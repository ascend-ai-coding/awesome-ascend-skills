# SparseFlashMla Pattern
一句话：SparseFlashMla 在 AscendC 算子里定义"注意力 KV 流 = ori 段（KV 窗口）+ cmp 段（压缩前缀）双段拼接，softmax 分母额外计入 attention sinks"；实现上要把两段不同可见性的 KV 组织进同一条 flash attention 主循环，窗口界与压缩阈值逐查询行变化。

## 先读这个

命中信号：算子名 `SparseFlashMla`，或输入含 `cmp_kv` / `cmp_sparse_indices`（双段 KV 独有信号）；`sinks`、`metadata` 为辅助信号（当前版本 sinks 必填、metadata 由配套 metadata 算子生成，但单独出现不足以判定本模式——标准 MLA 算子也可带 sinks）。
本模式只负责：双段 KV 的可见性规则（窗口 + 压缩前缀阈值）、sinks 的归一化语义、场景判定与循环结构。不要把它简化成"只是滑动窗口注意力"；它与 DeepSeek MLA（`q_nope/q_rope` + latent cache，Dqk≠Dv，见 `mla_pattern.md`）是两个不同模式。

> **平台范围**：本文只覆盖 **arch22（Atlas A2/A3，910B）** 分支。打包算子在 A5/A6 系列的参数约束不同（mask_mode 取值更宽、cmp_ratio 1~128、topk_value_mode 支持 1/2），不在本文范围，移植时须重新核对。

## 1. 识别条件

- KV 流由两段拼接：ori 段 = 原始 KV 的滑动窗口/因果可见区间；cmp 段 = 前缀压缩 KV。
- softmax 分母额外计入 `sinks`（attention sinks，只归一化、不贡献数值）。
- 硬约束：`D = 512`（q/kv head dim，tiling 层强制）、`KV_N = 1`（K/V 共享存储）；dtype 仅 fp16/bf16；Q_N ∈ {1,2,4,8,16,32,64,128}。
- 布局：`layout_q`/`layout_kv` 支持 BSND/TND（A2/A3 的 DSpark 场景仅支持 TND/PA_BBND）；paged KV 的 block_size 为 16 的倍数、最大 1024。
- 场景由**输入张量存在性**唯一确定（算子无 cmp_topk 之类的判定 attr）：

| 场景 | ori_kv | ori_sparse_indices | cmp_kv | cmp_sparse_indices |
|----------|--------|--------------------|--------|--------------------|
| SWA | ✓ | – | – | – |
| SWA 稀疏 ori_kv（DSpark） | ✓ | ✓（配 `ori_topk_length`） | – | – |
| HCA | ✓ | – | ✓ | – |
| CSA | ✓ | – | ✓ | ✓ |
| ORI_CMP_SPARSE | ✓ | ✓ | ✓ | ✓ |

SWA/HCA/DSpark 走 SWA kernel 模板（HCA = SWA + cmp 前缀段），CSA/ORI_CMP_SPARSE 走 CSA kernel 模板。

### A2/A3 平台参数约束（op tiling 层 OP_CHECK 校验，非 aclnn 层）

| 条件 | A2/A3 要求 |
|---------|------------|
| `topk_value_mode` | 仅 1 |
| 无 `cmp_kv`（SWA/DSpark） | `cmp_ratio = 1` 且 `cmp_mask_mode = 0` |
| HCA | `cmp_ratio = 128`、`cmp_mask_mode = 3` |
| CSA | `cmp_ratio ∈ {1, 2, 4}`、`cmp_mask_mode = 3`；`cmp_sparse_indices` 末维 K2 ∈ [1, 8192] 任意整数 |
| DSpark（`ori_sparse_indices` 非空） | `ori_mask_mode = 0`，`ori_win_left/right` 非负 |
| 非 DSpark | `ori_mask_mode = 4`（BAND）、`ori_win_left = 127`、`ori_win_right = 0` |
| 通用 | `ori_mask_mode ∈ {0,3,4}`、`cmp_mask_mode ∈ {0,3}`；`ori_mask_mode ≠ 4` 时 win 必须为 -1；win ∈ [-1, INT32_MAX] |
| `cmp_residual_kv` | `cmp_mask_mode=3` 且 `cmp_ratio ≠ 1` 时必填（恢复压缩前长度） |

参数不合法在 op tiling 层被拒绝（`OP_CHECK_IF` 图校验）。

## 2. 核心规则

### 双段 KV 单循环

```text
for kv tile t in one serial loop:          # 边界 total_blocks 为查询行 s 的 TIR 表达式
  t <  ori_blocks: ori 段块（窗口 [start(s), end(s)) 覆盖的块）
  t >= ori_blocks: cmp 段块（前缀 [:threshold(s)）覆盖的块）
  越界列不跳过，掩码列全填充（exp 下溢为 0），循环体分支无关
```

关键点：

- 两段进同一条 flash-softmax 状态机，共享 `m_i / sumexp / acc_o`；不拆成两个循环。
- 窗口界/阈值随查询行变化 → 内层循环边界动态；该循环不做软件流水，靠 AUTO_SYNC 管理。
- 窗口起止按块不裁剪：边界块内越界列用掩码填掉，与打包算子的"按块裁剪 + 边界块逐列掩码"等价。

### sinks 语义

softmax 分母 `L = Σ exp(scores - m) + exp(sinks - m)`，分子不变（`m = max(rowmax(scores), sinks)`）。

- sinks 是虚拟 key：参与 row max 和分母，不进 PV 数值。
- 当前版本 sinks **必填**；shape `(N_q,)`（= q 头数），fp32；无 sinks 语义时传零向量（等价 golden 无 sinks）。
- flash 状态初值即该语义：`m_i = sinks`（T.copy）、`sumexp = 1`（= exp(sinks - m_i)）、`acc_o = 0`。

### CSA gather（A2/A3 契约）

- `cmp_sparse_indices` 是**块粒度逻辑索引**：元素值 × sparseBlockSize = 行基址，gather 以 sparse block 为单位取数（不是逐行 gather）。
- **-1 = 无效/填充项，跳过**（与 `ori_sparse_indices` 的 "-1 表示无效或填充 slot" 语义一致），不是扫描终止符。
- cmp 段可见性（压缩阈值）由 `cmp_mask_mode = 3`（RightDownCausal）掩码表达；gather 只负责按索引取块。
- `cmp_topk_length` 在 A2/A3 **不支持传入非空 tensor**；有效条目由索引张量内的 -1 标记表达（A5/A6 系列才走长度张量契约，不在本文范围）。

### 掩码填充值

- 打包算子参考实现的掩码填充常量为 **-2e38**（`SOFTMAX_MIN_NUM`），以**选择性直填**写入被掩 lane（带 mask 的 Duplicate），既不是 -2^30，也不是对全部列做加法。
- TileLang 侧等价写法：乘加式 `scores*valid + MASK_FILL*(1-valid)` 或直接置位，MASK_FILL 取 exp 下溢为 0 的足够负值即可（与参考一致取 -2e38）。

### metadata-first 两段式（打包算子形态）

- `SparseFlashMlaMetadata`（配套 metadata 算子）生成任务列表 `metadata`；主算子按任务表调度，任务空间 2D 网格 `(bN2 × gSq)`。
- 任务表布局：FA 每核 9 字段、FD 每核 8 字段（kernel 侧数组上限 FA 80 / FD 80 核；实际使用核数与字段含义由任务表内容决定，末核等特殊编码以 metadata 算子输出为准）。

## 3. 地址/形状公式

```text
q:       [B, S1, N1, D]      dtype fp16/bf16
ori_kv:  [B, S2, 1, D]       （N2 == 1，K/V 共享存储）
cmp_kv:  [B, S2c, 1, D]      可选
sinks:   (N_q,) float32       当前版本必填（零向量 = 无 sinks 语义）
out:     [B, S1, N1, D]
```

主算子 kernel 入口 22 个 GM 参数（20 个用户张量 + workspace + tiling），输入含
`q / ori_kv / cmp_kv / ori_sparse_indices / cmp_sparse_indices / ori_block_table / cmp_block_table /
cu_seqlens_q / cu_seqlens_ori_kv / cu_seqlens_cmp_kv / seqused_q / seqused_ori_kv / seqused_cmp_kv /
cmp_residual_kv / ori_topk_length / cmp_topk_length / sinks / metadata`，输出 `attn_out / softmax_lse`。

smScale：打包算子 `softmax_scale` attr 由调用方显式传入（默认 1.0）；`softmax_scale <= 0 → 1/sqrt(D)`
是 TileLang 配置形态自己的约定，不是打包算子的行为。

ori 窗口（查询行 s，q_len=S1，ori_len=S2）：

```text
MASK_NO(0):     [0, S2)
MASK_CAUSAL(3): [0, max(causal, 0))
MASK_BAND(4):   [start, end)
  causal = S2 - S1 + s + 1
  start  = 0                       if win_left  == -1 else max(causal - win_left - 1, 0)
  end    = S2                      if win_right == -1 else min(causal + win_right, S2)
```

cmp 前缀阈值：

```text
cmp_mask_mode == NO:     threshold = min(S2 // ratio, S2c)          # 全量可比
cmp_mask_mode == CAUSAL: threshold = min(max(causal, 0) // ratio, S2c)
HCA 压缩长度: S2c = max(S2 // ratio, 1)     # 用 floor；ceil 会使阈值超出真实 ori_len
恢复压缩前长度: ori_len = cmp_len * ratio + cmp_residual          # cmp_residual_kv 输入
```

双段独立传入（TileLang 形态，host 禁止 pad/拼接）：

```text
kernel 签名: kernel(q, ori_kv, cmp_kv, sinks, idx, output)   # 双段独立 tensor
ori_blocks  = ceil(o_end / BLOCK_N)         # o_end = 窗口右端
cmp_blocks  = ceil(threshold / BLOCK_N)
total_blocks = ori_blocks + cmp_blocks
t ∈ [0, total_blocks):
  is_ori  = t < ori_blocks
  seg_t   = is_ori ? t : t - ori_blocks
  seg_end = is_ori ? o_end : threshold
  valid   = clamp(seg_end - seg_t*BLOCK_N, 0, BLOCK_N)      # 尾块真实 extent
  T.copy(is_ori ? ori_kv[cid, seg_t*BLOCK_N : seg_t*BLOCK_N+valid, 0, :]
                : cmp_kv[cid, seg_t*BLOCK_N : seg_t*BLOCK_N+valid, 0, :],
         kv_l1[:, :valid], pad_value=0.0)                   # pad 行由乘加式掩码双保险清零
```

mask 常量：`MASK_NO=0 / MASK_CAUSAL=3 / MASK_BAND=4`，`UNLIMITED=-1`；填充值见 §2（-2e38 直填）。

## 4. 与其它模式的边界

```text
场景判定(SWA/HCA/CSA/DSpark) -> 窗口+阈值逐行计算 -> 双段单循环 load -> flash-softmax(+sinks 分母) -> PV -> 归一化输出
```

- DeepSeek MLA（`mla_pattern.md`）：`q_nope/q_rope`、Dqk≠Dv 的 latent 形态；本模式 K/V 共享存储、无 rope、单一 D=512。两者都改 flash 主循环的 KV 来源，但维度消费规则不同。
- 普通 SWA / FlashAttention：是本模式 ori 段的特例（无 cmp 段、sinks=0、单段循环）。
- Sparse/TopK：gather 只决定取哪些 cmp/ori 块（块粒度索引，-1 跳过），不改变窗口与 sinks 规则。
- Mask/Causal：只作用在 scores 的可见性；本模式还额外引入压缩前缀可见性与 sinks 分母。
- prefill/decode 统一：窗口界为 s 的 TIR 表达式，decode 行（S1=1 或 s=0）自动折为编译期常量，同一份 kernel 覆盖两种形态。

## 5. 检查点

- `smScale` 按 D=512 算（TileLang 配置形态 `softmax_scale <= 0` 时取 `1/sqrt(D)`；打包算子由调用方显式传入）。
- ori/cmp 两段必须在同一条 serial 循环内消费；全部跨 pipe 拷贝集中在一个 serial 循环（AUTO_CV_SYNC 同步点计数不变量），拆循环复用 workspace 会破坏同步配对。
- sinks 只进分母不进分子；flash 初值 `m_i=sinks`、`sumexp=1`、`acc_o=0`，epilogue `out = acc_o / sumexp`。
- 掩码必须乘加式 `scores*valid + MASK_FILL*(1-valid)` 或选择性直填；禁止把 MASK_FILL 与有效 score 直接相加（fp32 大数相吞有效列）。MASK_FILL 取 -2e38 级（与打包算子参考实现一致），勿用未经验证的其他常数冒充 spec。
- host 侧禁止 pad/拼接（`torch.zeros`+切片赋值、`torch.cat`/`F.pad` 违反 ascend-constraints §5 并触发 aclnn）；kernel 规则是双段独立传入（ori_kv/cmp_kv）、同一 `T.serial` 内经 `T.if` 选段、尾块 `valid = clamp(seg_end - seg_t*BLOCK_N, 0, BLOCK_N)` 用 `T.copy` BufferRegion 动态切片 + `pad_value` 搬入，越界列由乘加式掩码清零。
- A2/A3 直调按 §1 平台表对齐参数（tiling 层 OP_CHECK 拒绝）：HCA ratio=128、CSA ratio∈{1,2,4}、非 DSpark 的 ori_mask_mode=4/win=127/0、DSpark 的 ori_mask_mode=0、cmp_mask_mode=3（CSA/HCA）。
- 精度验证输入须同时覆盖正分布（`rand*2`）与含符号分布（`rand*4-2`）：含负输入使 softmax 输出过零、近零元素 rel_err 放大，由混合容差判据（atol+rtol）兜底；超判据按真实精度缺陷修复，禁止限定输入符号规避。
