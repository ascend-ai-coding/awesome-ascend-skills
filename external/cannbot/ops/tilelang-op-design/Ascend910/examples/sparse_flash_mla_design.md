# SparseFlashMla 算子设计文档

> 本文按 [`design-template.md`](design-template.md) 模板 §1–§10 填写，是
> `examples/` 目录下的独立交付文件：关键实现代码片段内联在对应章节——
> §3.3（tile 级 kernel 签名与双段循环）、§4.4（双段 KV 直传与尾块 valid-extent 拷贝）、
> §6.3（block 级 2D 任务反解与 ring 流水）。模板 §11
> 交付清单（含 proto.yaml）不在本文范围。实现已在 Ascend 910B 硬件验证通过。

---

## 1. 概述

### 1.1 算子名称

SparseFlashMla（压缩稀疏 MLA 注意力）

### 1.2 功能描述

对 MLA 布局的注意力做**压缩稀疏**加速：注意力 KV 流由两段拼接而成——ori 段
（原始 KV 的滑动窗口/因果可见区间）+ cmp 段（前缀压缩 KV），softmax 分母额外
计入 attention sinks（只归一化、不贡献数值）。场景由**输入张量存在性**唯一确定
（打包算子无 cmp_topk 之类判定 attr；本文只覆盖 arch22 = A2/A3/910B 分支）：

| scenario | ori_kv | ori_sparse_indices | cmp_kv | cmp_sparse_indices |
|----------|--------|--------------------|--------|--------------------|
| SWA | ✓ | – | – | – |
| SWA 稀疏 ori_kv（DSpark） | ✓ | ✓（配 `ori_topk_length`） | – | – |
| HCA | ✓ | – | ✓ | – |
| CSA | ✓ | – | ✓ | ✓ |
| ORI_CMP_SPARSE | ✓ | ✓ | ✓ | ✓ |

HCA 的 cmp 段为压缩前缀 `floor(causal / ratio)` 行；CSA 在 HCA 之上以
`cmp_sparse_indices` 做块粒度稀疏取数（-1 项跳过，见 §10.1）。A2/A3 平台参数
约束（tiling 层 OP_CHECK）：HCA `cmp_ratio=128`、CSA `cmp_ratio∈{1,2,4}`、
CSA/HCA `cmp_mask_mode=3`、DSpark `ori_mask_mode=0` 且 win 非负、非 DSpark
`ori_mask_mode=4`/`win_left=127`/`win_right=0`、`topk_value_mode=1`、
`cmp_mask_mode=3 且 ratio≠1` 时 `cmp_residual_kv` 必填。

本目录实现覆盖 **SWA / HCA**；CSA（块粒度索引 gather）为显式扩展点（§10.1）。
与 DeepSeek MLA（`q_nope/q_rope`、Dqk≠Dv 的 latent 形态）区分：本算子无 rope
拆分、K/V 共享存储。打包算子契约与平台约束已在 §3.5.2 差异对照表中完整给出。

> **⚠️ Host 侧 Buffer 操作约束**（详见 ../references/ascend-constraints.md §5）：host
> 侧不改动 NPU 张量内容、不触发 aclnn——**禁止** `torch.zeros`+切片赋值拼接、
> `torch.cat`/`F.pad` 等 host padding。ori/cmp 两段 KV 以独立 tensor 原样传入
> kernel，尾块由 kernel 内 `T.copy` valid-extent 动态切片 + `pad_value` 处理
> （见 §4.4/§5.4）；golden 与输入构造在 CPU 完成。

### 1.3 数学公式

$$
\text{scores} = Q \cdot K^{\top} \cdot s, \qquad
P = \frac{\exp(\text{scores} - m)}{L}, \qquad
O = P \cdot V
$$

其中每查询行 $K = V = \text{concat}\big(ori\_kv[\,win(q\_pos)\,],\ cmp\_kv[\,:\,thr(q\_pos)\,]\big)$，
$m = \max(\text{rowmax}(\text{scores}),\ \text{sinks})$，分母
$L = \sum \exp(\text{scores} - m) + \exp(\text{sinks} - m)$（sinks 为虚拟 key）。

### 1.4 算法描述

多步分解（flash decoding 形态，逐行 Q 滑窗）：

1. **窗口计算**（标量/TIR 表达式）：对每查询行 $s$ 求 ori 窗口 $[start(s), end(s))$
   与 cmp 前缀阈值 $thr(s)$；BAND 模式 $causal = ori\_len - q\_len + s + 1$，
   $start = max(causal - win\_left - 1, 0)$，$end = min(causal + win\_right, ori\_len)$。
2. **双段 KV 单循环**：ori 段块 + cmp 段块拼进**同一个 serial 循环**（$t < ori\_blocks$
   走 ori 段，否则走 cmp 段），共享同一 flash-softmax 状态机；越界列不跳过，
   以 `MASK_FILL = -2e38` 掩码（exp 下溢为 0），循环体分支无关。
3. **flash-softmax 状态机**（与 golden `softmax_with_sinks` 严格一致）：

   ```text
   m_i     := sinks        （初始 = 虚拟 key 的 max）
   sumexp  := 1            （= exp(sinks - m_i)）
   per block: m_old = m_i;  m_i = max(m_i, rowmax(score));
              alpha  = exp(m_old - m_i)
              sumexp = alpha * sumexp + rowsum(exp(score - m_i))
              acc_o  = alpha * acc_o + P @ V
   epilogue: out = acc_o / sumexp
   ```

4. **epilogue**：`acc_o / sumexp` → cast 目标 dtype → 写回 GM。

### 1.5 数据流图

```text
                    ┌─────────── 每 (batch, 查询行 s, head 块) ───────────┐
 q [B,S1,N1,D] ──T.copy──> q_l1 ─┐
 ori/cmp 块 t（valid-extent 切片）──T.copy──> kv_l1 ──┤ T.gemm_v0(QK^T) ─> scores_l0c ─> scores_ub
 idx [BLOCK_N] ──T.copy──> idx_ub │   ×softmax_scale + 窗口掩码(乘加式, MASK_FILL)
                                  │   flash-softmax: m_i / sumexp / P
                                  │ T.copy P ─> p_l1 ─> T.gemm_v0(PV) ─> acc_o_new
                                  │ acc_o = alpha*acc_o + acc_o_new   (UB)
                                  └ epilogue: ÷sumexp → cast → output [B,S1,N1,D]
```

---

## 2. 编程模式选型

### 2.1 模式结论

**混合**（tile 级实现 = Developer 式 CV 融合；block 级骨架 = Expert 显式核分离）。
**融合算子判定：是**（GEMM + softmax 后处理，见 §8）。

### 2.2 选型理由

算子主体是 QK/PV 两个 Cube GEMM + 窗口掩码/online softmax/归一化 Vector 后处理，
必须 AIC/AIV 协同。tile 级实现把 CV 分离、同步、workspace 中转全部交给
`TL_ASCEND_AUTO_CV_COMBINE + TL_ASCEND_AUTO_CV_SYNC + TL_ASCEND_AUTO_SYNC`
四个 pass（Developer 模式），消除手写 workspace/vid 负担且已硬件验证；
block 级骨架演示与打包算子调度对齐的 Expert 形态（显式 `T.Scope("C"/"V")` +
`set/wait_cross_flag` + GM workspace ring），供生成 MLA 类算子时参考。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_L1 / T.alloc_L0C / T.alloc_ub` 显式指定（L0A/L0B 由 gemm 内部管理） |
| 计算方式 | `T.gemm_v0`（Cube）+ `T.tile.*` / `T.reduce_*`（Vector） |
| 作用域 | tile 级：编译器自动 CV 分离；block 级骨架：显式 `T.Scope("C") / T.Scope("V")` |
| 同步方式 | tile 级：四个 AUTO pass 自动同步；block 级骨架：`set/wait_cross_flag` 手动信号 |

---

## 3. API 映射设计

### 3.1 公式拆解

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | $S = Q \cdot K^{\top} \cdot s$ | QK GEMM（`transpose_B`），KV 块来自双段拼接流 |
| 2 | $\text{scores} \gets \text{scores} \cdot \text{valid} + \text{MASK\_FILL} \cdot (1 - \text{valid})$ | 乘加式窗口掩码（避免 fp32 大数相消，§10.2 坑 #4） |
| 3 | $m_i = \max(m_i, \text{rowmax}(\text{scores}))$，$\alpha = e^{m_{old} - m_i}$ | online softmax rescale |
| 4 | $sumexp = \alpha \cdot sumexp + \text{rowsum}(e^{\text{scores} - m_i})$ | 分母累计（初值 1 = sinks 贡献） |
| 5 | $P = e^{\text{scores} - m_i}$，cast 目标 dtype | softmax 分子即 PV 左操作数 |
| 6 | $acc_o = \alpha \cdot acc_o + P \cdot V$ | PV GEMM 每块 re-init，结果读回 UB 累加 |
| 7 | $out = acc_o / sumexp$ | epilogue 归一化 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | QK^T | `T.gemm_v0(q_l1, kv_l1, scores_l0c, transpose_B=True, init=True)` | L1 操作数 → L0C 全量初始化输出 | Developer(Cube) |
| 2 | 掩码 | `T.tile.mul/add/max/min` 组合 + 每行 `T.tile.mul/add` | 由 `idx_ub` 算术生成 valid∈{0,1} 列掩码 | Developer(Vector) |
| 3/4 | rescale/rowsum | `T.reduce_max(scores_ub, m_i, dim=-1)` / `T.reduce_sum(..., rowsum, dim=-1)` + `T.tile.max/sub/exp/mul/add` | UB 行归约 | Developer(Vector) |
| 5 | P cast | `T.copy(scores_ub, p_ub)`（f32→bf16 cast）+ `T.copy(p_ub, p_l1, tmp=p_ub_nz)` | UB→L1，nz tmp 抑制非规范化 | Developer |
| 6 | PV | `T.gemm_v0(p_l1, kv_l1, acc_o_l0c, init=True)` + `T.copy(acc_o_l0c, acc_o_new)` | 每块 re-init；L0C→UB 读回 | Developer(Cube) |
| 7 | 归一化/写回 | `T.tile.div` + `T.copy(acc_o_half, output[...])` | f32 归一化后 cast 写出 | Developer(Vector) |
| — | flash 状态初始化 | `T.copy(sinks[...], m_i)` + `T.tile.fill(sumexp, 1.0)` + `T.tile.fill(acc_o_ub, 0.0)` | sinks 为 `m_i` 初值；`sumexp=1` = exp(sinks-m_i) | Developer |

### 3.3 计算伪代码

```python
# 与下方实际 kernel 签名一致的结构（grid = batch）
with T.Kernel(batch, is_npu=True) as (cid, vid):
    q_l1   = T.alloc_L1([BLOCK_H, dim], dtype)          # 32×512 bf16 = 32KB
    kv_l1  = T.alloc_L1([BLOCK_N, dim], dtype)          # 128×512 bf16 = 128KB
    p_l1   = T.alloc_L1([BLOCK_H, BLOCK_N], dtype)
    scores_l0c = T.alloc_L0C([BLOCK_H, BLOCK_N], "float")
    acc_o_l0c  = T.alloc_L0C([BLOCK_H, dim], "float")   # 32×512 f32 = 64KB（D=512 上限关键）
    # scores_ub / acc_o_ub / acc_o_new / p_ub / m_i / sumexp / ...（AIV 每核半块行）

    T.copy(idx, idx_ub)                                  # [BLOCK_N] f32 arange 掩码常量
    for s in T.serial(q_seq):                            # 查询行
        causal = ori_seq - q_seq + s + 1                 # TIR 表达式
        o_start/o_end/cmp_thr = 窗口/阈值（TIR 表达式，decode 行折为常量）
        for h in T.serial(head_blocks):                  # head 块（BLOCK_H=32）
            T.copy(q[cid, s, h_lo:h_lo+BLOCK_H, :], q_l1)
            T.copy(sinks[...], m_i); T.tile.fill(sumexp, 1.0); T.tile.fill(acc_o_ub, 0.0)
            for t in T.serial(total_blocks):             # 双段拼接 KV 块循环（动态边界）
                # seg_t/seg_end/valid 索引算术见 §3.3.1（s 的 TIR 表达式）
                if t < ori_blocks:                       # valid-extent 动态切片 + pad_value
                    T.copy(ori_kv[cid, seg_t*BLOCK_N : seg_t*BLOCK_N+valid, 0, :], kv_l1[:, :valid], pad_value=0.0)
                else:
                    T.copy(cmp_kv[cid, seg_t*BLOCK_N : seg_t*BLOCK_N+valid, 0, :], kv_l1[:, :valid], pad_value=0.0)
                T.gemm_v0(q_l1, kv_l1, scores_l0c, transpose_B=True, init=True)
                T.copy(scores_l0c, scores_ub)
                # ×scale → 乘加式窗口掩码 → flash-softmax → P cast → UB→L1
                T.gemm_v0(p_l1, kv_l1, acc_o_l0c, init=True)
                T.copy(acc_o_l0c, acc_o_new)
                # acc_o = alpha*acc_o + acc_o_new
            T.tile.div(acc_o_ub, acc_o_ub, sumexp)       # epilogue
            T.copy(acc_o_half, output[cid, s, ...])
```

### 3.3.1 实际 kernel 签名与关键实现（代码节选）

共享常量（mask mode 取值与算子 spec / 打包算子 model.py 一致）：

```python
MASK_NO = 0
MASK_CAUSAL = 3
MASK_BAND = 4
UNLIMITED = -1
MASK_FILL = -2e38            # 掩码列填充值（与打包算子参考实现 SOFTMAX_MIN_NUM 一致；exp 下溢为 0，fp32 下安全）
BLOCK_N = 128            # KV 块
BLOCK_H = 32             # query-head 块（L0C 容量约束，§5.3）
BLOCK_H_HALF = BLOCK_H // 2   # 每 AIV 半块行（MIX 双 Vector 核 vid 分工）
```

编译期配置与 kernel 入口签名：

```python
@dataclass(frozen=True)
class SparseFlashMlaConfig:
    batch: int; q_seq: int; heads: int; dim: int
    ori_seq: int; cmp_seq: int = 0; cmp_ratio: int = 1
    ori_mask_mode: int = MASK_BAND; cmp_mask_mode: int = MASK_CAUSAL
    ori_win_left: int = 127; ori_win_right: int = 0
    softmax_scale: float = -1.0   # <= 0 -> 默认 1 / sqrt(dim)
    dtype: str = "bfloat16"

def sparse_flash_mla_fwd(cfg: SparseFlashMlaConfig):
    # 返回编译后的 kernel，调用契约：kernel(q, ori_kv, cmp_kv, sinks, idx, output)
    #   q       [B, S1, N1, D]        输入 dtype
    #   ori_kv  [B, S2, 1, D] / cmp_kv [B, S2c, 1, D]  双段独立直传（host 不拼接
    #           不 pad，ascend-constraints §5）；尾块 valid extent 见 §4.4/§5.4
    #   sinks   [N1] f32；idx [BLOCK_N] f32 arange 掩码辅助常量
    #   output  [B, S1, N1, D] 调用方预分配，kernel 原地写
    ...
```

每查询行窗口界是 `s` 的 TIR 表达式（helper 只返回表达式、不发语句，符合 §10.1
helper 陷阱约束）：

```python
def _ori_window_start(causal, ori_seq, mask_mode, wl):
    if mask_mode == MASK_NO or wl == UNLIMITED: return 0
    if mask_mode == MASK_CAUSAL: return 0
    return T.max(causal - wl - 1, 0)

def _ori_window_end(causal, ori_seq, mask_mode, wr):
    if mask_mode == MASK_NO: return ori_seq
    if mask_mode == MASK_CAUSAL: return T.max(causal, 0)
    if wr == UNLIMITED: return ori_seq
    return T.max(T.min(causal + wr, ori_seq), 0)

def _cmp_thr(causal, ori_seq, cmp_seq, cmp_ratio, mask_mode):
    if cmp_seq <= 0: return 0
    if mask_mode == MASK_NO: return T.min(ori_seq // cmp_ratio, cmp_seq)
    return T.max(T.min(causal // cmp_ratio, cmp_seq), 0)
```

双段单循环的精确索引算术（本实现的核心技巧）：

```python
ori_blocks   = T.ceildiv(o_end, BLOCK_N)
cmp_blocks   = T.ceildiv(cmp_thr, BLOCK_N)
total_blocks = ori_blocks + cmp_blocks     # 双段同一 serial，共享 flash 状态
for t in T.serial(total_blocks):
    # 段内块下标 / 段末位（s 的 TIR 表达式）→ 本块 valid extent（尾块运行期值）
    is_ori  = t < ori_blocks
    seg_t   = T.if_then_else(is_ori, t, t - ori_blocks)
    seg_end = T.if_then_else(is_ori, o_end, cmp_thr)
    valid   = T.max(T.min(seg_end - seg_t * BLOCK_N, BLOCK_N), 0)
    if is_ori:
        T.copy(ori_kv[cid, seg_t*BLOCK_N : seg_t*BLOCK_N + valid, 0, :],
               kv_l1[:, :valid], pad_value=0.0)      # valid-extent 动态切片，pad 行零填充
    else:
        T.copy(cmp_kv[cid, seg_t*BLOCK_N : seg_t*BLOCK_N + valid, 0, :],
               kv_l1[:, :valid], pad_value=0.0)
    # ... QK → 掩码 → flash-softmax → PV（语句顺序约束见 §7.2）
```

> pass_configs 四个 AUTO pass（§4.7/§7.3）作用于 `tilelang.compile(main, pass_configs=...)`。

### 3.4 API 可行性确认

| API | 来源 | 验证 |
|-----|------|------|
| `T.gemm_v0`（L1 操作数入参） | `tilelang-op-develop` 编码规范 | ✅ |
| `T.alloc_L1/L0C/ub`、`T.copy`（含跨 pipe、nz tmp） | 同上 + `tilelang-programming-model-guide` | ✅ |
| `T.tile.mul/add/sub/exp/max/min/fill` | Vector 速查表 | ✅ |
| `T.reduce_max/T.reduce_sum(dim=-1)` | Vector 归约速查表 | ✅ |
| `T.Kernel(..., is_npu=True) as (cid, vid)`、`T.if_then_else` | 编程模型指南 | ✅ |

---

## 3.5 技术约束确认

### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No | 一维网格（tile 级 grid=batch；block 级 grid=used_core_num） |
| threads 参数限制（仅 1 或 2） | Yes | MIX 形态 `(cid, vid)` 双 Vector 核，每 AIV 持 `BLOCK_H/2` 行 |
| 动态循环边界不支持 | Yes | 内层 `t` 循环边界 `total_blocks` 为查询行 s 的 TIR 表达式——该循环不做软件流水，靠 AUTO_SYNC 管理，硬件验证通过 |
| 流水线不支持动态边界 | Yes | 双段 KV 循环用 `T.serial`，不用 `T.Pipelined`（吞吐让位正确性，见 §6.3） |

### 3.5.2 参考实现差异说明

参考实现为 CANN **打包算子**（kernel 入口 22 个 GM 参数 = 20 用户张量 + workspace +
tiling；arch22 = A2/A3/910B 分支，A5/A6 分支不在本文范围）：

| 差异项 | 参考实现（打包算子） | 本项目（TileLang） | 转换方案 |
|--------|---------------------|-------------------|----------|
| 任务划分 | 配套 metadata 算子（`SparseFlashMlaMetadata`）预生成任务表（FA 每核 9 字段 + FD 每核 8 字段，数组上限 80+80 核），2D 网格 (bN2 × gSq) | 一维 grid=batch，核内顺序扫 (s, head 块) | block 级骨架给出同构 2D 线性化反解方案（§6.3） |
| 双段 KV | ori/cmp 两段独立寻址（22 指针，含 cu_seqlens_*/seqused_*/block_table 等 18 个输入张量） | 两段独立 tensor 直传 kernel，单循环内 `T.if` 选段 + 段内 valid-extent 拷贝 | 免 host 拼接/pad（ascend-constraints §5），见 §4.4 |
| AIC/AIV 流水 | 手写 workspace ring + set/wait 信号 | AUTO_CV_COMBINE/SYNC pass 自动推导，无 workspace | §7 |
| 掩码 | 窗口起止按块裁剪 + 边界块逐列掩码（填充常量 -2e38 选择性直填） | 不裁剪，越界列 MASK_FILL 全掩码（分支无关循环体） | 等价（exp 下溢为 0） |
| CSA gather | `cmp_sparse_indices` **块粒度**逻辑索引（元素 × sparseBlockSize = 行基址，-1 项跳过） | **未覆盖**（扩展点，§10.1） | `T.tile.gather` 或按块动态 GM 拷贝 |

### 3.5.3 本项目同类实现参考

| 参考实现 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| flash_attention 设计模板（block/tile 双级） | 高度相似 | flash-softmax 状态机、C/V split、Q 分块循环结构 |
| §6.3 block 级骨架 | 同算子 block 级 | 2D 任务网格反解、双段 KV ring 流水（Expert 形态） |
| §3.5.2 打包算子差异对照 | 算子领域知识 | 打包算子契约、910B 平台约束、metadata 语义 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

打包算子完整输入（18 个，arch22 契约；TileLang 形态只消费其中双段直传所需子集）：

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| q | [B, S1, N1, D] | bfloat16 / float16 | 查询，BSND/TND 布局 |
| ori_kv | [B, S2, N2, D] | bfloat16 / float16 | 原始 KV（K==V 共享存储），N2=1 |
| cmp_kv | [B, S2c, N2, D] | bfloat16 / float16 | 压缩前缀 KV（HCA/CSA 场景；纯 SWA 不传） |
| ori_sparse_indices | [B, S1, N2, K] / [T1, N2, K] | int32 | DSpark：ori 段块粒度逻辑索引（-1 = 无效/填充项跳过） |
| ori_topk_length | [B, S1, N2] / [T1, N2] | int32 | DSpark：每个 q token 的有效索引条目数 |
| cmp_sparse_indices | [B, S1, N2, K2] | int32 | CSA：cmp 段块粒度逻辑索引（-1 项跳过；K2 ∈ [1, 8192]） |
| cmp_topk_length | [B, S1, 1] | int32 | 仅 A5/A6 使用；**A2/A3 必须传空** |
| ori_block_table / cmp_block_table | 分页页表 | int32 | paged KV 逻辑块→物理块映射 |
| cu_seqlens_q / cu_seqlens_ori_kv / cu_seqlens_cmp_kv | [B+1] | int32 | 各段 cumulative 长度 |
| seqused_q / seqused_ori_kv / seqused_cmp_kv | [B] | int32 | 各段实际使用长度 |
| cmp_residual_kv | [B] | int32 | 压缩余数，`cmp_mask_mode=3 且 cmp_ratio≠1` 时必填 |
| sinks | (N_q,) | float32 | attention sinks（**当前版本必填**；零向量 = 无 sinks 语义） |
| metadata | 1D | int32 | 任务表，由 `SparseFlashMlaMetadata` 生成 |

硬约束：`D = 512`（tiling 层强制）、`N2 = 1`（KV_N=1 算子约束，此时 `group = heads`，
无 head-sharing 分派）、`N1 % 32 == 0`（本设计 BLOCK_H 整除 wrapper 契约；打包算子
Q_N 支持 1~128）、dtype 仅 fp16/bf16、paged block_size 为 16 的倍数且 ≤ 1024。

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| output | [B, S1, N1, D] | bfloat16 / float16 | 注意力输出（调用方预分配，kernel 原地写） |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| q_l1 | (32, 512) | bf16 | L1 | 当前 head 块 Q tile |
| kv_l1 | (128, 512) | bf16 | L1 | 当前 KV 块（双段共用） |
| p_l1 | (32, 128) | bf16 | L1 | P 操作数（UB→L1） |
| scores_l0c | (32, 128) | f32 | L0C | QK 输出 |
| acc_o_l0c | (32, 512) | f32 | L0C | PV 输出（**BLOCK_H=32 的成因，见 §5.3**） |
| scores_ub | (16, 128) | f32 | UB | softmax 分数（每 AIV 半块行） |
| acc_o_ub / acc_o_new | (16, 512) | f32 | UB | flash 状态 O / 本块 PV 读回 |
| acc_o_half | (16, 512) | bf16 | UB | epilogue cast 缓冲 |
| p_ub / p_ub_nz | (16, 128) | bf16 | UB | P cast / nz tmp |
| m_i / m_old / sumexp / rowsum | (16,) | f32 | UB | flash 状态标量行 |
| idx_ub / mask_lo / mask_hi / mask_col | (128,) | f32 | UB | 列掩码算术 |

### 4.4 内存搬运路径

```text
host 侧（零计算、零数据搬运、零 aclnn；ascend-constraints §5）：
  ori_kv [B,S2,1,D]  ──┐
                        ├─> 独立 tensor 直传 kernel（不拼接、不 pad）
  cmp_kv [B,S2c,1,D] ──┘   尾块 valid extent 由 kernel 内 T.copy 动态切片处理

GM[q] --T.copy--> L1[q_l1]                        # 每 (s, head块)
GM[ori_kv/cmp_kv] --T.copy(valid-extent 切片 + pad_value)--> L1[kv_l1]   # 每 KV 块
L1[q_l1] + L1[kv_l1] --T.gemm_v0--> L0C[scores_l0c] --T.copy--> UB[scores_ub]
UB[scores_ub] --掩码/softmax--> UB[p_ub] --T.copy--> L1[p_l1]
L1[p_l1] + L1[kv_l1] --T.gemm_v0--> L0C[acc_o_l0c] --T.copy--> UB[acc_o_new]
UB[acc_o_ub] --÷sumexp/cast--> UB[acc_o_half] --T.copy--> GM[output]
```

host wrapper（零布局代码：不做拼接、不做 pad，符合 ascend-constraints §5）：

```python
# ori_kv [B,S2,1,D]、cmp_kv [B,S2c,1,D] 原样传入，不创建新 buffer、不触发 aclnn
# （禁止 torch.zeros+切片赋值 / torch.cat / F.pad——host padding 属违规，
#   尾块改由 kernel 内 T.copy valid-extent 动态切片 + pad_value 处理，§5.4）
# 纯 SWA 无 cmp_kv：传占位 [B, 1, 1, D]，kernel 内 cmp_blocks == 0 永不读取
idx = torch.arange(BLOCK_N, dtype=torch.float32, device=q.device)  # 掩码辅助常量
kernel(q, ori_kv, cmp_kv, sinks, idx, output)  # output 调用方预分配
```

### 4.5 UB 内存预算

每 AIV 核（`hb = BLOCK_H/2 = 16` 行半块）：

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| scores_ub | (16, 128) | f32 | 8,192 |
| acc_o_ub | (16, 512) | f32 | 32,768 |
| acc_o_new | (16, 512) | f32 | 32,768 |
| acc_o_half | (16, 512) | bf16 | 16,384 |
| p_ub / p_ub_nz | (16, 128) ×2 | bf16 | 8,192 |
| m_i/m_old/sumexp/rowsum | (16,) ×4 | f32 | 256 |
| idx_ub/mask_lo/mask_hi/mask_col | (128,) ×4 | f32 | 2,048 |
| **总计** | | | **≈ 100,608 (98.3KB)** / 196,608 (192KB, Ascend910) ✓ |

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| ori_seq / cmp_seq / q_seq | host 侧 python 常量（`SparseFlashMlaConfig` 编译期配置） | 受 ori/cmp 段长与窗口 TIR 表达式约束；kernel 每实例单 shape |

> 本实现为编译期配置形态（每 shape 编译一个 kernel 实例，host wrapper 契约见
> §4.4）；`start/end/cmp_thr` 在 kernel 内是
> s 的 TIR 表达式，prefill/decode 同一份代码，decode 行自动折为常量。

### 4.7 JIT 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}
kernel = tilelang.compile(main, pass_configs=pass_configs, target="ascendc")
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 混合（Cube QK/PV GEMM + Vector softmax/掩码/归一化）

**判定依据**: 含 `T.gemm_v0` 矩阵乘（QK^T、PV）+ 行归约（rowmax/rowsum）
+ 逐列掩码 element-wise，必须 AIC/AIV 协同。

### 5.2 Block 划分

```python
BLOCK_N = 128   # KV 块：DMA 整块拷贝对齐（bf16 尾轴 512B），单块 128KB L1
BLOCK_H = 32    # query-head 块：L0C 容量约束的成因（见 §5.3）
head_blocks = (heads + BLOCK_H - 1) // BLOCK_H   # kernel 内 h 循环
# tile 级：grid = batch（一核一个 batch 行），核内 (s, h, t) 三层串行
# block 级：used_core_num = min(20, q_work)，2D 线性化任务区间（见 block_level 文件）
```

### 5.3 约束分析

- **对齐约束**: `BLOCK_N=128`、`D=512`，bf16 尾轴 512B ≫ 32B DMA 对齐 ✓；
  KV 拷贝为 valid-extent 动态切片（尾块端点随 s 的 TIR 表达式在运行期计算，见 §5.4）。
- **UB 容量**: 每 AIV ≈ 98.3KB < 192KB ✓（`TL_ASCEND_MEMORY_PLANNING` 可复用
  dead buffer）。
- **L0 容量（关键）**: PV 输出 `[BLOCK_H, 512]` f32。`BLOCK_H=128` 时需
  `128×512×4 = 256KB > 128KB` L0C 容量，**超限**；取 `BLOCK_H=32` 后
  `32×512×4 = 64KB` + QK 分数 `32×128×4 = 16KB` = 80KB ≤ 128KB ✓。
  head 维因此以 32 行分块循环。
- **L1 容量**: q_l1 32KB + kv_l1 128KB + p_l1 8KB = 168KB < 512KB（910B L1）✓。

### 5.4 注意事项

- **非整除必须显式设计（kernel 内处理，禁止 host pad）**：`ori_seq`/`cmp_seq`
  不整除 `BLOCK_N` 时，kernel 对每个 KV 块按
  `valid = clamp(seg_end - seg_t*BLOCK_N, 0, BLOCK_N)` 计算段内有效行数，用
  `T.copy` 的 BufferRegion 动态切片搬入 `kv_l1[:, :valid]`（`pad_value=0.0`
  填 pad 行）；pad 列落在所有查询窗口之外，再由乘加式掩码双保险清零
  （§6.4）。符合 ascend-constraints §5「无需 host padding」与
  api-kernel-memory「T.copy 动态 shape 切片」契约；代价是循环体内引入
  段选择/尾块分支（`T.if`），取舍原则是 host 合规优先于循环体零分支。
- `q_seq` 无分块（逐行扫）；`heads` 必须 `BLOCK_H` 整除（wrapper 契约）。

### 5.5 数据搬运性能可行性（数据重排算子必填）

非数据重排算子，跳过。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| batch | block 级并行 | `T.Kernel(batch, is_npu=True)` | tile 级简化：一核一个 batch 行 |
| 查询行 s | 串行 | `T.serial(q_seq)` | 每行窗口不同（TIR 表达式），flash 状态行间独立 |
| head 块 h | 串行 | `T.serial(head_blocks)` | BLOCK_H=32 L0C 约束（§5.3） |
| KV 块 t | 串行（动态边界） | `T.serial(total_blocks)` | ori+cmp 双段拼接单循环，边界为 s 的 TIR 表达式 |
| 行内元素 | 向量化 | `T.tile.*` / `T.reduce_*` | UB 内整行并行 |

### 6.2 循环伪代码

```python
with T.Kernel(batch, is_npu=True) as (cid, vid):
    # 缓冲分配（§4.3）
    T.copy(idx, idx_ub)
    for s in T.serial(q_seq):                    # 查询行
        causal  = ori_seq - q_seq + s + 1        # TIR 表达式
        o_start = _ori_window_start(causal, ...) # max(causal-wl-1, 0) 等
        o_end   = _ori_window_end(causal, ...)
        cmp_thr = _cmp_thr(causal, ...)
        ori_blocks   = ceildiv(o_end, BLOCK_N)
        total_blocks = ori_blocks + ceildiv(cmp_thr, BLOCK_N)
        for h in T.serial(head_blocks):          # head 块
            # 装载 Q、初始化 flash 状态（m_i=sinks、sumexp=1、acc_o=0）
            for t in T.serial(total_blocks):     # 双段单循环
                is_ori  = t < ori_blocks
                seg_t   = is_ori ? t : t - ori_blocks
                seg_end = is_ori ? o_end : cmp_thr
                valid   = clamp(seg_end - seg_t*BLOCK_N, 0, BLOCK_N)
                if is_ori:                       # valid-extent 动态切片 + pad_value
                    T.copy(ori_kv[cid, seg_t*BLOCK_N : seg_t*BLOCK_N+valid, 0, :], kv_l1[:, :valid], pad_value=0.0)
                else:
                    T.copy(cmp_kv[cid, seg_t*BLOCK_N : seg_t*BLOCK_N+valid, 0, :], kv_l1[:, :valid], pad_value=0.0)
                # QK → 掩码 → flash-softmax → P cast → PV → acc_o 累加
            # epilogue: ÷sumexp → cast → 写 output[cid, s, ...]
```

### 6.3 流水线优化

- 双段 KV 循环**不使用** `T.Pipelined`：循环边界是 s 的 TIR 表达式（动态），
  软件流水不支持动态边界（§3.5.1）；且 AUTO_CV_SYNC 要求所有跨 pipe 拷贝
  集中在**一个 serial 循环**内（拆分循环复用同 workspace 会破坏
  cube/vec 同步点计数不变量，实测错序/死锁，§10.2 坑 #2）。
- 打包算子同构的 MIX ring 流水方案（Expert 形态，性能演进方向；骨架代码节选
  见本节下方）——任务空间 = bN2 × gSq
  （bN2 = batch×n2，gSq = group×q_outer_blocks），每核线性区间算术反解
  (bz, kvHead, qHead, qRowStart)，与打包算子 AICPU metadata 任务表同构：

```python
# ring 流水骨架：t ∈ [0, total_blocks + prelaunch)
#   t <  total_blocks：生产 C1(t)/V1(t)（QK、softmax 写 workspace）
#   t >= prelaunch   ：消费 C2(t-2)/V2(t-2)（PV、merge），滞后 2 拍
for t in T.serial(total_blocks + prelaunch):
    if t < total_blocks:
        with T.Scope("C"):
            # KV 预载 + QK gemm（细算 TODO(tile-level)）
            T.set_cross_flag("FIX", SIG_S_READY)
        with T.Scope("V"):
            T.wait_cross_flag(SIG_S_READY)
            T.set_cross_flag("MTE3", SIG_P_READY)
    if t >= prelaunch:
        with T.Scope("C"):
            T.wait_cross_flag(SIG_P_READY)
            T.set_cross_flag("FIX", SIG_O_READY)
        with T.Scope("V"):
            T.wait_cross_flag(SIG_O_READY)
# ring 不变式：ring_slots = prelaunch + 1 = 3；SIG_S_READY=0 / SIG_P_READY=1 /
# SIG_O_READY=2；workspace slot = t % 3 轮转；两 AIV 各持 BLOCK_H/2 行
```

block 级 2D 任务划分与线性区间反解（与打包算子 AICPU metadata 任务表同构；
`m_base_size=512` 为任务粒度，Q 常驻 L1）：

```python
MAX_CORES = 20
total_bnkv = batch * 1          # n2 == 1
q_outer_blocks = math.ceil(q_seq / m_base_size)
total_gsq  = group * q_outer_blocks      # group = heads（KV_N=1 约束）
q_work     = total_bnkv * total_gsq
used_core_num  = min(MAX_CORES, q_work)
tasks_per_core = math.ceil(q_work / used_core_num)

with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
    # 本核线性任务区间（静态均分；s2 不拆核）
    start_q = (q_work * cid) // used_core_num
    end_q   = (q_work * (cid + 1)) // used_core_num
    for task_local in T.serial(tasks_per_core):
        cur_q = start_q + task_local
        if cur_q < end_q:
            # 任务反解 (bN2, gSq) -> (bz, kvHead, qHead, qRowStart)
            cur_bnkv    = cur_q // total_gsq
            cur_gsq     = cur_q % total_gsq
            group_id    = cur_gsq // q_outer_blocks
            q_outer_idx = cur_gsq % q_outer_blocks
            q_head      = group_id          # n2 == 1: q_head == group_id
            q_row_start = q_outer_idx * m_base_size
```

### 6.4 尾块处理

见 §5.4：kernel 内 valid-extent 动态切片 + pad_value + 掩码双保险方案（无 host pad）。
窗口边界块的逐列可见性由 `lo2/hi2` 裁剪后的乘加式掩码表达：

```python
lo2 = min(max(lo - t * BLOCK_N, 0), BLOCK_N)   # 本块内有效列区间
hi2 = min(max(hi - t * BLOCK_N, 0), BLOCK_N)
valid = clamp(idx + 1 - lo2, 0, 1) * clamp(hi2 - idx, 0, 1)   # ∈ {0,1}
scores = scores * valid + MASK_FILL * (1 - valid)
```

---

## 7. 同步策略

### 7.1 同步模式

**自动同步**（tile 级实现）：`TL_ASCEND_AUTO_SYNC`（DMA/计算依赖）+
`TL_ASCEND_AUTO_CV_COMBINE`（CV 核分离）+ `TL_ASCEND_AUTO_CV_SYNC`
（跨 pipe workspace 中转与 cube/vec 同步点配对）。

### 7.2 同步点说明

自动同步下的**语句顺序约束**（AUTO_CV_SYNC 推导的隐性同步点，硬件实测总结）：

| 位置 | 同步约束 | 理由 |
|------|----------|------|
| P cast / UB→L1 先于 PV gemm | AIV 写 P（AIC 读）不得等待 AIC 的 acc_o 中转 | 避免环形跨核等待（§10.2 坑 #1） |
| PV 每块 re-init，O 累加在 UB | L0C→UB 读回后 `acc_o = alpha*acc_o + new` | L0C 无法跨块驻留累加状态 |
| 全部跨 pipe 拷贝在同一 serial 循环 | AUTO_CV_SYNC 按 workspace 配对 cube/vec 同步点计数 | 拆循环复用 workspace 破坏不变量（§10.2 坑 #2） |

block 级骨架（Expert 形态）的手动同步点：`T.set_cross_flag("FIX"/"MTE3", SIG_*)`
/ `T.wait_cross_flag(SIG_*)` 见 §6.3。

### 7.3 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,         # 依赖同步
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,   # UB 复用规划
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # 自动 CV 分离
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,      # 跨 pipe 同步
}
```

---

## 8. 融合算子设计

### 8.1 融合算子判定

**判定结果**: 是

**判定依据**: 含 QK/PV 两个 Cube GEMM，中间需 softmax/掩码/归一化 Vector
后处理，典型 CV 融合形态。

### 8.2 CV 交互设计（按编程模式）

**tile 级实现 = Developer 模式（消除 workspace/vid）**：
- `T.Kernel(batch, is_npu=True) as (cid, vid)`（MIX 双 Vector 核）
- 签名无 `workspace_*` 参数，装饰器无 `workspace_idx`
- Cube↔Vector 经 L0C→UB、UB→L1 片上中转，同步交给 AUTO_CV_SYNC pass
- 两个 AIV 各持 `BLOCK_H/2` 行（`vid` 分工，`sinks[h_lo + vid*hb : ...]`）

**block 级骨架 = Expert/混合回退参考**（§6.3）：

| workspace | Shape | dtype | 用途 |
|-----------|-------|-------|------|
| workspace_s | (cores, ring_slots, m_base_size, BLOCK_N) | f32 | C1 QK 分数 → V1 softmax |
| workspace_p | (cores, ring_slots, m_base_size, BLOCK_N) | bf16 | V1 P → C2 PV |
| workspace_o | (cores, ring_slots, m_base_size, dim) | f32 | C2 PV 输出 → V2 merge |
| workspace_meta | (cores, ring_slots, m_base_size, 3) | f32 | 跨 s2 合并元数据 |

**workspace_idx**: [4, 5, 6, 7, 8]（block 级骨架签名参数位置，仅回退写法）

### 8.3 Cube 核计算流程

```python
# Developer（tile 级实际代码）：L1 操作数 → L0C，直连片上 buffer
T.copy(<段内 valid-extent 切片>, kv_l1)   # 见 §4.4/§5.4
T.gemm_v0(q_l1, kv_l1, scores_l0c, transpose_B=True, init=True)  # QK
T.copy(scores_l0c, scores_ub)                                     # L0C → UB（CV 中转）
# ... AIV softmax ...
T.copy(p_ub, p_l1, tmp=p_ub_nz)                                   # UB → L1
T.gemm_v0(p_l1, kv_l1, acc_o_l0c, init=True)                      # PV（每块 re-init）
T.copy(acc_o_l0c, acc_o_new)                                      # L0C → UB

# 回退（block 级骨架）：经 workspace 中转 + set/wait_cross_flag 信号
# 见 §6.3 的双段 ring 流水
```

### 8.4 Vector 核计算流程

```python
# Developer（tile 级实际代码）：从片上 buffer 直读，无 workspace、无 vid 偏移计算
T.tile.mul(scores_ub, scores_ub, sm_scale)          # scale
# 乘加式窗口掩码（§6.4）
T.copy(m_i, m_old); T.reduce_max(scores_ub, m_i, dim=-1); T.tile.max(m_i, m_i, m_old)
T.tile.sub/exp/...                                  # flash-softmax 更新
T.copy(scores_ub, p_ub)                             # P cast（f32→bf16）
# acc_o = alpha*acc_o + acc_o_new；epilogue: T.tile.div + cast + T.copy 写出
```

### 8.5 pass_configs 配置

同 §7.3。block 级 Expert 形态不设 AUTO_CV_*，改用手动信号（骨架文件注释）。

### 8.6 注意事项

- 核间流水线与核内流水线不能同时开启（本设计均不用软件流水，规避）。
- 语句顺序必须避免环形跨核等待（§7.2）。
- block 级 workspace ring 的 slot 轮转与信号编号必须一一对应（骨架常量
  `SIG_S_READY=0 / SIG_P_READY=1 / SIG_O_READY=2`，`ring_slots = prelaunch + 1`）。

---

## 9. 验证方案

用例、golden、精度容差与性能哨兵按 [`design-template.md`](design-template.md) §9
在交付阶段产出；本文不含测试数据、精度与性能实测结果。

**输入分布要求**：测试数据必须同时覆盖正分布（`rand * 2`）与含符号分布
（`rand * 4 - 2`）两类。含负输入使 softmax 输出过零，近零元素逐元素相对误差被放大，
但精度判定采用混合容差判据（逐元素 `|actual - golden| ≤ atol + rtol·|golden|`），
atol 项正是为近零元素设计的兜底；含负分布下混合容差判据不通过时，**按真实精度缺陷
修复**，禁止用限定输入符号规避。

---

## 10. 风险点与注意事项

### 10.1 已知约束

- **CSA 未覆盖**：`cmp_sparse_indices` **块粒度** gather（元素 × sparseBlockSize =
  行基址，-1 项跳过；A2/A3 的有效条目由索引张量内的 -1 标记表达，`cmp_topk_length`
  必须传空）需片上按块 gather（`T.tile.gather` 或按块动态 GM 拷贝），
  本设计显式扩展点；传入 CSA 输入时 wrapper 抛 `NotImplementedError`，禁止静默降级。
- **D=512 / N2=1 硬约束**：BLOCK_H 由 L0C 容量倒推（§5.3），其他 head_dim
  需重新核算。
- **helper 函数陷阱**：tilelang-ascend trace 会**静默丢弃** kernel body 内
  调用的 helper 函数的语句（返回值 TIR 表达式的标量 bound helper 可用）——
  kernel body 必须直写（本实现 `_ori_window_*` 只返回表达式，符合约束）。
- **编译期配置形态**：每 shape 一个 kernel 实例，不适合频繁变 shape 的在线
  场景（block 级骨架的动态任务网格是演进方向）。

### 10.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| 环形跨核等待死锁 | AIV 写 P 排在 AIC acc_o 中转之后 | 挂死 | P cast / UB→L1 先于 PV gemm（§7.2） |
| 拆循环复用 workspace | 跨 pipe 拷贝拆到第二个 serial 循环 | AUTO_CV_SYNC 同步点计数不匹配，错序/死锁 | 全部跨 pipe 拷贝集中一个 serial 循环 |
| helper 语句被静默丢弃 | body 内调用含 T.copy/T.tile 的函数 | 计算缺失，精度错误且无报错 | 语句直写 body，helper 只返回 TIR 表达式 |
| 加法式掩码大数相消 | `scores + MASK_FILL`（fp32 有效值被大数吞没） | 有效列被吞 | 乘加式 `scores*valid + MASK_FILL*(1-valid)`（§6.4）；MASK_FILL 取 -2e38（与打包算子参考实现一致） |
| L0C 超限编译失败 | BLOCK_H=128 配 D=512（256KB > 128KB） | 编译失败 | BLOCK_H=32（§5.3） |
| decode 精度超标 | 输入含负值，softmax 输出过零 | 近零元素逐元素相对误差放大 | 按混合容差判据（atol+rtol）评估；不通过按真实精度缺陷修复（§9），禁止限定正分布输入 |
| HCA/CSA 结果系统性错位 | `cmp_seq` 用 ceil(s2/ratio) | cmp 阈值恢复超出真实 ori_len | `cmp_seq = max(s2 // ratio, 1)`（floor） |

### 10.3 特殊场景处理

- **prefill/decode 统一**：窗口界为 s 的 TIR 表达式，decode 行（q_seq=1 或 s=0）
  自动折为编译期常量，同一份 kernel 覆盖两种形态。
- **纯 SWA 无 cmp_kv**：wrapper 传占位 `cmp_kv`（kernel 内 `cmp_blocks == 0`，
  永不读取），避免签名分支。
- **无 sinks**：传 `[N1]` 零向量（golden 无 sinks 语义等价）。
- **HCA 压缩长度**：取 `cmp_seq = max(S2 // ratio, 1)`，与打包算子
  `seqused_cmp_kv` 推导一致（ceil 会使 cmp 阈值超出真实 ori_len）。
