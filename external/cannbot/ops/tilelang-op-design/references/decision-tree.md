# 算子特征分析决策树（Ascend 版）

## 目录

- [1. 函数设计原则](#1-函数设计原则)
- [2. 决策树](#2-决策树)
- [3. NPU 硬件约束](#3-npu-硬件约束)
- [4. API 映射规则](#4-api-映射规则)

> 本项目为 TileLang-Ascend，与 GPU 版 TileLang 在 Kernel 维度、threads、循环边界、GEMM API、内存分配等方面有显著差异。完整对比与约束清单见 SKILL 子目录索引下的 `ascend-constraints.md`。外部参考实现仅用于理解数学逻辑，API 映射必须查阅本项目。

---

## 1. 函数设计原则

1. **维度参数自推导**：算子调用函数（如 `conv_im2col_gemm`）应从输入 tensor shape 提取 B/C/H/W 等维度，不依赖模块级全局变量。这保证多场景顺序测试时不发生变量污染。
2. **Host 预处理显式声明**：若计算的一部分在 Python 侧完成（如 im2col），必须在 §1 算法描述和 §4 数据流中明确标注。

## 2. 决策树

**重要**：`T.reduce_sum/max/min` 和 `T.tile.*` 在 Developer 和 Expert 模式下**都可使用**。模式选择取决于是否需要手动控制内存层级和同步，而非使用了哪个 API。

```
算子数学公式
├─ 含 matmul / @ / 矩阵乘
│   ├─ 仅 matmul → 纯 Cube
│   │   模式: Developer (推荐) 或 Expert
│   │   API: T.gemm_v0 / T.mma
│   │   内存: GM→L1→L0A/L0B→L0C→GM
│   │   pass_configs: 全开启（Developer）
│   │   Kernel: T.Kernel(任务数, is_npu=True) as (cid, _)
│   │
│   └─ matmul + element-wise 前处理/后处理 → CV 融合算子
│       ├─ Developer 模式（推荐，默认消除 workspace/vid）
│       │   模式: Developer + AUTO_CV_COMBINE
│       │   API: T.tile.* (Vector) + T.gemm_v0 (Cube)
│       │   内存: GM→L1→L0C→片上直连→UB→GM（默认无 workspace 中转）
│       │   pass_configs: AUTO_SYNC + AUTO_CV_COMBINE + AUTO_CV_SYNC
│       │   同步: AUTO_SYNC + AUTO_CV_SYNC 自动处理
│       │   V 核: threads=2 自动并行（消 vid）；复杂场景才回退 workspace+vid
│       │   写法: 见 tilelang-programming-model-guide mode-examples.md §6
│       │
│       ├─ Expert 模式（极致性能）
│       │   模式: Expert + T.Scope("C"/"V") + T.set_cross_flag
│       │   同步: 手动核间同步（T.set_cross_flag / T.wait_cross_flag）
│       │
│    典型算子: W4A8 GEMM, Flash Attention, 量化 GEMM
│
├─ 含注意力 / flash-softmax（Attention、FlashAttention、MLA 变体）
│   判定: QK^T + online softmax + PV 三段式，matmul + 归约 + element-wise 的 CV 融合
│   ├─ 标准 Attention（FlashAttention 等）
│   │   → 上方「matmul + element-wise → CV 融合算子」分支（Developer 默认）
│   └─ 稀疏 MLA 变体（SparseFlashMla：输入含 cmp_kv / cmp_sparse_indices）
│       → 强制分支：见本节下方「稀疏 MLA 注意力变体结构选型」
│       权威示例: examples/sparse_flash_mla_design.md
│
├─ 含卷积（nn.Conv* / F.conv*）→ Conv 类算子，强制三模式选型（详见下文「Conv 类算子结构选型」）
│   结构: img2col + GEMM（LoadData3D 随路展开 / T.tile.im2col 设计记号）
│   模板: 先读 examples/conv2d-template.md，再走 design-template.md 通用骨架
│
├─ 纯 element-wise（逐元素运算）
│   参考: examples/elementwise/*.py, examples/activation/*.py
│   ├─ 单步运算 → Developer 模式
│   │   API: T.Parallel + 算术符号
│   │   内存: T.alloc_shared（编译器映射到 UB）
│   │
│   └─ 多步运算（如 softmax、layer_norm）
│       参考: examples/softmax/*.py, examples/normalization/*.py
│       ├─ 需精细 buffer 控制 → Expert 模式
│       └─ 无需精细控制 → Developer 模式
│
├─ 含归约（reduce_sum / reduce_max / reduce_min）
│   参考: examples/reduce/*.py
│   API: T.reduce_sum / T.reduce_max / T.reduce_min
│   内存: T.alloc_shared → UB
│
├─ 含分组/动态批次
│   参考: examples/grouped_gemm/*.py（重要！）
│   关键技术:
│   - block_metadata 预计算表（替代三维 Kernel）
│   - 静态循环边界 + 条件判断（替代动态边界）
│   Kernel: T.Kernel(total_blocks) + 手动索引分解
│
├─ 池化类（AvgPool/MaxPool/AdaptivePool 及反向 Grad）
│   判定: 滑动窗口 reduce（前向）/ scatter-add（反向）
│   领域参考: references/pooling/（5 篇设计篇，⚠️ 命中即先读，随本 skill 携带）
│   ├─ 前向 reduce: layout-strategy.md（NDHWC 布局）、row-granularity.md（行粒度 tile）、
│   │   reduce-d-fastpath.md（KH==1&&KW==1 快路径）
│   ├─ adaptive 窗口（AdaptiveAvg/MaxPool）: adaptive-avg-pool3d-lessons.md
│   └─ 反向 grad: backward-patterns.md（窗口闭式公式、scatter-add、divisor 共享、block 语义对调）
│   转译/实现篇（设计完成后）: tilelang2ascend-translator Pooling 类别
│   （references/pooling-patterns/，含 tilelang-translation/ub-management/alignment-guards/
│   precision-patterns/backward-implementation/grad-v2-lessons/pooling-anti-patterns）
│
└─ 其他复杂算子
    强制步骤: 先搜索本项目 examples/
```

**稀疏 MLA 注意力变体结构选型（SparseFlashMla 强制分支）**

稀疏 MLA 注意力（算子名 `SparseFlashMla` / 输入含 `cmp_kv`、`cmp_sparse_indices` 特征，平台 arch22 = A2/A3/910B）必须按以下顺序显式选型，并在设计文档中记录结论：

**场景判定（先于一切）**——由输入张量存在性唯一确定（无 cmp_topk 之类判定 attr）：

| scenario | ori_kv | ori_sparse_indices | cmp_kv | cmp_sparse_indices |
|----------|--------|--------------------|--------|--------------------|
| SWA | ✓ | – | – | – |
| SWA 稀疏 ori_kv（DSpark） | ✓ | ✓（配 ori_topk_length） | – | – |
| HCA | ✓ | – | ✓ | – |
| CSA | ✓ | – | ✓ | ✓ |
| ORI_CMP_SPARSE | ✓ | ✓ | ✓ | ✓ |

A2/A3 平台约束（tiling 层 OP_CHECK）：HCA `cmp_ratio=128`、CSA `cmp_ratio∈{1,2,4}`、
CSA/HCA `cmp_mask_mode=3`、DSpark `ori_mask_mode=0` 且 win 非负、非 DSpark
`ori_mask_mode=4`/`win_left=127`/`win_right=0`；D=512、KV_N=1、Q_N∈{1..128}。

**硬约束**（算子定义，违反即不成立）：

- `D = 512`、`KV_N = 1`（K/V 共享存储）、`N1 % BLOCK_H == 0`
- `BLOCK_H` 由 L0C 容量倒推：PV 输出 tile `[BLOCK_H, D]` fp32，`BLOCK_H=128, D=512` 时 `128×512×4 = 256KB > 128KB` L0C 上限 → 取 `BLOCK_H=32`（64KB + QK 分数 16KB = 80KB ≤ 128KB），head 维以 32 行分块循环
- `KV_N=1` 时 `group = heads`，q_head 与 kv_head 一一对应，无 head-sharing 分派

**TileLang 覆盖范围门禁**：

- SWA / HCA → 本分支完整支持（双段 KV 单循环 + flash-softmax），**权威示例 `examples/sparse_flash_mla_design.md`**（910B 硬件验证通过）
- CSA（`cmp_sparse_indices` **块粒度** gather：元素 × sparseBlockSize = 行基址，-1 项跳过）→ **显式扩展点**：需片上按块 gather（`T.tile.gather` 或按块动态 GM 拷贝）；设计未覆盖时必须在「风险点」声明，wrapper 对 CSA 输入显式抛 `NotImplementedError`，禁止静默降级
- 与 DeepSeek MLA（`q_nope/q_rope`、Dqk≠Dv）区分：本分支无 rope 拆分、K/V 共享存储；命中信号与契约见 `community/ops/tilelang2ascend-translator/references/attention-patterns/references/sparse_flash_mla_package_op.md`

**强制设计规则**（不满足须在「风险点」写明理由）：

- **双段 KV 拼单循环**：ori 窗口段 + cmp 前缀段必须在**同一个 `T.serial`** 中按块拼接（共享 flash-softmax 状态机）；s2 不拆核（状态跨块串联），多核并行只能沿 (batch, head, q 行) 切
- **掩码而不是跳过**：越界 KV 列不裁剪循环，用乘加式掩码 `scores*valid + MASK_FILL*(1-valid)`（`MASK_FILL=-2e38`，与打包算子参考实现一致，exp 下溢为 0）；**禁止加法式** `scores + MASK_FILL`（fp32 大数相消吞掉有效列）；窗口界为查询行 s 的 TIR 表达式，prefill/decode 同一份代码
- **flash-softmax 状态机契约**：sinks 为虚拟 key——`m_i` 初值 = sinks（`T.copy(sinks[...], m_i)`）、`sumexp` 初值 = 1（`T.tile.fill`，= exp(sinks-m_i)）、sinks 只进分母不进分子；epilogue `out = acc_o / sumexp`
- **动态边界不用软件流水**：双段循环边界是 s 的 TIR 表达式（动态），该循环用 `T.serial` 不用 `T.Pipelined`；`AUTO_CV_SYNC` 要求全部跨 pipe 拷贝集中在**一个 serial 循环**内（拆循环复用 workspace 破坏 cube/vec 同步点计数不变量，实测错序/死锁）
- **语句顺序防环形等待**：AIV 写 P（AIC 读）不得排在 AIC 的 acc_o 中转之后——P cast / UB→L1 先于 PV gemm；PV 每块 `init=True` re-init，O 累加在 UB（`alpha*acc_o + new`）完成
- **helper 函数陷阱**：tilelang-ascend trace 会静默丢弃 kernel body 内调用的 helper 函数的**语句**（返回 TIR 表达式的标量 bound helper 可用）——kernel body 必须直写
- **非整除尾块（decode 形态）**：S 维不整除 BLOCK_N 时**禁止 host pad/拼接**（`torch.zeros`+切片赋值、`torch.cat`/`F.pad` 均违反 ascend-constraints §5，且在 NPU 上触发 aclnn，评测环境可能裁剪）。kernel 内按 `valid = clamp(seg_end - seg_t*BLOCK_N, 0, BLOCK_N)` 用 `T.copy` BufferRegion 动态切片搬入 `kv_l1[:, :valid]`（`pad_value=0.0` 填 pad 行），pad 列落在所有查询窗口之外、由乘加式掩码双保险清零；双段在同一 `T.serial` 内经 `T.if` 选段，共享 flash-softmax 状态
- **输入分布要求**：softmax 类算子测试数据必须同时覆盖正分布（`rand*2`）与含符号分布（`rand*4-2`）；含负输入使输出过零、近零元素 rel_err 放大，由混合容差判据（atol+rtol）兜底，超判据按真实精度缺陷修复，禁止限定输入符号规避

**完整 sparse_flash_mla 设计样例**：权威示例 `examples/sparse_flash_mla_design.md`——按 `examples/design-template.md` 模板 §1–§10 填写（§11 交付清单/proto.yaml 不在示例内），含场景判定（§1.2/§2）、CV 同步不变量（§7/§8）与坑位清单（§10.2）。生成 MLA 类算子时先读该示例，再走通用骨架。

### Conv 类算子结构选型（conv2d / img2col+GEMM 强制分支）

Conv 类算子（`nn.Conv*` / `F.conv*`）必须按以下三模式显式选型，并在设计文档中记录结论：

| 模式 | 适用条件 | 省去 |
|------|---------|------|
| fused img2col（LoadData3D 随路展开，用 `T.tile.im2col` 设计记号表达） | Ci%C0==0（真实 Ci 不吃 pad 通道）且 xNz+B 面板 ≤ 480KB | im2col GM 物化（最大收益） |
| direct 1x1 flipped GEMM（x NCHW 本身即 `[K=Ci, M=H*W]` 操作数） | kH=kW=1, s=1, p=0, Ci%C0==0, (H*W)%C0==0 | wrepack + im2col 两级发射 |
| normal（im2col 物化 + K-major wrepack） | 通用回退 | — |

约束：
- **C0 = 32B/元素宽**（fp16/bf16=16，fp32=8；NZ/ZN 分形 `[16 行, C0 列]`）：现行实现 fused/direct 只路由 fp16/bf16、gate 硬编码 16；窗口化 band 路径经 xtrans_pad 上片 CiPad，不受 Ci 对齐限制。fp32 输入先按 conv2d-template.md §12 前置路由。
- **路由判定顺序**：先判 direct 1x1（条件最严格，命中即省 wrepack + im2col 两级
  发射，与 fused 重叠场景严格更优）→ 再判 fused img2col（L1 预算用物理分配而非
  逻辑 Co）→ 否则 normal 回退；与 `examples/conv2d-template.md` `select_mode`
  实现一致（表格按收益排序，非判定顺序）
- 满足 fused/direct 条件却退回 normal 时，必须在设计文档「风险点」中写明理由
- direct 1x1 的 `(H*W)%C0==0` 源于 B 操作数（raw x NCHW `[Ci, H*W]`）GM→L1 走 Nd2Nz 通路要求每行 32B 对齐（fp16/bf16 下 = 16 元素）：H*W 非对齐时 raw x 行宽无法 32B 对齐、Nd2Nz 加载失效，此时须回落 normal/fused（xpad UB-relay 补齐方案正确但实测 net negative，见 conv2d_base.cpp P0-b）
- **禁止默认标量逐元素 gather** 的 im2col 设计；k'=(ci,kh,kw) 解码按段连续块/向量化表达；标量豁免仅限占比 <5% 的偶发小尾块（主数据路径不可豁免）
- dtype 适配（fp32 走原生 fp32 cube / bf16 拆分替代）时必须声明「保留的优化结构清单」：fused/normal 模式、零标量、批量转置、epilogue 跨行块处理逐项标注 保留/丢弃+理由
- 权重 repack 结果是（weight 内容, shape, dtype）的确定函数，设计中标注可由 host 侧 LRU 缓存
- **尾块处理：host 不 padding 到整块再 crop**——优先 kernel 内尾块：GM 侧只访问 valid 范围（精确 slice 或 T.copy 的 pad_value），UB/L1 按完整 tile footprint 分配、无效 lane 填幺元或 mask，输出只回写有效矩形（裁剪发生在 kernel 写回 extent 内，不是先写 padded 输出再由 host 切片）；GEMM K 尾块 A/B 片段补 0（0 对累加是幺元），M/N 尾块只影响被丢弃的行列。host padding 仅为"后端路径确实无法表达 valid 范围"时的受控 fallback，须把额外分配与拷贝计入成本；归约/归一化类算子 padding 行必须幺元填充或 mask，分母用逻辑长度；padding 不改善并行度，M < block 时应缩小 block、沿 N/batch 分核，或 M 极小时改向量/GEMV 路径，而不是 padding
- **完整 conv2d 设计模板**（§0 三模式选型 + API 映射 + Tiling + K 顺序契约 + 风险点 + 交付清单，权威示例 `archive_tasks/conv2d_base/design/`）见 [`examples/conv2d-template.md`](../examples/conv2d-template.md)。生成 conv2d 类算子时先读本模板再走 `examples/design-template.md` 通用骨架。

## 3. NPU 硬件约束

**⚠️ NPU 硬件约束（必查）**：

设计 Tiling 策略时，必须考虑：

1. **分形限制**（Fractal Limits）：
   - L0A: M ≥ 16, K ≥ 32
   - L0B: K ≥ 32, N ≥ 16
   - L0C: M ≥ 16, N ≥ 16
2. **对齐要求**：
   - UB/L1: 32 Byte
   - L0A/L0B: 512 Byte
   - L0C: 64 Byte
3. **存储大小上限**：
   - L0A/L0B: 64KB
   - L0C: 128KB
   - L1: 512KB
   - UB: 192KB

违反约束会导致编译错误或运行时错误。详见 `../../tilelang-api-best-practices/references/api-kernel-memory.md`。

## 4. API 映射规则

| 类别 | Ascend 专用 API（推荐） | 通用 API（本项目不推荐/不支持） |
|------|------------------------|-------------------------------|
| GEMM | `T.gemm_v0` | `T.gemm`（可能不支持） |
| 内存分配（Expert）| `T.alloc_L1`, `T.alloc_L0C`, `T.alloc_ub` | - |
| 内存分配（Developer）| `T.alloc_shared`, `T.alloc_fragment` | - |
| Kernel | `T.Kernel(一维, is_npu=True)` | `T.Kernel(三维)` ❌ |
| 同步 | `T.barrier_all()`, `T.Scope("C")` | 自动同步（Developer 模式） |
| 循环 | `T.serial`, `T.unroll` | `T.Pipelined(动态边界)` ❌ |