# ChunkCat AscendC Tiling 设计经验

> 所属类别：Conversion 数据转换类；场景判定与同类子场景索引见 [patterns.md](patterns.md)。

本文件面向**开源项目 skill 的使用者**：当你需要为 ChunkCat 类算子（dim=0 切块 + 沿下一维级联 + 零填充）
设计 Ascend C 的 Host Tiling 与 Kernel 时，按本文的“问题 → 设计 → 实现方式 → 效果”路径即可复现一套
经过验证的高性能实现。文中不依赖任何本地文件路径或迭代编号，只保留可复用设计逻辑。

---

## 0. 基础环境与适用边界

### 0.1 基础环境

本经验在以下环境验证通过，具体数值和 API 行为均以该环境为基准：

| 项目 | 基准环境 |
|---|---|
| NPU（本文基准证据平台） | Ascend 910B3（DAV_2201 类），UB 192 KiB；性能数据在单一健康 NPU 设备上采集 |
| NPU（950PR 追加证据） | Ascend950PR（DAV_3510 / arch35），UB 248 KiB（平台 INI `ub_size=253952`，运行时以 `GetCoreMemSize(CoreMemType::UB, ...)` 为准），AIC:AIV = 28:56 |
| CANN | 9.1.0（提供 `aclnnChunkCat`，且 AscendC 的 `DataCopyExtParams`、`Adds` 等行为按此版本；910B3 与 950PR 同版本，但 arch35 代码路径不同，需分别实测） |
| torch_npu | 910B3：2.9.0 ／ 950PR：2.7.1.post10 |
| 算子调用 | 独立编译的 AscendC kernel（`torch.ops` 扩展），对标 CANN `aclnnChunkCat` 参考 |
| 性能口径 | msprof quick（Task_Duration），warmup=3，repeat=1 |
| 平台参数真源 | `npu-arch` skill（本文只写相对预算公式与平台锚，不复制硬件真源数值） |

> 其它 CANN / torch_npu 版本或其它 SoC（如 910B2、950 等）上，`DataCopyParams` 字段宽度、
> `DataCopyExtParams` 多行支持、`Adds` 的 bf16 限制、事件同步开销都可能不同，必须重新验证，
> 不能直接套用本文数值结论。

### 0.2 适用边界

本经验适用于满足以下全部条件的算子/输入：

- 语义：沿 `dim=0` 切块 + 沿 dim 后一维级联 + 不足 chunk 处零填充；
- 输入：1–4 个张量，全部连续（contiguous）、同一 dtype、rank 1–8；
- dtype：fp32 / fp16 / bf16；
- 输出：`[num_chunks, sum_i(chunk_i * prod(shape_i[1:]))]` 的连续新张量；
- 性能特征：data movement 主导，无跨核通信，无元素值计算。

### 0.3 不适用 / 不能直接推广

- `dim != 0`、非连续输入、输入组内 dtype 不一致、输入 tensor 数 > 4；
- 需要改变元素值的运算（计算受限算子不能照搬“搬运优先”的结论）；
- 任意 stride / mask / 负 stride 等与 ChunkCat 语义无关的更一般 StridedSlice / Gather / Scatter；
- **其它 SoC 或 CANN 版本**：以下结论均需重新验证，不能直接沿用——
  1. **整体结构**：本文 §4–§7 是 910B3 实现的结构（元素级 `Adds`、bf16 `ReinterpretCast<int16_t>`、`FAST_ROW_E → SEGMENT_BATCH → SEGMENT_SINGLE` 三路由、二维 block 网格 + 列优先 UB tile、输入/输出 1:1）。**Ascend 950PR（DAV_3510）的推荐实现未采用其中任何一项**（改为字节级统一路径、按输出字节单元开核、S 槽复用 + 独立零区），两条路线不得互相套用参数；
  2. **UB 预算与 1:1 切分**：本文按「预留约 8 KB 后输入/输出 1:1」划分；该结构与 `colLimit` 族公式（§5.1–§5.3）在 950PR 上**未重测**。950PR 的容量核算改用「`S·(U+slack) + Z ≤ 可用 UB`」，见 §5.3。**注意**：INI 的 `ub_size`（253952 = 248 KiB）是物理容量，运行时可用值可能更小（另一工作区实测运行时属性仅 216 KiB）；**预算一律取运行时接口/属性值**；
  3. **同步开销**：本文「1 行/tile 时事件开销与搬运同量级」的估计按 910B3 计时；950PR 推荐实现改用 5 槽深流水 + 延迟一格写出。注意 **910B3 的「双缓冲/ping-pong 是反模式」在 950PR 上不成立**（950PR 实测用更深流水），两侧结论不可互引；
  4. **stride 字段宽度与行距**：`DataCopyExtParams.srcStride/dstStride` 在 950PR 上为 `int64_t`（910B3 为 `uint32_t`）；`DataCopyPad` 新增 `PaddingMode` 模板参数（`Compact` 使 UB 侧稠密、只在整块末尾补一次 32B）——该模板参数**只对 arch35 系可用，910B3 上无法编译**；
  5. **多行 `DataCopyExtParams` 行为**：910B3 上软件回退极慢甚至 hang（见 §7.4/§8），950PR 上**未复现**该回退；但 950PR 文档给出的 `blockCount` 上限为 `[0,65535]`（910B3/A3 为 `[0,4095]`），且「~29–32 行隐式上限、超出静默丢零」的说法在 950PR 上已被真机证伪。两平台不得互推；
  6. **核数**：核数相关经验一律改为 `GetCoreNumAiv()` / `aclrtGetDeviceInfo(ACL_DEV_ATTR_VECTOR_CORE_NUM, …)` 运行时取值（910B3 与 950PR 的 AIV 核数不同；950PR 本机 SKU 实测 56）；
  7. **平台专属机制**：NDDMA（`MultiCopyParams`）、`SetLoopModePara`、RegBase/MicroAPI、SIMT 在该 950PR 实现中**均未采用**，只有接口/文档级存在性证据，**不得写成「已验证可跑」**；
- 需要跨核通信或非规则输出 ownership 的算子。

### 0.4 在算子生成流程中的位置

本文件是 **AscendC 实现阶段** 的设计参考（对应生成流程中的 Phase 3/4 的 native AscendC 路线）。
它假设阅读者已经完成算子语义分析、case 生成和工程脚手架初始化，能读懂
`torch.ops` 扩展、`TPipe/TBuf/DataCopy` 等 AscendC 基础 API。

- 如果生成路由选择 **ascendc_native**：本文件的 §3–§6 可直接作为 Kernel/Tiling 设计依据，
  §7 提供了从零搭建可编译工程的最小骨架。
- 如果生成路由选择 **tilelang2ascendc**：本文件仍可作为生成后 AscendC kernel 应达到的结构目标，
  但不能替代 TileLang 阶段 3 的 DSL 设计表达（block/tile 映射、copy/fill 原语等）。
  ChunkCat 是搬运主导 + 零填充 + 需要精确 Padded DMA 的算子，通常会被路由策略判为更适合
  native AscendC；只有在路由策略明确选择 tilelang 时，才需要另行补 TileLang 设计。

---

## 1. 算子问题定义

### 1.1 语义

- 输入：1–4 个连续张量，同一 dtype（fp32 / fp16 / bf16），rank 1–8。
- 参数：`dim`（仅支持 0）、`num_chunks`。
- 计算：每个输入按 dim0 切成 `num_chunks` 块（`chunk = ceil(dim0/num_chunks)`），
  再把所有输入的对应块沿 dim 后一维拼接；rank=1 时退化为“切块后堆叠成二维矩阵”；末尾不足一个 chunk 的部分补零。
- 输出：`[num_chunks, sum_i(chunk_i * prod(shape_i[1:]))]`，dtype 与输入一致。
- 本质：**不改变元素值**，是带 padding 的 data movement / concat。

### 1.2 性能本质：搬运受限，不是计算受限

在 Ascend 910B3 上的性能基线如下（msprof 深度 profiling，阶段耗时占比）：

| 阶段 | ChunkCat kernel | CANN 参考 |
|---|---:|---:|
| MTE2（GM→UB） | **51%–73%** | 49%–60% |
| MTE3（UB→GM） | 22%–28% | 35%–45% |
| Vector | 4%–7% | 8%–40% |
| Scalar | 4%–12% | 3%–28% |

结论：**绝大多数时间花在 MTE2/MTE3 数据搬运上，Vector/Scalar 计算占比很小。**
这决定了后续所有设计判断：

- 优化方向应围绕“减少搬运次数、增大每次搬运的连续段、减少搬运之间的同步等待”。
- 任何增加搬运次数或把 tile 切得更碎的方案，都必须先证明重叠收益能覆盖额外开销。

---

## 2. 为什么必须“先分路由”而不是单一路径

### 2.1 问题

ChunkCat 的输入几何差异极大，同一个 kernel 会同时遇到：

1. **32B 对齐差异**：`prod(shape[1:])` 可能是 32B 元素宽度的整数倍，也可能不是。非 32B 对齐时，
   直接整段搬运会越界或要求地址对齐补齐，必须走 Padded DMA。
2. **padding 差异**：`num_chunks` 不整除 dim0、`num_chunks > dim0`、或多个输入宽度不一致时，
   输出行的某些段完全由零组成。这些零**必须被显式写入**，否则输出里是未初始化内存。
3. **多 tensor 段差异**：一行输出可能由 2/3/4 个输入 tensor 的 segment 拼接而成。
   segment 数量、宽度、是否连续，决定了一次 CopyOut 能做多大。
4. **行宽差异**：单 tensor 列宽可能小于一个 UB tile（1D 大向量、小 shape），也可能远大于一个 tile
   （宽 2D，如 4096 列）。搬运策略应随之不同。
5. **stride 限制差异**：本文多行批量使用 `DataCopy + DataCopyParams` 场景，其
   `blockLen/srcStride/dstStride` 按 **32B 块数**理解。注意 `DataCopyPad` 场景不同：
   `DataCopyExtParams.blockLen` 为 **1 字节**；`srcStride/dstStride` 按操作数位置区分
   （GM 侧 1 字节、UB 侧 32B 块）。宽 2D 行间 gap（按本文多行 `DataCopy` 的 32B 块计）
   很容易超过 uint16 上限，多行批量 DMA 在硬件上无法表达，必须退到单行。

如果只写一条“万能路径”，结果是要么在这些差异处精确性失败，要么大量 case 被迫走最保守的慢路径。

### 2.2 设计：把“选路”变成架构，而不是 if-else 补丁

路由架构的职责是先把 case 区分开，再在每条 route 内部做针对性的搬运设计：

```text
Host: shape/dtype/num_chunks
  → 归一化为输出矩阵 + 路由特征（width、chunk、对齐标志，colLimit 定义见 §5.2）
  → 计算 block 网格和 UB tile（这是“调度”）
Kernel: 每个 tile 的每一行
  → 先尝试 FAST_ROW_E：整行多 tensor segment 合并搬运
  → 未命中 → 回退 ProcessTensorSegment：逐 tensor / 逐段处理
```

这样设计的原因：

- **正确性和性能分离**：特化路径（FAST_ROW_E）只处理最有利的几何，任何条件不满足就整体回退；
  正确性由最保守的 fallback 保证，而不是让每条路径都尝试处理所有 corner case。
- **同步策略可独立最优化**：FAST_ROW_E 的“多 segment 批量同步”只在所有 segment 对齐且连续时
  才安全；把它下沉到通用路径会引入对齐和 padding 风险。
- **每条路径的 tile/搬运基块不同**：1D 大向量适合多行批量 `blockCount>1`；宽 2D 必须列切分、
  单行搬运；不把它们混在一起，tiling 参数才不会互相污染。
- **可预测回退**：route 优先级 F1 → F2 → F3 固定，任一特化不满足就落到更通用的路径。
  这样优化一条快速路径时不会隐式改变对其它 case 的语义。

### 2.3 效果

- 常见的大块 case 走 FAST_ROW_E：整行多 segment 合并搬运，只做一次 CopyOut。
- 1D/窄行多行 case 走 SEGMENT_BATCH：一次 DMA 搬多行，减少 MTE2/MTE3 次数。
- 非对齐/带 padding 的 case 走 SEGMENT_SINGLE：牺牲性能保正确。
- 最终基准：全 case geomean **1.1323x**；剔除一个异常的 1D 大向量 case 后 **0.9657x**。

---

## 3. 路由特征：Host 应该先算清什么

Host 只需要输入元数据，就能决定绝大多数路由。一次算清后写入 TilingData，Kernel 只读标志和尺寸，
不在 Device 上重新计算或猜测。

| 特征 | 定义 | 为什么需要它 |
|---|---|---|
| `numInputs` | 实际 tensor 数 1–4 | 决定一行输出最多有几个 segment，也决定 Kernel 遍历段数 |
| `numChunk` / `outputRow` | 输出行数 = `num_chunks` | block 网格的行方向规模 |
| `outputCol` | 输出总列宽 = `sum(chunk_i * prod(shape_i[1:]))` | block 网格列方向规模，以及 CopyOut 行 stride |
| `width[t]` | 输入 t 在输出矩阵中的列宽 | 各行输出列区间的 segment 边界 |
| `chunkSize[t]` | `ceil(dim0_t / numChunks)` | 该输入每个 chunk 对应的源行数上限；最后不足即为 padding 行 |
| `restSize[t]` | `prod(shape_t[1:])` | 列对齐判断依据（是否为 32 字节整元素数 `32/itemSize` 的整数倍） |
| `isAllAlign` | 每个输入 `restSize` 都是 `32/itemSize` 的整数倍 | 决定是否所有 segment 天然满足 32B 对齐；也影响 block 网格搜索的行块上限 |
| `isHalfAlign` | `!isAllAlign && 任一输入 restSize 对齐` | 混合对齐标记，Host 写入 TilingData 以保持 ABI 完整 |
| `isOneConcat` | `outputRow == 1 && !isAllAlign` | 单行且非全对齐的标记，Host 写入 TilingData 以保持 ABI 完整 |
| `dtypeFlag` / `itemSize` | fp32=4B，fp16/bf16=2B | 所有容量、colLimit、stride 的计算基数 |

**关键点**：`isAllAlign` 不是“整个算子只有一个路径”的开关，而是 block 网格搜索时的能力信号。
真正是否走 FAST_ROW_E，由 Kernel 每行根据 segment 级对齐和连续性判定。

**后文符号统一**：`srcRowWidth` 恒等于 `width[t]`（该输入张量展平后的 chunk 槽宽），
`dstRowWidth` 恒等于 `outputCol`。`copyCols` 是某一行中该 segment 的有效列数，
`fullSegCols` 是该 segment 在输出中占据的总列宽（含右侧零填充）。这些符号按最终 Kernel 的字段语义使用。

---

## 4. Kernel 内部路由：三条路径

### 4.1 路由总表

| 优先级 | Route | 准入条件 | 实现 | fallback |
|---:|---|---|---|---|
| 1 | **FAST_ROW_E**（整行合并快速路径） | 一行内每个 segment 同时满足：`copyCols>0`；`copyCols*sizeof(T)%32==0`；源地址 `segSrcOffset*sizeof(T)%32==0`；`(srcRowWidth-copyCols)*sizeof(T)%32==0`；无 padding；各 segment 在输出中连续无 gap | 多 segment 连续 MTE2 到同一 UB 缓冲不同偏移 → 一次 Wait → 多 segment 连续 Adds → 一次 Wait → **一次 CopyOut 写整行** | 回退 `ProcessTensorSegment` 逐 tensor 处理 |
| 2 | **SEGMENT_BATCH**（segment 内多行批量，最终基准未启用） | 同一 segment 连续多行 `copyCols` 相同；无 padding；源/目的地址及行宽差 32B 对齐；`DataCopy` 的 `srcStride/dstStride <= 65535` 且 `blockCount <= 4095` | 用 `DataCopy + DataCopyParams(blockCount=numRows)` 一次搬多行；仅当 Kernel 以 `tileRowCount>1` 进入 `ProcessTensorSegment` 时触发 | 单行 `SEGMENT_SINGLE` |
| 3 | **SEGMENT_SINGLE**（最终正确性路径，最终基准实际使用的 fallback） | 不满足上述任一条件 | 单行处理：对齐走 `DataCopy`；非对齐/带 padding 走 `DataCopyPad + DataCopyExtParams`，显式补零，`Adds` 有效部分 | 无（最通用） |

### 4.2 FAST_ROW_E：整行多段合并 + 两阶段批量同步

**问题**：当一行输出由多个 tensor 的 segment 拼接而成时，如果每个 segment 都执行
“MTE2 搬入 → Wait → Adds → SetFlag/WaitFlag → 再搬下一个 segment”，每多一个 segment
就多两次同步。同步等待本身有固定开销，且会打断 MTE2 的连续下发，导致搬运单元空转。

**设计**：同一行所有 segment 的目标 UB 地址是互不重叠的（按 segment 宽度累加偏移）。
MTE2 是顺序执行的，Vector 同队列的 Adds 也是顺序执行的。因此可以：

1. 把所有 segment 的 MTE2 连续下发到同一个输入 UB 缓冲的不同偏移；
2. 全部下发后，只做一次 `WaitFlag<MTE2_V>`；
3. 把所有 Adds 连续执行到输出 UB 缓冲的不同偏移；
4. 全部执行后，只做一次 `WaitFlag<V_MTE3>`；
5. 最后把拼好的整行一次 CopyOut。

**为什么安全**：同一队列内 MTE2 保序、Vector 保序；一次 `WaitFlag` 等待的是该事件之前
所有同队列命令，所以最终一次 Wait 能覆盖之前所有 MTE2 写入；Adds 同理。
**为什么快**：同步次数从 `2 × numSegments` 降为 `2`；MTE2 命令可以连续排入硬件队列，
减少等待气泡；CopyOut 一次搬出整行，比逐段搬出减少 MTE3 次数。

**准入条件（必须同时满足）**：

- 每段 `copyCols > 0` 且 `copyCols * sizeof(T) % 32 == 0`：Adds 和 CopyOut 的地址偏移必须 32B 对齐；
- 每段的源地址 `segSrcOffset * sizeof(T) % 32 == 0`：`DataCopy` 的 GM 源地址必须 32B 对齐；
- 每段的源行宽差 `(srcRowWidth - copyCols) * sizeof(T) % 32 == 0`：多段合并时
  `DataCopy` 的源布局必须可被 32B 块表达；任一段不满足就走 `DataCopyPad`；
- 无 padding：任何 padding 段都必须先写零再搬有效数据，不能与纯合并路径共用；
- 各段输出列区间连续（无 gap）：如果有 gap，把段直接拼在 UB 里再 CopyOut 会写错位置。

**效果**：这是最终实现里常规多 tensor / 对齐 case 的主路径。实测 4-tensor 的宽行 case 提升约
+1.8%；3-tensor、2-tensor case 不退化。在最终基准中，这条路径 + 两阶段同步是最后保留的正向优化。

### 4.3 SEGMENT_BATCH：segment 内多行批量（可选扩展，最终基准未启用）

**问题**：对 1D 大向量或窄行 case，一个 segment 在多行上拥有完全相同的 `copyCols`。
若逐行做“MTE2→Adds→MTE3”，搬运次数与行数成正比，MTE2/MTE3 的启动开销被放大。

**设计**：当一个 segment 连续多行的有效宽度相同、且源/目的行间 stride 可被硬件表达时，
用一次多 block DMA 搬运多行。注意 CopyIn（GM→UB）和 CopyOut（UB→GM）的 stride 语义不同，
要分别计算：

```text
CopyIn（GM→UB，目的端 UB 连续）:
  blockCount = numRows,                        // 行数
  blockLen   = copyBytes / 32,                 // 每行的 32B 块数
  srcStride  = (srcRowWidth - copyCols)*elemSize / 32,  // GM 行间 gap，32B 块
  dstStride  = 0                                // UB 连续排布

CopyOut（UB→GM，源端 UB 有行槽）:
  blockCount = numRows,
  blockLen   = copyBytes / 32,
  srcStride  = (ubRowStrideElems - copyCols)*elemSize / 32, // UB 行间 gap，32B 块
  dstStride  = (dstRowWidth - copyCols)*elemSize / 32       // GM 行间 gap，32B 块
```

**为什么必须加准入条件**（以下约束用于 `DataCopy + DataCopyParams`，非 `DataCopyPad`）：

- `DataCopyParams` 的 `srcStride/dstStride` 字段是 uint16，本文多行 `DataCopy` 按 **32B 块** 填写，
  最大值 65535 块（约 2MB）；`blockCount` 还需 ≤ 4095。宽 2D 行间 gap
  （`srcRowWidth - copyCols`，如 4096 列 × 2B = 8KB 以上）按块换算后很容易超过 stride 上限，
  直接截断会搬错地址（硬件按错误 stride 跳行），所以 stride 超限必须回退单行。
- 源地址、目的地址、行宽差必须 32B 对齐，否则 DMA 无法按 32B 块表达。
- 一旦某行 `copyCols` 变化（例如进入最后不足 chunk 的行），批必须 flush，不能继续合并。

**在最终基准中的状态**：最终基准的 `ProcessTile` 逐行调用 `ProcessRowE`，fallback 也按
`tileRowCount=1` 进入 `ProcessTensorSegment`，因此 **SEGMENT_BATCH 在最终基准主循环中不会触发**。
它保留为扩展路径：若未来让 Kernel 以 `ubRowFactor>1` 的 tile 进入 `ProcessTensorSegment`，
且满足上述准入条件，才能激活多行批量。历史上该机制在开发期曾把 1D 大向量 case 从约 1.44x
提升到约 4.19x（中间数据，非最终基准；最终基准中该 case 因参考侧异常为 14.4x 离群值）。
宽 2D case 因 stride 超限无法受益，这是硬件/API 的硬边界，不要强行使用。

### 4.4 SEGMENT_SINGLE：最终正确性 fallback

**问题**：非 32B 对齐、或 segment 带有 padding 时，上面两条路径都不能直接工作：
非对齐时 `DataCopy` 无法表达精确字节数，带 padding 时必须把零写入 UB 对应区域。

**设计**：单行处理，优先保证正确：

1. **搬运**：对齐单行走 `DataCopy`；非对齐/带 padding 走 `DataCopyPad + DataCopyExtParams`。
   `DataCopyExtParams.blockLen` 是 uint32 **字节**，可精确表达当前所有行宽的字节数；
   `DataCopyPad` 若用 `DataCopyParams`，其 `blockLen` 是 uint16 **字节**（最大 65535 字节），
   宽行会截断，所以这里必须用 Ext。
   `DataCopyPadExtParams` 可以指定 `paddingValue = 0` 补齐右侧 32B 对齐。
2. **补零**：输出 segment 有 padding 时，先向输出 UB 的 padding 区域 Duplicate 零：
   - aligned 路径：只清 `[copyCols, fullSegCols)`；
   - 非 aligned 路径：必须整段清 `[0, numRows*fullSegCols)`，
     因为 `Duplicate` 的子视图要求 32B 对齐起始地址。
   然后 `Adds` 只做 `validElems` 个元素，让 padding 保持零。
3. **写回**：`CopyOut` 写完整 segment（含零 padding）。

**效果**：所有真实 case 20/20 精度通过。该路径是性能和正确性之间的安全网，不应被任何特化路径
“无 fallback 地抢占”。

### 4.5 bf16 分支：为什么用 `ReinterpretCast<int16_t> + Adds`

AscendC 的 `Adds` 不支持 bf16 类型模板。bf16 的位模式与 int16 相同（都是 16 位），
ChunkCat 不改变元素值，因此所有 bf16 vector 拷贝都可以先 `ReinterpretCast<int16_t>()`
再执行 `Adds(..., int16_t(0), count)`。这样 bf16 与 fp16/fp32 共用同一套 kernel 结构和同步逻辑，
避免为 bf16 单独维护一条直拷路径（历史上单独的 bf16 直拷路径有约 10% 偶发精度失败）。

---

## 5. Host Tiling 设计

### 5.1 二维 block 网格搜索

**问题**：如果只按 `outputRow = num_chunks` 行并行开核，当 `num_chunks` 很小（例如 5）时，
最多只能开 5 个核，其余核空闲。必须把输出列也切分，让核数 = `blockRowNum × blockColNum`。

**设计**：搜索 `(r, c)`，让 `r*c` 在 `r ≤ outputRow、c ≤ outputCol、r*c ≤ aivCoreNum` 下最大；
相同乘积时优先更大的 `c`（列分块更细，核间列粒度更小、负载更均衡）。

```text
maxRowBlocks   = allAlign ? min(coreNum, outputRow)
                          : min(min(coreNum, outputRow), 16)   // 非全对齐不要开太多行块
maxRowBlocks   = max(maxRowBlocks, 1)
bestR/bestC    = argmax(r*c)，tie 取 c 更大者；r=1..maxRowBlocks，c=min(outputCol, coreNum/r)
blockRowFactor       = ceil(outputRow / bestR)
tailBlockRowFactor   = outputRow - blockRowFactor * (bestR - 1)   // <=0 时取 blockRowFactor
blockColFactor       = ceil(outputCol / bestC)
tailBlockColFactor   = outputCol - blockColFactor * (bestC - 1)   // <=0 时取 blockColFactor
blockDim             = bestR * bestC
```

**为什么非全对齐时行块上限取 16**：全对齐时所有行结构相同，多开行块不会引入额外复杂度；
非全对齐时按行分块越多，进入非对齐/补零 corner case 的行越多，过早打开太多行块反而让更多核
处理慢路径，不如集中列向切分。具体上限值随平台和 case 集可调，16 是验证过的默认值。

**实现**：Host 算出 block 网格、tile 尺寸后，`blockDim = blockRowNum × blockColNum`。
Kernel 只根据 `blockIdx` 反解出自己的 `(blockRowGroup = blockIdx / blockColNum,
blockColGroup = blockIdx % blockColNum)`，再按上述 factor 计算本核行/列区间与尾块区间。

### 5.2 列优先 UB tile 切分

**问题**：一块 UB 要同时容纳输入 tile 和输出 tile。如果先按行切（例如固定“每 tile 2 行”），
在宽行场景每行只分到很窄的列宽，列方向的 tile 数大幅增加，MTE2/MTE3 次数随之增加。
列优先切分能最大化单次搬运的列连续长度。

**设计**（这就是 `colLimit → ubColLoop → ubColFactor → rowLimit` 公式）：

```text
colLimit       = inUbSize / itemSize            // 输入 UB 能容纳的最大列元素数；>=1
blockColCount  = max(blockColFactor, tailBlockColFactor)
blockRowCount  = max(blockRowFactor, tailBlockRowFactor)
ubColLoop      = ceil(blockColCount / colLimit) // 列方向需要几个 tile
ubColFactor    = align_up(ceil(blockColCount / ubColLoop), 32/itemSize)  // 对齐到 32 字节的整元素数
ubColFactor    = min(ubColFactor, colLimit)     // 必须 clip，否则 rowLimit 可能为 0
ubColFactor    = max(ubColFactor, 1)
rowLimit       = (ubColLoop == 1) ? colLimit / ubColFactor : 1
ubRowLoop      = ceil(blockRowCount / rowLimit)
ubRowFactor    = ceil(blockRowCount / ubRowLoop)
ubRowFactor    = min(ubRowFactor, blockRowCount) // 尾块不超行数
ubRowFactor    = max(ubRowFactor, 1)
```

**为什么 `ubColLoop == 1` 才允许多行**：如果整列已经能塞进一个 tile，才有多余 UB 空间
去容纳更多行；如果列本身就要拆成多个 tile，再按行拆只会让每 tile 的列宽更窄，导致更多搬运次数。
因此规则是 **“先保证列宽最大，列放不下就不叠行”**。

**效果**：列优先策略是常规 2D case 性能接近参考的关键；它让每 tile 的列宽从早期方案约提升
33%–100%，直接减少 tile 迭代次数和 MTE2 等待。

### 5.3 UB 输入/输出容量划分

**设计**：`ubBudget = GetCoreMemSize(CoreMemType::UB) - reserved`，其中 `reserved` 为安全余量（平台相关，910B3 经验值约 8 KB）；输入/输出**各占一半**（`inUbSize = outUbSize = ubBudget / 2`）。本算子输入输出 dtype 相同，因此两半缓冲区都按该 dtype 使用；
本实现采用“输入搬进 srcLocal、Adds 到 dstLocal、再 CopyOut”的三段式，输入 buffer 只需容纳
一块有效 tile。

> **禁止写死 UB 大小**：910B3 示例 `ubBudget ≈ 192 KiB`、`reserved ≈ 8 KB`；950PR（DAV_3510）可用 UB 示例 248 KiB（INI `ub_size=253952`）。
> `reserved` 与 1:1 比例在 950PR 上**未重测**，换平台时必须按实测回填；源码里出现 `ubSize - 8KB` / `192 * 1024` 一律视为未去本地化的反例（见 §7.3）。
>
> **950PR 的容量核算是另一套（并列，不替换本节 1:1）**：推荐实现不复用「输入/输出各半」，而是
> `S` 个 staging 槽被载入与写出复用、零区 `Z` 独立分配，约束为
> `S·(U + slack) + Z ≤ 可用 UB`（示例：`S=5`、`U=32 KiB`、`slack=64 B`、`Z=65600 B` ⇒ `229760 B ≈ 224 KiB`，约占 248 KiB 的 90%）。
> 该结构下 **`colLimit / ubColLoop / ubColFactor / rowLimit` 族公式不适用**；对应量是输出字节单元 `U` 与「单元内行数 = `ceil(U/row_bytes)`」，且 `row_bytes > U` 时单元落在单行内、多行描述符自动退化为 1D（A）。接近 UB 上限的组合（如单缓冲 192 KiB + 65600 B 零区）在 950PR 上会以设备错误 `507035` 失败，**不得作为默认配置**；实际可用 UB 边界**未系统测定**（C）。

**为什么各占一半，而不是输入 2/3 + 输出 1/3**：输出 tile 需要容纳整行所有 segment 的拼接结果，
输出 buffer 过小会直接挤压列宽 `ubColFactor`，在宽行 case 成为瓶颈。等分后输出列宽上升约 50%–100%，
是常规 2D case 达到目标性能的重要一步。

### 5.4 全宽 `colLimit` 与 `DataCopyExtParams`

**问题**：宽行时单次搬运字节数可能超过 65535。若在 `DataCopyPad` 场景使用 `DataCopyParams`，
其 `blockLen` 是 uint16 **字节**（最大 65535 字节，约 64KB），宽行不够用，必须改用
`DataCopyExtParams`（`blockLen` 为 uint32 字节，上限 2097151 字节）。
注意 `DataCopy + DataCopyParams` 是另一条路径（严格 32B 对齐，本文按 32B 块填写），
两条路径的单位语义不同，禁止混用。

**设计**：所有 `DataCopyPad` 路径使用 `DataCopyExtParams + DataCopyPadExtParams<T>`：

- `blockLen` 是 **uint32 字节数**，单行搬运不受 16 位溢出影响；
- `DataCopyPadExtParams` 同时携带右侧 padding 的元素数和 padding 值；
- alignment 良好的多行批量仍用 `DataCopy + DataCopyParams`（本文按 32B 块单位填写）。

**效果**：全宽 `colLimit` 可以安全使用，不需要人为 cap 列宽。历史上曾因 cap 列宽
（16384 或 16320/32640）而损失约 33% 列宽，移除 cap 后常规 case 性能接近参考；
前提是 Pad 路径已切到 ext 参数。

### 5.5 输出分配：`at::empty` + Kernel 显式写零

**问题**：

- `at::zeros` 会对大输出额外执行一次 ZerosLike kernel，这是纯开销，
  且 ChunkCat 本应写满所有输出位置。
- 直接改成 `at::empty` 而不补零，padded 区域会保留未初始化内存，精度失败。

**设计**：Host 用 `at::empty` 分配输出；Kernel 对每个输出位置负责：

- 有效数据由搬运路径写入；
- padding 区域由显式 Duplicate 零填充写入；
- 完全无效的行/段（例如 `num_chunks > dim0`）整段清零后 CopyOut。

**效果**：省掉 Host 侧的 ZerosLike kernel 开销，同时保持精度；这是大输出 case 提速
超过 10% 的来源。写零优化只清 `[copyCols, fullSegCols)` 的尾部 padding，比整段清零更省 Vector 指令。

---

## 6. 同步设计：为什么同步次数是隐藏成本

### 6.1 问题

每 tile 的典型执行是 `MTE2 → Vector → MTE3`。数据搬运本身很快，但每次 `SetFlag/WaitFlag`
都有固定开销。在“每 tile 只有 1 行、MTE3 耗时 <1us”的细粒度 tile 上，事件等待开销与搬运
本身在同一量级。如果对一行内的每个 segment 都做一次同步，开销随 segment 数线性放大。

### 6.2 设计

- **segment 内不设中间同步**：同一行列的多个 MTE2 写入 UB 不同偏移，硬件顺序执行，
  最后只等一次；多个 Adds 同理最后只等一次。
- **tile 级只保留必要的 MTE2→V→MTE3 串行同步**：`MTE2_V` 保证 MTE2 完成后 Vector 开始；
  `V_MTE3` 保证 Vector 完成后 MTE3 开始；CopyOut 后再用 `MTE3_MTE2/MTE3_V` 保证下一 tile 复用
  UB 缓冲区前 MTE3 已读完。

### 6.3 事件 ID

每个 `HardEvent` 类型独立分配事件 ID（MTE2_V、V_MTE3、MTE3_MTE2、MTE3_V）。
不要拿一个 ID 复用到多个事件类型；类型共享一个 ID 会造成事件状态机互相覆盖、偶发竞态。

---

## 7. 实现落地最小骨架

按本文件不足以直接生成可编译代码的地方，都应在工程脚手架/基础 skill 中补全。下面是本算子特有、
必须精确实现的 C++ 关键块。

### 7.1 TilingData 结构（Host/Kernel 共用的 ABI）

字段顺序必须 Host/Kernel/二进制完全一致；Host 端 `memcpy` 成字节 tensor 后传到 Device。

```cpp
struct ChunkCatTilingData {
    int32_t dtypeFlag;         // 0=fp32, 1=fp16, 2=bf16
    int32_t numInputs;         // 1..4
    int32_t numChunk;
    int32_t outputRow;         // = numChunks
    int32_t outputCol;
    int32_t dim0[4];           // 每个输入的 dim0
    int32_t width[4];          // ceil(dim0/numChunk) * prod(shape[1:])
    int32_t blockRowNum;
    int32_t blockColNum;
    int32_t blockRowFactor;
    int32_t tailBlockRowFactor;
    int32_t blockColFactor;
    int32_t tailBlockColFactor;
    int32_t inUbSize;          // bytes
    int32_t outUbSize;         // bytes
    int32_t ubRowFactor;
    int32_t ubColFactor;
    int32_t isAllAlign;
    int32_t isHalfAlign;
    int32_t isOneConcat;
};
```

### 7.2 Host 端最小流程

下面为贴近实际代码的伪代码/骨架（非逐字可编译，需按工程模板补全头文件与 launch 宏）。

```text
// 1) 输入合法性
TORCH_CHECK(dim == 0); TORCH_CHECK(num_chunks > 0); TORCH_CHECK(!x.empty());
TORCH_CHECK(x.size() <= 4);
// dtype 只允许 fp32/fp16/bf16；所有 tensor 同 dtype、同 device、contiguous。

// 2) 计算输出列宽
outputCol = sum_i( ceil(x_i.size(0)/num_chunks) * prod(x_i.shape[1:]) );

// 3) 输出只分配，不显式清零
at::Tensor y = at::empty({num_chunks, outputCol}, x[0].options());

// 4) 获取平台能力
auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
int32_t coreNum = platform->GetCoreNumAiv();
uint64_t ubSize = 192 * 1024;   // ⚠️ 反例：不要写死。此行是 910B3 取值的示范（见 §5.3），950PR 上 248 KiB 且 1:1 结构不适用
platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);   // 运行时真源；必须由它覆盖上面的初值

// 5) computeTiling（§5 公式）并传 Device
ChunkCatTilingData tiling;
computeTiling(&tiling, x, num_chunks, outputCol, coreNum, ubSize);
at::Tensor tilingCpu = at::empty({static_cast<int64_t>(sizeof(ChunkCatTilingData))},
                                  at::device(at::kCPU).dtype(at::kByte));
memcpy(tilingCpu.data_ptr(), &tiling, sizeof(ChunkCatTilingData));
auto tilingNpu = tilingCpu.to(at::kPrivateUse1);

// 6) 输入补齐到 4 个并启动
std::vector<at::Tensor> padded = x;
while (padded.size() < 4) padded.push_back(at::empty({0}, x[0].options()));
uint32_t blockDim = tiling.blockRowNum * tiling.blockColNum;
EXEC_KERNEL_CMD(chunk_cat, blockDim, padded[0], padded[1], padded[2], padded[3], y, tilingNpu);
```

`EXEC_KERNEL_CMD` / `kernels/aclrtlaunch_chunk_cat.h` 由工程模板或构建脚本按算子名生成；
`platform_ascendc`、`torch_kernel_helper.h` 均来自标准工程骨架。

### 7.3 Kernel 端最小流程

```text
Init:
  blockRowGroup_ = blockIdx / blockColNum
  blockColGroup_ = blockIdx % blockColNum
  计算本核 blockStartRow/blockRowCount/blockStartCol/blockColCount（尾块用 tail*Factor）
  InitBuffer(srcBuf_, inUbSize)  // TBuf<TPosition::VECCALC>
  InitBuffer(dstBuf_, outUbSize)
  srcLocal_ = srcBuf_.Get<T>(); dstLocal_ = dstBuf_.Get<T>();
  四个事件 ID = AllocEventID<MTE2_V / V_MTE3 / MTE3_MTE2 / MTE3_V>

Process:
  rowLoop = ceil(blockRowCount / ubRowFactor)
  colLoop = ceil(blockColCount / ubColFactor)
  for i in rowLoop*colLoop:
     rowTile, colTile; tileRowCount, tileColCount（尾块截断）
     ProcessTile(tile): 逐行 ProcessRowE(row, tileStartCol, tileColCount)

ProcessRowE:
  收集本行各 tensor 的 segStart/segEnd/segWidth/copyCols
  若 不满足 FAST_ROW_E 准入 → 逐 tensor ProcessTensorSegment 后 return
  Phase1: 循环 DataCopy/DataCopyPad 到 srcLocal_[srcOffset]（无 SetFlag/WaitFlag）
          SetFlag<MTE2_V>; WaitFlag<MTE2_V>
  Phase2: 循环 Adds 到 dstLocal_[ubOffset]（无 SetFlag/WaitFlag）
          SetFlag<V_MTE3>; WaitFlag<V_MTE3>
  根据 firstSegStart 做一次 DataCopy/DataCopyPad CopyOut
  最后 SetFlag/WaitFlag<MTE3_MTE2>、<MTE3_V> 再返回
```

### 7.4 本设计涉及的 API 字段语义与限制

| API / 字段 | 平台 | 语义 / 单位 | 已确认限制 | 本文用法 |
|---|---|---|---|---|
| `DataCopy` + `DataCopyParams`（GM↔UB、UB↔UB） | 910B3（原行） | `blockLen/srcStride/dstStride` 均为 **32B 块数**，`uint16` | `blockCount` 1..4095；`blockLen` 1..65535 块 | 对齐多行/单行批量搬运 |
| `DataCopy` + `DataCopyParams` | **950PR** | 结构体与字段宽度**未变**（`uint16`，仍按 32B 块）；该重载在 3510 上**不含** `PaddingMode` 模板参数 | 推荐实现未使用 ⇒ 容量/对齐结论**未验证，禁止沿用 910B3 数值** | 不使用（全程 `DataCopyPad`） |
| `DataCopyPad` + `DataCopyParams`（GM↔UB） | 910B3（原行） | `blockLen` 为 **字节**（字段 `uint16`）；stride **按操作数位置**：GM 侧字节、UB 侧 32B 块 | `blockCount` 1..4095；`blockLen` 最大 65535 字节，宽行不够用 | 仅窄行/非对齐小数据量兜底 |
| `DataCopyPad` + `DataCopyParams` | **950PR** | 重载存在且**新增 `PaddingMode` 模板参数**（默认 `Normal`）；`blockLen` 仍为字节（`uint16`） | 该形态 `blockCount ∈ [0,4095]`（与 Ext 形态的 `[0,65535]` 不同） | 不使用 |
| `DataCopyPad` + `DataCopyExtParams`（GM↔UB） | 910B3（原行） | `blockLen` 为 **字节**（`uint32`）；stride **按操作数位置**：GM 侧字节、UB 侧 32B 块 | `blockCount` 1..4095；`blockLen` 上限 2097151 字节 | 所有宽行/精确字节 `DataCopyPad` 路径（本文默认） |
| `DataCopyPad` + `DataCopyExtParams` | **950PR** | `blockLen` 字节；**`srcStride/dstStride` 为 `int64_t`**；GM→UB：`blockLen`/`srcStride` 单位 1 B、`dstStride` 单位 32 B；UB→GM：`srcStride` 单位 32 B、`dstStride` 单位 1 B | 950 文档：`blockCount ∈ [0,65535]`、`blockLen ∈ [0,2²¹−1]`；GM→UB `srcStride ∈ [−blockLen, 2⁴⁰−1]`（负值 = 重复搬同一块/广播）、`dstStride ∈ [0,65535]`；UB→GM 反向；`blockCount`/`blockLen` 任一为 0 ⇒ 接口视为 NOP | 主搬运：单行稠密载入 + `blockCount=行数` 的 strided 写出；零区广播用 `srcStride=0` |
| `DataCopyExtParams` 多行 | 910B3（原行） | — | 910B3 软件回退极慢/hang，不要用 | 仅 `blockCount=1` |
| `DataCopyExtParams` 多行 | **950PR** | 头文件/文档可编译可表达 | **性能未验证**：固定 38 例上该 strided 多行分支执行 0 次（`row_bytes ≤ 32 KiB 且多 tensor` 的用例数 = 0）；「~29–32 行隐式上限、超出静默丢零」在本机已被真机证伪（2785 配置 0 失败） | 代码保留、固定 38 例未触发 |
| `DataCopyPad<T, PaddingMode::Compact>` | **950PR 专属** | 模板参数；UB 侧**稠密**（占用 = 有用字节）、只在整块末尾补一次 32 B；`Normal` 则每块补到 `ceil32(blockLen)` | **编译期门禁**：该模板参数只对 arch35 系存在 ⇒ **910B3（2201）上无法编译**；GM→UB 方向下 `Compact` 使 `dstStride`/`leftPadding`/`rightPadding` 无效，UB→GM 方向文档未写该规则 | 2D 写出统一 `Compact` |
| `DataCopy` / `DataCopyPad` 的 UB 端地址 | 通用 | UB 起始地址必须 32B 对齐 | 非对齐报 VEC/MTE 地址错误 | 用 `AlignedElems` 维护 UB 偏移 |
| `DataCopy(dst,src,count)` | **950PR** | 元素数；`count*sizeof(T)` 非 32B 倍数时**静默向下截断** | ⇒ ChunkCat 输出总字节不保证 32B 倍数，载入/写出必须全用 `DataCopyPad` | 不使用裸 `DataCopy(count)` |
| `Adds(dst,src,scalar,count)` Level2 | 910B3（原行） | 元素数 | 在 CANN 9.1.0 已内部处理 repeat；bf16 不支持，需 `ReinterpretCast<int16_t>` 做位拷贝 | 主拷贝路径 |
| `Adds` 的 bf16 限制 | **950PR** | **不适用**：推荐实现是纯字节 DMA，**完全不使用 `Adds`** | 3510 的 `Adds` 是否支持 bf16 **未验证** ⇒ 禁止把 910B3 的结论（正反任一种）搬过来 | 不使用 |
| `Duplicate` 子视图 | 通用 | 元素数 | 起始地址必须 32B 对齐 | 只清对齐 padding 区；非对齐清整段（950PR 用于一次性生成 UB 零区） |
| `GlobalTensor::SetL2CacheHint<CacheRwMode>` | **950PR** | 模式编码进 GM 地址 bit[60:63]；在 3510 上确实生效（非空实现） | **实测为负收益，不采用**：`DISABLE` 使流式例再慢约 2.5%、L2 驻留例掉到 0.33×；`PERSISTENT` 亦有害 | 不使用 |
| `SetLoopModePara` / `ResetLoopModePara` | **950PR 专属** | 两层循环外提，可再压指令数 | 需 `blockLen` 32B 对齐或额外的 `Reset` 纪律；必须成对 | 未采用（与 2D 描述符收益重叠） |

---

## 8. 反模式：不要在这些方向上浪费时间

| 方向 | 为什么失败 | 可复用判断 |
|---|---|---|
| 双缓冲 / TQue depth-2 / ping-pong 流水线 | UB 总量固定，双缓冲必须把 tile 减半 → tile 数翻倍，启动开销抵消重叠收益；且“1 行/tile”时 MTE3 时间极短，事件同步本身占比高。长流水线还出现数据错误 | 只有当 tile 可含多行（如 8+ 行）使 MTE3 时间远超事件开销时，才值得重新评估 |
| 直接 MTE2→MTE3（去掉 Adds / direct-dst） | ChunkCat 是搬运受限，Vector 不是瓶颈；省掉 Adds 只省少量 Vector，且 direct 路径对齐条件更严、同步收益微小 | 先做 profiling 确认 Vector 是否真的是瓶颈，否则保留 Adds |
| 多行 `DataCopyExtParams` | 在 910B3 上多行 ext 参数走软件回退，极慢甚至 hang；它只适合单行精确字节长度 | 多行批量一律用 `DataCopy + DataCopyParams`；ext 只用于单行/精确字节 |
| 把 `DataCopyParams.blockLen` 一律当 32B 块数 | `DataCopy + DataCopyParams` 按 32B 块填写；但 `DataCopyPad + DataCopyParams` 的 `blockLen` **就是字节**（uint16，最大 65535 字节）。两条路径语义不同，混用会截断或搬错长度 | 精确字节走 `DataCopyPad + DataCopyExtParams`；对齐多块走 `DataCopy + DataCopyParams` |
| 用 `at::empty` 但不显式补零 | 输出 padding 区域未初始化，精度失败 | `at::empty` 必须与 Kernel 显式补零成对出现 |
| 给所有 segment 逐段 WaitFlag | 同步开销线性放大 | 同队列多写入统一 Wait |
| 多事件类型共用一个事件 ID / 用 `FetchEventID` | 事件状态机互相覆盖 | 每类型独立 `AllocEventID` |
| 某类 case 慢就全局截断列宽 | 后续证明 short cap 会造成列宽损失，是性能瓶颈 | colLimit 用全宽 `inUbSize/itemSize`；Adds Level2 API 已内部处理 repeat 限制 |
| 大 tile（96/192 KiB）提升持续吞吐 **（950PR 否定）** | 950PR 的 HBM 混合速率对单元尺寸 32/64/96/192 KiB 与槽数 1–4 不敏感；96 KiB 小例更差；192 KiB 单缓冲直接 UB 上限失败（`507035`） | 先做"单元尺寸扫描"再决定；`S·(U+slack) + Z ≤ 可用 UB` 必须编译期可算 |
| L2 流式提示（`SetL2CacheHint`）提升 **（950PR 否定）** | `DISABLE` 使流式例再慢约 2.5%、L2 驻留例掉到 0.33×；`PERSISTENT` 亦有害（+13%~+19%） | 一次性只读数据不要关 L2；先在同一 session 内做开/关配对再决定 |
| 把 910B3 的"双缓冲是反模式"搬到 950PR（反之亦然） | 950PR 推荐实现故意用 5 槽深流水，且实测收益来自尾波量化（单元尺寸）、**流水深度效应未被单独隔离** | 该结论必须带平台；跨平台引用即错 |
| 对齐时改用非 Pad 的裸 `DataCopy(count)` **（950PR）** | `count*sizeof(T)` 非 32B 倍数时**静默向下截断**；38 例上每 tensor 描述符条数已是 1，收益上限 <0.1% | 若要用，必须先证明字节数是 32B 倍数；否则保持 `DataCopyPad` |
| 把 `DataCopyPad<Compact>` / `SetLoopModePara` 代码写进 910B3 目标 | `PaddingMode` 模板参数只对 arch35 系存在，910B3 上**无法编译** | 用平台谓词/条件编译隔离；910B3 走 `Normal` |
| 在 950PR 上继续用编译期常量 UB（不使用 `GetCoreMemSize`） | 推荐实现本身这么做过（有显式 248 KiB 核算），但这是**反例**，不能作为规范 | 规范要求由 `GetCoreMemSize(CoreMemType::UB, …)` 驱动；950PR 的 248 KiB 只能作示例值 |
| 需要 UB 内向量处理时仍用 `TQueBind` 当纯输入 | AIV bufId 同步错位 → 多 tile 时**确定性错行**（每次运行错行集合不同） | 整块 GM→UB→GM 往返才用 `TQueBind`；UB 内有向量读时改用 canonical `TQue<VECIN,D>`(MTE2→V) + `TQue<VECOUT,D>`(V→MTE3) |
| `at::empty` 输出 + host 未校验输入维长一致性 | 条目偏多时静默留下**未初始化输出行**（既不报错也不是零），`38/38` 通过也可能覆盖不到 | host 加维长一致性 `TORCH_CHECK`；用「每个输出字节都被写过」的毒值覆盖实验自证（注意 2 字节 dtype 的毒值假阳性坑） |

---

## 9. 设计检查表

开发同类 ChunkCat 算子时，按以下顺序落地：

1. **语义归一化**：输出写成 `(chunks, outputCol)`；计算 per-tensor 列宽。
2. **路由架构**：先定 3 条 route 及优先级（FAST_ROW_E → SEGMENT_BATCH → SEGMENT_SINGLE），
   写清每条 route 的准入谓词和 fallback；不要写“万能 kernel”。
3. **Host tiling**：
   - 2D block 网格 `r*c` 尽量铺满核，相同乘积优先更大的 `c`；非全对齐限制最大行块数；
   - UB 预留后 1:1 切分；
   - 列优先 `colLimit / ubColLoop / ubColFactor / rowLimit`，并对 `ubColFactor`/`ubRowFactor` 做 clip；
   - 输出用 `at::empty`。
4. **Kernel 搬运**：
   - FAST_ROW_E：全对齐（含源地址/源行宽差）、无 padding、连续 → 两阶段同步 + 一次 CopyOut；
   - SEGMENT_BATCH：`DataCopy + DataCopyParams` 且 `blockCount>1`，stride ≤ 65535、`blockCount ≤ 4095`；`copyCols` 变化立即 flush 旧批；
   - SEGMENT_SINGLE：`DataCopyPad + DataCopyExtParams`，padding 显式清零。
5. **工程落地**：按 §7 实现 TilingData 结构、Host 最小流程（合法检查、tiling 传输、补 4 输入启动）
   和 Kernel Init/Process 骨架。
6. **安全性**：
   - `DataCopy + DataCopyParams`：字段按 32B 块理解；`DataCopyPad + DataCopyParams`：`blockLen` 为字节、stride 按 GM 侧字节 / UB 侧 32B 块区分；
   - 精确字节用 `DataCopyPad + DataCopyExtParams`；
   - `blockLen` 字段能用 16 位表达才用它；
   - Adds 地址偏移必须 32B 对齐；bf16 走 `ReinterpretCast<int16_t> + Adds`。
7. **同步**：segment 循环内不做 SetFlag/WaitFlag；每个 HardEvent 独立 ID。
8. **验证**：用下表结构覆盖 20 组 precision case；用同一设备和 warmup 后的 msprof quick 计时；
   同时报告含/不含 1D 异常大向量 case 的 geomean。
9. **平台分派**：950PR 走「字节单元 + A/B/C 三段分解」，910B3 走 `FAST_ROW_E → SEGMENT_BATCH → SEGMENT_SINGLE`；两条路线由显式平台谓词分开，原平台路径保持可达，不得静默改道。
10. **950PR 容量核算**：`S·(U + slack) + Z ≤ 可用 UB`（示例 `5×(32768+64)+65600 = 229760 B`）；不得照搬 1:1 切分；单缓冲接近 UB 上限（如 192 KiB + 65600 B）会以 507035 失败。
11. **平台 API 门禁**：`PaddingMode::Compact` 只对 arch35 系可编译；`DataCopyExtParams.blockCount` 上限按平台写（950PR `[0,65535]` / 910B3 `[0,4095]`）；裸 `DataCopy(count)` 在 `count*sizeof(T)` 非 32B 倍数时静默截断。
12. **尾波量化收益谓词**：纯几何量 `N = ceil(out_bytes/U) < 896`；不得按 `case_id`/`nc`/形状取值特判，且不得把收益表述为「存储通路吞吐提升」。

### 9.1 建议的验证 case 集（ChunkCat 20 组）

| # | 输入 shape | dtype | num_chunks | 覆盖点 |
|---|---:|---|---|---|
| 0 | [17] | fp32 | 1 | 1D 退化，chunk 不切 |
| 1 | [3] | fp32 | 8 | 全 padding（nc > dim0） |
| 2 | [3,2,2,2,2,2,2,2] ×2 | fp32 | 4 | 8D 最大秩 |
| 3 | [8192,2048] ×2 | fp32 | 5 | 质数 nc，非整除补零 |
| 4 | [16384,2048] ×2 | fp16 | 16 | 2 的幂 nc，整除 |
| 5 | [8192,64,64] ×2 | bf16 | 8 | bf16 高维 |
| 6 | [33554432], [8388608] | fp16 | 5 | 1D 大向量（性能异常 case，需剔除报告） |
| 7 | [16384,4096] ×2 | fp32 | 8 | fp32 宽 2D |
| 8 | [32768,4096] ×2 | fp16 | 32 | fp16 宽 2D |
| 9 | [32768,4096] ×2 | bf16 | 64 | bf16 宽 2D |
| 10 | [24576,4096] ×2 | fp16 | 64 | 大 nc + 宽 2D |
| 11 | [16384,4096] ×4 | fp16 | 64 | 4 输入多 segment |
| 12 | [32768,16,16,16] ×2 | fp16 | 16 | 4D 高维 |
| 13 | [32768,4096], [33554432] | fp16 | 32 | 1D/2D 混合 |
| 14 | [49152,4096] ×2 | fp16 | 32 | 超大 2D |
| 15 | [65536,4096] ×2 | bf16 | 128 | bf16 + 极端 nc |
| 16 | [32768,4096] ×3 | fp16 | 64 | 3 输入多 segment |
| 17 | [65536,4096] ×2 | fp16 | 64 | 大输出 |
| 18 | [24576,4096] ×2 | fp32 | 8 | fp32 大 nc |
| 19 | [49152,4096] ×2 | fp16 | 64 | 大 nc + 宽 2D |

---

## 10. 证据边界

- 证据平台：Ascend 910B3，UB 约 192KB（有效预算使用方式见上文），CANN 9.1.0，torch_npu 2.9.0。
- 最终基准性能：全 case geomean **1.1323x**；剔除 1D 大向量异常 case 后 **0.9657x**
  （该 case 因参考侧异常得到 14.4x 离群加速，会拉高 geomean，所以必须双口径报告）；
  中位 speedup **0.964x**，常规 2D case 约 **0.86x–1.11x**。
- 本设计结论是基于“ChunkCat 是搬运受限 + 规则输出矩阵”这一特性。对计算受限、
  不规则 stride、或需要跨核通信的算子，只能复用“路由优先 + 显式 fallback + 同步批量化”的方法，
  不要照搬 tile 数值、UB 切分比例或 route 内容。
- 上述 API 限制（`DataCopyParams` uint16 stride、ext 多行在 910B3 上性能差、Adds 无 bf16 模板、
  事件近似开销等）在其它 SoC 或 CANN 版本上需重新验证。

### 10.1 950PR（DAV_3510）追加证据边界

- **证据平台**：Ascend950PR（DAV_3510 / arch35），CANN 9.1.0，torch_npu 2.7.1.post10，UB 248 KiB
  （INI `ub_size=253952`；运行时可用值口径另计，预算取运行时值）。采集口径与上文 910B3 一致
  （同 device、同 session、paired 测量），逐字节 `precision_consistency` 门禁先行于性能采集。
- **结论**：38 例全量 geomean **1.0400**，中位 1.0041，min 0.9985，max 1.6560，`total_speedup` 1.0044；
  **双口径**为「全 38 例 1.0400 / 剔除 3 条最小与 L2 驻留几何后其余 35 例 1.0074」。950PR 上**不存在**
  910B3 文档里那种「1D 大向量离群例」（同类几何比值 1.0179），因此 910B3 的第二口径**不能照搬**。
- 与 910B3 的 1.1323 / 0.9657 **不可互相换算或外推**：分母（CANN `aclnnChunkCat` vs 910B3 参考）、
  用例集与实现结构均不同；同 lab 不同用例集/不同参考的 geomean 也无换算关系。
- **950PR 未验证清单**：`DataCopyExtParams` 多行推理/批量补零路径在固定 38 例上执行 0 次（正确性由同域随机几何
  与毒值抽查覆盖，不构成性能证据）；`Compact` 在 UB→GM 方向的 `dstStride` 生效性；事件 id 上限；
  DAV_3510 实际可用 UB 边界；bf16 与 `Adds` 的支持性；NDDMA / `SetLoopModePara` / RegBase / SIMT
  在该算子上均无真机正面证据（仅接口/文档级）。
- **共同口径限制**：`warmup=3/repeats=1`、无 device 前后置校验；官方固定「先参考后候选」给后测侧
  0.12–0.24% 的系统优势，任何 <0.3% 的候选间差**不可判**。