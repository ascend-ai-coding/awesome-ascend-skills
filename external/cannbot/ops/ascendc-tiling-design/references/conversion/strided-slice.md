# StridedSlice / Slice Ascend C Tiling 设计经验

> 所属类别：Conversion 数据转换类；场景判定与同类子场景索引见 [patterns.md](patterns.md)。

本文沉淀自一个完整的 StridedSliceV2 Ascend C Tiling 优化过程，并经过多轮正/负反馈核验。
只保留对后续同类算子设计有决策价值的内容：**先讲为什么，再讲怎么做**。

## 0. 性能优先原则

本文的目标不是帮助 Agent 先生成一个“能过正确性”的搬运算子，再把性能留给后续迭代；目标是指导 Agent **一次独立生成就优先落地高性能路由**。性能必须参与设计、测量和结论判断。

必须遵守以下原则：

- **正确性是准入条件，性能是核心交付维度**：能够输出正确结果不代表实现完成；为了方便通过精度而采用逐元素搬运、逐 run DMA、单 block DMA 或统一保守 fallback，不能在未评估性能影响的情况下作为最终方案。
- **每条核心经验都必须参与设计**：拿到 shape/stride/dtype 后，必须逐项检查并落实本文的路由、ownership group（写所有权分组，见 §3 表）、批量 DMA、Gather/窗口、tile、开核和同步经验。不能只读取经验文件后自行选择一个更容易实现的通用 descriptor 路径。
- **路由必须真实执行**：Host 选择的 `routeIndex` 必须在 Kernel 中实际 `switch`/分发到对应 route；只把 route ID 写入 TilingData、但所有 route 仍执行同一个 generic kernel，不算使用了分路由经验。
- **fallback 只能覆盖未被特化的边界**：`PACKED`/单 block fallback 必须是最后的安全网，不能吞掉等长短行、多 prefix span、跨 wrap、宏窗口或非对齐长 run 等本文已有专用方案覆盖的常见几何。
- **`blockCount=1` 不是非对齐的默认答案**：只有在专用路由的准入条件确实不满足、或已有明确硬件证据证明批量方式不安全时，才允许使用；使用时必须记录命中几何、硬件证据和预计性能影响。
- **性能明显退化必须反查经验覆盖**：一次独立生成出现明显性能退化、常见 case 大量落入 fallback 或核心路径未命中时，优先判断是核心经验未实现、路由被 fallback 吞掉、API/硬件约束表述不够可执行，不能直接解释为“正确但性能就这样”。

因此，本文后续所有“性能效果”都应理解为“在正确性通过后必须争取复现的设计信号”，而不是可以跳过实现的建议。

---

## 1. 适用与不适用

### 1.1 适用

- 算子语义：Slice / StridedSlice，输出是输入元素的有序子集，不改变元素值，输出为连续新张量。
- 输入约束：rank ≤ 8，正向 stride，元素大小 2 / 4 / 8 字节，输出非空。
- 平台假设：
  - **基准证据平台**：Ascend 910B3（DAV_2201 类），UB 192 KiB（196608 B）。
  - **追加证据平台**：Ascend 950PR（DAV_3510 / arch35），UB 248 KiB（平台 INI `ub_size=253952`；运行时以 `GetCoreMemSize(CoreMemType::UB, ...)` 为准），AIC:AIV = 28:56，CANN 9.1.0，torch_npu 2.7.1.post10。两平台 CANN 同版本，但 arch35 的代码路径与可用机制不同，见 §5.12。
- **与平台无关的原则**：写回一律走 MTE3，以物理 32B 块为最小写所有权；tile 容量、开核数与 UB 切分一律由 `GetCoreMemSize(CoreMemType::UB)` / `GetCoreNum*()` 驱动，不写死字节数或核数（写死值只能作为标注了平台的示例）。

### 1.2 不适用 / 不能直接照搬

- 负 stride、rank > 8、元素大小 1 字节的 Slice。
- Transpose / Pad / Gather / Scatter：几何不同，只能借鉴“先分路由”的方法，不能复制分支。
- 固定 tile 数字（48K/80K/96K/160K 等）不可无条件外推到其他 SoC 或 UB 容量。950PR（DAV_3510）上这些数值同样**未重测**，不得从 910B3 外推。
- §5.12 列出的 dav-3510 机制（`DataCopyPad<Compact>` / `SetLoopModePara` / RegBase Gather / SIMT）在 950PR 上**不能默认启用**：它们的可用性、单位与边界已实测，但收益/泛化范围受限（多为单条正例或探针级证据）；其中 SIMT 与 AscendC 框架在**同一编译单元内**互斥（独立 TU + 设备链接可用，见 §5.11.5）。
- 跨 session 的微秒级差异、小于 3% 的 geomean 变化，不能作为确定性收益。
- “更小的输出就该少开核”“Tiny 输入走 Fast Path 一定快”等直觉已被实测反证，不能默认成立。

---

## 2. 为什么必须是“分路由”架构

### 2.1 问题：不同几何的最优执行方式根本不同

StridedSlice 的输入视图可能表现为：

- 一整块连续内存（等价于一次大块拷贝）；
- 大量等长、恒定源 pitch 的短行（需要多 block 批量 DMA）；
- 多个前缀维、行会跨 pitch 边界（需要 span 批量或 Gather 复合）；
- 大且 32B 对齐的单 run（多核列分片才有收益）；
- 周期性的窗口抽取（一次 2D MTE2 + GatherMask 最优）；
- 完全不规则的碎片（只能标量 packed fallback）。

同一个 kernel 硬套所有几何，在实测中真实发生过两类灾难：

1. **逐 run 搬运等长短行**：一个 `[654,819] int16` 类几何，逐 run 走地址/队列/同步时慢到数百 μs，改为批量 DMA 后降到个位数 μs。
2. **把多 prefix 批量 DMA 泛化到 rank-one 快路径**：目标 case 改善，但 rank-one 快路径被吞掉后慢了约 70 倍，全量 geomean 大幅回退。

结论：**分路由不是优化项，是这类算子的基础架构**。先按几何特征选择执行方式，再在每条路径内部谈 tile、UB、同步、多核。

### 2.2 设计：语义归一化 → 特征派生 → Host 路由 → Kernel 只执行

```text
输入 shape/stride/dtype
  → 归一化为“最大连续后缀 run”的几何特征
  → Host 按优先级选择 route，并把 route ID 写入 TilingData
  → Kernel 必须 switch(routeID)，执行被选中的那一条路径
  → 每条路径内部再决定 tile、buffer、同步、开核
```

### 2.3 为什么路由放在 Host

- 路由所需信息全部来自 shape/stride/dtype/平台参数，与运行值无关，天然适合 Host 一次性计算。
- 如果放在 device，每个 core 都要重复做一遍 tiling-only 判断；改为 Host 传 `routeIndex` 后，代码路径显著简化。
- Host 决策结果可写入 TilingData，测试时可以直接 dump `routeIndex`，性能 case 才能精确归因到“哪条 route 变快/变慢”。
- Kernel 只执行一条路径，buffer 分配和同步方式可以按 route 独立选择，不用互相迁就。

### 2.4 为什么不用一个“大 kernel + 内部 if-else”

- 不同 route 需要的 buffer 类型不同：有的用普通 TQue，有的用 TQueBind 双槽，有的只要一个 TBuf，有的还要 Gather 索引表。全塞进一个 kernel 会让 Init 和同步相互妥协。
- 独立 kernel symbol 让每个 route 的 buffer/loop/sync 最简单，也避免了“改一条 route 影响另一条 route 的编译布局”这类难归因的退化，避免了编译文件过大导致的ICACHE MISS。
- 最终版本为每条 route 保留独立 kernel 入口，同时用同一个 `routeIndex` 分发。

### 2.5 效果

在一个 18 case、覆盖宏窗口 Gather / direct Padded DMA、多前缀 span、对齐 run 等几何的性能集上，该架构最终相对 NPU 参考实现达到约 **1.83× 几何平均加速**。

> **口径说明**：本文各路由小节给出的“效果”数字，来自该路由生效时的实测 session，**不是同一次 session、也不是同一套 case 集**。
> 参考侧耗时本身会随 session 波动（实测中同一参考实现、同一批 18 个用例在两个 session 之间的单例耗时差异可达 −25% ~ +10%），
> 小样例集与大样例集的绝对耗时更是相差几个数量级，因此：
> - 只有“同一次 session 内的相对参考加速比”和“同一 case、同一规模集上的自耗时改善”可以直接引用；
> - 不同小节之间的微秒数、以及跨 session 的 geomean 变化，不能直接相减或横向比较。

---

## 3. 路由特征：先知道“问什么”

| 特征 | 定义 | 为什么需要它 |
|---|---|---|
| `prefixRank` | 末尾最大物理连续后缀折叠后，剩余的前缀维数量 | 直接决定走哪一类搬运：0=整块连续；1=等长行；>1=多前缀 span/composite |
| `runBytes` | 单个连续后缀 run 的输出字节数 | 路由门槛、批量大小、UB 行槽都以它为单位 |
| `runCount` | run 总数 | 决定 ownership group 数、开核宽度、循环次数 |
| `ownershipGroupRuns` | `32 / gcd(runBytes, 32)` 个 run 组成一个最小写回组 | 非 32B 短 run 必须拼满 32B 后由单一 owner 写回，避免相邻写互相覆盖。含义：**从任一 32B 边界起，连续取这么多个 run 的字节数恰好是 32B 的整数倍**，所以核边界只能落在这个粒度的整数倍上 |
| `tileBytes` | 每个 core 一轮搬运的 UB 容量 | 先用于路由判定，再动态回填为吞吐最优值 |
| `windowSpanBytes` / `windowBatchRows` | 窗口类路由的源跨度与批行数 | 窗口路由专用，控制读放大和索引表大小 |

**为什么先折叠“最大连续后缀”**：输出是连续新张量，源视图总能切成若干连续 run；run 是搬运算子的最小自然单位。用 run 而非元素做路由，维度越高收益越大。

---

## 4. 路由决策树：顺序就是优先级

> 下面的顺序是经过正/负反馈验证后的最终优先级：**先匹配者胜**。每条路由都必须有明确准入条件；不满足就落到更通用的路径。

```text
Q0  能否直接表达为“有界完整后缀窗口”？
     ├─ 非对齐连续长 run（64B..4096B，跨 prefix 也允许） → 直接 Padded DMA（不建 Gather 表）
     ├─ 短 run、读放大有界、pitch 合法                     → 宏窗口 + 批量 Gather
     └─ 否 → 进入标准路由
Q1  输出整体是否物理连续？                                  → 整块连续搬运
Q2  是否是大而 32B 对齐的单 run，且单组不够铺满核？           → 列分片
Q3  是否是单前缀、等长、恒定 pitch 的行？                    → 仿射行批量 DMA
Q4  是否是精确的周期窗口几何？                              → 周期窗口专用
Q5  是否是多前缀、且 inner extent 可被 ownership group 整除？ → 前缀 span 批量
Q6  是否是非 32B 对齐、组跨 wrap 但组字节 ≤ 8K？             → 复合组 Gather
Q7  是否是 2D、固定内步长、4B 元素的仿射行？                 → 仿射折叠 + Gather
Q8  剩余 run 是否 32B 对齐？                               → 普通对齐 run
Q9  其余一切                                              → packed fallback
```

> 两点实现说明：
> - 下面的 `ogRuns` 是 `ownershipGroupRuns` 的简写，`PREFIX`/`COMPOSITE`/`AFFINE` 等是路由名称；上面的 `Qn`/`Pn` 编号只是为了叙述“先匹配者胜”的优先级，并不等于路由 ID，各分支的稳定 ID 在 §5.11.2 的 `ComputeRouteIndex` 里统一给出。
> - Q5 前缀 span 批量内部再分两种搬运方式：`runBytes` 与源 gap 都是 32B 整数倍时用 `DataCopy`（步长以 32B 块计），否则用 `DataCopyPad`；这不改变路由优先级。

### 4.1 为什么顺序不能随意换

- **Q0 窗口先于标准路由**：窗口会重写几何（合并可仿射的前缀、折叠内步长），是多种碎片几何的统一解法；但它只接受严格有界的情形，不接受时原几何原样进入标准路由。
- **Q1 连续整块最先**：整体连续是成本最低、最简单的路径，不能被任何“更特化但更贵”的分支抢占。
- **Q2 列分片在仿射行之前**：它只取决于 `runBytes > tileBytes` 与核数/组数关系，判断非常便宜；不命中就继续向下，不会污染后续几何。
- **Q3 单前缀仿射行必须排在 Q5 多前缀之前**：历史上曾把多 prefix span 批量做成泛化路径，结果 rank-one 快路径被吞掉，出现约 70 倍回退。rank-one 是等长行最优解，必须优先保护。
- **Q4 周期窗口在 Q5 之前**：它是极窄的精确门禁，命中的一定是该专用路径最优；但它只覆盖一个验证过的几何，宁可不命中也不能放宽误抢。
- **Q5 可整除 span 在 Q6 复合组之前**：span 内批量不需要 Gather，成本更低；跨 wrap 才需要 Gather 复合。
- **Q8 对齐路径在 Q9 packed 之前**：能直接 DMA 的不要落到标量搬运。
- **Q9 packed 永远最后且永远可达**：任何特化失败都必须有正确性 fallback。

---

### 4.2 如何独立判断一个几何该走哪条路由

不要背路由表。拿到一个 `shape / stride / dtype` 后，按下面这条推理链走，每一步都在做“要不要继续”的判断。

```text
输入 tensor (shape, element-strides, dtype)
  │
  ├── 1. 归一化
  │      从最后一个维向前折叠物理连续后缀
  │      → prefixRank / runBytes / runCount / ownershipGroupRuns
  │
  ├── 2. 宏窗口裁决
  │      TryDirectPaddedWindow  → 非对齐连续长 run [64,4096]，直接 Padded DMA
  │      TryMacroWindow         → 短 run ≤256B，读放大 ≤8×，可批量 Gather
  │      命中 → ROUTE_MACRO_WINDOW；未命中 → 进常规路由
  │
  └── 3. 常规路由链（依次问，第一个命中就停）
        P0  prefixRank==0 ?                       → CONTIGUOUS
        P0.5 runBytes>tile && 32B对齐 && 核可铺满列 ? → COLUMN_SLICED
        P1  prefixRank==1 && pitch合法 ?          → AFFINE
        P2a 精确5D周期几何 ?                      → PERIODIC
        P2b prefixRank>=2 && innerExtent%ogRuns==0 ? → PREFIX
        P3  prefixRank>=2 && innerExtent%ogRuns!=0
              && runBytes%32!=0 && group≤8K ?      → COMPOSITE
        P3b 2D && elementSize=4 && 内维stride>elem ? → STRIDED_AFFINE
        P4  runBytes%32==0 ?                      → ALIGNED
        P5  否则                                   → PACKED
```

#### 关键判断点

1. **归一化时要无条件折叠 size==1 的维**，不要因为 stride 不连续就停。
2. **宏窗口优先于一切常规路由**。它失败的条件是：`runBytes>256`、读放大>8×、pitch 不合法、或 batch rows 算出来为 0。
3. **AFFINE 与 PREFIX 的分界是 `prefixRank`**：prefixRank==1 才可能是 AFFINE；prefixRank>=2 才可能进 PREFIX/COMPOSITE。
4. **PREFIX 与 COMPOSITE 的分界是整除**：`innerExtent % ownershipGroupRuns == 0` → PREFIX；否则 → COMPOSITE。
5. **COMPOSITE 与 ALIGNED 的分界是 32B 对齐**：`runBytes % 32 != 0` 才可能 COMPOSITE；对齐的走 ALIGNED。

#### 自测题

给定 `shape=[100, 50]`、`element-strides=[1, 4]`、`dtype=int32`：

- `sourceStrideBytes = [4, 16]`
- 从 d=1 开始：`stride=16 ≠ expected=4` → 不折叠
- `prefixRank=2, runBytes=4, runCount=5000, ogRuns=8`
- 宏窗口：runBytes=4<64，且窗口枚举时 pitch=4<span=788 → 不命中
- 常规路由：AFFINE 需要 prefixRank=1 → 否；PREFIX 需要 `50%8==0` → 否；COMPOSITE 满足（非 32B、不能整除、组≤8K）→ **COMPOSITE**

---

## 5. 各分支详解：问题 → 设计 → 实现 → 效果

### 5.1 整块连续搬运（CONTIGUOUS）

**问题**：当所有维都连续时，整个输出就是一次 GM→UB→GM 的连续拷贝；继续按 run 循环会白白增加循环和同步。

**设计**：`prefixRank == 0` 时把总字节按 tile 切块，均分给所有 core。**为什么按 tile 而不是按字节平均分**：每个 core 一次能搬运的数据受 UB 限制，tile 是真实工作单元；按 tile 分块保证每轮搬运量 ≤ UB，且块边界天然是 32B 对齐。

**实现**：`DataCopy`/`DataCopyPad` 搬运。对齐且大块时优先用 TQueBind 双槽双缓冲；非对齐小块用普通队列 + PipeBarrier。

**效果与边界**：连续尾维 case 在实测中相对参考实现达到 1.1× 以上。非连续短行场景不能走这条路，否则会因逐 run 而退化。

### 5.2 大对齐 run 的列分片（COLUMN_SLICED）

**问题**：一个非常大的 32B 对齐 run（例如单个 run 数 MB）如果每核只分到一个 run，很多 core 会闲置；但把一个 run 切成列块，每核就能分到一块工作。

**设计**：当 `runBytes > tileBytes` 且 ownership group 数小于核数、但 `groups × 列块数` 足以铺核时，把调度网格展开为 `(group, columnTile)`。**为什么列边界必须 32B 对齐**：对齐 run 按 32B 边界切，各 core 的 GM 写区间天然不重叠，不需要额外同步。

**实现**：与普通对齐 run 共用 kernel；`runBytes > tileBytes` 时进入列分片循环，每核处理若干 `(group, columnTile)` cell。

**效果与边界**：单大 run case 从约 50 μs 降到 43 μs，达到相对参考 1.30×。若核数已经足够按 group 铺满，不必列分片——多一次切分只增加 loop 开销。

### 5.3 单前缀等长行批量（AFFINE_ROWS）

**问题**：`prefixRank == 1` 且所有行等长、源 pitch 恒定时，逐 run 搬运会让每行都付出一次 DMA 参数构造、队列和同步开销。实测中最差的一个这类 case 逐 run 时慢到数百 μs，批量后降到几 μs。

**设计**：用一次 `DataCopyPad(blockCount=rows, blockLen=runBytes, srcStride=gap)` 一次搬多行，输出同样用多 block `DataCopyPad` 写回。**为什么输出用 DataCopyPad 而不是 DataCopy**：当 run 非 32B 对齐时，UB 行槽是 padded 的，DataCopyPad 写回会自动丢弃 padding，输出保持连续。

**实现分三种子路径**：

| 子路径 | 条件 | 为什么这样选 |
|---|---|---|
| 对齐 TBuf + 精确事件 | `runBytes % 32 == 0` | 不需要 padding，单 TBuf 即可；精确事件比粗 barrier 开销小 |
| 对齐 + bound 双缓冲 | `runBytes % 32 == 0 && useBoundQueue != 0` | 双槽 TQueBind 允许 MTE2/MTE3 重叠，适合大 tile 批量；非对齐时行槽带 padding，不能走 bound 路径 |
| 非对齐 + 普通队列 | 其余 | 通用安全路径，PipeBarrier 保证依赖 |

**效果与边界**：等长短行类 case 从数百 μs 降到个位数 μs，部分 case 达到参考的 1.2×。非等长、pitch 不恒定、或 batch 会跨 prefix 边界时不能使用。

### 5.4 多前缀可整除 span 批量（PREFIX_SPANS）

**问题**：多前缀维、内层行等长且 pitch 恒定时，如果 inner extent 能被 ownership group 整除，则每个 batch 不会跨内层 wrap，可以安全地批量搬运。

**设计**：每批只处理同一个 innermost prefix span 内的行。**为什么强调整除**：ownership group 是写回安全边界；不能整除意味着批会跨 wrap，需要交给复合组路径。

**实现**：aligned 情形用 `DataCopy`（blockLen/srcStride 以 32B 块为单位），非对齐用 `DataCopyPad`。同步保留 PipeBarrier——实测曾尝试全量替换为精确事件，在部分路径上出现约 2 倍回退，因此这里只保留通用 barrier。

**效果与边界**：多 prefix 中规则场景可用；跨 wrap 或 inner extent 不整除时交给复合组或 packed。

### 5.5 跨 wrap 复合组（COMPOSITE_OWNERSHIP_GROUPS）

**问题**：多 prefix、run 非 32B 对齐、且 ownership group 会跨 prefix wrap。不能简单批量 DMA，因为 UB 行槽 padding 和输出 32B 所有权会同时出现。

**设计**：以 ownership group 为单位处理：先把组内几个 run 用 `DataCopyPad` 搬进 UB 的 padded 行，再 `Gather` 紧凑成连续输出，最后**一个 group 只发一次 MTE3** 写回。**为什么必须单次 MTE3**：group 内相邻短 run 可能共享同一个 32B 输出块，多次写会互相覆盖；一次写保证该块只有一个 owner。

**实现**：Gather 偏移表由 Host 预计算，kernel 一次性从 TilingData 旁挂缓冲区读入。**为什么把表放 Host 算**：device 端 scalar SetValue 双循环构造 offset 表非常慢；改为 Host 构造 + 单次 DataCopy 后，复合类 case 相对参考的加速比提升 30%–95%（例如 5D int64 从 0.452× 提升到 0.882×）。

**效果与边界**：5D int64 等跨 wrap case 从 12 μs 降到 7 μs 以下。组字节超过 8K 或无法用 Gather 表示时不使用。

### 5.6 2D 固定内步长仿射折叠（STRIDED_AFFINE）

**问题**：2D 视图中，每行元素源 stride 固定但不是紧邻（例如每行 5 个元素、元素间隔 8B）。物理连续后缀折叠后每个元素仍是一个 run，runCount 巨大，固定开销随行数线性放大。

**设计**：把“一行固定 stride 的元素”折叠成一个逻辑 run，`runCount` 从 `n×5` 降到 `n`。**为什么只对 2D、4B 元素开放**：每个元素需要独立 32B 对齐的 UB 槽才能安全 DMA；2B 元素曾遇到 DataCopyPad 对齐问题，因此在获得新证据前不放行。

**实现**：Host 先尝试折叠，折叠后必须再次确认 route 可用；不可用要完整回滚折叠，否则会污染其他路由的几何。kernel 把每个元素读入独立 32B slot，再 `Gather` 紧凑。

**效果与边界**：对 `[n, 5] stride8` 类输入有效；折叠失败必须回滚是正确性红线。

### 5.7 周期窗口专用（PERIODIC_PREFIX_WINDOW）

**问题**：某些视图是精确的周期抽取（例如每 256B 源行取前 64B，256 个 plane）。逐 plane 的 MTE2/MTE3 和 barrier 次数是主要开销。

**设计**：把完整 plane 按核划分（每个核 6–7 个 plane），每核只做**一次 2D MTE2 + 一次 GatherMask + 一次连续 MTE3**。**为什么固定 40 核**：plane 总数 256，40 核时每核 6–7 个 plane 最平衡；实测做过 20 核对照，慢约 10%，因此 40 核写死为准入条件。**平台标注**：这里的 40 核是 **910B3** 侧的实测结论；换平台（如 950PR，AIV 56）后核数必须由 `GetCoreNumAiv()` 取得，并按同样的「每核 6–7 个 plane」平衡原则重新标定，**禁止把 40 写死到 950PR**。

**实现**：准入谓词非常严格（rank、prefixRank、runBytes、runCount、核数、内层 stride 全部精确匹配，并证明前缀维可仿射展平）。命中后 2D `DataCopyPad` 一次读入所有 plane，`GatherMask` 一次抽取窗口，单次 MTE3 写回。

**效果与边界**：该 case 从 10.4 μs 降到 5.8 μs。这条路由非常窄，但它的存在价值正是“精确门禁 + 专用最优”，其他几何绝不能误入。

### 5.8 宏窗口 + 批量 Gather（MACRO_WINDOW / Gather 子路径）

**问题**：高 rank、短 fragment 的场景，逐 fragment 的 DMA/同步/Gather 次数太多。为每个 fragment 建表、搬入、压缩、写回，固定开销占比极高。

**设计**：把“完整后缀窗口”当作宏行，一次搬入窗口的源 span，用一张复用的 Gather 表把窗口内有效字节压成连续输出。**为什么窗口必须有界**：源 span 大于有效输出字节太多时会读放大过大，省下的 issue 开销被 HBM 读流量抵消。因此准入要求 `span ≤ 8 × 有效字节`、**原始物理连续后缀 `runBytes ≤ 256B`**（即“短 fragment”门禁；窗口改写后的宏行字节不受 256B 限制），且 inner pitch 合法。

**实现（可编码）**：

Host 端：

```text
def TryMacroWindow(tiling):
    if prefixRank == 0 or runBytes > 256 or outputRank < 2: return false
    # 从后往前枚举最大后缀
    suffixElements = 1; span = elementBytes
    for d in (outputRank-1) .. 1:
        extent = outputShape[d]
        if extent == 0 or suffixElements > 4096/elementBytes/extent: break
        suffixElements *= extent
        if extent > 1 and sourceStrideBytes[d] > (16384 - span)/(extent-1): break  # 先界住 span 再乘
        span += (extent-1)*sourceStrideBytes[d]
        bytes = suffixElements*elementBytes
        if span > bytes*8: continue  # 读放大超 8×
        # 找 innerPrefix 的 pitch
        innerPrefix = d-1
        while innerPrefix >= 0 && outputShape[innerPrefix]==1: innerPrefix--
        if innerPrefix < 0: continue
        pitch = sourceStrideBytes[innerPrefix]
        if pitch < span or pitch-span > UINT32_MAX: continue
        group = 32/gcd(bytes,32)
        rows = min(4095, 65536/Align32(span), 16384/bytes)
        rows = rows/group*group
        if rows == 0: continue
        if span == bytes && bytes == runBytes && prefixRank > 1 && outputShape[prefixRank-1] % group == 0:
            continue  # 连续后缀+可整除prefix → 直接MTE，不建表
        # 接受此后缀
        # 重写几何：折叠可仿射前缀，删除 singleton
        ...
    # 生成 offset 表
    offsets = windowBatchRows * (runBytes / 2) 个 int32
    for each row, element: offsets[row * units + e * (elem/2) + part] = row * Align32(span) + source_of_element(e) + part*2
    # 重写 outputShape/sourceStrideBytes：合并连续可仿射前缀
```

Kernel 端：

```text
// Init: 一个输入槽 depth1, 两个输出槽 depth2 (或 bound 双槽)
slot = Align32(span)
out_bytes = windowBatchRows * runBytes
out_slots = 2; in_slots = 1
InitBuffer(inQueue, 1, windowBatchRows * slot)
InitBuffer(outQueue, 2, out_bytes)
InitBuffer(scratch, out_bytes * 2)  // 放 offset 表

// Process:
// 注意：应在循环前一次性读取全量表：
// DataCopy(offsets.ReinterpretCast<uint8_t>(), table_, windowBatchRows * runBytes * 2U)
for start in first .. last step windowBatchRows:
    rows = min(windowBatchRows, last-start)
    input = AllocTensor(inQueue)
    for each span row in this batch:
        DataCopyPad(input[paddedRow], x_gm[source],
            {spanRows, windowSpanBytes, gap, 0, 0}, {false,0,0,0})
    EnQue(inQueue); input = DeQue()
    output = AllocTensor(outQueue)
    Gather(outputU16, inputU16, offsetsU32, 0, rows * (runBytes / 2))
    PipeBarrier<V>  // 或用 EnQue/DeQue 替代
    EnQue(outQueue); output = DeQue()
    DataCopyPad(y_gm[start * runBytes], output, {1, rows*runBytes, 0, 0, 0})
    FreeTensor(input); FreeTensor(output)
```

**为什么批行数可以大于 innerExtent（关键原理）**：

- MACRO_WINDOW 的“宏行”不是 innerExtent 内的单行，而是**完整后缀窗口**；
- 一个 batch 可以横跨多个 prefix 位置。只要 Kernel 使用源游标（source cursor）维护“当前宏行对应到第几个 prefix 位置以及源地址”，`windowBatchRows` 完全可以大于 `innerExtent`；
- 因此 Host 在计算 `rows` 时**不要用 `min(rows, innerExtent)` 截断**，也不要在 `innerExtent % group != 0` 时直接退化为单核；
- 多核安全只由 ownership group 边界保证：每个核从完整 ownership group 开始，只有全局尾组可以是不完整组；
- 如果错误地按 `innerExtent` 限制 batch，小 innerExtent + 大 runCount 的场景会变成“单核 + 极多小批量”，固定开销被放大成数百倍性能退化。

**为什么连续后缀且 prefix 可整除时退回直接 MTE**：此时无需 Gather，建表反而多余。

**效果与边界**：多个高 rank 短 fragment case 获得 3–18 倍自身耗时改善。稀疏窗口、大 run、跨 wrap 的不规则复合不走此路。

### 5.9 非对齐长 run 直接 Padded DMA（MACRO_WINDOW / direct 子路径）

**问题**：有些连续后缀 run 较长（64B..4096B）但非 32B 对齐，且行会跨 prefix 边界。以前这类输入会落到 packed 标量 fallback，出现 O(outputBytes) 的标量拷贝和 O(outputBytes/32) 的同步，慢到数千 μs。

**设计**：这类 run 的输出行本身是连续的，只是源行不 32B 对齐。可以用双槽 TQueBind + 输入 `DataCopyPad` 把每行读进 padded UB 槽，输出用一次多 block `DataCopyPad` 把 padding 自动剥掉，直接写出连续行。**为什么不建 Gather 表**：行内没有跨 stride 的选元素操作，唯一问题是 padding；DataCopyPad 输出端自动处理了。

**实现**：

准入要求：

- `prefixRank ≥ 2`；
- `64B ≤ runBytes ≤ 4096B`；
- `runBytes % 32 != 0`；
- 源 pitch 与 run 的差在 uint32 范围内；
- `windowSpanBytes == runBytes`（行内没有元素选择，只有非 32B 对齐问题）。

Buffer 结构：

- 使用 `TQueBind<VECIN, VECOUT, 2>` 双槽，输入与输出**共享同一个 slot**；
- 每个槽大小：`windowBatchRows * Align32(runBytes)`（行内按 `Align32(runBytes)` 存放；输出写回时每行只取前 `runBytes` 字节，padding 由 MTE3 的 `DataCopyPad` 自动剥掉）；
- 不分配 Gather offset 表。

Host 批行数：

```text
rows = min(4095, UB预算 / Align32(runBytes))
rows = rows / group * group   # group = 32 / gcd(runBytes, 32)
```

Kernel 输入：

- 使用源游标跨 prefix carry；
- 在一个 prefix span 内连续多行时：
  ```cpp
  DataCopyExtParams cp{spanRows, runBytes, gap, 0, 0}; // gap = sourcePitch - runBytes
  DataCopyPad(input[local * slot], x_[src], cp, pad);
  ```
- 输入行写入 `Align32(runBytes)` 的 padded slot。

Kernel 输出：

- 一次多 block `DataCopyPad`：
  ```cpp
  DataCopyExtParams out{rows, runBytes, 0, 0, 0};
  DataCopyPad(y_[start * runBytes], data, out);
  ```
- MTE3 自动剥掉每行 padding，写出连续行。

**为什么不能把这类几何误归入 MACRO_WINDOW + Gather**：

- direct Padded DMA 的行内没有跨 stride 选元素操作，唯一问题是 padding；
- Gather 需要额外建表、读表、向量压缩，纯属多余；
- 对 1638B / 80B 这类非对齐长 run，direct Padded DMA 可以把最差 case 从数千 μs 降到十几 μs；误走 Gather 或 PACKED 会慢一个数量级以上。

**效果与边界**：最差的非对齐长 run case 从约 8800 μs 降到 18 μs；80B 行 case 也改善约 18%。run 小于 64B 时批收益小于固定开销，run 大于 4096B 时改用其他路径更稳。

### 5.10 packed fallback（最后安全网，禁止吞掉核心路径）

**问题**：所有特化都不匹配时，必须有一条永远正确的路径。

**设计**：以 32B 输出块为单位，把不规则源片段逐段读进 32B scratch，拼满后由单一 owner 一次性写回。**为什么最小单位是 32B 而不是逐元素写**：MTE3 写回以 32B 块为粒度，多个短写若共享一个物理块会互相覆盖；UB 内先拼满 32B 再写，保证每个物理块只有一个 owner。

**实现**：标量 `SetValue` 逐字节搬运到 32B scratch，满 32B 写一次；尾块按实际长度写。慢，但正确且不越界。

**硬性约束**：

- packed fallback 只能在所有适用的 `AFFINE_ROWS`、`PREFIX_SPANS`、`COMPOSITE`、`STRIDED_AFFINE`、`PERIODIC`、`MACRO_WINDOW` 和非对齐长 run direct-Padded-DMA 门禁均不满足时使用。
- 若实现选择用 `blockCount=1` 作为 DMA fallback，必须把它绑定到 packed/fallback route；不能仅因为 source pitch 非 32B 对齐，就让所有本可批量处理的几何都退化到单 block。
- 性能报告必须记录 fallback 命中的 route、原因、descriptor 数和平均 `blockCount`。常见 canonical case 大量命中 fallback 时，设计不能判定为完成。

**效果与边界**：最终 fallback。优化的目标就是让常见几何尽量不落到这里。

### 5.11 完整实现路径：从几何到可生成代码

上面 5.1–5.10 已经讲清了为什么这么设计。下面给出实现一个可运行、可接近 1.8× 水平的算子所必需掌握的**编码公式**。分为三块：头文件规格、Host 算法、Kernel 骨架。每块的输出就是可写的代码。

---

#### 5.11.1 Host Kernel 共享 ABI —— 先写头文件

```cpp
#include <cstdint>
constexpr uint32_t MAX_RANK = 8U;
struct StridedSliceV2Tiling {
    uint32_t outputRank;        // 输出维数
    uint32_t prefixRank;        // 折叠后的前缀维数
    uint32_t usedCoreNum;       // 开核数
    uint32_t tileBytes;         // 单核批处理容量
    uint64_t outputElements;    // 输出总元素数
    uint64_t runElements;       // 单个 suffix run 的元素数
    uint64_t runCount;          // outputElements / runElements
    uint64_t runBytes;          // runElements * elementSize
    uint64_t ownershipGroupRuns;// 32 / gcd(runBytes, 32)
    uint64_t outputShape[MAX_RANK];
    uint64_t sourceStrideBytes[MAX_RANK]; // x.stride(d) * elementSize
    uint32_t routeIndex;        // host 决策结果
    uint32_t useBoundQueue;     // 0/1/2
    uint32_t windowSpanBytes;   // 宏窗口的源 span
    uint32_t windowBatchRows;   // 窗口批行数
};
static_assert(sizeof(StridedSliceV2Tiling) == 200);
constexpr uint64_t STRIDED_SLICE_COMPOSITE_TABLE_OFFSET = (200 + 31) / 32 * 32;  // 224
```

Host 把这张结构 `memcpy` 进 CPU ByteTensor，再把 `.to(kPrivateUse1)` 传给 kernel。

---

#### 5.11.2 Host 归一化与路由判定 —— 决定 routeIndex 和 tiling 参数的完整算法

**1. 后缀折叠（决定 runBytes / runCount / prefixRank）：**

```text
outputShape[d] = x.size(d)
sourceStrideBytes[d] = x.stride(d) * elementSize
prefixRank = outputRank
runElements = 1
expectedStride = elementSize
for d in (outputRank-1) .. 0:
  if x.size(d) != 1 && sourceStrideBytes[d] != expectedStride: break
  runElements *= x.size(d)
  expectedStride *= x.size(d)
  prefixRank = d
runCount = outputElements / runElements
runBytes = runElements * elementSize
```

**2. 旧 tile 定 route 的方向：**

旧 tile 只用于路由判定，不用于最终吞吐：

```text
oldTile = (runBytes >= 16384) ? 81920 : 49152  // 80 KiB 与 48 KiB：按 UB 预算取的经验值（910B3）
                                               // 950PR（UB 248 KiB）上未重测，换平台须重新标定
```

**3. 宏窗口预处理（优先于标准路由）：**

先尝试 `TryDirectPaddedWindow`：

```text
if prefixRank >= 2 && 64 <= runBytes <= 4096 && runBytes % 32 != 0:
  pitch = sourceStrideBytes[prefixRank - 1]
  gap = pitch - runBytes
  if gap <= 0xFFFFFFFF:  // uint32
    group = 32 / gcd(runBytes, 32)
    rows = min(4095, batchBudget / Align32(runBytes))   // batchBudget 为输入槽字节预算（910B3 经验值 80 KiB；950PR 未重测）
    rows = rows / group * group
    if rows > 0: routeIndex = MACRO_WINDOW; direct = true
```

未命中再试 `TryMacroWindow`，其完整条件与几何重写在 5.8 节。

**4. 标准路由判定（按优先级，先匹配者胜）：**
> 注：`ComputeRouteIndex` 会用到 `usedCoreNum`；实现中先按第 5 步估算开核数、再调用它，之后 AFFINE 路由还可能按 §6.3 的窄门控二次减核。这里按“先讲路由、再讲开核回填”的顺序叙述，只为便于阅读。

```text
ownershipGroupRuns = 32 / gcd(runBytes, 32)
groupCount = ceil(runCount / ownershipGroupRuns)
colChunks  = ceil(runBytes / oldTile)     // 列分片时一个 run 被切成几块

function ComputeRouteIndex:
  if prefixRank == 0                       → CONTIGUOUS
  if runBytes % 32 == 0 && runBytes > oldTile
     && groupCount < usedCoreNum <= groupCount * colChunks → COLUMN_SLICED
  if prefixRank == 1 && CanBatchAffineRows → AFFINE_ROWS
  if CanUsePeriodicPrefixWindow            → PERIODIC
  if CanBatchDivisiblePrefixSpans          → PREFIX_SPANS
  if CanCopyCompositeOwnershipGroups       → COMPOSITE
  if CanUseStridedAffine                   → STRIDED_AFFINE
  if runBytes % 32 == 0                    → ALIGNED_RUNS
  else                                     → PACKED
```

`CanBatchAffineRows` 等子函数的精确条件，见 §5.3–§5.10 各分支小节的准入条件，以及 §5.11.2 中对应子函数的判据。

**路由实现门禁（生成 DoD）**：

1. Host 的 route 判定必须使用本文定义的几何特征和优先级，不能只根据第一个 descriptor 的 `blockCount` 或 `blockLen` 粗略贴 route 标签。
2. TilingData 中的 `routeIndex` 必须被 Kernel 消费；Kernel 必须对 route 做真实分发。若各 route 最终调用同一个 `ProcessDescriptor`，视为路由未实现。
3. 每个已实现 route 必须拥有自己的 buffer/同步/搬运策略，并至少有一个命中 case、一个阈值边界 case 和一个保护 fallback case。
4. `blockCount=1` 只能作为明确记录原因的 fallback；批量 DMA 不得仅因“非 32B 对齐”而被全局关闭。应优先使用本文对应的 Padded DMA、ownership group、Gather 或宏窗口路径。
5. 若因 CANN/SoC 实测 fault 暂时不能启用某条批量路径，必须保留该路径的设计与验证记录，并继续为其实现更小 tile、分组 DMA、Gather 或其他可测量替代方案；不能把暂时 fallback 当成最终高性能实现。

**5. 开核数与动态 tile 回填：**

```text
// 开核 = 可调度组数（不含 per-core tile 二次压缩）
schedulableGroups = (prefixRank == 0) ? ceil(runBytes / oldTile)
                 : (runBytes % 32 == 0 && runBytes > oldTile) ? groupCount * colChunks
                 : groupCount
usedCoreNum = min(availableCores, max(schedulableGroups, 1))

// 决定 bound queue 深度
useBoundQueue = (route in [CONTIGUOUS, AFFINE, ALIGNED, COLUMN_SLICED] && tile==80K && runBytes%32==0)
                ? (tile >= 80K ? 2 : 1) : 0    // 80K 为 910B3 经验阈值；950PR 未重测

// 回填真正 tile
ubBudget = GetCoreMemSize(CoreMemType::UB)   // 运行时取值；910B3 示例 192 KiB，950PR 示例 248 KiB（INI）/ 216 KiB（运行时属性，预算取后者）
maxTile = useBoundQueue ? ubBudget / useBoundQueue : (ubBudget - 16K) / 2
tileCap = maxTile
if prefixRank == 0: tileCap = min(tileCap, (runBytes * runCount + usedCoreNum - 1) / usedCoreNum)
tileBytes = Align32(tileCap)
// 对 AFFINE/PREFIX/COMPOSITE/STRIDED_AFFINE 四条 route，tile 至少容纳一个完整 ownership group
if route in [AFFINE, PREFIX, COMPOSITE, STRIDED]: minTile = ownershipGroupRuns * Align32(runBytes)
tileBytes = max(tileBytes, minTile)
```

---

#### 5.11.3 每条路由的 Kernel 骨架模板

所有 Kernel 共享相同的连接参数：

```cpp
extern "C" __global__ __aicore__ void kernel_name(GM_ADDR x, GM_ADDR y, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    KernelClass k(&pipe);
    k.Init(x, y, tiling);
    k.Process();
}
```

每个 Kernel 类均通过 `CopyTiling` 加载共享结构，然后按 `routeIndex` 执行唯一的 Process。下面以 COMPOSITE 为例展示必须的具体公式，其余路由按同样的“分配本核 group → 批量搬入 → 紧凑/写回”骨架替换即可，各分支的公式见对应小节。

**COMPOSITE 骨架关键公式：**

```cpp
// Process 中每核分得的组：
uint64_t groupCount = ceil(runCount / ownershipGroupRuns);
uint64_t first = groupCount * core / usedCoreNum * ownershipGroupRuns;
uint64_t last = min(groupCount * (core + 1) / usedCoreNum * ownershipGroupRuns, runCount);

// per-group 循环：
for group in [first .. last step ownershipGroupRuns]:
    // 1. 读 offset 表
    GlobalTensor<uint8_t> tableGm(tilingGm + STRIDED_SLICE_COMPOSITE_TABLE_OFFSET);
    DataCopy(offsetsLocal, tableGm, fullTableBytes);

    // 2. 用 DataCopyPad + Gather 紧凑（见 5.5）
    for each row in group:
        DataCopyPad(input[paddedRow], x_gm[source], {spanRows, runBytes, gap, 0, 0}, pad);
    EnQue(input);
    input = DeQue();
    Gather(outputU16, inputU16, offsetsU32, 0, groupRows * (runBytes / 2));
    EnQue(output);
    DeQue();
    DataCopyPad(y_gm[group * runBytes], output, {1, validBytes, 0, 0, 0});
```

所有路由使用同一组 API（DataCopyPad / Gather / TQue / PipeBarrier / SetFlag、WaitFlag），具体参数单位见 5.11.4。

---

#### 5.11.4 AscendC API 参数单位速查

| API | 参数意义 | 单位 | 平台差异（910B3 / 950PR） |
|---|---|---|---|
| `DataCopyParams{blockCount, blockLen, srcStride, dstStride}` | 多 block 跨 stride 搬运（字段均为 uint16；新版头文件已把 `srcStride/dstStride` 标记为待弃用，等价于 `srcGap/dstGap`） | blockCount 为 block 数；blockLen/srcStride/dstStride **以 32B 块为单位** | 两平台同（单位均为 32B 块） |
| `DataCopyExtParams{blockCount, blockLen, srcStride, dstStride, rsv}` | 多 block 跨 stride 搬运（blockCount 为 uint16，其余为 uint32） | `blockLen` **字节**；`srcStride/dstStride` 是**不含 `blockLen` 的间隙（gap）**，方向决定单位：**GM 侧 1 字节、UB 侧 32B 块**；blockCount ≤ 4095（本平台上位 API 可达 65535） | **950PR 上 `srcStride/dstStride` 为 `int64_t`**（910B3 为 `uint32_t`）；`blockLen` 上限 2097151 字节 |
| `DataCopyPadExtParams{isPad, leftPadding, rightPadding, paddingValue}` | padding 模式；用于 DataCopyPad | isPad=true 时自动补到 32B 右对齐 | 950PR 新增 `PaddingMode::Normal/Compact` 模板参数：`Compact` 下 UB 侧行距 = `blockLen`（不向上 32B 对齐），910B3/`Normal` 按 `align32(blockLen)` 补齐 |
| `TQueBind<VECIN,VECOUT,D> queue` | 绑定队列，D≥2 时双缓冲 | `InitBuffer(queue, D, tile)`，tile 为每槽字节数 | 两平台同 |
| `PipeBarrier<PIPE_ALL>` | 粗粒度等待 | 通用，开销中等 | 两平台同 |
| `SetFlag<MTE2_MTE3>(event)` / `WaitFlag` | 精确事件 | 只用于简单路径 | 两平台同；**NDDMA 多维 `DataCopy` 无 pipe 注解，必须靠它显式同步**（950PR 专属场景） |
| `Gather(dstU16, srcU16, offsetsU32, 0, count)` | 按偏移表紧凑，count 为待取元素数 | 偏移表是 `LocalTensor<uint32_t>`；把 src/dst 都重解释为 uint16 后，表项与 offset 均按 **uint16 元素** 计 | 两平台同；950PR 另有 RegBase `Gather`（见 §5.12），越界语义不同 |
| `GatherMask(dstU32, srcU32, maskU32, false, 0, GatherMaskParams(...), retained)` | 位掩码抽取，掩码低位优先 | 末尾 `retained` 是输出参数（本次实际保留的元素数）；`GatherMaskParams(src0BlockStride, repeatTimes, src0RepeatStride, src1RepeatStride)` | 两平台同 |
| DataCopyPad 重载 | UB←GM（MTE2）为 (dstLocal, srcGm, DataCopyExtParams, DataCopyPadExtParams) 四参；GM←UB（MTE3）为 (dstGm, srcLocal, DataCopyExtParams) 三参，**不接受 pad 参数** | — | 两平台同 |

---

#### 5.11.5 分步构建顺序

1. **TilingData 结构**（5.11.1）→ 确定 ABI 后 host/kernel 两端可以同时开发。
2. **Host 归一化 + 基础路由**（5.11.2）→ 先实现 CONTIGUOUS + PACKED，所有 shape 先跑通。
3. **每加一条路由，分别实现这三块**：
   - Host 准入 if 条件（5.11.2 的子函数）+ 对应 tiling 字段赋值；
   - Kernel 类 + 入口 cpp（5.11.3 骨架）；
   - 测试至少 1 个命中 case + 1 个边界 case + 1 个保护 case（最终验收按 §9.4 的 2/2/2 口径）。
4. **最后补 tile/core 动态联动和 bound queue**：此时路由已全，只改 Host 的 tile 回填和 `useBoundQueue` 决策，kernel 端只对应调整 `InitBuffer` 的参数。

5. **平台/工具链前提（启用 §5.12 的 dav-3510 机制前必须确认）**：`--enable-simt` 与 AscendC 框架在**同一编译单元内**互斥——该选项会定义 `__NPU_COMPILER_INTERNAL_PURE_SIMT__`，`__simd_vf__` 宏消失、`asc_vf_call` 被排除、`#include "kernel_operator.h"` 报错。因此 **RegBase 与 SIMT 分属不同候选、不混编**；SIMT 只能放到**独立编译单元**（只含 `simt_api/*`）+ 给该 TU 单独加 `--enable-simt`，再由设备链接合进同一个 `.so`（该形态已实测上板可用，61/61）。RegBase 启用前先做最小冒烟编译（`bisheng --npu-arch=dav-3510` + `RegTensor` / `__VEC_SCOPE__` / `LoadAlign`）；Loop Mode 的 `SetLoopModePara` / `ResetLoopModePara` 必须成对调用。

这样每条路由的增强都是一次局部改动，不影响其他路由。

#### 5.11.6 可执行实现规格（Gather 表 / 多行 Gather / GatherMask / AIV_ONLY）

> StridedSlice 是**纯数据搬运算子**，
> 所有入口均为 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`，**不需要 AIC / Cube / mix kernel**。
> 只有需要 Cube 计算（如 MatMul）或 AIC+AIV 分职时才考虑 mix；本算子不要引入 mix kernel。

##### 5.11.6.1 Offset/Gather 表在 TilingData 后的传递方式

- 固定 ABI：`StridedSliceV2Tiling` 固定 `sizeof == 200B`。
- 表紧跟在 struct 后，偏移量用常量：

```cpp
constexpr uint64_t STRIDED_SLICE_COMPOSITE_TABLE_OFFSET =
    (sizeof(StridedSliceV2Tiling) + 31U) / 32U * 32U;  // 即 224
```

- Host 端构造 `tilingCpu` 为 `at::empty({tableOffset + fullTableBytes}, kByte)`（`tableOffset` 是表相对 TilingData 起点的偏移，取 32B 对齐后的 224，**不能用结构体大小 200**）：
  - 前 200B `memcpy` tiling struct；
  - `tableOffset` 后 `memcpy` offsets 表；
  - `tilingCpu.to(kPrivateUse1, true, true)` 传给 kernel。
- Kernel 端在 `Init` 中：

```cpp
CopyTiling(&tiling_, tilingGm);
table_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(tilingGm) +
                      STRIDED_SLICE_COMPOSITE_TABLE_OFFSET);
// 一次性读表：
AscendC::DataCopy(offsets.ReinterpretCast<uint8_t>(), table_, tableBytes);
AscendC::PipeBarrier<PIPE_MTE2>();
```

- `Gather` 的偏移表统一以 **uint16_t 单元** 表示（把 `LocalTensor` 和 `GlobalTensor` 都 `ReinterpretCast<uint16_t>()`）。

##### 5.11.6.2 COMPOSITE / STRIDED_AFFINE 的 Host 表生成公式

**COMPOSITE（每个 run 一行，UB 中行宽 `Align32(runBytes)`）**

```cpp
const uint32_t unitsPerRun    = runBytes / sizeof(uint16_t);
const uint32_t fullCount     = ownershipGroupRuns * unitsPerRun;
const uint32_t paddedRunBytes = Align32(runBytes);
for (row = 0; row < ownershipGroupRuns; ++row)
    for (unit = 0; unit < unitsPerRun; ++unit)
        table[row * unitsPerRun + unit] =
            row * paddedRunBytes + unit * sizeof(uint16_t);
```

Kernel 使用：

```cpp
Gather(output.ReinterpretCast<uint16_t>(),
       input.ReinterpretCast<uint16_t>(),
       offsets,          // LocalTensor<uint32_t>
       0U,
       groupRows * unitsPerRun);
```

**STRIDED_AFFINE（2D 固定内步长，每个元素独立 32B slot）**

```cpp
const uint64_t innerExtent        = outputShape[1];
const uint64_t elementBytes     = runBytes / innerExtent;
const uint64_t paddedElementBytes = Align32(elementBytes);
const uint64_t rowSlotBytes    = innerExtent * paddedElementBytes;
const uint32_t unitsPerElement = elementBytes / sizeof(uint16_t);
const uint32_t unitsPerRun    = runBytes / sizeof(uint16_t);
for (row = 0; row < ownershipGroupRuns; ++row)
    for (e = 0; e < innerExtent; ++e)
        for (unit = 0; unit < unitsPerElement; ++unit)
            table[row * unitsPerRun + e * unitsPerElement + unit] =
                row * rowSlotBytes + e * paddedElementBytes + unit * sizeof(uint16_t);
```

Kernel 的 CopyIn 需要先把每个元素放到独立 32B 对齐 slot：

```cpp
for (row = 0; row < groupRows; ++row)
    for (e = 0; e < innerExtent; ++e)
        DataCopyPad(input[row * rowSlotBytes + e * paddedElementBytes],
                   xGm_[row * rowPitch + e * innerStride],
                   {1, elementBytes, 0, 0, 0},
                   {true, 0, 0, 0});
```

然后同一张表 `Gather` 压缩。

##### 5.11.6.3 MACRO_WINDOW 的 Host 表生成公式

```cpp
// selectedElements: 从后往前合并的后缀元素数
// selectedSpan:    该后缀的源 span 字节（不是 runBytes）
// selectedRows:    每批宏行数
units = bytes / 2;   // bytes = selectedElements * elementBytes
for (e = 0; e < selectedElements; ++e) {
    // 将 e 按 outputShape 的 suffix 维拆成坐标，计算 source 字节偏移
    for (row = 0; row < selectedRows; ++row)
        for (part = 0; part < elementBytes / 2; ++part)
            table[row * units + e * (elementBytes / 2) + part] =
                row * Align32(selectedSpan) + source + part * 2;
}
// 重写几何：删除 singleton 前缀维，合并可仿射 carry；
// runBytes = bytes; runCount = outputElements / selectedElements;
// ownershipGroupRuns = 32 / gcd(bytes, 32);
// windowSpanBytes = selectedSpan; windowBatchRows = selectedRows;
```

Kernel 的 CopyIn 使用窗口 span：

```cpp
slot = Align32(windowSpanBytes);
// 输入队列 depth1，输出队列 depth2（或 TQueBind）
DataCopyExtParams cp{spanRows, windowSpanBytes, gap, 0, 0};   // gap = sourcePitch - windowSpanBytes
DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
DataCopyPad(input[local * slot], x_[src], cp, pad);
// ...
Gather(outputU16, inputU16, offsetsU32, 0, rows * (runBytes / 2));
// ...
DataCopyPad(y_[start * runBytes], output, {1, rows * runBytes, 0, 0, 0});
```

##### 5.11.6.3.1 MACRO_WINDOW 实现红线与已知失败模式（实测补充）

> 实测补充：只按 §5.8 / §5.11.6.3 的公式实现、而未满足下面的红线时，
> 会出现 **Gather 输出错位** 和 **UB out-of-bounds**，并被迫回退到逐 run 慢路径。
> 因此实现 MACRO_WINDOW 时必须同时满足以下红线，否则不要启用该 route，也不要简单回退。

1. **Host 表行跨步必须等于 Kernel 的 `Align32(selectedSpan)`，禁止硬编码 32B**
   - 错误写法（实际出现）：
     ```cpp
     uint64_t paddedSpan = 32U;  // debug: assume one 32B DataBlock per row
     ```
   - 正确：Host 表使用 `row * Align32(selectedSpan)`，Kernel 使用
     `slot_ = Align32(t_.windowSpanBytes)` 作为输入行槽；两者必须一致。
   - 只要 `selectedSpan > 32B`，硬编码 32B 会导致表行偏移与 UB 实际布局不一致，Gather 必然错位。

2. **Gather 必须统一使用 `uint16_t` 视角，禁止直接用 `LocalTensor<T>` 调 Gather**
   - 错误写法（实际出现）：
     ```cpp
     Gather(output, input, offsets, 0U, rowCount * units);
     ```
   - 正确写法：
     ```cpp
     AscendC::Gather(output.ReinterpretCast<uint16_t>(),
                     input.ReinterpretCast<uint16_t>(),
                     offsets,               // LocalTensor<uint32_t>
                     0U,
                     validBytes / 2U);      // validBytes = rows * runBytes
     ```
   - offset 表条目是 **uint16_t 视图下的元素偏移**（等价于把 `row * Align32(selectedSpan) + source + part * 2` 这个字节地址除以 2），
     不是 T 元素索引，也不是“每个元素一个 offset”。

3. **表大小和 Gather count 的单位必须一致**
   - 表元素数 = `selectedRows * (bytes / 2)`，不是 `selectedRows * selectedElements`。
   - `Gather` 的 count = `实际行数 * (runBytes / 2)`，不是 `rowCount * selectedElements`。
   - 对 2B 元素（fp16/bf16/int16）`runBytes / 2` 就是元素数；对 4B/8B 元素必须按 `uint16`（2 字节）单元拆分，即单元数 = `runBytes / 2`。
   - 正确公式：`units = bytes / 2U`，`offsets.resize(selectedRows * units)`。

4. **CopyIn 必须按“窗口 span 行”批量搬运，不能复用 AFFINE_ROWS 的 `RunSourceOffset + innerExtent` 循环**
   - 错误：在 MACRO_WINDOW 中仍按输出 run 逐行 `RunSourceOffset(row)`，再用 `innerExtent`
     切分批量，导致几何语义和 Gather 表不匹配。
   - 正确：使用源游标（`InitSourceCursor` / `AdvanceSourceCursor`）维护当前宏行的
     源地址和最后一个前缀维坐标，并在一个源 span 内用：
     ```cpp
     AscendC::DataCopyExtParams cp{spanRows, t_.windowSpanBytes, gap, 0U, 0U};
     // gap = t_.sourceStrideBytes[prefixRank-1] - t_.windowSpanBytes
     AscendC::DataCopyPad(input[local * slot_], x_[src], cp, pad);
     ```
     批量读取连续宏行；不要退化成逐 run 小 DMA。
   - 注意还有一个稠密仿射特例：
     `prefixRank == 1 && sourceStrideBytes[0] == slot_ && slot_ <= 128` 时，
     用单 block、加长 blockLen 合并读取，不要错误地逐行发描述符。

5. **UB 分配方式必须与 buffer 结构匹配；不要改用一次性 `LocalMemAllocator` 而不重算容量**
   - 推荐的非 direct 路径分配是：
     - `in_`：depth 1，`windowBatchRows * Align32(spanBytes)`
     - `out_`：depth 2，`windowBatchRows * runBytes`
     - `index_`：`windowBatchRows * runBytes * 2` 字节，用于 offset 表
   - 配合上述 TQue/TQueBind 结构时，`rows = min(4095, min(65536 / Align32(span), 16384 / bytes))`
     是保守且可用的上限（每批输入 ≤64KB、每槽输出 ≤16KB、`out_` depth 2 共 32KB、表 ≤32KB）。**平台标注**：这些常量为 910B3 经验值，950PR 未重测（其 950PR 等价示例为单条 `TQueBind` depth 2 的 tile ≤112 KiB、两条队列共享 120 KiB）。
   - 如果自行改成 `LocalMemAllocator` 一次性分配 input+output+offset，则必须重新核算：
     ```text
     inputBytes  = batchRows * Align32(spanBytes)
     outputBytes = batchRows * runBytes
     tableBytes  = batchRows * (runBytes / 2) * sizeof(uint32_t)
     totalUB     = inputBytes + outputBytes + tableBytes + queue/double-buffer overhead
     必须 totalUB <= 可用 UB，再按 ownershipGroupRuns 对齐回退 batchRows
     ```
   - 使用 TQueBind / 双缓冲时，`inputBytes/outputBytes` 还要乘以对应 depth。

6. **尾组不能按 `batchRows` 读取或写出超出实际行数的数据**
   - 每批实际行数 `rows = min(batchRows, end - start)`；
   - 若在循环前一次性读全量表：
     `DataCopy(offsets.ReinterpretCast<uint8_t>(), table_, windowBatchRows * runBytes * 2U)`，
     则表必须按 `windowBatchRows` 生成；
   - `Gather` 的 count 和 MTE3 写回长度都必须按实际 `rows`，不能按 `batchRows`。

7. **`batchRows` 不能被 `innerExtent` 截断，也不能因 `innerExtent` 不整除 group 就退化为单核**
   - 原理：MACRO_WINDOW 的“宏行”是完整后缀窗口，不是 innerExtent 内的单行；一个 batch 可以跨多个
     prefix 位置，源游标负责跨 prefix carry。因此 `batchRows` 可以远大于 `innerExtent`。
   - 错误写法（实际出现）：
     ```cpp
     if (rowsAligned > 0 && rowsAligned <= innerExtent &&
         (innerExtent % group) == 0) {
         rows = rowsAligned; multiCoreSafe = true;
     } else {
         rows = min(rowsRaw, innerExtent); multiCoreSafe = false;
     }
     ```
     这会使得小 innerExtent + 大 runCount 的场景变成“单核 + 极多小批量”，固定开销被放大到数百倍。
   - 正确做法：`rows` 直接取完整的 `rowsAligned`；即使 `innerExtent < rowsAligned` 或
     `innerExtent % group != 0`，仍由源游标跨 prefix carry 完成批量读取。
   - 多核安全只由 ownership group 边界保证：每个核从完整 ownership group 开始，只有全局尾组允许不完整。
   - 如果实现不支持跨 prefix carry，那是实现缺陷，不是该几何不适用 MACRO_WINDOW 的理由。

8. **候选选择要优先减少总宏行数（总批次数），不能只看 innerExtent 或 span/bytes**
   - 原理：MACRO_WINDOW 的总成本 ≈ 批次数 × 每批固定开销 + DMA 流量；减少描述符和同步次数才是 Gather 的收益来源。
   - 错误：只选 `innerExtent` 最大的候选，或只挑 `span / bytes` 读放大最小的候选，都可能选中 span 巨大、rows 很小的差路径，导致读放大和批次数同时恶化。
   - 正确：在所有满足密度界（`span ≤ 8 × bytes`）与 span 上界的候选中，优先取**后缀最大**的一个（`runBytes` 最大 → 总宏行数/批次数最少）；同等条件下再比 `batchRows` 大、`innerExtent` 大。
   - 本条与 §5.11.6.3.2 第 4 条的评分顺序必须一致，不能出现两套优先级。

9. **当 `windowSpanBytes == runBytes` 且满足非对齐长 run 条件时，必须优先 direct Padded DMA，不要走 Gather**
   - 原理：这类行内没有跨 stride 的选元素操作，唯一问题是非 32B 对齐 padding；Gather 的表生成、读表、
     向量压缩都是多余开销。
   - 准入条件：`prefixRank >= 2`、`64B <= runBytes <= 4096B`、`runBytes % 32 != 0`、
     `sourcePitch - runBytes` 在 uint32 范围内。
   - 实现：使用 `TQueBind` 双槽 + 多 block `DataCopyPad`，不建 Gather 表；输入写 padded slot，
     输出多 block 自动剥 padding。
   - 若把这类几何误归入 MACRO_WINDOW + Gather 或 PACKED，性能会差一个数量级以上。

10. **启用门禁：必须先用命中 case + 边界 case + 保护 case 验证，再作为默认路径**
   - 至少验证：一个高 rank 短 fragment 命中 case（如 `[20000,2,2,3,3,4,4,5] fp32`）、
     一个尾组 case、一个 UB 容量边界 case；
   - 未通过这些验证前，不允许把 `routeIndex = MACRO_WINDOW` 作为默认路径；
   - 若调试中出现 Gather 输出错误或 UB 越界，优先按上述 1–9 逐项排查，
     **不能直接回退到 AFFINE_ROWS 并把 MACRO_WINDOW 标记为“不可用”**；回退只允许在完成排查并记录具体不满足的红线后使用。

11. **MACRO_WINDOW 不是“所有高 rank 短 fragment 都必须走 Gather”；固定开销过高时必须寻找更轻量路径**
   - 原理：Gather 的收益来自“一次批量读入 + 向量压缩”替代大量小 DMA；但如果每个 batch 还要承担
     表读取、TQue/同步、Gather 本身的固定开销，且有效数据量很小，那么 Gather 路径可能反而比
     直接批量 DMA 更慢。
   - 判定：如果某个 MACRO_WINDOW 命中 case 的耗时远高于同量级 direct MTE / direct Padded DMA，
     应优先检查是否可以在不建表的情况下完成：
     1. `span == bytes` 且行内无跨 stride 选元素 → 走 direct MTE / direct Padded DMA；
     2. `prefixRank == 1` 且源 pitch 等于 padded slot → 走 dense 仿射单 block 合并读取；
     3. 小 case 固定开销主导 → 减少每批同步/表读取，或改用更轻量的 AFFINE/PREFIX 路径。
   - 不要因为“已经实现了 MACRO_WINDOW”就把所有短 fragment 都塞进去；路由选择应以实测性能为准。

12. **MACRO_WINDOW 遇到 UB OOB 时不能整体禁用并退回 per-run CURSOR_BATCH（逐 run 游标批处理：每个输出 run 各发一次 MTE2 描述符的保守路径），否则高 rank 短 fragment 会数量级退化**
   - 原理：per-run CURSOR_BATCH 每个输出 run 仍产生一个 MTE2 描述符，run 数可达几十万甚至上百万；
     即使 MTE3/同步做了 batch，MTE2 issue 开销仍然主导，微秒级参考 case 会被拖到数百微秒。
   - 错误做法：MACRO_WINDOW 集成时出现 UB out-of-bounds，就把 `routeIndex = MACRO_WINDOW` 用
     `if (false)` 关掉，让所有高 rank 短 fragment 落到 CURSOR_BATCH。这会从 geomean ~1.0x 直接掉到 0.4x。
   - 正确做法：UB OOB 通常来自以下位置，必须逐一定位并修复：
     1. offset 表长度 / Gather count 超过 output 或 input buffer 容量；
     2. input/output/table 的 UB 分配没有按 batchRows 和 depth 计入总容量；
     3. `ReinterpretCast<uint16_t>()` 后 count 或表单位不匹配；
     4. 源游标跨 prefix carry 时读取越过最后一个有效行；
     5. TQue/LocalMemAllocator 混用导致缓冲区重叠。
   - 只有在完成上述排查并记录具体不可满足的红线后，才允许临时禁用；禁用时必须同时提供可达到相近性能的
     替代路径，不能退回 per-run 慢路径。

##### 5.11.6.3.2 MACRO_GATHER 高性能实现规格（流水 / 候选选择 / direct 批行）

> 这里把 MACRO_WINDOW 中“建 Gather 表压缩”的实现简称为 **MACRO_GATHER**，用于区别于同属 MACRO_WINDOW 的 direct Padded DMA 子路径（见 §5.9）。
> 同样功能正确的 MACRO_GATHER，如果 buffer、同步、候选选择实现不好，性能可能差一个数量级。
> 下面给出高性能实现必须满足的结构性要求。

1. **必须使用队列化流水，禁止单缓冲 + 每批全量 barrier**
   - 原理：Gather 路径需要 MTE2→V→MTE3 三段流水；单缓冲会把三段完全串行，并且每批
     `PipeBarrier<PIPE_ALL>` 会阻塞所有流水线。
   - 推荐结构：
     - `in_`：`TQue<VECIN, 1>`，大小 `windowBatchRows * Align32(windowSpanBytes)`
     - `out_`：`TQue<VECOUT, 2>`，大小 `windowBatchRows * runBytes`
     - `index_`：`TBuf<VECCALC>`，大小 `windowBatchRows * runBytes * 2`
   - 每批使用 `AllocTensor → EnQue → DeQue → Gather → EnQue → DeQue → DataCopyPad`，
     依赖由 TQue 建立，不要每批 `SetFlag/WaitFlag` + `PipeBarrier<PIPE_ALL>`。
   - `out_` 使用 depth 2，使 MTE3 写回与下一批 MTE2/Vector 重叠。

2. **Offset 表只读一次，并让第一次队列依赖覆盖表加载**
   - 表在循环前用一次 `DataCopy/DataCopyPad` 读入；
   - 不要每批重复读表；
   - 读表后不必单独等待，第一次 `in_.EnQue/DeQue` 会建立依赖。

3. **CopyIn 使用源游标，避免每段重算地址**
   - 使用 `InitSourceCursor` / `AdvanceSourceCursor` 维护当前宏行的 prefix 坐标与源地址；
   - 在一个 prefix span 内用 `DataCopyExtParams{spanRows, windowSpanBytes, gap, 0, 0}` 批量读取；
   - Gather 路径的输入 padding 不是必须的：`DataCopyPadExtParams` 可用 `isPad=false`；
     只有 direct Padded DMA 才需要靠右 padding 来剥掉行 padding。

4. **候选选择不能只看 span/bytes，要优先减少总宏行数**
   - 原理：MACRO_GATHER 的总成本 ≈ 批次数 × 每批固定开销 + DMA 流量；
     选择更大的后缀虽然 `span/bytes` 读放大略高，但会减少 runCount、增大 runBytes，通常能降低总批次数。
   - 推荐评分顺序：
     1. 估算总批次数 `ceil(runCount / batchRows)` 尽量小；
     2. `runBytes` 尽量大（减少每行固定开销）；
     3. `span/bytes` 读放大尽量小；
     4. `innerExtent` 大小只作为次级偏好。
   - 不要仅因为某个候选 `span/bytes` 更小就选择它；如果它导致 runCount 翻倍、runBytes 减半，通常是坏选择。

5. **direct Padded DMA 的 batchRows 要按输入槽大小计算，不要把 output/table 混入同一 per-row 预算**
   - direct Padded DMA 使用 `TQueBind<VECIN,VECOUT,D>`，输入和输出共享同一 slot；
   - 每个 slot 的字节数按 `Align32(runBytes)` 计算；
   - 推荐：`rows = min(4095, batchBudget / Align32(runBytes)) / group * group`（`batchBudget` 为输入槽字节预算：910B3 经验值 80 KiB，950PR 未重测），
     再用 depth 2 检查 `2 * rows * Align32(runBytes) <= UB 预算`；
   - 如果把 `Align32(runBytes) + runBytes + 2*runBytes` 一起算进 per-row UB，会得到过小的 batchRows，
     导致批次数和同步次数成倍增加。

6. **batchRows 要按每核 ownership group 裁剪**
   - `maxCoreRows = ceil(ownershipGroups / usedCoreNum) * ownershipGroupRuns`；
   - `windowBatchRows = min(windowBatchRows, maxCoreRows)`；
   - 避免某些核 batch 过大、某些核空闲，提升多核负载均衡。

##### 5.11.6.4 PERIODIC 的 GatherMask 专用参数

仅命中下面这一种精确几何：5D、prefixRank=4、runBytes=64B、每行源 pitch=256B、plane pitch=8192B、每 plane 16 行、910B3 的 40 核（950PR 上核数须用 `GetCoreNumAiv()` 重新标定，见 §5.7）。

```cpp
// mask：每个 repeat 取低 16 个元素
mask64.SetValue(0, 0x000000000000ffffULL);
mask64.SetValue(1, 0ULL);

GatherMask(output.ReinterpretCast<uint32_t>(),
           input.ReinterpretCast<uint32_t>(),
           mask64.ReinterpretCast<uint32_t>(),
           false, 0U,
           GatherMaskParams(1U, localPlanes * 16U, 8U, 0U),
           retained);
```

##### 5.11.6.5 入口与编译形态

- 所有 kernel 入口：`extern "C" __global__ __aicore__ void ...` + `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);`
- 每个 route 独立一个 `.cpp` + `.h`，不在一个文件里塞多个 `TPipe` 配置。
- 不要用 `TBuf<VECCALC>` 作为 GM↔UB 的搬运缓冲；优先 `TQue<VECIN/VECOUT>` / `TQueBind` / `LocalMemAllocator<Hardware::UB>`。对齐单缓冲路径允许用 `TBuf<VECOUT>` + 精确事件（见 §5.3、§7.2）。
- 本算子不需要 AIC/Cube/mix kernel；不要为“看起来复杂”的搬运引入 mix kernel。

---

### 5.12 dav-3510 专属搬运与 gather 语义（Ascend 950PR 追加证据）

> **平台前提**：本节机制只在 Ascend 950PR（DAV_3510 / arch35，CANN 9.1.0，UB 248 KiB，torch_npu 2.7.1.post10）可用；
> Ascend 910B3（DAV_2201）没有这些接口。本节是**并列追加**——未启用时按 §5.1–§5.11 的 910B3 路径执行，原路径保持可达。
> 每条结论标证据等级：**A** = 真机复现（可重跑）；**B** = 头文件/随包源码可查；**C** = 调研或推断（含反证条件）。
> 本节所有结论**不外推**到其它用例集/设备/算子（本 lab 为单一固定 61 例用例集、固定参考）。

#### 5.12.1 `DataCopyPad<Compact>` 的 UB 侧行距（A）

- 多 block 搬运时，`PaddingMode::Compact` 下 **UB 侧行距 = `blockLen`，不向上 32B 对齐**；GM→UB 与 UB→GM **两个方向都成立**（A：span 非 32B 倍数时第 r 行整体偏移 `r*(pitch-span)`，读侧错 4 个元素、写侧错 6 个元素，逐字节探针）。
- 推论：`Compact` 只能表达「紧密排布」；**需要 32B 对齐 UB 行距时必须显式用 Loop Mode**（§5.12.2），不能用 stride 参数凑。反向同样成立：910B3/`Normal` 的 `align32(blockLen)` 假设在 `Compact` 下不成立，两条路径的单位语义不能混用。

#### 5.12.2 `SetLoopModePara` 的 stride 语义（A）

| 项 | 结论 |
|---|---|
| 结构 | `LoopModeParams{loop1Size, loop2Size, loop1SrcStride, loop1DstStride, loop2SrcStride, loop2DstStride}`（B） |
| 单位 | **字节**；`loop*Size < 2^21`（B：官方文档明确「单位为字节」「UB 侧必须 32 字节对齐」；pitch 读法由 A 级探针确定——官方原文用「间隔」一词，不要按 gap 填） |
| 支持范围 | **仅 Ascend 950PR & 950DT**；A3 / A2（910B）/ 310B / 310P / 910 均不支持（B：产品支持矩阵 + 头文件 `#if __NPU_ARCH__ == 3510`） |
| 语义 | `loop1DstStride`（`OUT_TO_UB`）/ `loop1SrcStride`（`UB_TO_OUT`）是**每层迭代的跨步（pitch，不是 gap）**；`loop2*` 同理作用于外层（A，探针逐字节确认） |
| 对齐 | **UB 侧** stride 必须 **32B 对齐**且 < 2^21；GM 侧 stride 40 bit 任意（A/B） |
| 方向 | `OUT_TO_UB` 与 `UB_TO_OUT` 都支持；3510 上普通 `DataCopy` 与 `DataCopyPad` 走同一 `*_align_v2` 路径，LoopMode 对两者生效（A/B） |
| 层数 | `DataCopyExtParams`（blockLen × blockCount）+ LoopMode（loop1 内 × loop2 外）= 最多 **4 层硬件循环**，更外层落软件循环（A） |
| 纪律 | SPR 全局状态，**必须成对 `Set`/`Reset`**，否则污染后续所有搬运（A/C） |

**反例（C）**：只把 stride 用 LoopMode 表达出来并不会变快——未合并连续维时比基线慢 2–5×。收益来自「把可合并的维压进硬件循环 + 消除逐块地址计算」，不是「描述得更细」。官方 `strided_slice` key 300 路径（最内层源步长 ∈ (1,16]、末维超集 span ≥ cacheline 256 B、输出 > 1 KB）即此用法。

#### 5.12.3 RegBase `Gather` 的索引越界语义（A，**更正既有口径**）

- **索引寄存器恒覆盖整个 VL**，**被 mask 掉的 lane 同样会形成地址**。因此「被 mask 的 lane 不会产生越界」这一既有口径，在 CANN 9.1.0 / dav-3510 上**不成立**（至少对超出缓冲的地址）。
- 实测后果：最后一个 tile 的最后一行越出 tile buffer（约 1.2 KiB）→ **vector core exception 507035**（plog 定位到 gather 内核）。
- **缓冲分配公式（硬约束）**：

```text
bufferBytes >= (rows - 1) * pitch + max(pitch, reach)
reach = (chunks - 1) * VL * gap + (VL - 1) * gap + elemSize + 32
VL = 256 B（fp32 64 lane / fp16 128 / int8 256）
```

- 该条**更正**了早期「mask 未筛选 lane 不越界」的结论（原结论为 B 级调研，现已由真机 A 级证据推翻）。若按 §5.11.6 的 Gather 表公式分配缓冲，必须同时满足上式，否则异常。
- RegBase 其余硬约束：`VL = 256 B`（fp32 64 lane / fp16 128 / int8 256）、`RegTensor ≤ 32`、`MaskReg ≤ 8`（超出落预留 UB 并劣化）、`UpdateMask` 为 post-decrement。

#### 5.12.4 Gather 类搬运的公共经验（按块表/索引选行，A）

> 来源算子语义：按 `block_tables` + `seq_lens` 从 cache 中选取若干行/段写出（不是 Slice，但同属「按索引做搬运算子」）。
> 与 Slice 的差异：索引来自张量（Host 不可预知的运行值），因此**不能**像 COMPOSITE 那样把 offset 表在 Host 预算后旁挂。

1. **UB 预算必须运行时推导**：`ubBytes = GetRuntimeUBSize()` 类运行时取值；`slotBytes`（UB 槽）是「合并快路径 / 有界分块路径」的唯一开关，不依赖具体用例。不要写死 `192 * 1024`。
2. **两种搬运形态**：`Norm` 模式是连续字节流（单段大 DMA）；`PA_NZ` 是「块表 × 列维」的 2D 列维搬运。
   - `rowBytes ≤ slot` → **合并快路径**：`tileRows > 1`，一次描述符搬多行；
   - `rowBytes > slot` → **有界分块路径**：`tileRows = 1`，按 `cells` 分块，`chunkBytes = min(64 KiB, slot)`。
   - 两条路径在 `cells == nzBlocks` 时逐指令等价（分块路径的自然退化），保留热路径特化是刻意的。
3. **越界位置必须显式写零（位模式）**：`column ≥ tableCols` 或 `blk ∉ [0, numBlocks)` 时整行写 0 的**位模式**；不要在设备端做标量类型转换。写零也要分片（`piece = floor32(transferBytes)`），避免超长单描述符。
4. **`tileRows` 由 slot 反推**，不由 batch 数决定；`cells` 上限要先按 `nzBlocks`、再按 `65535` 截断（后者在 `rowBytes/32 ≤ 4736` 且 `slot ≤ 64 KiB` 时恒不触发，属防御）。
5. **效果口径**：该实现（含上述路径）在固定 38 例上 `precision_consistency` 38/38 逐字节通过，quick geomean **2.48–2.50×**（相对 CANN `aclnnGatherPaKvCache`，单会话、`warmup=1/repeats=1`）。收益主体在宽行分块（曾崩溃的 148 KiB 行现 2.18×），**合并路径的机理未隔离**（无「去掉合并」对照），不得把整体 2.5× 全部归因给合并。
6. **队列类型即同步契约（A）**：整块 GM→UB→GM 往返用一个 `TQueBind<VECIN,VECOUT,2>` 即可（`EnQue/DeQue` 自动插事件）；一旦在 UB 内做向量处理，**必须**改用 canonical `TQue<VECIN,2>`（MTE2→V）与 `TQue<VECOUT,2>`（V→MTE3）。把 `TQueBind` 当纯输入喂给向量读会导致 AIV bufId 同步错位 → 多 tile 时**确定性错行**（每次运行错行集合不同）。
7. **UB 报值口径要留痕（A/C）**：实测探针 `GetRuntimeUBSize()` = 253952 B（248 KiB）；另有文档按 524288 做算术但**无归档探针支持**。两者会改变 `slot`（81920 vs 131072）与快路径阈值，因此 host 首次 launch 前应把报值打印一次/写入 design，避免事后无法机读复核。
8. **报值 ≠ 硬件上限（A/C）**：曾有实现按 `4×rowBytes` 名义分配，实测 `rowBytes=131072` 通过、`131104` 起触发 `507035`（`ACL_ERROR_RT_VECTOR_CORE_EXCEPTION`），148 KiB 必挂 ⇒ UB 槽**只能向下推导 + 有界分块**，不得「把槽上抬到整行」。报值 248 KiB 与实际失败临界的关系有两种待分清假说（在用足迹 ≈256 KiB / 单描述符 `blockLen` 上界 2^17 B），**未证实**。

#### 5.12.5 防御分支 / 死路径的处置（A/B，跨子场景可复用）

1. **「本次用例没触发」≠「不可达」**：给防御分支写 reachability 结论时，必须给「调用点不变式 + 判据恒真/恒假的代数推导 + 域枚举反例数」三件套。把前者写成后者，会让后续读者既无法判断能否删除，也会把「未覆盖」误当「已证明」。
2. **可删的死 clamp** 只依赖常量与已在位的不变量（如 `slotBytes < 32 → 32`、`chunkBytes < 32 → 32`、`cells == 0 → 1`、`cells > 65535` 截断）。删除后语义完全不变，但**保留**的兜底仍有独立价值。
3. **静默丢活是反模式**：`b >= batch → break` 这类「不写输出就退出」的分支，在合法输入下依赖 host 未强制的不变量（`seq_lens` 条目数 == `batch`）；一旦 host 推导与设备侧不一致，剩余输出行会停留在 `at::empty` 的**未初始化内存**里，既不报错也不是零。Host 侧应补 `TORCH_CHECK(length_count == batch (+1))`。
4. **运行时 UB 报值要留痕**：路径分类依赖 `GetRuntimeUBSize()` 的报值，若不入痕，事后无法机读复核「为什么这个 case 走了快/慢路径」。

#### 5.12.6 950PR 路由准入的写法：相对可表达性（A）

- **判据必须比较「自身 P/N/G 路径对该 nest 的可表达性」，而不是绝对质量阈值**。反例：用绝对阈值（`runSpan < 32B` ∧ 源跨度 ≥ 4×输出）筛 SIMT 时，命中的 14 例整体 0.849 < 未命中 47 例 1.036，净 −1.18%；在参考也走 SIMT 的 8 例上 +12.2%，但多抓的 6 例（自身 DMA 本可表达）−24.8%。改成相对可表达性判据后命中 8 例，整体约 +1.55%。
- **命中集 ≠ 收益集**：任何新路径上线前先做**同一 run 内 on/off 配对**；不能凭「路由与参考一致」推断性能相当（参考侧 tiling key 只作「方向同族」旁证，且其中 3 条分支边界是拟合值）。
- **收益可能集中在单条用例**：G 路径相对基线约 +0.69% 的增益几乎全部来自 1 条用例（去掉该例后 +0.806%，低于 1% 聚合门槛）⇒ 报告必须给「去掉 top-1/top-3 后的余量」，不得写成可泛化分支。
- **tile 不是越大越好**：950PR 上 tile 64→112 KiB 把 work item 数砍约 43%，但在本 61 例上**无可测时间收益**（pad 路径 1.01387 → 1.01255，反而慢 0.23%，落在逐例 log 比 RMS 1.6%–2.3% 内）⇒ 大 tile 需要干净对照证明，不能默认成立。
- **会话可区分门槛**：本 lab 为「逐例 ≳1.1%、聚合 ≳1%」；低于此的候选间差不可判（同产物单批批间 `SD(log)≈0.28%`）。

---

## 6. tile 与开核：为什么这样联动

### 6.1 为什么 tile 要“先定路由，再回填”

- 路由判定需要 tile 参数，但路由决策应该稳定；直接用 UB 满容量 tile 会影响路由的命中。
- 因此采用两阶段：先用一组保守的旧 tile 完成 route 和 bound 队列选择；route 确定后，再按 `UB 容量 - 固定开销` 回填该 route 能用的最大 tile。
- 效果：既不改变路由命中，又能让每条路由尽量吃满 UB，减少每轮循环次数。

### 6.2 为什么动态 tile 优于固定分档

- 早期固定 48K/80K/96K 分档，出现过“大 affine 变好、大 aligned 变差”的互斥现象：同一 tile 对不同路由不是同时最优。
- 动态回填后，bound 路径按 `UB / depth`、普通队列按 `(UB - 16K) / 2` 计算，tile 自动适配路由类型和 buffer 结构。

### 6.3 为什么开核数不能由 `totalBytes / tileBytes` 决定

- `tileBytes` 是单核批处理容量，不是全局调度量子。若按总字节除以 tile 二次压缩核数，复合/prefix 类多组几何会从 40 核塌缩到 1–12 核，出现数倍回退。
- 正确做法：开核数 = `min(可用核数, 可调度组数)`，其中可调度组来自 ownership group（必要时按连续 tile 或列分片展开）。
- 减核只能做成 route-specific 的窄门控：例如“中等 aligned affine 且每组只有 1 run 且核数远超组数”时，才按每核约 32 KiB 有效输出减核。全局按字节减核已被反证。

---

## 7. UB / 同步 / 写所有权：三条公共经验

### 7.1 写所有权先于性能

- 非 32B 输出短 run 不能让多个核或多次写共享一个物理 32B 块。
- 先在 UB 内把短 run 拼成 32B 块，再由单一 owner 单次 MTE3 写回。
- 所有批量路径的核边界都必须是完整 ownership group；只有全局最后一个组允许不完整。

### 7.2 同步方式按路径选，不搞全量替换

- 单 TBuf 的简单对齐路径：`SetFlag/WaitFlag` 精确事件即可，开销低。
- 多 stage / 需要跨队列保护的路径：保留 `PipeBarrier`，因为粗 barrier 换掉后某些路径会因依赖错位而回退。
- TQueBind 双槽只给大 tile、depth=2 且 UB 峰值核算通过的路径，用于 MTE2/MTE3 重叠。
- **950PR 追加项（按平台标注）**：`PaddingMode::Compact` 改变了「UB 行距是否需要 32B 对齐」的判断（§5.12.1），Loop Mode 改变的是**跨迭代行距**而非单次描述符（§5.12.2）；二者都不引入额外的 MTE2→V→MTE3 事件需求，但 Loop Mode 必须成对 `Set`/`Reset`，否则污染后续搬运。这不改变「精确事件不是越多越好」的结论。

### 7.3 Host 与 Kernel 的镜像一致性

- `routeIndex` 和 `useBoundQueue` 是 Host/Kernel 的单一事实源。
- Kernel 不重新计算 tiling-only 谓词；否则两端谓词迟早漂移，出现“Host 选 A 路径、Kernel 按 B 路径初始化 buffer”的错误。
- 窗口类路由会重写几何（`prefixRank/runBytes/runCount/ownershipGroupRuns`），Kernel 必须直接消费重写后的值。

---

## 8. 设计决策与反馈经验表

> 下表把关键决策整理为“问题 → 设计 → 为什么 → 效果/结论”。证据强度：**采纳**=正反馈且落地；**反证**=有效测量否定；**局部**=只对特定场景有效。

| 场景 / 问题 | 设计决策 | 为什么 | 实现要点 | 效果 / 结论 |
|---|---|---|---|---|
| 大量等长、恒定 pitch 的短行 | 单前缀仿射行批量 DMA | 逐 run 的 DMA/队列/同步固定开销远超数据量 | `DataCopyPad(blockCount)` 批量读，多 block 写回自动去 padding | 最短行 case 从数百 μs 降到几 μs；**采纳** |
| 多 prefix 批量优化想覆盖 rank-one | 先保护 rank-one，再处理多 prefix | 泛化路径会吞掉已验证最优路径，出现约 70 倍回退 | 决策树先单前缀仿射行，再多前缀 span | **反证**：rank-one 必须独立且优先 |
| 多 prefix、inner extent 可整除 | 前缀 span 批量 | batch 不跨 wrap，无需 Gather | 每批限制在 innermost span 内；aligned 用 DataCopy，非对齐用 DataCopyPad | 规则多 prefix 场景有效；**采纳** |
| 多 prefix、非 32B、组跨 wrap 且组 ≤ 8K | 复合组：padded DMA + Gather + 单次 MTE3 | 组内短 run 共享 32B 输出块，必须单 owner 单次写 | Host 预计算 offset 表，kernel 单次 DataCopy 读表 | 复合 int64 case 相对参考的加速比提升 30%–95%；**采纳** |
| 2D 固定内步长导致 runCount 巨大 | 仿射折叠 | 折叠后 runCount 从 n×m 降到 n | Host 折叠 + 折叠失败完整回滚；kernel 每元素独立 32B slot + Gather | 对 2D/4B 元素有效；**局部采纳** |
| 大而 32B 对齐的单 run 铺不满核 | `(group, columnTile)` 列分片 | 单 run 单核时多数核闲置；对齐切块保证写区间不重叠 | runBytes > tileBytes 时展开列分片网格 | 单大 run 达到参考 1.30×；**采纳** |
| 完整周期窗口（每行固定窗口） | 严格门禁 + 单次 2D MTE2 + GatherMask | 逐 plane 的 MTE/barrier 是主要开销；少核不够并行 | 每核 6–7 plane，单次 MTE2/GatherMask/MTE3 | 目标 case 10.4→5.8 μs；20 核对照慢 10%，40 核写死；**采纳** |
| 高 rank 短 fragment 逐 fragment 处理 | 宏窗口 + 批量 Gather | 复用一张表、批量搬运，减少 issue/同步；窗口必须密度有界 | `span ≤ 8×有效字节`、原始物理连续后缀 `runBytes ≤ 256B`、Host 预计算 offset 表 | 多个碎片 case 自身耗时改善 3–18 倍；**采纳** |
| 非对齐长 run（64B..4096B）跨 prefix | 直接 Padded DMA，不建 Gather 表 | 行内无选元素操作，唯一问题是 padding；DataCopyPad 输出端可自动剥 padding | 双槽 TQueBind + 输入 padded 行 + 输出多 block DataCopyPad | 最差 case 从约 8800 μs 降到 18 μs；**采纳** |
| 所有特化都不匹配 | packed fallback | 正确性必须永远可达 | 32B scratch 拼块，单 owner 写回 | 最终兜底；目标是把常见几何尽量不落到这里 |
| tile 用固定 48K/80K 分档 | 动态 tile 回填 | 固定分档对 aligned/affine 不同时最优，出现互斥退化 | 先旧 tile 定 route，再按 UB 容量和队列类型回填 | 消除大 aligned 退化；**采纳** |
| 开核按总字节 / tile 二次压缩 | 开核按 ownership group 派生 | tile 是批容量不是调度量子；二次压缩会把多组几何从 40 核压到 1–12 核 | 不按 `totalBytes/tileBytes` 重算 usedCoreNum | **反证**：曾因构建证据不一致撤回，且会导致数倍回退 |
| 全局按字节限核（≤4K 单核、4K/core） | 不采用 | 同 shape 不同 slice 行为不一致，误伤 scalar-run 与高维 | 只允许 route-specific 的 32 KiB/core 减核 | **反证** |
| Tiny 输入跳过队列/路由的 Fast Path | 不采用 | tiny 固定开销不在队列初始化，跳过无用 | 所有输入正常走 route | **反证**：唯一命中 case 反而更慢 |
| 所有路径 PIPE_ALL 改精确事件 | 只在窄 aligned 路径用事件 | 某些路径（尤其 aligned runs）插入事件后回退约 2 倍 | aligned 单 TBuf 用 SetFlag/WaitFlag；其余保留 PipeBarrier/TQue | **反证后局部吸收** |
| Host 成本模型做严格准入 | 不采用 | 未校准的 proxy margin 与实测不符 | 不使用成本模型 gate | **反证** |
| 小尾窗整窗搬入 + Gather slice | 不采用 | 准入 case 实测远慢于参考，未触发分支的 sibling 反而快 | 不建子 32B 小尾窗特化 | **反证** |
| device 端标量构造 offset 表 | Host 预计算 offset 表 | device scalar SetValue 双循环很慢 | Host 生成 + 旁挂 TilingData + 单次 DataCopy | 复合类 case 大幅改善；**采纳** |
| 单一大 kernel 内嵌全路由 | 每 route 独立 kernel symbol | 各 route buffer/同步不同，独立实现最简、不互相干扰 | 9 个独立 kernel 入口 + 统一 routeIndex 分发 | **采纳** |

---

## 9. 开发检查表

### 9.1 先证明需要路由

- [ ] 支持域内是否同时存在连续大块、等长短行、跨 wrap、不规则碎片等至少两类几何？
- [ ] 是否找到至少一个“错误路径会慢 10 倍以上”的代表 case？没有这样的 case，分路由价值有限。

### 9.2 特征与决策树

- [ ] 是否从 shape/stride/dtype 派生了 `prefixRank / runBytes / runCount / ownershipGroupRuns`？
- [ ] 每个分支是否有稳定 ID、准入谓词、优先级、fallback？
- [ ] Host 是否按本文完整优先级选择 route，而不是只根据第一个 descriptor 粗略命名？
- [ ] Kernel 是否真实消费 `routeIndex` 并执行对应 route？仅写入 TilingData 不算完成。
- [ ] 谓词是否互斥或有明确优先级？支持域内是否无空洞？
- [ ] 最终 fallback 是否永远可达？
- [ ] `blockCount=1` 是否只出现在明确的 fallback/保护路径？是否记录了硬件证据和性能代价？
- [ ] 是否确认常见短 run、多 prefix、跨 wrap、宏窗口和非对齐长 run 没有被 fallback 吞掉？
- [ ] 是否保护了已验证的快路径不被新的更宽分支抢占？

### 9.3 每个分支实现前先回答

- [ ] 这条分支解决什么问题？为什么需要它？
- [ ] 命中后最坏会慢多少？读放大/冗余拷贝是否有上界？
- [ ] 写回是否满足物理 32B 单 owner？核边界是否落在完整 ownership group？
- [ ] 同步方式是否与该路径的数据流匹配？
- [ ] 该分支的 tile 和核数是否与该路径的实际工作单元联动？

### 9.4 验证

- [ ] 每个分支至少 2 个命中 case、2 个边界 case（阈值两侧）、2 个保护 case（应继续走旧路径）。
- [ ] 性能记录必须能确认实际 route ID，不能只凭 shape 猜。
- [ ] 性能记录必须同时包含每个 case 的 route、descriptor 数、平均 `blockCount`、Gather/窗口是否启用；否则无法证明核心经验真正生效。
- [ ] 性能结论必须来自同 device 同 session 的 paired 测量；失败/部分 session 不进入统计。
- [ ] 对同 shape 不同 slice 的情况也要测（曾出现一个快一个慢的陷阱）。
- [ ] 若性能相对参考明显退化、常见 case 大量命中 fallback 或 route 未真实执行，必须回到本文逐条检查核心经验是否被实现；不能以“精度已通过”作为结束条件。

---

## 10. 不能泛化的结论

- 固定 tile 字节数不是通用最优值；只有“按 UB 容量和 route 类型动态回填”的方法可复用。
- “输出小就该单核”“每核至少 N KB”等按字节限核规则已被反证，只能做 route-specific 减核。
- 跨 session 的微小 geomean 变化不能作为优化有效的证据。
- 精确事件同步不是越多越好；只有简单对齐单 buffer 路径才稳。
- 本经验基于正 stride、rank ≤ 8、2/4/8 字节元素的 Slice；其他算子族应重新恢复自己的路由空间。
- **平台更正（950PR / CANN 9.1.0）**：「RegBase Gather 被 mask 的 lane 不产生越界」这一早期调研结论**已被推翻**（A 级真机复现，vector core exception 507035）；缓冲必须按 §5.12.3 的 `(rows-1)*pitch + max(pitch, reach)` 分配。
- §5.12 的 dav-3510 机制结论（`DataCopyPad<Compact>` 行距、Loop Mode pitch 语义、RegBase Gather 越界、SIMT 与 AscendC 框架互斥、Gather 类搬运的运行时 UB 预算）**只覆盖实测的单一用例集与设备**，不外推到其它几何/dtype/平台；其中多项只有单条正例或探针级证据，标 C 的部分不得当作设计硬约束。
