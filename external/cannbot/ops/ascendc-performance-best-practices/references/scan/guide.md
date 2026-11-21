# Scan / Cum 前缀扫描类算子高性能实现索引

> 适用架构：arch35（A5 / DAV_3510）；本文 RegBase / Reg / MicroAPI / VF 内容为 arch35 独占。

Scan（前缀扫描）族算子——Cumsum、Cummin、Cummax、Cumprod、LogCumsumExp 等——沿某一轴做**含位置依赖的累积运算**（prefix scan）。与 Reduce 族"沿 R 求单个统计量"不同，Scan 族要求**输出每个位置的前缀结果**（`y[i] = op(x[0..i])`），因此不能简单折叠成一个标量，核心难点在于**如何把串行前缀依赖并行化**。本指南沉淀 Cumsum 在 Ascend Vector 核心上的高性能实现，提炼为一组**可复用 Kernel 模板**，并说明如何适配到同族其他算子（含带索引的 Cummin/Cummax/ArgCum）。

> 本指南是**优秀实践目录**，而非选型决策树：每个模板列出适用条件与解决的问题，由实现者按实际 shape、UB 容量、并行度权衡采用。Scan 族形态多样（大 R 串行 / 小 R 多行 / 窄 N 折叠 / 1D 超长），不存在普适"唯一最优"，理解各模板权衡是正确适配的前提。

## 文档结构

```
guide.md（本文件）
└── templates/
    ├── usage_guide.md                          模板集成指南（.template → .h、入口写法、常见编译问题、模板已知问题）
    ├── scan_tiling_data.template               packed TilingData 公共定义（→ .h）
    ├── scan_tiling.template                    Host tiling 选型参考实现（→ .h）
    ├── dav310/
    │   ├── scan_base.template                  Device 侧公共工具（CastTrait / 促进类型 / CeilDiv/Log2 / GetCacheID）（→ .h）
    │   ├── cum_tile_resident_scan.template     tile 常驻扫描 + ping-pong 双缓冲（N>1 / dim 非末维主力）（→ .h）
    │   ├── cum_row_kogge_stone.template        行内 Kogge-Stone + 行批处理 + ping-pong 双缓冲（N==1 / dim=-1 起步版，含索引）（→ .h）
    │   ├── cum_lane_parallel_scan.template     lane 并行顺序扫描 + 两行交错（算力受限时替掉 Kogge-Stone）（→ .h）
    │   ├── cum_oneway_sklansky.template        One-way Sklansky UB 内并行前缀扫描（→ .h）
    │   ├── cum_streaming_scan.template         流式前缀扫描（参考实现，非性能路径）（→ .h）
    │   └── cum_core_partition_scan.template    跨核 R 分段扫描 + SyncAll + 向量化 ReadCarry（1D 超长 R / 低并行度）（→ .h）
    ├── cum_tile_resident_scan.md               tile 常驻扫描 + 双缓冲说明文档
    ├── cum_row_kogge_stone.md                  行内 Kogge-Stone 说明文档
    ├── cum_lane_parallel_scan.md               lane 并行顺序扫描说明文档
    ├── cum_oneway_sklansky.md                  One-way Sklansky 说明文档
    ├── cum_streaming_scan.md                   流式扫描（含索引）说明文档
    └── cum_core_partition_scan.md              跨核 R 分段扫描说明文档
```

**阅读顺序**：先读本指南「通用范式」「三层组合骨架」理解结构 → 读 [usage_guide.md §1.1 选型](templates/usage_guide.md) 定路线 → 读对应模板 `.md` 了解流程与关键技术 → 落代码时对照同名 `.template`（去掉 `.template` 后缀转成 `.h`）。

按形状直接跳转：

| 融合成 `(M, R, N)` 后 | 读这个 | 瓶颈在哪 |
|---|---|---|
| `N > 1`（dim 非最后一维） | [cum_tile_resident_scan.md](templates/cum_tile_resident_scan.md) | 搬运粒度与流水重叠 |
| `N == 1`（dim=-1，扫描轴在 GM 上连续） | [cum_row_kogge_stone.md](templates/cum_row_kogge_stone.md) | 先摆平搬运 |
| 搬运已摆平、瓶颈变成向量指令条数 | [cum_lane_parallel_scan.md](templates/cum_lane_parallel_scan.md) | `log2(R)` 的放大 |
| `M*N` 远小于核数且 R 很长（典型 1D） | [cum_core_partition_scan.md](templates/cum_core_partition_scan.md) | 并行度不足 |
| Cumsum/Cumprod 无索引、长单行 R | [cum_oneway_sklansky.md](templates/cum_oneway_sklansky.md) | — |
| 只想先跑通、不追性能 | [cum_streaming_scan.md](templates/cum_streaming_scan.md) | 参考实现，非性能路径 |

> **两条主线的类型选型结论相反，不要互相套用**：`N > 1` 的 tile 常驻路访存连续，
> 16 位可以边搬边 Cast；`N == 1` 的 lane 并行路走 gather，16 位**必须**先升 fp32
> （16 位 gather 没有全速路径，实测慢 1.6~1.9 倍）。详见各自的 `.md`。

> ⚠️ **集成前必读**：[usage_guide.md](templates/usage_guide.md)（含选型实测轨迹、模板已知问题）。

## Scan 计算的通用范式

Scan 族算子均可抽象为**沿扫描轴 R 的结合律二元运算 `combine`（带单位元 identity）**，输出每个前缀位置的累积结果：

```
y[.., i, ..] = x[.., 0, ..] ⊕ x[.., 1, ..] ⊕ ... ⊕ x[.., i, ..]

算子          combine ⊕         identity      附加输出
Cumsum        加法 (+)          0             —
Cumprod       乘法 (×)          1             —
Cummax        max              -inf          argmax 索引（并列取更晚位置，对齐 torch）
Cummin        min              +inf          argmin 索引（并列取更晚位置，对齐 torch）
LogCumsumExp  logaddexp        -inf          —（需在线 max 稳定化）
```

因此 **Cumsum 的骨架（切分 + 流水 + 前缀合并的组织方式）对 Scan 族通用**，适配只需替换 `combine`/`identity`，并在带索引算子上追加**索引选择**（见「适配同族算子」）。

### 与 Reduce 的关键区别

| 维度 | Reduce（Softmax/Norm/EuclideanNorm） | **Scan（Cumsum/Cummin/…）** |
|------|--------------------------------------|-----------------------------|
| 沿 R 输出 | 1 个统计量（标量/向量） | **每个前缀位置 1 个值**（输出与输入同形） |
| R 累加顺序 | 可交换（可任意分核合并） | **有序**（前缀依赖，跨核/跨块合并必须保序） |
| 并行难点 | 分核求部分和再合并 | **前缀依赖的并行化**（Sklansky 对数深度 / 分段扫描 + 保序进位） |
| 跨核合并 | 部分和相加（顺序无关） | **进位（carry）必须按 R 顺序传播** |

> Reduce 的 group/BlockSplitR 因"可交换"可随意分核合并部分和；Scan 的跨核分段则必须**保序传播进位**（前段的累积结果作为后段的初值），这是 Scan 特有的复杂度。

## 三层组合骨架（Cumsum 提取的核心设计）

Cumsum 最可复用的不是某个单一 kernel，而是**三层可组合的前缀扫描结构**——每层解决"数据规模超出上一层容量"时的进位传播：

```
InnerScan<combine>            —— 向量寄存器/UB 内的前缀扫描（Sklansky 对数深度 或 流式串行）
    ⟲ 被 UbChunkScan 包裹     —— R 超出单次 UB，切成多个 chunk 串行，chunk 间用对数进位 memo
        ⟲ 被 CorePartitionScan 包裹  —— R 跨核分段，各核算局部前缀，跨核保序传播进位
```

对应到 Cumsum 的 8 个浮点模板（`inner ∈ {oneway, twoway}` × `是否 UbChunk` × `是否 CorePartition`）：

| tiling key | 模板 | 层级 | 触发 |
|-----------|------|------|------|
| 1001 | oneway_ss | Inner | R 全量入 UB，切 M（或借 N）够满核 |
| 1002 | twoway_ss | Inner（折叠） | 窄 N（`alignN ≤ VREG/4`）折叠多个 R 位置进向量道 |
| 1011 | ub_ss_oneway_ss | Inner+UbChunk | R 超 UB，切 M 够满核，单核串行多 chunk |
| 1012 | ub_ss_twoway_ss | Inner(折叠)+UbChunk | 同上，窄 N 折叠 |
| 1101 | core_ss_oneway_ss | Inner+CorePartition | M/N 并行度不足，跨核借 R，单核 R 段入 UB |
| 1102 | core_ss_twoway_ss | Inner(折叠)+CorePartition | 同上，窄 N 折叠 |
| 1111 | core_ss_ub_ss_oneway_ss | 全三层 | 跨核借 R 后单核 R 段仍超 UB |
| 1112 | core_ss_ub_ss_twoway_ss | 全三层（折叠） | 同上，窄 N 折叠 |

> **命名约定**：`oneway`=直接沿连续 R 行扫描；`twoway`=把长 R 折叠成 2D，先段内扫再段间扫（窄 N 时提升向量道利用率）；`ss`=Sklansky scan；`ub_ss`=在多个 UB chunk 间加一层进位扫描；`core_ss`=在多核 R 分段间加一层进位扫描。

## 公共框架与工具（scan_base）

所有模板复用 [dav310/scan_base.template](templates/dav310/scan_base.template)（转换后 `scan_base.h`）中的工具：

| 工具 | 说明 |
|------|------|
| `CastTrait` + `CastToFp32From<T>` / `CastFromFp32To<T>` | fp16/bf16 载入即升 fp32、输出即降回；fp32/int32 直通。`combine`=min/max 在 fp32 下无误差，Cumsum/Cumprod 用 fp32 累加保精度 |
| 促进类型 `GetPromoteType<T>` | `half/bfloat16_t → float`，`float → float`，`int32_t → int32_t` |
| `CeilDiv` / `CalPow2` / `CalLog2`（向上取整 log2） | Sklansky 层数与分块计数 |
| `GetCacheID(idx)` = `popcount(idx ^ (idx+1)) - 1` | UbChunk 层"二进制进位 memo 层级"：像二进制计数器一样合并 chunk 进位，O(log N) 层 |
| 常量 | `VL=64`（256B VReg / 4B）、`BLOCK_SIZE=32`、`VREG_BYTES=256` |

## Inner 层：Sklansky 并行前缀扫描

**Sklansky（分治扇出）** 把 R 个位置的前缀扫描做成 `ceil(log2(R))` 层，每层对**整行向量载荷**（`alignN × mFold` 元素）做一次结合律 `combine`：

```
第 k 层（0-based）：半组宽 = 2^k，组宽 = 2^(k+1)
  正向锚点 = 组内第 2^k-1 行；把锚点 combine 到其后 2^k 行
  8 行正向示例：
    层0: 锚 0,2,4,6  →  1,3,5,7
    层1: 锚 1,5      →  2,3 与 6,7
    层2: 锚 3        →  4,5,6,7
```

向量化（`__VEC_SCOPE__`）：扫描维是 R（标量步进层号 k），每个前缀节点的载荷是一整行向量（`RegTensor<P>`）；每次 combine 是**整行向量运算**：`DataCopy` 载锚点行 + 目标行 → `Add/Min/Max` → `DataCopy` 写回，尾部用 `UpdateMask` 精确掩码。关键 Reg API：`RegTensor` / `MaskReg` / `UpdateMask` / `DataCopy` / `Add`(cumsum) / `Min`/`Max` + `Compare`+`Select`(cummin/cummax，见流式模板) / `Duplicate` / `Cast`。

详见 [cum_oneway_sklansky.md](templates/cum_oneway_sklansky.md)。**twoway** 在此之上把长 R 折叠成 `[foldCount, foldLen]`：先各 fold 内 Sklansky，再 `Gather` 紧凑重排后 fold 间 Sklansky——窄 N 场景把向量道利用率从 `alignN/VL` 拉到接近 100%。

## 带索引的极值扫描（Cummin/Cummax）：两条路线

对**带位置输出的极值扫描**，除 values 还要输出 argmin/argmax indices。索引选择用
`Compare`+`Select` 两路同 mask 更新——Sklansky 的"整行相加"语义不直接适配索引传播。
按扫描轴在 GM 上是否连续走两条不同的路：

### N > 1（dim 非最后一维）：tile 常驻扫描

→ 模板 [cum_tile_resident_scan.md](templates/cum_tile_resident_scan.md)

每行一次 `DataCopyPad(blockCount = rLen)` 把整段 R 搬进 UB，布局 `[R][lane]`；
扫描沿 R 顺序推进，running-(min,idx) **常驻寄存器**，循环体内零 DMA。
lane 维度用 N（N 够宽时）或借 M 轴折叠（N 窄时）。

R 装不下时切成多个 `rChunk`、块间靠 UB 上的 carry 续传，**并且必须做 ping-pong 双缓冲**：
不做时四条流水占比之和只有 101%（完全串行），加上后升到 132%~173%，实测快 4%~31%。

```
combine（min，last-occurrence 对齐 torch.cummin，NaN 粘滞）：
  leMask  = (x <= curMin)
  nanMask = (x != x)
  takeB   = leMask | nanMask          # Or 合并成一个掩码，值和索引各省一条 Select
  curMin  = takeB ? x : curMin
  curIdx  = takeB ? r : curIdx        # 同 mask 更新索引 → 并列取更晚的 r
```

> ⚠️ **不要**退化成"每个 r 搬入搬出"的逐点流式：`2*R` 次小搬运 + 每 r 三个
> `SetFlag/WaitFlag` 完全串行，实测慢 **8~50x**（0.13x vs 0.58x 综合分）。
> [cum_streaming_scan.md](templates/cum_streaming_scan.md) 描述的是这套 combine 语义的
> 参考实现，**其逐点搬运的数据流不要用于性能路径**。

### N == 1（dim=-1）：行内 Kogge-Stone

扫描轴在 GM 上连续，整行一次搬运只需 1 个描述符，UB 内沿 R 做 `log2(R)` 级合并。
详见 [cum_row_kogge_stone.md](templates/cum_row_kogge_stone.md)。这条路上还有四个
叠加收益：int64 索引在 UB 内 Cast 后直写、fp16/bf16 原生 16 位比较、掩码合并、
**k 行一批搬入 + 行间双缓冲**（`R ≤ rChunk` 时相邻 m 行在 GM 上连续，
一次 `DataCopyPad(blockCount=k)` 就能搬进来，k 行只付一对 flag）。

数值：需要标量比较进位时 fp16/bf16 升 fp32（min/max 不引入误差）；
纯向量比较时 16 位可以原生比较，每 vreg 128 lane。索引扫描期用 int32、搬出前转 int64。

### 搬运摆平之后：lane 并行顺序扫描替掉 Kogge-Stone

Kogge-Stone 的向量指令条数是 `log2(R) · (R/VLC) · ~14`，`log2(R)` 这个因子在
R 大或 lane 少（fp32/int32 每 vreg 只有 64 lane）时就是主要开销。

改成 **VLC 条 lane 各领一段连续区间沿段内顺序推进**：相位一各 lane 归约自己那段，
相位二对 VLC 个段结果做一次短扫描拿 exclusive 前缀，相位三各 lane 带前缀重扫写回。
指令条数降到 `~15 · (R/VLC) + 14 · log2(VLC)`，与 `log2(R)` 脱钩。
段内元素在 UB 上按 lane 跨步，用 `Gather`/`Scatter` 做「虚拟转置」——
不真的搬数据，一条 vgather 横跨 VLC 个段各取一个元素。

关键前提：**索引类型必须与比较类型等宽**（16 位比较配 `int16_t` 索引），
否则索引侧要拆包，收益被吃掉。另外段长就是 lane 间的 UB 地址步长，
**取 2 的幂会撞 bank，实测慢 5.5 倍**。
详见 [cum_lane_parallel_scan.md](templates/cum_lane_parallel_scan.md)。

判断该不该换：把 `indices` 的搬出（int64，占总流量 2/3）临时注掉跑一次，
**耗时不变即纯算力受限**。cann-bench Cummin 上 fp32/int32 用例实测提速 25%~61%，
整体从 0.93x 拉到 1.69x。

## CorePartition 层：跨核 R 分段扫描（1D 超长 R）

当 **M×N 极小（切 A/行只能用满极少核）且 R 超长** 时（典型：1D `[1000003]`），流式/Sklansky 都只用 1~少数核，串行 R 成为瓶颈。此时把**扫描轴 R 跨核切分**，各核算局部前缀，再**保序传播进位**：

```
分段: R 切成 rBorrow 段（≈ coreNum），段 p 覆盖 [pStart, pEnd)
Phase A（每核）: 对本段做局部前缀扫描 → 段内 y/idx；段末 (min,idx) = 本段总计
—— 保序进位（Scan 特有，与 Reduce 可交换合并不同）——
Phase B: 对 rBorrow 个段总计做 exclusive 前缀 combine → carry[p]（段 p 的进位=前面所有段的累积）
Phase C（每核）: 段 p 用 carry[p] 与段内局部前缀 combine → 最终 y/idx（p=0 无进位）
```

Cumsum 用**核内 SyncAll + 原地 y 上的对数层 Sklansky** 实现（无 workspace，进位直接读被扫核的输出行），
代价是要 `SetScheduleMode(1)`、blockDim 覆盖所有参与同步的核、host 预算 6 层调度表，
且为了塞进 6 层把核数静默截断到 64。

**先改成两次 launch 跑通**：phase A 只读求段内合并结果写 workspace，phase B 读回前序段的进位
重扫并写出，stream 顺序即全设备屏障。注意**别让 phase A 顺手写出局部前缀**——那样
phase B 要回读整个输出再写一遍，访存量从 `x + 输出` 涨到 `x + 2×输出`。
另外 phase A 只需要段内的 `(min, argmin)`，那是**归约**而不是扫描，没有 log 因子，
别复用扫描代码再丢掉输出。实测 `[1000003]` 从 2002 µs 降到 34 µs。

**再合成单次 launch**：`kernel launch 的固定开销约 9 µs`，在小用例上和整个搬运时间同量级。
把两次 launch 换成「段归约 → `AscendC::SyncAll()` → 同核继续扫描」，实测 35.7 µs → 19.2 µs。
两个约束：`SyncAll` 前要 `PipeBarrier<PIPE_ALL>()` 保证段结果已落 GM；
**轮空的核也必须走到 `SyncAll`**，否则同步永远凑不齐（提前 `return` 会死锁）。

详见 [cum_core_partition_scan.md](templates/cum_core_partition_scan.md)。

## 适配同族算子

Scan 骨架对同族算子通用，适配只需替换 `combine`/`identity`（+ 带索引算子的索引选择）：

| 算子 | Inner combine | identity | 索引 | 数值注意 |
|------|--------------|----------|------|---------|
| **Cumsum** | `Add` | 0 | — | fp32 累加；大 R 用二分进位保精度 |
| **Cumprod** | `Mul` | 1 | — | fp32 累加；注意溢出/下溢 |
| **Cummax** | `Compare<GE>`+`Compare<NE>`+`Or`+`Select` | -inf | argmax（并列取更晚） | NaN 必须显式粘滞，见下 |
| **Cummin** | `Compare<LE>`+`Compare<NE>`+`Or`+`Select` | +inf | argmin（并列取更晚） | 同上（本库 cummin 实例） |
| **LogCumsumExp** | logaddexp | -inf | — | 在线维护 running max 做数值稳定 |

### 适配要点

1. **维度映射**：扫描轴对应模板的 R；其余轴融合为 M（R 前维度乘积）与 N（R 后维度乘积）。`dim=-1` → N=1；`dim=0` → M=1。
2. **combine 替换**：把 Cumsum 的 `Add` 换成目标算子的结合律运算（Sklansky 层内、UbChunk 进位、CorePartition 进位三处一致替换）。
3. **索引跟踪（Cummin/Cummax）**：`Compare`+`Select` 两路同 mask 同时更新 value 与 index；
   并列语义对齐 torch 的 **last occurrence** → 用 `LE`（`x <= cur` 时取更晚的）。
4. **NaN 必须显式粘滞**：只写一路 `Compare` 不够。cann-bench 的 `value_range=["nan","nan"]`
   生成的是 **NaN 与实数混合**的输入；若只判 `x > cur`，因 `(实数 > NaN) = false`
   会把 NaN "恢复"成实数。正解是再加一路 `Compare<NE>(x, x)` 判 NaN，
   与 `LE` 的结果 `Or` 成一个掩码：`(le | nan) ? x : cur`。
5. **精度**：需要标量比较进位的路径上 fp16/bf16 升 fp32；纯向量比较的路径上
   16 位可以原生比较（每 vreg 128 lane，但索引要配 2 个寄存器）。
   编译器不支持 16 位浮点的标量类型转换，这是两条路分岔的根本原因。

### 模板选择参考（非决策树）

| 场景 | 倾向模板 |
|------|---------|
| Cummin/Cummax，`N > 1`（dim 非最后一维） | **tile 常驻扫描**（整段 R 进 UB，running state 常驻寄存器） |
| Cummin/Cummax，`N == 1`（dim=-1） | **row_kogge_stone**（整行连续搬入 + UB 内对数深度扫描） |
| Cumsum/Cumprod，R 全载 UB，切 M 满核 | oneway_ss（窄 N → twoway_ss） |
| R 超单核 UB，切 M 满核 | ub_ss（窄 N → twoway 变体） |
| `M*N` 远小于核数 + R 超长（1D 大 R） | **core_partition_scan（跨核借 R + 保序进位）** |
| 行极多且每行极短（`N<128 && M>12800 && R<12800`） | SIMT；**不满足这个门槛就别用**，实测慢 5~8 倍 |

> 实际选择还需结合 M/N 规模、UB 预算、是否带索引综合判断。选型公式见性能优化 skill 的 `references/tiling/scan/`。

## 关键技术总览

| 技术 | 涉及模板 | 核心要点 |
|------|---------|---------|
| **Reg VF（`__VEC_SCOPE__`）** | 全部 | `RegTensor`+`MaskReg` 直接操作 256B VReg，逐 `VL=64` 块处理，尾块 `UpdateMask` |
| **FP16/BF16 ↔ FP32** | 全部 | 载入即升 fp32 保证 combine/比较稳定，输出降回，`CastTrait` 控饱和/舍入 |
| **Sklansky 对数深度前缀** | oneway/twoway_ss | `ceil(log2(R))` 层整行向量 combine，破解长 R 串行延迟 |
| **行内 Kogge-Stone（沿 R 方向）** | row_kogge_stone | 扫描轴在 GM 上连续时，整行 1 个描述符搬进 UB，级内分块**从高地址往低地址**原地更新；跨块进位不能靠哨兵，要单独一趟 pass |
| **lane 并行顺序扫描（虚拟转置）** | lane_parallel_scan | VLC 条 lane 各领一段连续区间沿段内推进，`Gather/Scatter` 存取，指令条数与 `log2(R)` 脱钩。段长即 lane 间步长，**取 2 的幂撞 bank 慢 5.5 倍**；三个相位必须同一个 vec scope，lane 间平移用寄存器间 `Gather` |
| **两行交错填指令延迟** | lane_parallel_scan | 单行扫描每步依赖上一步，实测约 4.6 cycle/指令；两行进同一 vec scope 让两条依赖链互相填空，再快 8%~18%。模板已提供 `LaneParallelScanPair` 完整实现 |
| **行批处理 + ping-pong 双缓冲（N==1）** | row_kogge_stone | 一次 `DataCopyPad(blockCount=k)` 搬 k 行摊薄固定开销；x/val/out 各两份 ping-pong 事件使搬入/计算/搬出重叠；不做时四条流水占比 ~101% 完全串行，加后 132~173%。模板已提供 `ScanRowBatchLoop`/`BatchCopyIn`/`BatchCompute`/`BatchCopyOut` 骨架 |
| **16 位原生比较** | row_kogge_stone | `C = T` 每 vreg 128 lane，进出各省一趟 Cast；索引仍 int32 需配 2 个寄存器 + `UnPack<LOWEST/HIGHEST>` 拆掩码。**只在 R 大时划算** |
| **int64 索引 UB 内直写** | 全部（带索引） | 扫描用 int32、搬出前 `Cast` 成 int64，与 int32 工作区复用同一 buffer（写指针天然落后于读指针）；工作区偏移**必须按 vreg 对齐** |
| **窄 N 折叠（twoway）** | twoway 系 | 长 R 折叠进空闲向量道，`Gather` 紧凑重排，段内→段间两级扫描 |
| **二进制进位 memo（GetCacheID）** | ub_ss 系 | 多 UB chunk 进位按二进制计数器层级合并，O(log N) 层而非 O(N) 串行 |
| **流式 running-state + Compare/Select** | streaming_scan | 沿 R 逐步更新常驻 (value,idx)，两路同 mask，天然支持索引与 NaN 传播 |
| **跨核保序进位（CorePartition）** | core_ss 系 | R 分段 → 局部前缀 → exclusive 进位（保序，Scan 特有）→ 段内合并 |
| **strided DataCopyPad（Compact）** | 流式/全部 | `blockCount` 一次搬多行、`srcStride/dstStride` 跨 R×N 跳跃，Compact 打包成连续 W |
| **多 work-item 复用 UB 的 WAR 屏障** | 流式 | 一核处理多 work-item 时，上一 item 末次 CopyOut(MTE3) 与下一 item InitRunning(V) 需 `MTE3→V` 屏障，否则末位 idx 被覆写（实测踩坑，见 usage_guide.md §4） |
