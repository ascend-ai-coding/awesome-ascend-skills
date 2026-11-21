# 多核两级归并排序方案（MrgSort 外部归并）

> **前提**：本方案针对 [patterns.md](patterns.md) 中的 **Pattern C（多核两级归并）**——数据量超过 `tileSize × coreNum`，每个核需要处理多个 tile。
>
> 如果 N 不满足 Pattern C 条件，请先阅读 [patterns.md](patterns.md) 选择正确的方案。

---

## 1. 变量定义与两级归并

### 1.1 变量定义

| 符号 | 含义 | 来源 |
|------|------|------|
| `N` | 总元素数 | 输入 |
| `K` | Top-K 值 | 输入 |
| `dtype` | 数据类型 | 输入参数 |
| `coreNum` | 使用核数 | tiling 计算 |
| `tileSize` | 单 tile 元素数 | UB 容量推导（工程值 4096） |
| `M` | 归并路数（= 4） | MrgSort API 常量 |
| `ubSize` | UB 大小 | `GetPlatformInfo().GetUbSize()` |
| `S_c` | 单核 tile 数 = `ceil(ceil(N/tileSize)/coreNum)` | tiling 计算 |
| `E_c` | 单核元素数 = `S_c × tileSize` | tiling 计算 |

### 1.2 两级形式化定义

```
第一级：核内多 tile 归并
  输入：每核 S_c 个有序 tile（每个 tile 大小 = tileSize）
  操作：各核独立对 S_c 个 tile 进行 M-way 归并（ceil(log_M(S_c)) 轮）
  输出：每核 1 个有序数列（共 coreNum 个）
  特点：同一核内的数据，无需跨核同步

第二级：跨核归并
  输入：coreNum 个有序数列（分散在各核 workspace 区域）
  操作：将 coreNum 路逐步归并至 ≤ M 路
    - Round 0：按 M 路一组，从各核 workspace 读取，输出到 workspace 公共区域
    - Round 1+：对上一轮的 group 输出继续归并，直至路数 ≤ M
  输出：≤ M 个有序数列 → Core 0 最终归并 → K 个元素 → GM
  特点：每一轮归并后都需要 SyncAll 等待所有核完成
```

### 1.3 为什么需要两级归并

Pattern C 下，数据量 N > tileSize × coreNum，每个核被分配的元素数 E_c > tileSize。这意味着：

```
E_c = ceil(ceil(N/tileSize) / coreNum) × tileSize > tileSize
```

每个核需要排序多个 tile，排序后得到多个有序 tile。如果直接将这些 tile 跨核归并，则归并轮次 = `ceil(log_M(totalTiles))`，跨核归并每一轮都需要 `SyncAll` 等待所有核完成，直接跨核归并的 `SyncAll` 开销不可接受。因此必须先在各核内部归并（第一级），再将各核算结果跨核归并（第二级），得到最终的 Top-K 结果。

---

## 2. 四阶段架构推导

### 2.1 从两级归并到四阶段

将两级归并映射到 AscendC kernel 执行模型，自然得到四个阶段：

```
第一级（各核独立并行）：
  Phase 1: tile 排序 — 各核独立并行，将 GM 数据读入 UB，排序后写入 workspace
  Phase 2: 核内归并 — 各核独立并行，对 S_c 个 tile 进行多轮 M-way 归并

第二级（跨核）：
  Phase 3: 跨核归并 — 将 coreNum 路逐步归并至 ≤ M 路
  Phase 4: 最终归并+输出 — Core 0 将 ≤ M 路归并，Extract 分离 value/index 输出到 GM
```

| Phase | 职责 | 核数参与 | 数据流 |
|-------|------|---------|--------|
| **Phase 1** | 各核并行 tile 排序 | 全核 | GM → UB → Sort → workspace |
| **Phase 2** | 各核独立归并自己的全部 tiles，直到各核输出单个有序数列 | 全核 | workspace → UB → MrgSort → workspace |
| **Phase 3** | 跨核归并（逐步减少路数至 ≤ M） | 递减 | workspace → UB → MrgSort → workspace |
| **Phase 4** | Core 0 最终归并（≤ M 路）+ Extract 输出 | 1 核 | workspace → UB → Extract → GM |

### 2.2 Phase 3/4 边界推导

Phase 3 的退出条件由 M-way 归并的物理限制决定：

```
Phase 3 循环条件：while listNum > M
```

**为什么是 `listNum > M`**：当路数 ≤ M(=4) 时，单次 MrgSort 调用即可完成最终归并，不需要多轮 group 循环。此时交给 Phase 4 处理更高效——Phase 4 包含 Extract 分离 value/index 并输出到 GM，是完整的"归并+输出"阶段。

### 2.3 归并轮次公式

```
Phase 2 归并轮次 = ceil(log_M(S_c))
Phase 3 归并轮次 = ceil(log_M(coreNum)) - 1   （最后一轮由 Phase 4 完成）
Phase 4          = 1 轮 ≤M-way 归并
```

---

## 3. Tiling 切分建模

### 3.1 Tile 切分公式

```
totalTiles      = ceil(N / tileSize)
frontCoreTiles  = ceil(totalTiles / coreNum)         // 前 coreNum-1 核各处理的 tile 数
usedCore        = ceil(totalTiles / frontCoreTiles)  // 实际使用核数
lastCoreTiles   = totalTiles - (usedCore - 1) × frontCoreTiles
lastTileSize    = N - tileSize × (totalTiles - 1)          // 最后一个 tile 可能残缺
elementsPerCore = frontCoreTiles × tileSize
```

### 3.2 UB 批次计算

归并和输出阶段的单次处理量由 UB 容量决定：

```
// Phase 2/3 归并阶段：UB 容纳 2 个队列（输入+输出），每个队列 4 路
mergeBytesPerElem  = 2 × M × 8B  = 64B
onceMaxElementsMerge  = (ubSize / mergeBytesPerElem)  / 32 × 32

// Phase 4 输出阶段：额外需要 Extract 和输出 buffer
outputBytesPerElem = 2 × M × 8B + M × (4B + 4B + sizeof(dtype)) = 104B (float16)
onceMaxElementsOutput = (ubSize / outputBytesPerElem) / 32 × 32
```

**数值示例**（192KB UB, float16）：

| 阶段 | bytes/elem | onceMaxElements |
|------|-----------|----------------|
| Phase 2/3 归并 | 64B | `(192KB/64B)/32×32 = 3008` |
| Phase 4 输出 | 104B | `(192KB/104B)/32×32 = 1824` |

### 3.3 Workspace 大小公式

Workspace 按**核数**分配（非按总元素数），使用双缓存机制：

```
wsPerCoreBytes = (E_c × 8B × 2 + 32 - 1) / 32 × 32   // 每核 proposal 空间(双缓存)，32B 对齐
wsTotalFloats  = usedCore × wsPerCoreBytes / sizeof(float)
wsHalfFloats   = wsTotalFloats / 2                      // 双缓存各占一半

workspace[0] = workSpace[0 .. wsHalfFloats-1]
workspace[1] = workSpace[wsHalfFloats .. 2×wsHalfFloats-1]
```

**为什么按核数分配而非总元素数**：
- 每个核在 workspace 中有独立的存储区域
- 按核数分配保证各核数据不越界，尾核（coreNum-1）的偏移 `(coreNum-1) × E_c × 2` < `wsHalfFloats`

### 3.4 TilingData 结构

```cpp
struct SortTilingData {
    int64_t coreNum;               // 实际使用核数
    int64_t frontCoreTiles;        // 前 coreNum-1 核的 tile 数
    int64_t lastCoreTiles;         // 尾核 tile 数
    int64_t totalTiles;            // 总 tile 数
    int64_t elementsPerCore;       // 单核处理元素数
    int64_t tileSize;              // tileSize（= 4096）
    int64_t lastTileSize;          // 末 tile 元素数（≤ tileSize）
    int64_t totalElements;         // 总元素数 N
    int64_t onceMaxElementsMerge;  // Phase 2/3 单次归并最大元素数
    int64_t onceMaxElementsOutput; // Phase 4 单次输出最大元素数
};
```

---

## 4. 四阶段方案设计

### 4.1 Phase 1：tile 排序

#### 4.1.1 设计目标

将 N 个元素按 tileSize 切分，各核并行排序自己的 tile，结果写入 workspace[0]。

#### 4.1.2 UB 布局

Phase 1 需要 6 个 buffer，每元素 UB 占用 34B（float16）：

| Buffer | 大小 | 用途 |
|--------|------|------|
| inputQueue | tileSize × sizeof(dtype) | 原始输入 |
| inputValueTempBuf（可选） | tileSize × sizeof(float) | dtype→float 转换 |
| sortedValueIndexUb | tileSize × sizeof(uint32_t) | 索引数组 |
| concatTempBuf | tiling 侧使用 `GetConcatTmpSize` 接口获取 | Concat tmp buffer |
| sortTempBuf | tiling 侧使用 `GetSortTmpSize` 接口获取 | Sort tmp buffer |
| sortedValueUb | tileSize × 8B | Sort 输出 proposal |

**建议**：工程中取 tileSize = 4096，经过实测，这通常是最优取值。

#### 4.1.3 算法步骤

```
for tileIdx in [0, myTiles):
  1. 计算偏移: globalOffset = blockIdx × S_c × tileSize + tileIdx × tileSize
  2. 输入: DataCopyPad(GM → UB)，Pad 填充 -inf
  3. 类型转换: Cast(dtype → float)
  4. 初始化索引: ArithProgression(offset, offset+1, ..., offset+tileSize-1)
  5. 合成 proposal: Concat(value + index)
  6. 排序: Sort<float, true>(proposal)
  7. 输出: DataCopyPad(UB → workspace[0])
```

**数据依赖**：步骤 4（ArithProgression）使用 VECTOR 管道，步骤 2（DataCopyPad）使用 MTE2 管道。步骤 4 前必须 `WaitFlag<MTE2_V>` 确保索引 buffer 写入的源数据已就绪。

#### 4.1.4 workspace 地址排布

```
workspace[0] 排布（Phase 1 输出）：
┌────────────────────coreStride──────────────────┐┌─────coreStride──────────────┐
│ core0                                           ││ core1                       │...
│ tile0[tileSize×2] tile1[tileSize×2] ... tile[S_c-1][tileSize×2]     ││ tile0..tile(S_c-1)          │
└─────────────────────────────────────────────────┘└─────────────────────────────┘
  GetCoreWsOffset(0) = 0                           GetCoreWsOffset(1) = E_c × 2
  tile偏移 = GetCoreWsOffset(core) + tileIdx × tileSize × 2
```

**关键**：`GetCoreWsOffset(i) = i × E_c × 2`（直接乘法，不用 `GetSortLen` 包装）。

---

### 4.2 Phase 2：核内多 tile 归并

#### 4.2.1 设计目标

每个核独立将自己的 S_c 个有序 tile 通过多轮 M-way 归并合并为 1 个有序数列。

**核间无依赖**：各核操作完全独立，无需同步。

#### 4.2.2 UB 布局

Phase 2 归并阶段仅需要 2 个 buffer（M-way 输入 + M-way 输出），每元素 UB 占用 64B：

| Buffer | 大小 | 用途 |
|--------|------|------|
| copyInQueue | M × onceMaxElementsMerge × 8B | M 路归并输入（proposal 格式） |
| sortedQueue | M × onceMaxElementsMerge × 8B | M 路归并输出（proposal 格式） |

**约束**：`onceMaxElementsMerge × 64B ≤ ubSize`，因此 `onceMaxElementsMerge ≤ ubSize / 64B`。

**对比 Phase 1**：Phase 2 无需 Cast buffer、索引数组、Concat/Sort tmp buffer，buffer 数从 6 降至 2，每元素 UB 占用增至 64B（因为需要容纳 M 路输入输出）。Phase 2 的 buffer 在 Phase 1 结束后通过 `pipe_.Reset()` 重新分配。

#### 4.2.3 归并状态变量

| 变量 | 含义 | 初始值 | 更新方式 |
|------|------|--------|---------|
| `listNum` | 当前待归并数列数 | `S_c` | `ceil(listNum / M)` |
| `currentElements` | 每路地址间隔（元素数） | `tileSize` | `currentElements × M` |
| `currentTailElements` | 尾块实际长度 | `lastTileSize`（尾核）否则 `tileSize` | `currentElements × (remainListNum-1) + currentTailElements` |
| `truncationFlag` | 上一轮输出是否被截断 | `false` | 每轮归并后设置 |

#### 4.2.4 截断逻辑

**定义**：截断是指在归并过程中，当某轮归并的输出量足以覆盖 Top-K 需求时，后续归并只需读取前 K 个有效元素，丢弃超出 K 的部分，而非继续处理全量数据。

**原因**：Top-K 场景下，最终只需要最大的 K 个元素。归并排序保证了输出有序，因此当一轮归并的输出 ≥ K 时，可以确定最终结果必然在这批输出的前 K 个中。后续轮次继续处理全量数据毫无意义——超出 K 的部分最终会被丢弃，反而浪费计算和带宽。因此需要引入截断，在满足条件后将每路有效长度限制为 K，只归并前 K 个元素。

**标志位**：`truncationFlag` 是跨 Phase 2/3 持久化的布尔标志，用于描述「上一轮归并输出」是否被截断：

| 值 | 含义 | 下一轮 effectiveLength |
|----|------|----------------------|
| `false` | 上一轮输出完整，全量有效 | `currentElements` |
| `true` | 上一轮输出已截断，仅前 K 个有效 | `min(currentElements, K)` |

**核心语义**：
```
truncationFlag 描述的是「上一轮输出」是否被截断：
  false → 读入全量数据（effectiveLength = currentElements）
  true  → 只读入 K 个（effectiveLength = min(currentElements, K)）

关键顺序：归并执行时 truncationFlag 仍为旧值
  → 首次触发截断时读入完整数据（正确）
  → 下一轮时 truncationFlag 已为 true，读入截断后的 K 个（正确）
```

**触发条件**：`currentElements × M ≥ K`（归并后的理论输出 ≥ K 时，下一轮即可截断）

**设置时机**：在每轮归并执行**后**根据触发条件设置，标记本轮输出状态，影响下一轮的读入长度。

**触发时机取决于 shape 与 K**：
- K 较小 → Phase 2 某轮即触发 → 后续轮次全部截断
- K 较大 → Phase 2 全程正常 → Phase 3 才首次触发

#### 4.2.5 伪代码

```
listNum = S_c
currentElements = tileSize
currentTailElements = (blockIdx == coreNum-1) ? lastTileSize : tileSize
truncationFlag = false
workSpaceFlag = 0

while listNum != 1:
    nextElements = currentElements × M

    if nextElements >= K:
        Phase2TruncatedMergeRound()    // truncationFlag 为旧值 → 读入完整数据
        truncationFlag = true          // 归并后标记
    else:
        Phase2NormalMergeRound()
        truncationFlag = false

    // 参数更新（截断和非截断分支共用）
    currentGroupNum = ceil(listNum / M)
    remainListNum = listNum - (currentGroupNum - 1) × M
    currentTailElements = currentElements × (remainListNum - 1) + currentTailElements
    listNum = currentGroupNum
    currentElements = currentElements × M

    workSpaceFlag = (workSpaceFlag + 1) % 2   // 交换读写缓存
```

#### 4.2.6 尾块长度更新公式

```
currentTailElements = currentElements × (remainListNum - 1) + currentTailElements
```

**推导**：最后一个 group 有 `remainListNum` 路（可能 < M）。其中前 `remainListNum-1` 路都是完整长度 `currentElements`，第 `remainListNum` 路继承了上一轮的尾块残缺长度 `currentTailElements`。

**更新顺序约束**：必须先用旧的 `currentElements` 计算尾块长度，再更新 `currentElements`。

#### 4.2.7 workspace 地址排布

Phase 2 的数据始终位于核专属 workspace 区域内（基址 = `GetCoreWsOffset(blockIdx)`），每轮归并 source→destination workspace 交替。

**Round 1**（输入：workspace[0] 核区域的 S_c 个 tiles）：

```
核 blockIdx 的 workspace 区域（基址 = coreBase）：

 输入（workspace[0]）：
 ┌────listStride────┐┌────listStride────┐                   ┌────listStride────┐
 │ list0 [tileSize×2]       ││ list1 [tileSize×2]       │ ... list(S_c-1)  │ list(S_c-1)[tileSize×2] │
 └───────────────────┘└───────────────────┘                   └──────────────────┘
       ↓ M-way 归并 (group 0)     ↓ M-way 归并 (group 1)  ...

 输出（workspace[1]）：
 ┌──────groupStride──────────┐┌──────groupStride──────────┐
 │ group0 [currentElements×M×2]││ group1 [currentElements×M×2]│...
 └───────────────────────────┘└───────────────────────────┘
```

**Round N**（输入/输出类似，listStride 和 groupStride 随轮次增长）：

| 轮次 | listStride | groupStride |
|------|-----------|-------------|
| Round 1 | `tileSize × 2` | `M × tileSize × 2` |
| Round 2 | `(M×tileSize) × 2` | `M × (M×tileSize) × 2` |
| Round r | `(M^(r-1)×tileSize) × 2` | `M × (M^(r-1)×tileSize) × 2` |

**通用地址公式**：
```
offsets[i] = coreBase + (groupIdx × M + i) × currentElements × 2
wsOutOffset = coreBase + groupIdx × M × currentElements × 2
```

---

### 4.3 Phase 3：跨核归并

#### 4.3.1 设计目标

将 Phase 2 输出的 coreNum 个有序数列通过跨核归并逐步减少路数，直至 ≤ M 路。

**状态继承**：Phase 3 继承 Phase 2 的 `truncationFlag`、`workSpaceFlag`、`currentElements`。

#### 4.3.2 UB 布局

**复用 Phase 2 UB**：Phase 3 的归并操作与 Phase 2 完全相同（M-way 归并），直接复用 Phase 2 的 2 个 buffer，无需重新分配：

| Buffer | 大小 | 用途 |
|--------|------|------|
| copyInQueue | M × onceMaxElementsMerge × 8B | M 路归并输入（proposal 格式） |
| sortedQueue | M × onceMaxElementsMerge × 8B | M 路归并输出（proposal 格式） |

每元素 UB 占用不变（64B），`onceMaxElementsMerge` 取值不变。

#### 4.3.3 Round 0 vs Round 1+ 的区别

Phase 3 的核心复杂性在于 **Round 0 的数据来源特殊**：

| 维度 | Round 0 | Round 1+ |
|------|---------|----------|
| **数据来源** | 各核 workspace 区域（core 粒度） | 上一轮 group 输出（proposal 粒度） |
| **listStrideInGroup** | `E_c × 2`（核间隔） | `currentElements × 2` |
| **输出 groupSize** | `K × M × 2` | `K × M × 2`（截断后）或 `M × currentElements × 2` |
| **偏移计算** | `GetCoreWsOffset(groupIdx × M + i)` | `(groupIdx × M + i) × currentElements × 2` |

**为什么 Round 0 特殊**：Round 0 需要从各核独立的 workspace 区域读取数据。由于各核的 workspace 区域由 `coreStride = E_c × 2` 分隔，且各核的输出量可能不同（截断状态下为 K），Round 0 使用 `GetCoreWsOffset()` 直接定位到各核区域。

Round 1+ 数据已经在 workspace 公共区域以 proposal 格式连续排布，使用普通的 `currentElements × 2` 作为偏移。

#### 4.3.4 有效长度计算

Round 1+ 的每路有效长度取决于截断状态：

```
effectiveLen = truncationFlag ? min(currentElements, K) : min(currentElements, K)
```

**推导**：无论截断与否，每路有效长度都应取 `min(currentElements, K)`——截断后 `currentElements ≥ K` 则退化为 K，未截断时若 `currentElements < K` 则直接取 `currentElements`。Phase 3 必须在未截断分支也使用 `min()` 的原因：Round 0 的输出可能已被 `stopThreshold` 限制为 K，但 `truncationFlag` 要到本轮归并后才检查设置，此时 `currentElements` 可能大于实际有效数据。

#### 4.3.5 伪代码

```
workSpaceFlag = Phase 2 结束时的值
workspaceInput = workspace[workSpaceFlag]
workspaceOutput = workspace[1 - workSpaceFlag]

// Round 0：从各核 workspace 区域读取
listLen = truncationFlag ? K : min(currentElements, K)
groups = ceil(coreNum / M)

for each group in Round 0:
    for i in [0, M):
        coreIdx = groupIdx × M + i
        if coreIdx < coreNum:
            offsets[i] = GetCoreWsOffset(coreIdx)
            listRemainElements[i] = listLen

    stopThreshold = sum(listRemainElements) - K
    while sum(listRemainElements) > stopThreshold:
        CopyIn → MrgSort → CopyOut
        UpdateSortInfo()

SyncAll()
workSpaceFlag = (workSpaceFlag + 1) % 2

// Round 1+：从 proposal 排布读取，循环直到 listNum ≤ M
listNum = groups
currentElements = listLen × M

while listNum > M:
    effectiveLen = truncationFlag ? min(currentElements, K) : min(currentElements, K)
    groups = ceil(listNum / M)

    for each group:
        for i in [0, M):
            blockNum = groupIdx × M + i
            if blockNum < listNum:
                offsets[i] = blockNum × currentElements × 2
                listRemainElements[i] = effectiveLen

        stopThreshold = sum(listRemainElements) - K
        while sum(listRemainElements) > stopThreshold:
            CopyIn → MrgSort → CopyOut
            UpdateSortInfo()

    if currentElements × M ≥ K:
        truncationFlag = true

    SyncAll()
    workSpaceFlag = (workSpaceFlag + 1) % 2
    listNum = groups
    currentElements = currentElements × M
```

#### 4.3.6 SyncAll 时机

Phase 3 的 SyncAll 发生在**每轮归并结束后**（而非每个 group 后）：

```
Round 0 (coreNum→groups):
  for each group in [0, groups):    ← 各核并行处理不同 group
    MrgSort group 内的 4 路数据
  SyncAll()                          ← 所有 group 完成，数据可见

Round 1+ (while listNum > M):
  for each group in [0, groups):
    MrgSort group 内的 4 路数据
  SyncAll()                          ← 每轮结束同步
```

**为什么不是每个 group 后 SyncAll**：一个 group 内的 4 路归并由单个核完成（数据处理在核内），round 内各 group 由不同核并行执行。SyncAll 只需在所有核完成本轮所有 group 后执行一次。

---

### 4.4 Phase 4：最终归并 + 输出

#### 4.4.1 设计目标

Core 0 将 Phase 3 输出的 ≤ M 路有序数列进行最终归并，Extract 分离 value 和 index，输出 K 个元素到 GM。

**仅 Core 0 执行**，其他核等待。

#### 4.4.2 UB 布局

Phase 4 需要 5 个 buffer，每元素 UB 占用 104B（float16）：

| Buffer | 大小 | 用途 |
|--------|------|------|
| sortedQueue | 4 × onceMaxElementsOutput × 8B | 归并输出 |
| copyInQueue | 4 × onceMaxElementsOutput × 8B | 归并输入 |
| castValueQueue | 4 × onceMaxElementsOutput × 4B | Extract value |
| castIndexQueue | 4 × onceMaxElementsOutput × 4B | Extract index |
| outValueQueue | 4 × onceMaxElementsOutput × sizeof(dtype) | Cast 输出（float → dtype） |

**为何 index 不需要额外输出 buffer**：Extract 分离出的 index 已是 `uint32_t` 最终格式，可直接 `DataCopyPad` 到 GM。而 value 是 float，需 `Cast` 回原始 dtype（如 float16），`Cast` API 要求独立的输出 buffer，因此需要 `outValueQueue`。

```
数据流：MrgSort → Extract ┬→ castValueQueue(float) → Cast → outValueQueue(dtype) → GM
                          └→ castIndexQueue(uint32_t) ────────────→ DataCopyPad → GM
```

**UB 切换**：Phase 3 结束后调用 `pipe_.Reset()` 释放 Phase 2/3 的 UB，重新分配 Phase 4 的 5 个 buffer。

#### 4.4.3 伪代码

```
workspaceInput = workspace[workSpaceFlag]   // 继承 Phase 3 输出位置

// 设置各路有效长度
for i in [0, listNum):
    if i < listNum - 1:
        listRemainElements[i] = currentElements
    else:
        listRemainElements[i] = currentTailElements

allRemainElements = K × listNum
stopThreshold = allRemainElements - K

while allRemainElements > stopThreshold:
    CopyIn → MrgSort → Extract → Cast(dtype) → CopyOut(GM)
    UpdateSortInfo()

SyncAll()   // 等待 Core 0 完成
```

#### 4.4.4 workspace 地址排布

```
输入 workspace[workSpaceFlag]（Phase 3 输出）：
 ┌──────────────────────────────────────────────────────┐
 │ list0              list1        list2        list3   │  (≤M 路)
 │ [currentElements×2] 或 [currentTailElements×2]       │
 └──────────────────────────────────────────────────────┘
   listStrideInGroup = currentElements × 2

输出：GM（仅 Core 0）
 ┌──────────────┐
 │ K 个 value    │ + K 个 index
 └──────────────┘
```

---

## 5. 关键设计机制

### 5.1 workspace 双缓存

**机制**：`workSpaceFlag_` 在两个 workspace 半区之间交替，实现读写分离。

```
workSpaceFlag = 0  →  输入 = workspace[0], 输出 = workspace[1]
workSpaceFlag = 1  →  输入 = workspace[1], 输出 = workspace[0]
```

**交换时机**：每一轮归并结束后交换。

| 阶段 | 归并前 flag | 输入 workspace | 输出 workspace | 归并后 flag |
|------|-----------|---------------|---------------|------------|
| Phase 1 | - | GM | workspace[0] | 0（初始化） |
| Phase 2 Round 1 | 0 | workspace[0] | workspace[1] | 1 |
| Phase 2 Round 2 | 1 | workspace[1] | workspace[0] | 0 |
| ... | 交替 | ... | ... | ... |
| Phase 3 Round 0 | Phase2 末值 | workspace[flag] | workspace[1-flag] | 翻转 |
| Phase 3 Round 1+ | 逐轮翻转 | workspace[flag] | workspace[1-flag] | 逐轮翻转 |
| Phase 4 | Phase3 末值 | workspace[flag] | GM | 无交换 |

### 5.2 SyncAll 时序

| 阶段 | SyncAll 时机 | 次数 | 原因 |
|------|-------------|------|------|
| Phase 1 | 无 | 0 | 各核并行，无核间依赖 |
| Phase 2 | Phase 2 全部结束后 | 1 | Phase 3 需要所有核的归并结果 |
| Phase 3 | 每轮归并结束后 | 每轮 1 次 | 下一轮需要上一轮的完整输出 |
| Phase 4 | Core 0 完成后 | 1 | 等待输出完成 |

**总次数公式**：`SyncAll = 1 + ceil(log_M(coreNum)) - 1 + 1 = ceil(log_M(coreNum)) + 1`

### 5.3 UB 分阶段重置

**原因**：不同阶段需要的 buffer 组合不同。Phase 2/3 不需要 Extract buffer，Phase 4 不需要 Phase 1 的 sort buffer。分阶段分配可以让归并阶段使用更大批次。

```
Phase 1 结束后:  pipe_.Reset() → InitBuffer(Phase 2/3 的 2 个 buffer, 64B/elem)
Phase 3 结束后:  pipe_.Reset() → InitBuffer(Phase 4 的 5 个 buffer, 104B/elem)
```

**效果**：归并阶段每批可处理 3008 个元素（vs. 若共用 Phase 1 buffer 则只能处理 ~1500 个）。

### 5.4 截断标志跨阶段语义

`truncationFlag` 是**跨 Phase 2/3 持久化**的标志：

```
truncationFlag = false  →  上一轮输出完整  →  effectiveLength = currentElements
truncationFlag = true   →  上一轮输出截断  →  effectiveLength = min(currentElements, K)

设置时机（关键）：
  每轮归并执行后设置 → 标记本轮输出是否被截断 → 影响下一轮的读入长度
```

这确保了截断语义在 Phase 2 和 Phase 3 之间无缝传递。

---

## 6. 设计检查清单

### 6.1 方案适用性

- [ ] N > tileSize × coreNum？（Pattern C 前提）
- [ ] ascend910b 平台？（MrgSort4 禁止使用）
- [ ] tileSize = 4096？

### 6.2 动态参数

- [ ] Phase 2 循环条件为 `while (listNum != 1)`？
- [ ] Phase 3 循环条件为 `while (listNum > M)`？
- [ ] `currentElements` 更新为 `currentElements × M`？
- [ ] `onceMaxElementsMerge/Output` 从 tiling 传入而非硬编码？

### 6.3 地址计算

- [ ] `GetCoreWsOffset(i) = i × E_c × 2`（直接乘法，非 `GetSortLen` 包装）？
- [ ] workspace 大小先按字节计算再转 float 数？
- [ ] Phase 2 尾块长度用公式 `currentElements × (remainListNum-1) + currentTailElements`？
- [ ] 参数更新顺序：先算尾块长度，再更新 `currentElements`？

### 6.4 归并循环

- [ ] 每个归并循环都调用 `UpdateSortInfo()`？（否则死循环）
- [ ] 归并循环终止条件为 `allRemainElements > stopThreshold`？
- [ ] `stopThreshold = allRemainElements - K` 在循环前捕获？

### 6.5 截断逻辑

- [ ] `truncationFlag` 在归并执行**后**设置？（确保首次触发时读入完整数据）
- [ ] Phase 3 有效长度 `effectiveLen = min(currentElements, K)`（Round 0 输出可能已被 K 截断）？
- [ ] Phase 2/3 截断条件统一为 `currentElements × M ≥ K`？

---

---

## 7. Kernel 实现模板

### 7.1 Process() 流程框架

```cpp
__aicore__ inline void Process()
{
    // Phase 1: 各核并行 tile 排序
    uint32_t myTiles = (blockIdx_ < coreNum_ - 1) ? frontCoreTiles_ : lastCoreTiles_;
    for (uint32_t t = 0; t < myTiles; t++) {
        int64_t globalOffset = blockIdx_ * frontCoreTiles_ * tileSize_ + t * tileSize_;
        uint32_t curTileSize = (blockIdx_ == coreNum_ - 1 && t == myTiles - 1) ? lastTileSize_ : tileSize_;
        SortInSingleCore(curTileSize, globalOffset);
    }
    // Phase 2, 3 UB 分配
    pipe_.Reset();
    onceMaxElements_ = onceMaxElementsMerge_;
    pipe_.InitBuffer(sortedQueue_, TOPK_DOUBLE_BUFFER, TOPK_LIST_MAX * onceMaxElements_ * TOPK_PROPOSAL_SIZE);
    pipe_.InitBuffer(copyInQueue_, TOPK_DOUBLE_BUFFER, TOPK_LIST_MAX * onceMaxElements_ * TOPK_PROPOSAL_SIZE);
    Phase2IntraCoreMerge();
    // Phase 2 完全结束后全核同步
    SyncAll();
    // Phase 3内部每轮归并后全核同步
    Phase3InterCoreMerge();
    // Phase 4 UB 分配
    pipe_.Reset();
    onceMaxElements_ = onceMaxElementsOutput_;
    pipe_.InitBuffer(sortedQueue_, TOPK_DOUBLE_BUFFER, TOPK_LIST_MAX * onceMaxElements_ * TOPK_PROPOSAL_SIZE);
    pipe_.InitBuffer(copyInQueue_, TOPK_DOUBLE_BUFFER, TOPK_LIST_MAX * onceMaxElements_ * TOPK_PROPOSAL_SIZE);
    pipe_.InitBuffer(castValueQueue_, TOPK_DOUBLE_BUFFER, TOPK_LIST_MAX * onceMaxElements_ * sizeof(float));
    pipe_.InitBuffer(castIndexQueue_, TOPK_DOUBLE_BUFFER, TOPK_LIST_MAX * onceMaxElements_ * sizeof(uint32_t));
    pipe_.InitBuffer(outValueQueue_, TOPK_DOUBLE_BUFFER, TOPK_LIST_MAX * onceMaxElements_ * sizeof(DTYPE_X));
    Phase4FinalMergeAndOutput();
}
```
