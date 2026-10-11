# Sort 类算子场景路由

> 本文档用于**场景判定**和**方案选择**。确定场景后，按链接进入对应详细文档。

---

## 1. 归并排序原理

### 1.1 分治思想

归并排序（Merge Sort）的核心是**分治**（Divide and Conquer）：

```
Divide:   将无序序列递归拆分为子序列，直至子序列长度为 1（自然有序）
Conquer:  将有序子序列两两归并，逐层合并为更大的有序序列
```

```
无序序列: [5, 3, 8, 1, 9, 2, 7, 4]
        ↓ Divide
        [5,3,8,1]   [9,2,7,4]
        ↓ Divide
      [5,3] [8,1]   [9,2] [7,4]
        ↓ Divide
     [5] [3] [8] [1] [9] [2] [7] [4]    ← 长度为 1，自然有序
        ↓ Conquer (归并)
      [3,5] [1,8]   [2,9] [4,7]
        ↓ Conquer
        [1,3,5,8]   [2,4,7,9]
        ↓ Conquer
        [1,2,3,4,5,7,8,9]
```

> 稳定排序：归并排序是**稳定排序**——相同值的元素，归并后保持原始相对顺序。

### 1.2 M-way 归并

AscendC 的 `MrgSort` API 支持 **M 路归并**（`MRG_SORT_ELEMENT_LEN`，通常 M=4），即一次 API 调用可将 M 路有序数列归并为一个有序数列。

```
输入：M 路各自有序的数列
    list0: [a0, a1, a2, ...]  (有序)
    list1: [b0, b1, b2, ...]  (有序)
    ...
    list(M-1): [...]

MrgSort M-way → 输出：归并后的一条有序数列
```

### 1.3 并行排序归并策略

当数据量超过 UB 容量时，单次 Sort 无法处理全部数据。此时利用归并排序的分治特性，结合多核并行能力：

```
大数据集 → 按 UB 容量切分为多个 tile → 各核并行排序 tile（Divide）
         → 多核多轮归并（Conquer）→ 最终有序结果
```

```
GM (Global Memory)
  ↓ DataCopy
UB (Unified Buffer, 大小 = ubSize)  ← Sort/MrgSort 操作在 UB 内执行
  ↓ Sort/MrgSort API
UB → GM (输出)
```

关键认知：**UB 容量决定了一次能排序多少元素，进而决定了需要多少层级归并**——数据量越大，需要的归并层级越多，方案越复杂。超出 UB 容量的数据必须拆分为多个 **tile**，分别排序后再归并。

---

## 2. 核心约束：UB 内存

### 2.1 UB 容量

```
ubSize = GetPlatformInfo().GetUbSize()   // 从平台信息获取，不同芯片取值不同
```

### 2.2 排序操作的 UB 空间占用模型

**每元素 UB 占用**：

```
concatTmpPerElem = GetConcatTmpSize(platform, t, sizeof(float)) / t
sortTmpPerElem   = GetSortTmpSize(platform, t, sizeof(float)) / t
sortBytesPerElem = sizeof(dtype) + sizeof(float) + sizeof(uint32_t) + PROPOSAL_SIZE
                   + concatTmpPerElem + sortTmpPerElem
```

以常见 dtype 为例（实践中 `concatTmpPerElem ≈ sortTmpPerElem ≈ sizeof(float) * 2 = 8B`）：

| dtype | sizeof(dtype) | sortBytesPerElem 近似值 |
|-------|--------------|----------------------|
| float16/bf16 | 2B | 2 + 4 + 4 + 8 + 8 + 8 = **34B** |
| float32 | 4B | 4 + 4 + 8 + 8 + 8 = **32B** |

> **注意**：若输入 dtype 为 float32，则无需 Cast，UB 中可省去类型转换 buffer，此时 `sortBytesPerElem = sizeof(float) + sizeof(uint32_t) + PROPOSAL_SIZE + concatTmpPerElem + sortTmpPerElem`（≈ 32B）。设计方案时需根据实际 dtype 组合确定公式。`concatTmpPerElem` 和 `sortTmpPerElem` 应以 API 返回值为准，不以上述近似值为准。

**Sort 排序 UB 总空间分配**：

假设单次 UB 中处理 `elemCount` 个元素，则单次 Sort 操作需要在 UB 中同时容纳以下 buffer：

| Buffer | 大小（元素数 × 单元素字节） | 说明 |
|--------|--------------------------|------|
| 输入数据 | `elemCount × sizeof(dtype)` | 原始输入，dtype = float16/bf16/float32 等 |
| 类型转换（可选） | `elemCount × sizeof(float)` | `Cast(dtype → float)`，Sort API 要求 float 输入 |
| 索引数组 | `elemCount × sizeof(uint32_t)` | `ArithProgression` 生成的索引 |
| Concat tmp buffer | `GetConcatTmpSize(platform, elemCount, sizeof(float))` | `Concat` API 将 value+index 合成为 proposal 所需的临时 buffer，大小由 API 接口获取 |
| Sort tmp buffer | `GetSortTmpSize(platform, elemCount, sizeof(float))` | `Sort` API 排序所需的临时 buffer，大小由 API 接口获取 |
| Sort 输出 | `elemCount × PROPOSAL_SIZE` | Sort 输出，proposal 格式（value + index = 8B） |

其中 `PROPOSAL_SIZE = 8`（1 个 proposal = 4B value + 4B index），`PROPOSAL_FACTOR = 4`（proposal 大小因子，详见 API 文档）。

> **关键**：Concat tmp buffer 和 Sort tmp buffer 的大小**必须通过 API 接口获取**（`GetConcatTmpSize` / `GetSortTmpSize`），不应手动估算。与 `ubSize` 一样，它们取决于平台实现，不同芯片版本可能有差异。

### 2.3 UB 容量对 tileSize 的约束

单次 Sort 操作的 UB 约束：

```
elemCount × sortBytesPerElem ≤ ubSize
```

由此推导 **elemCount 的理论上限**，又称为 **tileSize**：

```
tileSize = elemCount_max = ubSize / sortBytesPerElem
```

tileSize 需对齐到 Sort API 要求的粒度（`TOPK_SORT_NUM = 32`）：

```
tileSize = floor(tileSize / 32) × 32
```

工程实践中 tileSize 通常取小于理论最大值的固定值（如 4096，这也是实测性能最优值），留出 UB 余量给对齐填充和 API 内部峰值开销。

---

## 3. 场景建模

### 3.1 输入变量

| 变量 | 含义 | 来源 |
|------|------|------|
| `N` | 总元素数 | 输入 shape |
| `dtype` | 数据类型 | 输入参数（float16/float32 等） |
| `K` | Top-K 值 | 输入参数 |
| `M` | 归并路数 | `MRG_SORT_ELEMENT_LEN`（通常 = 4） |
| `coreNum` | 可用核数 | 硬件规格 / tiling 计算 |
| `ubSize` | UB 大小 | `GetPlatformInfo().GetUbSize()` |

### 3.2 Pattern 划分

根据上述约束，由tileSize 和 coreNum 的关系划分出三种 Pattern：

- 当 N ≤ tileSize 时：只需要单核处理 1 个 tile，可直接使用 `Sort` API 实现排序并输出
- 当 tileSize < N ≤ tileSize × coreNum 时：每个核至多处理 1 个 tile，排序后直接进入跨核归并——即**一级归并**
- 当 N > tileSize × coreNum 时：每个核需处理多个 tile，必须先核内归并再跨核归并——即**两级归并**

---

#### 3.2.1 Pattern A：单核排序

**判定条件**：`N ≤ tileSize`

**特征**：数据量不超单次 Sort 的 UB 容量，无需 tile 切分。

**方案**：单核调用 `Sort<T, true>` 完成排序，数据流为 `GM → UB → Sort → GM`。

> **详细设计**：Pattern A 的具体方案见相关设计文档（待补充）。

---

#### 3.2.2 Pattern B：多核一级归并

**判定条件**：`tileSize < N ≤ tileSize × coreNum`

**特征**：数据需要多核分担，但每个核最多处理 1 个 tile（`ceil(N/tileSize) ≤ coreNum`）。核内无需多 tile 归并——各核排序后直接跨核归并。

**方案**：
```
各核并行排序自己的 tile → 跨核归并（将 usedCoreNum 路有序数列合并为最终结果）
```

> **详细设计**：Pattern B 的具体方案见相关设计文档（待补充）。

---

#### 3.2.3 Pattern C：多核两级归并

**判定条件**：`tileSize × coreNum < N`

**特征**：数据量超过多核单 tile 覆盖范围，每个核需处理多个 tile。必须使用**两级归并**：
- **第一级**：各核内部对自己的多个 tile 进行归并，输出 1 个有序数列
- **第二级**：将 coreNum 个有序数列跨核归并，最终输出 Top-K 结果

**方案概要**：
```
各核并行排序多个 tile → 各核独立多 tile 归并(第一级) → 跨核归并(第二级) → 最终输出 K 个元素到 GM
```

**与 Pattern B 的关键区别**：新增了各核独立的多 tile 归并环节。
**关键设计：减少 SyncAll 开销**：为什么不能将各核的所有 tile 直接进行跨核归并？跨核归并每一轮都需要 `SyncAll` 等待所有核完成，而核内归并完全不需要。若跳过第一级直接将 `totalTiles` 个 tile 跨核归并，归并轮次 = `ceil(log_M(totalTiles))`，每轮一次 SyncAll，开销不可接受。先在各核内部归并可将待归并路数从 `totalTiles` 降至 `coreNum`，大幅减少跨核归并轮次和 SyncAll 次数。

**→ 详细方案设计见 [alg-two-level-mrgsort.md](alg-two-level-mrgsort.md)**

---

## 4. 决策树

```
给定: N(总元素数), dtype(数据类型), coreNum(可用核数), K(Top-K值), ubSize(UB大小)

Step 1: 计算 tileSize
  tileSize = ubSize / sortBytesPerElem，对齐到 32，工程建议取值 4096

Step 2: 比较 N 与 tileSize
  ├─ N ≤ tileSize
  │   → Pattern A: 单核排序
  │     数据不超 UB 容量，单核 Sort 一次完成
  │
  └─ N > tileSize
      └─ 比较 N 与 tileSize × coreNum
          ├─ N ≤ tileSize × coreNum
          │   → Pattern B: 多核一级归并
          │     每核 ≤1 个 tile，无需核内多 tile 归并
          │
          └─ N > tileSize × coreNum
              → Pattern C: 多核两级归并
                每核 >1 个 tile，需两级归并
                → 详见 [alg-two-level-mrgsort.md](alg-two-level-mrgsort.md)
```

**数值示例**（float16, tileSize=4096, coreNum=20）：

| N | tileSize×coreNum 阈值 | 判定 | 方案 |
|---|----------------------|------|------|
| 2,000 | 81,920 | N ≤ 4096 | Pattern A |
| 50,000 | 81,920 | 4096 < N ≤ 81,920 | Pattern B |
| 500,000 | 81,920 | N > 81,920 | Pattern C → [alg-two-level-mrgsort.md](alg-two-level-mrgsort.md) |

---

## 子场景参考

| 主题 | 文档 |
|------|------|
| Pattern C 详细方案设计 | [alg-two-level-mrgsort.md](alg-two-level-mrgsort.md) |
