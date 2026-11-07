# Sort / TopK 族策略选型指南

> 算子族：`sort` / `topk`
> 适用算子：`Sort`, `TopK`, `Argsort`, `NMS` 等含排序/选择的算子
> 关联设计：[knowledge-strategy-architecture-v3.2](../../../../../docs/design/knowledge-strategy-architecture-v3.2.md) §3.2

## 选型决策树

```
总元素数 N 与 tileSize、coreNum 的关系？
├─ N ≤ tileSize
│   → Pattern A：单核排序（Sort API 一次完成）
├─ tileSize < N ≤ tileSize × coreNum
│   → Pattern B：多核一级归并（每核 ≤1 tile，排序后直接跨核归并）
└─ N > tileSize × coreNum
    → Pattern C：多核两级归并（先核内归并，再跨核归并）
```

## 族特定策略

| 策略 | Tier | Playbook | 触发场景 | 核心手段 |
|---|---|---|---|---|
| **P96** 硬件 Sort/MrgSort | L1 | ✅ | 任意含排序/选择的算子 | `Sort<T,true>` + `MrgSort` 替代标量排序 |
| **P111** 两级归并 Tiling | L1 | ✅ | Top-K 大数据量（Pattern C） | tileSize=4096、四阶段、workspace 双缓存、截断 |
| **P1** 双缓冲 | L0 | ✅ | 排序 tile 循环 | CopyIn / Sort / CopyOut 流水化 |
| **P7** 32B 对齐 | L0 | — | 尾块、非对齐输入 | DataCopyPad、tileSize 对齐到 32 |

---

## Ascend C Sort Tiling 系统化设计

> 来源：插件本地 [`tiling-design/sort/patterns.md`](../tiling-design/sort/patterns.md)、[`alg-two-level-mrgsort.md`](../tiling-design/sort/alg-two-level-mrgsort.md)
>
> 本章节沉淀 Sort / TopK 族在 Ascend C 上的 Tiling 与归并设计方法论。

### 1. 归并排序原理

- **分治**：递归拆分到长度为 1，再两两归并。
- **M-way 归并**：`MrgSort` API 一次调用可合并 `MRG_SORT_ELEMENT_LEN`（通常 M=4）路有序数列。
- **大数据集策略**：按 UB 容量切 tile，各核并行排序 tile，再逐级归并。

### 2. UB 容量模型

```cpp
ubSize = GetPlatformInfo().GetUbSize();
tileSize = ubSize / sortBytesPerElem;
tileSize = floor(tileSize / 32) * 32;   // 对齐到 TOPK_SORT_NUM=32
```

**每元素 UB 占用（float16 近似）**：

| Buffer | 大小 | 说明 |
|---|---|---|
| 输入数据 | `tileSize × sizeof(dtype)` | 原始输入 |
| 类型转换（可选） | `tileSize × sizeof(float)` | `Cast(dtype→float)` |
| 索引数组 | `tileSize × sizeof(uint32_t)` | `ArithProgression` 生成 |
| Concat tmp | `GetConcatTmpSize(...)` | 必须由 API 获取 |
| Sort tmp | `GetSortTmpSize(...)` | 必须由 API 获取 |
| Sort 输出 | `tileSize × PROPOSAL_SIZE` | proposal 格式（4B value + 4B index）|

> 工程建议：`tileSize = 4096`（实测通常最优）。

### 3. Pattern 判定

| Pattern | 条件 | 方案 |
|---|---|---|
| A | `N ≤ tileSize` | 单核 `Sort<T,true>` 一次完成 |
| B | `tileSize < N ≤ tileSize × coreNum` | 各核并行排序 → 跨核归并 |
| C | `N > tileSize × coreNum` | 各核排序多个 tile → 核内归并 → 跨核归并 |

### 4. Pattern C：多核两级归并

#### 4.1 变量定义

| 符号 | 含义 |
|---|---|
| `S_c` | 单核 tile 数 = `ceil(ceil(N/tileSize)/coreNum)` |
| `E_c` | 单核元素数 = `S_c × tileSize` |
| `M` | 归并路数 = 4 |

#### 4.2 四阶段架构

| Phase | 职责 | 核数参与 | 同步 |
|---|---|---|---|
| Phase 1 | tile 排序 | 全核 | 无 |
| Phase 2 | 核内多 tile 归并 | 全核 | Phase 2 结束后 1 次 SyncAll |
| Phase 3 | 跨核归并 | 递减 | 每轮后 SyncAll |
| Phase 4 | Core 0 最终归并 + Extract 输出 | 1 核 | Core 0 完成后 1 次 SyncAll |

```cpp
__aicore__ inline void Process()
{
    // Phase 1
    for (uint32_t t = 0; t < myTiles; t++) {
        SortInSingleCore(curTileSize, globalOffset);
    }
    // Phase 2/3
    pipe_.Reset();
    InitMergeBuffers();
    Phase2IntraCoreMerge();
    SyncAll();
    Phase3InterCoreMerge();
    // Phase 4
    pipe_.Reset();
    InitOutputBuffers();
    Phase4FinalMergeAndOutput();
}
```

#### 4.3 截断逻辑

Top-K 场景下，当 `currentElements × M ≥ K` 时，后续归并只需读取前 K 个元素。`truncationFlag` 在每轮归并**执行后**设置，影响下一轮读入长度。

### 5. Tiling 切分公式

```cpp
totalTiles      = ceil(N / tileSize);
frontCoreTiles  = ceil(totalTiles / coreNum);
usedCore        = ceil(totalTiles / frontCoreTiles);
lastCoreTiles   = totalTiles - (usedCore - 1) * frontCoreTiles;
lastTileSize    = N - tileSize * (totalTiles - 1);
elementsPerCore = frontCoreTiles * tileSize;
```

### 6. UB 批次计算

```cpp
// Phase 2/3 归并：2 个队列 × M 路 × 8B
mergeBytesPerElem      = 2 * M * 8;   // 64B
onceMaxElementsMerge   = (ubSize / mergeBytesPerElem) / 32 * 32;

// Phase 4 输出：额外 Extract + Cast buffer
outputBytesPerElem     = 2 * M * 8 + M * (4 + 4 + sizeof(dtype)); // float16 ≈ 104B
onceMaxElementsOutput  = (ubSize / outputBytesPerElem) / 32 * 32;
```

### 7. Workspace 双缓存

```cpp
wsPerCoreBytes = (E_c * 8B * 2 + 31) / 32 * 32; // 双缓存，32B 对齐
wsTotalFloats  = usedCore * wsPerCoreBytes / sizeof(float);
wsHalfFloats   = wsTotalFloats / 2;
workspace[0]   = workSpace[0 .. wsHalfFloats-1];
workspace[1]   = workSpace[wsHalfFloats .. 2*wsHalfFloats-1];
```

### 8. 设计检查清单

- [ ] 是否根据 `N` 正确选择 Pattern A/B/C？
- [ ] `tileSize` 是否对齐到 32 且工程建议取 4096？
- [ ] UB 中 Concat/Sort tmp buffer 是否通过 `GetConcatTmpSize` / `GetSortTmpSize` 获取？
- [ ] Phase 2 循环条件是否为 `while (listNum != 1)`？
- [ ] Phase 3 循环条件是否为 `while (listNum > M)`？
- [ ] `truncationFlag` 是否在每轮归并**后**设置？
- [ ] 每个归并循环是否调用 `UpdateSortInfo()` 防止死循环？
- [ ] Workspace 偏移是否使用 `GetCoreWsOffset(i) = i × E_c × 2`？
