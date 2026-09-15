# Sort 族 — Tiling 流程（M-way Merge Sort）

> Sort 族 Top-K 排序算子 Tiling 推导流程。基于 M-way 归并排序的分治策略。

---

## 1. 决策树

```
给定: N, K, dtype, ubSize, coreNum

Step 1: 计算 tileSize
  sortBytesPerElem = sizeof(dtype) + sizeof(float) + sizeof(uint32)
                     + PROPOSAL_SIZE + concatTmpPerElem + sortTmpPerElem
  tileSize = min(ubSize / sortBytesPerElem, 4096)  对齐到32

Step 2: 比较 N 与 tileSize
  ├─ N ≤ tileSize
  │   → Pattern A: 单核排序
  │     直接 Sort API 完成
  │
  └─ N > tileSize
      └─ 比较 N 与 tileSize × coreNum
          ├─ N ≤ tileSize × coreNum
          │   → Pattern B: 多核一级归并
          │     每核≤1个tile, 无需核内多tile归并
          │
          └─ N > tileSize × coreNum
              → Pattern C: 多核两级归并
                每核>1个tile, 需两阶段归并
```

---

## 2. Pattern C: 四阶段架构

```
Phase 1 — tile排序 (全核并行):
  各核将GM数据读入UB → Cast+ArithProgression+Concat → Sort → workspace[0]

Phase 2 — 核内多tile归并 (全核并行, 无同步):
  各核独立将S_c个有序tile通过M-way归并合并为1个有序数列
  循环条件: while listNum != 1

Phase 3 — 跨核归并 (递减核数, 每轮SyncAll):
  将coreNum路逐步归并至≤M路
  循环条件: while listNum > M

Phase 4 — 最终归并+输出 (仅Core 0):
  将≤M路归并为最终结果 → Extract分离value/index → GM
```

## 3. Tile 切分公式

```
totalTiles      = CeilDiv(N, tileSize)
frontCoreTiles  = CeilDiv(totalTiles, coreNum)
usedCore        = CeilDiv(totalTiles, frontCoreTiles)
lastCoreTiles   = totalTiles - (usedCore-1) × frontCoreTiles
lastTileSize    = N - tileSize × (totalTiles-1)
elementsPerCore = frontCoreTiles × tileSize
```

## 4. UB 预算 (分Phase)

| Phase | Buffer数 | bytes/elem | onceMaxElements |
|-------|:------:|-----------|----------------|
| Phase 1 (Sort) | 6 | ~34B (fp16) | = tileSize = 4096 |
| Phase 2/3 (Merge) | 2 | 64B | `(ubSize/64)/32×32` |
| Phase 4 (Output) | 5 | 104B (fp16) | `(ubSize/BPE)/32×32` |

## 5. 归并轮次

```
Phase 2 归并轮次 = ceil(log_M(S_c))
Phase 3 归并轮次 = ceil(log_M(coreNum)) - 1  (最后一轮由Phase 4完成)
Phase 4          = 1轮 ≤M-way归并
总SyncAll次数 = ceil(log_M(coreNum)) + 1
```

## 6. 截断优化 (Top-K)

当 `currentElements × M ≥ K` 时启用截断:
- 归并输出≥K时, 后续只处理前K个有效元素
- 截断标志 `truncationFlag` 跨Phase 2/3持久化
- 首次触发时读入完整数据, 下一轮开始读截断长度

## 7. Workspace 双缓存

```
wsPerCore = E_c × 8B × 2  (32B对齐, 双缓存)
workspace[0]: 写缓存
workspace[1]: 读缓存
每轮归并后 workSpaceFlag = 1 - workSpaceFlag (交替)
```
