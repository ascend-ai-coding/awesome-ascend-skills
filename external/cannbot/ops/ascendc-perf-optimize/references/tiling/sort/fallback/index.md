# Sort 排序类 — 兜底算法（M-way Merge Sort）

> Sort 族 Top-K 排序算子的官方参考 Tiling 算法。
>
> 基于 M-way 归并排序，按数据规模分为三个 Pattern：**A**（单核排序）/ **B**（多核一级归并）/ **C**（多核两级归并）。

---

## 1. 路由流程

```
给定: N(总元素数), K(Top-K值), dtype, ubSize, coreNum

Step 1 — 计算 tileSize:
  tileSize = min(ubSize / sortBytesPerElem, 4096), 对齐到32

Step 2 — Pattern 判定:
  ├─ N ≤ tileSize → Pattern A: 单核 Sort 一次完成
  ├─ N ≤ tileSize × coreNum → Pattern B: 多核一级归并
  └─ N > tileSize × coreNum → Pattern C: 多核两级归并
```

## 2. Pattern C 四阶段架构

| Phase | 职责 | 核参与 | 数据流 |
|-------|------|:----:|--------|
| Phase 1 | 各核并行 tile 排序 | 全核 | GM→UB→Sort→workspace |
| Phase 2 | 核内多 tile 归并 | 全核 | workspace→UB→MrgSort→workspace |
| Phase 3 | 跨核归并 (路数→≤M) | 递减 | workspace→UB→MrgSort→workspace |
| Phase 4 | Core0 最终归并+输出 | 1核 | workspace→UB→Extract→GM |

## 3. 索引

| 文档 | 内容 |
|------|------|
| [sort_tiling.py](script/sort_tiling.py) | Python 参考实现（Pattern A/B/C 路由 + 两级归并） |
| [tiling-flow.md](tiling-flow.md) | Tiling 推导流程 + 四阶段架构 + 截断优化 |
| [tiling-fields.md](tiling-fields.md) | TilingData 字段语义 + 约束检查 |
