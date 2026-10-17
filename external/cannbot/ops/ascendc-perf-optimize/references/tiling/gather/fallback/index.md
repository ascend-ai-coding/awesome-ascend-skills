# Gather 族 — 通用 Tiling 算法概览

> 代际限定：本文 RegBase / Reg / MicroAPI / VF 相关实现仅适用于 arch35（A5 / DAV_3510）；通用 Tiling 建模不受此限制。

> Gather 族算子的六模板 Tiling 算法。所有 Gather 算子共享同一套模板语言。

## 文档索引

| 文档 | 内容 |
|------|------|
| [tiling-flow.md](tiling-flow.md) | 六模板决策树 + 合轴公式 + UB 预算公式 + 多核切分 |
| [tiling-fields.md](tiling-fields.md) | TilingData 字段语义（规划中） |

## 模板一览

| 模板 | 触发条件 | 核心策略 |
|------|---------|---------|
| SIMD 两维 | 2D + innerSize ≥ SIMD 阈值 | gatherSize 分核，innerSize 行向量化 |
| SIMT 两维 | 2D（兜底） | SIMT 2048 线程标量并行 |
| SIMD 末轴 Gather | innerSize==1 + GD 全载 UB | Gather 向量化 |
| GA 全载 | inner 小 + GA 全载 UB | Gather / VF gather |
| Full-Load | GD×inner 全载 UB + 2D | x 全载，逐索引 copy |
| SIMD 行拷贝 | inner 大 | DataCopyPad 逐行 + index 缓存 |
| SIMT 标量 | 兜底 | SIMT 2048 线程 + fast-div |

## 典型场景

| 场景 | 推荐模板 | 原因 |
|------|---------|------|
| torch.gather dim=0, 2D 大张量 | SIMT 标量 | 散点读无连续性，线程并行最优 |
| Embedding lookup (hidden 大) | SIMD 行拷贝 | 每 token 取 hidden 个连续元素 |
| 末轴 gather (dim=last) | SIMD 末轴 Gather | Gather 一指令 gather 多索引 |
| 小 hidden + 大 vocab | GA 全载 | vocab×hidden 全载 UB，Gather |
