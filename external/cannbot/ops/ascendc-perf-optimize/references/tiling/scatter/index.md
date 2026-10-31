# Scatter 族 Tiling 算法路由


> Scatter 族（Scatter、ScatterAdd、ScatterReduce，对应 CANN ScatterElementsV2 / ONNX ScatterElements）Tiling 算法入口。默认使用 fallback 兜底算法。
>
> **不含** ScatterNd / ScatterNdUpdate / ScatterUpdate / ScatterList —— 这些是多维联合索引或整行替换，寻址与一致性模型不同，本族的核心不变量（冲突只发生在纤维内）对它们不成立，不要套本目录的决策树。

---

## 1. 已注册算法

| 优先级 | 算法 | 选择条件 | 入口文件 |
|--------|------|---------|---------|
| 0（兜底） | 通用六路径 | 目标架构/能力匹配后的兜底 | [fallback/index.md](fallback/index.md) → [fallback/tiling-flow.md](fallback/tiling-flow.md) |

> ⚠️ 路由命中后**必须深入 fallback/ 子目录**读取 `tiling-flow.md`（六路径决策树 + 线程数/block 数推导 + workspace 预算）、`tiling-fields.md`（TilingData 字段语义）。禁止仅读本文件的表格后停止。

---

## 2. 路由规则

```
算子类型匹配:
  所有按轴 dim + 单值 indices 的 Scatter 算子 → fallback/（通用六路径算法）
    → 必读 fallback/index.md（算法概览与文档索引）
    → 必读 fallback/tiling-flow.md（六路径决策树 + 线程数/block 数推导 + workspace 预算）
  后续贡献的专用算法 → 按本表注册条件匹配
```

---

## 3. 与 Gather 族的关系

两族同属"访存地址由运行期数据决定"，但**可优化空间不对称**，Tiling 的决策对象不同：

| | Gather（读侧聚集） | Scatter（写侧散射） |
|---|---|---|
| 不规则在哪 | 读 | 写 |
| 能否变规则 | **能**：物理 / kernel 内 Transpose、full-load 把不规则读变连续 | **不能**：写地址由 index 值决定 |
| Tiling 决策什么 | 要不要转、转得下吗、转了划算吗（能力门控 + UB 容量 + 搬运成本） | **并行划分**（纤维 vs 元素）与**冲突消解**（免原子 / 原子 / 仲裁），以及由此决定的线程数、block 数、workspace |

读写两侧都不规则的算子（如 `index_put_` 的部分形态）两族都要看。

---

## 4. 贡献新算法

在 `scatter/` 下创建新目录，在此文件的路由表中注册，注明选择条件和优先级。
