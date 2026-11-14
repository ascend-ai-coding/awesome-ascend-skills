# Reduction 族 Tiling 算法路由


> Reduction 族（ReduceSum, Softmax, LayerNorm, ArgMax, EuclideanNorm）Tiling 算法入口。默认使用 fallback 兜底算法。

---

## 1. 已注册算法

| 优先级 | 算法 | 选择条件 | 入口文件 |
|--------|------|---------|---------|
| 0（兜底） | 通用五模板 | 目标架构/能力匹配后的兜底 | [fallback/index.md](fallback/index.md) → [fallback/tiling-flow.md](fallback/tiling-flow.md) |
| 1（前置窄门控） | 多轴交错连续载入（tail-R fast / tail-A fast-RARA） | dtype∈{fp32,fp16,bf16} + 合轴后仍多个交替 A/R（偶数轴≥6 ARAR...AR / 奇数轴≥7 ARAR...A）+ 内块连续 + 32B 对齐 + burst/内块行阈值 | [fallback/tiling-flow.md §3.1/§11–13](fallback/tiling-flow.md#31-多轴交错前置判定step-05)（案例 [example/euclidean_norm](fallback/example/euclidean_norm/experience.md)） |

> ⚠️ 路由命中后**必须深入 fallback/ 子目录**读取 `tiling-flow.md`（五模板决策树 + UB 预算公式）、`tiling-fields.md`（TilingData 字段语义）、`example/<算子名>/experience.md`（算子实践案例）。禁止仅读本文件的表格后停止。

---

## 2. 路由规则

```
算子类型匹配:
  所有 Reduction 算子 → fallback/（通用五模板算法）
    → 必读 fallback/index.md（算法概览与文档索引）
    → 必读 fallback/tiling-flow.md（五模板决策树 + UB 预算公式）
    → 必读 fallback/tiling-fields.md（TilingData 字段语义）
    → 必读 fallback/example/<算子名>/experience.md（如 Softmax → example/softmax/experience.md）
  后续贡献的专用算法 → 按本表注册条件匹配
```

---

## 3. 贡献新算法

在 `reduction/` 下创建新目录，在此文件的路由表中注册，注明选择条件和优先级。
