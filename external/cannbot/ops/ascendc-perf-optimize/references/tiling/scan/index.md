# Scan / Cum 族 Tiling 算法路由


> Scan 族（Cumsum, Cummin, Cummax, Cumprod, LogCumsumExp）前缀扫描算子 Tiling 算法入口。默认使用 fallback 兜底算法。

---

## 1. 已注册算法

| 优先级 | 算法 | 选择条件 | 入口文件 |
|--------|------|---------|---------|
| 0（兜底） | 通用扫描模板（流式 / oneway·twoway·ub·core 三层组合） | 目标架构/能力匹配后的兜底 | [fallback/index.md](fallback/index.md) → [fallback/tiling-flow.md](fallback/tiling-flow.md) |

> ⚠️ 路由命中后**必须深入 fallback/ 子目录**读取 `tiling-flow.md`（选型决策树 + UB 预算 + 多核）、`tiling-fields.md`（TilingData 字段语义）。禁止仅读本文件表格后停止。

---

## 2. 路由规则

```
算子类型匹配:
  所有 Scan/Cum 算子 → fallback/（通用扫描模板算法）
    → 必读 fallback/index.md（算法概览与文档索引）
    → 必读 fallback/tiling-flow.md（选型决策树 + UB 预算 + 保序进位多核）
    → 必读 fallback/tiling-fields.md（TilingData 字段语义）
  后续贡献的专用算法 → 按本表注册条件匹配
```

---

## 3. 贡献新算法

在 `scan/` 下创建新目录，在此文件路由表中注册，注明选择条件和优先级。
