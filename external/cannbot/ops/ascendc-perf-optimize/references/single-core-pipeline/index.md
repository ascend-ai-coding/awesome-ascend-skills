# 单核流水优化策略

> 核内流水分析，通过仿真图或者 profiling 数据判定 bound 类型，按 bound 展开优化策略。
> 结合仿真图或者 profiling 数据诊断瓶颈，输出优化方案并回修 tiling 参数。

---

## Bound 判定流程

```
给定：仿真图 或者 profiling 数据（含 MTE2/CUBE/Vector 利用率、带宽、指令发射率）

Step 0 — MC² 前置检查（通算融合算子必须）：
  若算子为 MC² 通算融合（存在 CrossCoreWaitFlag 挂在 MTE2 流水线上），
  须先用隔离测试判定 MTE2 ratio 是否被通信等待污染
  （参考 [bound_diagnosis.md](../comm-compute/bound_diagnosis.md)「MC² 场景 MTE2 污染判定」）。
  ├─ mte2_polluted = true → 跳过 memory.md，路由到 [comm-compute/](../comm-compute/) 通信掩盖策略
  └─ mte2_polluted = false → 继续以下 Step 1 正常路由

Step 1 — 初步判定：
  ├─  scalar耗时占比高 或者 计算量小 → Scalar Bound / 小 case → scalar.md
  ├─ MTE2/访存单元利用率高 + 带宽为主要限制 → 访存 Bound → memory.md
  ├─ Vector 单元利用率高 + 向量指令为主要限制 → Vec Bound → vec.md
  └─ 各单元利用率均不高 → 无 Bound → no-bound.md

Step 2 — 确认与细分：
  根据仿真图时间轴确认判定结论，排除误判（如访存等计算 vs 计算等访存）。
```

## Bound 路由

| Bound 类型 | 判定条件 | 优化策略文档 |
|-----------|---------|------------|
| Scalar Bound / 小 case | scalar占比最高或者计算量极小或 shape 很小 | [scalar.md](scalar.md) |
| 访存 Bound | MTE2/AIC 带宽为主导瓶颈 | [memory.md](memory.md) |
| Vec Bound | Vector 指令为主导瓶颈 | [vec.md](vec.md) |
| 无 Bound | 无明显瓶颈 | [no-bound.md](no-bound.md) |
