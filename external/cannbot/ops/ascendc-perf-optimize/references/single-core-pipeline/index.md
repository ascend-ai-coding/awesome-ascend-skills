# 单核流水优化策略

> 通用诊断适用于 arch22 / arch35，具体指标与卡片按目标能力选用；RegBase / Reg / MicroAPI / VF / RVEC 相关内容仅适用于 arch35（A5），示例阈值不作为跨代默认值。


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
  └─ 各单元利用率均不高 → 算**重叠因子**（算法见 metrics.md §2.1）：
       Σ(各泳道耗时) / aiv_time，与理想上界 Σ/max 比。
       达成率低（如 51%）= 流水并行不足 → 无 Bound → no-bound.md

Step 2 — 确认与细分：
  根据仿真图时间轴确认判定结论，排除误判（如访存等计算 vs 计算等访存）。

Step 3 — 下钻（判定成立后，需要定位"谁挡住谁"时）：
  msprof 只回答"重叠够不够"，回答不了"谁挡住谁"——各泳道 ratio 是时间占比，
  不含因果。起 CANN Simulator 采指令级 trace（`/ops-simulator`，仅支持 950），
  看空窗期间各泳道占用来定位阻塞源。命令与读法见 techniques/nobound-02。
```

## Bound 路由

| Bound 类型 | 判定条件 | 诊断页 |
|-----------|---------|------------|
| Scalar Bound / 小 case | scalar占比最高或者计算量极小或 shape 很小 | [scalar.md](scalar.md) |
| 访存 Bound | MTE2/AIC 带宽为主导瓶颈 | [memory.md](memory.md) |
| Vec Bound | Vector 指令为主导瓶颈 | [vec.md](vec.md) |
| 无 Bound | 无明显瓶颈 | [no-bound.md](no-bound.md) |

## 两级知识结构

本目录按两级组织，避免诊断与实施耦合：

- **L1 路由 / 诊断**：本 `index.md`（bound 判定）→ 诊断页（`vec/scalar/memory/no-bound.md`，只判定 + 特征）。
- **L2 技术卡片**：`techniques/`（每张卡自足：何时用 → 原理 → before/after → 性能数据 → 红线 → 关联）。
- **口径**：采集与指标统一见 [metrics.md](metrics.md)（msprof）。

## bound → 技术卡片映射

诊断出 bound 后，进对应诊断页看细分特征，再到 `techniques/` 取卡片实施：

| Bound | 诊断页 | 候选卡片 |
| --- | --- | --- |
| VEC | vec.md | vec-01,vec-02,vec-03,vec-04,vec-05,vec-06,vec-07,vec-08,vec-09,vec-10,vec-11,vec-12,vec-13,vec-14 |
| Scalar/小 case | scalar.md | vec-04,vec-09,vec-10,scalar-01 |
| 访存 | memory.md | vec-01,vec-05,vec-06,mem-01,mem-02,mem-03,mem-04,mem-05,mem-06,mem-07 |
| 无 bound | no-bound.md | vec-01,vec-08,vec-10,nobound-01,nobound-02 |

> 卡片清单与状态见 [techniques/INDEX.md](techniques/INDEX.md)。
