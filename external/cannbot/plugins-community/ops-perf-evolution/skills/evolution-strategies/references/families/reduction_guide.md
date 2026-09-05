# Reduction / Normalization 族策略选型指南

> 算子族：`reduction` / `normalization` / `softmax`
> 适用算子：`ReduceSum`, `ReduceMax`, `ReduceMean`, `LayerNorm`, `RMSNorm`, `Softmax`, `LogSoftmax`, `GroupNorm` 等
> 关联设计：[knowledge-strategy-architecture-v3.2](../../../../../docs/design/knowledge-strategy-architecture-v3.2.md) §3.2

## 选型决策树

```
归约维度大小？
├── 小维 (≤ 128) → 单核内 reduce + 树状归约 (P68)
├── 中维 (128 - 2048)
│   ├── 沿 inner 轴 → BlockReduceMax/Sum (P55) 单指令归约
│   └── 沿 outer 轴 → 多核切分 + workspace 归约 (P40)
└── 大维 (> 2048) → 两轮 reduce (Welford 算法 A2) + double buffer (P1)

是否数值稳定性敏感？
├── 是 (Softmax / LogSoftmax) → A5 max-trick + A1 FP32 中间计算
└── 否 (ReduceSum / Mean) → 直接 reduce 即可
```

## 族特定策略

> ⚠️ **当前状态**：根据 INDEX.json 筛查，reduction 族当前**没有专属 op_families 标签**。所有 reduction 类策略都打成了 `omni`（通用）或 `softmax/attention`（更窄）。
>
> **运维 TODO**（不在本 Phase 范围）：扫描以下策略，把 `omni` 中明显偏 reduction 类型的（如 P68 低延迟归约）改打 `reduction` 标签：
> - P68 低延迟归约
> - A1 FP32 中间计算
> - A2 Welford 算法（如果有对应策略卡）

## 推荐组合

| 组合 | 适用 | 说明 |
|---|---|---|
| **P1 + P5 + P10** | 标准 RMSNorm / LayerNorm | 双缓冲 + 流水同步 + 向量化拷贝（最基础） |
| **A5 + A1 + P5** | Softmax / LogSoftmax | 数值稳定基础组合（max-trick + FP32 中间 + 同步） |
| **P55 + P67** | inner 轴归约的 attention/norm | BlockReduceMax 单指令替代 + Vector Counter |
| **P68 + Welford** | 大维归约的精度敏感算子 | 树状归约 + 两轮算法（避免单次溢出） |

## 涉及的通用策略（omni）

参见 attention_guide.md 中的 P1/P5/P10 等 L0 通用策略——这些对 reduction 族同样适用。

## Playbook 状态

- ✅ P1, P5, P10 已有 Playbook（reduction 族最常用三件套）
- ❌ A1, A5, P68 待 Phase C 补
