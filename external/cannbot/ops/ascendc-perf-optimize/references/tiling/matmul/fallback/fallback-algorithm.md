# MatMul 族 — 兜底算法（官方参考）

> MatMul 族所有变体（a16w16 / mxfp4 / mxfp8 / batch_matmul / group_matmul）共享的官方参考 Tiling 算法。
>
> 按 **dtype / Scale / 特有维度** 找对应变体，再按 **Shape 条件** 选择策略（SWAT / FullLoad / StreamK）。

---

## 1. 路由流程

```
给定：算子名 + Shape + dtype

Step 0 — 变体路由：
  ├─ dtype = FP16/BF16, 无 Scale, 无 batch/group → a16w16（基线）
  ├─ dtype = FP4/FP8, 有 per-group Scale         → mxfp4/8
  ├─ 有 batch 维 (B, M, K) × (B, K, N)           → batch_matmul
  ├─ 有 group 维 (g groups, M_i 不等)             → group_matmul
  └─ 未匹配 → 找最相似的已知变体，标注差异

Step 1 — 策略选择：
  ├─ StreamK 判定通过   → StreamK（K 轴切分多核）
  ├─ FullLoad 判定通过  → FullLoad（A 或 B 驻留 L1）
  └─ 否则              → SWAT（默认回退）
```

---

## 2. 算法策略

| 策略 | 适用条件 | 核心思想 |
|------|---------|---------|
| **SWAT**（默认） | 所有 Shape | K 轴流式迭代，七步推导 |
| **FullLoad** | A 或 B 能全量驻留 L1，对侧 T ≥ 2 | 小矩阵一次载入驻留 |
| **StreamK** | K ≥ 32768 且 B=1 且无 group | K 轴切分给多核并行 |

---

## 3. 索引

| 文档 | 内容 |
|------|------|
| [tiling-flow.md](tiling-flow.md) | 通用 Tiling 七步推导 + SWAT/FullLoad/StreamK 算法细节 |
| [tiling-fields.md](tiling-fields.md) | 跨变体 TilingData 字段语义对照表 |
| [tiling-variants.md](tiling-variants.md) | 各变体（a16w16 / mxfp4 / batch / group）差异说明 |
