# `determinism.accumulation_order` 语义定义

> `spec.yaml` 中 `determinism.accumulation_order` + `bitwise_reproducible` 的规范来源。
> 枚举值定义于 spec 层（非 regbase 范式原生术语），语义映射见本文档。
> 枚举见 `schemas/op-spec.json`: `["stable_in_axis", "tree", "sequential", "none"]`。

## 0. 三个维度

| 维度 | 回答的问题 | 控制字段 | 备注 |
|------|-----------|---------|------|
| **A. 累加顺序** | 跨核切分后 partial 按何种顺序合并？ | `accumulation_order` | |
| **B. kernel 自身可复现性** | 同输入同设备多次执行 kernel，结果是否 bitwise 一致？ | `bitwise_reproducible` | 当前 scope 内可由 A 推导（stable_in_axis/sequential/none → `true`） |
| **C. 数值容差** | 浮点累加顺序位差用多大容差吸收？ | `numerical_tolerance.per_dtype` | |

## 1. 枚举值

### `stable_in_axis` —— 归约类

归约轴（R）可跨核切分（regbase `reduce-split-strategy.md:12,32`：A 并行度不足时借 R 轴分核）；
跨核时各核沿局部 R 段做 partial reduce 写 workspace，Phase 2 串联合并（`reduce-design-overview.md:229`）。
范式 §3.3.11 指引：固定切分 + 固定累加顺序 → 同设备同输入多次执行 bitwise 可复现。
具体切分策略（优先 A 轴 / A 并行度不足时借 R 轴走 Group 两阶段合并）由 regbase 范式按核数与并行度决定，见 `reduce-split-strategy.md` §零/§二，**不由本字段规定**。
**不规定核内累加算法**（由 regbase `needs_bisection` 决定，见 §2）。

适用：Reduction / ReductionComposite（sum/mean/max/min/prod 均适用）；ArgReduce 待对照其范式确认。

### `tree` —— Contraction 类

累加轴（如 matmul K 轴）可跨核切分，partial 按固定树形顺序合并。详见 Contraction 范式文档。

### `sequential` —— Recurrence 类

累加严格沿元素序列顺序，存在 inter-element 依赖，**禁止并行化累加轴**；batch/其他维度可自由切。
适用：Recurrence（cumsum 等）。

### `none` —— 无累加约束

算子不涉及累加/规约。适用：Elementwise / Broadcast / LayoutTransform 等。

## 2. 与 regbase 的关系

`stable_in_axis`（维度 A）与 `needs_bisection`（regbase 实现层）**正交**：前者是 spec 层对累加顺序确定性的声明，后者决定核内累加算法（sum/mean→二分树，max/min/prod→线性，见 `patterns.md:83`），禁止强绑定。regbase `patterns.md §3.3.11`"固定切分 + 固定累加顺序 → bitwise 可复现"仅针对 `needs_bisection=true`（sum/mean）。

## 3. 快速决策表

| 算子类别 | accumulation_order | bitwise_reproducible | numerical_tolerance |
|----------|-------------------|----------------------|---------------------|
| Reduction sum/mean | `stable_in_axis` | `true` | 浮点非 0 容差 |
| Reduction max/min/prod | `stable_in_axis` | `true` | max/min 0 容差；prod 浮点非 0 |
| Recurrence (cumsum) | `sequential` | `true` | 顺序与 golden 一致→可 0 容差；否则非 0 |
| Elementwise | `none` | `true` | 0 容差 |

## 4. 常见错误

1. R 轴跨核切分（Group 模板）时 partial 合并顺序未固定 → **HIGH**。
2. 整数 dtype reduce（sum/mean/prod）不可假定 0 容差——regbase kernel 内部 Cast 到 fp32 累加，fp32 域结合律不成立 → **HIGH**（假定 0 容差而无依据）。
   - `tolerance_defaults.yaml` 对 int32/int64 的默认值是 `bitwise_equal` + 0，`generate_spec.py` 对 reduction 类算子（Reduction / ReductionComposite）自动覆盖为 fp32 容差。
   - **max/min 除外**（精确比较、不累加，保持 `bitwise_equal` + 0）。
   - ArgReduce 输出是索引（离散精确值，不受 fp32 累加影响），保留 `bitwise_equal` + 0。
   - 非生成器产出的 spec 须手动确保。
3. 浮点 dtype reduce 用 0 容差 → **HIGH**。
4. `stable_in_axis` 与 `needs_bisection=true` 强绑定（对 max/min/prod 误声明"二分缓存树"）→ **MED**。
5. §9.10 泛写"固定切分"未引用具体 `accumulation_order` 值 → **MED**。
