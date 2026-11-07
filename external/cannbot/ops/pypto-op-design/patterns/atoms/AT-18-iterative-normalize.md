---
type: pattern/atom
title: Iterative Normalize (Sinkhorn 族)
description: 迭代归一化模板。最常见的实例是 Sinkhorn（行/列交替归一化趋于双随机矩阵），但本模板同样适用于任意"固定点迭代 + 多轴归一化"算法。
tags:
- iterative-normalize
flow_pattern:
- V
examples:
- mhc_sinkhorn（泛化：Greenkhorn / Power Iter / Sym-Norm）
---

## AT-18: Iterative Normalize (Sinkhorn 族)

**描述**: 迭代归一化模板。最常见的实例是 Sinkhorn（行/列交替归一化趋于双随机矩阵），但本模板同样适用于任意"固定点迭代 + 多轴归一化"算法。

**CV 排布**: 纯 V

**计算流（Sinkhorn 实例）**:
```
for iter in range(num_iters):
    row_sum = sum(x, dim=-1, keepdim=True)
    x = div(x, row_sum)
    col_sum = sum(x, dim=-2, keepdim=True)
    x = div(x, col_sum)
```

**计算流（泛化骨架）**:
```
for iter in range(num_iters):
    for axis in normalize_axes:                      # 可能是 1 个 / 2 个 / k 个轴
        agg = reduce(x, dim=axis, keepdim=True)      # sum / max / l2-norm 等
        x = normalize_op(x, agg)                     # div / sub_max / div_l2
    if converge_check_enabled:
        if delta(x_prev, x) < eps: break             # 提前终止（PyPTO 不支持，需固定 iter）
```

**泛化变体（适用算子族）**:

| 变体 | 归一化 op | 归一化轴序 | 迭代数 | 典型场景 |
|------|---------|-----------|--------|---------|
| **Sinkhorn**（当前样本） | `div` | 行 → 列交替 | 10~20 | MoE 软路由 / Optimal Transport |
| **Greenkhorn / Newton-Sinkhorn** | `div` 带 Newton 校正 | 行 → 列 | 5~10 | 加速 Sinkhorn |
| **OT 行/列双归一化** | `div` | 行 / 列任一固定 | 1（无迭代） | 简化分配 |
| **Reweighted Softmax 收敛** | `softmax → div` | 单轴反复 | 3~5 | Adaptive Softmax / 路由 fine-tune |
| **Power Iteration**（特征向量） | `l2_normalize` | 矩阵-向量乘后单轴 | 10~50 | 大模型权重 SVD 初始化 |
| **Symmetric Normalize**（图相关） | `div(sqrt(D_row) * sqrt(D_col))` | 双轴对称 | 1~3 | GNN 邻接矩阵归一化 |

**编程约束（PyPTO 特化）**:
- 迭代数必须是**编译期常量**（用 Python `for iter in range(N)`），不能用 SymbolicScalar
- 不支持运行时收敛判断，必须用固定迭代数
- 每次迭代都要在新的 tile 上独立做 reduce，编译器不会自动 CSE 跨迭代

**使用算子**: mhc_sinkhorn（其他变体可参考本骨架实现）



---
