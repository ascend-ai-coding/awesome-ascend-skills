---
type: pattern/atom
title: Online Softmax
description: 在 KV 序列分块场景下，逐块累积 softmax 的最大值、指数和与加权和，最终归一化。
tags:
- softmax
flow_pattern:
- V
examples:
- FA
- BSA
- PageAttn
- SparseAttn
---

## AT-01: Online Softmax

**描述**: 在 KV 序列分块场景下，逐块累积 softmax 的最大值、指数和与加权和，最终归一化。

**CV 排布**: 纯 V（但通常紧跟在 C1 matmul 之后，作为 C1-V1-C2 中 V1 阶段）

**输入**:
- `scores: Tensor[M, K_tile]` — QK^T 的当前块分数（FP32）
- `mi_update: Tensor[M, 1]` — 上一块的行最大值（FP32，累积器）
- `li_update: Tensor[M, 1]` — 上一块的指数和（FP32，累积器）
- `oi_update: Tensor[M, D]` — 上一块的加权和（FP32，累积器）
- `scale: float` — 缩放系数 (1/√d)

**输出**:
- 更新后的 `mi_update`, `li_update`, `oi_update`
- 若 `is_loop_end`：最终归一化结果 `oi_final = oi_update / li_update`

**计算流**:
```
# V1 阶段：计算当前块 softmax 分量
scores_scaled = mul(scores, scale)
mij = amax(scores_scaled, dim=-1, keepdim=True)
pij = exp(sub(scores_scaled, mij))
lij = sum(pij, dim=-1, keepdim=True)

# 三路分支
if is_loop_begin AND is_loop_end:
    p_norm = div(pij, lij)
    p_bf16 = cast(p_norm, BF16)
    oij = matmul(p_bf16, V)          # -> C2
    oi_final = oij                    # 无需累积
elif is_loop_begin:
    p_bf16 = cast(pij, BF16)
    oij = matmul(p_bf16, V)          # -> C2, FP32 accumulation
    mi_update[:] = mij
    li_update[:] = lij
    oi_update[:] = oij
else:
    # Online 累积更新
    p_bf16 = cast(pij, BF16)
    oij = matmul(p_bf16, V)
    mi_new = maximum(mi_update, mij)
    alpha = exp(sub(mi_update, mi_new))
    beta = exp(sub(mij, mi_new))
    li_new = add(mul(alpha, li_update), mul(beta, lij))
    oi_new = add(mul(oi_update, alpha), mul(oij, beta))
    if is_loop_end:
        oi_final = div(oi_new, li_new)
    else:
        mi_update[:] = mi_new
        li_update[:] = li_new
        oi_update[:] = oi_new
```

**dtype 路由**: 输入 BF16/FP16 → 计算全程 FP32 → 输出 BF16/FP16

**动态轴**: KV 序列长度动态，tile 数量 `ceildiv(seq_len, tile_size)` 动态

**跨 loop 状态**: 3 个 FP32 累积器 (`mi`, `li`, `oi`) 跨 KV tile 迭代传递

**特征维度标签**:
- 计算流特征: softmax 归一化 + online 累积
- 数据连续性: 连续 view（KV 连续存储时）/ 非连续 gather（稀疏 KV）
- 跨 loop 状态更新: 最大值/指数和/加权和 的 running accumulation

**实例化参数**:
| 参数 | 说明 | 典型值 |
|------|------|--------|
| `scale` | 1/√head_dim | 0.125 (d=64) / 0.0625 (d=256) |
| `mask_type` | 无 / causal / sliding_window / sparse | 依算子类型 |
| `mask_value` | 屏蔽填充值 | -65504.0 (FP16) / -3.4e38 (FP32) |



---
