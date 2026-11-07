---
type: pattern/atom
title: Standard Softmax
description: 在非分块（或单次全量）场景下的标准 softmax，无需 online 累积。
tags:
- softmax
flow_pattern:
- V
examples:
- SparseAttnTND
- WinAttn
- CompressFA
---

## AT-02: Standard Softmax

**描述**: 在非分块（或单次全量）场景下的标准 softmax，无需 online 累积。

**CV 排布**: 纯 V

**输入**:
- `scores: Tensor[M, N]` — 原始分数（FP32）
- `scale: float`
- `mask: Tensor[M, N]` (可选) — 布尔掩码

**输出**:
- `p_norm: Tensor[M, N]` — softmax 概率（FP32 或 BF16）；或（`normalization_position=post-pv` 时）未归一化指数 `pij` 与指数和 `lij`，归一化在 C2 之后完成

**计算流（两种归一化位置变体）**:

**变体 A — pre-pv 归一化（默认）**：
```
scores_scaled = mul(scores, scale)
[可选] scores_masked = add(mul(scores_scaled, mask), mul(inv_mask, LARGE_NEG))  # 屏蔽位置置 LARGE_NEG；不应用 mask 时 = scores_scaled
mij = amax(scores_masked, dim=-1, keepdim=True)
pij = exp(sub(scores_masked, mij))
lij = sum(pij, dim=-1, keepdim=True)
p_norm = div(pij, lij)
p_out = cast(p_norm, BF16)
# C2: matmul(p_out, V)
```

**变体 B — post-pv 延迟归一化（golden 强制 P 未归一化 bf16 场景）**：
```
scores_scaled = mul(scores, scale)
mij = amax(scores_scaled, dim=-1, keepdim=True)
pij = exp(sub(scores_scaled, mij))
lij = sum(pij, dim=-1, keepdim=True)
p_bf16 = cast(pij, BF16)              # P 未归一化即量化（舍入点在 P@V 前）
q1 = matmul(p_bf16, V)                # C2，FP32 累加
out = cast(div(q1, lij), BF16)        # 归一化延迟到 C2 之后
```

**变体选择规则**：golden 若要求 P 在 P@V 前以**未归一化**形态 cast bf16（如 online softmax 变体的单遍化，保持与在线 golden 相同的 P 舍入点），必须用变体 B；变体 A 会移动 P 的 bf16 舍入点，与该类 golden 精度不可比。

**与 AT-01 的区别**: 无累积器、无三路分支、无跨 loop 状态。适用于 KV 全量可见的场景（如稀疏注意力每个 token 的 topk KV）。

**实例化参数**:
| 参数 | 说明 |
|------|------|
| `apply_mask` | 是否应用掩码 |
| `sink_integration` | 是否融合 attention sink（WinAttn, CompressFA） |
| `normalization_position` | `pre-pv`（变体 A，默认）/ `post-pv`（变体 B，延迟归一化） |



---
