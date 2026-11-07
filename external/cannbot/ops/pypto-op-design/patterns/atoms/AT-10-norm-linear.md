---
type: pattern/atom
title: RMSNorm + Linear (Fused)
description: 先做 RMSNorm 归一化，再做 Linear 投影。这是 Prolog 算子和 Pre-Attention 算子的标准子结构。
tags:
- norm-linear-fused
flow_pattern:
- V
- C
examples:
- MLAProlog
- Qwen3PreAttn
---

## AT-10: RMSNorm + Linear (Fused)

**描述**: 先做 RMSNorm 归一化，再做 Linear 投影。这是 Prolog 算子和 Pre-Attention 算子的标准子结构。

**CV 排布**: V → C

**计算流**:
```
# V 阶段: RMSNorm
normed = AT-03(x, gamma, eps)
normed_bf16 = cast(normed, BF16)

# C 阶段: Linear
projected = matmul(normed_bf16, weight, dtype=BF16, b_trans=True)
```

**使用算子**: MLAProlog (q_a_proj → norm → q_b_proj), Qwen3PreAttn (input_norm → QKV_proj)



---
