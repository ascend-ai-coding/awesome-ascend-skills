---
type: pattern/atom
title: RMSNorm + Linear + Quant (Fused)
description: Norm → Quant → MatMul → Dequant 的完整量化投影管线。
tags:
- norm-linear-quant-fused
flow_pattern:
- V
- C
- V
examples:
- MLAPrologQuant
- GLMAttnFusion
---

## AT-11: RMSNorm + Linear + Quant (Fused)

**描述**: Norm → Quant → MatMul → Dequant 的完整量化投影管线。

**CV 排布**: V → C → V

**计算流**:
```
# V 阶段: RMSNorm
normed = AT-03(x, gamma, eps)

# V 阶段: Quant
normed_int8, norm_scale = AT-05(normed)

# C 阶段: INT8 MatMul
y_int32 = matmul(normed_int8, w_int8, dtype=INT32)

# V 阶段: Dequant
y = AT-06(y_int32, norm_scale, w_scale)
```

**使用算子**: MLAPrologQuant, GLMAttnFusion Phase1



---
