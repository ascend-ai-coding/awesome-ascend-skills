---
type: pattern/atom
title: Per-Token Quantization (FP → INT8)
description: 对每个 token 的向量做对称量化，输出 INT8 张量 + FP32 scale。
tags:
- quant
flow_pattern:
- V
examples:
- GLMMoEFusion
- MLAPrologQuant
- GMM SwiGLU
---

## AT-05: Per-Token Quantization (FP → INT8)

**描述**: 对每个 token 的向量做对称量化，输出 INT8 张量 + FP32 scale。

**CV 排布**: 纯 V

**输入**:
- `x: Tensor[M, D]` — 输入（BF16/FP32）

**输出**:
- `x_q: Tensor[M, D]` — 量化结果（INT8）
- `scale: Tensor[M, 1]` — 反量化 scale（FP32）

**计算流**:
```
x_fp32 = cast(x, FP32)
x_abs = abs(x_fp32)
x_max = amax(x_abs, dim=-1, keepdim=True)
scale = div(full([M,1], 127.0, FP32), x_max)
x_scaled = mul(x_fp32, scale)
x_int32 = cast(x_scaled, INT32, rounding_mode=RINT)
x_fp16 = cast(x_int32, FP16, rounding_mode=ROUND)
x_q = cast(x_fp16, INT8, rounding_mode=TRUNC, saturate=True)
inv_scale = div(full([M,1], 1.0, FP32), scale)
```

**高频组合**: 与 Dequant (AT-06) 配对使用。夹在两次 MatMul 之间形成 Quant-MatMul-Dequant 管线。



---
