---
type: pattern/atom
title: FP8 Quantization (Symmetric Per-Token)
description: 将 FP32/BF16 张量量化为 FP8 E4M3，输出 FP8 数据 + FP32 scale。
tags:
- quant
flow_pattern:
- V
examples:
- PageAttnFP8
---

## AT-08: FP8 Quantization (Symmetric Per-Token)

**描述**: 将 FP32/BF16 张量量化为 FP8 E4M3，输出 FP8 数据 + FP32 scale。

**CV 排布**: 纯 V

**计算流**:
```
x_fp32 = cast(x, FP32)
x_max = amax(abs(x_fp32), dim=-1, keepdim=True)
scale = div(full([r,c], 448.0, FP32), x_max)   # 448 = FP8E4M3 max
x_scaled = mul(x_fp32, scale)
x_fp8 = cast(x_scaled, DT_FP8E4M3)
inv_scale = div(full([r,c], 1.0, FP32), scale)
return (x_fp8, inv_scale)
```

**使用算子**: PageAttnFP8（在 softmax 后对 P 重新量化为 FP8 再做 PV matmul）

**与 AT-05 的区别**: 输出 dtype 为 FP8E4M3（非 INT8），scale 上限为 448.0（非 127.0）



---
