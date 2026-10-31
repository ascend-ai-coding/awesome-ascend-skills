---
type: pattern/atom
title: Linear Projection (Quantized MatMul)
description: 带可选量化的线性投影，支持 INT8 W8A8、MXFP8 和标准 BF16 三种模式。
tags:
- matmul
flow_pattern:
- C
- V
examples:
- GLMAttnFusion
- GMM SwiGLU
- GLMMoEFusion
---

## AT-09: Linear Projection (Quantized MatMul)

**描述**: 带可选量化的线性投影，支持 INT8 W8A8、MXFP8 和标准 BF16 三种模式。

**CV 排布**: C + V（量化/反量化为 V，MatMul 为 C）

**输入**:
- `x: Tensor[M, K]` — 输入
- `w: Tensor[K, N]` 或 `[N, K]` — 权重
- `bias: Tensor[N]` — 偏置（可选）
- 量化参数（可选）: `x_scale`, `w_scale`, `quant_bias`, `deq_scale`

**输出**:
- `y: Tensor[M, N]` — 投影结果

**计算流（按量化模式分支）**:

**BF16 模式**:
```
y = matmul(x, w, dtype=BF16, b_trans=True)
```

**INT8 W8A8 模式**:
```
x_int8, x_scale = AT-05(x)                    # 量化激活
y_int32 = matmul(x_int8, w_int8, dtype=INT32)  # 整数矩阵乘
y = AT-06(y_int32, x_scale, w_scale)           # 反量化
```

**MXFP8 模式**:
```
y = scaled_mm(x_fp8, w_fp8, FP32, x_scale, w_scale)
```

**实例化参数**:
| 参数 | 说明 |
|------|------|
| `quant_mode` | none / int8_w8a8 / mxfp8 |
| `has_bias` | 是否有偏置 |
| `out_dtype` | 输出 dtype（BF16/FP32） |



---
