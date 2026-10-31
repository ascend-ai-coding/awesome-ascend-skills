---
type: pattern/atom
title: P @ V MatMul
description: Attention 的 softmax 概率 P 与 V 的矩阵乘。
tags:
- matmul
flow_pattern:
- C
examples:
- 所有 Attention 算子
---

## AT-13: P @ V MatMul

**描述**: Attention 的 softmax 概率 P 与 V 的矩阵乘。

**CV 排布**: C

**计算流**:
```
P_bf16 = cast(P_fp32, BF16)       # 或 FP16 / FP8
O = matmul(P_bf16, V, dtype=FP32)  # 或 dtype=BF16
```

**FP8 变体**:
```
P_fp8, P_scale = AT-08(P_fp32)     # 量化 P
O_int = matmul(P_fp8, V_fp8, FP32)  # FP8 matmul
O = dequant_dynamic(O_int, P_scale, V_scale)
```

**实例化参数**:
| 参数 | 说明 |
|------|------|
| `p_dtype` | P 的 dtype (BF16/FP16/FP8) |
| `o_dtype` | 输出 dtype (FP32 用于累积, BF16 用于最终) |
| `set_matrix_size` | 是否设置 matmul 尺寸 hint |



---
