---
type: pattern/atom
title: QK^T MatMul + Scale
description: Attention 核心的 Q @ K^T 计算并缩放。
tags:
- matmul
flow_pattern:
- C
examples:
- 所有 Attention 算子
---

## AT-12: QK^T MatMul + Scale

**描述**: Attention 核心的 Q @ K^T 计算并缩放。

**CV 排布**: C

**计算流**:
```
scores = matmul(Q, K, dtype=FP32, b_trans=True)
scores_scaled = mul(scores, scale)   # scale = 1/√d
```

**实例化参数**:
| 参数 | 说明 |
|------|------|
| `q_dtype` | Q 的输入 dtype (BF16/FP16/FP8) |
| `dequant_after` | FP8 输入时是否需要在 matmul 后反量化 |
| `q_scale / k_scale` | FP8 模式下的反量化 scale |

**FP8 变体**:
```
scores_int = matmul(Q_fp8, K_fp8, dtype=FP32, b_trans=True)
scores = dequant_dynamic(scores_int, q_scale, k_scale_T)
scores_scaled = mul(scores, scale)
```



---
