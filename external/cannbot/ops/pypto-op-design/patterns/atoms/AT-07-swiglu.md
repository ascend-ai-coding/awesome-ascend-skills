---
type: pattern/atom
title: SwiGLU Activation
description: Swish-Gated Linear Unit = `silu(gate) * value`，FFN 层标准激活。
tags:
- activation
flow_pattern:
- V
examples:
- FusedSwiGLU
- GMM SwiGLU
- GLMMoEFusion
---

## AT-07: SwiGLU Activation

**描述**: Swish-Gated Linear Unit = `silu(gate) * value`，FFN 层标准激活。

**CV 排布**: 纯 V（通常在两个 MatMul 之后）

**输入**:
- `gate: Tensor[M, N]` — 门控路径（FP32）
- `value: Tensor[M, N]` — 值路径（FP32）

**输出**:
- `y: Tensor[M, N]` — 激活结果

**计算流（两种实现）**:

**变体 A — 使用 exp（无 sigmoid 原语时）**:
```
sigmoid_gate = div(ones, add(ones, exp(neg(gate))))
silu = mul(gate, sigmoid_gate)
y = mul(silu, value)
```

**变体 B — 直接 silu 算子**:
```
silu = sigmoid(gate) * gate
y = mul(silu, value)
```



---
