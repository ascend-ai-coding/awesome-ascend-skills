---
type: pattern/atom
title: RoPE (Rotary Position Embedding)
description: 旋转位置编码，应用于 Query/Key 的 head_dim 维度。
tags:
- positional-encoding
flow_pattern:
- V
examples:
- MLAProlog
- Qwen3PreAttn
- InterleaveRope
---

## AT-04: RoPE (Rotary Position Embedding)

**描述**: 旋转位置编码，应用于 Query/Key 的 head_dim 维度。

**CV 排布**: 纯 V

**三种变体**:

### 变体 A — Half-Split（最常用）
```
# 输入 x: [B, N, D], cos/sin: [B, D/2]
x_left = view(x, [..., 0:D/2])
x_right = view(x, [..., D/2:D])
o1 = sub(mul(x_left, cos), mul(x_right, sin))
o2 = add(mul(x_right, cos), mul(x_left, sin))
out = concat([o1, o2], dim=-1)
```
**使用算子**: MLAProlog, Qwen3PreAttn, Compressor

### 变体 B — Interleave GatherMask
```
xe = gathermask(x, mode=1)   # 偶数位
xo = gathermask(x, mode=2)   # 奇数位
ye = sub(mul(xe, cos), mul(xo, sin))
yo = add(mul(xe, sin), mul(xo, cos))
out = assemble interleaved ye, yo
```
**使用算子**: InterleaveRope

### 变体 C — Rotate-Half (Reshape-Transpose)
```
x_2d = reshape(x, [..., D/2, 2])
x_2d_t = transpose(x_2d, -1, -2)  # swap last two dims
x_rot = reshape(x_2d_t, [..., D])
# x_rot = [-x2, x1]
out = concat([mul(-x_rot_right, cos) + mul(x_rot_left, sin), ...])
```
**使用算子**: MLAPrologQuant

**实例化参数**:
| 参数 | 说明 |
|------|------|
| `variant` | A / B / C |
| `rotary_dim` | 旋转维度（通常 = head_dim 或 head_dim 的子集） |
| `partial_rotate` | 是否仅旋转部分维度（前 rotary_dim），剩余维度直通 |



---
