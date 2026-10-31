---
type: pattern/atom
title: RMSNorm
description: Root Mean Square Layer Normalization，LLM 中最常用的归一化层。
tags:
- norm
flow_pattern:
- V
examples:
- InplaceAddRmsNorm
- MLAProlog
- mhc_pre
- Compressor
---

## AT-03: RMSNorm

**描述**: Root Mean Square Layer Normalization，LLM 中最常用的归一化层。

**CV 排布**: 纯 V

**输入**:
- `x: Tensor[*, D]` — 输入张量（BF16/FP16）
- `gamma: Tensor[D]` — 缩放参数（BF16，可选）
- `eps: float` — 防止除零的小值

**输出**:
- `y: Tensor[*, D]` — 归一化结果（BF16/FP16）
- `rstd: Tensor[*, 1]` — 倒数标准差（可选输出）

**计算流（两种变体）**:

**变体 A — rsqrt（推荐，硬件融合指令）**:
```
x_fp32 = cast(x, FP32)
x_sq = mul(x_fp32, x_fp32)
mean_sq = mul(sum(x_sq, dim=-1, keepdim=True), 1.0/D)
var = add(mean_sq, eps)
rstd = rsqrt(var)
y = mul(x_fp32, rstd)
[可选] y = mul(y, cast(gamma, FP32))
y_out = cast(y, BF16)
```

**变体 B — sqrt + div**:
```
...同上到 mean_sq...
var = add(mean_sq, eps)
std = sqrt(var)
rstd = div(ones, std)
...
```

**实例化参数**:
| 参数 | 说明 | 变体 |
|------|------|------|
| `has_gamma` | 是否乘 gamma | 有 gamma (MLAProlog) / 无 gamma (mhc_pre) |
| `has_bias` | 是否加 bias | GLMAttnFusion 使用 |
| `rsqrt_mode` | rsqrt vs sqrt+div | rsqrt (Qwen3) / sqrt+div (GLM) |
| `output_rstd` | 是否输出 rstd | InplaceAddRmsNorm 输出 |

**高频组合**: 通常与 Linear Projection 组合为 AT-10 (RMSNorm + Linear)



---
