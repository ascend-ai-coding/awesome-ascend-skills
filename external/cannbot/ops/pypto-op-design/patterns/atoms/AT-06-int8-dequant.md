---
type: pattern/atom
title: Per-Token Dequantization (INT → FP)
description: 将量化后的整数乘积累积结果还原为浮点。
tags:
- dequant
flow_pattern:
- V
examples:
- GLMMoEFusion
- MLAPrologQuant
- PageAttnFP8
---

## AT-06: Per-Token Dequantization (INT → FP)

**描述**: 将量化后的整数乘积累积结果还原为浮点。

**CV 排布**: 纯 V

**输入**:
- `x_int: Tensor[M, N]` — 整数累加结果（INT32）
- `scale_a: Tensor[M, 1]` — 激活反量化 scale（FP32）
- `scale_b: Tensor[1, N]` 或 `Tensor[M, 1]` — 权重反量化 scale（FP32）

**输出**:
- `x_fp32: Tensor[M, N]` — 浮点结果

**计算流**:
```
x_fp32 = cast(x_int, FP32)
x_fp32 = mul(x_fp32, cast(scale_a, FP32))
x_fp32 = mul(x_fp32, cast(scale_b, FP32))
```

**变体 — MXFP8 (scaled_mm)**:
```
result = pypto.scaled_mm(A_fp8, B_fp8, out_dtype=FP32, scaled_A, scaled_B)
```
此时量化/反量化由 `scaled_mm` 内部处理。

### 变体 C — 稀疏 Gather + Dequant（UB 内融合）

**适用场景**：量化 KV cache 的稀疏注意力，key 需同时做 gather + dequant。
**实例**：`models/deepseek_v32_exp/sparse_flash_attention_quant_impl.py` 中 `gather_in_ub()`

```
for i in range(kv_block_count):
    k_block = view(k_cache_2d, [BS, D], [phys_idx*BS, 0],
                   valid_shape=[valid_bs, D])
    k_scale_block = view(k_scale_2d, [BS, 1], [phys_idx*BS, 0],
                         valid_shape=[valid_bs, 1])
    k_fp32 = cast(k_block, FP32)
    k_fp32 = mul(k_fp32, k_scale_block)          # UB 内反量化
    kj_assemble[i*BS:(i+1)*BS, :] = k_fp32       # 拼装到连续 buffer
```



---
