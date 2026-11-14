# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Independent CPU mathematics for the BF16 / OCP MXFP8 projection."""

import torch


def inverse_rope(attn_out, sin, cos):
    """Apply separate FP32 products/additions, then round the tail to BF16."""
    x = attn_out.float()
    result = x.clone()
    even, odd = x[..., 448::2], x[..., 449::2]
    c, s = cos[:, None, ::2], sin[:, None, ::2]
    result[..., 448::2] = even * c + odd * s
    result[..., 449::2] = odd * c - even * s
    return result.to(torch.bfloat16)


def quantize_mx(x):
    """Quantize BF16 values in groups of 32; return payload and E8M0 bytes."""
    if x.device.type != "cpu" or x.dtype != torch.bfloat16 or x.shape[-1] % 64:
        raise ValueError("CPU BF16 input with K divisible by 64 required")
    grouped = x.contiguous().reshape(*x.shape[:-1], -1, 32)
    exponent = ((grouped.view(torch.int16).to(torch.int32) & 0x7F80) >> 7).amax(-1)
    exponent = exponent.clamp_min(8)
    scale = (exponent - 8).to(torch.uint8)
    reciprocal_bits = torch.where(exponent == 8, 0, (262 - exponent) << 7)
    reciprocal = reciprocal_bits.to(torch.int16).view(torch.bfloat16).float()
    scaled = grouped.float() * reciprocal[..., None]
    payload = scaled.clamp(-448, 448).to(torch.float8_e4m3fn).reshape(x.shape)
    return payload, scale.reshape(*x.shape[:-1], x.shape[-1] // 64, 2)


def dequantize_mx(payload, scales):
    """Decode the per-32 E8M0 scale in the public packed scale layout."""
    code = scales.view(torch.uint8).reshape(*payload.shape[:-1], -1).int()
    factor = torch.pow(2.0, code.double() - 127)
    return (
        payload.double().reshape(*payload.shape[:-1], -1, 32) * factor[..., None]
    ).reshape(payload.shape)


def attn_epilogue_golden_cpu(
    attn_out, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos
):
    """CPU mathematical oracle with BF16 rounding at each projection boundary.

    FP64 dot products isolate mathematical rounding from device accumulation.
    Device acceptance additionally uses the upstream native oracle, because its
    FP32 accumulation order is part of the upstream zero-tolerance criterion.
    """
    if any(
        v.device.type != "cpu"
        for v in (attn_out, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos)
    ):
        raise ValueError("the CPU oracle requires CPU tensors")
    t = attn_out.shape[0]
    rotated = inverse_rope(attn_out, rope_sin, rope_cos).reshape(t, 8, 4096)
    aq, asc = quantize_mx(rotated)
    activation = dequantize_mx(aq, asc)
    wa = dequantize_mx(woa, descale_woa)
    y = torch.bmm(activation.permute(1, 0, 2), wa.transpose(1, 2)).permute(1, 0, 2)
    y = y.contiguous().reshape(t, 8192).to(torch.bfloat16)
    yq, ys = quantize_mx(y)
    result = dequantize_mx(yq, ys) @ dequantize_mx(wob, descale_wob).t()
    return result.to(torch.bfloat16)


def assert_exact(actual, expected, label="output"):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f"{label}: shape/dtype mismatch")
    a, b = actual.detach().cpu(), expected.detach().cpu()
    if a.is_floating_point() and not bool(torch.isfinite(a.float()).all()):
        raise AssertionError(f"{label}: nonfinite values")
    torch.testing.assert_close(a, b, rtol=0, atol=0, msg=label)
    if a.dtype == torch.bfloat16 and not torch.equal(
        a.view(torch.int16), b.view(torch.int16)
    ):
        raise AssertionError(f"{label}: BF16 bit patterns differ")


def assert_payload(actual, expected, label="payload"):
    assert_exact(actual.view(torch.uint8), expected.view(torch.uint8), label)
