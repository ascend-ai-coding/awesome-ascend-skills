# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent NumPy references and expected Fixpipe case classifications.

MATRIX classifies Acc→Vec as exact, rounded/saturated, byte-carried or rejected. MAT_MATRIX separates q@q numerical
readback, HF8 execution-only checks and UINT8 readback rejection. Acc→GM has an explicit outcome override for scaled
FP32 output. These tables define expectations; runtime source and toolchain identity belong to the source-gated device
runner.

Controlled small integer inputs and power-of-two scales permit exact comparisons. FP8 uses an independent finite E4M3FN
decoder/nearest-value quantizer. HF8 has no independent host codec in this example, so its execution and
device-conversion consistency checks remain distinct from numerical format validation.
"""

import struct

import numpy as np


# Expected Vec outcomes: exact, rounded, byte-carried, or parse-only rejection.
MATRIX = (
    ("FP32", "INT8", "int8", "rounded"),
    ("FP32", "UINT8", "uint8", "rounded"),
    ("FP32", "HF8", "uint8", "opaque"),
    ("FP32", "FP16", "float16", "exact"),
    ("FP32", "BF16", "bfloat16", "exact"),
    ("FP32", "FP8E4M3FN", "uint8", "opaque"),
    ("FP32", "FP32", "float32", "rejected"),
    ("INT32", "INT8", "int8", "rounded"),
    ("INT32", "UINT8", "uint8", "rounded"),
    ("INT32", "FP16", "float16", "exact"),
    ("INT32", "BF16", "bfloat16", "exact"),
)

# Mat checks: exact q@q, execution-only HF8, or unsupported UINT8 readback.
MAT_MATRIX = (
    ("FP32", "INT8", "q_at_q"),
    ("FP32", "UINT8", "no_readback"),
    ("FP32", "FP16", "q_at_q"),
    ("FP32", "BF16", "q_at_q"),
    ("FP32", "HF8", "runs_only"),
    ("FP32", "FP8E4M3FN", "q_at_q"),
    ("INT32", "INT8", "q_at_q"),
    ("INT32", "UINT8", "no_readback"),
    ("INT32", "FP16", "q_at_q"),
    ("INT32", "BF16", "q_at_q"),
)

# The GM route supports scaled FP32 output.
STORE_OUTCOME_OVERRIDES = {("FP32", "FP32"): "exact"}


def store_outcome(src, dst):
    """Select the GM outcome override or the matching Vec-table classification."""
    override = STORE_OUTCOME_OVERRIDES.get((src, dst))
    if override is not None:
        return override
    for s, d, _, outcome in MATRIX:
        if (s, d) == (src, dst):
            return outcome
    raise KeyError(f"{src}→{dst} 不在 MATRIX 里")


# Non-quantized destination combinations used as explicit controls.
NO_SCALE_DESTINATIONS = {
    "FP32": ("FP32", "FP16", "BF16"),
    "INT32": ("INT32",),
}

# Integer saturation bounds.
SATURATION = {"INT8": (-128, 127), "UINT8": (0, 255)}

# Per-channel signedness is encoded in the payload rather than inferred from the destination.
SIGNED_FLAG_FOR = {"INT8": True, "UINT8": False}


def make_inputs(m, k, n, src, seed=0):
    """Create deterministic small integer operands in the actual FP16 or INT8 input dtype."""
    rng = np.random.default_rng(seed)
    a = rng.integers(-4, 4, size=(m, k))
    b = rng.integers(-2, 2, size=(k, n))
    if src == "FP32":
        return a.astype(np.float16), b.astype(np.float16)
    return a.astype(np.int8), b.astype(np.int8)


def accumulate(a, b):
    """Compute the unblocked mathematical product in NumPy FP64."""
    return a.astype(np.float64) @ b.astype(np.float64)


def scaled_exact(a, b, scale):
    """Return the exact scaled mathematical product for the controlled inputs."""
    return accumulate(a, b) * scale


def scaled_rounded(a, b, scale, dst):
    """Round to nearest-even and saturate to the declared signed or unsigned integer range."""
    lo, hi = SATURATION[dst]
    return np.clip(np.rint(accumulate(a, b) * scale), lo, hi)


def decode_fp8_e4m3(raw):
    """Decode finite E4M3FN bytes, including subnormals; the all-ones exponent/maximal mantissa encodes NaN."""
    raw = np.asarray(raw, dtype=np.uint8).astype(np.int32)
    sign = np.where(raw >> 7 == 1, -1.0, 1.0)
    exp = (raw >> 3) & 0xF
    mant = raw & 0x7
    # 非规格化数：exp == 0 时隐含位是 0，指数固定为 1-bias
    normal = 2.0 ** (exp.astype(np.float64) - 7.0) * (1.0 + mant / 8.0)
    subnormal = 2.0 ** (1.0 - 7.0) * (mant / 8.0)
    return sign * np.where(exp == 0, subnormal, normal)


def per_channel_scales(n, *, exponents=(-2, -1, 0, 1, 2)):
    """Create a [1,N] table of varying powers of two to distinguish per-column scaling from broadcasting."""
    return np.array(
        [[2.0 ** exponents[j % len(exponents)] for j in range(n)]], dtype=np.float64
    )


def fp8_e4m3_values():
    """Enumerate the 253 distinct finite E4M3FN values through the independent byte decoder."""
    raw = np.arange(256, dtype=np.uint8)
    finite = raw[(raw & 0x7F) != 0x7F]
    return np.unique(decode_fp8_e4m3(finite))


def quantize_to_fp8_e4m3(x):
    """Select the nearest finite representable value, with deterministic even-mantissa ties and saturation at the finite
    endpoints.
    """
    values = fp8_e4m3_values()
    flat = np.asarray(x, dtype=np.float64).reshape(-1)
    idx = np.clip(np.searchsorted(values, flat), 1, len(values) - 1)
    lo, hi = values[idx - 1], values[idx]
    take_hi = np.abs(hi - flat) < np.abs(flat - lo)
    tie = np.abs(hi - flat) == np.abs(flat - lo)
    # 等距时取「尾数为偶」的那个：E4M3 的可表示值里相邻两个尾数必有一奇一偶,
    # 偶的那个是 lo 还是 hi 取决于奇偶性，按 lo 的索引判。
    take_hi = np.where(tie, (idx - 1) % 2 == 1, take_hi)
    out = np.where(take_hi, hi, lo)
    return out.reshape(np.asarray(x).shape)


def quantized_in_l1(a, b, scale, dst):
    """Form the quantized L1 reference used by q@q; controlled BF16 inputs are exactly representable."""
    scaled = accumulate(a, b) * scale
    if dst in SATURATION:
        lo, hi = SATURATION[dst]
        return np.clip(np.rint(scaled), lo, hi)
    if dst == "FP16":
        return scaled.astype(np.float16).astype(np.float64)
    if dst == "BF16":
        return scaled
    if dst == "FP8E4M3FN":
        return quantize_to_fp8_e4m3(scaled)
    raise ValueError(f"没有 {dst} 的编码器——这一格只能验跑通")


# The scale retains ten explicit mantissa bits.
SCALE_MANTISSA_BITS = 10


def truncate_scale_to_hardware(scale):
    """Clear thirteen low FP32 mantissa bits, retaining ten explicit bits. Normal finite values are the reference
    domain.
    """
    bits = struct.unpack("!I", struct.pack("!f", float(scale)))[0]
    drop = 23 - SCALE_MANTISSA_BITS
    return struct.unpack("!f", struct.pack("!I", bits & ~((1 << drop) - 1)))[0]


# Payload fields from asc_set_l0c_copy_prequant_impl.h: scale mask 0xFFFFE000, offset shift 37, signed flag shift 46.
QUANT_PAYLOAD_LAYOUT = (
    "scale: bit13..31 | offset: bit37..45 (9-bit two's complement) | is_signed: bit46"
)

QUANT_OFFSET_SHIFT = 37
QUANT_OFFSET_MASK = 0x1FF
QUANT_SIGNED_SHIFT = 46


def encode_quant_payload(scale, *, offset=0, is_signed=False):
    """Pack fp19 scale, signed nine-bit offset and bit46. The explicit quant_offset parameter is tested
    independently.
    """
    bits = struct.unpack("!I", struct.pack("!f", float(scale)))[0]
    payload = bits & 0xFFFFE000
    payload |= (int(offset) & QUANT_OFFSET_MASK) << QUANT_OFFSET_SHIFT
    payload |= int(bool(is_signed)) << QUANT_SIGNED_SHIFT
    return payload


def sign_extend_9bit(offset):
    """Interpret the low nine offset bits as two's complement."""
    value = int(offset) & QUANT_OFFSET_MASK
    return value - 512 if value >= 256 else value


def quantized_with_offset(acc, scale, offset, dst="INT8"):
    """Scale with fp19 truncation, add the signed offset, round and saturate."""
    lo, hi = SATURATION[dst]
    scaled = np.asarray(acc, dtype=np.float64) * truncate_scale_to_hardware(scale)
    return np.clip(np.rint(scaled + sign_extend_9bit(offset)), lo, hi)
