#!/usr/bin/env python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
Elementwise Tiling Algorithm — Fallback Reference Implementation.

Covers: Sin, Cos, Abs, Exp and other 1D elementwise ops.
All inputs/outputs share identical shape, no cross-element dependency.
"""

import math
import logging
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_TILING_BITS = 32768      # 4KB per core minimum (bits)
ELEM_ALIGN_FACTOR = 512      # multi-core element alignment
ALIGN_256 = 256              # UB alignment (bytes)


@dataclass
class ElewiseTilingInput:
    """Input parameters for compute_elewise_tiling."""
    shape: List[int]
    dtype: str = "fp16"
    op_type: str = "add"
    ub_size: int = 248 * 1024
    core_num: int = 64
    buffer_num: int = 3
    extra_size: int = 0
    input_same_magnitude: bool = False


@dataclass
class _ElewiseConfig:
    """Packed elementwise config to reduce argument count."""
    ub_size: int
    extra_size: int
    buffer_num: int
    elem_bytes: int
    min_dtype_bits: int
    core_num: int


@dataclass
class ElewiseTilingData:
    """Elementwise tiling output."""

    # Problem
    dim0: int                    # total element count (flattened)
    dtype: str = "fp16"
    elem_bytes: int = 2

    # Multi-core split
    core_num: int = 0            # actual cores used
    block_former: int = 0         # elements per core (512-aligned)
    block_num: int = 0            # virtual block count
    block_tail: int = 0           # tail block elements

    # UB split
    ub_former: int = 0            # elements per UB tile (256B-aligned)
    ub_loop_former: int = 0       # UB loops for first block
    ub_tail_former: int = 0       # tail size for first block's last UB
    ub_loop_tail: int = 0         # UB loops for tail block
    ub_tail_tail: int = 0         # tail size for tail block's last UB

    # Precision branch
    use_fp32_cast: bool = False   # True when FP16/BF16 Add/Sub needs FP32 intermediate
    k_extra_bufs: int = 0         # extra FP32 buffers for precision path

    # Buffer info
    buffer_num: int = 0
    ub_size: int = 0
    extra_size: int = 0

    def to_dict(self) -> dict:
        return {
            "dim0": self.dim0,
            "dtype": self.dtype,
            "elem_bytes": self.elem_bytes,
            "core_num": self.core_num,
            "block_former": self.block_former,
            "block_num": self.block_num,
            "block_tail": self.block_tail,
            "ub_former": self.ub_former,
            "ub_loop_former": self.ub_loop_former,
            "ub_tail_former": self.ub_tail_former,
            "ub_loop_tail": self.ub_loop_tail,
            "ub_tail_tail": self.ub_tail_tail,
            "use_fp32_cast": self.use_fp32_cast,
        }


# ---------------------------------------------------------------------------
# Precision branch: determine if FP32 cast is needed
# ---------------------------------------------------------------------------

def check_fp32_cast(
    op_type: str,
    dtype: str,
    input_same_magnitude: bool = False,
) -> Tuple[bool, int]:
    """
    Check if the operation needs FP32 intermediate to avoid precision loss.

    Applies to FP16/BF16 Add/Sub where "big eats small" can occur.

    Returns:
        (use_fp32_cast, k_extra_bufs)
    """
    if dtype not in ("fp16", "bf16"):
        return False, 0
    if op_type not in ("add", "sub"):
        return False, 0
    if input_same_magnitude:
        return False, 0
    return True, 2


# ---------------------------------------------------------------------------
# Multi-core split
# ---------------------------------------------------------------------------

def calc_multicore_split(
    dim0: int,
    available_cores: int,
    min_dtype_bits: int,
) -> Tuple[int, int, int, int]:
    """
    Multi-core split with minimum 4KB per core.

    Returns: (core_num, block_former, block_num, block_tail)
    """
    core_num = (dim0 * min_dtype_bits + MIN_TILING_BITS - 1) // MIN_TILING_BITS
    core_num = min(core_num, available_cores)
    core_num = max(core_num, 1)

    block_former = ((dim0 + core_num - 1) // core_num + ELEM_ALIGN_FACTOR - 1)
    block_former = (block_former // ELEM_ALIGN_FACTOR) * ELEM_ALIGN_FACTOR

    block_num = (dim0 + block_former - 1) // block_former
    block_tail = dim0 - (block_num - 1) * block_former
    if block_tail <= 0:
        block_tail = block_former

    return core_num, block_former, block_num, block_tail


# ---------------------------------------------------------------------------
# UB split
# ---------------------------------------------------------------------------

def _calc_ub_former_basic(cfg: _ElewiseConfig, k_extra_bufs: int) -> int:
    """Calculate UB former with basic double-buffer division."""
    double_buffer = 2
    if k_extra_bufs > 0:
        buffer_divisor = double_buffer * cfg.buffer_num * cfg.elem_bytes + k_extra_bufs * 4
    else:
        buffer_divisor = double_buffer * cfg.buffer_num * cfg.elem_bytes
    max_elem_num = (cfg.ub_size - cfg.extra_size) // buffer_divisor
    align_factor = ALIGN_256 // cfg.elem_bytes
    ub_former = (max_elem_num // align_factor) * align_factor
    if ub_former <= 0:
        ub_former = align_factor
    return ub_former


def calc_ub_split(cfg: _ElewiseConfig, k_extra_bufs: int = 0) -> int:
    """UB split with 256B alignment. Double-buffered queues."""
    return _calc_ub_former_basic(cfg, k_extra_bufs)


def calc_loop_info(
    block_former: int,
    block_tail: int,
    ub_former: int,
) -> Tuple[int, int, int, int]:
    """Calculate UB loop counts and tail sizes for first and tail blocks."""
    ub_loop_former = (block_former + ub_former - 1) // ub_former
    ub_tail_former = block_former - (ub_loop_former - 1) * ub_former
    if ub_tail_former <= 0:
        ub_tail_former = ub_former

    ub_loop_tail = (block_tail + ub_former - 1) // ub_former
    ub_tail_tail = block_tail - (ub_loop_tail - 1) * ub_former
    if ub_tail_tail <= 0:
        ub_tail_tail = ub_former

    return ub_loop_former, ub_tail_former, ub_loop_tail, ub_tail_tail


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_elewise_tiling(params: ElewiseTilingInput) -> ElewiseTilingData:
    """
    Compute elementwise tiling for 1D elementwise ops.

    Args:
        params: ElewiseTilingInput with shape, dtype, op_type, and platform config.

    Returns:
        ElewiseTilingData with all tiling parameters
    """
    dtype_map = {"fp16": 2, "float16": 2, "bf16": 2, "bfloat16": 2,
                 "fp32": 4, "float32": 4, "float": 4, "int8": 1}
    elem_bytes = dtype_map.get(params.dtype, 2)
    min_dtype_bits = elem_bytes * 8

    dim0 = 1
    for d in params.shape:
        dim0 *= d

    use_fp32, k_extra = check_fp32_cast(params.op_type, params.dtype, params.input_same_magnitude)

    used_cores, block_former, block_num, block_tail = calc_multicore_split(
        dim0, params.core_num, min_dtype_bits,
    )

    cfg = _ElewiseConfig(
        ub_size=params.ub_size, extra_size=params.extra_size, buffer_num=params.buffer_num,
        elem_bytes=elem_bytes, min_dtype_bits=min_dtype_bits, core_num=used_cores,
    )
    ub_former = calc_ub_split(cfg, k_extra)

    ub_loop_former, ub_tail_former, ub_loop_tail, ub_tail_tail = calc_loop_info(
        block_former, block_tail, ub_former,
    )

    return ElewiseTilingData(
        dim0=dim0,
        dtype=params.dtype,
        elem_bytes=elem_bytes,
        core_num=used_cores,
        block_former=block_former,
        block_num=block_num,
        block_tail=block_tail,
        ub_former=ub_former,
        ub_loop_former=ub_loop_former,
        ub_tail_former=ub_tail_former,
        ub_loop_tail=ub_loop_tail,
        ub_tail_tail=ub_tail_tail,
        use_fp32_cast=use_fp32,
        k_extra_bufs=k_extra,
        buffer_num=params.buffer_num,
        ub_size=params.ub_size,
        extra_size=params.extra_size,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    tiling = compute_elewise_tiling(ElewiseTilingInput(
        shape=[1000, 256],
        dtype="fp16",
        op_type="add",
    ))
    logger.info(json.dumps(tiling.to_dict(), indent=2))
