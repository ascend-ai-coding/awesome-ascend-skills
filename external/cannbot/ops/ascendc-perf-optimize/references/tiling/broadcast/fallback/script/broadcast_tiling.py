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
Broadcast Tiling Algorithm — Fallback Reference Implementation.

Covers: Add, Mul, Sub and other binary elementwise ops with broadcasting semantics.
Supports: DimensionCollapse → OneDim / UB-Broadcast / NDDMA-Broadcast routing.
"""

import math
import logging
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CACHE_LINE = 128          # bytes, OneDim alignment
REPEAT_BYTES = 256        # bytes, multi-dim alignment
NDDMA_MAX_DIMS = 5        # NDDMA hardware limit


@dataclass
class _TilingConfig:
    """Packed tiling config to reduce argument count across internal helpers."""
    ub_size: int
    extra_size: int
    buffer_num: int
    core_num: int
    max_dtype_bytes: int
    max_dtype_bits: int
    min_dtype_bits: int = 0


@dataclass
class BroadcastTilingInput:
    """Input parameters for compute_broadcast_tiling."""
    input_shapes: List[List[int]]
    output_shape: List[int]
    dtype: str = "fp16"
    ub_size: int = 248 * 1024
    core_num: int = 64
    extra_size: int = 0
    is_dav3510: bool = True
    force_nddma: bool = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BroadcastTilingData:
    """Unified broadcast tiling output."""

    # Problem dimensions
    output_shape: List[int]           # collapsed output shape
    input_shapes: List[List[int]]     # collapsed input shapes (N inputs)
    input_strides: List[List[int]]    # collapsed strides (stride=0 for broadcast ax)
    shape_len: int                    # collapsed dimension count
    scalar_flags: List[bool]          # per-input: is scalar (all dims=1)?

    # UB split
    ub_split_axis: int = 0
    ub_former: int = 0               # elements per UB tile on split axis
    ub_outer: int = 0                # tile count on split axis
    ub_tail: int = 0                 # last tile size on split axis

    # Multi-core split
    fused_product: int = 0           # ub_outer * outer dims product
    block_former: int = 0            # tiles per core
    block_num: int = 0               # virtual block count
    block_tail: int = 0
    core_num: int = 0

    # Branch selection
    branch: str = "onedim"           # onedim | ub_broadcast | nddma_broadcast | dynamic_ub
    sch_mode: int = 0                # 1=without_loop, 2=with_loop (NDDMA)
    is_dav3510: bool = False          # chip generation

    # UB budget
    max_elem_num: int = 0
    buffer_num: int = 0
    extra_size: int = 0

    def to_dict(self) -> dict:
        return {
            "output_shape": self.output_shape,
            "input_strides": self.input_strides,
            "scalar_flags": self.scalar_flags,
            "ub_split_axis": self.ub_split_axis,
            "ub_former": self.ub_former,
            "ub_outer": self.ub_outer,
            "ub_tail": self.ub_tail,
            "block_former": self.block_former,
            "block_num": self.block_num,
            "block_tail": self.block_tail,
            "core_num": self.core_num,
            "branch": self.branch,
            "sch_mode": self.sch_mode,
            "max_elem_num": self.max_elem_num,
        }


# ---------------------------------------------------------------------------
# Step 0: Dimension Collapse
# ---------------------------------------------------------------------------

def _pad_shapes(input_shapes: List[List[int]], output_shape: List[int]) -> List[List[int]]:
    """Left-pad all input shapes to match output rank."""
    rank = len(output_shape)
    padded = []
    for s in input_shapes:
        if len(s) < rank:
            padded.append([1] * (rank - len(s)) + list(s))
        else:
            padded.append(list(s))
    return padded


def _axis_has_broadcast_needs(input_shapes: List[List[int]], ax: int, src_idx: int) -> bool:
    """Check if axis 'ax' of input 'src_idx' needs broadcasting (dim=1 while another has >1)."""
    if input_shapes[src_idx][ax] != 1:
        return False
    for j, shape_j in enumerate(input_shapes):
        if j != src_idx and shape_j[ax] > 1:
            return True
    return False


def _compute_flags(input_shapes: List[List[int]]) -> List[int]:
    """Compute broadcast flag bitmaps per axis."""
    rank = len(input_shapes[0])
    num_inputs = len(input_shapes)
    flags = [0] * rank
    for ax in range(rank):
        flag = 0
        for i in range(num_inputs):
            if _axis_has_broadcast_needs(input_shapes, ax, i):
                flag |= (1 << i)
        flags[ax] = flag
    return flags


def _merge_axes(shape: List[int], flags: List[int]) -> Tuple[List[int], List[int]]:
    """Merge adjacent axes with identical broadcast flags."""
    if len(shape) <= 1:
        return list(shape), list(flags)
    merged_shape = [shape[0]]
    merged_flags = [flags[0]]
    for i in range(1, len(shape)):
        if flags[i] == merged_flags[-1]:
            merged_shape[-1] *= shape[i]
        else:
            merged_shape.append(shape[i])
            merged_flags.append(flags[i])
    return merged_shape, merged_flags


def _compute_strides(shape: List[int], is_broadcast_input: List[bool]) -> List[int]:
    """Compute strides right-to-left. Broadcast axes (stride=0) handled in single pass."""
    rank = len(shape)
    strides = [1] * rank
    for i in range(rank - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    for i in range(rank):
        if is_broadcast_input[i]:
            strides[i] = 0
    return strides


def dimension_collapse(
    input_shapes: List[List[int]],
    output_shape: List[int],
) -> Tuple[List[List[int]], List[int], List[List[int]], List[List[bool]]]:
    """
    Collapse dimensions: pad → flag → merge → strides.

    Returns: (collapsed_inputs, collapsed_output, all_strides, broadcast_masks)
    """
    padded = _pad_shapes(input_shapes, output_shape)

    flags = _compute_flags(padded)
    merged_output, merged_flags = _merge_axes(output_shape, flags)
    merged_rank = len(merged_output)

    merged_inputs = []
    for inp in padded:
        m_inp, _ = _merge_axes(inp, flags)
        merged_inputs.append(m_inp)

    all_strides = []
    all_masks = []
    for inp in merged_inputs:
        is_bc = [inp[i] == 1 and merged_output[i] > 1 for i in range(merged_rank)]
        all_masks.append(is_bc)
        all_strides.append(_compute_strides(inp, is_bc))

    return merged_inputs, merged_output, all_strides, all_masks


# ---------------------------------------------------------------------------
# Step 1: UB split calculation
# ---------------------------------------------------------------------------

def _calc_max_elem_num(cfg: _TilingConfig) -> int:
    """Calculate maximum elements that fit in UB, aligned to 256B."""
    raw = (cfg.ub_size - cfg.extra_size) * 8 // (cfg.buffer_num * cfg.max_dtype_bits)
    align = REPEAT_BYTES * 8 // cfg.min_dtype_bits
    return (raw // align) * align


def calc_ub_split(
    output_dims: List[int],
    max_elem_num: int,
) -> Tuple[int, int, int, int]:
    """
    UB split: from innermost axis outward, find the first axis that doesn't fit.

    Returns: (ub_split_axis, ub_former, ub_outer, ub_tail)
    """
    shape_len = len(output_dims)
    cur_product = 1
    ub_split_axis = 0
    all_fit = True

    for i in range(shape_len - 1, -1, -1):
        cur_product *= output_dims[i]
        if cur_product > max_elem_num:
            ub_split_axis = i
            cur_product //= output_dims[i]
            all_fit = False
            break

    if all_fit:
        cur_product //= output_dims[0]

    if shape_len == 1:
        ub_former = max_elem_num
    else:
        ub_former = max_elem_num // cur_product if cur_product > 0 else max_elem_num

    ub_outer = math.ceil(output_dims[ub_split_axis] / ub_former)
    ub_tail = output_dims[ub_split_axis] - (ub_former - 1) * ub_former if ub_outer > 1 else output_dims[ub_split_axis]

    if ub_tail <= 0:
        ub_tail = output_dims[ub_split_axis]

    return ub_split_axis, ub_former, ub_outer, ub_tail


# ---------------------------------------------------------------------------
# Step 2: Multi-core split
# ---------------------------------------------------------------------------

def _compute_fused_product(output_dims: List[int], ub_split_axis: int, ub_outer: int) -> int:
    """Compute fused product across ub_split_axis and outer dims."""
    product = ub_outer
    for i in range(ub_split_axis):
        product *= output_dims[i]
    return product


def calc_multicore_split(
    fused_product: int,
    core_num: int,
) -> Tuple[int, int, int]:
    """
    Multi-core split: flatten ub_split_axis and outer axes, distribute evenly.

    Returns: (block_former, block_num, block_tail)
    """
    block_former = math.ceil(fused_product / core_num)
    block_num = math.ceil(fused_product / block_former)
    block_tail = fused_product - (block_num - 1) * block_former
    return block_former, block_num, block_tail


def _should_shrink_cores(block_num: int, core_num: int, ub_former: int, cfg: _TilingConfig) -> bool:
    """Check if core utilization is poor enough to warrant shrinking."""
    if block_num >= core_num // 2:
        return False
    return ub_former * cfg.max_dtype_bytes * cfg.buffer_num > 8 * 1024


def _compute_shrunk_ub_former(output_dims: List[int], cfg: _TilingConfig, current_ub_former: int) -> int:
    """Compute smaller ub_former to feed more cores."""
    dim_length = 1
    for d in output_dims:
        dim_length *= d
    dim_per_core = dim_length * 2 // cfg.core_num
    aligned_per_core = (math.ceil(dim_per_core * cfg.max_dtype_bytes / CACHE_LINE)
                      * CACHE_LINE // cfg.max_dtype_bytes)
    if aligned_per_core <= 0:
        return current_ub_former
    new_ub_former = min(current_ub_former, aligned_per_core)
    lowest = (8 * 1024 // cfg.buffer_num // CACHE_LINE) * CACHE_LINE // cfg.max_dtype_bytes
    new_ub_former = max(new_ub_former, lowest)
    return new_ub_former


def _optimize_core_utilization(
    output_dims: List[int],
    ub_split_axis: int,
    cfg: _TilingConfig,
) -> Tuple[int, int, int, int]:
    """If block_num < core_num / 2, shrink max_elem_num to feed more cores."""
    max_elem_num = _calc_max_elem_num(cfg)
    ub_split_axis, ub_former, ub_outer, ub_tail = calc_ub_split(output_dims, max_elem_num)
    fused_product = _compute_fused_product(output_dims, ub_split_axis, ub_outer)
    block_former, block_num, block_tail = calc_multicore_split(fused_product, cfg.core_num)

    if _should_shrink_cores(block_num, cfg.core_num, ub_former, cfg):
        new_ub_former = _compute_shrunk_ub_former(output_dims, cfg, ub_former)
        if new_ub_former != ub_former:
            ub_former = new_ub_former
            ub_outer = math.ceil(output_dims[ub_split_axis] / ub_former)
            fused_product = _compute_fused_product(output_dims, ub_split_axis, ub_outer)
            block_former, block_num, block_tail = calc_multicore_split(fused_product, cfg.core_num)

    return block_num, cfg.core_num, ub_former, ub_outer


# ---------------------------------------------------------------------------
# Step 3: Branch selection
# ---------------------------------------------------------------------------

def _check_nlast_ub_broadcast(
    output_dims: List[int],
    input_strides: List[List[int]],
) -> bool:
    """Check if non-last axis broadcast should route to dynamic_ub."""
    shape_len = len(output_dims)
    last_axis = shape_len - 1
    for inp_strides in input_strides:
        if inp_strides[last_axis] == 0:
            return False
    for ax in range(shape_len - 1):
        for inp_strides in input_strides:
            if inp_strides[ax] == 0 and output_dims[ax] > 1:
                return output_dims[last_axis] >= CACHE_LINE // 2
    return False


def select_broadcast_branch(
    output_dims: List[int],
    input_strides: List[List[int]],
    is_dav3510: bool,
    dtype: str,
    force_nddma: bool = False,
) -> Tuple[str, int]:
    """
    Select the broadcast implementation branch.

    Returns: (branch_name, sch_mode)
    """
    if len(output_dims) == 1:
        return "onedim", 0

    if not is_dav3510:
        return "ub_broadcast", 0

    if force_nddma:
        sch_mode = 1 if len(output_dims) <= NDDMA_MAX_DIMS else 2
        return "nddma_broadcast", sch_mode

    if _check_nlast_ub_broadcast(output_dims, input_strides):
        return "dynamic_ub", 0

    dtype_bytes = {"fp16": 2, "bf16": 2, "int8": 1, "fp32": 4, "float": 4}.get(dtype, 2)
    if dtype_bytes <= 2:
        tail_dim = output_dims[-1]
        if (tail_dim * dtype_bytes) % 32 == 0:
            return "dynamic_ub", 0

    sch_mode = 1 if len(output_dims) <= NDDMA_MAX_DIMS else 2
    return "nddma_broadcast", sch_mode


# ---------------------------------------------------------------------------
# Step 4: OneDim tiling
# ---------------------------------------------------------------------------

def _maybe_shrink_onedim(
    dim_len: int, block_num: int, ub_former: int,
    cfg: _TilingConfig, idtype_bytes: int,
) -> Tuple[int, int, int, int, int]:
    """Shrink UB block size for OneDim when core utilization is poor."""
    if block_num >= cfg.core_num // 2:
        return ub_former, 0, 0, 0, block_num
    if ub_former * cfg.max_dtype_bytes * cfg.buffer_num <= 8 * 1024:
        return ub_former, 0, 0, 0, block_num
    dim_per_core = dim_len * 2 // cfg.core_num
    aligned_per_core = (math.ceil(dim_per_core * cfg.max_dtype_bytes / CACHE_LINE)
                      * CACHE_LINE // cfg.max_dtype_bytes)
    if aligned_per_core <= 0:
        return ub_former, 0, 0, 0, block_num
    new_ub_former = min(ub_former, aligned_per_core)
    lowest = (8 * 1024 // cfg.buffer_num // CACHE_LINE) * CACHE_LINE // cfg.max_dtype_bytes
    new_ub_former = max(new_ub_former, lowest)
    ub_former = new_ub_former
    ub_outer = math.ceil(dim_len / ub_former)
    ub_tail = dim_len - (ub_outer - 1) * ub_former
    if ub_tail <= 0:
        ub_tail = ub_former
    block_former = math.ceil(ub_outer / cfg.core_num)
    block_tail = ub_outer - (block_former - 1) * block_former
    if block_tail <= 0:
        block_tail = block_former
    block_num = math.ceil(ub_outer / block_former)
    return ub_former, ub_outer, ub_tail, block_tail, block_num


def tiling_onedim(output_dims: List[int], cfg: _TilingConfig, idtype_bytes: int) -> BroadcastTilingData:
    """OneDim tiling: single collapsed dimension, linear processing."""
    dim_len = output_dims[0]

    ub_former_byte = (cfg.ub_size - cfg.extra_size) // cfg.buffer_num
    ub_former = (ub_former_byte // CACHE_LINE) * CACHE_LINE // idtype_bytes

    ub_outer = math.ceil(dim_len / ub_former)
    ub_tail = dim_len - (ub_outer - 1) * ub_former
    if ub_tail <= 0:
        ub_tail = ub_former

    block_former = math.ceil(ub_outer / cfg.core_num)
    block_tail = ub_outer - (block_former - 1) * block_former
    if block_tail <= 0:
        block_tail = block_former
    block_num = math.ceil(ub_outer / block_former)

    ub_former, ub_outer, ub_tail, block_tail, block_num = _maybe_shrink_onedim(
        dim_len, block_num, ub_former, cfg, idtype_bytes,
    )

    return BroadcastTilingData(
        output_shape=output_dims,
        input_shapes=[[dim_len]],
        input_strides=[[1]],
        shape_len=1,
        scalar_flags=[],
        ub_split_axis=0,
        ub_former=ub_former,
        ub_outer=ub_outer,
        ub_tail=ub_tail,
        block_former=block_former,
        block_num=block_num,
        block_tail=block_tail,
        core_num=cfg.core_num,
        branch="onedim",
    )


# ---------------------------------------------------------------------------
# Step 5: Multi-dim tiling (UB-Broadcast / NDDMA)
# ---------------------------------------------------------------------------

def tiling_multidim(
    output_dims: List[int],
    input_strides: List[List[int]],
    cfg: _TilingConfig,
    branch: str,
    sch_mode: int = 0,
) -> BroadcastTilingData:
    """Multi-dim broadcast tiling (UB/NDDMA paths)."""
    shape_len = len(output_dims)

    max_elem_num = _calc_max_elem_num(cfg)
    ub_split_axis, ub_former, ub_outer, ub_tail = calc_ub_split(output_dims, max_elem_num)

    fused_product = _compute_fused_product(output_dims, ub_split_axis, ub_outer)
    block_former, block_num, block_tail = calc_multicore_split(fused_product, cfg.core_num)

    block_num, used_cores, _, _ = _optimize_core_utilization(
        output_dims, ub_split_axis, cfg,
    )

    fused_product = _compute_fused_product(output_dims, ub_split_axis, ub_outer)

    return BroadcastTilingData(
        output_shape=output_dims,
        input_shapes=[],
        input_strides=input_strides,
        shape_len=shape_len,
        scalar_flags=[],
        ub_split_axis=ub_split_axis,
        ub_former=ub_former,
        ub_outer=ub_outer,
        ub_tail=ub_tail,
        fused_product=fused_product,
        block_former=block_former,
        block_num=block_num,
        block_tail=block_tail,
        core_num=used_cores,
        branch=branch,
        sch_mode=sch_mode,
        max_elem_num=max_elem_num,
        buffer_num=cfg.buffer_num,
        extra_size=cfg.extra_size,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_broadcast_tiling(params: BroadcastTilingInput) -> BroadcastTilingData:
    """
    Compute broadcast tiling for binary/multi-input elementwise ops.

    Args:
        params: BroadcastTilingInput with shapes, dtype, and platform config.

    Returns:
        BroadcastTilingData with all tiling parameters
    """
    dtype_bytes_map = {"fp16": 2, "float16": 2, "bf16": 2, "bfloat16": 2,
                       "int8": 1, "fp32": 4, "float32": 4, "float": 4}
    dtype_bits_map = {"fp16": 16, "float16": 16, "bf16": 16, "bfloat16": 16,
                      "int8": 8, "fp32": 32, "float32": 32, "float": 32}

    dtype = params.dtype
    max_dtype_bytes = max(dtype_bytes_map.get(d, 2) for d in [dtype])
    max_dtype_bits = max_dtype_bytes * 8
    min_dtype_bits = max_dtype_bytes * 8
    idtype_bytes = dtype_bytes_map.get(dtype, 2)

    cfg = _TilingConfig(
        ub_size=params.ub_size,
        extra_size=params.extra_size,
        buffer_num=len(params.input_shapes) + 1,
        core_num=params.core_num,
        max_dtype_bytes=max_dtype_bytes,
        max_dtype_bits=max_dtype_bits,
        min_dtype_bits=min_dtype_bits,
    )

    collapsed_inputs, collapsed_output, all_strides, all_masks = dimension_collapse(
        params.input_shapes, params.output_shape,
    )
    scalar_flags = [all(d == 1 for d in inp) for inp in collapsed_inputs]

    branch, sch_mode = select_broadcast_branch(
        collapsed_output, all_strides, params.is_dav3510, params.dtype,
        params.force_nddma,
    )

    if branch == "onedim":
        result = tiling_onedim(collapsed_output, cfg, idtype_bytes)
    else:
        result = tiling_multidim(
            collapsed_output, all_strides, cfg, branch, sch_mode,
        )

    result.input_shapes = collapsed_inputs
    result.scalar_flags = scalar_flags
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    tiling = compute_broadcast_tiling(BroadcastTilingInput(
        input_shapes=[[4, 3, 8], [1, 3, 8]],
        output_shape=[4, 3, 8],
        dtype="fp16",
        is_dav3510=True,
    ))
    logger.info(json.dumps(tiling.to_dict(), indent=2))
