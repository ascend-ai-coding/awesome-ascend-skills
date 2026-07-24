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
Conversion Tiling Algorithm — Small-Channel Transpose.

Covers: Transpose, Permute, NCHW→NHWC small-channel data rearrangement.
Models problem as [channel_count, flat_length] → [flat_length, channel_count].
"""

import math
import logging
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALIGN_ELEM = 32         # tile_elements alignment (32 elements for 32B/FP32 compatibility)
REPEAT_MAX = 255         # TransDataTo5HD repeats上限
VNCHWCONV_BLOCK = 16     # vnchwconv half block size


@dataclass
class ConversionTilingInput:
    """Input parameters for compute_conversion_tiling."""
    input_shape: List[int]
    perm: List[int]
    dtype: str = "fp16"
    ub_size: int = 248 * 1024
    core_num: int = 64
    reserved_bytes: int = 0


@dataclass
class _ConversionConfig:
    """Packed conversion config to reduce argument count."""
    ub_size: int
    reserved_bytes: int
    core_num: int


@dataclass
class ConversionTilingData:
    """Small-channel transpose tiling output."""

    # Problem model: [channel_count, flat_length] → [flat_length, channel_count]
    channel_count: int                      # small channel count
    flat_length: int                        # flattened long axis length

    # Tile parameters
    tile_elements: int                      # effective elements per tile
    tile_elements_aligned: int              # aligned tile width (32-align)
    repeats: int                            # TransDataTo5HD repeats (= tile_elements_aligned / 16)
    total_tiles: int                        # total tile count
    block_dim: int                          # actual vector cores used

    # Buffer sizes
    ub_bytes_per_tile: int                  # UB budget per tile
    ub_size: int                            # total UB size
    reserved_bytes: int                     # reserved bytes

    # Per-tile loop info
    tiles_per_core: int = 0
    core_num: int = 0

    def to_dict(self) -> dict:
        return {
            "channel_count": self.channel_count,
            "flat_length": self.flat_length,
            "tile_elements": self.tile_elements,
            "tile_elements_aligned": self.tile_elements_aligned,
            "repeats": self.repeats,
            "total_tiles": self.total_tiles,
            "block_dim": self.block_dim,
            "ub_bytes_per_tile": self.ub_bytes_per_tile,
            "tiles_per_core": self.tiles_per_core,
            "core_num": self.core_num,
        }


# ---------------------------------------------------------------------------
# Axis collapse: reshape any permute into [channel_count, flat_length]
# ---------------------------------------------------------------------------

def collapse_to_channel_flat(
    input_shape: List[int],
    perm: List[int],
) -> Tuple[int, int]:
    """
    Collapse a transpose into [channel_count, flat_length]→[flat_length, channel_count] form.

    channel_count = product of all channel-like dimensions being moved
    flat_length = product of all remaining dimensions kept in order
    """
    c_candidates = []
    for i, p in enumerate(perm):
        if p != i:
            c_candidates.append((input_shape[i], i))

    if c_candidates:
        channel_count = min(c_candidates)[0]
    else:
        channel_count = 1

    flat_length = 1
    for s in input_shape:
        flat_length *= s
    flat_length //= channel_count

    return channel_count, flat_length


# ---------------------------------------------------------------------------
# UB budget calculation
# ---------------------------------------------------------------------------

def calc_ub_budget(channel_count: int, tile_elements_aligned: int) -> int:
    """
    UB budget formula: tile_elements_aligned * (16 * channel_count + 32) bytes.
    """
    return tile_elements_aligned * (16 * channel_count + 32)


# ---------------------------------------------------------------------------
# Tile size calculation
# ---------------------------------------------------------------------------

def calc_tile_size(
    channel_count: int,
    flat_length: int,
    cfg: _ConversionConfig,
) -> Tuple[int, int, int, int]:
    """
    Calculate tile_elements / tile_elements_aligned / repeats / total_tiles.

    Strategy: maximize single tile size while spreading tiles across all cores.
    """
    per_elem_bytes = 16 * channel_count + 32
    ub_budget = cfg.ub_size - cfg.reserved_bytes
    tile_n_max = (ub_budget // per_elem_bytes // ALIGN_ELEM) * ALIGN_ELEM
    repeats_max_bytes = 255 * VNCHWCONV_BLOCK
    tile_n_max = min(tile_n_max, repeats_max_bytes)

    tile_n = ((flat_length + cfg.core_num - 1) // cfg.core_num + ALIGN_ELEM - 1)
    tile_n = (tile_n // ALIGN_ELEM) * ALIGN_ELEM

    if tile_n > tile_n_max:
        min_tiles = math.ceil(flat_length / tile_n_max)
        aligned_tiles = ((min_tiles + cfg.core_num - 1) // cfg.core_num) * cfg.core_num
        tile_n = ((flat_length + aligned_tiles - 1) // aligned_tiles + ALIGN_ELEM - 1)
        tile_n = (tile_n // ALIGN_ELEM) * ALIGN_ELEM

    tile_na = tile_n
    if tile_na % VNCHWCONV_BLOCK != 0:
        tile_na = ((tile_na + VNCHWCONV_BLOCK - 1) // VNCHWCONV_BLOCK) * VNCHWCONV_BLOCK

    repeats = tile_na // VNCHWCONV_BLOCK
    total_tiles = math.ceil(flat_length / tile_n)

    return tile_n, tile_na, repeats, total_tiles


# ---------------------------------------------------------------------------
# Multi-core split
# ---------------------------------------------------------------------------

def calc_multicore(
    total_tiles: int, core_num: int,
) -> Tuple[int, int, int]:
    """
    Multi-core allocation: distribute tiles evenly.

    Returns: (block_dim, tiles_per_core, used_cores)
    """
    block_dim = min(core_num, total_tiles)
    tiles_per_core = math.ceil(total_tiles / block_dim)
    return block_dim, tiles_per_core, block_dim


# ---------------------------------------------------------------------------
# Offset table generation
# ---------------------------------------------------------------------------

def generate_offset_table(tile_elements_aligned: int, channel_count: int) -> List[int]:
    """
    Generate Gather offset table for TransDataTo5HD output.

    For each 16-half block, only the first channel_count positions are valid.
    """
    offsets = [0] * (tile_elements_aligned * channel_count)
    for p in range(tile_elements_aligned):
        for c in range(channel_count):
            offsets[p * channel_count + c] = (p * VNCHWCONV_BLOCK + c) * 2
    return offsets


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_conversion_tiling(params: ConversionTilingInput) -> ConversionTilingData:
    """
    Compute small-channel transpose tiling.

    Args:
        params: ConversionTilingInput with input_shape, perm, dtype, and platform config.

    Returns:
        ConversionTilingData with all tiling parameters
    """
    cfg = _ConversionConfig(ub_size=params.ub_size, reserved_bytes=params.reserved_bytes, core_num=params.core_num)

    channel_count, flat_length = collapse_to_channel_flat(params.input_shape, params.perm)

    if channel_count > 16:
        raise ValueError(
            f"channel_count={channel_count} > 16, small-channel transpose may not be optimal. "
            f"Consider a general transpose approach."
        )

    tile_n, tile_na, repeats, total_tiles = calc_tile_size(
        channel_count, flat_length, cfg,
    )

    block_dim, tiles_per_core, used_cores = calc_multicore(total_tiles, params.core_num)
    ub_bytes = calc_ub_budget(channel_count, tile_na)

    return ConversionTilingData(
        channel_count=channel_count,
        flat_length=flat_length,
        tile_elements=tile_n,
        tile_elements_aligned=tile_na,
        repeats=repeats,
        total_tiles=total_tiles,
        block_dim=block_dim,
        ub_bytes_per_tile=ub_bytes,
        ub_size=params.ub_size,
        reserved_bytes=params.reserved_bytes,
        tiles_per_core=tiles_per_core,
        core_num=used_cores,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    H, W = 224, 224
    tiling = compute_conversion_tiling(ConversionTilingInput(
        input_shape=[3, H, W],
        perm=[1, 2, 0],
        dtype="fp16",
    ))
    logger.info(json.dumps(tiling.to_dict(), indent=2))
