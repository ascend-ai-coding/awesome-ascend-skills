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
Sort Tiling Algorithm — Fallback Reference Implementation.

Covers: Top-K Sort with M-way merge sort.
Patterns: A (single-core) / B (multi-core one-level merge) / C (multi-core two-level merge).
"""

import math
import logging
from dataclasses import dataclass
from typing import Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
M_WAY = 4                       # M-way merge (MrgSort API constant)
PROPOSAL_SIZE = 8               # proposal format: 4B value + 4B index
TOPK_SORT_NUM = 32              # Sort API alignment granularity
DEFAULT_TILE_SIZE = 4096        # empirically optimal tile size


@dataclass
class SortTilingData:
    """Sort tiling output."""

    # Problem
    element_count: int            # total element count (was N)
    top_k: int                    # Top-K value (was K)
    dtype: str = "fp16"
    dtype_bytes: int = 2

    # Routing
    pattern: str = "A"           # "A" | "B" | "C"
    tile_size: int = 0           # elements per tile
    total_tiles: int = 0         # total tile count
    core_num: int = 0            # actual cores used

    # Per-core parameters
    tiles_per_core: int = 0      # tiles per core (= S_c for Pattern C)
    elements_per_core: int = 0   # elements per core (= E_c)
    front_core_tiles: int = 0    # tile count for first (coreNum-1) cores
    last_core_tiles: int = 0     # tile count for last core
    last_tile_size: int = 0      # last tile may be partial

    # Merge parameters
    merge_rounds_phase2: int = 0     # ceil(log_M(S_c))
    merge_rounds_phase3: int = 0     # ceil(log_M(coreNum)) - 1
    once_max_merge: int = 0          # max elements per merge batch (Phase 2/3)
    once_max_output: int = 0         # max elements per output batch (Phase 4)

    # Workspace
    workspace_bytes: int = 0

    # UB budget
    ub_size: int = 0
    sort_bytes_per_elem: int = 0     # Phase 1: ~34B (fp16) / ~32B (fp32)
    merge_bytes_per_elem: int = 64   # Phase 2/3: 2*M*8B = 64B
    output_bytes_per_elem: int = 0   # Phase 4: varies by dtype

    def to_dict(self) -> dict:
        return {
            "element_count": self.element_count,
            "top_k": self.top_k,
            "dtype": self.dtype,
            "pattern": self.pattern,
            "tile_size": self.tile_size,
            "total_tiles": self.total_tiles,
            "core_num": self.core_num,
            "tiles_per_core": self.tiles_per_core,
            "elements_per_core": self.elements_per_core,
            "front_core_tiles": self.front_core_tiles,
            "last_core_tiles": self.last_core_tiles,
            "last_tile_size": self.last_tile_size,
            "merge_rounds_phase2": self.merge_rounds_phase2,
            "merge_rounds_phase3": self.merge_rounds_phase3,
            "once_max_merge": self.once_max_merge,
            "once_max_output": self.once_max_output,
            "workspace_bytes": self.workspace_bytes,
        }


# ---------------------------------------------------------------------------
# Step 1: Calculate tile_size from UB constraints
# ---------------------------------------------------------------------------

def calc_sort_bytes_per_elem(dtype: str) -> int:
    """
    Per-element UB cost for Phase 1 sort operation.

    For FP16: sizeof(dtype) + sizeof(float) + sizeof(uint32) + PROPOSAL_SIZE
              + concatTmpPerElem + sortTmpPerElem ≈ 34B
    For FP32: sizeof(float) + sizeof(uint32) + PROPOSAL_SIZE
              + concatTmpPerElem + sortTmpPerElem ≈ 32B (no Cast needed)
    """
    dtype_sizes = {"fp16": 2, "bf16": 2, "fp32": 4, "float": 4}
    dsize = dtype_sizes.get(dtype, 2)
    if dsize <= 2:
        return dsize + 4 + 4 + PROPOSAL_SIZE + 8 + 8
    return dsize + 4 + PROPOSAL_SIZE + 8 + 8


def calc_output_bytes_per_elem(dtype: str) -> int:
    """Phase 4 output: 2*M*8B (merge I/O) + M*(4B value + 4B index + sizeof(dtype))."""
    dsize_map = {"fp16": 2, "bf16": 2, "fp32": 4, "float": 4}
    dsize = dsize_map.get(dtype, 2)
    return 2 * M_WAY * PROPOSAL_SIZE + M_WAY * (4 + 4 + dsize)


def calc_tile_size(ub_size: int, dtype: str) -> int:
    """Calculate tile_size from UB capacity."""
    sort_bpe = calc_sort_bytes_per_elem(dtype)
    theoretical = (ub_size // sort_bpe // TOPK_SORT_NUM) * TOPK_SORT_NUM
    return min(DEFAULT_TILE_SIZE, theoretical)


# ---------------------------------------------------------------------------
# Step 2: Pattern routing
# ---------------------------------------------------------------------------

def route_pattern(
    element_count: int,
    tile_size: int,
    core_num: int,
) -> Tuple[str, int, int]:
    """
    Route to Pattern A / B / C based on element_count vs tile_size * core_num.

    Returns: (pattern, total_tiles, elements_per_core)
    """
    tile_product = tile_size * core_num

    if element_count <= tile_size:
        return "A", 1, element_count

    total_tiles = math.ceil(element_count / tile_size)

    if element_count <= tile_product:
        return "B", total_tiles, total_tiles * tile_size // core_num + tile_size

    return "C", total_tiles, 0


# ---------------------------------------------------------------------------
# Step 3: Pattern C — tile distribution
# ---------------------------------------------------------------------------

def calc_pattern_c_tiling(
    element_count: int,
    tile_size: int,
    core_num: int,
) -> Tuple[int, int, int, int, int]:
    """
    Pattern C: distribute tiles across cores for two-level merge.

    Returns: (total_tiles, front_core_tiles, last_core_tiles, used_cores, last_tile_size)
    """
    total_tiles = math.ceil(element_count / tile_size)
    front_core_tiles = math.ceil(total_tiles / core_num)
    used_cores = math.ceil(total_tiles / front_core_tiles)
    last_core_tiles = total_tiles - (used_cores - 1) * front_core_tiles
    last_tile_size = element_count - tile_size * (total_tiles - 1)

    return total_tiles, front_core_tiles, last_core_tiles, used_cores, last_tile_size


# ---------------------------------------------------------------------------
# Step 4: Workspace calculation
# ---------------------------------------------------------------------------

def calc_workspace_size(elements_per_core: int, used_cores: int) -> int:
    """Workspace for two-level merge sort, double-buffered, 32B aligned."""
    ws_per_core = ((elements_per_core * PROPOSAL_SIZE * 2 + 31) // 32) * 32
    return used_cores * ws_per_core


# ---------------------------------------------------------------------------
# Step 5: Merge batch sizes
# ---------------------------------------------------------------------------

def calc_batch_sizes(ub_size: int, dtype: str) -> Tuple[int, int]:
    """
    Calculate onceMaxElements for merge (Phase 2/3) and output (Phase 4).

    Returns: (once_max_merge, once_max_output)
    """
    merge_bytes_per_elem = 2 * M_WAY * PROPOSAL_SIZE
    output_bpe = calc_output_bytes_per_elem(dtype)

    once_max_merge = (ub_size // merge_bytes_per_elem // TOPK_SORT_NUM) * TOPK_SORT_NUM
    once_max_output = (ub_size // output_bpe // TOPK_SORT_NUM) * TOPK_SORT_NUM

    return once_max_merge, once_max_output


# ---------------------------------------------------------------------------
# Step 6: Merge rounds
# ---------------------------------------------------------------------------

def calc_merge_rounds(tiles_per_core: int, core_num: int) -> Tuple[int, int]:
    """
    Calculate merge rounds for Phase 2 and Phase 3.

    Phase 2: intra-core merge → ceil(log_M(tiles_per_core))
    Phase 3: inter-core merge → ceil(log_M(core_num)) - 1

    Returns: (phase2_rounds, phase3_rounds)
    """
    if tiles_per_core <= 0:
        return 0, 0
    p2 = math.ceil(math.log(max(tiles_per_core, 2), M_WAY))
    p3 = max(0, math.ceil(math.log(max(core_num, 2), M_WAY)) - 1)
    return p2, p3


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _fill_pattern_a(result: SortTilingData, element_count: int) -> None:
    """Populate fields for Pattern A (single-core, single-tile)."""
    result.core_num = 1
    result.tiles_per_core = 1
    result.elements_per_core = element_count
    result.front_core_tiles = 1
    result.last_core_tiles = 1
    result.last_tile_size = element_count
    result.merge_rounds_phase2 = 0
    result.merge_rounds_phase3 = 0
    result.pattern = "A"


def _fill_pattern_b(
    result: SortTilingData, total_tiles: int, tile_size: int,
    element_count: int, core_num: int,
) -> None:
    """Populate fields for Pattern B (multi-core one-level merge)."""
    result.core_num = min(core_num, total_tiles)
    result.tiles_per_core = 1
    result.elements_per_core = tile_size
    result.front_core_tiles = 1
    result.last_core_tiles = 1
    result.last_tile_size = element_count - (total_tiles - 1) * tile_size
    result.merge_rounds_phase2 = 0
    result.merge_rounds_phase3 = math.ceil(math.log(max(total_tiles, 2), M_WAY)) - 1
    result.pattern = "B"


def _fill_pattern_c(
    result: SortTilingData, element_count: int, tile_size: int,
    core_num: int, ub_size_and_dtype: Tuple[int, str],
) -> None:
    """Populate fields for Pattern C (multi-core two-level merge)."""
    ub_size, dtype = ub_size_and_dtype
    total_tiles, front_core_tiles, last_core_tiles, used_cores, last_tile_size = \
        calc_pattern_c_tiling(element_count, tile_size, core_num)

    elements_per_core = front_core_tiles * tile_size
    tiles_per_core_first = front_core_tiles

    p2_rounds, p3_rounds = calc_merge_rounds(tiles_per_core_first, used_cores)
    once_max_merge, once_max_output = calc_batch_sizes(ub_size, dtype)
    workspace_bytes = calc_workspace_size(elements_per_core, used_cores)

    result.core_num = used_cores
    result.tiles_per_core = tiles_per_core_first
    result.elements_per_core = elements_per_core
    result.front_core_tiles = front_core_tiles
    result.last_core_tiles = last_core_tiles
    result.last_tile_size = last_tile_size
    result.total_tiles = total_tiles
    result.merge_rounds_phase2 = p2_rounds
    result.merge_rounds_phase3 = p3_rounds
    result.once_max_merge = once_max_merge
    result.once_max_output = once_max_output
    result.workspace_bytes = workspace_bytes
    result.pattern = "C"


def compute_sort_tiling(
    element_count: int,
    top_k: int,
    dtype: str = "fp16",
    ub_size: int = 248 * 1024,
    core_num: int = 64,
) -> SortTilingData:
    """
    Compute sort (Top-K) tiling.

    Args:
        element_count: Total element count
        top_k: Top-K value (number of largest elements to return)
        dtype: Data type
        ub_size: Unified Buffer size in bytes
        core_num: Available AI Core count

    Returns:
        SortTilingData with all tiling parameters
    """
    dtype_map = {"fp16": 2, "float16": 2, "bf16": 2, "bfloat16": 2,
                 "fp32": 4, "float32": 4, "float": 4}
    dtype_bytes = dtype_map.get(dtype, 2)
    sort_bpe = calc_sort_bytes_per_elem(dtype)

    tile_size = calc_tile_size(ub_size, dtype)
    pattern, total_tiles, _ = route_pattern(element_count, tile_size, core_num)

    result = SortTilingData(
        element_count=element_count, top_k=top_k, dtype=dtype, dtype_bytes=dtype_bytes,
        ub_size=ub_size,
        sort_bytes_per_elem=sort_bpe,
        merge_bytes_per_elem=2 * M_WAY * PROPOSAL_SIZE,
        output_bytes_per_elem=calc_output_bytes_per_elem(dtype),
        tile_size=tile_size,
        total_tiles=total_tiles,
    )

    if pattern == "A":
        _fill_pattern_a(result, element_count)
    elif pattern == "B":
        _fill_pattern_b(result, total_tiles, tile_size, element_count, core_num)
    else:
        _fill_pattern_c(result, element_count, tile_size, core_num, (ub_size, dtype))

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    tiling = compute_sort_tiling(element_count=500000, top_k=100, dtype="fp16")
    logger.info("=== Sort(element_count=500000, top_k=100, fp16) ===")
    logger.info(json.dumps(tiling.to_dict(), indent=2))

    tiling2 = compute_sort_tiling(element_count=50000, top_k=50, dtype="fp16")
    logger.info("\n=== Sort(element_count=50000, top_k=50, fp16) ===")
    logger.info(json.dumps(tiling2.to_dict(), indent=2))

    tiling3 = compute_sort_tiling(element_count=2000, top_k=10, dtype="fp32")
    logger.info("\n=== Sort(element_count=2000, top_k=10, fp32) ===")
    logger.info(json.dumps(tiling3.to_dict(), indent=2))
