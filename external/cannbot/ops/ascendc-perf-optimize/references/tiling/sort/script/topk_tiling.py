#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Executable DAV_3510 TopK host-tiling reference model.

The model preserves routing order, shape normalization, UB accounting, workspace
layout, and the mode-specific field contract. CANN SDK queries that cannot be
reproduced outside the SDK are surfaced in ``sdk_queries`` instead of guessed.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

INT64_MAX = (1 << 63) - 1
INT32_MAX = (1 << 31) - 1
BIN_NUM = 256
ALIGN_BYTES = 32
SIMT_RESERVED_BYTES = 32 * 1024
SYSTEM_WORKSPACE_BYTES = 16 * 1024 * 1024
TOPK_K_MAX = 2048
SORT_INT32_RANGE_MAX = 1_073_741_823
SORT_UB_CONST_INT32 = 4096
SORT_UB_CONST_INT64 = 7168
SORT_MIN_TILE = 1024

DTYPE_BYTES = {
    "int8": 1,
    "uint8": 1,
    "int32": 4,
    "int64": 8,
    "fp16": 2,
    "fp32": 4,
    "bf16": 2,
}

TILING_KEY = {
    "int8": 1001,
    "uint8": 2001,
    "int32": 1003,
    "int64": 1004,
    "fp16": 3002,
    "fp32": 3003,
    "bf16": 4002,
}

MODE = {
    "single_core": 1,
    "multi_core": 2,
    "multi_core_optimized": 4,
    "sort_and_topk": 5,
    "fp32_inter_core_merge": 6,
    "fp32_intra_core_merge": 7,
    "non_last_small_axis": 8,
}


def ceil_div(value: int, divisor: int) -> int:
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    return (value + divisor - 1) // divisor


def align_up(value: int, alignment: int) -> int:
    return ceil_div(value, alignment) * alignment


def checked_product(values: Sequence[int], name: str) -> int:
    result = 1
    for value in values:
        if value <= 0:
            raise ValueError(f"{name} dimensions must be positive")
        result *= value
        if result > INT64_MAX:
            raise OverflowError(f"{name} product exceeds int64")
    return result


@dataclass(frozen=True)
class PlatformSpec:
    core_num: int
    ub_size: int
    ub_block_size: int = ALIGN_BYTES
    system_workspace_bytes: int = SYSTEM_WORKSPACE_BYTES

    @property
    def fallback_ub_budget(self) -> int:
        return self.ub_size - 8 * 1024

    def validate(self) -> None:
        if self.core_num <= 0:
            raise ValueError("core_num must be positive")
        if self.ub_size <= SIMT_RESERVED_BYTES:
            raise ValueError("ub_size must exceed the 32 KiB SIMT reservation")
        if self.ub_block_size <= 0:
            raise ValueError("ub_block_size must be positive")


@dataclass(frozen=True)
class TopKRequest:
    shape: Tuple[int, ...]
    k: int
    dim: int = -1
    dtype: str = "fp16"
    largest: bool = True
    sorted: bool = True
    indices_dtype: str = "int64"


@dataclass(frozen=True)
class SdkOverrides:
    """Optional results from the target CANN SDK.

    ``merge_tmp_bytes`` is the result of GetConcatTmpSize for the selected
    small MergeSort route. ``sort_tmp_bytes`` is the result of
    GetSortMaxMinTmpSize for SortAndTopK.
    """

    merge_tmp_bytes: Optional[int] = None
    sort_tmp_bytes: Optional[int] = None


@dataclass
class TopKTilingPlan:
    route: str
    tiling_key: int
    block_dim: int
    local_memory_bytes: int
    workspace_bytes: int
    normalized: Dict[str, int]
    mode_type: Optional[int] = None
    tiling_data: Dict[str, int] = field(default_factory=dict)
    launch_params: Dict[str, Any] = field(default_factory=dict)
    sdk_queries: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_request(request: TopKRequest) -> Dict[str, int]:
    if not request.shape or len(request.shape) > 8:
        raise ValueError("rank must be in [1, 8]")
    if request.dtype not in DTYPE_BYTES:
        raise ValueError(f"unsupported dtype: {request.dtype}")
    if request.indices_dtype not in ("int32", "int64"):
        raise ValueError("indices_dtype must be int32 or int64")
    rank = len(request.shape)
    dim = request.dim + rank if request.dim < 0 else request.dim
    if dim < 0 or dim >= rank:
        raise ValueError("dim is out of range")
    axis_len = request.shape[dim]
    if request.k < 1 or request.k > min(axis_len, TOPK_K_MAX):
        raise ValueError("k must be in [1, min(axis_len, 2048)]")
    outer = checked_product(request.shape[:dim] or (1,), "outer")
    inner = checked_product(request.shape[dim + 1:] or (1,), "inner")
    rows = outer * inner
    if rows > INT64_MAX:
        raise OverflowError("row count exceeds int64")
    return {
        "rank": rank,
        "dim": dim,
        "outer": outer,
        "axis_len": axis_len,
        "inner": inner,
        "rows": rows,
        "non_last": int(dim + 1 < rank),
    }


def topk_api_tmp_size(
    need_data_num: int,
    k: int,
    dtype: str,
    largest: bool,
    sorted_output: bool,
    now_tile_size: int,
    single_core_mode: bool = False,
) -> int:
    """Mirror the fixed Arch35 RADIX_SELECT temporary-buffer layout."""
    inner = align_up(need_data_num, ALIGN_BYTES)
    effective_k = min(need_data_num if single_core_mode else now_tile_size, k)
    aligned_k = align_up(effective_k, ALIGN_BYTES)
    dtype_size = DTYPE_BYTES[dtype]
    input_index = inner * 4
    candidate_value = max(inner, 256) * dtype_size if dtype_size == 8 else inner * dtype_size
    candidate_index = max(input_index, 512)
    work_value = inner * dtype_size if (not largest or dtype != "uint8") else 0
    sorted_output_bytes = 512 + aligned_k * (11 + dtype_size) if sorted_output else 0
    return input_index + candidate_value + candidate_index + work_value + sorted_output_bytes


def _single_block_runtime(axis_len: int, tile: int, dtype_size: int, index_size: int, k: int) -> int:
    batches = tile // axis_len
    aligned_axis = align_up(axis_len, ALIGN_BYTES)
    return batches * (
        aligned_axis * dtype_size
        + align_up(k * dtype_size, ALIGN_BYTES)
        + align_up(k * index_size, ALIGN_BYTES)
        + align_up(k * 4, ALIGN_BYTES)
    )


def _single_core_runtime(
    axis_len: int, tile: int, dtype_size: int, index_size: int, k: int, sorted_output: bool
) -> int:
    tile_count = ceil_div(axis_len, tile)
    balanced_tile = ceil_div(axis_len, tile_count)
    output_count = min(balanced_tile, k)
    axis_index_size = 4 if axis_len <= INT32_MAX else 8
    total = align_up(balanced_tile, ALIGN_BYTES) * dtype_size
    total += align_up(output_count * dtype_size, ALIGN_BYTES)
    total += align_up(output_count * index_size, ALIGN_BYTES)
    total += align_up(output_count * 4, ALIGN_BYTES)
    total += BIN_NUM * 4 + BIN_NUM * axis_index_size
    total += align_up(tile_count * 4, ALIGN_BYTES)
    total += align_up(balanced_tile * 4, ALIGN_BYTES)
    total += BIN_NUM * axis_index_size
    if sorted_output and k * dtype_size <= 8000:
        total += align_up(k * index_size, ALIGN_BYTES)
    return total


def _multi_core_runtime(
    axis_len: int,
    tile: int,
    core_num: int,
    dtype_size: int,
    index_size: int,
    k: int,
) -> int:
    tile_count = ceil_div(axis_len, tile)
    rounds = ceil_div(tile_count, core_num)
    total = index_size * BIN_NUM * (rounds + 1) + align_up(4 * rounds, ALIGN_BYTES) * 2
    factor = dtype_size * 2 + index_size * 2
    if tile < k:
        factor += dtype_size + index_size + 4
    else:
        total += align_up(k * 4, ALIGN_BYTES)
        total += align_up(k * dtype_size, ALIGN_BYTES)
        total += align_up(k * index_size, ALIGN_BYTES)
    return total + factor * tile


def _optimized_runtime(tile: int, dtype_size: int, index_size: int, k: int, ub_block: int) -> int:
    return (
        align_up(tile, ub_block) * dtype_size
        + align_up(tile, ub_block) * 4
        + align_up(k * dtype_size, ub_block)
        + align_up(k * 4, ub_block)
        + align_up(k * index_size, ub_block)
    )


def _small_axis_plan(request: TopKRequest, normalized: Dict[str, int], platform: PlatformSpec) -> TopKTilingPlan:
    axis_len = normalized["axis_len"]
    outer = normalized["outer"]
    inner = normalized["inner"]
    step = 16 if axis_len == 17 else 32
    tiles_per_outer = max(platform.core_num // outer, 1)
    desired = align_up(ceil_div(inner, tiles_per_outer), step)
    tile_width = max(min(desired, 4096), step)
    usable_ub = platform.ub_size - SIMT_RESERVED_BYTES
    while tile_width >= step:
        row_bytes = align_up(tile_width * 2, ALIGN_BYTES)
        mask_bytes = align_up(ceil_div(tile_width, 8), ALIGN_BYTES)
        required = axis_len * row_bytes * 2 + row_bytes * 2 + mask_bytes * 2
        if required <= usable_ub:
            break
        tile_width -= step
    if tile_width < step:
        raise ValueError("small-axis network does not fit in UB")
    tile_count = outer * ceil_div(inner, tile_width)
    params = {
        "outer": outer,
        "axisLen": axis_len,
        "inner": inner,
        "isLargest": int(request.largest),
        "tileWidth": tile_width,
        "usedCores": min(platform.core_num, tile_count),
        "localMemoryBytes": usable_ub,
    }
    return TopKTilingPlan(
        route="fixed_small_axis",
        tiling_key=900001,
        block_dim=params["usedCores"],
        local_memory_bytes=usable_ub,
        workspace_bytes=0,
        normalized=normalized,
        launch_params=params,
    )


def _fallback_use_transpose_table(d_chunk: int, merge_pad: int, compute_size: int) -> bool:
    if d_chunk > 128 or d_chunk < 32 or d_chunk % 32:
        return False
    if d_chunk // 32 not in (1, 2, 3, 4):
        return False
    return merge_pad * compute_size >= 4 * d_chunk and merge_pad >= d_chunk


def _fallback_radix_floor(count: int, dtype_size: int) -> int:
    aligned = align_up(count, 32)
    if dtype_size == 8:
        return 8 * max(aligned, 512) + max(4 * aligned, 2048) + 8 * max(aligned, 512)
    return dtype_size * aligned + max(4 * aligned, 1024) + dtype_size * aligned


def _fallback_sort_tmp(count: int, dtype_size: int) -> int:
    return 512 + (11 + dtype_size) * align_up(count, 32) + 1024


def _fallback_ub_total(
    g: int,
    d_chunk: int,
    storage_size: int,
    compute_size: int,
    k: int,
    tmp_size: int,
    params: Dict[str, int],
    use_transpose_table: bool,
) -> int:
    k_pad = params["kPad"]
    k_idx_pad = params["kIdxPad"]
    merge_pad = params["mergePad"]
    is_16 = storage_size != compute_size
    is_64 = compute_size == 8
    total = g * d_chunk * compute_size + d_chunk * 4
    total += 2 * (g * k_pad * compute_size + g * k_idx_pad * 4)
    total += tmp_size + g * k_idx_pad * 8
    if not (is_64 and params["branch"] == 0):
        total += g * merge_pad * compute_size + g * merge_pad * 4
        total += 8 * 128 * 4
    if is_16:
        total += g * d_chunk * storage_size + g * align_up(k, 32 // storage_size) * storage_size
    if is_64:
        total += max(d_chunk, merge_pad) * (compute_size + 4)
    if params["branch"] == 1:
        total += g * d_chunk * storage_size + d_chunk * 4
        if use_transpose_table:
            total += g * d_chunk * 4
    return total


def _fallback_params(request: TopKRequest, normalized: Dict[str, int], platform: PlatformSpec) -> Dict[str, int]:
    storage_size = DTYPE_BYTES[request.dtype]
    compute_size = 4 if storage_size == 2 else storage_size
    branch = 0 if normalized["inner"] == 1 else 1
    k_pad = align_up(request.k, 32 // (4 if compute_size == 8 else compute_size))
    k_idx_pad = align_up(request.k, 8)
    merge_pad = align_up(2 * k_pad, 32)
    params = {
        "outer": normalized["outer"],
        "dimLen": normalized["axis_len"],
        "inner": normalized["inner"],
        "rows": normalized["rows"],
        "k": request.k,
        "isLargest": int(request.largest),
        "branch": branch,
        "kPad": k_pad,
        "kIdxPad": k_idx_pad,
        "mergePad": merge_pad,
    }
    d_max = 4096 if branch == 0 else 4000
    first = min(d_max, align_up(normalized["axis_len"], 32))
    candidates = list(dict.fromkeys((first, 2048, 1024, 512, 256, 128, 64, 32)))
    used_estimate = min(normalized["rows"], platform.core_num)
    rows_per_core = ceil_div(normalized["rows"], used_estimate)
    best: Optional[Tuple[int, int, int, int]] = None
    ub_budget = platform.fallback_ub_budget

    for d_chunk in candidates:
        if d_chunk > first or d_chunk < 32:
            continue
        max_g = 128
        if compute_size == 8 and branch == 0:
            if d_chunk < merge_pad:
                continue
            max_g = min(max_g, d_chunk // merge_pad)
        if branch == 1 and storage_size == 1:
            max_g = min(max_g, 65535 // d_chunk)
        elif branch == 1 and storage_size == 2:
            max_g = min(max_g, 131071 // (d_chunk * 2))
        use_table = branch == 1 and _fallback_use_transpose_table(d_chunk, merge_pad, compute_size)
        sort_count = max(d_chunk, merge_pad) if compute_size == 8 else k_pad
        floor = align_up(
            max(
                _fallback_sort_tmp(sort_count, compute_size),
                _fallback_radix_floor(d_chunk, compute_size),
                _fallback_radix_floor(merge_pad, compute_size),
            ),
            32,
        )
        if compute_size != 8:
            floor = max(
                floor,
                topk_api_tmp_size(d_chunk, min(request.k, d_chunk), request.dtype, request.largest, False, d_chunk),
                topk_api_tmp_size(merge_pad, request.k, request.dtype, request.largest, False, merge_pad),
            )
            floor = align_up(floor, 32)
        feasible_g = 0
        for g in range(max_g, 0, -1):
            if (
                _fallback_ub_total(g, d_chunk, storage_size, compute_size, request.k, floor, params, use_table)
                <= ub_budget
            ):
                feasible_g = g
                break
        if feasible_g == 0:
            continue
        chunks = ceil_div(normalized["axis_len"], d_chunk)
        groups = ceil_div(rows_per_core, feasible_g)
        cost = 2 * chunks - 1 if compute_size == 8 else chunks * groups
        candidate = (cost, -d_chunk, feasible_g, floor)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("fallback has no UB-feasible (dChunk, G) candidate")
    _, neg_d_chunk, g, tmp_size = best
    params.update(
        {
            "dChunk": -neg_d_chunk,
            "g": g,
            "tmpSize": tmp_size,
            "usedCores": min(normalized["rows"], platform.core_num),
        }
    )
    params["rowsPerCore"] = ceil_div(normalized["rows"], params["usedCores"])
    return params


def _fallback_plan(request: TopKRequest, normalized: Dict[str, int], platform: PlatformSpec) -> TopKTilingPlan:
    params = _fallback_params(request, normalized, platform)
    return TopKTilingPlan(
        route="fallback_l" if params["branch"] == 0 else "fallback_s",
        tiling_key=900002,
        block_dim=params["usedCores"],
        local_memory_bytes=platform.fallback_ub_budget,
        workspace_bytes=0,
        normalized=normalized,
        launch_params=params,
        sdk_queries=[
            "For every fallback dChunk candidate, query GetTopKMaxMinTmpSize for dChunk and mergePad; "
            "set tmpSize to max(all SDK results, radix/sort implementation floor), then rerun the (dChunk, G) search."
        ],
        notes=[
            "The derived tmp floor is sufficient for model exploration, not a production replacement for SDK queries."
        ],
    )


def _mode8_plan(request: TopKRequest, normalized: Dict[str, int], platform: PlatformSpec) -> TopKTilingPlan:
    axis_len = normalized["axis_len"]
    use_merge = request.dtype == "fp32" or (request.dtype in ("fp16", "bf16") and axis_len <= 1024)
    sort_count = align_up(axis_len, 32)
    sort_dtype_size = 4 if use_merge and request.dtype == "bf16" else DTYPE_BYTES[request.dtype]
    tmp_size = (
        sort_count * 8
        if use_merge
        else topk_api_tmp_size(axis_len, request.k, request.dtype, request.largest, request.sorted, axis_len)
    )
    candidates_by_size = {
        8: (4, 2, 1),
        4: (8, 4, 2, 1),
        2: (16, 8, 4, 2, 1),
        1: (32, 16, 8, 4, 2, 1),
    }
    best: Optional[Tuple[int, int, Dict[str, int]]] = None
    for chunk in candidates_by_size[DTYPE_BYTES[request.dtype]]:
        chunk = min(chunk, normalized["inner"])
        input_row = align_up(chunk * DTYPE_BYTES[request.dtype], platform.ub_block_size)
        axis_row = align_up(sort_count * sort_dtype_size, platform.ub_block_size)
        if use_merge:
            axis_row = max(axis_row, align_up(sort_count * 8, platform.ub_block_size))
        output_count = sort_count if use_merge else request.k
        value_row = align_up(output_count * sort_dtype_size, platform.ub_block_size)
        if use_merge:
            value_row = max(value_row, align_up(output_count * 8, platform.ub_block_size))
        index_row = align_up(output_count * 4, platform.ub_block_size)
        input_row_elems = input_row // DTYPE_BYTES[request.dtype]
        axis_row_elems = axis_row // sort_dtype_size
        if DTYPE_BYTES[request.dtype] <= 2 and (
            (axis_len - 1) * input_row_elems > 65535 or (chunk - 1) * axis_row_elems > 65535
        ):
            continue
        cast_row = (
            align_up(sort_count * DTYPE_BYTES[request.dtype], platform.ub_block_size)
            if use_merge and request.dtype == "bf16"
            else 0
        )
        peak = (
            axis_len * input_row
            + chunk * axis_row
            + chunk * value_row
            + chunk * index_row
            + chunk * cast_row
            + tmp_size
        )
        if peak > platform.ub_size - SIMT_RESERVED_BYTES:
            continue
        loops = ceil_div(normalized["inner"], chunk)
        tile_count = normalized["outer"] * loops
        active = min(platform.core_num, tile_count)
        details = {
            "innerChunk": chunk,
            "innerLoopNum": loops,
            "inputRowBytes": input_row,
            "valueAxisBytes": axis_row,
            "indexAxisBytes": value_row,
            "outputIndexRowBytes": index_row,
            "peakUb": peak,
        }
        score = (active, chunk)
        if best is None or score > best[:2]:
            best = (active, chunk, details)
    if best is None:
        raise ValueError("mode 8 has no UB-feasible innerChunk")
    active, _, details = best
    fields = {
        "isLargest": int(request.largest),
        "isSort": int(request.sorted),
        "sortLoopTimes": 1,
        "unsortedDimParallel": active,
        "unsortedDimNum": normalized["inner"],
        "lastDimNeedCore": 1,
        "numTileDataSize": sort_count,
        "platformCoreNum": platform.core_num,
        "topkAcApiTmpBufferSize": 0 if use_merge else tmp_size,
        "oneCoreRowNum": normalized["outer"],
        "modeType": MODE["non_last_small_axis"],
        "isInInt32Range": int(axis_len <= INT32_MAX),
        "lastAxisNum": axis_len,
        "topKRealValue": request.k,
        "lastDimTileNum": 1,
        "outputLastDimValue": request.k,
        "keyParams0": 1 if use_merge else 0,
        "keyParams1": details["innerChunk"],
        "keyParams2": details["innerLoopNum"],
        "keyParams3": details["inputRowBytes"],
        "keyParams4": details["valueAxisBytes"],
        "keyParams5": details["indexAxisBytes"],
        "tmpUbSize": tmp_size if use_merge else 0,
    }
    return TopKTilingPlan(
        route="common_non_last_mode8",
        tiling_key=TILING_KEY[request.dtype],
        block_dim=active,
        local_memory_bytes=platform.ub_size - SIMT_RESERVED_BYTES,
        workspace_bytes=platform.system_workspace_bytes,
        normalized=normalized,
        mode_type=MODE["non_last_small_axis"],
        tiling_data=fields,
        launch_params=details,
    )


def _find_tile(
    start: int,
    minimum: int,
    step: int,
    fits: Any,
) -> int:
    tile = start
    while tile >= minimum:
        if fits(tile):
            return tile
        tile -= step
    return 0


def _base_fields(request: TopKRequest, normalized: Dict[str, int], platform: PlatformSpec) -> Dict[str, int]:
    return {
        "isLargest": int(request.largest),
        "isSort": int(request.sorted),
        "unsortedDimNum": normalized["rows"],
        "platformCoreNum": platform.core_num,
        "isInInt32Range": int(normalized["axis_len"] <= INT32_MAX),
        "lastAxisNum": normalized["axis_len"],
        "topKRealValue": request.k,
        "outputLastDimValue": request.k,
    }


def _common_workspace(
    mode_type: int,
    fields: Dict[str, int],
    request: TopKRequest,
    platform: PlatformSpec,
) -> Tuple[int, List[str]]:
    tile_count = fields.get("lastDimTileNum", 1)
    parallel = fields.get("unsortedDimParallel", 1)
    index_size = 4 if fields["isInInt32Range"] else 8
    if mode_type == MODE["single_core"]:
        user = tile_count * BIN_NUM * parallel * index_size
    elif mode_type == MODE["multi_core_optimized"] and fields["isInInt32Range"]:
        factor = tile_count * parallel
        user = align_up(DTYPE_BYTES[request.dtype] * factor * request.k, 32)
        user += align_up(4 * factor * request.k, 32)
    else:
        user = BIN_NUM * index_size * parallel
        user += 2 * align_up(tile_count * index_size * parallel, 32)
    queries: List[str] = []
    needs_sort_with_index = request.sorted and (
        (mode_type == MODE["multi_core"] and request.k > 2000)
        or (mode_type == MODE["single_core"] and (request.k > 2000 or request.k * DTYPE_BYTES[request.dtype] > 8000))
    )
    if needs_sort_with_index:
        sort_loops = ceil_div(fields["unsortedDimNum"], platform.core_num)
        sort_active = (
            platform.core_num if sort_loops > 1 else (fields["unsortedDimNum"] % platform.core_num or platform.core_num)
        )
        sort_tmp_estimate = _fallback_sort_tmp(request.k, DTYPE_BYTES[request.dtype])
        fields.update(
            {
                "tilingKeyForSort": TILING_KEY[request.dtype],
                "sortLoopTimesForSort": sort_loops,
                "unsortedDimParallelForSort": sort_active,
                "unsortedDimNumForSort": fields["unsortedDimNum"],
                "lastDimTileNumForSort": 1,
                "lastDimNeedCoreForSort": 1,
                "numTileDataSizeForSort": request.k,
                "sortAcApiNeedBufferSizeForSort": sort_tmp_estimate,
                "oneCoreRowNumForSort": 1,
                "outputLastDimValueForSort": request.k,
                "isInInt32RangeForSort": 1,
                "lastAxisNumForSort": request.k,
                "modeTypeForSort": 1,
                "keyParams0": 0,
                "keyParams1": 0,
                "keyParams2": 0,
                "keyParams3": 0,
                "keyParams4": 0,
                "keyParams5": 0,
                "tmpUbSize": sort_tmp_estimate,
            }
        )
        user += align_up(request.k * fields["unsortedDimNum"] * DTYPE_BYTES[request.dtype], ALIGN_BYTES)
        user += align_up(request.k * fields["unsortedDimNum"] * DTYPE_BYTES[request.indices_dtype], ALIGN_BYTES)
        queries.append(
            "GetSortMaxMinTmpSize(value_dtype, uint32 index, hasSrcIndex=true) must replace the "
            "SortWithIndex tmp estimate; k<=2048 selects SortWithIndex small-size mode 1."
        )
    return user + platform.system_workspace_bytes, queries


def _sort_and_topk_tile(
    axis_len: int,
    rows: int,
    k: int,
    dtype_size: int,
    index_size: int,
    platform: PlatformSpec,
    sort_tmp_override: Optional[int],
) -> Tuple[Dict[str, int], int]:
    usable = platform.ub_size - SIMT_RESERVED_BYTES
    is_int32 = axis_len <= SORT_INT32_RANGE_MAX
    ub_extra = SORT_UB_CONST_INT32 if is_int32 else SORT_UB_CONST_INT64
    tile_factor = (10 if is_int32 else 14) + dtype_size

    def query_tmp(tile: int) -> int:
        if sort_tmp_override is not None:
            return sort_tmp_override
        return _fallback_sort_tmp(min(axis_len, tile), 1)

    tile = ((usable - ub_extra) // tile_factor // BIN_NUM) * BIN_NUM
    while tile >= BIN_NUM and query_tmp(tile) > usable - (ub_extra + tile_factor * tile):
        tile -= BIN_NUM
    if tile < BIN_NUM:
        raise ValueError("SortAndTopK has no UB-feasible radix-sort tile")

    last_tiles = ceil_div(axis_len, tile)
    small_tile = axis_len <= SORT_MIN_TILE and last_tiles == 1
    if last_tiles % platform.core_num != 0 and not small_tile:
        if rows == 1 and last_tiles == 1:
            tile = max(align_up(ceil_div(axis_len, platform.core_num), BIN_NUM), SORT_MIN_TILE)
        elif rows == 1 or last_tiles >= platform.core_num:
            all_cores = align_up(last_tiles, platform.core_num)
            tile = max(align_up(ceil_div(axis_len, all_cores), BIN_NUM), SORT_MIN_TILE)
        elif rows > 1 and rows < platform.core_num and last_tiles == 1:
            h_cores = platform.core_num // rows
            tile = align_up(axis_len // h_cores, BIN_NUM)
        elif rows > 1 and last_tiles > 1:
            tile = align_up(axis_len // last_tiles, BIN_NUM)
            adjusted_last_tiles = ceil_div(axis_len, tile)
            batch_cores = platform.core_num // adjusted_last_tiles if adjusted_last_tiles else platform.core_num
            if adjusted_last_tiles < platform.core_num and rows < platform.core_num and rows < batch_cores:
                batch_cores = rows
                h_cores = platform.core_num // batch_cores
                tile = align_up(ceil_div(axis_len, h_cores), BIN_NUM)
            if batch_cores == 1 and adjusted_last_tiles < platform.core_num:
                all_cores = align_up(adjusted_last_tiles, platform.core_num)
                tile = max(align_up(ceil_div(axis_len, all_cores), BIN_NUM), SORT_MIN_TILE)

    tmp_size = query_tmp(tile)
    remaining = usable - (ub_extra + tile_factor * tile)
    if tmp_size > remaining:
        raise ValueError("SortAndTopK adjusted tile does not fit the target Sort tmp size")
    tmp_size += (max(remaining - tmp_size - platform.ub_block_size, 0) // platform.ub_block_size) * (
        platform.ub_block_size
    )

    last_tiles = ceil_div(axis_len, tile)
    parallel = 1 if platform.core_num <= last_tiles else min(platform.core_num // last_tiles, rows)
    loops = ceil_div(rows, parallel)
    last_need = min(platform.core_num, last_tiles)
    core_need = parallel * last_need
    counter_size = 4 if is_int32 else 8
    ub_elements = tmp_size // counter_size
    if ub_elements == 0:
        raise ValueError("SortAndTopK Sort tmp cannot hold one counter")

    global_hist = BIN_NUM * last_tiles * dtype_size * parallel
    exclusive_hist = BIN_NUM * dtype_size * parallel
    key5 = max(ceil_div(global_hist, core_need), platform.ub_block_size)
    key0 = ceil_div(global_hist, key5)
    key3 = ceil_div(key5, ub_elements)
    key2 = min(key5, ub_elements)
    key4 = max(ceil_div(exclusive_hist, core_need), platform.ub_block_size)
    key1 = ceil_div(exclusive_hist, key4)

    extraction_tile = (usable - 1) // ALIGN_BYTES * ALIGN_BYTES // max(dtype_size, index_size)
    extraction_tiles = ceil_div(k, extraction_tile)
    fields = {
        "isInInt32Range": int(is_int32),
        "numTileDataSize": tile,
        "unsortedDimParallel": parallel,
        "lastDimTileNum": last_tiles,
        "sortLoopTimes": loops,
        "lastDimNeedCore": last_need,
        "keyParams0": key0,
        "keyParams1": key1,
        "keyParams2": key2,
        "keyParams3": key3,
        "keyParams4": key4,
        "keyParams5": key5,
        "tmpUbSize": tmp_size,
        "sortAndTopkTileDataSize": extraction_tile,
        "sortAndTopkBlockTileNum": extraction_tiles // platform.core_num,
        "sortAndTopkTailTileNum": extraction_tiles % platform.core_num,
    }

    exclusive_ws = align_up(key1 * key4 * counter_size, platform.ub_block_size)
    global_ws = align_up(key3 * key2 * key0 * counter_size, platform.ub_block_size)
    out_index_ws = align_up(axis_len * parallel * counter_size, platform.ub_block_size)
    sorted_index_ws = align_up(axis_len * parallel * index_size, platform.ub_block_size)
    hist_tile_ws = last_tiles * BIN_NUM * parallel * 2 * 2
    uint8_input_ws = align_up(last_tiles * tile * parallel, platform.ub_block_size)
    out_value_ws = align_up(axis_len * parallel * dtype_size, platform.ub_block_size) * 2
    user_workspace = (
        exclusive_ws + global_ws + out_index_ws + sorted_index_ws + hist_tile_ws + uint8_input_ws + out_value_ws
    )
    fields["coreNumNeed"] = core_need
    return fields, user_workspace


def _small_merge_plan(
    request: TopKRequest,
    normalized: Dict[str, int],
    platform: PlatformSpec,
    overrides: SdkOverrides,
) -> TopKTilingPlan:
    axis_len = normalized["axis_len"]
    aligned_axis = align_up(axis_len, 32)
    compute_size = 4 if request.dtype == "bf16" else DTYPE_BYTES[request.dtype]
    query = "GetConcatTmpSize(platform, align32(axis_len), compute_dtype_bytes)"
    merge_tmp = overrides.merge_tmp_bytes
    sdk_queries: List[str] = []
    if merge_tmp is None:
        merge_tmp = aligned_axis * 8
        sdk_queries.append(query + "; the Python fallback uses align32(axis_len) * 8 only for plan exploration.")
    buffer_count = 1 if axis_len >= 2048 else 2
    init_space = aligned_axis * 4 + aligned_axis * 2 * compute_size * 8
    usable = platform.ub_size - SIMT_RESERVED_BYTES
    row_bytes = (
        aligned_axis
        * buffer_count
        * (2 * DTYPE_BYTES[request.dtype] + DTYPE_BYTES[request.indices_dtype] + compute_size)
    )
    row_capacity = max((usable - merge_tmp - init_space) // max(row_bytes, 1), 1)
    virtual_rows = ceil_div(normalized["rows"], row_capacity)
    loop_count = ceil_div(virtual_rows, platform.core_num)
    active = platform.core_num if loop_count > 1 else (virtual_rows % platform.core_num or platform.core_num)
    fields = _base_fields(request, normalized, platform)
    fields.update(
        {
            "sortLoopTimes": loop_count,
            "unsortedDimParallel": active,
            "lastDimNeedCore": 1,
            "numTileDataSize": axis_len,
            "mergSortAcApiNeedBufferSize": merge_tmp,
            "oneCoreRowNum": row_capacity,
            "lastDimTileNum": 1,
            "keyParams4": buffer_count,
        }
    )
    workspace, extra_queries = _common_workspace(0, fields, request, platform)
    return TopKTilingPlan(
        route="common_small_merge",
        tiling_key=TILING_KEY[request.dtype] + 10000,
        block_dim=active,
        local_memory_bytes=usable,
        workspace_bytes=workspace,
        normalized=normalized,
        mode_type=0,
        tiling_data=fields,
        sdk_queries=sdk_queries + extra_queries,
    )


def _last_axis_common_plan(
    request: TopKRequest,
    normalized: Dict[str, int],
    platform: PlatformSpec,
    overrides: SdkOverrides,
) -> TopKTilingPlan:
    axis_len = normalized["axis_len"]
    rows = normalized["rows"]
    dtype_size = DTYPE_BYTES[request.dtype]
    index_size = DTYPE_BYTES[request.indices_dtype]
    usable = platform.ub_size - SIMT_RESERVED_BYTES

    if request.dtype in ("fp16", "fp32", "bf16") and axis_len <= 1024:
        return _small_merge_plan(request, normalized, platform, overrides)

    single_block_start = 10240 if dtype_size == 8 else 15360
    single_block_tile = _find_tile(
        single_block_start,
        axis_len,
        BIN_NUM,
        lambda tile: topk_api_tmp_size(axis_len, request.k, request.dtype, request.largest, request.sorted, tile)
        + _single_block_runtime(axis_len, tile, dtype_size, index_size, request.k)
        <= usable,
    )
    if single_block_tile >= axis_len:
        batches = single_block_tile // axis_len
        per_loop = platform.core_num * batches
        loops = ceil_div(rows, per_loop)
        tail_total = rows % per_loop
        fields = _base_fields(request, normalized, platform)
        fields.update(
            {
                "sortLoopTimes": loops,
                "unsortedDimParallel": platform.core_num,
                "lastDimNeedCore": 1,
                "numTileDataSize": axis_len,
                "lastDimTileNum": 1,
                "batchNumInUb": batches,
                "tailLoopBatchNum": tail_total // platform.core_num,
                "tailBatchNum": tail_total % platform.core_num,
                "topkAcApiTmpBufferSize": topk_api_tmp_size(
                    axis_len, request.k, request.dtype, request.largest, request.sorted, single_block_tile
                ),
            }
        )
        workspace, queries = _common_workspace(0, fields, request, platform)
        return TopKTilingPlan(
            route="common_single_block",
            tiling_key=TILING_KEY[request.dtype],
            block_dim=platform.core_num,
            local_memory_bytes=usable,
            workspace_bytes=workspace,
            normalized=normalized,
            mode_type=0,
            tiling_data=fields,
            sdk_queries=queries,
        )

    if request.dtype == "fp32" and 0.25 <= request.k / axis_len < 0.5:
        split = ceil_div(axis_len, 2048)
        once_max = usable // (128 if index_size == 8 else 96)
        if split > 1 and split <= platform.core_num and rows * split <= platform.core_num and once_max >= 32:
            aligned_input_elems = align_up(axis_len * 8, 32) // 8
            fields = _base_fields(request, normalized, platform)
            fields.update(
                {
                    "sortLoopTimes": 1,
                    "unsortedDimParallel": rows,
                    "lastDimNeedCore": split,
                    "numTileDataSize": axis_len // split,
                    "oneCoreRowNum": 1,
                    "modeType": MODE["fp32_inter_core_merge"],
                    "lastDimTileNum": split,
                    "keyParams0": (once_max // 32) * 32,
                    "lastDimTileNumTimes": ceil_div(split, platform.core_num),
                }
            )
            return TopKTilingPlan(
                route="common_fp32_inter_core_merge",
                tiling_key=23003,
                block_dim=rows * split,
                local_memory_bytes=usable,
                workspace_bytes=rows * aligned_input_elems * 8 * 2 + platform.system_workspace_bytes,
                normalized=normalized,
                mode_type=MODE["fp32_inter_core_merge"],
                tiling_data=fields,
            )
        block_sort = (usable // 64 // 32) * 32
        extract = (usable // ((8 + 4 + 4 + index_size) * 2) // 32) * 32
        blocks = ceil_div(axis_len, block_sort) if block_sort else 0
        if rows >= platform.core_num // 2 and 1 < blocks <= 256 and extract:
            active = min(platform.core_num, rows)
            aligned_count = blocks * block_sort
            fields = _base_fields(request, normalized, platform)
            fields.update(
                {
                    "sortLoopTimes": 1,
                    "unsortedDimParallel": active,
                    "lastDimNeedCore": active,
                    "numTileDataSize": block_sort,
                    "oneCoreRowNum": ceil_div(rows, active),
                    "modeType": MODE["fp32_intra_core_merge"],
                    "lastDimTileNum": blocks,
                    "keyParams0": ceil_div(rows, active),
                    "keyParams1": block_sort * 2,
                    "keyParams2": aligned_count * 2,
                    "keyParams3": aligned_count,
                    "keyParams4": extract,
                    "keyParams5": INT32_MAX // block_sort,
                    "lastDimTileNumTimes": ceil_div(blocks, platform.core_num),
                }
            )
            return TopKTilingPlan(
                route="common_fp32_intra_core_merge",
                tiling_key=33003,
                block_dim=active,
                local_memory_bytes=usable,
                workspace_bytes=active * aligned_count * 8 * 2 + platform.system_workspace_bytes,
                normalized=normalized,
                mode_type=MODE["fp32_intra_core_merge"],
                tiling_data=fields,
            )

    single_core_start = 10240 if dtype_size == 8 else 15360
    single_core_tile = 0
    if rows >= platform.core_num:
        single_core_tile = _find_tile(
            single_core_start,
            BIN_NUM,
            BIN_NUM,
            lambda tile: topk_api_tmp_size(
                min(axis_len, tile), request.k, request.dtype, request.largest, request.sorted, tile, True
            )
            + _single_core_runtime(axis_len, tile, dtype_size, index_size, request.k, request.sorted)
            <= usable,
        )
    if single_core_tile:
        tile_count = ceil_div(axis_len, single_core_tile)
        balanced_tile = ceil_div(axis_len, tile_count)
        fields = _base_fields(request, normalized, platform)
        fields.update(
            {
                "sortLoopTimes": rows // platform.core_num,
                "unsortedDimParallel": platform.core_num,
                "numTileDataSize": balanced_tile,
                "lastDimTileNum": tile_count,
                "modeType": MODE["single_core"],
                "tailBatchNum": rows % platform.core_num,
                "tailTileNum": axis_len % tile_count,
                "topkAcApiTmpBufferSize": topk_api_tmp_size(
                    balanced_tile, request.k, request.dtype, request.largest, request.sorted, single_core_tile, True
                ),
            }
        )
        workspace, queries = _common_workspace(MODE["single_core"], fields, request, platform)
        return TopKTilingPlan(
            route="common_single_core_radix",
            tiling_key=TILING_KEY[request.dtype],
            block_dim=platform.core_num,
            local_memory_bytes=usable,
            workspace_bytes=workspace,
            normalized=normalized,
            mode_type=MODE["single_core"],
            tiling_data=fields,
            sdk_queries=queries,
        )

    start = 5120 if dtype_size == 8 else 7680
    optimized_tile = _find_tile(
        start,
        request.k,
        32,
        lambda tile: topk_api_tmp_size(tile, request.k, request.dtype, request.largest, request.sorted, tile)
        + _optimized_runtime(tile, dtype_size, index_size, request.k, platform.ub_block_size)
        <= usable,
    )
    if (
        optimized_tile
        and optimized_tile < axis_len
        and request.k * ceil_div(axis_len, optimized_tile) <= optimized_tile
    ):
        mode_type = MODE["multi_core_optimized"]
        tile = optimized_tile
        route = "common_multi_core_optimized_radix"
    else:
        tile = 0
        if axis_len < 10_000_000:
            tile = _find_tile(
                start,
                BIN_NUM,
                BIN_NUM,
                lambda candidate: topk_api_tmp_size(
                    candidate,
                    max(ceil_div(axis_len, candidate), request.k),
                    request.dtype,
                    request.largest,
                    request.sorted,
                    candidate,
                )
                + _multi_core_runtime(axis_len, candidate, platform.core_num, dtype_size, index_size, request.k)
                <= usable,
            )
        mode_type = MODE["multi_core"]
        route = "common_multi_core_radix"

    if tile:
        tile_count = ceil_div(axis_len, tile)
        if axis_len <= tile * platform.core_num // 2:
            parallel = min(rows, max(platform.core_num // tile_count, 1))
            loops = ceil_div(rows, parallel)
            last_need = tile_count
        else:
            parallel = 1
            loops = rows
            last_need = min(platform.core_num, tile_count)
        fields = _base_fields(request, normalized, platform)
        fields.update(
            {
                "sortLoopTimes": loops,
                "unsortedDimParallel": parallel,
                "lastDimNeedCore": last_need,
                "numTileDataSize": tile,
                "lastDimTileNum": tile_count,
                "modeType": mode_type,
                "lastDimTileNumTimes": ceil_div(tile_count, platform.core_num),
                "topkAcApiTmpBufferSize": topk_api_tmp_size(
                    tile, max(tile_count, request.k), request.dtype, request.largest, request.sorted, tile
                ),
            }
        )
        workspace, queries = _common_workspace(mode_type, fields, request, platform)
        return TopKTilingPlan(
            route=route,
            tiling_key=TILING_KEY[request.dtype],
            block_dim=platform.core_num,
            local_memory_bytes=usable,
            workspace_bytes=workspace,
            normalized=normalized,
            mode_type=mode_type,
            tiling_data=fields,
            sdk_queries=queries,
        )

    sort_fields, user_workspace = _sort_and_topk_tile(
        axis_len,
        rows,
        request.k,
        dtype_size,
        index_size,
        platform,
        overrides.sort_tmp_bytes,
    )
    core_need = sort_fields.pop("coreNumNeed")
    fields = {
        "isLargest": 2 if request.largest else 0,
        "isSort": int(request.sorted),
        "platformCoreNum": platform.core_num,
        "lastAxisNum": axis_len,
        "unsortedDimNum": rows,
        "topKRealValue": request.k,
        "modeType": MODE["sort_and_topk"],
    }
    fields.update(sort_fields)
    sdk_queries = []
    if overrides.sort_tmp_bytes is None:
        sdk_queries.append(
            "Query GetSortMaxMinTmpSize(uint8 input, uint32 index) for every SortAndTopK tile adjustment; "
            "replace the Python estimate and rerun the tile, histogram, and workspace calculation."
        )
    return TopKTilingPlan(
        route="common_sort_and_topk",
        tiling_key=TILING_KEY[request.dtype],
        block_dim=core_need,
        local_memory_bytes=usable,
        workspace_bytes=user_workspace + platform.system_workspace_bytes,
        normalized=normalized,
        mode_type=MODE["sort_and_topk"],
        tiling_data=fields,
        sdk_queries=sdk_queries,
    )


def compute_topk_tiling(
    request: TopKRequest,
    platform: PlatformSpec,
    overrides: SdkOverrides = SdkOverrides(),
    allow_transpose: bool = True,
) -> TopKTilingPlan:
    platform.validate()
    normalized = normalize_request(request)
    non_last = bool(normalized["non_last"])
    axis_len = normalized["axis_len"]

    if non_last and request.k == axis_len and axis_len in (7, 17) and request.dtype in ("fp16", "bf16"):
        return _small_axis_plan(request, normalized, platform)

    batched_int32 = (
        request.dtype == "int32"
        and non_last
        and axis_len >= 32
        and normalized["rows"] >= 512
        and (request.k <= 32 or 4 * request.k <= axis_len)
    )
    use_fallback = (non_last and axis_len > 2048) or batched_int32 or request.dtype == "int64"
    if use_fallback:
        transpose_candidate = (
            allow_transpose
            and normalized["outer"] == 1
            and non_last
            and request.dtype in ("int32", "fp32")
            and axis_len > 2048
            and normalized["inner"] >= platform.core_num * 32
            and 3 * request.k <= axis_len
        )
        if transpose_candidate:
            inner_request = TopKRequest(
                shape=(normalized["inner"], axis_len),
                k=request.k,
                dim=1,
                dtype=request.dtype,
                largest=request.largest,
                sorted=request.sorted,
                indices_dtype=request.indices_dtype,
            )
            inner_plan = compute_topk_tiling(inner_request, platform, overrides, allow_transpose=False)
            input_bytes = axis_len * normalized["inner"] * 4
            values_offset = align_up(input_bytes, 512)
            indices_offset = align_up(values_offset + normalized["inner"] * request.k * 4, 512)
            inner_offset = align_up(indices_offset + normalized["inner"] * request.k * 8, 512)
            total = inner_offset + inner_plan.workspace_bytes
            if inner_offset <= 2 * input_bytes and total <= INT64_MAX:
                return TopKTilingPlan(
                    route="transpose_fallback",
                    tiling_key=900003,
                    block_dim=inner_plan.block_dim,
                    local_memory_bytes=inner_plan.local_memory_bytes,
                    workspace_bytes=total,
                    normalized=normalized,
                    launch_params={
                        "innerTilingKey": inner_plan.tiling_key,
                        "inputOffset": 0,
                        "valuesOffset": values_offset,
                        "indicesOffset": indices_offset,
                        "innerWorkspaceOffset": inner_offset,
                        "innerWorkspaceBytes": inner_plan.workspace_bytes,
                        "transposeRequests": [
                            {"shape": [axis_len, normalized["inner"]], "perm": [1, 0], "elementBytes": 4},
                            {"shape": [normalized["inner"], request.k], "perm": [1, 0], "elementBytes": 4},
                            {"shape": [normalized["inner"], request.k], "perm": [1, 0], "elementBytes": 8},
                        ],
                        "innerPlan": inner_plan.to_dict(),
                    },
                    sdk_queries=inner_plan.sdk_queries,
                )
        return _fallback_plan(request, normalized, platform)

    if non_last:
        return _mode8_plan(request, normalized, platform)
    return _last_axis_common_plan(request, normalized, platform, overrides)


def _self_test() -> None:
    # 仅自测使用的具名样例，不作为实际设备默认配置。
    platform = PlatformSpec(core_num=64, ub_size=248 * 1024)
    cases = [
        (TopKRequest((2, 7, 64), 7, 1, "fp16"), "fixed_small_axis"),
        (TopKRequest((4, 8192), 64, -1, "int64"), "fallback_l"),
        (TopKRequest((2, 4096, 64), 64, 1, "int32"), "fallback_s"),
        (TopKRequest((8192, 8192), 256, 0, "int32"), "transpose_fallback"),
        (TopKRequest((8, 64, 64), 8, 1, "fp16"), "common_non_last_mode8"),
        (TopKRequest((4, 512), 16, -1, "fp16"), "common_small_merge"),
        (TopKRequest((1, 128), 1, -1, "int8"), "common_single_block"),
        (TopKRequest((1, 8052), 2013, -1, "fp32"), "common_fp32_inter_core_merge"),
        (TopKRequest((32, 8052), 2013, -1, "fp32"), "common_fp32_intra_core_merge"),
        (TopKRequest((64, 16384), 1, -1, "int8"), "common_single_core_radix"),
        (TopKRequest((1, 16384), 1, -1, "int8"), "common_multi_core_optimized_radix"),
        (TopKRequest((1, 1_000_000), 2048, -1, "int32"), "common_multi_core_radix"),
        (TopKRequest((1, 10_000_001), 128, -1, "int8"), "common_sort_and_topk"),
    ]
    plans = []
    for request, expected_route in cases:
        plan = compute_topk_tiling(request, platform)
        assert plan.route == expected_route, (expected_route, plan.route)
        plans.append(plan)
    for plan in plans:
        assert plan.block_dim > 0
        assert plan.workspace_bytes >= 0
    print("topk_tiling.py self-test: PASS")


def _parse_shape(raw: str) -> Tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("shape must contain at least one dimension")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=_parse_shape, help="comma-separated ND shape")
    parser.add_argument("--k", type=int)
    parser.add_argument("--dim", type=int, default=-1)
    parser.add_argument("--dtype", choices=sorted(DTYPE_BYTES), default="fp16")
    parser.add_argument("--indices-dtype", choices=("int32", "int64"), default="int64")
    parser.add_argument("--smallest", action="store_true")
    parser.add_argument("--unsorted", action="store_true")
    parser.add_argument("--core-num", type=int, help="目标设备查询得到的 AIV 核数")
    parser.add_argument("--ub-size", type=int, help="目标设备查询得到的单核 UB 字节数")
    parser.add_argument("--merge-tmp-bytes", type=int)
    parser.add_argument("--sort-tmp-bytes", type=int)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    if args.shape is None or args.k is None or args.core_num is None or args.ub_size is None:
        parser.error("--shape, --k, --core-num and --ub-size are required unless --self-test is used")
    request = TopKRequest(
        shape=args.shape,
        k=args.k,
        dim=args.dim,
        dtype=args.dtype,
        largest=not args.smallest,
        sorted=not args.unsorted,
        indices_dtype=args.indices_dtype,
    )
    platform = PlatformSpec(core_num=args.core_num, ub_size=args.ub_size)
    overrides = SdkOverrides(args.merge_tmp_bytes, args.sort_tmp_bytes)
    print(json.dumps(compute_topk_tiling(request, platform, overrides).to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
