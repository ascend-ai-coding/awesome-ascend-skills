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
Reduction Tiling Algorithm — Reference Implementation.

Five templates: AR-SmallR / AR-FullLoad / AR-Recompute / ARA-FullLoad / ARA-Recompute.
Covers: ReduceSum, ReduceMax, Softmax, LayerNorm, ArgMax.
"""

import math
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

VECTOR_REG_WIDTH = 256
A0_TILE_UNIT_FP32 = 64
A0_TILE_UNIT_FP16 = 128
REPEAT_MAX = 255
R_BIN_SIZE = 128


@dataclass
class ReductionTilingInput:
    """Input parameters for compute_reduction_tiling."""
    shape: List[int]
    axes: List[int]
    dtype: str = "fp32"
    op_type: str = "sum"
    ub_size: int = 248 * 1024
    core_num: int = 64
    extra_flags: Optional[dict] = None


@dataclass
class _ReductionConfig:
    dtype_bytes: int
    ub_size: int
    core_num: int
    tmp_buf: int = 0


@dataclass
class ReductionTilingData:
    """Reduction tiling output."""

    template: str = "AR-FullLoad"
    mode: str = "AR"
    outer_dim: int = 1
    reduce_dim: int = 0
    inner_dim: int = 1

    core_num: int = 0
    used_core_num: int = 0
    tiles_per_core: int = 0

    a0_tile_len: int = 0
    r_chunk: int = 0
    rows_per_ub: int = 0
    row_per_loop: int = 0

    tmp_buf_size: int = 0

    is_full_load: bool = True
    use_pattern_ra: bool = False
    enable_group_reduce: bool = False
    enable_welford: bool = False
    enable_dichotomy: bool = False
    with_index: bool = False
    workspace_size: int = 0

    dtype: str = "fp32"
    dtype_bytes: int = 4

    @property
    def sub_mode(self) -> str:
        """Backward-compatible alias: FullLoad / Recompute / SmallR."""
        if "SmallR" in self.template:
            return "SmallR"
        if "Recompute" in self.template:
            return "Recompute"
        return "FullLoad"

    def to_dict(self) -> dict:
        return {
            "template": self.template,
            "mode": self.mode,
            "outer_dim": self.outer_dim,
            "reduce_dim": self.reduce_dim,
            "inner_dim": self.inner_dim,
            "core_num": self.core_num,
            "used_core_num": self.used_core_num,
            "a0_tile_len": self.a0_tile_len,
            "r_chunk": self.r_chunk,
            "rows_per_ub": self.rows_per_ub,
            "row_per_loop": self.row_per_loop,
            "tmp_buf_size": self.tmp_buf_size,
            "is_full_load": self.is_full_load,
            "use_pattern_ra": self.use_pattern_ra,
            "enable_group_reduce": self.enable_group_reduce,
            "workspace_size": self.workspace_size,
        }


def collapse_axes(shape: List[int], axes: List[int]) -> Tuple[List[int], List[str]]:
    """Collapse N-d shape + axes into A/R alternating sequence."""
    rank = len(shape)
    is_reduce = [False] * rank
    for ax in axes:
        is_reduce[ax] = True

    merged_dims = [shape[0]]
    merged_tags = ["R" if is_reduce[0] else "A"]

    for i in range(1, rank):
        tag = "R" if is_reduce[i] else "A"
        if tag == merged_tags[-1]:
            merged_dims[-1] *= shape[i]
        else:
            merged_dims.append(shape[i])
            merged_tags.append(tag)

    return merged_dims, merged_tags


def route_mode(
    collapsed_dims: List[int],
    collapsed_tags: List[str],
) -> Tuple[str, int, int, int]:
    """Determine AR vs ARA mode. Returns (mode, outer_dim, reduce_dim, inner_dim)."""
    n = len(collapsed_dims)
    if n == 1:
        if collapsed_tags[0] == "R":
            return "AR", 1, collapsed_dims[0], 1
        raise ValueError("Single axis must be reduce axis")

    r_idx = None
    for i, tag in enumerate(collapsed_tags):
        if tag == "R":
            r_idx = i
            break

    if r_idx is None:
        raise ValueError("No reduce axis found")

    if r_idx == n - 1:
        outer_dim = 1
        for i in range(r_idx):
            outer_dim *= collapsed_dims[i]
        return "AR", outer_dim, collapsed_dims[r_idx], 1
    elif r_idx == 0:
        inner_dim = 1
        for i in range(r_idx + 1, n):
            inner_dim *= collapsed_dims[i]
        return "ARA", 1, collapsed_dims[r_idx], inner_dim
    else:
        outer_dim = 1
        for i in range(r_idx):
            outer_dim *= collapsed_dims[i]
        inner_dim = 1
        for i in range(r_idx + 1, n):
            inner_dim *= collapsed_dims[i]
        return "ARA", outer_dim, collapsed_dims[r_idx], inner_dim


def calc_tmp_buf_size(r_length_align: int, type_size: int) -> int:
    """Calculate sharedTmpBuffer size for Reduce API."""
    per_repeat = VECTOR_REG_WIDTH // (type_size * 8)
    per_block = 32 // type_size
    repeats = (r_length_align + per_repeat - 1) // per_repeat
    tmp_buf = ((repeats + per_block - 1) // per_block) * per_block * type_size
    return max(tmp_buf, 4096)


def _r_small_threshold(dtype_bytes: int) -> int:
    r_tile_unit = 8 if dtype_bytes == 4 else 16
    return r_tile_unit * 2


def _build_ar_smallr_result(
    outer_dim: int, reduce_dim: int, cfg: _ReductionConfig,
) -> ReductionTilingData:
    a1_tile_unit = A0_TILE_UNIT_FP32
    r_tile_unit = 8 if cfg.dtype_bytes == 4 else 16
    r_align = math.ceil(reduce_dim / r_tile_unit) * r_tile_unit
    per_tile = r_align * (cfg.dtype_bytes * 4 + 8)
    max_a1_tiles = cfg.ub_size // (a1_tile_unit * (per_tile + 4))
    a1_tile_count = min(max(max_a1_tiles, 1), math.ceil(outer_dim / a1_tile_unit))
    a1_tile_len = a1_tile_count * a1_tile_unit
    a1_outer = math.ceil(outer_dim / a1_tile_len)
    tiles_per_core = math.ceil(a1_outer / cfg.core_num)
    used_cores = math.ceil(a1_outer / tiles_per_core) if tiles_per_core > 0 else 1

    return ReductionTilingData(
        template="AR-SmallR", mode="AR",
        outer_dim=outer_dim, reduce_dim=reduce_dim, inner_dim=1,
        core_num=cfg.core_num, used_core_num=used_cores,
        tiles_per_core=tiles_per_core,
        a0_tile_len=a1_tile_len,
        is_full_load=True,
        dtype_bytes=cfg.dtype_bytes,
    )


def _build_ar_fullload_result(
    outer_dim: int, reduce_dim: int, rows_per_core: int, used_cores: int,
    cfg: _ReductionConfig,
) -> ReductionTilingData:
    return ReductionTilingData(
        template="AR-FullLoad", mode="AR",
        outer_dim=outer_dim, reduce_dim=reduce_dim, inner_dim=1,
        core_num=cfg.core_num, used_core_num=used_cores,
        tiles_per_core=rows_per_core,
        rows_per_ub=reduce_dim,
        tmp_buf_size=cfg.tmp_buf,
        is_full_load=True,
        dtype_bytes=cfg.dtype_bytes,
    )


def _build_ar_recompute_result(
    outer_dim: int, reduce_dim: int, rows_per_core: int, used_cores: int,
    cfg: _ReductionConfig,
) -> ReductionTilingData:
    chunk_overhead = 2112
    per_elem = cfg.dtype_bytes * 5 + 4
    r_chunk = (cfg.ub_size - chunk_overhead) // per_elem
    r_chunk = min(max(r_chunk, 1), reduce_dim)
    return ReductionTilingData(
        template="AR-Recompute", mode="AR",
        outer_dim=outer_dim, reduce_dim=reduce_dim, inner_dim=1,
        core_num=cfg.core_num, used_core_num=used_cores,
        tiles_per_core=rows_per_core,
        r_chunk=r_chunk,
        rows_per_ub=r_chunk,
        tmp_buf_size=cfg.tmp_buf,
        is_full_load=False,
        dtype_bytes=cfg.dtype_bytes,
    )


def tiling_ar(outer_dim: int, reduce_dim: int, cfg: _ReductionConfig) -> ReductionTilingData:
    cfg.tmp_buf = calc_tmp_buf_size(reduce_dim, cfg.dtype_bytes)
    rows_per_core = math.ceil(outer_dim / cfg.core_num)
    used_cores = math.ceil(outer_dim / rows_per_core) if rows_per_core > 0 else 1

    if reduce_dim <= _r_small_threshold(cfg.dtype_bytes):
        smallr = _build_ar_smallr_result(outer_dim, reduce_dim, cfg)
        if smallr.a0_tile_len > 0:
            return smallr

    overhead = 64 + cfg.tmp_buf
    r_max = (cfg.ub_size - overhead) // (2 * cfg.dtype_bytes)
    r_max = min(r_max, REPEAT_MAX)
    r_max = max(r_max, 1)

    if reduce_dim <= r_max:
        return _build_ar_fullload_result(outer_dim, reduce_dim, rows_per_core, used_cores, cfg)
    return _build_ar_recompute_result(outer_dim, reduce_dim, rows_per_core, used_cores, cfg)


def _build_ara_fullload(
    dims: Tuple[int, int, int], base: int, cfg: _ReductionConfig,
) -> ReductionTilingData:
    outer_dim, reduce_dim, inner_dim = dims
    ub_per_tile = 2 * reduce_dim * base * cfg.dtype_bytes + cfg.tmp_buf
    fixed_cost = 2 * base * cfg.dtype_bytes + cfg.tmp_buf
    factor_max = (cfg.ub_size - fixed_cost) // ub_per_tile if ub_per_tile > 0 else 0
    factor_max = max(factor_max, 1)

    a0_factor_max = (inner_dim + base - 1) // base
    total_tiles_max = outer_dim * a0_factor_max
    a0_inner_max = (total_tiles_max + cfg.core_num - 1) // cfg.core_num if cfg.core_num > 0 else 1

    a0_inner = min(a0_inner_max, factor_max, a0_factor_max)
    a0_inner = max(a0_inner, 1)
    a0_tile_len = a0_inner * base

    a0_outer = (inner_dim + a0_tile_len - 1) // a0_tile_len
    total_tiles = outer_dim * a0_outer
    tiles_per_core = (total_tiles + cfg.core_num - 1) // cfg.core_num
    used_cores = (total_tiles + tiles_per_core - 1) // tiles_per_core if tiles_per_core > 0 else 1

    return ReductionTilingData(
        template="ARA-FullLoad", mode="ARA",
        outer_dim=outer_dim, reduce_dim=reduce_dim, inner_dim=inner_dim,
        core_num=cfg.core_num, used_core_num=used_cores,
        tiles_per_core=tiles_per_core,
        a0_tile_len=a0_tile_len,
        rows_per_ub=a0_tile_len,
        row_per_loop=reduce_dim,
        tmp_buf_size=cfg.tmp_buf,
        is_full_load=True,
        use_pattern_ra=True,
        dtype_bytes=cfg.dtype_bytes,
    )


def _build_ara_recompute(
    dims: Tuple[int, int, int], r_max: int, base: int, cfg: _ReductionConfig,
) -> ReductionTilingData:
    outer_dim, reduce_dim, inner_dim = dims
    r_per_loop = max(r_max, 1)

    a0_tile_len = min(
        inner_dim,
        base * ((cfg.ub_size - cfg.tmp_buf) // (2 * r_per_loop * base * cfg.dtype_bytes))
    )
    a0_tile_len = max(a0_tile_len, base)

    a0_outer = (inner_dim + a0_tile_len - 1) // a0_tile_len
    total_tiles = outer_dim * a0_outer
    tiles_per_core = (total_tiles + cfg.core_num - 1) // cfg.core_num
    used_cores = (total_tiles + tiles_per_core - 1) // tiles_per_core if tiles_per_core > 0 else 1

    return ReductionTilingData(
        template="ARA-Recompute", mode="ARA",
        outer_dim=outer_dim, reduce_dim=reduce_dim, inner_dim=inner_dim,
        core_num=cfg.core_num, used_core_num=used_cores,
        tiles_per_core=tiles_per_core,
        a0_tile_len=a0_tile_len,
        r_chunk=r_per_loop,
        row_per_loop=r_per_loop,
        tmp_buf_size=cfg.tmp_buf,
        is_full_load=False,
        use_pattern_ra=True,
        dtype_bytes=cfg.dtype_bytes,
    )


def tiling_ara(outer_dim: int, reduce_dim: int, inner_dim: int, cfg: _ReductionConfig) -> ReductionTilingData:
    base = A0_TILE_UNIT_FP32 if cfg.dtype_bytes == 4 else A0_TILE_UNIT_FP16
    cfg.tmp_buf = calc_tmp_buf_size(reduce_dim * base, cfg.dtype_bytes)

    overhead = 2 * base * cfg.dtype_bytes + cfg.tmp_buf
    r_max = (cfg.ub_size - overhead) // (2 * base * cfg.dtype_bytes)
    r_max = min(r_max, REPEAT_MAX)
    r_max = max(r_max, 1)

    dims = (outer_dim, reduce_dim, inner_dim)
    if reduce_dim <= r_max:
        return _build_ara_fullload(dims, base, cfg)
    return _build_ara_recompute(dims, r_max, base, cfg)


def check_group_reduce(
    reduce_dim: int, a_dims_product: int,
    ub_size: int, dtype_bytes: int, core_num: int,
) -> Tuple[bool, int, int]:
    """Check if Group Reduce is needed. Returns (enable, workspace_size, group_r)."""
    max_r_single_core = (ub_size - 4096) // (2 * dtype_bytes)
    if reduce_dim <= max_r_single_core:
        return False, 0, 1
    if a_dims_product >= core_num:
        return False, 0, 1
    group_r = min(core_num, reduce_dim // max_r_single_core + 1)
    workspace_size = core_num * 256 * 4
    return True, workspace_size, group_r


def compute_reduction_tiling(params: ReductionTilingInput) -> ReductionTilingData:
    """
    Compute reduction tiling using the five-template decision tree.

    Templates: AR-SmallR / AR-FullLoad / AR-Recompute / ARA-FullLoad / ARA-Recompute
    """
    dtype_map = {"fp16": 2, "float16": 2, "bf16": 2, "bfloat16": 2,
                 "fp32": 4, "float32": 4, "float": 4}
    dtype_bytes = dtype_map.get(params.dtype, 4)
    extra = params.extra_flags or {}

    cfg = _ReductionConfig(dtype_bytes=dtype_bytes, ub_size=params.ub_size, core_num=params.core_num)

    collapsed_dims, collapsed_tags = collapse_axes(params.shape, params.axes)
    mode, outer_dim, reduce_dim, inner_dim = route_mode(collapsed_dims, collapsed_tags)

    if mode == "AR":
        result = tiling_ar(outer_dim, reduce_dim, cfg)
    else:
        result = tiling_ara(outer_dim, reduce_dim, inner_dim, cfg)

    a_product = 1
    for i, tag in enumerate(collapsed_tags):
        if tag == "A":
            a_product *= collapsed_dims[i]

    enable_gr, ws_size, _ = check_group_reduce(
        reduce_dim, a_product, params.ub_size, dtype_bytes, params.core_num,
    )
    if extra.get("force_group_reduce", False) or enable_gr:
        result.enable_group_reduce = True
        result.workspace_size = ws_size

    if params.op_type in ("var", "std", "norm") and not result.is_full_load:
        result.enable_welford = extra.get("force_welford", True)

    if params.op_type == "sum" and extra.get("force_dichotomy", False):
        result.enable_dichotomy = True

    if params.op_type in ("argmax", "argmin") or extra.get("with_index", False):
        result.with_index = True

    result.dtype = params.dtype
    result.dtype_bytes = dtype_bytes
    return result


if __name__ == "__main__":
    import json

    tiling = compute_reduction_tiling(ReductionTilingInput(
        shape=[1000, 256], axes=[0], dtype="fp32",
    ))
    logger.info("=== ReduceSum([1000,256], axis=0) ===")
    logger.info(json.dumps(tiling.to_dict(), indent=2))

    tiling2 = compute_reduction_tiling(ReductionTilingInput(
        shape=[2, 100, 4], axes=[1], dtype="fp32",
    ))
    logger.info("\n=== ReduceMax([2,100,4], axis=1) ===")
    logger.info(json.dumps(tiling2.to_dict(), indent=2))
