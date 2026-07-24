#!/usr/bin/env python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
MC2 Communication-Compute Fusion Tiling Script (Equal-Split Fallback).

Determines the optimal uniform M-dimension split for MC2 fusion operators
(matmul_all_reduce, allgather_matmul, matmul_reducescatter, alltoall_matmul)
based on baseM/baseN from the matmul tiling data.

Objective (lexicographic):
  1. Maximize core utilization (prefer 100%)
  2. Maximize tile count (prefer more splits for better pipeline overlap)

Usage:
  python mc2_tiling.py -M 4096 -N 4096 --baseM 128 --baseN 128 --n-core 8
  python mc2_tiling.py -M 2048 -N 256 --baseM 128 --baseN 128 --n-core 8
  python mc2_tiling.py -M 3072 -N 512 --baseM 256 --baseN 128 --n-core 24
"""

import argparse
import logging
import math
import sys
from dataclasses import dataclass, field
from typing import List, Optional


ALIGN_GRANULARITY = 16
UTIL_THRESHOLD_FULL = 1.0
UTIL_THRESHOLD_RELAXED = 0.80
SHORT_BLOCK_MIN = 16


@dataclass
class TilingResult:
    """Output of the equal-split tiling algorithm."""
    tile_cnt: int = 0
    long_block_cnt: int = 0
    long_m_size: int = 0
    short_block_cnt: int = 0
    short_m_size: int = 0
    short_block_pos: int = 0
    utilization: float = 0.0
    n_block_cnt: int = 0
    m_block_cnt: int = 0
    total_blocks: int = 0
    degrade_level: int = 0
    align_granularity: int = 0
    degrade_reason: str = ""

    def __str__(self) -> str:
        lines = [
            "=== MC2 Equal-Split Tiling Result ===",
            f"  longMSize      = {self.long_m_size}",
            f"  longBlockCnt   = {self.long_block_cnt}",
            f"  shortBlockCnt  = {self.short_block_cnt}",
            f"  shortMSize     = {self.short_m_size}",
            f"  shortBlockPos  = {self.short_block_pos} ({'front' if self.short_block_pos == 0 else 'back'})",
            f"  tileCnt        = {self.tile_cnt}",
            f"  ---",
            f"  nBlockCnt      = {self.n_block_cnt}",
            f"  mBlockCnt      = {self.m_block_cnt}",
            f"  totalBlocks    = {self.total_blocks}",
            f"  utilization    = {self.utilization:.1%}",
            f"  ---",
            f"  degradeLevel   = {self.degrade_level} {self.degrade_reason}",
            f"  alignGranularity = {self.align_granularity}",
        ]
        return "\n".join(lines)


@dataclass
class DegradeConfig:
    """Parameters controlling degrade strategy for uniform tiling search."""
    align_granularity: int
    util_threshold: float
    degrade_level: int
    degrade_reason: str


def calc_n_block_cnt(n: int, base_n: int) -> int:
    return math.ceil(n / base_n)


def calc_min_m_block_cnt(n_core: int, n_block_cnt: int) -> int:
    return max(1, math.ceil(n_core / n_block_cnt))


def calc_utilization(long_m_size: int, base_m: int, n_block_cnt: int, n_core: int) -> tuple:
    m_block_cnt = math.ceil(long_m_size / base_m)
    total_blocks = m_block_cnt * n_block_cnt
    util = min(total_blocks, n_core) / n_core
    return m_block_cnt, total_blocks, util


def find_divisors(n: int) -> List[int]:
    divs = set()
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return sorted(divs)


def find_optimal_uniform(m: int, base_m: int, n_block_cnt: int, n_core: int,
                         degrade: DegradeConfig) -> Optional[TilingResult]:
    min_m_block_cnt = max(1, math.ceil(degrade.util_threshold * n_core / n_block_cnt))
    min_long_m_size = min_m_block_cnt * base_m

    if min_long_m_size > m:
        return None

    best = None
    divisors = find_divisors(m)

    for d in divisors:
        if d < min_long_m_size:
            continue
        if d % degrade.align_granularity != 0:
            continue

        m_block_cnt, total_blocks, util = calc_utilization(d, base_m, n_block_cnt, n_core)
        tile_cnt = m // d

        candidate = TilingResult(
            tile_cnt=tile_cnt,
            long_block_cnt=tile_cnt,
            long_m_size=d,
            short_block_cnt=0,
            short_m_size=0,
            short_block_pos=0,
            utilization=util,
            n_block_cnt=n_block_cnt,
            m_block_cnt=m_block_cnt,
            total_blocks=total_blocks,
            degrade_level=degrade.degrade_level,
            align_granularity=degrade.align_granularity,
            degrade_reason=degrade.degrade_reason,
        )

        if best is None:
            best = candidate
        else:
            if (util > best.utilization or
                (util == best.utilization and tile_cnt > best.tile_cnt)):
                best = candidate

    return best


def _build_single_tile(m: int, base_m: int, n_block_cnt: int, n_core: int,
                       degrade_reason: str) -> TilingResult:
    m_block_cnt, total_blocks, util = calc_utilization(m, base_m, n_block_cnt, n_core)
    return TilingResult(
        tile_cnt=1,
        long_block_cnt=1,
        long_m_size=m,
        short_block_cnt=0,
        short_m_size=0,
        short_block_pos=0,
        utilization=util,
        n_block_cnt=n_block_cnt,
        m_block_cnt=m_block_cnt,
        total_blocks=total_blocks,
        degrade_level=3,
        align_granularity=ALIGN_GRANULARITY,
        degrade_reason=degrade_reason,
    )


def fallback_short_block(m: int, base_m: int, n_block_cnt: int, n_core: int,
                         util_threshold: float) -> TilingResult:
    min_m_block_cnt = max(1, math.ceil(util_threshold * n_core / n_block_cnt))
    min_long_m_size = min_m_block_cnt * base_m

    if m < base_m:
        return _build_single_tile(m, base_m, n_block_cnt, n_core, "(M < baseM, single tile)")

    if min_long_m_size > m:
        return _build_single_tile(m, base_m, n_block_cnt, n_core, "(M < minLongMSize, single tile)")

    long_m_size = min_long_m_size
    while long_m_size % ALIGN_GRANULARITY != 0:
        long_m_size += 1

    long_block_cnt = m // long_m_size
    short_m_size = m - long_block_cnt * long_m_size

    while short_m_size > 0 and short_m_size < SHORT_BLOCK_MIN and long_block_cnt > 1:
        long_block_cnt -= 1
        short_m_size = m - long_block_cnt * long_m_size

    short_block_cnt = 1 if short_m_size > 0 else 0

    m_block_cnt, total_blocks, util = calc_utilization(long_m_size, base_m, n_block_cnt, n_core)
    tile_cnt = long_block_cnt + short_block_cnt

    return TilingResult(
        tile_cnt=tile_cnt,
        long_block_cnt=long_block_cnt,
        long_m_size=long_m_size,
        short_block_cnt=short_block_cnt,
        short_m_size=short_m_size,
        short_block_pos=1,
        utilization=util,
        n_block_cnt=n_block_cnt,
        m_block_cnt=m_block_cnt,
        total_blocks=total_blocks,
        degrade_level=2,
        align_granularity=ALIGN_GRANULARITY,
        degrade_reason="(short block fallback)",
    )


def equal_split_tiling(m: int, n: int, base_m: int, base_n: int, n_core: int) -> TilingResult:
    n_block_cnt = calc_n_block_cnt(n, base_n)

    result = find_optimal_uniform(m, base_m, n_block_cnt, n_core,
                                  DegradeConfig(base_m, UTIL_THRESHOLD_FULL, 0, ""))
    if result is not None:
        return result

    result = find_optimal_uniform(m, base_m, n_block_cnt, n_core,
                                  DegradeConfig(ALIGN_GRANULARITY, UTIL_THRESHOLD_FULL, 1, "(relaxed to 16-align)"))
    if result is not None:
        return result

    result = fallback_short_block(m, base_m, n_block_cnt, n_core, UTIL_THRESHOLD_FULL)
    if result.utilization >= UTIL_THRESHOLD_RELAXED:
        return result

    result = find_optimal_uniform(m, base_m, n_block_cnt, n_core,
                                  DegradeConfig(base_m, UTIL_THRESHOLD_RELAXED, 3, "(relaxed util to 80%)"))
    if result is not None:
        return result

    result = find_optimal_uniform(m, base_m, n_block_cnt, n_core,
                                  DegradeConfig(ALIGN_GRANULARITY, UTIL_THRESHOLD_RELAXED, 3,
                                                "(relaxed util to 80%, 16-align)"))
    if result is not None:
        return result

    return fallback_short_block(m, base_m, n_block_cnt, n_core, UTIL_THRESHOLD_RELAXED)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="MC2 Equal-Split Tiling")
    parser.add_argument("-M", type=int, required=True, help="Total M dimension")
    parser.add_argument("-N", type=int, required=True, help="Total N dimension")
    parser.add_argument("--baseM", type=int, required=True, help="baseM from matmul tiling data")
    parser.add_argument("--baseN", type=int, required=True, help="baseN from matmul tiling data")
    parser.add_argument("--n-core", type=int, required=True, help="usedCoreNum (AIC count)")
    args = parser.parse_args()

    if any(v <= 0 for v in (args.M, args.N, args.baseM, args.baseN, args.n_core)):
        logging.error("Error: all arguments must be positive")
        sys.exit(1)

    result = equal_split_tiling(args.M, args.N, args.baseM, args.baseN, args.n_core)
    logging.info(result)


if __name__ == "__main__":
    main()
