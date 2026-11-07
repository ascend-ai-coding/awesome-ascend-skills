# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。Clipped sliding-window maxima over independent FP32 rows.

Contract: distinct contiguous finite FP32 x[M,N] and y[M,N], with
1 <= M <= 64, 1 <= N <= 131072 and Host integer 0 <= radius <= 31.
The radius is passed as an INT64 scalar; Host code only checks metadata.

Workers own disjoint 128-output blocks and may read overlapping input halos.
A source tile spans up to 190 elements, padded to 192 FP32 elements at UB 0;
the separate 128-element output tile starts at 0x1000 (UB high-water 4608 B).
Two masked VF segments reuse the source without overwriting the second halo.
Logical output origin, clipped load origin and tile indices remain separate.
Auto-mutex tile groups order DMA, VF access and buffer reuse.

Verified against behavior 872df763f, CANN 9.2.0 and Ascend950PR. Tests cover
clipped windows, register/block tails, finite extrema, radius changes within
a cached specialization, multiple workers, poisoned repeats and guards.
The listed verification matrix is not an exhaustive support-domain sweep.
"""

from functools import lru_cache

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf


FP32 = pl.DT_FP32
_BLOCK = 128
_TILE_WIDTH = 192


@pl.vector_function
def _window_max(source, output, origin, live, active, radius, output_offset):
    """Reduce each output lane over its clipped neighborhood."""
    mask = vf.update_mask(active, dtype=FP32)
    lanes = vf.arange(0, dtype=pl.DT_INT32)
    zero_i = vf.full(0, mask, dtype=pl.DT_INT32)
    live_i = vf.full(live, mask, dtype=pl.DT_INT32)
    centers = vf.adds(lanes, origin, mask)
    neg_inf = vf.full(float("-inf"), mask, dtype=FP32)
    result = vf.full(float("-inf"), mask, dtype=FP32)

    for step in pl.range(0, 2 * radius + 1):
        offset = step - radius
        indices = vf.adds(centers, offset, mask)
        valid_left = vf.ge(indices, zero_i, mask)
        valid_right = vf.lt(indices, live_i, mask)
        valid = vf.and_(valid_left, valid_right, mask)
        safe_indices = vf.select(indices, centers, valid)
        gather_indices = vf.bit_cast(safe_indices, dtype=pl.DT_UINT32)
        candidate = vf.gather(source, gather_indices, mask)
        candidate = vf.select(candidate, neg_inf, valid)
        result = vf.max(result, candidate, mask)

    vf.store_align(output, result, mask, [0, output_offset])


@lru_cache(maxsize=16)
def _make_kernel(rows, cols):
    blocks_per_row = (cols + _BLOCK - 1) // _BLOCK

    @pl.jit(auto_mutex=True)
    def kernel(
        x: pl.Tensor[[rows, cols], FP32],
        y: pl.Tensor[[rows, cols], FP32],
        radius: pl.DT_INT64,
    ):
        with pl.section_vector():
            source_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, _TILE_WIDTH],
                    dtype=FP32,
                    target_memory=pl.MemorySpace.Vec,
                ),
                addrs=[0],
                mutex_ids=[0],
            )
            output_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, _BLOCK],
                    dtype=FP32,
                    target_memory=pl.MemorySpace.Vec,
                ),
                addrs=[0x1000],
                mutex_ids=[1],
            )
            total_blocks = rows * blocks_per_row
            for task in pl.range(pl.get_block_idx(), total_blocks, pl.get_block_num()):
                row = task // blocks_per_row
                block = task % blocks_per_row
                out_start = block * _BLOCK
                active = pl.min(_BLOCK, cols - out_start)
                load_start = pl.max(0, out_start - radius)
                load_end = pl.min(cols, out_start + _BLOCK + radius)
                live = load_end - load_start
                origin = out_start - load_start

                source = source_group.next()
                output = output_group.next()
                pl.set_validshape(source, [1, live])
                pl.load(source, x, [row, load_start])
                first_active = pl.min(64, active)
                _window_max(source, output, origin, live, first_active, radius, 0)
                if active > 64:
                    _window_max(
                        source,
                        output,
                        origin + 64,
                        live,
                        active - 64,
                        radius,
                        64,
                    )
                pl.set_validshape(output, [1, active])
                pl.store(y, output, [row, out_start])

    return kernel


def run(x, y, radius, workers=1):
    """Compute the clipped one-dimensional neighborhood maximum per row."""
    if not isinstance(radius, int) or isinstance(radius, bool):
        raise TypeError("radius must be a Host integer")
    if not isinstance(workers, int) or isinstance(workers, bool):
        raise TypeError("workers must be a Host integer")
    if workers < 1:
        raise ValueError("workers must be positive")

    rows, cols = tuple(x.shape)
    if not 1 <= rows <= 64 or not 1 <= cols <= 131072:
        raise ValueError("shape must satisfy 1 <= M <= 64 and 1 <= N <= 131072")
    if not 0 <= radius <= 31:
        raise ValueError("radius must satisfy 0 <= radius <= 31")
    if tuple(y.shape) != (rows, cols):
        raise ValueError("y must have the same shape as x")
    if x.dtype != y.dtype or str(x.dtype) != "torch.float32":
        raise TypeError("x and y must be FP32")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("x and y must be contiguous")

    _make_kernel(rows, cols)[None, workers](x, y, radius)
