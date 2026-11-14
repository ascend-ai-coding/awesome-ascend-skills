# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始（VF 原始接口，无语法糖）。

Grouped FP32-to-INT8 quantization. The kernel schedules independent row/group
segments across vector workers; the Host wrapper only specializes and launches.

Support domain: 1 <= M <= 64, 1 <= N <= 8192, 1 <= G <= N. Each group uses
aligned local X/Q tiles; VF offsets advance by 64 elements from local zero,
while native DMA places the result at its possibly unaligned logical GM column.
Tile addresses use padded physical capacities, and auto_mutex orders reuse.
Numerical conversion and clipping precede byte extraction: PACK4_B32 itself
neither rounds nor saturates. No intermediate INT8 register is needed.

Verified on behavior 872df763f, CANN 9.2.0, Ascend950PR: 90 combinations with
independent FP32-product reference, exact ties, signed saturation, odd G,
G=1/G=N, upper N, guards, multiple workers and repeated calls. This tests the
listed parameter matrix, not every combination in the support domain.
"""

from functools import lru_cache

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32
INT32 = pl.DT_INT32
INT8 = pl.DT_INT8
VF_FP32_LANES = 64


@lru_cache(maxsize=None)
def _make_kernel(rows: int, cols: int, group_size: int):
    groups = (cols + group_size - 1) // group_size
    x_width = ((group_size + VF_FP32_LANES - 1) // VF_FP32_LANES) * VF_FP32_LANES
    scale_width = ((groups + 7) // 8) * 8
    q_width = ((group_size + 31) // 32) * 32
    x_bytes = x_width * 4
    scale_bytes = scale_width * 4
    q_addr = x_bytes + scale_bytes

    @pl.vector_function
    def _quantize_group(x_tile, scale_tile, q_tile, active, group):
        scale_reg = vf.load_align(scale_tile, [0, group], dist=pl.LoadDist.BRC)
        for part in pl.range(0, active, VF_FP32_LANES):
            count = pl.min(VF_FP32_LANES, active - part)
            mask_f32 = vf.update_mask(count, dtype=FP32)
            values = vf.load_align(x_tile, [0, part])
            products = vf.mul(values, scale_reg, mask_f32)
            rounded = vf.astype(
                products,
                mask_f32,
                dtype=INT32,
                round_mode=pl.VFRoundMode.CAST_RINT,
            )
            mask_i32 = vf.update_mask(count, dtype=INT32)
            lower_clamped = vf.maxs(rounded, pl.const(-127, INT32), mask_i32)
            clipped = vf.mins(lower_clamped, pl.const(127, INT32), mask_i32)
            vf.store_align(
                q_tile,
                clipped,
                mask_i32,
                [0, part],
                dist=pl.StoreDist.PACK4_B32,
            )

    @pl.jit(auto_mutex=True)
    def _quantize(
        x: pl.Tensor[[rows, cols], FP32],
        inverse_scales: pl.Tensor[[rows, groups], FP32],
        q: pl.Tensor[[rows, cols], INT8],
    ):
        x_type = pl.TileType(
            shape=[1, x_width], dtype=FP32, target_memory=pl.MemorySpace.Vec
        )
        scale_type = pl.TileType(
            shape=[1, scale_width], dtype=FP32, target_memory=pl.MemorySpace.Vec
        )
        q_type = pl.TileType(
            shape=[1, q_width], dtype=INT8, target_memory=pl.MemorySpace.Vec
        )
        x_group = pl.make_tile_group(type=x_type, addrs=[0], mutex_ids=[0])
        scale_group = pl.make_tile_group(
            type=scale_type, addrs=[x_bytes], mutex_ids=[1]
        )
        q_group = pl.make_tile_group(type=q_type, addrs=[q_addr], mutex_ids=[2])

        with pl.section_vector():
            work_items = rows * groups
            for item in pl.range(pl.get_block_idx(), work_items, pl.get_block_num()):
                row = item // groups
                group = item % groups
                offset = group * group_size
                active = pl.min(group_size, cols - offset)
                x_tile = x_group.next()
                scale_tile = scale_group.next()
                q_tile = q_group.next()
                pl.set_validshape(x_tile, [1, active])
                pl.set_validshape(scale_tile, [1, groups])
                pl.set_validshape(q_tile, [1, active])
                pl.load(x_tile, x, [row, offset])
                pl.load(scale_tile, inverse_scales, [row, 0])
                _quantize_group(x_tile, scale_tile, q_tile, active, group)
                pl.store(q, q_tile, [row, offset])

    return _quantize


def run(x, inverse_scales, q, group_size, workers=1):
    """Launch grouped quantization; only tensor metadata is read on the Host."""
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    rows, cols = x.shape
    if not (1 <= rows <= 64 and 1 <= cols <= 8192 and group_size <= cols):
        raise ValueError("expected 1 <= M <= 64, 1 <= N <= 8192, 1 <= group_size <= N")
    groups = (cols + group_size - 1) // group_size
    if tuple(inverse_scales.shape) != (rows, groups):
        raise ValueError("inverse_scales must have shape [M, ceil(N / group_size)]")
    if tuple(q.shape) != (rows, cols):
        raise ValueError("q must have shape [M, N]")
    if workers <= 0:
        raise ValueError("workers must be positive")
    return _make_kernel(rows, cols, group_size)[None, workers](x, inverse_scales, q)
