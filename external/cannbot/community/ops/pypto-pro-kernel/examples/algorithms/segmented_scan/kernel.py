# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。VF INT32 行内 inclusive segmented scan。

契约：连续 x/heads/out[M,N]，heads 为 0/1；遇到头标记先重置，再累加当前输入。整数累加不得溢出。
每行由一个 Vector worker 持有；值与头标记共享 lane 参与条件，寄存器间携带已合并的尾值。
三组单槽 UB 全行常驻，占 3 * ceil(N/64) * 256 字节；扩大 N 前须重新核对容量。
固定有效区限制 GM 尾写；auto_mutex 排序三组读写，Host 仅选形状和 worker 数。
验证边界：PyPTO 行为版本 872df763f、CANN 9.2.0、Ascend950PR；x 在 [-4,4]。
同目录测试覆盖无/全/稀疏重置、寄存器边界、重复调用和守卫，保持精确 INT32 比较。
"""

from functools import lru_cache

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf


INT32 = pl.DT_INT32
UINT32 = pl.DT_UINT32
LANES = 64


@pl.vector_function
def _scan_rows(x_tile, heads_tile, y_tile, cols):
    """Scan 64-lane chunks and carry each chunk's final segmented sum."""
    int_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=INT32)
    uint_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=UINT32)
    lane_ids = vf.arange(0, dtype=UINT32)
    index_mask = vf.full(63, uint_mask, dtype=UINT32)
    zeros = vf.full(0, int_mask, dtype=INT32)
    carry = vf.full(0, int_mask, dtype=INT32)

    for col in pl.range(0, cols, LANES):
        active = pl.min(LANES, cols - col)
        active_mask = vf.update_mask(active, dtype=INT32)
        x = vf.load_align(x_tile, [0, col])
        heads = vf.load_align(heads_tile, [0, col])

        scanned = x
        reset = heads
        offset = 1
        for _ in pl.range(0, 6):
            # The scan network doubles its covered prefix at each power of two.
            wrapped = vf.adds(lane_ids, LANES - offset, uint_mask)
            wrapped = vf.and_(wrapped, index_mask, uint_mask)
            left_sum = vf.gather(scanned, wrapped)
            left_reset = vf.gather(reset, wrapped)

            sum_with_left = vf.add(scanned, left_sum, int_mask)
            no_reset = vf.eq(reset, zeros, int_mask)
            sum_with_reset = vf.select(sum_with_left, scanned, no_reset)
            lower_lanes = vf.lt(lane_ids, offset, uint_mask)
            scanned = vf.select(scanned, sum_with_reset, lower_lanes)
            reset_with_left = vf.max(reset, left_reset, int_mask)
            reset = vf.select(reset, reset_with_left, lower_lanes)
            offset = offset * 2

        no_head_in_prefix = vf.eq(reset, zeros, int_mask)
        with_carry = vf.add(scanned, carry, int_mask)
        result = vf.select(with_carry, scanned, no_head_in_prefix)
        vf.store_align(y_tile, result, active_mask, [0, col])
        carry = vf.full(result, int_mask, pos=pl.DuplicatePos.HIGHEST)


@lru_cache(maxsize=None)
def make_kernel(cols: int):
    """Build a dynamic-row kernel specialized for the logical column count."""
    if cols <= 0:
        raise ValueError("cols must be positive")
    padded_cols = ((cols + LANES - 1) // LANES) * LANES
    row_bytes = padded_cols * 4

    @pl.jit(auto_mutex=True)
    def kernel(
        x: pl.Tensor[[pl.DYNAMIC, cols], INT32],
        heads: pl.Tensor[[pl.DYNAMIC, cols], INT32],
        out: pl.Tensor[[pl.DYNAMIC, cols], INT32],
    ):
        tile_type = pl.TileType(
            shape=[1, padded_cols],
            dtype=INT32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[1, cols],
        )
        x_group = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
        heads_group = pl.make_tile_group(
            type=tile_type, addrs=[row_bytes], mutex_ids=[1]
        )
        out_group = pl.make_tile_group(
            type=tile_type, addrs=[2 * row_bytes], mutex_ids=[2]
        )

        with pl.section_vector():
            for row in pl.range(pl.get_block_idx(), x.shape[0], pl.get_block_num()):
                x_tile = x_group.next()
                heads_tile = heads_group.next()
                out_tile = out_group.next()
                pl.load(x_tile, x, [row, 0])
                pl.load(heads_tile, heads, [row, 0])
                _scan_rows(x_tile, heads_tile, out_tile, cols)
                pl.store(out, out_tile, [row, 0])

    return kernel


def run(x, heads, out, workers=1):
    """Launch the native kernel; the host only selects shape and worker count."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    if tuple(heads.shape) != tuple(x.shape) or tuple(out.shape) != tuple(x.shape):
        raise ValueError("x, heads, and out must have identical shapes")
    make_kernel(int(x.shape[1]))[None, workers](x, heads, out)
