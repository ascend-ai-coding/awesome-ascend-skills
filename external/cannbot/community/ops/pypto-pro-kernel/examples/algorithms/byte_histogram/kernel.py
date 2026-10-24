# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。VF 直方图按输入分组，INT32 部分计数再由 VF 汇总。

Each DMA brings 32768 bytes; a VF loop consumes 256-byte registers. Both
UINT16 bin windows stay in registers until the group is complete, so even a
single hot bin cannot overflow. Distinct names hold widened unsigned counts
and signed views. Workers own input groups, then disjoint output-bin slices;
auto_mutex protects UB reuse and the second launch orders the partial merge.
Empty inputs use a native zero kernel without passing a zero-sized tensor.

Validated on behavior version 872df763f, CANN 9.2.0, Ascend950PR: exact counts,
N through 131101 including tails and 65537, concentrated/edge/cyclic/random
values, repeated calls, multiple workers and input/output guards. The merge
loads all partial rows for a bin slice into UB; very large N needs a separate
capacity plan or another reduction level. This is not an arbitrary-size claim.
"""

import pypto_pro.language as pl
import torch
from pypto_pro.language import Vf as vf


UINT8 = pl.DT_UINT8
UINT16 = pl.DT_UINT16
UINT32 = pl.DT_UINT32
INT32 = pl.DT_INT32
GROUP_SIZE = 32768
VF_UINT8_LANES = 256
VF_INT32_LANES = 64
BIN_COUNT = 256
BIN_GROUPS = BIN_COUNT // VF_INT32_LANES


@pl.vector_function
def _histogram_group(src_tile, partial_tile, active):
    """Histogram one input group; a group contains at most 32768 bytes."""
    mask_u16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=UINT16)
    mask_i32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=INT32)
    hist0 = vf.full(0, mask_u16, dtype=UINT16)
    hist1 = vf.full(0, mask_u16, dtype=UINT16)

    for col in pl.range(0, active, VF_UINT8_LANES):
        active_lanes = pl.min(VF_UINT8_LANES, active - col)
        mask_u8 = vf.update_mask(active_lanes, dtype=UINT8)
        values = vf.load_align(src_tile, [0, col], dtype=UINT8)
        hist0 = vf.histograms(
            values,
            mask_u8,
            bin_type=pl.BinType.BIN0,
            hist_type=pl.HistType.FREQUENCY,
        )
        hist1 = vf.histograms(
            values,
            mask_u8,
            bin_type=pl.BinType.BIN1,
            hist_type=pl.HistType.FREQUENCY,
        )

    counts0_lo_u32 = vf.unpack(hist0, part=pl.PackPart.LOWER, dtype=UINT32)
    counts0_hi_u32 = vf.unpack(hist0, part=pl.PackPart.UPPER, dtype=UINT32)
    counts1_lo_u32 = vf.unpack(hist1, part=pl.PackPart.LOWER, dtype=UINT32)
    counts1_hi_u32 = vf.unpack(hist1, part=pl.PackPart.UPPER, dtype=UINT32)
    counts0_lo = vf.bit_cast(counts0_lo_u32, dtype=INT32)
    counts0_hi = vf.bit_cast(counts0_hi_u32, dtype=INT32)
    counts1_lo = vf.bit_cast(counts1_lo_u32, dtype=INT32)
    counts1_hi = vf.bit_cast(counts1_hi_u32, dtype=INT32)

    vf.store_align(partial_tile, counts0_lo, mask_i32, [0, 0])
    vf.store_align(partial_tile, counts0_hi, mask_i32, [0, 64])
    vf.store_align(partial_tile, counts1_lo, mask_i32, [0, 128])
    vf.store_align(partial_tile, counts1_hi, mask_i32, [0, 192])


@pl.jit(auto_mutex=True)
def _histogram_partials(
    x: pl.Tensor[[1, pl.DYNAMIC], UINT8],
    partials: pl.Tensor[[pl.DYNAMIC, BIN_COUNT], INT32],
):
    input_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, GROUP_SIZE],
            dtype=UINT8,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=[0],
        mutex_ids=[0],
    )
    partial_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, BIN_COUNT],
            dtype=INT32,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=[0x8000],
        mutex_ids=[1],
    )

    with pl.section_vector():
        group_count = (x.shape[1] + GROUP_SIZE - 1) // GROUP_SIZE
        for group_idx in pl.range(pl.get_block_idx(), group_count, pl.get_block_num()):
            src_tile = input_group.next()
            partial_tile = partial_group.next()
            input_offset = group_idx * GROUP_SIZE
            active = pl.min(GROUP_SIZE, x.shape[1] - input_offset)
            pl.set_validshape(src_tile, [1, active])
            pl.load(src_tile, x, [0, input_offset])
            _histogram_group(src_tile, partial_tile, active)
            pl.store(partials, partial_tile, [group_idx, 0])


@pl.vector_function
def _sum_partial_group(src_tile, dst_tile, group_count):
    """Sum one 64-bin slice across the bounded partial rows."""
    mask_i32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=INT32)
    total = vf.full(0, mask_i32, dtype=INT32)
    for row in pl.range(0, group_count, 1):
        values = vf.load_align(src_tile, [row, 0])
        total = vf.add(total, values, mask_i32)
    vf.store_align(dst_tile, total, mask_i32, [0, 0])


def _make_reduce_kernel(group_count):
    @pl.jit(auto_mutex=True)
    def kernel(
        partials: pl.Tensor[[group_count, BIN_COUNT], INT32],
        counts: pl.Tensor[[1, BIN_COUNT], INT32],
    ):
        partial_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[group_count, VF_INT32_LANES],
                dtype=INT32,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=[0],
            mutex_ids=[0],
        )
        output_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, VF_INT32_LANES],
                dtype=INT32,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=[group_count * VF_INT32_LANES * 4],
            mutex_ids=[1],
        )

        with pl.section_vector():
            for bin_group in pl.range(
                pl.get_block_idx(), BIN_GROUPS, pl.get_block_num()
            ):
                src_tile = partial_group.next()
                dst_tile = output_group.next()
                pl.load(src_tile, partials, [0, bin_group * VF_INT32_LANES])
                _sum_partial_group(src_tile, dst_tile, group_count)
                pl.store(counts, dst_tile, [0, bin_group * VF_INT32_LANES])

    return kernel


@pl.vector_function
def _zero_count_slice(dst_tile):
    mask_i32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=INT32)
    zeros = vf.full(0, mask_i32, dtype=INT32)
    vf.store_align(dst_tile, zeros, mask_i32, [0, 0])


@pl.jit(auto_mutex=True)
def _zero_counts(counts: pl.Tensor[[1, BIN_COUNT], INT32]):
    output_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, VF_INT32_LANES],
            dtype=INT32,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=[0],
        mutex_ids=[0],
    )
    with pl.section_vector():
        for bin_group in pl.range(pl.get_block_idx(), BIN_GROUPS, pl.get_block_num()):
            dst_tile = output_group.next()
            _zero_count_slice(dst_tile)
            pl.store(counts, dst_tile, [0, bin_group * VF_INT32_LANES])


def run(x, counts, workers=1):
    """Write the exact UINT8 histogram into the caller's INT32 output."""
    if x.ndim != 1 or counts.ndim != 1 or counts.shape[0] != BIN_COUNT:
        raise ValueError("expected x[N] and counts[256]")
    if x.dtype != torch.uint8 or counts.dtype != torch.int32:
        raise TypeError("expected UINT8 input and INT32 output")
    if not x.is_contiguous() or not counts.is_contiguous():
        raise ValueError("expected contiguous input and output")
    if x.device != counts.device:
        raise ValueError("input and output must share a device")
    if workers < 1:
        raise ValueError("workers must be positive")

    n = x.shape[0]
    counts_2d = counts.view(1, BIN_COUNT)
    if n == 0:
        _zero_counts[None, workers](counts_2d)
        return

    group_count = (n + GROUP_SIZE - 1) // GROUP_SIZE
    partials = torch.empty((group_count, BIN_COUNT), dtype=torch.int32, device=x.device)
    x_2d = x.view(1, n)
    _histogram_partials[None, workers](x_2d, partials)
    _make_reduce_kernel(group_count)[None, workers](partials, counts_2d)
