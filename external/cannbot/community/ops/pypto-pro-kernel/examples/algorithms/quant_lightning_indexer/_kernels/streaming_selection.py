# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始（VF）。Bounded-memory selection for large candidate capacities.

One vector worker owns a row. A 16-bit radix search counts keys at or above
each midpoint, then emits greater keys followed by ties. All DMA buffers
are bounded by 8192 elements, independent of the requested output capacity.
"""

from functools import lru_cache
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf
from .selection import emit_selection

BF = pl.DT_BF16
U16 = pl.DT_UINT16
I32 = pl.DT_INT32


@pl.vector_function
def count_keys(src, count_tile, active, threshold):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U16)
    zero = vf.full(0, p, dtype=U16)
    one = vf.full(1, p, dtype=U16)
    sign = vf.full(0x8000, p, dtype=U16)
    neg = vf.full(0xFFFF, p, dtype=U16)
    target = vf.full(threshold, p, dtype=U16)
    counts = vf.full(0, p, dtype=U16)
    for c in pl.range(0, active, 128):
        valid = vf.update_mask(pl.min(128, active - c), dtype=U16)
        raw = vf.load_align(src, c)
        bits = vf.bit_cast(raw, dtype=U16)
        negative = vf.ge(bits, sign, p)
        flip = vf.select(neg, sign, negative)
        key = vf.xor(bits, flip, p)
        selected = vf.ge(key, target, valid)
        increment = vf.select(one, zero, selected)
        counts = vf.add(counts, increment, p)
    total = vf.reduce_sum(counts, p)
    vf.store_align(count_tile, total, p, 0)


@pl.vector_function
def collect_keys(src, index_tile, count_tile, active, threshold, equal):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U16)
    zero = vf.full(0, p, dtype=U16)
    one = vf.full(1, p, dtype=U16)
    sign = vf.full(0x8000, p, dtype=U16)
    neg = vf.full(0xFFFF, p, dtype=U16)
    target = vf.full(threshold, p, dtype=U16)
    counts = vf.full(0, p, dtype=U16)
    vf.clear_spr()
    ar = vf.unalign_reg_for_store()
    for c in pl.range(0, active, 128):
        valid = vf.update_mask(pl.min(128, active - c), dtype=U16)
        raw = vf.load_align(src, c)
        bits = vf.bit_cast(raw, dtype=U16)
        negative = vf.ge(bits, sign, p)
        flip = vf.select(neg, sign, negative)
        key = vf.xor(bits, flip, p)
        if equal:
            selected = vf.eq(key, target, valid)
        else:
            selected = vf.gt(key, target, valid)
        increment = vf.select(one, zero, selected)
        counts = vf.add(counts, increment, p)
        index = vf.arange(c, dtype=U16)
        compact = vf.squeeze(
            index, selected, gather_mode=pl.SqueezeMode.STORE_REG, dtype=U16
        )
        vf.squeeze_store_unalign(index_tile, compact, ar)
    vf.squeeze_store_unalign_post(index_tile, ar)
    total = vf.reduce_sum(counts, p)
    vf.store_align(count_tile, total, p, 0)


@pl.vector_function
def fill_padding(indices, values, size):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=I32)
    minus_one = vf.full(-1, p, dtype=I32)
    inf_bits = vf.full(0xFF80, p, dtype=pl.DT_UINT32)
    for c in pl.range(0, size, 64):
        active = vf.update_mask(pl.min(64, size - c), dtype=I32)
        vf.store_align(indices, minus_one, active, c)
        vf.store_align(values, inf_bits, active, c, dist=pl.StoreDist.PACK_B32)


@lru_cache(None)
def make_streaming_select(has_offset=False):
    chunk = 8192

    @pl.jit(auto_mutex=True)
    def kernel(
        scores: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], BF],
        offsets: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], I32],
        indices: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], I32],
        values: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], BF],
    ):
        rows = scores.shape[0]
        width = scores.shape[1]
        topk = indices.shape[1]
        sg_type = pl.TileType(
            shape=[1, chunk],
            dtype=BF,
            target_memory=pl.MemorySpace.Vec,
        )
        sg = pl.make_tile_group(
            type=sg_type,
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        xg_type = pl.TileType(
            shape=[1, chunk + 128],
            dtype=U16,
            target_memory=pl.MemorySpace.Vec,
        )
        xg = pl.make_tile_group(
            type=xg_type,
            addrs=2 * chunk,
            mutex_ids="auto",
            depth=1,
        )
        cg_type = pl.TileType(
            shape=[1, 128],
            dtype=U16,
            target_memory=pl.MemorySpace.Vec,
        )
        cg = pl.make_tile_group(
            type=cg_type,
            addrs=4 * chunk + 256,
            mutex_ids="auto",
            depth=1,
        )
        ig_type = pl.TileType(
            shape=[1, chunk],
            dtype=I32,
            target_memory=pl.MemorySpace.Vec,
        )
        ig = pl.make_tile_group(
            type=ig_type,
            addrs=4 * chunk + 512,
            mutex_ids="auto",
            depth=1,
        )
        vg_type = pl.TileType(
            shape=[1, chunk],
            dtype=BF,
            target_memory=pl.MemorySpace.Vec,
        )
        vg = pl.make_tile_group(
            type=vg_type,
            addrs=8 * chunk + 512,
            mutex_ids="auto",
            depth=1,
        )
        with pl.section_vector():
            for row in pl.range(pl.get_block_idx(), rows, pl.get_block_num()):
                src = sg.current()
                local = xg.current()
                ct = cg.current()
                oi = ig.current()
                ov = vg.current()
                offset = 0
                if has_offset:
                    offset = pl.getval(offsets, row)
                lo = 127
                if topk < width:
                    hi = 65535
                    for bit in pl.range(16):
                        midpoint = (lo + hi + 1) // 2
                        total = pl.const(0, pl.DT_INT64)
                        for start in pl.range(0, width, chunk):
                            active = pl.min(chunk, width - start)
                            pl.set_validshape(src, [1, active])
                            pl.load(src, scores, [row, start])
                            count_keys(src, ct, active, midpoint)
                            total = total + pl.getval(ct, 0)
                        if total >= topk:
                            lo = midpoint
                        else:
                            hi = midpoint - 1
                written = 0
                for phase in pl.range(2):
                    for start in pl.range(0, width, chunk):
                        if written < topk:
                            active = pl.min(chunk, width - start)
                            pl.set_validshape(src, [1, active])
                            pl.load(src, scores, [row, start])
                            if phase == 0:
                                collect_keys(src, local, ct, active, lo, False)
                            else:
                                collect_keys(src, local, ct, active, lo, True)
                            available = pl.getval(ct, 0)
                            count = pl.min(available, topk - written)
                            if count > 0:
                                emit_selection(
                                    local,
                                    src,
                                    oi,
                                    ov,
                                    count,
                                    count,
                                    start,
                                    local,
                                    False,
                                    offset,
                                )
                                pl.set_validshape(oi, [1, count])
                                pl.set_validshape(ov, [1, count])
                                pl.store(indices, oi, [row, written])
                                pl.store(values, ov, [row, written])
                                written = written + count
                for start in pl.range(written, topk, chunk):
                    count = pl.min(chunk, topk - start)
                    fill_padding(oi, ov, count)
                    pl.set_validshape(oi, [1, count])
                    pl.set_validshape(ov, [1, count])
                    pl.store(indices, oi, [row, start])
                    pl.store(values, ov, [row, start])

    return kernel
