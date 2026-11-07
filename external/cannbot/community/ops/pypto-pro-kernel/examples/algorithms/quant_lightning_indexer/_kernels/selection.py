#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""写法：原始（VF）。BF16 radix selection; VF threshold primitives adapted from PyPTO lightning_indexer ST.
Each worker selects one <=16384-element partition using <=150 KiB UB;
local pruning is exact because a discarded item has K predecessors.
The merge applies the same comparator to the retained partition prefixes.
"""

from functools import lru_cache
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

U16 = pl.DT_UINT16
I32 = pl.DT_INT32
U32 = pl.DT_UINT32
BF = pl.DT_BF16


@pl.vector_function
def keys_from_scores(scores, keys, size, valid_size):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U16)
    sign = vf.full(0x8000, p, dtype=U16)
    allbits = vf.full(0xFFFF, p, dtype=U16)
    zero = vf.full(0, p, dtype=U16)
    for c in pl.range(0, size, 128):
        value = vf.load_align(scores, [0, c])
        bits = vf.bit_cast(value, dtype=U16)
        negative = vf.ge(bits, sign, p)
        flip = vf.select(allbits, sign, negative)
        key = vf.xor(bits, flip, p)
        active = vf.update_mask(pl.max(0, pl.min(128, valid_size - c)), dtype=U16)
        safe_key = vf.select(key, zero, active)
        vf.store_align(keys, safe_key, p, [0, c])


@pl.vector_function
def emit_ordered(scores, out_idx, out_val, topk, count, offset, write_values=True):
    """All keys fit: preserve logical order and pad without selection."""
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U32)
    inf_bits = vf.full(0xFF80, p, dtype=U32)
    minus_one = vf.full(-1, p, dtype=I32)
    for c in pl.range(0, topk, 64):
        active = vf.update_mask(pl.min(64, topk - c), dtype=U32)
        present = vf.update_mask(pl.max(0, pl.min(64, count - c)), dtype=U32)
        raw = vf.load_align(
            scores,
            pl.min(c, (count - 1) // 64 * 64),
            dist=pl.LoadDist.UNPK_B16,
            dtype=U32,
        )
        valid = vf.ne(raw, inf_bits, present)
        index = vf.arange(offset + c, dtype=I32)
        oi = vf.select(index, minus_one, valid)
        ov = vf.select(raw, inf_bits, valid)
        vf.store_align(out_idx, oi, active, c)
        if write_values:
            vf.store_align(out_val, ov, active, c, dist=pl.StoreDist.PACK_B32)


@pl.vector_function
def emit_selection(
    indices16,
    scores,
    out_idx,
    out_val,
    topk,
    count,
    begin,
    mapping,
    mapped,
    offset,
    write_values=True,
):
    p32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U32)
    for c in pl.range(0, topk, 64):
        active = vf.update_mask(pl.min(64, topk - c), dtype=U32)
        valid = vf.update_mask(pl.max(0, pl.min(64, count - c)), dtype=U32)
        raw = vf.load_align(
            indices16,
            [0, pl.min(c, pl.max(0, count - 1) // 64 * 64)],
            dist=pl.LoadDist.UNPK_B16,
            dtype=U32,
        )
        zero = vf.full(0, p32, dtype=U32)
        idx = vf.select(raw, zero, valid)
        value = vf.gather(scores, idx, active)
        # Gather with b32 indices yields bf16 payload in low halfwords.
        index = vf.adds(idx, begin, active)
        negone = vf.full(-1, p32, dtype=I32)
        si = vf.bit_cast(index, dtype=I32)
        if mapped:
            si = vf.gather(mapping, idx, active)
        shifted = vf.adds(si, offset, active)
        bits = vf.bit_cast(value, dtype=U32)
        invalid_bits = vf.full(0xFF80, p32, dtype=U32)
        finite = vf.ne(bits, invalid_bits, valid)
        oi = vf.select(shifted, negone, finite)
        vf.store_align(out_idx, oi, active, [0, c])
        ov = vf.select(bits, invalid_bits, finite)
        if write_values:
            vf.store_align(out_val, ov, active, [0, c], dist=pl.StoreDist.PACK_B32)


@lru_cache(None)
def make_select(local_width, topk, chunk=16384, mapped=False, add_offset=False):
    output_width = (topk + 127) // 128 * 128

    @pl.jit(auto_mutex=True)
    def kernel(
        scores: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], BF],
        index_map: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], I32],
        offsets: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], I32],
        out_idx: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, topk], I32],
        out_val: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, topk], BF],
    ):
        rows = scores.shape[0]
        width = scores.shape[1]
        part_count = out_idx.shape[1]
        sg_type = pl.TileType(
            shape=[1, local_width],
            dtype=BF,
            target_memory=pl.MemorySpace.Vec,
        )
        sg = pl.make_tile_group(
            type=sg_type,
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        kg_type = pl.TileType(
            shape=[1, local_width],
            dtype=U16,
            target_memory=pl.MemorySpace.Vec,
        )
        kg = pl.make_tile_group(
            type=kg_type,
            addrs=2 * local_width,
            mutex_ids="auto",
            depth=1,
        )
        hg_type = pl.TileType(
            shape=[1, 256],
            dtype=U32,
            target_memory=pl.MemorySpace.Vec,
        )
        hg = pl.make_tile_group(
            type=hg_type,
            addrs=4 * local_width,
            mutex_ids="auto",
            depth=1,
        )
        lg_type = pl.TileType(
            shape=[1, 256],
            dtype=U32,
            target_memory=pl.MemorySpace.Vec,
        )
        lg = pl.make_tile_group(
            type=lg_type,
            addrs=4 * local_width + 1024,
            mutex_ids="auto",
            depth=1,
        )
        bg_type = pl.TileType(
            shape=[1, 256],
            dtype=U32,
            target_memory=pl.MemorySpace.Vec,
        )
        bg = pl.make_tile_group(
            type=bg_type,
            addrs=4 * local_width + 2048,
            mutex_ids="auto",
            depth=1,
        )
        ig_type = pl.TileType(
            shape=[1, 128],
            dtype=U32,
            target_memory=pl.MemorySpace.Vec,
        )
        ig = pl.make_tile_group(
            type=ig_type,
            addrs=4 * local_width + 3072,
            mutex_ids="auto",
            depth=1,
        )
        ng_type = pl.TileType(
            shape=[1, 128],
            dtype=U32,
            target_memory=pl.MemorySpace.Vec,
        )
        ng = pl.make_tile_group(
            type=ng_type,
            addrs=4 * local_width + 3584,
            mutex_ids="auto",
            depth=1,
        )
        tg_type = pl.TileType(
            shape=[1, 128],
            dtype=U16,
            target_memory=pl.MemorySpace.Vec,
        )
        tg = pl.make_tile_group(
            type=tg_type,
            addrs=4 * local_width + 4096,
            mutex_ids="auto",
            depth=1,
        )
        xg_type = pl.TileType(
            shape=[1, local_width + 128],
            dtype=U16,
            target_memory=pl.MemorySpace.Vec,
        )
        xg = pl.make_tile_group(
            type=xg_type,
            addrs=4 * local_width + 4352,
            mutex_ids="auto",
            depth=1,
        )
        og_type = pl.TileType(
            shape=[1, output_width],
            dtype=I32,
            target_memory=pl.MemorySpace.Vec,
        )
        og = pl.make_tile_group(
            type=og_type,
            addrs=6 * local_width + 4608,
            mutex_ids="auto",
            depth=1,
        )
        vg_type = pl.TileType(
            shape=[1, output_width],
            dtype=BF,
            target_memory=pl.MemorySpace.Vec,
        )
        vg = pl.make_tile_group(
            type=vg_type,
            addrs=6 * local_width + 4608 + 4 * output_width,
            mutex_ids="auto",
            depth=1,
        )
        mg_type = pl.TileType(
            shape=[1, local_width if mapped else 64],
            dtype=I32,
            target_memory=pl.MemorySpace.Vec,
        )
        mg = pl.make_tile_group(
            type=mg_type,
            addrs=6 * local_width + 4608 + 6 * output_width,
            mutex_ids="auto",
            depth=1,
        )
        with pl.section_vector():
            for work in pl.range(
                pl.get_block_idx(), rows * part_count, pl.get_block_num()
            ):
                row = work // part_count
                part = work % part_count
                size = pl.min(local_width, width - part * chunk)
                st = sg.current()
                kt = kg.current()
                ht = hg.current()
                lt = lg.current()
                bt = bg.current()
                it = ig.current()
                nt = ng.current()
                tt = tg.current()
                xt = xg.current()
                ot = og.current()
                vt = vg.current()
                mt = mg.current()
                pl.set_validshape(st, [1, size])
                pl.load(st, scores, [row, part * chunk])
                if mapped:
                    pl.set_validshape(mt, [1, size])
                    pl.load(mt, index_map, [row, part * chunk])
                keys_from_scores(st, kt, local_width, size)
                histograms_high_vf(ht, kt, local_width // 256)
                find_high_bin_and_histograms_low(
                    lt,
                    it,
                    ht,
                    kt,
                    bt,
                    nt,
                    local_width - pl.min(topk, size) + 1,
                    local_width // 256,
                )
                find_kth_vf(tt, lt, it, bt, nt)
                collect_indices_gt_eq_vf(xt, kt, tt, 0, local_width // 128)
                offset = 0
                if add_offset:
                    offset = pl.getval(offsets, row)
                emit_selection(
                    xt,
                    st,
                    ot,
                    vt,
                    topk,
                    pl.min(topk, size),
                    part * chunk,
                    mt,
                    mapped,
                    offset,
                )
                pl.set_validshape(ot, [1, topk])
                pl.set_validshape(vt, [1, topk])
                pl.store(out_idx, ot, [row, part, 0])
                pl.store(out_val, vt, [row, part, 0])

    return kernel


@pl.vector_function
def histograms_high_vf(hist_high_tile, score_u16_tile, hist_loop: pl.DT_INT64, base=0):
    """Histogram high byte (BYTE_1, MSB) — chistv2 accumulates into dst reg."""
    preg_b8 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT8)
    preg_b16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    preg_b32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)

    cout0 = vf.full(0, preg_b16, dtype=pl.DT_UINT16)
    cout1 = vf.full(0, preg_b16, dtype=pl.DT_UINT16)

    for i in pl.range(0, hist_loop):
        vreg_low, vreg_high = vf.load_align(
            score_u16_tile,
            base + i * 256,
            dist=pl.LoadDist.DINTLV_B8,
            dtype=pl.DT_UINT8,
        )
        cout0 = vf.histograms(
            vreg_high,
            preg_b8,
            bin_type=pl.BinType.BIN0,
            hist_type=pl.HistType.ACCUMULATE,
        )
        cout1 = vf.histograms(
            vreg_high,
            preg_b8,
            bin_type=pl.BinType.BIN1,
            hist_type=pl.HistType.ACCUMULATE,
        )

    c0_even = vf.astype(cout0, preg_b16, layout=pl.CastLayout.ZERO, dtype=pl.DT_UINT32)
    c0_odd = vf.astype(cout0, preg_b16, layout=pl.CastLayout.ONE, dtype=pl.DT_UINT32)
    c1_even = vf.astype(cout1, preg_b16, layout=pl.CastLayout.ZERO, dtype=pl.DT_UINT32)
    c1_odd = vf.astype(cout1, preg_b16, layout=pl.CastLayout.ONE, dtype=pl.DT_UINT32)
    vf.store_align(
        hist_high_tile, c0_even, c0_odd, preg_b32, dist=pl.StoreDist.INTLV_B32
    )
    vf.store_align(
        hist_high_tile + 128, c1_even, c1_odd, preg_b32, dist=pl.StoreDist.INTLV_B32
    )
    vf.mem_bar()


@pl.vector_function
def find_high_bin_and_histograms_low(
    hist_low_tile,
    idx_high_tile,
    hist_high_tile,
    score_u16_tile,
    sqz_idx_buf,
    nk_value_tile,
    bottom_k,
    hist_loop,
    base=0,
):
    p8 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT8)
    p16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U16)
    p32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U32)
    target = vf.full(bottom_k, p32, dtype=U32)
    absent = vf.full(256, p32, dtype=U32)
    best = vf.full(256, p32, dtype=U32)
    for i in pl.range(4):
        idx = vf.arange(i * 64, dtype=U32)
        counts = vf.load_align(hist_high_tile, i * 64)
        match = vf.ge(counts, target, p32)
        chosen = vf.select(idx, absent, match)
        best = vf.min(best, chosen, p32)
    minimum = vf.reduce_min(best, p32)
    high = vf.full(minimum, p32)
    prev = vf.maxs(high, 1, p32)
    one = vf.full(1, p32, dtype=U32)
    prev = vf.sub(prev, one, p32)
    counts = vf.gather(hist_high_tile, prev, p32)
    zero = vf.full(0, p32, dtype=U32)
    has_prev = vf.gt(high, zero, p32)
    preceding = vf.select(counts, zero, has_prev)
    next_k = vf.sub(target, preceding, p32)
    repeated = vf.muls(high, 0x01010101, p32)
    vf.store_align(idx_high_tile, repeated, p32, 0)
    vf.store_align(nk_value_tile, next_k, p32, 0)
    high8 = vf.bit_cast(repeated, dtype=pl.DT_UINT8)
    count0 = vf.full(0, p16, dtype=U16)
    count1 = vf.full(0, p16, dtype=U16)
    for i in pl.range(hist_loop):
        low, hi = vf.load_align(
            score_u16_tile,
            base + i * 256,
            dist=pl.LoadDist.DINTLV_B8,
            dtype=pl.DT_UINT8,
        )
        match = vf.eq(hi, high8, p8)
        count0 = vf.histograms(
            low, match, bin_type=pl.BinType.BIN0, hist_type=pl.HistType.ACCUMULATE
        )
        count1 = vf.histograms(
            low, match, bin_type=pl.BinType.BIN1, hist_type=pl.HistType.ACCUMULATE
        )
    e0 = vf.astype(count0, p16, layout=pl.CastLayout.ZERO, dtype=U32)
    o0 = vf.astype(count0, p16, layout=pl.CastLayout.ONE, dtype=U32)
    e1 = vf.astype(count1, p16, layout=pl.CastLayout.ZERO, dtype=U32)
    o1 = vf.astype(count1, p16, layout=pl.CastLayout.ONE, dtype=U32)
    vf.store_align(hist_low_tile, e0, o0, p32, dist=pl.StoreDist.INTLV_B32)
    vf.store_align(hist_low_tile + 128, e1, o1, p32, dist=pl.StoreDist.INTLV_B32)


@pl.vector_function
def find_kth_vf(
    kth_value_tile, hist_low_tile, idx_high_tile, sqz_idx_buf, nk_value_tile
):
    p16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U16)
    p32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U32)
    target = vf.load_align(nk_value_tile, 0)
    absent = vf.full(256, p32, dtype=U32)
    best = vf.full(256, p32, dtype=U32)
    for i in pl.range(4):
        idx = vf.arange(i * 64, dtype=U32)
        counts = vf.load_align(hist_low_tile, i * 64)
        match = vf.ge(counts, target, p32)
        chosen = vf.select(idx, absent, match)
        best = vf.min(best, chosen, p32)
    minimum = vf.reduce_min(best, p32)
    low = vf.full(minimum, p32)
    high = vf.load_align(idx_high_tile, 0)
    mask = vf.full(255, p32, dtype=U32)
    high = vf.and_(high, mask, p32)
    high = vf.shift_left(high, 8, p32)
    key = vf.or_(high, low, p32)
    upper = vf.shift_left(key, 16, p32)
    both = vf.or_(key, upper, p32)
    key16 = vf.bit_cast(both, dtype=U16)
    vf.store_align(kth_value_tile, key16, p16, 0)


@pl.vector_function
def collect_indices_gt_eq_vf(
    output_idx_tile,
    score_u16_tile,
    kth_value_tile,
    begin_idx: pl.DT_INT64,
    vf_loop: pl.DT_INT64,
    base=0,
):
    """Collect indices where value > kth (GT), then value == kth (EQ).

    GT indices come first, EQ indices follow. The downstream pl.store
    with set_validshape(topk) truncates to exactly topk entries.
    Output indices are DT_INT32 (matching reference implementation).
    """
    preg_b16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)

    vf.clear_spr()
    align_idx = vf.unalign_reg_for_store()

    kth_value = vf.load_align(
        kth_value_tile, 0, dist=pl.LoadDist.BRC_B16, dtype=pl.DT_UINT16
    )

    # Pass 1: collect GT indices
    for i in pl.range(0, vf_loop):
        idx_c = vf.arange(begin_idx + i * 128, dtype=pl.DT_INT16)
        vreg_input = vf.load_align(score_u16_tile, base + i * 128)
        pout_gt = vf.gt(vreg_input, kth_value, preg_b16)
        sqz_idx_out = vf.squeeze(
            idx_c, pout_gt, gather_mode=pl.SqueezeMode.STORE_REG, dtype=pl.DT_UINT16
        )
        vf.squeeze_store_unalign(output_idx_tile, sqz_idx_out, align_idx)

    # Pass 2: collect EQ indices (appends after GT)
    for i in pl.range(0, vf_loop):
        idx_c = vf.arange(begin_idx + i * 128, dtype=pl.DT_INT16)
        vreg_input = vf.load_align(score_u16_tile, base + i * 128)
        pout_eq = vf.eq(vreg_input, kth_value, preg_b16)
        sqz_idx_out = vf.squeeze(
            idx_c, pout_eq, gather_mode=pl.SqueezeMode.STORE_REG, dtype=pl.DT_UINT16
        )
        vf.squeeze_store_unalign(output_idx_tile, sqz_idx_out, align_idx)

    vf.squeeze_store_unalign_post(output_idx_tile, align_idx)
