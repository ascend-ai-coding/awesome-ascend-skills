# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始（保留 L0 操作数；原生 VF 与 B 层流水）。

Each Cube worker owns a six-query group and a contiguous key shard. Q/K
are packed E2M1, with four E8M0 scales per 128 logical elements. Both L0A
query halves remain live throughout the shard, and each L0B key panel feeds
two distinct accumulators. This is manual GEMM exception 6: ordinary sugar
would stage the unchanged operands again. Per-slot auto_mutex is retained.
QK, weights and every sequential head FMA have the upstream BF16 rounding.

Each AIV owns three queries. For N=256 the Acc view [96,256] -> [192,128]
permutes 16-column groups, without moving bytes. The consumer reduces the
matching head rows; cache_select restores logical key indices on emission.
Partial masks follow that same permutation. Two interleaved UB slots use
address(slot,row,col)=256*slot+512*row+2*col, with row<192 and col<128.
Their accessed byte sets are disjoint. The conservative bounding boxes
overlap, explaining MemSpaceCollision; full-width accesses are prohibited.
Cube/VF handshakes protect each slot. A drained MIX barrier separates all
partial writes from the final merge, after which the UB views can be reused.

Resource spans: Mat ends at 198656 bytes (47872 bytes of live payload),
Left=12288, Right=32768, ScaleLeft=768, ScaleRight=2048, Acc=196608 bytes.
Vec ends at 108544+8*L bytes for an L-token shard, including all slots,
cached keys, histograms and selection scratch; dispatch enforces L<=19200.
The kernel emits only bounded partial Top-K lists to GM and merges them
in the same native call. No full score matrix is materialized.
"""

from functools import lru_cache

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

from .selection import (
    find_high_bin_and_histograms_low,
    find_kth_vf,
    collect_indices_gt_eq_vf,
    histograms_high_vf,
    keys_from_scores,
)
from .cache_select import clear_cache, widen_histogram, emit_cached

F4 = pl.DT_FP4E2M1
E8 = pl.DT_FP8E8M0
BF = pl.DT_BF16
F32 = pl.DT_FP32
I32 = pl.DT_INT32


def vector_span(shard_width):
    """Highest declared UB byte; hardware capacity remains framework-owned."""
    return 108544 + 8 * shard_width


@lru_cache(None)
def make_paged(
    max_k,
    page_size,
    k_page_stride,
    ks_page_stride,
    paged=True,
    mask=3,
    ratio=2,
    has_residual=True,
    tile_n=256,
    split=7,
    has_offset=False,
    return_values=False,
):
    assert paged and page_size == 128 and tile_n == 256
    topk = 512
    preload = 0
    handover_depth = 2
    vg_ids = [24, 25]
    forward_ids = list(range(handover_depth))
    backward_ids = list(range(handover_depth, 2 * handover_depth))
    mq = 6
    mm = mq * 32
    half = mm // 2
    shard_tiles = (max_k + tile_n * split - 1) // (tile_n * split)
    shard_width = shard_tiles * tile_n
    total_tiles = (max_k + tile_n - 1) // tile_n
    parts = (total_tiles + shard_tiles - 1) // shard_tiles
    merge_width = parts * topk
    assert merge_width <= shard_width
    kr = 128
    kn = 256

    @pl.vector_function
    def convert_weights(src, dst, count):
        p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=F32)
        for c in pl.range(0, 128, 64):
            active = vf.update_mask(pl.max(0, pl.min(64, count - c)), dtype=F32)
            value = vf.load_align(src, c)
            narrow = vf.astype(value, active, dtype=BF, layout=pl.CastLayout.ZERO)
            bits = vf.bit_cast(narrow, dtype=pl.DT_UINT32)
            vf.store_align(dst, bits, p, c, dist=pl.StoreDist.PACK_B32)

    @pl.vector_function
    def reduce_heads(
        src, weight, dst, hist, position, e0, o0, e1, o1, e2, o2, all_valid
    ):
        p16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=BF)
        zb = vf.full(0, p16, dtype=pl.DT_UINT16)
        zero = vf.bit_cast(zb, dtype=BF)
        sign = vf.full(32768, p16, dtype=pl.DT_UINT16)
        flipneg = vf.full(65535, p16, dtype=pl.DT_UINT16)
        p8 = vf.update_mask(128, dtype=pl.DT_UINT8)
        acc0_0 = vf.full(zero, p16)
        acc0_1 = vf.full(zero, p16)
        acc1_0 = vf.full(zero, p16)
        acc1_1 = vf.full(zero, p16)
        acc2_0 = vf.full(zero, p16)
        acc2_1 = vf.full(zero, p16)
        for h in pl.range(32):
            w0 = vf.load_align(weight, [0, h], dist=pl.LoadDist.BRC_B16)
            qk0_0 = vf.load_align(src, [0 * 32 + h + 0 * half, 0])
            relu0_0 = vf.max(qk0_0, zero, p16)
            acc0_0 = vf.mul_add_dst(relu0_0, w0, p16)
            qk0_1 = vf.load_align(src, [0 * 32 + h + 1 * half, 0])
            relu0_1 = vf.max(qk0_1, zero, p16)
            acc0_1 = vf.mul_add_dst(relu0_1, w0, p16)
            w1 = vf.load_align(weight, [1, h], dist=pl.LoadDist.BRC_B16)
            qk1_0 = vf.load_align(src, [1 * 32 + h + 0 * half, 0])
            relu1_0 = vf.max(qk1_0, zero, p16)
            acc1_0 = vf.mul_add_dst(relu1_0, w1, p16)
            qk1_1 = vf.load_align(src, [1 * 32 + h + 1 * half, 0])
            relu1_1 = vf.max(qk1_1, zero, p16)
            acc1_1 = vf.mul_add_dst(relu1_1, w1, p16)
            w2 = vf.load_align(weight, [2, h], dist=pl.LoadDist.BRC_B16)
            qk2_0 = vf.load_align(src, [2 * 32 + h + 0 * half, 0])
            relu2_0 = vf.max(qk2_0, zero, p16)
            acc2_0 = vf.mul_add_dst(relu2_0, w2, p16)
            qk2_1 = vf.load_align(src, [2 * 32 + h + 1 * half, 0])
            relu2_1 = vf.max(qk2_1, zero, p16)
            acc2_1 = vf.mul_add_dst(relu2_1, w2, p16)
        h0lo = vf.load_align(hist, 0 * 256)
        h0hi = vf.load_align(hist, 0 * 256 + 128)
        bits0_0 = vf.bit_cast(acc0_0, dtype=pl.DT_UINT16)
        negative0_0 = vf.ge(bits0_0, sign, p16)
        flip0_0 = vf.select(flipneg, sign, negative0_0)
        rawkey0_0 = vf.xor(bits0_0, flip0_0, p16)
        if all_valid:
            key0_0 = rawkey0_0
        else:
            mask0_0 = vf.update_mask(e0, dtype=BF)
            key0_0 = vf.select(rawkey0_0, zb, mask0_0)
        vf.store_align(dst, key0_0, p16, [0, position + 0 * 128])
        high0_0 = vf.shift_right(key0_0, 8, p16)
        packed0_0 = vf.pack(high0_0, dtype=pl.DT_UINT8, part=pl.PackPart.LOWER)
        h0lo = vf.histograms(
            packed0_0, p8, bin_type=pl.BinType.BIN0, hist_type=pl.HistType.ACCUMULATE
        )
        h0hi = vf.histograms(
            packed0_0, p8, bin_type=pl.BinType.BIN1, hist_type=pl.HistType.ACCUMULATE
        )
        bits0_1 = vf.bit_cast(acc0_1, dtype=pl.DT_UINT16)
        negative0_1 = vf.ge(bits0_1, sign, p16)
        flip0_1 = vf.select(flipneg, sign, negative0_1)
        rawkey0_1 = vf.xor(bits0_1, flip0_1, p16)
        if all_valid:
            key0_1 = rawkey0_1
        else:
            mask0_1 = vf.update_mask(o0, dtype=BF)
            key0_1 = vf.select(rawkey0_1, zb, mask0_1)
        vf.store_align(dst, key0_1, p16, [0, position + 1 * 128])
        high0_1 = vf.shift_right(key0_1, 8, p16)
        packed0_1 = vf.pack(high0_1, dtype=pl.DT_UINT8, part=pl.PackPart.LOWER)
        h0lo = vf.histograms(
            packed0_1, p8, bin_type=pl.BinType.BIN0, hist_type=pl.HistType.ACCUMULATE
        )
        h0hi = vf.histograms(
            packed0_1, p8, bin_type=pl.BinType.BIN1, hist_type=pl.HistType.ACCUMULATE
        )
        vf.store_align(hist, h0lo, p16, 0 * 256)
        vf.store_align(hist, h0hi, p16, 0 * 256 + 128)
        h1lo = vf.load_align(hist, 1 * 256)
        h1hi = vf.load_align(hist, 1 * 256 + 128)
        bits1_0 = vf.bit_cast(acc1_0, dtype=pl.DT_UINT16)
        negative1_0 = vf.ge(bits1_0, sign, p16)
        flip1_0 = vf.select(flipneg, sign, negative1_0)
        rawkey1_0 = vf.xor(bits1_0, flip1_0, p16)
        if all_valid:
            key1_0 = rawkey1_0
        else:
            mask1_0 = vf.update_mask(e1, dtype=BF)
            key1_0 = vf.select(rawkey1_0, zb, mask1_0)
        vf.store_align(dst, key1_0, p16, [1, position + 0 * 128])
        high1_0 = vf.shift_right(key1_0, 8, p16)
        packed1_0 = vf.pack(high1_0, dtype=pl.DT_UINT8, part=pl.PackPart.LOWER)
        h1lo = vf.histograms(
            packed1_0, p8, bin_type=pl.BinType.BIN0, hist_type=pl.HistType.ACCUMULATE
        )
        h1hi = vf.histograms(
            packed1_0, p8, bin_type=pl.BinType.BIN1, hist_type=pl.HistType.ACCUMULATE
        )
        bits1_1 = vf.bit_cast(acc1_1, dtype=pl.DT_UINT16)
        negative1_1 = vf.ge(bits1_1, sign, p16)
        flip1_1 = vf.select(flipneg, sign, negative1_1)
        rawkey1_1 = vf.xor(bits1_1, flip1_1, p16)
        if all_valid:
            key1_1 = rawkey1_1
        else:
            mask1_1 = vf.update_mask(o1, dtype=BF)
            key1_1 = vf.select(rawkey1_1, zb, mask1_1)
        vf.store_align(dst, key1_1, p16, [1, position + 1 * 128])
        high1_1 = vf.shift_right(key1_1, 8, p16)
        packed1_1 = vf.pack(high1_1, dtype=pl.DT_UINT8, part=pl.PackPart.LOWER)
        h1lo = vf.histograms(
            packed1_1, p8, bin_type=pl.BinType.BIN0, hist_type=pl.HistType.ACCUMULATE
        )
        h1hi = vf.histograms(
            packed1_1, p8, bin_type=pl.BinType.BIN1, hist_type=pl.HistType.ACCUMULATE
        )
        vf.store_align(hist, h1lo, p16, 1 * 256)
        vf.store_align(hist, h1hi, p16, 1 * 256 + 128)
        h2lo = vf.load_align(hist, 2 * 256)
        h2hi = vf.load_align(hist, 2 * 256 + 128)
        bits2_0 = vf.bit_cast(acc2_0, dtype=pl.DT_UINT16)
        negative2_0 = vf.ge(bits2_0, sign, p16)
        flip2_0 = vf.select(flipneg, sign, negative2_0)
        rawkey2_0 = vf.xor(bits2_0, flip2_0, p16)
        if all_valid:
            key2_0 = rawkey2_0
        else:
            mask2_0 = vf.update_mask(e2, dtype=BF)
            key2_0 = vf.select(rawkey2_0, zb, mask2_0)
        vf.store_align(dst, key2_0, p16, [2, position + 0 * 128])
        high2_0 = vf.shift_right(key2_0, 8, p16)
        packed2_0 = vf.pack(high2_0, dtype=pl.DT_UINT8, part=pl.PackPart.LOWER)
        h2lo = vf.histograms(
            packed2_0, p8, bin_type=pl.BinType.BIN0, hist_type=pl.HistType.ACCUMULATE
        )
        h2hi = vf.histograms(
            packed2_0, p8, bin_type=pl.BinType.BIN1, hist_type=pl.HistType.ACCUMULATE
        )
        bits2_1 = vf.bit_cast(acc2_1, dtype=pl.DT_UINT16)
        negative2_1 = vf.ge(bits2_1, sign, p16)
        flip2_1 = vf.select(flipneg, sign, negative2_1)
        rawkey2_1 = vf.xor(bits2_1, flip2_1, p16)
        if all_valid:
            key2_1 = rawkey2_1
        else:
            mask2_1 = vf.update_mask(o2, dtype=BF)
            key2_1 = vf.select(rawkey2_1, zb, mask2_1)
        vf.store_align(dst, key2_1, p16, [2, position + 1 * 128])
        high2_1 = vf.shift_right(key2_1, 8, p16)
        packed2_1 = vf.pack(high2_1, dtype=pl.DT_UINT8, part=pl.PackPart.LOWER)
        h2lo = vf.histograms(
            packed2_1, p8, bin_type=pl.BinType.BIN0, hist_type=pl.HistType.ACCUMULATE
        )
        h2hi = vf.histograms(
            packed2_1, p8, bin_type=pl.BinType.BIN1, hist_type=pl.HistType.ACCUMULATE
        )
        vf.store_align(hist, h2lo, p16, 2 * 256)
        vf.store_align(hist, h2hi, p16, 2 * 256 + 128)

    @pl.pipeline.stage
    def cube_step(
        t,
        qbase,
        qvalid,
        b,
        kbase,
        kl,
        q,
        k,
        qs,
        ks,
        table,
        qg,
        kg,
        qsg,
        ksg,
        la,
        lb,
        sa,
        sb,
        cg,
        cgv,
        vg,
        kp0,
        kp1,
        sp0,
        sp1,
    ):
        qt = qg.current()
        qst = qsg.current()
        if t == 0:
            pl.set_validshape(qt, [qvalid * 32, 128])
            pl.set_validshape(qst, [qvalid * 32, 4])
            pl.load(qt, q, [qbase * 32, 0])
            pl.load(qst, qs, [qbase * 32, 0, 0])
        kt = kg.next()
        kst = ksg.next()
        active = pl.min(tile_n, pl.max(0, kl - kbase))
        pl.set_validshape(kt, [128, active])
        pl.set_validshape(kst, [4, active])
        physical = kbase
        physical_s = kbase
        page = 0
        if active > 0:
            page = pl.getval(table, b * table.shape[1] + kbase // page_size)
        physical = page * k_page_stride + kbase % page_size
        physical_s = page * ks_page_stride + kbase % page_size
        p0 = kp0.next()
        p1 = kp1.next()
        s0 = sp0.next()
        s1 = sp1.next()
        page1 = page
        if active > 128:
            page1 = pl.getval(table, b * table.shape[1] + kbase // page_size + 1)
        pl.set_validshape(p0, [128, 128])
        pl.set_validshape(p1, [128, 128])
        pl.load(p0, k, [physical, 0], order=[1, 0])
        pl.load(p1, k, [page1 * k_page_stride, 0], order=[1, 0])
        pl.load(s0, ks, [physical_s, 0, 0], order=[1, 0])
        pl.load(s1, ks, [page1 * ks_page_stride, 0, 0], order=[1, 0])
        a0 = la[0]
        a1 = la[1]
        as0 = sa[0]
        as1 = sa[1]
        if t == 0:
            pl.move(a0, qt, [0, 0])
            pl.move(a1, qt, [half, 0])
            pl.move(as0, qst, [0, 0])
            pl.move(as1, qst, [half, 0])
        br = lb.next()
        bs = sb.next()
        pl.move(br, kt)
        pl.move(bs, kst)
        ct0 = cg.next()
        pl.matmul_mx(ct0, a0, br, scale_a=as0, scale_b=bs)
        vt = vg.next()
        folded0 = cgv[0]
        pl.set_validshape(vt, [half * 2, tile_n // 2])
        pl.move(vt, folded0, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
        ct1 = cg.next()
        pl.matmul_mx(ct1, a1, br, scale_a=as1, scale_b=bs)
        folded1 = cgv[1]
        pl.move(vt, folded1, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec1)

    @pl.pipeline.stage
    def vector_step(
        t,
        qbase,
        qvalid,
        qlocal,
        b,
        kbase,
        kl,
        used,
        residual,
        weights,
        out_idx,
        out_val,
        part,
        steps,
        visible0,
        visible1,
        visible2,
        wg,
        wbg,
        og,
        vg,
        hg,
        hhg,
        hlg,
        bg,
        ig,
        ng,
        tg,
        xg,
        oig,
        ovg,
    ):
        vt = vg.next()
        wt = wg.current()
        wbt = wbg.current()
        sub = pl.get_subblock_idx()
        if t == 0:
            pl.set_validshape(wt, [pl.max(0, pl.min(3, qvalid - sub * 3)), 32])
            if qvalid > sub * 3:
                pl.load(wt, weights, [qbase + sub * 3, 0])
            convert_weights(wt, wbt, pl.max(0, pl.min(3, qvalid - sub * 3)) * 32)
        ot = og.current()
        hist = hg.current()
        if t == 0:
            clear_cache(ot, hist, shard_width, steps * tile_n)
        valid0 = pl.max(0, pl.min(tile_n, visible0 - kbase))
        valid1 = pl.max(0, pl.min(tile_n, visible1 - kbase))
        valid2 = pl.max(0, pl.min(tile_n, visible2 - kbase))
        e0 = valid0 // 32 * 16 + pl.min(16, valid0 % 32)
        o0 = valid0 // 32 * 16 + pl.max(0, valid0 % 32 - 16)
        e1 = valid1 // 32 * 16 + pl.min(16, valid1 % 32)
        o1 = valid1 // 32 * 16 + pl.max(0, valid1 % 32 - 16)
        e2 = valid2 // 32 * 16 + pl.min(16, valid2 % 32)
        o2 = valid2 // 32 * 16 + pl.max(0, valid2 % 32 - 16)
        if kbase + tile_n <= pl.min(visible0, pl.min(visible1, visible2)):
            reduce_heads(vt, wbt, ot, hist, t * tile_n, e0, o0, e1, o1, e2, o2, True)
        else:
            reduce_heads(vt, wbt, ot, hist, t * tile_n, e0, o0, e1, o1, e2, o2, False)
        nr = pl.max(0, pl.min(3, qvalid - sub * 3))
        if t == steps - 1:
            ht = hhg.current()
            lt = hlg.current()
            bt = bg.current()
            it = ig.current()
            nt = ng.current()
            tt = tg.current()
            xt = xg.current()
            oi = oig.current()
            ov = ovg.current()
            for r in pl.range(nr):
                widen_histogram(hist, ht, r)
                find_high_bin_and_histograms_low(
                    lt,
                    it,
                    ht,
                    ot,
                    bt,
                    nt,
                    shard_width - topk + 1,
                    shard_width // 256,
                    r * shard_width,
                )
                find_kth_vf(tt, lt, it, bt, nt)
                collect_indices_gt_eq_vf(
                    xt, ot, tt, 0, shard_width // 128, r * shard_width
                )
                emit_cached(
                    xt,
                    ot,
                    oi,
                    ov,
                    topk,
                    r * shard_width,
                    part * shard_width,
                    False,
                    0,
                    True,
                )
                pl.store(out_idx, oi, [qbase + sub * 3 + r, part * topk])
                pl.store(out_val, ov, [qbase + sub * 3 + r, part * topk])

    @pl.jit(
        arch="a5", auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=preload)
    )
    def kernel(
        q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], F4],
        k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], F4],
        weights: pl.Tensor[[pl.DYNAMIC, 32], F32],
        qs: pl.Tensor[[pl.DYNAMIC, 2, 2], E8],
        ks: pl.Tensor[[pl.DYNAMIC, 2, 2], E8],
        cuq: pl.Tensor[[pl.DYNAMIC], I32],
        usedq: pl.Tensor[[pl.DYNAMIC], I32],
        usedk: pl.Tensor[[pl.DYNAMIC], I32],
        residuals: pl.Tensor[[pl.DYNAMIC], I32],
        table: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], I32],
        out_idx: pl.Tensor[[pl.DYNAMIC, parts * topk], I32],
        out_val: pl.Tensor[[pl.DYNAMIC, parts * topk], BF],
        final_idx: pl.Tensor[[pl.DYNAMIC, topk], I32],
        final_val: pl.Tensor[[pl.DYNAMIC, topk], BF],
        offsets: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], I32],
        max_q: pl.DT_INT64,
    ):
        rows = weights.shape[0]
        batches = usedq.shape[0]
        qgroups = (max_q + mq - 1) // mq
        grid = batches * qgroups * parts
        qg_type = pl.TileType(
            shape=[mm, 128],
            dtype=F4,
            target_memory=pl.MemorySpace.Mat,
        )
        qg = pl.make_tile_group(
            type=qg_type,
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        qsg_type = pl.TileType(
            shape=[mm, 4],
            dtype=E8,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZZ,
        )
        qsg = pl.make_tile_group(
            type=qsg_type,
            addrs=64 * mm,
            mutex_ids="auto",
            depth=1,
        )
        la_type = pl.TileType(
            shape=[half, 128],
            dtype=F4,
            target_memory=pl.MemorySpace.Left,
            compact=1,
        )
        la = pl.make_tile_group(
            type=la_type,
            addrs=0,
            mutex_ids="auto",
            depth=2,
        )
        lb_type = pl.TileType(
            shape=[128, tile_n],
            dtype=F4,
            target_memory=pl.MemorySpace.Right,
            compact=1,
        )
        lb = pl.make_tile_group(
            type=lb_type,
            addrs=0,
            mutex_ids="auto",
            depth=2,
        )
        sa_type = pl.TileType(
            shape=[half, 4],
            dtype=E8,
            target_memory=pl.MemorySpace.ScaleLeft,
            layout=pl.ZZ,
            compact=1,
        )
        sa = pl.make_tile_group(
            type=sa_type,
            addrs=[0, 64 * half >> 4],
            mutex_ids="auto",
            depth=2,
        )
        sb_type = pl.TileType(
            shape=[4, tile_n],
            dtype=E8,
            target_memory=pl.MemorySpace.ScaleRight,
            layout=pl.NN,
            compact=1,
        )
        sb = pl.make_tile_group(
            type=sb_type,
            addrs=[0, 64 * tile_n >> 4],
            mutex_ids="auto",
            depth=2,
        )
        cg = pl.make_tile_group(
            type=pl.TileType(
                shape=[half, tile_n],
                dtype=F32,
                target_memory=pl.MemorySpace.Acc,
                layout=pl.NZ,
                fractal=1024,
                compact=1,
            ),
            addrs=[0, half * tile_n * 4],
            mutex_ids=[10, 11],
        )
        cgv = pl.make_tile_group(
            type=pl.TileType(
                shape=[half * 2, tile_n // 2],
                dtype=F32,
                target_memory=pl.MemorySpace.Acc,
                layout=pl.NZ,
                fractal=1024,
                compact=1,
            ),
            addrs=[0, half * tile_n * 4],
            mutex_ids=[10, 11],
        )
        vg = pl.make_tile_group(
            type=pl.TileType(
                shape=[half * 2, tile_n], dtype=BF, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[0, tile_n],
            depth=handover_depth,
            mutex_ids=vg_ids,
            fwd_ids=forward_ids,
            bwd_ids=backward_ids,
        )
        wg = pl.make_tile_group(
            type=pl.TileType(
                shape=[4, 32], dtype=F32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[half * tile_n * 2 * handover_depth + 256],
            mutex_ids=[26],
        )
        wbg = pl.make_tile_group(
            type=pl.TileType(shape=[4, 32], dtype=BF, target_memory=pl.MemorySpace.Vec),
            addrs=[half * tile_n * 2 * handover_depth + 256 + 512],
            mutex_ids=[27],
        )
        og = pl.make_tile_group(
            type=pl.TileType(
                shape=[3, shard_width],
                dtype=pl.DT_UINT16,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=[half * tile_n * 2 * handover_depth + 256 + 768],
            mutex_ids=[0],
        )
        hg = pl.make_tile_group(
            type=pl.TileType(
                shape=[3, 256], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[
                half * tile_n * 2 * handover_depth + 256 + 768 + 3 * shard_width * 2
            ],
            mutex_ids=[1],
        )
        hhg = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, 256], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[
                half * tile_n * 2 * handover_depth
                + 256
                + 768
                + 3 * shard_width * 2
                + 1536
            ],
            mutex_ids=[2],
        )
        hlg = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, 256], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[
                half * tile_n * 2 * handover_depth
                + 256
                + 768
                + 3 * shard_width * 2
                + 1536
                + 1024
            ],
            mutex_ids=[3],
        )
        bg = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, 256], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[
                half * tile_n * 2 * handover_depth
                + 256
                + 768
                + 3 * shard_width * 2
                + 1536
                + 1024
                + 1024
            ],
            mutex_ids=[4],
        )
        ig = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, 128], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[
                half * tile_n * 2 * handover_depth
                + 256
                + 768
                + 3 * shard_width * 2
                + 1536
                + 1024
                + 1024
                + 1024
            ],
            mutex_ids=[5],
        )
        ng = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, 128], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[
                half * tile_n * 2 * handover_depth
                + 256
                + 768
                + 3 * shard_width * 2
                + 1536
                + 1024
                + 1024
                + 1024
                + 512
            ],
            mutex_ids=[6],
        )
        tg = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, 128], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[
                half * tile_n * 2 * handover_depth
                + 256
                + 768
                + 3 * shard_width * 2
                + 1536
                + 1024
                + 1024
                + 1024
                + 512
                + 512
            ],
            mutex_ids=[7],
        )
        xg = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, shard_width + 128],
                dtype=pl.DT_UINT16,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=[
                half * tile_n * 2 * handover_depth
                + 256
                + 768
                + 3 * shard_width * 2
                + 1536
                + 1024
                + 1024
                + 1024
                + 512
                + 512
                + 256
            ],
            mutex_ids=[8],
        )
        oig = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, topk], dtype=I32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[
                half * tile_n * 2 * handover_depth
                + 256
                + 768
                + 3 * shard_width * 2
                + 1536
                + 1024
                + 1024
                + 1024
                + 512
                + 512
                + 256
                + (shard_width + 128) * 2
            ],
            mutex_ids=[9],
        )
        ovg = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, topk], dtype=BF, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[
                half * tile_n * 2 * handover_depth
                + 256
                + 768
                + 3 * shard_width * 2
                + 1536
                + 1024
                + 1024
                + 1024
                + 512
                + 512
                + 256
                + (shard_width + 128) * 2
                + topk * 4
            ],
            mutex_ids=[10],
        )
        kg = pl.make_tile_group(
            type=pl.TileType(
                shape=[128, tile_n],
                dtype=pl.DT_FP4E2M1,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.ZN,
            ),
            addrs=[65536, 131072],
            mutex_ids=[20, 21],
        )
        kp0 = pl.make_tile_group(
            type=pl.TileType(
                shape=[kr, kn],
                dtype=pl.DT_FP4E2M1,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.ZN,
            ),
            addrs=[65536, 131072],
            mutex_ids=[20, 21],
        )
        kp1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[kr, kn],
                dtype=pl.DT_FP4E2M1,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.ZN,
            ),
            addrs=[69632, 135168],
            mutex_ids=[20, 21],
        )
        ksg = pl.make_tile_group(
            type=pl.TileType(
                shape=[4, tile_n],
                dtype=pl.DT_FP8E8M0,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NN,
            ),
            addrs=[196608, 196608 + 4 * tile_n],
            mutex_ids=[22, 23],
        )
        sp0 = pl.make_tile_group(
            type=pl.TileType(
                shape=[4, 128],
                dtype=pl.DT_FP8E8M0,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NN,
            ),
            addrs=[196608, 197632],
            mutex_ids=[22, 23],
        )
        sp1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[4, 128],
                dtype=pl.DT_FP8E8M0,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NN,
            ),
            addrs=[197120, 198144],
            mutex_ids=[22, 23],
        )
        core = pl.get_block_idx() // pl.get_subblock_num()
        for item in pl.range(core, grid, pl.get_block_num()):
            shard = item % parts
            qgroup = item // parts % qgroups
            b = item // (parts * qgroups)
            qstart = pl.getval(cuq, b)
            qend = pl.getval(cuq, b + 1)
            qlocal = qgroup * mq
            qbase = qstart + qlocal
            qvalid = pl.min(mq, pl.max(0, qend - qbase))
            kl = pl.getval(usedk, b)
            used = pl.getval(usedq, b)
            residual = 0
            if has_residual:
                residual = pl.getval(residuals, b)
            sub = pl.get_subblock_idx()
            visible0 = kl
            if mask == 3:
                visible0 = pl.min(
                    kl,
                    pl.max(
                        0,
                        (ratio * kl + residual - used + qlocal + sub * 3 + 0 + 1)
                        // ratio,
                    ),
                )
            if qlocal + sub * 3 + 0 >= used:
                visible0 = 0
            visible1 = kl
            if mask == 3:
                visible1 = pl.min(
                    kl,
                    pl.max(
                        0,
                        (ratio * kl + residual - used + qlocal + sub * 3 + 1 + 1)
                        // ratio,
                    ),
                )
            if qlocal + sub * 3 + 1 >= used:
                visible1 = 0
            visible2 = kl
            if mask == 3:
                visible2 = pl.min(
                    kl,
                    pl.max(
                        0,
                        (ratio * kl + residual - used + qlocal + sub * 3 + 2 + 1)
                        // ratio,
                    ),
                )
            if qlocal + sub * 3 + 2 >= used:
                visible2 = 0
            steps = pl.max(0, pl.min(shard_tiles, total_tiles - shard * shard_tiles))
            for t in pl.range(pl.min(qvalid, 1) * steps):
                kb = (shard * shard_tiles + t) * tile_n
                with pl.section_cube():
                    cube_step(
                        t,
                        qbase,
                        qvalid,
                        b,
                        kb,
                        kl,
                        q,
                        k,
                        qs,
                        ks,
                        table,
                        qg,
                        kg,
                        qsg,
                        ksg,
                        la,
                        lb,
                        sa,
                        sb,
                        cg,
                        cgv,
                        vg,
                        kp0,
                        kp1,
                        sp0,
                        sp1,
                    )
                with pl.section_vector():
                    vector_step(
                        t,
                        qbase,
                        qvalid,
                        qlocal,
                        b,
                        kb,
                        kl,
                        used,
                        residual,
                        weights,
                        out_idx,
                        out_val,
                        shard,
                        steps,
                        visible0,
                        visible1,
                        visible2,
                        wg,
                        wbg,
                        og,
                        vg,
                        hg,
                        hhg,
                        hlg,
                        bg,
                        ig,
                        ng,
                        tg,
                        xg,
                        oig,
                        ovg,
                    )
        with pl.section_cube():
            pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        with pl.section_vector():
            pl.system.bar_all()
            pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
            cache = og.current()
            sk = pl.reinterpret(cache, shape=[1, merge_width])
            val = pl.reinterpret(cache, dtype=BF, shape=[1, merge_width])
            shared = vg[0]
            idx = pl.reinterpret(shared, dtype=I32, shape=[1, merge_width])
            ht = hhg.current()
            lt = hlg.current()
            bt = bg.current()
            it = ig.current()
            nt = ng.current()
            tt = tg.current()
            xt = xg.current()
            oi = oig.current()
            ov = ovg.current()
            for row in pl.range(
                pl.get_block_idx(), rows, pl.get_block_num() * pl.get_subblock_num()
            ):
                pl.load(val, out_val, [row, 0])
                pl.load(idx, out_idx, [row, 0])
                keys_from_scores(val, sk, merge_width, merge_width)
                histograms_high_vf(ht, sk, merge_width // 256)
                find_high_bin_and_histograms_low(
                    lt, it, ht, sk, bt, nt, merge_width - topk + 1, merge_width // 256
                )
                find_kth_vf(tt, lt, it, bt, nt)
                collect_indices_gt_eq_vf(xt, sk, tt, 0, merge_width // 128)
                row_offset = 0
                if has_offset:
                    row_offset = pl.getval(offsets, row)
                emit_cached(
                    xt,
                    sk,
                    oi,
                    ov,
                    topk,
                    0,
                    0,
                    True,
                    idx,
                    False,
                    row_offset,
                    return_values,
                )
                pl.store(final_idx, oi, [row, 0])
                if return_values:
                    pl.store(final_val, ov, [row, 0])

    return kernel
