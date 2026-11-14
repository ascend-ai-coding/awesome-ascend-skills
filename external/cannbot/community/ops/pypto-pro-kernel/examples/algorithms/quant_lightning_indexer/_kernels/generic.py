# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：语法糖。MXFP4 QK with the QLI BF16 sequential head reduction.

Each block owns a six-query group and a contiguous key shard. Q/K carry
logical [M,128]/[N,128] FP4 values, scales [M,2,2]/[N,2,2]. Page gathers
happen on Cube. The two AIVs receive three queries each by directed BF16
FIX copies. At N=128 the two receiving slots use 49152 bytes/AIV, the
weight/output scratch less than 2 KiB; L1 < 64 KiB, L0A/B < 32 KiB,
L0C 196608 bytes. Every head FMA rounds to BF16, as the upstream evaluator
requires. This general path materializes scores. For <=16384 keys a drained
MIX barrier hands them to row-owned selection in the same native task.
Candidate blocks reuse those scores and the selection scratch. Larger
inputs use bounded hierarchical selection. All batch/query/key axes
are dynamic; only physical tile capacities and layout choices specialize.
"""

from functools import lru_cache

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

from .candidates import block_max, write_count
from .selection import (
    keys_from_scores,
    histograms_high_vf,
    find_high_bin_and_histograms_low,
    find_kth_vf,
    collect_indices_gt_eq_vf,
    emit_selection,
    emit_ordered,
)

F4 = pl.DT_FP4E2M1
E8 = pl.DT_FP8E8M0
BF = pl.DT_BF16
F32 = pl.DT_FP32
I32 = pl.DT_INT32


@lru_cache(None)
def make_score(
    page_size,
    k_page_stride,
    ks_page_stride,
    paged=True,
    mask=3,
    ratio=2,
    has_residual=True,
    tile_n=128,
    preload=0,
    k_stride=64,
    ks_stride=4,
    has_cuq=True,
    has_usedq=True,
    has_usedk=True,
    has_cuk=False,
    has_table=True,
    select_width=0,
    topk=512,
    has_offset=False,
    candidate_capacity=0,
):
    mq = 6
    mm = mq * 32
    half = mm // 2
    vec_n = max(128, tile_n)
    output_width = (max(topk, candidate_capacity) + 127) // 128 * 128
    block_width = (select_width + 2047) // 2048 * 256
    selection_capacity = max(256, select_width)

    @pl.vector_function
    def reduce_heads(src, weight, dst, row, valid_count):
        p32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=F32)
        p16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=BF)
        zero_bits = vf.full(0, p16, dtype=pl.DT_UINT16)
        zero = vf.bit_cast(zero_bits, dtype=BF)
        inf_bits = vf.full(0xFF80, p16, dtype=pl.DT_UINT16)
        neginf = vf.bit_cast(inf_bits, dtype=BF)
        for c in pl.range(vec_n // 128):
            acc = vf.full(zero, p16)
            for h in pl.range(32):
                v = vf.load_align(src, [(row * 32 + h), c * 128])
                v = vf.max(v, zero, p16)
                w32 = vf.load_align(weight, [row, h], dist=pl.LoadDist.BRC_B32)
                w16e = vf.astype(w32, p32, dtype=BF, layout=pl.CastLayout.ZERO)
                w16o = vf.astype(w32, p32, dtype=BF, layout=pl.CastLayout.ONE)
                w16 = vf.xor(w16e, w16o, p16)
                acc = vf.mul_add_dst(v, w16, p16)
            valid = vf.update_mask(
                pl.max(0, pl.min(128, valid_count - c * 128)), dtype=BF
            )
            value = vf.select(acc, neginf, valid)
            vf.store_align(dst, value, p16, [row, c * 128])

    @pl.pipeline.stage
    def cube_step(
        t,
        qbase,
        qvalid,
        b,
        kbase,
        kl,
        kstart,
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
        vg,
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
        # One page per step for the first interface probe (N=page_size=128).
        active = pl.min(tile_n, pl.max(0, kl - kbase))
        pl.set_validshape(kt, [128, active])
        pl.set_validshape(kst, [4, active])
        physical = kstart + kbase
        physical_s = kstart + kbase
        if paged:
            page = kbase // page_size
            if has_table:
                page = pl.getval(table, b * table.shape[1] + kbase // page_size)
            physical = page * k_page_stride + kbase % page_size
            physical_s = page * ks_page_stride + kbase % page_size
        vt = vg.next()
        if active > 0:
            pl.load(kt, k, [physical, 0], order=[1, 0])
            pl.load(kst, ks, [physical_s, 0, 0], order=[1, 0])
            ct = cg.next()
            pl.gemm_mx(ct, qt, kt, qst, kst, l0a=la, l0b=lb, scale_a=sa, scale_b=sb)
            pl.set_validshape(vt, [half, tile_n])
            pl.move(vt, ct, [0, 0], acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
            pl.move(vt, ct, [half, 0], acc_to_vec_mode=pl.AccToVecMode.SingleModeVec1)

    @pl.vector_function
    def fill_invalid(tile):
        mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
        bits = vf.full(0xFF80, mask, dtype=pl.DT_UINT16)
        values = vf.bit_cast(bits, dtype=BF)
        for i in pl.range(0, 3 * vec_n, 128):
            vf.store_align(tile, values, mask, i)

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
        scores,
        wg,
        og,
        vg,
    ):
        vt = vg.next()
        wt = wg.current()
        sub = pl.get_subblock_idx()
        if t == 0:
            pl.set_validshape(wt, [pl.max(0, pl.min(3, qvalid - sub * 3)), 32])
            if qvalid > sub * 3:
                pl.load(wt, weights, [qbase + sub * 3, 0])
        ot = og.next()
        if kbase < kl:
            for r in pl.range(pl.max(0, pl.min(3, qvalid - sub * 3))):
                visible = kl
                if mask == 3:
                    visible = pl.min(
                        kl,
                        pl.max(
                            0,
                            (ratio * kl + residual - used + qlocal + sub * 3 + r + 1)
                            // ratio,
                        ),
                    )
                if qlocal + sub * 3 + r >= used:
                    visible = 0
                reduce_heads(vt, wt, ot, r, pl.max(0, pl.min(tile_n, visible - kbase)))
        else:
            fill_invalid(ot)
        nr = pl.max(0, pl.min(3, qvalid - sub * 3))
        if nr > 0:
            pl.set_validshape(ot, [nr, pl.min(tile_n, scores.shape[1] - kbase)])
            pl.store(scores, ot, [qbase + sub * 3, kbase])

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
        cuk: pl.Tensor[[pl.DYNAMIC], I32],
        scores: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], BF],
        max_q: pl.DT_INT64,
        offsets: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], I32],
        final_idx: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], I32],
        final_val: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], BF],
        candidates: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], I32],
        lengths: pl.Tensor[[pl.DYNAMIC, 1], I32],
    ):
        rows = scores.shape[0]
        max_k = scores.shape[1]
        batches = 1
        if has_cuq:
            batches = cuq.shape[0] - 1
        qgroups = (max_q + mq - 1) // mq
        split = pl.min(7, (max_k + tile_n - 1) // tile_n)
        shard_tiles = (max_k + tile_n * split - 1) // (tile_n * split)
        grid = batches * qgroups * split
        key_view = pl.make_tensor(k, [k.shape[0], 128], [k_stride * 2, 1])
        scale_view = pl.make_tensor(ks, [ks.shape[0], 2, 2], [ks_stride, 2, 1])
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
        kg_type = pl.TileType(
            shape=[128, tile_n],
            dtype=F4,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZN,
        )
        kg = pl.make_tile_group(
            type=kg_type,
            addrs=64 * mm,
            mutex_ids="auto",
            depth=2,
        )
        qsg_type = pl.TileType(
            shape=[mm, 4],
            dtype=E8,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZZ,
        )
        qsg = pl.make_tile_group(
            type=qsg_type,
            addrs=64 * mm + 128 * tile_n,
            mutex_ids="auto",
            depth=1,
        )
        ksg_type = pl.TileType(
            shape=[4, tile_n],
            dtype=E8,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NN,
        )
        ksg = pl.make_tile_group(
            type=ksg_type,
            addrs=68 * mm + 128 * tile_n,
            mutex_ids="auto",
            depth=2,
        )
        la_type = pl.TileType(
            shape=[mm, 128],
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
            shape=[mm, 4],
            dtype=E8,
            target_memory=pl.MemorySpace.ScaleLeft,
            layout=pl.ZZ,
            compact=1,
        )
        sa = pl.make_tile_group(
            type=sa_type,
            addrs=[0, 64 * mm >> 4],
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
        cg_type = pl.TileType(
            shape=[mm, tile_n],
            dtype=F32,
            target_memory=pl.MemorySpace.Acc,
            compact=1,
        )
        cg = pl.make_tile_group(
            type=cg_type,
            addrs=0,
            mutex_ids="auto",
            depth=2,
        )
        vg_type = pl.TileType(
            shape=[half, vec_n],
            dtype=BF,
            target_memory=pl.MemorySpace.Vec,
        )
        vg = pl.make_tile_group(
            type=vg_type,
            addrs=0,
            mutex_ids="auto",
            depth=2,
            fwd_ids=[0, 1],
            bwd_ids=[2, 3],
        )
        wg_type = pl.TileType(
            shape=[3, 32],
            dtype=F32,
            target_memory=pl.MemorySpace.Vec,
        )
        wg = pl.make_tile_group(
            type=wg_type,
            addrs=4 * half * vec_n,
            mutex_ids="auto",
            depth=1,
        )
        og_type = pl.TileType(
            shape=[3, vec_n],
            dtype=BF,
            target_memory=pl.MemorySpace.Vec,
        )
        og = pl.make_tile_group(
            type=og_type,
            addrs=4 * half * vec_n + 384,
            mutex_ids="auto",
            depth=2,
        )
        ss_type = pl.TileType(
            shape=[1, selection_capacity],
            dtype=BF,
            target_memory=pl.MemorySpace.Vec,
        )
        ss = pl.make_tile_group(
            type=ss_type,
            addrs=4 * half * vec_n + 384 + 12 * vec_n,
            mutex_ids="auto",
            depth=1,
        )
        sk_type = pl.TileType(
            shape=[1, selection_capacity],
            dtype=pl.DT_UINT16,
            target_memory=pl.MemorySpace.Vec,
        )
        sk = pl.make_tile_group(
            type=sk_type,
            addrs=4 * half * vec_n + 384 + 12 * vec_n + 2 * selection_capacity,
            mutex_ids="auto",
            depth=1,
        )
        sh_type = pl.TileType(
            shape=[1, 256],
            dtype=pl.DT_UINT32,
            target_memory=pl.MemorySpace.Vec,
        )
        sh = pl.make_tile_group(
            type=sh_type,
            addrs=4 * half * vec_n + 384 + 12 * vec_n + 4 * selection_capacity,
            mutex_ids="auto",
            depth=1,
        )
        sl_type = pl.TileType(
            shape=[1, 256],
            dtype=pl.DT_UINT32,
            target_memory=pl.MemorySpace.Vec,
        )
        sl = pl.make_tile_group(
            type=sl_type,
            addrs=4 * half * vec_n + 1408 + 12 * vec_n + 4 * selection_capacity,
            mutex_ids="auto",
            depth=1,
        )
        sbuf_type = pl.TileType(
            shape=[1, 256],
            dtype=pl.DT_UINT32,
            target_memory=pl.MemorySpace.Vec,
        )
        sbuf = pl.make_tile_group(
            type=sbuf_type,
            addrs=4 * half * vec_n + 2432 + 12 * vec_n + 4 * selection_capacity,
            mutex_ids="auto",
            depth=1,
        )
        si_type = pl.TileType(
            shape=[1, 128],
            dtype=pl.DT_UINT32,
            target_memory=pl.MemorySpace.Vec,
        )
        si = pl.make_tile_group(
            type=si_type,
            addrs=4 * half * vec_n + 3456 + 12 * vec_n + 4 * selection_capacity,
            mutex_ids="auto",
            depth=1,
        )
        sn_type = pl.TileType(
            shape=[1, 128],
            dtype=pl.DT_UINT32,
            target_memory=pl.MemorySpace.Vec,
        )
        sn = pl.make_tile_group(
            type=sn_type,
            addrs=4 * half * vec_n + 3968 + 12 * vec_n + 4 * selection_capacity,
            mutex_ids="auto",
            depth=1,
        )
        st_type = pl.TileType(
            shape=[1, 128],
            dtype=pl.DT_UINT16,
            target_memory=pl.MemorySpace.Vec,
        )
        st = pl.make_tile_group(
            type=st_type,
            addrs=4 * half * vec_n + 4480 + 12 * vec_n + 4 * selection_capacity,
            mutex_ids="auto",
            depth=1,
        )
        sx_type = pl.TileType(
            shape=[1, selection_capacity + 128],
            dtype=pl.DT_UINT16,
            target_memory=pl.MemorySpace.Vec,
        )
        sx = pl.make_tile_group(
            type=sx_type,
            addrs=4 * half * vec_n + 4736 + 12 * vec_n + 4 * selection_capacity,
            mutex_ids="auto",
            depth=1,
        )
        so_type = pl.TileType(
            shape=[1, output_width],
            dtype=I32,
            target_memory=pl.MemorySpace.Vec,
        )
        so = pl.make_tile_group(
            type=so_type,
            addrs=4 * half * vec_n + 4992 + 12 * vec_n + 6 * selection_capacity,
            mutex_ids="auto",
            depth=1,
        )
        sv_type = pl.TileType(
            shape=[1, output_width],
            dtype=BF,
            target_memory=pl.MemorySpace.Vec,
        )
        sv = pl.make_tile_group(
            type=sv_type,
            addrs=4 * half * vec_n
            + 4992
            + 12 * vec_n
            + 6 * selection_capacity
            + 4 * output_width,
            mutex_ids="auto",
            depth=1,
        )
        core = pl.get_block_idx() // pl.get_subblock_num()
        for item in pl.range(core, grid, pl.get_block_num()):
            shard = item % split
            qgroup = item // split % qgroups
            b = item // (split * qgroups)
            qstart = 0
            qend = rows
            if has_cuq:
                qstart = pl.getval(cuq, b)
                qend = pl.getval(cuq, b + 1)
            qlocal = qgroup * mq
            qbase = qstart + qlocal
            qvalid = pl.min(mq, pl.max(0, qend - qbase))
            kl = max_k
            kstart = 0
            if has_cuk:
                kstart = pl.getval(cuk, b)
                kl = pl.getval(cuk, b + 1) - kstart
            if has_usedk:
                kl = pl.getval(usedk, b)
            used = qend - qstart
            if has_usedq:
                used = pl.getval(usedq, b)
            if used == 0:
                kl = 0
            residual = 0
            if has_residual:
                residual = pl.getval(residuals, b)
            for t in pl.range(
                pl.min(qvalid, 1)
                * pl.max(
                    0,
                    pl.min(
                        shard_tiles,
                        (max_k + tile_n - 1) // tile_n - shard * shard_tiles,
                    ),
                )
            ):
                kb = (shard * shard_tiles + t) * tile_n
                with pl.section_cube():
                    cube_step(
                        t,
                        qbase,
                        qvalid,
                        b,
                        kb,
                        kl,
                        kstart,
                        q,
                        key_view,
                        qs,
                        scale_view,
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
                        vg,
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
                        scores,
                        wg,
                        og,
                        vg,
                    )

        if select_width:
            with pl.section_cube():
                pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
            with pl.section_vector():
                pl.system.bar_all()
                pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
                for row in pl.range(
                    pl.get_block_idx(), rows, pl.get_block_num() * pl.get_subblock_num()
                ):
                    src = ss.current()
                    keys = sk.current()
                    high = sh.current()
                    low = sl.current()
                    buf = sbuf.current()
                    index = si.current()
                    nth = sn.current()
                    threshold = st.current()
                    selected = sx.current()
                    oi = so.current()
                    ov = sv.current()
                    pl.set_validshape(src, [1, max_k])
                    pl.load(src, scores, [row, 0])
                    offset = 0
                    if has_offset:
                        offset = pl.getval(offsets, row)
                    if max_k <= topk:
                        emit_ordered(src, oi, ov, topk, max_k, offset)
                    else:
                        keys_from_scores(src, keys, select_width, max_k)
                        histograms_high_vf(high, keys, select_width // 256)
                        find_high_bin_and_histograms_low(
                            low,
                            index,
                            high,
                            keys,
                            buf,
                            nth,
                            select_width - pl.min(topk, max_k) + 1,
                            select_width // 256,
                        )
                        find_kth_vf(threshold, low, index, buf, nth)
                        collect_indices_gt_eq_vf(
                            selected, keys, threshold, 0, select_width // 128
                        )
                        emit_selection(
                            selected,
                            src,
                            oi,
                            ov,
                            topk,
                            pl.min(topk, max_k),
                            0,
                            src,
                            False,
                            offset,
                        )
                    pl.set_validshape(oi, [1, topk])
                    pl.set_validshape(ov, [1, topk])
                    pl.store(final_idx, oi, [row, 0, 0])
                    pl.store(final_val, ov, [row, 0, 0])
                    if candidate_capacity:
                        qstart = 0
                        qend = rows
                        batch = 0
                        if has_cuq:
                            for b in pl.range(batches):
                                lo = pl.getval(cuq, b)
                                hi = pl.getval(cuq, b + 1)
                                if row >= lo and row < hi:
                                    batch = b
                                    qstart = lo
                                    qend = hi
                        uq = qend - qstart
                        if has_usedq:
                            uq = pl.getval(usedq, batch)
                        kl = max_k
                        if has_cuk:
                            kl = pl.getval(cuk, batch + 1) - pl.getval(cuk, batch)
                        if has_usedk:
                            kl = pl.getval(usedk, batch)
                        rem = 0
                        if has_residual:
                            rem = pl.getval(residuals, batch)
                        visible = kl
                        if mask == 3:
                            visible = pl.min(
                                kl,
                                pl.max(
                                    0,
                                    (ratio * kl + rem - uq + row - qstart + 1) // ratio,
                                ),
                            )
                        if row - qstart >= uq:
                            visible = 0
                        partial = -1
                        if visible % 8 != 0:
                            partial = visible // 8
                        count = pl.reinterpret(buf, dtype=I32, shape=[1, 8])
                        write_count(
                            count, pl.min(candidate_capacity, (visible + 7) // 8)
                        )
                        pl.set_validshape(count, [1, 1])
                        pl.store(lengths, count, [row, 0])
                        # Each 128-score chunk is fully loaded before its 16
                        # maxima overwrite an already-consumed prefix of src.
                        block_max(src, src, max_k, 0, partial)
                        blocks = (max_k + 7) // 8
                        if candidate_capacity >= blocks:
                            emit_ordered(
                                src, oi, ov, candidate_capacity, blocks, 0, False
                            )
                        else:
                            keys_from_scores(src, keys, block_width, blocks)
                            histograms_high_vf(high, keys, block_width // 256)
                            find_high_bin_and_histograms_low(
                                low,
                                index,
                                high,
                                keys,
                                buf,
                                nth,
                                block_width - pl.min(candidate_capacity, blocks) + 1,
                                block_width // 256,
                            )
                            find_kth_vf(threshold, low, index, buf, nth)
                            collect_indices_gt_eq_vf(
                                selected, keys, threshold, 0, block_width // 128
                            )
                            emit_selection(
                                selected,
                                src,
                                oi,
                                ov,
                                candidate_capacity,
                                candidate_capacity,
                                0,
                                src,
                                False,
                                0,
                                False,
                            )
                        pl.set_validshape(oi, [1, candidate_capacity])
                        pl.store(candidates, oi, [row, 0, 0])

    return kernel
