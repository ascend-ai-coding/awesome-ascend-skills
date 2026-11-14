# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ruff: noqa: F821
"""写法：混合。Native inverse RoPE and two-stage MXFP8 attention output projection.

Inputs: BF16 attn_out[T,64,512], E4M3FN woa[8,1024,4096] and wob[5120,8192],
E8M0 scales [8,1024,64,2] and [5120,128,2], and FP32 sin/cos[T,64].
All tensors are contiguous ND storage on one NPU; T is positive. Scale bytes
are reinterpreted as E8M0, never numerically converted. Inputs must keep every
stage finite; NaN/Inf propagation is outside this contract.

Inverse RoPE rotates only the last 64 head features with separate FP32 products
and additions before BF16 rounding. Eight heads form each first projection;
the intermediate rounds to BF16 before the second per-32 MXFP8 quantization.
The output is contiguous BF16 [T,5120] with independent storage on the current
stream. This forward excludes bias, residual, normalization and communication.

One MIX launch uses three capacity keys per hardware layout; T stays dynamic.
Workspace bytes are 42240 * ceil(T/16)*16, excluding output. At most eight
workspace sets are cached by device, stream and padded T; graph capture uses
its own buffers and needs a matching warmup. The caller orders different streams.
"""

from collections import OrderedDict
import threading

import torch
import torch_npu
import pypto
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf
from pypto_pro.runtime.platform import get_platform_info
from pypto_pro.runtime.tilingkey import TilingKeyField

FP8, E8 = pl.DT_FP8E4M3FN, pl.DT_FP8E8M0


Q_ROWS = 64
R_ROWS = 8


class EpilogueKey:
    """Three capacity profiles per hardware layout; T stays in the runtime ABI."""

    BM = TilingKeyField(bits=2, values=[64, 128, 256])
    BK = TilingKeyField(bits=1, values=[256, 512])
    VSlots = TilingKeyField(bits=1, values=[1, 2])
    K0 = TilingKeyField(bits=1, values=[128, 256])
    BN1 = TilingKeyField(bits=1, values=[160, 256])
    BN = TilingKeyField(bits=1, values=[160, 192])

    def is_valid(self, key):
        return key in (
            (64, 512, 2, 256, 160, 192),
            (128, 256, 2, 256, 160, 192),
            (256, 256, 1, 256, 160, 192),
            (64, 512, 2, 256, 256, 160),
            (128, 256, 2, 128, 256, 160),
            (256, 256, 1, 128, 256, 160),
        )


@pl.vector_function
def _inverse_rope(x, cos, sin):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    i = vf.arange(0, dtype=pl.DT_UINT32)
    one = vf.full(1, dtype=pl.DT_UINT32)
    swapped = vf.xor(i, one, p)
    parity = vf.and_(i, one, p)
    odd = vf.eq(parity, one, p)
    c = vf.load_align(cos, 0)
    s = vf.load_align(sin, 0)
    ns = vf.muls(s, -1.0, p)
    signed_s = vf.select(ns, s, odd)
    for h in pl.range(8):
        b = vf.load_align(x, h * 512 + 448, dist=pl.LoadDist.UNPK_B16)
        f = vf.astype(b, p, dtype=pl.DT_FP32)
        partner = vf.gather(f, swapped)
        a = vf.mul(f, c, p)
        v = vf.mul(partner, signed_s, p)
        r = vf.add(a, v, p)
        bf = vf.astype(r, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
        vf.store_align(x, bf, p, h * 512 + 448, dist=pl.StoreDist.PACK_B32)


@pl.vector_function
def _quant_scales(x, s, reciprocal, count: pl.DT_INT64):
    # Two deinterleaved b16 registers cover 256 values. A 16-lane block maximum
    # therefore covers 32 consecutive input values. Scale bytes use even b8
    # positions after narrowing; squeeze emits one byte for each 32-value group.
    p16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    p8 = vf.create_mask(pattern=pl.MaskPattern.VL16, dtype=pl.DT_UINT8)
    short16 = vf.create_mask(pattern=pl.MaskPattern.VL16, dtype=pl.DT_UINT16)
    em = vf.full(0x7F80, dtype=pl.DT_UINT16)
    eight = vf.full(8, dtype=pl.DT_UINT16)
    c262 = vf.full(262, dtype=pl.DT_UINT16)
    zero = vf.full(0, dtype=pl.DT_UINT16)
    idx = vf.arange(0, dtype=pl.DT_UINT8)
    one = vf.full(1, dtype=pl.DT_UINT8)
    parity = vf.and_(idx, one, p8)
    choose = vf.eq(parity, 0, p8)
    vf.clear_spr()
    ar = vf.unalign_reg_for_store()
    for chunk in pl.range(count // 256):
        x0, x1 = vf.load_align(x, chunk * 256, dist=pl.LoadDist.DINTLV_B16)
        bits0 = vf.bit_cast(x0, dtype=pl.DT_UINT16)
        bits1 = vf.bit_cast(x1, dtype=pl.DT_UINT16)
        a0 = vf.and_(bits0, em, p16)
        a1 = vf.and_(bits1, em, p16)
        both = vf.max(a0, a1, p16)
        gm = vf.reduce_max(both, p16, datablock=True)
        exponent = vf.shift_right(gm, 7, p16)
        e = vf.maxs(exponent, 8, p16)
        sb = vf.sub(e, eight, p16)
        sb8 = vf.astype(sb, p16, dtype=pl.DT_UINT8)
        dense = vf.squeeze(sb8, choose, gather_mode=pl.SqueezeMode.STORE_REG)
        vf.squeeze_store_unalign(s, dense, ar)
        reverse = vf.sub(c262, e, p16)
        rev = vf.shift_left(reverse, 7, p16)
        tiny = vf.eq(e, eight, p16)
        rev = vf.select(zero, rev, tiny)
        dup, unused = vf.interleave(rev, rev)
        vf.store_align(reciprocal, dup, short16, chunk * 16)
    vf.squeeze_store_unalign_post(s, ar)


@pl.vector_function
def _quant_values(x, q, reciprocal, even, odd, count: pl.DT_INT64):
    p16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    p32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p8 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT8)
    for chunk in pl.range(count // 256):
        for half in pl.range(2):
            off = chunk * 256 + half * 128
            xv = vf.load_align(x, off)
            rs = vf.load_align(
                reciprocal, chunk * 16 + half * 8, dist=pl.LoadDist.E2B_B16
            )
            reciprocal_bf = vf.bit_cast(rs, dtype=pl.DT_BF16)
            y = vf.mul(xv, reciprocal_bf, p16)
            yz = vf.astype(y, p16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
            yo = vf.astype(y, p16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
            # Explicit finite E4M3 saturation also covers values rounding past 448.
            yz = vf.maxs(yz, -448.0, p32)
            yz = vf.mins(yz, 448.0, p32)
            yo = vf.maxs(yo, -448.0, p32)
            yo = vf.mins(yo, 448.0, p32)
            qz = vf.astype(yz, p32, dtype=FP8, saturate=pl.SaturateMode.ON)
            qo = vf.astype(yo, p32, dtype=FP8, saturate=pl.SaturateMode.ON)
            vf.store_align(
                even, qz, p32, chunk * 128 + half * 64, dist=pl.StoreDist.PACK4_B32
            )
            vf.store_align(
                odd, qo, p32, chunk * 128 + half * 64, dist=pl.StoreDist.PACK4_B32
            )
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    for chunk in pl.range(count // 256):
        for half in pl.range(2):
            a = vf.load_align(even, chunk * 128 + half * 64)
            b = vf.load_align(odd, chunk * 128 + half * 64)
            vf.store_align(
                q + chunk * 256 + half * 128, a, b, p8, dist=pl.StoreDist.INTLV_B8
            )


@pl.vector_function
def _zero_quant(q, s, count: pl.DT_INT64):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT8)
    zero = vf.full(0, dtype=pl.DT_UINT8)
    for i in pl.range(count // 256):
        vf.store_align(q, zero, p, i * 256)
    ps = vf.update_mask(count // 32, dtype=pl.DT_UINT8)
    vf.store_align(s, zero, ps, 0)


@pl.vector_function
def _round_y(src, bf, count: pl.DT_INT64, offset: pl.DT_INT64):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for i in pl.range(count // 64):
        f = vf.load_align(src, offset + i * 64)
        b = vf.astype(f, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
        vf.store_align(bf, b, p, i * 64, dist=pl.StoreDist.PACK_B32)


@pl.vector_function
def _pad_scales(src, dst, rows: pl.DT_INT64, cols: pl.DT_INT64):
    i = vf.arange(0, dtype=pl.DT_UINT8)
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT8)
    valid = vf.update_mask(cols, dtype=pl.DT_UINT8)
    for r in pl.range(rows):
        v = vf.load_align(src, (r * cols // 32) * 32)
        index = vf.adds(i, r * cols % 32, p)
        packed = vf.gather(v, index)
        vf.store_align(dst, packed, valid, r * 32)


@pl.vector_function
def _inverse_rope_slab(x, cos, sin, rows: pl.DT_INT64):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    i = vf.arange(0, dtype=pl.DT_UINT32)
    one = vf.full(1, dtype=pl.DT_UINT32)
    swapped = vf.xor(i, one, p)
    parity = vf.and_(i, one, p)
    odd = vf.eq(parity, one, p)
    for row in pl.range(rows):
        c = vf.load_align(cos, row * 64)
        s = vf.load_align(sin, row * 64)
        ns = vf.muls(s, -1.0, p)
        signed_s = vf.select(ns, s, odd)
        for h in pl.range(4):
            b = vf.load_align(x, row * 2048 + h * 512 + 448, dist=pl.LoadDist.UNPK_B16)
            f = vf.astype(b, p, dtype=pl.DT_FP32)
            partner = vf.gather(f, swapped)
            a = vf.mul(f, c, p)
            v = vf.mul(partner, signed_s, p)
            r = vf.add(a, v, p)
            bf = vf.astype(r, p, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT)
            vf.store_align(
                x, bf, p, row * 2048 + h * 512 + 448, dist=pl.StoreDist.PACK_B32
            )


@pl.vector_function
def _zero_quant_tail(q, s, offset: pl.DT_INT64, count: pl.DT_INT64):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT8)
    zero = vf.full(0, dtype=pl.DT_UINT8)
    for i in pl.range(count // 256):
        vf.store_align(q, zero, p, offset + i * 256)
    ps = vf.update_mask(count // 32, dtype=pl.DT_UINT8)
    vf.store_align(s, zero, ps, offset // 32)


@pl.vector_function
def _zero_quant_batch(q, s, count: pl.DT_INT64):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT8)
    zero = vf.full(0, dtype=pl.DT_UINT8)
    for i in pl.range(count // 256):
        vf.store_align(q, zero, p, i * 256)
    for block in pl.range(count // 8192):
        vf.store_align(s, zero, p, block * 256)
    remainder = (count // 32) % 256
    if remainder > 0:
        ps = vf.update_mask(remainder, dtype=pl.DT_UINT8)
        vf.store_align(s, zero, ps, (count // 8192) * 256)


@pl.vector_function
def _zero_quant_tail_batch(q, s, offset: pl.DT_INT64, count: pl.DT_INT64):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT8)
    zero = vf.full(0, dtype=pl.DT_UINT8)
    for i in pl.range(count // 256):
        vf.store_align(q, zero, p, offset + i * 256)
    for block in pl.range(count // 8192):
        vf.store_align(s, zero, p, offset // 32 + block * 256)
    remainder = (count // 32) % 256
    if remainder > 0:
        ps = vf.update_mask(remainder, dtype=pl.DT_UINT8)
        vf.store_align(s, zero, ps, offset // 32 + (count // 8192) * 256)


@pl.pipeline.stage
def _project_a(
    work,
    t,
    padded,
    a,
    sa,
    b,
    sb,
    wb,
    wbg,
    ag,
    bg,
    b_transposed,
    asg,
    bsg,
    al,
    als,
    bl,
    bls,
    accg,
    shared,
):
    nm = (t + BM - 1) // BM
    nn = (1024 + BN1 - 1) // BN1
    if BN1 == 256:
        group = (work // nn) % 8
        mi = work // (8 * nn) * BM
    else:
        group = work // (nm * nn)
        mi = work // nn % nm * BM
    ni = work % nn * BN1
    mv = pl.min(BM, padded - mi)
    nv = pl.min(BN1, 1024 - ni)
    acc = accg.next()
    ast = asg.current()
    bst = bsg.current()
    pl.set_validshape(acc, [mv, nv])
    pl.set_validshape(ast, [mv, 64 if BN1 == 256 else 128])
    pl.set_validshape(ag, [mv, BK])
    pl.set_validshape(al, [mv, K0])
    pl.set_validshape(als, [mv, K0 // 32])
    pl.set_validshape(bg, [nv, BK])
    pl.set_validshape(b_transposed, [BK, nv])
    pl.set_validshape(bst, [128, nv])
    pl.set_validshape(bl, [K0, nv])
    pl.set_validshape(bls, [K0 // 32, nv])
    if BN1 != 256:
        pl.load(ast, sa, [group * padded + mi, 0, 0])
    if work != pl.get_block_idx():
        pl.load(bst, sb, [group * 1024 + ni, 0, 0], order=[1, 0])
    for ki in pl.range(0, 4096, BK):
        if BN1 == 256 and ki % 2048 == 0:
            # All 32 AICs have a first tile; later tiles reuse immutable slabs.
            if work == pl.get_block_idx():
                if ki == 0:
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=3)
                else:
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=4)
            pl.load(ast, sa, [group * padded + mi, ki // 64, 0])
        at = ag.next()
        bt = bg[(ki // BK) % 2]
        pl.load(at, a, [group * padded + mi, ki])
        if work != pl.get_block_idx() or ki >= (2 if BN1 == 256 else 1) * BK:
            pl.load(bt, b, [group * 1024 + ni, ki])
        tr = b_transposed[(ki // BK) % 2]
        _mx_block(
            acc,
            at,
            tr,
            ast,
            bst,
            al,
            als,
            bl,
            bls,
            ki,
            4096,
            ki % 2048 if BN1 == 256 else ki,
        )
    target = shared.next()
    pl.set_validshape(target, [mv // 2, nv])
    pl.move(
        target,
        acc,
        acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM,
        phase=pl.STPhase.Final,
    )
    if work + pl.get_block_num() >= 8 * nm * nn:
        second_nn = (5120 + BN - 1) // BN
        core = pl.get_block_idx()
        if core < nm * second_nn:
            first = wbg[0]
            col = core % second_nn * BN
            pl.set_validshape(first, [pl.min(BN, 5120 - col), BK])
            pl.load(first, wb, [col, 0])


def _prefetch_a_weights(b, sb, bg, bsg, t):
    work = pl.get_block_idx()
    nm = (t + BM - 1) // BM
    nn = (1024 + BN1 - 1) // BN1
    if work < 8 * nm * nn:
        if BN1 == 256:
            group = (work // nn) % 8
        else:
            group = work // (nm * nn)
        col = work % nn * BN1
        nv = pl.min(BN1, 1024 - col)
        first = bg[0]
        second = bg[1]
        scales = bsg.current()
        pl.set_validshape(first, [nv, BK])
        pl.set_validshape(second, [nv, BK])
        pl.set_validshape(scales, [128, nv])
        pl.load(first, b, [group * 1024 + col, 0])
        if BN1 == 256:
            pl.load(second, b, [group * 1024 + col, BK])
        pl.load(scales, sb, [group * 1024 + col, 0, 0], order=[1, 0])


@pl.pipeline.stage
def _quant_y(work, t, padded, yq, ys, shared, bfg, qg, esg, rg, eg, og, spg):
    nm = (t + BM - 1) // BM
    nn = (1024 + BN1 - 1) // BN1
    if BN1 == 256:
        group = (work // nn) % 8
        mi = work // (8 * nn) * BM
    else:
        group = work // (nm * nn)
        mi = work // nn % nm * BM
    ni = work % nn * BN1
    nv = pl.min(BN1, 1024 - ni)
    half = pl.min(BM, padded - mi) // 2
    row = mi + pl.get_subblock_idx() * half
    tile = shared.next()
    bf = bfg.current()
    q = qg.current()
    es = esg.current()
    reciprocal = rg.current()
    even = eg.current()
    odd = og.current()
    scales = spg.current()
    if BN1 == 160 or BM <= 128:
        _round_y(tile, bf, half * BN1, 0)
        _quant_scales(bf, es, reciprocal, half * BN1)
        _quant_values(bf, q, reciprocal, even, odd, half * BN1)
        _pad_scales(es, scales, half, BN1 // 32)
        pl.set_validshape(q, [half, nv])
        pl.set_validshape(scales, [half, nv // 32])
        pl.store(yq, q, [row, group * 1024 + ni])
        pl.store(ys, scales, [row, group * 32 + ni // 32])
    else:
        for start in pl.range(0, half, Q_ROWS):
            rows = pl.min(Q_ROWS, half - start)
            _round_y(tile, bf, rows * BN1, start * BN1)
            _quant_scales(bf, es, reciprocal, rows * BN1)
            _quant_values(bf, q, reciprocal, even, odd, rows * BN1)
            _pad_scales(es, scales, rows, BN1 // 32)
            pl.set_validshape(q, [rows, nv])
            pl.set_validshape(scales, [rows, nv // 32])
            pl.store(yq, q, [row + start, group * 1024 + ni])
            pl.store(ys, scales, [row + start, group * 32 + ni // 32])


@pl.jit(
    arch="a5",
    auto_mutex=True,
    tiling_key=EpilogueKey,
    compile_timeout=300,
    pipeline=pl.pipeline.PipelineConfig(preload=0),
)
def attn_epilogue_kernel(
    o_ptr: pl.Ptr[pl.DT_UINT8],
    cos_ptr: pl.Ptr[pl.DT_UINT8],
    sin_ptr: pl.Ptr[pl.DT_UINT8],
    wa_ptr: pl.Ptr[pl.DT_UINT8],
    sa_ptr: pl.Ptr[pl.DT_UINT8],
    wb_ptr: pl.Ptr[pl.DT_UINT8],
    sb_ptr: pl.Ptr[pl.DT_UINT8],
    aq_ptr: pl.Ptr[pl.DT_UINT8],
    asc_ptr: pl.Ptr[pl.DT_UINT8],
    yq_ptr: pl.Ptr[pl.DT_UINT8],
    ys_ptr: pl.Ptr[pl.DT_UINT8],
    out_ptr: pl.Ptr[pl.DT_UINT8],
    t: pl.DT_INT64,
    padded: pl.DT_INT64,
):
    # AIVs quantize activation rows; AICs own disjoint grouped output tiles.
    # A tile's two AIVs quantize disjoint halves of its effective padded M extent.
    # FP32 products/additions round to BF16 before each per-32 MX quantization.
    # For BF16 exponent e, scale=max(e-8,0); e<=8 produces a zero reciprocal.
    # Otherwise power-of-two scaling has magnitude <512 before clipping to 448.
    # Inputs/tables/weights and FP32 intermediates must be finite; sin/cos are in
    # [-1,1]. There is no floating division. E8M0 scales are decoded by mad_mx.
    # Quantization uses at most 64 rows per chunk for the 32-core layout.
    # L0A/B use (256//K0) panels, each BM/max(BN1,BN)*K0 bytes. Each
    # accumulator uses 4*BM*N bytes with the largest fitting depth, at most two.
    # The first weight tile of GEMM2 reuses L1 address 2*BM*BK, after GEMM1's
    # final reads. The inter-phase barrier drains it before new mutex IDs apply.
    # On 32 cores, AQ is published in two immutable 2048-column slabs. Events
    # 5/6 globally order AIV stores before events 3/4 notify each paired AIC.
    # Each AIC consumes both once on MTE2. C->V uses automatic events 0/1.
    # Event 8 orders all YQ stores before event 7 releases each AIC on MTE2.
    # Cube bar_all drains its first projection before local address reuse.
    if BN1 == 256:
        _rope_quant_slabs(o_ptr, cos_ptr, sin_ptr, aq_ptr, asc_ptr, t, padded)
    else:
        _rope_quant_full(o_ptr, cos_ptr, sin_ptr, aq_ptr, asc_ptr, t, padded)
    a = pl.make_tensor(aq_ptr, [8 * padded, 4096], dtype=FP8)
    sa = pl.make_tensor(asc_ptr, [8 * padded, 64, 2], dtype=E8)
    b = pl.make_tensor(wa_ptr, [8192, 4096], dtype=FP8)
    sb = pl.make_tensor(sa_ptr, [8192, 64, 2], dtype=E8)
    wb = pl.make_tensor(wb_ptr, [5120, 8192], dtype=FP8)
    yq = pl.make_tensor(yq_ptr, [padded, 8192], dtype=pl.DT_UINT8)
    ys = pl.make_tensor(ys_ptr, [padded, 256], dtype=pl.DT_UINT8)
    # Byte offsets preserve the L1 prefetch alias and the C->V buffer lifetime.
    # Vec mutex IDs 0..8 belong to inverse RoPE; these groups start at ID 9.
    l0_slots = 256 // K0
    acc_id = 6 + 4 * l0_slots
    acc_slots = min(2, 65536 // (BM * BN1))
    q_rows = Q_ROWS if BN1 == 256 else BM // 2
    scale_cols = 64 if BN1 == 256 else 128
    vec_bf = 32768 + 2 * BM * BN1 * VSlots
    vec_q = vec_bf + 2 * q_rows * BN1
    vec_es = vec_q + (q_rows + (512 + BN1 - 1) // BN1) * BN1
    vec_recip = vec_es + ((q_rows * BN1 // 32 + 31) // 32) * 32 + 256
    vec_even = vec_recip + q_rows * BN1 // 8
    vec_odd = vec_even + q_rows * BN1 // 2 + 256
    vec_scale = vec_odd + q_rows * BN1 // 2 + 256
    ag = pl.make_tile_group(
        type=pl.TileType(shape=[BM, BK], dtype=FP8, target_memory=pl.MemorySpace.Mat),
        addrs=[0, BM * BK],
        mutex_ids=[0, 1],
    )
    bg = pl.make_tile_group(
        type=pl.TileType(
            shape=[max(BN1, BN), BK], dtype=FP8, target_memory=pl.MemorySpace.Mat
        ),
        addrs=[2 * BM * BK, (2 * BM + max(BN1, BN)) * BK],
        mutex_ids=[2, 3],
    )
    asg = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, scale_cols],
            dtype=E8,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZZ,
        ),
        addrs=[2 * (BM + max(BN1, BN)) * BK],
        mutex_ids=[4],
    )
    bsg = pl.make_tile_group(
        type=pl.TileType(
            shape=[128, BN1], dtype=E8, target_memory=pl.MemorySpace.Mat, layout=pl.NN
        ),
        addrs=[2 * (BM + max(BN1, BN)) * BK + BM * scale_cols],
        mutex_ids=[5],
    )
    al = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, K0], dtype=FP8, target_memory=pl.MemorySpace.Left, compact=1
        ),
        addrs=[0, BM * K0][:l0_slots],
        mutex_ids=[6, 7][:l0_slots],
    )
    bl = pl.make_tile_group(
        type=pl.TileType(
            shape=[K0, BN1], dtype=FP8, target_memory=pl.MemorySpace.Right, compact=1
        ),
        addrs=[0, BN1 * K0][:l0_slots],
        mutex_ids=[6 + l0_slots, 7 + l0_slots][:l0_slots],
    )
    als = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, K0 // 32],
            dtype=E8,
            target_memory=pl.MemorySpace.ScaleLeft,
            compact=1,
        ),
        addrs=[0, BM * K0 // 16][:l0_slots],
        mutex_ids=[6 + 2 * l0_slots, 7 + 2 * l0_slots][:l0_slots],
    )
    bls = pl.make_tile_group(
        type=pl.TileType(
            shape=[K0 // 32, BN1],
            dtype=E8,
            target_memory=pl.MemorySpace.ScaleRight,
            compact=1,
        ),
        addrs=[0, BN1 * K0 // 16][:l0_slots],
        mutex_ids=[6 + 3 * l0_slots, 7 + 3 * l0_slots][:l0_slots],
    )
    accg = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, BN1],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            compact=1,
        ),
        addrs=[0, 4 * BM * BN1][:acc_slots],
        mutex_ids=[acc_id, acc_id + 1][:acc_slots],
    )
    # Cross-core IDs are literal lists selected by the compile-time slot count.
    if VSlots == 2:
        shared = pl.make_tile_group(
            type=pl.TileType(
                shape=[BM // 2, BN1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[32768, 32768 + 2 * BM * BN1],
            mutex_ids=[9, 10],
            fwd_ids=[0, 1],
            bwd_ids=[0, 1],
        )
    else:
        shared = pl.make_tile_group(
            type=pl.TileType(
                shape=[BM // 2, BN1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[32768],
            mutex_ids=[9],
            fwd_ids=[0],
            bwd_ids=[0],
        )
    bfg = pl.make_tile_group(
        type=pl.TileType(
            shape=[q_rows, BN1], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[vec_bf],
        mutex_ids=[9 + VSlots],
    )
    qg = pl.make_tile_group(
        type=pl.TileType(
            shape=[q_rows + (512 + BN1 - 1) // BN1, BN1],
            dtype=pl.DT_UINT8,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=[vec_q],
        mutex_ids=[10 + VSlots],
    )
    esg = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, (q_rows * BN1 // 32 + 31) // 32 * 32 + 256],
            dtype=pl.DT_UINT8,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=[vec_es],
        mutex_ids=[11 + VSlots],
    )
    rg = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, q_rows * BN1 // 16],
            dtype=pl.DT_UINT16,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=[vec_recip],
        mutex_ids=[12 + VSlots],
    )
    eg = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, q_rows * BN1 // 2 + 256],
            dtype=FP8,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=[vec_even],
        mutex_ids=[13 + VSlots],
    )
    og = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, q_rows * BN1 // 2 + 256],
            dtype=FP8,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=[vec_odd],
        mutex_ids=[14 + VSlots],
    )
    spg = pl.make_tile_group(
        type=pl.TileType(
            shape=[q_rows, 32], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[vec_scale],
        mutex_ids=[15 + VSlots],
    )
    b_transposed = pl.reinterpret(bg, shape=[BK, max(BN1, BN)], layout=pl.ZN)
    # Prefetched data must use GEMM2's physical NZ row pitch.
    wb_prefetch = pl.reinterpret(bg, shape=[BN, BK])
    with pl.section_cube():
        _prefetch_a_weights(b, sb, bg, bsg, t)
        pl.system.bar_all()
        if BN1 != 256:
            pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
    if BN1 != 256:
        with pl.section_vector():
            pl.system.bar_all()
            pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
    nm = (t + BM - 1) // BM
    for work in pl.range(
        pl.get_block_idx() // pl.get_subblock_num(),
        8 * nm * ((1024 + BN1 - 1) // BN1),
        pl.get_block_num(),
    ):
        with pl.section_cube():
            _project_a(
                work,
                t,
                padded,
                a,
                sa,
                b,
                sb,
                wb,
                wb_prefetch,
                ag,
                bg,
                b_transposed,
                asg,
                bsg,
                al,
                als,
                bl,
                bls,
                accg,
                shared,
            )
        with pl.section_vector():
            _quant_y(work, t, padded, yq, ys, shared, bfg, qg, esg, rg, eg, og, spg)
    with pl.section_cube():
        pl.system.bar_all()
        if BN1 == 256:
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=7)
        else:
            pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
    with pl.section_vector():
        if BN1 == 256:
            pl.system.set_cross_core(
                pipe=pl.PipeType.MTE3,
                event_id=8,
                sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
            )
            pl.system.wait_cross_core(
                pipe=pl.PipeType.MTE3,
                event_id=8,
                sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
            )
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=7)
        else:
            pl.system.bar_all()
            pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
    _project_b(yq_ptr, ys_ptr, wb_ptr, sb_ptr, out_ptr, t, padded)


def _rope_quant_slabs(o_ptr, cos_ptr, sin_ptr, aq_ptr, asc_ptr, t, padded):
    r_rows = R_ROWS if BM == 256 else 4
    source = pl.make_tensor(o_ptr, [t, 32768], dtype=pl.DT_BF16)
    cosine = pl.make_tensor(cos_ptr, [t, 64], dtype=pl.DT_FP32)
    sine = pl.make_tensor(sin_ptr, [t, 64], dtype=pl.DT_FP32)
    aq = pl.make_tensor(aq_ptr, [8 * padded, 4096], dtype=pl.DT_UINT8)
    asc = pl.make_tensor(asc_ptr, [8 * padded, 128], dtype=pl.DT_UINT8)
    # Rope storage uses 13120*r_rows+2560 bytes. Its stores finish before
    # the second slab notification permits Cube to write the shared UB view.
    cos_addr = 8192 * r_rows
    sin_addr = cos_addr + 256 * r_rows
    q_addr = sin_addr + 256 * r_rows
    es_addr = q_addr + (r_rows + 1) * 2048
    recip_addr = es_addr + r_rows * 64
    even_addr = recip_addr + r_rows * 256
    odd_addr = even_addr + r_rows * 1024 + 256
    xg = pl.make_tile_group(
        type=pl.TileType(
            shape=[r_rows, 2048], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[0, r_rows * 4096],
        mutex_ids=[0, 1],
    )
    cg = pl.make_tile_group(
        type=pl.TileType(
            shape=[r_rows, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[cos_addr],
        mutex_ids=[2],
    )
    sg = pl.make_tile_group(
        type=pl.TileType(
            shape=[r_rows, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[sin_addr],
        mutex_ids=[3],
    )
    qg = pl.make_tile_group(
        type=pl.TileType(
            shape=[r_rows + 1, 2048],
            dtype=pl.DT_UINT8,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=[q_addr],
        mutex_ids=[4],
    )
    esg = pl.make_tile_group(
        type=pl.TileType(
            shape=[r_rows, 64], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[es_addr],
        mutex_ids=[5],
    )
    rg = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, r_rows * 128],
            dtype=pl.DT_UINT16,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=[recip_addr],
        mutex_ids=[6],
    )
    eg = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, r_rows * 1024 + 256], dtype=FP8, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[even_addr],
        mutex_ids=[7],
    )
    og = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, r_rows * 1024 + 256], dtype=FP8, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[odd_addr],
        mutex_ids=[8],
    )
    with pl.section_vector():
        worker = pl.get_block_idx()
        group = worker % 8
        lane = worker // 8
        lanes = pl.get_block_num() * pl.get_subblock_num() // 8
        batch_rows = r_rows
        if BM == 256:
            rows_per_lane = (padded + lanes - 1) // lanes
            batches = (rows_per_lane + r_rows - 1) // r_rows
            batch_rows = (rows_per_lane + batches - 1) // batches
        for slab in pl.range(2):
            for row in pl.range(lane * batch_rows, padded, lanes * batch_rows):
                rows = pl.min(batch_rows, padded - row)
                x = xg.next()
                q = qg.current()
                es = esg.current()
                reciprocal = rg.current()
                even = eg.current()
                odd = og.current()
                if row < t:
                    valid = pl.min(rows, t - row)
                    c = cg.current()
                    s = sg.current()
                    pl.set_validshape(x, [valid, 2048])
                    pl.set_validshape(c, [valid, 64])
                    pl.set_validshape(s, [valid, 64])
                    pl.load(x, source, [row, group * 4096 + slab * 2048])
                    pl.load(c, cosine, [row, 0])
                    pl.load(s, sine, [row, 0])
                    _inverse_rope_slab(x, c, s, valid)
                    _quant_scales(x, es, reciprocal, valid * 2048)
                    _quant_values(x, q, reciprocal, even, odd, valid * 2048)
                    if valid < rows:
                        if BM == 256:
                            _zero_quant_tail_batch(
                                q, es, valid * 2048, (rows - valid) * 2048
                            )
                        else:
                            _zero_quant_tail(q, es, valid * 2048, (rows - valid) * 2048)
                else:
                    if BM == 256:
                        _zero_quant_batch(q, es, rows * 2048)
                    else:
                        _zero_quant(q, es, rows * 2048)
                pl.set_validshape(q, [rows, 2048])
                pl.set_validshape(es, [rows, 64])
                pl.store(aq, q, [group * padded + row, slab * 2048])
                pl.store(asc, es, [group * padded + row, slab * 64])
            # Each immutable slab is globally complete before both AIV subcores
            # publish it to their AIC. MTE3 orders publication after all stores.
            if slab == 0:
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE3,
                    event_id=5,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.MTE3,
                    event_id=5,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=3)
            else:
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE3,
                    event_id=6,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.MTE3,
                    event_id=6,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=4)


def _rope_quant_full(o_ptr, cos_ptr, sin_ptr, aq_ptr, asc_ptr, t, padded):
    source = pl.make_tensor(o_ptr, [t * 8, 4096], dtype=pl.DT_BF16)
    cosine = pl.make_tensor(cos_ptr, [t, 64], dtype=pl.DT_FP32)
    sine = pl.make_tensor(sin_ptr, [t, 64], dtype=pl.DT_FP32)
    aq = pl.make_tensor(aq_ptr, [8 * padded, 4096], dtype=pl.DT_UINT8)
    asc = pl.make_tensor(asc_ptr, [8 * padded, 128], dtype=pl.DT_UINT8)
    # The activation workspace occupies Vec bytes [0, 26752).
    xg = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 4096], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[0, 8192],
        mutex_ids=[0, 1],
    )
    cg = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[16384],
        mutex_ids=[2],
    )
    sg = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[16640],
        mutex_ids=[3],
    )
    qg = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 4608], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[16896],
        mutex_ids=[4],
    )
    esg = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 128], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[21504],
        mutex_ids=[5],
    )
    rg = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 256], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[21632],
        mutex_ids=[6],
    )
    eg = pl.make_tile_group(
        type=pl.TileType(shape=[1, 2304], dtype=FP8, target_memory=pl.MemorySpace.Vec),
        addrs=[22144],
        mutex_ids=[7],
    )
    og = pl.make_tile_group(
        type=pl.TileType(shape=[1, 2304], dtype=FP8, target_memory=pl.MemorySpace.Vec),
        addrs=[24448],
        mutex_ids=[8],
    )
    with pl.section_vector():
        for r in pl.range(
            pl.get_block_idx(), padded * 8, pl.get_block_num() * pl.get_subblock_num()
        ):
            x = xg.next()
            q = qg.current()
            es = esg.current()
            reciprocal = rg.current()
            even = eg.current()
            odd = og.current()
            if r // 8 < t:
                c = cg.current()
                s = sg.current()
                pl.load(x, source, [r, 0])
                pl.load(c, cosine, [r // 8, 0])
                pl.load(s, sine, [r // 8, 0])
                _inverse_rope(x, c, s)
                _quant_scales(x, es, reciprocal, 4096)
                _quant_values(x, q, reciprocal, even, odd, 4096)
            else:
                _zero_quant(q, es, 4096)
            pl.set_validshape(q, [1, 4096])
            dest = (r % 8) * padded + r // 8
            pl.store(aq, q, [dest, 0])
            pl.store(asc, es, [dest, 0])


def _project_b(yq_ptr, ys_ptr, wb_ptr, sb_ptr, out_ptr, t, padded):
    k = 8192
    n = 5120
    a = pl.make_tensor(yq_ptr, [padded, k], dtype=FP8)
    sa = pl.make_tensor(ys_ptr, [padded, k // 64, 2], dtype=E8)
    b = pl.make_tensor(wb_ptr, [n, k], dtype=FP8)
    sb = pl.make_tensor(sb_ptr, [n, k // 64, 2], dtype=E8)
    output = pl.make_tensor(out_ptr, [padded, n], dtype=pl.DT_BF16)
    # The Cube pipe barrier drains the first projection before address reuse.
    # Continue its Cube mutex numbering; L0 scale planes follow data address >> 4.
    l0_slots = 256 // K0
    cube_id = 6 + 4 * l0_slots + min(2, 65536 // (BM * BN1))
    acc_slots = min(2, 65536 // (BM * BN))
    ag = pl.make_tile_group(
        type=pl.TileType(shape=[BM, BK], dtype=FP8, target_memory=pl.MemorySpace.Mat),
        addrs=[0, BM * BK],
        mutex_ids=[cube_id, cube_id + 1],
    )
    bg = pl.make_tile_group(
        type=pl.TileType(shape=[BN, BK], dtype=FP8, target_memory=pl.MemorySpace.Mat),
        addrs=[2 * BM * BK, (2 * BM + BN) * BK],
        mutex_ids=[cube_id + 2, cube_id + 3],
    )
    asg = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, k // 32],
            dtype=E8,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZZ,
        ),
        addrs=[2 * (BM + BN) * BK],
        mutex_ids=[cube_id + 4],
    )
    bsg = pl.make_tile_group(
        type=pl.TileType(
            shape=[k // 32, BN],
            dtype=E8,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NN,
        ),
        addrs=[2 * (BM + BN) * BK + BM * (k // 32)],
        mutex_ids=[cube_id + 5],
    )
    al = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, K0], dtype=FP8, target_memory=pl.MemorySpace.Left, compact=1
        ),
        addrs=[0, BM * K0][:l0_slots],
        mutex_ids=[cube_id + 6, cube_id + 7][:l0_slots],
    )
    bl = pl.make_tile_group(
        type=pl.TileType(
            shape=[K0, BN], dtype=FP8, target_memory=pl.MemorySpace.Right, compact=1
        ),
        addrs=[0, BN * K0][:l0_slots],
        mutex_ids=[cube_id + 6 + l0_slots, cube_id + 7 + l0_slots][:l0_slots],
    )
    als = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, K0 // 32],
            dtype=E8,
            target_memory=pl.MemorySpace.ScaleLeft,
            compact=1,
        ),
        addrs=[0, BM * K0 // 16][:l0_slots],
        mutex_ids=[cube_id + 6 + 2 * l0_slots, cube_id + 7 + 2 * l0_slots][:l0_slots],
    )
    bls = pl.make_tile_group(
        type=pl.TileType(
            shape=[K0 // 32, BN],
            dtype=E8,
            target_memory=pl.MemorySpace.ScaleRight,
            compact=1,
        ),
        addrs=[0, BN * K0 // 16][:l0_slots],
        mutex_ids=[cube_id + 6 + 3 * l0_slots, cube_id + 7 + 3 * l0_slots][:l0_slots],
    )
    accg = pl.make_tile_group(
        type=pl.TileType(
            shape=[BM, BN],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            compact=1,
        ),
        addrs=[0, 4 * BM * BN][:acc_slots],
        mutex_ids=[cube_id + 6 + 4 * l0_slots, cube_id + 7 + 4 * l0_slots][:acc_slots],
    )
    b_transposed = pl.reinterpret(bg, shape=[BK, BN], layout=pl.ZN)
    with pl.section_cube():
        nm = (t + BM - 1) // BM
        nn = (n + BN - 1) // BN
        for work in pl.range(pl.get_block_idx(), nm * nn, pl.get_block_num()):
            mi = work // nn % nm * BM
            ni = work % nn * BN
            mv = pl.min(BM, t - mi)
            nv = pl.min(BN, n - ni)
            acc = accg.next()
            ast = asg.current()
            bst = bsg.current()
            pl.set_validshape(acc, [mv, nv])
            pl.set_validshape(ast, [mv, k // 32])
            pl.set_validshape(bst, [k // 32, nv])
            pl.set_validshape(ag, [mv, BK])
            pl.set_validshape(bg, [nv, BK])
            pl.set_validshape(b_transposed, [BK, nv])
            pl.set_validshape(al, [mv, K0])
            pl.set_validshape(als, [mv, K0 // 32])
            pl.set_validshape(bl, [K0, nv])
            pl.set_validshape(bls, [K0 // 32, nv])
            pl.load(ast, sa, [mi, 0, 0])
            pl.load(bst, sb, [ni, 0, 0], order=[1, 0])
            for ki in pl.range(0, k, BK):
                at = ag.next()
                bt = bg[(ki // BK) % 2]
                pl.load(at, a, [mi, ki])
                if work != pl.get_block_idx() or ki != 0:
                    pl.load(bt, b, [ni, ki])
                bt_transposed = b_transposed[(ki // BK) % 2]
                _mx_block(acc, at, bt_transposed, ast, bst, al, als, bl, bls, ki, k, ki)
            pl.store(output, acc, [mi, ni], phase=pl.STPhase.Final)


def _mx_block(acc, at, bt_transposed, ast, bst, al, als, bl, bls, ki, k, sa_ki):
    _mx_step(acc, at, bt_transposed, ast, bst, al, als, bl, bls, ki, k, sa_ki, 0)
    if BK >= 2 * K0:
        _mx_step(acc, at, bt_transposed, ast, bst, al, als, bl, bls, ki, k, sa_ki, K0)
    if BK >= 4 * K0:
        _mx_step(
            acc, at, bt_transposed, ast, bst, al, als, bl, bls, ki, k, sa_ki, 2 * K0
        )
        _mx_step(
            acc, at, bt_transposed, ast, bst, al, als, bl, bls, ki, k, sa_ki, 3 * K0
        )
    if BK >= 8 * K0:
        _mx_step(
            acc, at, bt_transposed, ast, bst, al, als, bl, bls, ki, k, sa_ki, 4 * K0
        )
        _mx_step(
            acc, at, bt_transposed, ast, bst, al, als, bl, bls, ki, k, sa_ki, 5 * K0
        )
        _mx_step(
            acc, at, bt_transposed, ast, bst, al, als, bl, bls, ki, k, sa_ki, 6 * K0
        )
        _mx_step(
            acc, at, bt_transposed, ast, bst, al, als, bl, bls, ki, k, sa_ki, 7 * K0
        )


def _mx_step(acc, at, bt, ast, bst, al, als, bl, bls, ki, k, sa_ki, inner):
    left = al[(inner // K0) % (256 // K0)]
    left_scale = als[(inner // K0) % (256 // K0)]
    right = bl[(inner // K0) % (256 // K0)]
    right_scale = bls[(inner // K0) % (256 // K0)]
    # Unit flags overlap M and FIX across output tiles.
    pl.move(left, at, [0, inner])
    pl.move(left_scale, ast, [0, (sa_ki + inner) // 32])
    pl.move(right, bt, [inner, 0])
    pl.move(right_scale, bst, [(ki + inner) // 32, 0])
    if inner == 0 and ki == 0:
        pl.matmul_mx(
            acc, left, right, left_scale, right_scale, phase=pl.AccPhase.Partial
        )
    elif inner + K0 == BK and ki + BK == k:
        pl.matmul_mx_acc(
            acc, acc, left, right, left_scale, right_scale, phase=pl.AccPhase.Final
        )
    else:
        pl.matmul_mx_acc(
            acc, acc, left, right, left_scale, right_scale, phase=pl.AccPhase.Partial
        )


_SCRATCH = OrderedDict()
_LOCK = threading.RLock()


def _scratch(attn_out, padded):
    key = (
        attn_out.device,
        torch.npu.current_stream(attn_out.device).npu_stream,
        padded,
    )
    capture = torch.npu.is_current_stream_capturing()
    with _LOCK:
        if not capture and key in _SCRATCH:
            _SCRATCH.move_to_end(key)
            return _SCRATCH[key]
        specs = [
            ((8 * padded, 4096), torch.uint8),
            ((8 * padded, 128), torch.uint8),
            ((padded, 8192), torch.uint8),
            ((padded, 256), torch.uint8),
        ]
        result = tuple(
            torch.empty(shape, dtype=dtype, device=attn_out.device)
            for shape, dtype in specs
        )
        if not capture:
            _SCRATCH[key] = result
            if len(_SCRATCH) > 8:
                _SCRATCH.popitem(last=False)
        return result


def _tiling_key(t, cube_cores):
    wide = cube_cores == 32
    bm, bk, slots = (
        (64, 512, 2) if t <= 64 else ((128, 256, 2) if t <= 128 else (256, 256, 1))
    )
    return dict(
        BM=bm,
        BK=bk,
        VSlots=slots,
        K0=128 if wide and bm >= 128 else 256,
        BN1=256 if wide else 160,
        BN=160 if wide else 192,
    )


def _launch(
    attn_out, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos, buffers, out
):
    t = attn_out.shape[0]
    padded = buffers[2].shape[0]
    info = get_platform_info()
    args = (
        attn_out,
        rope_cos,
        rope_sin,
        woa,
        descale_woa,
        wob,
        descale_wob,
        *buffers,
        out,
        t,
        padded,
    )
    with pypto.options(pass_options={"enable_slice": False}):
        attn_epilogue_kernel[
            None, info.cube_core_num, _tiling_key(t, info.cube_core_num)
        ](*args)


def attn_epilogue(attn_out, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos):
    """Return a contiguous BF16 [T,5120] projection from prequantized ND weights."""
    if (
        not isinstance(attn_out, torch.Tensor)
        or attn_out.ndim != 3
        or tuple(attn_out.shape[1:]) != (64, 512)
    ):
        raise ValueError("attn_out must have shape [T,64,512]")
    t = attn_out.shape[0]
    if t < 1:
        raise ValueError("T must be positive")
    if attn_out.device.type not in ("npu", "privateuseone"):
        raise ValueError("inputs must be on an NPU")
    for name, tensor, shape, dtype in (
        ("attn_out", attn_out, (t, 64, 512), torch.bfloat16),
        ("woa", woa, (8, 1024, 4096), torch.float8_e4m3fn),
        ("wob", wob, (5120, 8192), torch.float8_e4m3fn),
        ("descale_woa", descale_woa, (8, 1024, 64, 2), torch.float8_e8m0fnu),
        ("descale_wob", descale_wob, (5120, 128, 2), torch.float8_e8m0fnu),
        ("rope_sin", rope_sin, (t, 64), torch.float32),
        ("rope_cos", rope_cos, (t, 64), torch.float32),
    ):
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if tensor.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}")
        if tensor.device != attn_out.device or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous on {attn_out.device}")
        if torch_npu.get_npu_format(tensor) not in (0, 2):
            raise ValueError(f"{name} must use ND storage")
    padded = (t + 15) // 16 * 16
    with torch.npu.device(attn_out.device):
        buffers = _scratch(attn_out, padded)
        out = torch.empty((t, 5120), dtype=torch.bfloat16, device=attn_out.device)
        _launch(
            attn_out,
            woa,
            wob,
            descale_woa,
            descale_wob,
            rope_sin,
            rope_cos,
            buffers,
            out,
        )
    return out
