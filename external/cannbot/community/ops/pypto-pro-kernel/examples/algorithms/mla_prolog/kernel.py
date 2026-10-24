# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：混合。MLA prolog with shape-specific Vector and Cube projections.

Inputs are BF16: token_x[B,S,He], w_dq[He,1536], w_dkv_kr[He,576],
w_uq_qr[1536,128*192], w_uk[128,128,512], gamma_cq[1536], gamma_ckv[512]
and rope_sin/cos[B,S,64]. He is 6144, 7168 or 7680; M=B*S is positive.
Outputs are BF16 query[B,S,128,512], query_rope[B,S,128,64],
c_kv[B,S,512] and k_rope[B,S,64]. Each head's 192 columns contain 128 query
channels followed by 64 RoPE channels.

The numerical contract retains FP32 intermediates. Down projections use four
K partitions and merge their FP32 partial sums before RMSNorm. Cube consumers
receive three BF16 components of each FP32 activation; plain BF16 intermediates
lose cancellation accuracy. Weights are reused across the three products.
Vector consumers keep FP32 activations directly and widen BF16 weights in VF.

Upper projection dispatch uses input metadata only:
* M=1: seven FP32 register accumulators cover a 448-column weight tile.
* M=4: sixteen accumulators cover four rows and 256 columns with shared weights.
* Other M<=64: each of 28 Cube groups owns an aligned 832/896-column slab.
  Its two AIVs load disjoint weight columns, convert ND to NZ and insert into
  shared Mat slots. The pipeline declares both handover and reuse events.
* M>64: Cube tiles cover 128 rows by 512 columns.

For M=1, a final Vector kernel computes query RoPE and the per-head W_UK
projection directly from FP32 qr. Other shapes pack query activations into
three BF16 components before the Cube W_UK projection. Calls therefore launch
four kernels for M=1 and five otherwise, all on the same stream.

Tile groups expose physical addresses, slot counts and synchronization. Host
code validates metadata, allocates intermediates and dispatches; it never reads
device values. No transformed weights persist across calls.
"""

from functools import lru_cache

import torch

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32
BF16 = pl.DT_BF16

HCQ = 1536
HCKV = 512
DR = 64
D = 128
NH = 128
NKR = HCKV + DR  # 576
HE_CHOICES = (6144, 7168, 7680)

# --- down projection -------------------------------------------------------
DOWN_MB = 128
DOWN_KP = 256
DOWN_ABYTES = DOWN_MB * DOWN_KP * 2
# Two column tilings: narrow tiles keep all 28 workers fed at decode, wide
# tiles cut the repeated A reads that dominate once the token count grows.
DOWN_NB_SMALL = 64  # 33 tiles over [w_dq | w_dkv_kr]
DOWN_NB_BIG = 192  # 11 tiles (must divide 1536 and 576)
DOWN_NS = 4  # k-split partials of the down GEMM

# --- query upper projection ------------------------------------------------
UP_MB = 128
UP_KP = 128
UP_KS = 32
UP_NB = 512
UP_NT = NH * (D + DR) // UP_NB  # 48 column tiles
UP_ABYTES = UP_MB * UP_KP * 2
UP_ADEPTH = 1  # Mat slots per activation tile
UP_BDEPTH = 2  # Mat slots for the weight tile
UP_A0 = 0
UP_A1 = UP_A0 + UP_ADEPTH * UP_ABYTES
UP_A2 = UP_A1 + UP_ADEPTH * UP_ABYTES
UP_B = UP_A2 + UP_ADEPTH * UP_ABYTES

# --- W_UK absorption -------------------------------------------------------
UK_MB = 128
UK_KS = 32
UK_NS = 1  # 512-wide output per task
UK_NB = HCKV // UK_NS
UK_ABYTES = UK_MB * D * 2
UK_ADEPTH = 2  # Mat slots per activation tile
UK_BDEPTH = 2  # Mat slots for the w_uk tile
UK_A0 = 0
UK_A1 = UK_A0 + UK_ADEPTH * UK_ABYTES
UK_A2 = UK_A1 + UK_ADEPTH * UK_ABYTES
UK_B = UK_A2 + UK_ADEPTH * UK_ABYTES

# --- q_c packing / query RoPE ---------------------------------------------
HB = 16  # heads per split task

CUBE_WORKERS = 28
VEC_WORKERS = 56


def _rmsnorm_vf(width):
    """Row RMSNorm over a compile-time width; emits three BF16 components."""
    n_regs = width // 64
    inv_width = 1.0 / width

    @pl.vector_function
    def rmsnorm_row(src, gamma, dst0, dst1, dst2, eps: pl.DT_FP32):
        preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        total = vf.full(0.0, preg, dtype=FP32)
        for r in pl.range(n_regs):
            reg = vf.load_align(src, r * 64)
            square = vf.mul(reg, reg, preg)
            part = vf.reduce_sum(square, preg)
            total = vf.add(total, part, preg)
        mean = vf.muls(total, inv_width, preg)
        mean = vf.adds(mean, eps, preg)
        rms = vf.sqrt(mean, preg)
        rms_b = vf.full(rms, preg)
        for r in pl.range(n_regs):
            reg = vf.load_align(src, r * 64)
            g16 = vf.load_align(gamma, r * 64, dist=pl.LoadDist.UNPK)
            scale = vf.astype(g16, preg, dtype=FP32, layout=pl.CastLayout.ZERO)
            norm = vf.div(reg, rms_b, preg)
            out = vf.mul(norm, scale, preg)
            c0 = vf.astype(
                out,
                preg,
                dtype=BF16,
                layout=pl.CastLayout.ZERO,
                round_mode=pl.VFRoundMode.CAST_RINT,
            )
            f0 = vf.astype(c0, preg, dtype=FP32, layout=pl.CastLayout.ZERO)
            rem1 = vf.sub(out, f0, preg)
            c1 = vf.astype(
                rem1,
                preg,
                dtype=BF16,
                layout=pl.CastLayout.ZERO,
                round_mode=pl.VFRoundMode.CAST_RINT,
            )
            f1 = vf.astype(c1, preg, dtype=FP32, layout=pl.CastLayout.ZERO)
            rem2 = vf.sub(rem1, f1, preg)
            c2 = vf.astype(
                rem2,
                preg,
                dtype=BF16,
                layout=pl.CastLayout.ZERO,
                round_mode=pl.VFRoundMode.CAST_RINT,
            )
            vf.store_align(dst0, c0, preg, r * 64, dist=pl.StoreDist.PACK_B32)
            vf.store_align(dst1, c1, preg, r * 64, dist=pl.StoreDist.PACK_B32)
            vf.store_align(dst2, c2, preg, r * 64, dist=pl.StoreDist.PACK_B32)

    return rmsnorm_row


@pl.vector_function
def _sum_rows(dst, s1, s2, s3, width: pl.DT_INT64):
    """dst += s1 + s2 + s3, one FP32 row of ``width`` lanes."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    for r in pl.range(width // 64):
        acc = vf.load_align(dst, r * 64)
        v1 = vf.load_align(s1, r * 64)
        acc = vf.add(acc, v1, preg)
        v2 = vf.load_align(s2, r * 64)
        acc = vf.add(acc, v2, preg)
        v3 = vf.load_align(s3, r * 64)
        acc = vf.add(acc, v3, preg)
        vf.store_align(dst, acc, preg, r * 64)


@pl.vector_function
def _rope_rows_f32(src, cos_tile, sin_tile, dst, rows: pl.DT_INT64, base: pl.DT_INT64):
    """y = x * cos + rotate_half(x) * sin for ``rows`` 64-wide FP32 rows.

    rotate_half(x)[j] = -x[j+32] for j < 32 and +x[j-32] otherwise: the
    (lane + 32) mod 64 register rotate plus a sign selected by lane >= 32.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    u_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    lanes = vf.arange(0, dtype=pl.DT_UINT32)
    half = vf.full(32, u_all, dtype=pl.DT_UINT32)
    last = vf.full(63, u_all, dtype=pl.DT_UINT32)
    rolled = vf.adds(lanes, 32, u_all)
    rolled = vf.and_(rolled, last, u_all)
    high = vf.ge(lanes, half, u_all)
    cb = vf.load_align(cos_tile, 0, dist=pl.LoadDist.UNPK)
    cos = vf.astype(cb, preg, dtype=FP32, layout=pl.CastLayout.ZERO)
    sb = vf.load_align(sin_tile, 0, dist=pl.LoadDist.UNPK)
    sin = vf.astype(sb, preg, dtype=FP32, layout=pl.CastLayout.ZERO)
    for r in pl.range(rows):
        x = vf.load_align(src, [r, base])
        partner = vf.gather(x, rolled)
        neg = vf.neg(partner, preg)
        signed = vf.select(partner, neg, high)
        a = vf.mul(x, cos, preg)
        b = vf.mul(signed, sin, preg)
        out = vf.add(a, b, preg)
        packed = vf.astype(
            out,
            preg,
            dtype=BF16,
            layout=pl.CastLayout.ZERO,
            round_mode=pl.VFRoundMode.CAST_RINT,
        )
        vf.store_align(dst, packed, preg, [r, 0], dist=pl.StoreDist.PACK_B32)


@pl.vector_function
def _split_rope(src, cos_tile, sin_tile, hi_tile, lo_tile, lo2_tile, qrope_tile):
    """Pack one token's ``HB`` heads: q_c hi/lo/lo2 (128 cols) + RoPE (64 cols)."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    u_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    lanes = vf.arange(0, dtype=pl.DT_UINT32)
    half = vf.full(32, u_all, dtype=pl.DT_UINT32)
    last = vf.full(63, u_all, dtype=pl.DT_UINT32)
    rolled = vf.adds(lanes, 32, u_all)
    rolled = vf.and_(rolled, last, u_all)
    high = vf.ge(lanes, half, u_all)
    cb = vf.load_align(cos_tile, 0, dist=pl.LoadDist.UNPK)
    cos = vf.astype(cb, preg, dtype=FP32, layout=pl.CastLayout.ZERO)
    sb = vf.load_align(sin_tile, 0, dist=pl.LoadDist.UNPK)
    sin = vf.astype(sb, preg, dtype=FP32, layout=pl.CastLayout.ZERO)
    for h in pl.range(HB):
        base = h * (D + DR)
        for c in pl.range(D // 64):
            x = vf.load_align(src, base + c * 64)
            c0 = vf.astype(
                x,
                preg,
                dtype=BF16,
                layout=pl.CastLayout.ZERO,
                round_mode=pl.VFRoundMode.CAST_RINT,
            )
            f0 = vf.astype(c0, preg, dtype=FP32, layout=pl.CastLayout.ZERO)
            rem1 = vf.sub(x, f0, preg)
            c1 = vf.astype(
                rem1,
                preg,
                dtype=BF16,
                layout=pl.CastLayout.ZERO,
                round_mode=pl.VFRoundMode.CAST_RINT,
            )
            f1 = vf.astype(c1, preg, dtype=FP32, layout=pl.CastLayout.ZERO)
            rem2 = vf.sub(rem1, f1, preg)
            c2 = vf.astype(
                rem2,
                preg,
                dtype=BF16,
                layout=pl.CastLayout.ZERO,
                round_mode=pl.VFRoundMode.CAST_RINT,
            )
            off = h * D + c * 64
            vf.store_align(hi_tile, c0, preg, off, dist=pl.StoreDist.PACK_B32)
            vf.store_align(lo_tile, c1, preg, off, dist=pl.StoreDist.PACK_B32)
            vf.store_align(lo2_tile, c2, preg, off, dist=pl.StoreDist.PACK_B32)
        xr = vf.load_align(src, base + D)
        partner = vf.gather(xr, rolled)
        neg = vf.neg(partner, preg)
        signed = vf.select(partner, neg, high)
        a = vf.mul(xr, cos, preg)
        b = vf.mul(signed, sin, preg)
        out = vf.add(a, b, preg)
        packed = vf.astype(
            out,
            preg,
            dtype=BF16,
            layout=pl.CastLayout.ZERO,
            round_mode=pl.VFRoundMode.CAST_RINT,
        )
        vf.store_align(qrope_tile, packed, preg, h * DR, dist=pl.StoreDist.PACK_B32)


def _rmsnorm_full_vf(width):
    n_regs = width // 64
    inv_width = 1.0 / width

    @pl.vector_function
    def rmsnorm_full(src, gamma, dst0, dst1, dst2, eps: pl.DT_FP32):
        preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        total = vf.full(0.0, preg, dtype=FP32)
        for r in pl.range(n_regs):
            reg = vf.load_align(src, r * 64)
            square = vf.mul(reg, reg, preg)
            part = vf.reduce_sum(square, preg)
            total = vf.add(total, part, preg)
        mean = vf.muls(total, inv_width, preg)
        mean = vf.adds(mean, eps, preg)
        rms = vf.sqrt(mean, preg)
        rms_b = vf.full(rms, preg)
        for r in pl.range(n_regs):
            reg = vf.load_align(src, r * 64)
            g16 = vf.load_align(gamma, r * 64, dist=pl.LoadDist.UNPK)
            scale = vf.astype(g16, preg, dtype=FP32, layout=pl.CastLayout.ZERO)
            norm = vf.div(reg, rms_b, preg)
            out = vf.mul(norm, scale, preg)
            vf.store_align(dst0, out, preg, r * 64)

    return rmsnorm_full


# M=1 Vec: A at 0; two 114688-byte B slots at 8192; C ends at 239360.
VUP_KP = 128
VUP_NB = 448
VUP_DEPTH = 2


@pl.vector_function
def _vup_clear(dst):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    zero = vf.full(0.0, preg, dtype=FP32)
    for j in pl.range(VUP_NB // 64):
        vf.store_align(dst, zero, preg, j * 64)


@pl.vector_function
def _vup_chunk(a, w, dst, k0: pl.DT_INT64):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    acc0 = vf.full(0.0, preg, dtype=FP32)
    acc1 = vf.full(0.0, preg, dtype=FP32)
    acc2 = vf.full(0.0, preg, dtype=FP32)
    acc3 = vf.full(0.0, preg, dtype=FP32)
    acc4 = vf.full(0.0, preg, dtype=FP32)
    acc5 = vf.full(0.0, preg, dtype=FP32)
    acc6 = vf.full(0.0, preg, dtype=FP32)
    for k in pl.range(VUP_KP):
        av = vf.load_align(a, k0 + k, dist=pl.LoadDist.BRC_B32)
        wb0 = vf.load_align(w, k * VUP_NB + 0, dist=pl.LoadDist.UNPK)
        wf0 = vf.astype(wb0, preg, dtype=FP32, layout=pl.CastLayout.ZERO)
        acc0 = vf.mul_add_dst(av, wf0, preg)
        wb1 = vf.load_align(w, k * VUP_NB + 64, dist=pl.LoadDist.UNPK)
        wf1 = vf.astype(wb1, preg, dtype=FP32, layout=pl.CastLayout.ZERO)
        acc1 = vf.mul_add_dst(av, wf1, preg)
        wb2 = vf.load_align(w, k * VUP_NB + 128, dist=pl.LoadDist.UNPK)
        wf2 = vf.astype(wb2, preg, dtype=FP32, layout=pl.CastLayout.ZERO)
        acc2 = vf.mul_add_dst(av, wf2, preg)
        wb3 = vf.load_align(w, k * VUP_NB + 192, dist=pl.LoadDist.UNPK)
        wf3 = vf.astype(wb3, preg, dtype=FP32, layout=pl.CastLayout.ZERO)
        acc3 = vf.mul_add_dst(av, wf3, preg)
        wb4 = vf.load_align(w, k * VUP_NB + 256, dist=pl.LoadDist.UNPK)
        wf4 = vf.astype(wb4, preg, dtype=FP32, layout=pl.CastLayout.ZERO)
        acc4 = vf.mul_add_dst(av, wf4, preg)
        wb5 = vf.load_align(w, k * VUP_NB + 320, dist=pl.LoadDist.UNPK)
        wf5 = vf.astype(wb5, preg, dtype=FP32, layout=pl.CastLayout.ZERO)
        acc5 = vf.mul_add_dst(av, wf5, preg)
        wb6 = vf.load_align(w, k * VUP_NB + 384, dist=pl.LoadDist.UNPK)
        wf6 = vf.astype(wb6, preg, dtype=FP32, layout=pl.CastLayout.ZERO)
        acc6 = vf.mul_add_dst(av, wf6, preg)
    old0 = vf.load_align(dst, 0)
    out0 = vf.add(old0, acc0, preg)
    vf.store_align(dst, out0, preg, 0)
    old1 = vf.load_align(dst, 64)
    out1 = vf.add(old1, acc1, preg)
    vf.store_align(dst, out1, preg, 64)
    old2 = vf.load_align(dst, 128)
    out2 = vf.add(old2, acc2, preg)
    vf.store_align(dst, out2, preg, 128)
    old3 = vf.load_align(dst, 192)
    out3 = vf.add(old3, acc3, preg)
    vf.store_align(dst, out3, preg, 192)
    old4 = vf.load_align(dst, 256)
    out4 = vf.add(old4, acc4, preg)
    vf.store_align(dst, out4, preg, 256)
    old5 = vf.load_align(dst, 320)
    out5 = vf.add(old5, acc5, preg)
    vf.store_align(dst, out5, preg, 320)
    old6 = vf.load_align(dst, 384)
    out6 = vf.add(old6, acc6, preg)
    vf.store_align(dst, out6, preg, 384)


@pl.jit(arch="a5", auto_mutex=True, name="mla_prolog_vup_fp32")
def _up_vector(
    a: pl.Tensor[[pl.DYNAMIC, HCQ], FP32],
    w: pl.Tensor[[HCQ, NH * (D + DR)], BF16],
    out: pl.Tensor[[pl.DYNAMIC, NH * (D + DR)], FP32],
):
    a_g = pl.make_tile_group(
        type=pl.TileType(shape=[1, HCQ], dtype=FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0,
        mutex_ids="auto",
        depth=1,
    )
    w_g = pl.make_tile_group(
        type=pl.TileType(
            shape=[VUP_KP, VUP_NB], dtype=BF16, target_memory=pl.MemorySpace.Vec
        ),
        addrs=8192,
        mutex_ids="auto",
        depth=VUP_DEPTH,
    )
    c_g = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, VUP_NB], dtype=FP32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=8192 + VUP_DEPTH * VUP_KP * VUP_NB * 2,
        mutex_ids="auto",
        depth=1,
    )
    with pl.section_vector():
        nt = (NH * (D + DR) + VUP_NB - 1) // VUP_NB
        for task in pl.range(pl.get_block_idx(), a.shape[0] * nt, pl.get_block_num()):
            row = task // nt
            n0 = (task % nt) * VUP_NB
            cols = pl.min(VUP_NB, NH * (D + DR) - n0)
            aa = a_g.current()
            cc = c_g.current()
            pl.load(aa, a, [row, 0])
            _vup_clear(cc)
            for k0 in pl.range(0, HCQ, VUP_KP):
                ww = w_g.current()
                pl.set_validshape(ww, [VUP_KP, cols])
                pl.load(ww, w, [k0, n0])
                _vup_chunk(aa, ww, cc, k0)
                w_g.next()
            pl.set_validshape(cc, [1, cols])
            pl.store(out, cc, [row, n0])


# M=4 Vec: A 24576 B; three 65536-byte B slots; C ends at 225280.
VUP4_KP = 128
VUP4_NB = 256
VUP4_DEPTH = 3


@pl.vector_function
def _vup_clear4(dst):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    zero = vf.full(0.0, preg, dtype=FP32)
    for j in pl.range(16):
        vf.store_align(dst, zero, preg, j * 64)


@pl.vector_function
def _vup_chunk4(a, w, dst, k0: pl.DT_INT64):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    acc0_0 = vf.full(0.0, preg, dtype=FP32)
    acc0_1 = vf.full(0.0, preg, dtype=FP32)
    acc0_2 = vf.full(0.0, preg, dtype=FP32)
    acc0_3 = vf.full(0.0, preg, dtype=FP32)
    acc1_0 = vf.full(0.0, preg, dtype=FP32)
    acc1_1 = vf.full(0.0, preg, dtype=FP32)
    acc1_2 = vf.full(0.0, preg, dtype=FP32)
    acc1_3 = vf.full(0.0, preg, dtype=FP32)
    acc2_0 = vf.full(0.0, preg, dtype=FP32)
    acc2_1 = vf.full(0.0, preg, dtype=FP32)
    acc2_2 = vf.full(0.0, preg, dtype=FP32)
    acc2_3 = vf.full(0.0, preg, dtype=FP32)
    acc3_0 = vf.full(0.0, preg, dtype=FP32)
    acc3_1 = vf.full(0.0, preg, dtype=FP32)
    acc3_2 = vf.full(0.0, preg, dtype=FP32)
    acc3_3 = vf.full(0.0, preg, dtype=FP32)
    for k in pl.range(VUP4_KP):
        av0 = vf.load_align(a, 0 + k0 + k, dist=pl.LoadDist.BRC_B32)
        av1 = vf.load_align(a, 1536 + k0 + k, dist=pl.LoadDist.BRC_B32)
        av2 = vf.load_align(a, 3072 + k0 + k, dist=pl.LoadDist.BRC_B32)
        av3 = vf.load_align(a, 4608 + k0 + k, dist=pl.LoadDist.BRC_B32)
        wb0 = vf.load_align(w, k * VUP4_NB + 0, dist=pl.LoadDist.UNPK)
        wf0 = vf.astype(wb0, preg, dtype=FP32, layout=pl.CastLayout.ZERO)
        acc0_0 = vf.mul_add_dst(av0, wf0, preg)
        acc1_0 = vf.mul_add_dst(av1, wf0, preg)
        acc2_0 = vf.mul_add_dst(av2, wf0, preg)
        acc3_0 = vf.mul_add_dst(av3, wf0, preg)
        wb1 = vf.load_align(w, k * VUP4_NB + 64, dist=pl.LoadDist.UNPK)
        wf1 = vf.astype(wb1, preg, dtype=FP32, layout=pl.CastLayout.ZERO)
        acc0_1 = vf.mul_add_dst(av0, wf1, preg)
        acc1_1 = vf.mul_add_dst(av1, wf1, preg)
        acc2_1 = vf.mul_add_dst(av2, wf1, preg)
        acc3_1 = vf.mul_add_dst(av3, wf1, preg)
        wb2 = vf.load_align(w, k * VUP4_NB + 128, dist=pl.LoadDist.UNPK)
        wf2 = vf.astype(wb2, preg, dtype=FP32, layout=pl.CastLayout.ZERO)
        acc0_2 = vf.mul_add_dst(av0, wf2, preg)
        acc1_2 = vf.mul_add_dst(av1, wf2, preg)
        acc2_2 = vf.mul_add_dst(av2, wf2, preg)
        acc3_2 = vf.mul_add_dst(av3, wf2, preg)
        wb3 = vf.load_align(w, k * VUP4_NB + 192, dist=pl.LoadDist.UNPK)
        wf3 = vf.astype(wb3, preg, dtype=FP32, layout=pl.CastLayout.ZERO)
        acc0_3 = vf.mul_add_dst(av0, wf3, preg)
        acc1_3 = vf.mul_add_dst(av1, wf3, preg)
        acc2_3 = vf.mul_add_dst(av2, wf3, preg)
        acc3_3 = vf.mul_add_dst(av3, wf3, preg)
    old0_0 = vf.load_align(dst, 0)
    out0_0 = vf.add(old0_0, acc0_0, preg)
    vf.store_align(dst, out0_0, preg, 0)
    old0_1 = vf.load_align(dst, 64)
    out0_1 = vf.add(old0_1, acc0_1, preg)
    vf.store_align(dst, out0_1, preg, 64)
    old0_2 = vf.load_align(dst, 128)
    out0_2 = vf.add(old0_2, acc0_2, preg)
    vf.store_align(dst, out0_2, preg, 128)
    old0_3 = vf.load_align(dst, 192)
    out0_3 = vf.add(old0_3, acc0_3, preg)
    vf.store_align(dst, out0_3, preg, 192)
    old1_0 = vf.load_align(dst, 256)
    out1_0 = vf.add(old1_0, acc1_0, preg)
    vf.store_align(dst, out1_0, preg, 256)
    old1_1 = vf.load_align(dst, 320)
    out1_1 = vf.add(old1_1, acc1_1, preg)
    vf.store_align(dst, out1_1, preg, 320)
    old1_2 = vf.load_align(dst, 384)
    out1_2 = vf.add(old1_2, acc1_2, preg)
    vf.store_align(dst, out1_2, preg, 384)
    old1_3 = vf.load_align(dst, 448)
    out1_3 = vf.add(old1_3, acc1_3, preg)
    vf.store_align(dst, out1_3, preg, 448)
    old2_0 = vf.load_align(dst, 512)
    out2_0 = vf.add(old2_0, acc2_0, preg)
    vf.store_align(dst, out2_0, preg, 512)
    old2_1 = vf.load_align(dst, 576)
    out2_1 = vf.add(old2_1, acc2_1, preg)
    vf.store_align(dst, out2_1, preg, 576)
    old2_2 = vf.load_align(dst, 640)
    out2_2 = vf.add(old2_2, acc2_2, preg)
    vf.store_align(dst, out2_2, preg, 640)
    old2_3 = vf.load_align(dst, 704)
    out2_3 = vf.add(old2_3, acc2_3, preg)
    vf.store_align(dst, out2_3, preg, 704)
    old3_0 = vf.load_align(dst, 768)
    out3_0 = vf.add(old3_0, acc3_0, preg)
    vf.store_align(dst, out3_0, preg, 768)
    old3_1 = vf.load_align(dst, 832)
    out3_1 = vf.add(old3_1, acc3_1, preg)
    vf.store_align(dst, out3_1, preg, 832)
    old3_2 = vf.load_align(dst, 896)
    out3_2 = vf.add(old3_2, acc3_2, preg)
    vf.store_align(dst, out3_2, preg, 896)
    old3_3 = vf.load_align(dst, 960)
    out3_3 = vf.add(old3_3, acc3_3, preg)
    vf.store_align(dst, out3_3, preg, 960)


@pl.jit(arch="a5", auto_mutex=True, name="mla_prolog_vup4_fp32")
def _up_vector4(
    a: pl.Tensor[[pl.DYNAMIC, HCQ], FP32],
    w: pl.Tensor[[HCQ, NH * (D + DR)], BF16],
    out: pl.Tensor[[pl.DYNAMIC, NH * (D + DR)], FP32],
):
    a_g = pl.make_tile_group(
        type=pl.TileType(shape=[4, HCQ], dtype=FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0,
        mutex_ids="auto",
        depth=1,
    )
    w_g = pl.make_tile_group(
        type=pl.TileType(
            shape=[VUP4_KP, VUP4_NB], dtype=BF16, target_memory=pl.MemorySpace.Vec
        ),
        addrs=24576,
        mutex_ids="auto",
        depth=VUP4_DEPTH,
    )
    c_g = pl.make_tile_group(
        type=pl.TileType(
            shape=[4, VUP4_NB], dtype=FP32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=24576 + VUP4_DEPTH * VUP4_KP * VUP4_NB * 2,
        mutex_ids="auto",
        depth=1,
    )
    with pl.section_vector():
        nt = (NH * (D + DR) + VUP4_NB - 1) // VUP4_NB
        mt = (a.shape[0] + 3) // 4
        for task in pl.range(pl.get_block_idx(), mt * nt, pl.get_block_num()):
            row = (task // nt) * 4
            rows = pl.min(4, a.shape[0] - row)
            n0 = (task % nt) * VUP4_NB
            cols = pl.min(VUP4_NB, NH * (D + DR) - n0)
            aa = a_g.current()
            cc = c_g.current()
            pl.set_validshape(aa, [rows, HCQ])
            pl.load(aa, a, [row, 0])
            _vup_clear4(cc)
            for k0 in pl.range(0, HCQ, VUP4_KP):
                ww = w_g.current()
                pl.set_validshape(ww, [VUP4_KP, cols])
                pl.load(ww, w, [k0, n0])
                _vup_chunk4(aa, ww, cc, k0)
                w_g.next()
            pl.set_validshape(cc, [rows, cols])
            pl.store(out, cc, [row, n0])


@pl.vector_function
def _uk_clear(dst):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    zero = vf.full(0.0, preg, dtype=FP32)
    for j in pl.range(8):
        vf.store_align(dst, zero, preg, j * 64)


@pl.vector_function
def _uk_chunk(a, w, dst, k0: pl.DT_INT64):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    acc0 = vf.full(0.0, preg, dtype=FP32)
    acc1 = vf.full(0.0, preg, dtype=FP32)
    acc2 = vf.full(0.0, preg, dtype=FP32)
    acc3 = vf.full(0.0, preg, dtype=FP32)
    acc4 = vf.full(0.0, preg, dtype=FP32)
    acc5 = vf.full(0.0, preg, dtype=FP32)
    acc6 = vf.full(0.0, preg, dtype=FP32)
    acc7 = vf.full(0.0, preg, dtype=FP32)
    for k in pl.range(32):
        av = vf.load_align(a, k0 + k, dist=pl.LoadDist.BRC_B32)
        wb0 = vf.load_align(w, k * HCKV + 0, dist=pl.LoadDist.UNPK)
        wf0 = vf.astype(wb0, preg, dtype=FP32, layout=pl.CastLayout.ZERO)
        acc0 = vf.mul_add_dst(av, wf0, preg)
        wb1 = vf.load_align(w, k * HCKV + 64, dist=pl.LoadDist.UNPK)
        wf1 = vf.astype(wb1, preg, dtype=FP32, layout=pl.CastLayout.ZERO)
        acc1 = vf.mul_add_dst(av, wf1, preg)
        wb2 = vf.load_align(w, k * HCKV + 128, dist=pl.LoadDist.UNPK)
        wf2 = vf.astype(wb2, preg, dtype=FP32, layout=pl.CastLayout.ZERO)
        acc2 = vf.mul_add_dst(av, wf2, preg)
        wb3 = vf.load_align(w, k * HCKV + 192, dist=pl.LoadDist.UNPK)
        wf3 = vf.astype(wb3, preg, dtype=FP32, layout=pl.CastLayout.ZERO)
        acc3 = vf.mul_add_dst(av, wf3, preg)
        wb4 = vf.load_align(w, k * HCKV + 256, dist=pl.LoadDist.UNPK)
        wf4 = vf.astype(wb4, preg, dtype=FP32, layout=pl.CastLayout.ZERO)
        acc4 = vf.mul_add_dst(av, wf4, preg)
        wb5 = vf.load_align(w, k * HCKV + 320, dist=pl.LoadDist.UNPK)
        wf5 = vf.astype(wb5, preg, dtype=FP32, layout=pl.CastLayout.ZERO)
        acc5 = vf.mul_add_dst(av, wf5, preg)
        wb6 = vf.load_align(w, k * HCKV + 384, dist=pl.LoadDist.UNPK)
        wf6 = vf.astype(wb6, preg, dtype=FP32, layout=pl.CastLayout.ZERO)
        acc6 = vf.mul_add_dst(av, wf6, preg)
        wb7 = vf.load_align(w, k * HCKV + 448, dist=pl.LoadDist.UNPK)
        wf7 = vf.astype(wb7, preg, dtype=FP32, layout=pl.CastLayout.ZERO)
        acc7 = vf.mul_add_dst(av, wf7, preg)
    old0 = vf.load_align(dst, 0)
    out0 = vf.add(old0, acc0, preg)
    vf.store_align(dst, out0, preg, 0)
    old1 = vf.load_align(dst, 64)
    out1 = vf.add(old1, acc1, preg)
    vf.store_align(dst, out1, preg, 64)
    old2 = vf.load_align(dst, 128)
    out2 = vf.add(old2, acc2, preg)
    vf.store_align(dst, out2, preg, 128)
    old3 = vf.load_align(dst, 192)
    out3 = vf.add(old3, acc3, preg)
    vf.store_align(dst, out3, preg, 192)
    old4 = vf.load_align(dst, 256)
    out4 = vf.add(old4, acc4, preg)
    vf.store_align(dst, out4, preg, 256)
    old5 = vf.load_align(dst, 320)
    out5 = vf.add(old5, acc5, preg)
    vf.store_align(dst, out5, preg, 320)
    old6 = vf.load_align(dst, 384)
    out6 = vf.add(old6, acc6, preg)
    vf.store_align(dst, out6, preg, 384)
    old7 = vf.load_align(dst, 448)
    out7 = vf.add(old7, acc7, preg)
    vf.store_align(dst, out7, preg, 448)


@pl.vector_function
def _uk_pack(src, dst):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    for j in pl.range(8):
        x = vf.load_align(src, j * 64)
        y = vf.astype(
            x,
            preg,
            dtype=BF16,
            layout=pl.CastLayout.ZERO,
            round_mode=pl.VFRoundMode.CAST_RINT,
        )
        vf.store_align(dst, y, preg, j * 64, dist=pl.StoreDist.PACK_B32)


@pl.jit(arch="a5", auto_mutex=True, name="mla_prolog_split_uk_vf")
def _split_uk_vector(
    qr: pl.Tensor[[pl.DYNAMIC, NH * (D + DR)], FP32],
    w: pl.Tensor[[NH, D, HCKV], BF16],
    sin: pl.Tensor[[pl.DYNAMIC, DR], BF16],
    cos: pl.Tensor[[pl.DYNAMIC, DR], BF16],
    query: pl.Tensor[[pl.DYNAMIC, NH * HCKV], BF16],
    query_rope: pl.Tensor[[pl.DYNAMIC, NH * DR], BF16],
):
    a_g = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, D + DR], dtype=FP32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=0,
        mutex_ids="auto",
        depth=1,
    )
    sn_g = pl.make_tile_group(
        type=pl.TileType(shape=[1, DR], dtype=BF16, target_memory=pl.MemorySpace.Vec),
        addrs=1024,
        mutex_ids="auto",
        depth=1,
    )
    cs_g = pl.make_tile_group(
        type=pl.TileType(shape=[1, DR], dtype=BF16, target_memory=pl.MemorySpace.Vec),
        addrs=1152,
        mutex_ids="auto",
        depth=1,
    )
    rope_g = pl.make_tile_group(
        type=pl.TileType(shape=[1, DR], dtype=BF16, target_memory=pl.MemorySpace.Vec),
        addrs=1280,
        mutex_ids="auto",
        depth=1,
    )
    w_g = pl.make_tile_group(
        type=pl.TileType(
            shape=[32, HCKV], dtype=BF16, target_memory=pl.MemorySpace.Vec
        ),
        addrs=4096,
        mutex_ids="auto",
        depth=3,
    )
    c_g = pl.make_tile_group(
        type=pl.TileType(shape=[1, HCKV], dtype=FP32, target_memory=pl.MemorySpace.Vec),
        addrs=102400,
        mutex_ids="auto",
        depth=1,
    )
    y_g = pl.make_tile_group(
        type=pl.TileType(shape=[1, HCKV], dtype=BF16, target_memory=pl.MemorySpace.Vec),
        addrs=104448,
        mutex_ids="auto",
        depth=1,
    )
    with pl.section_vector():
        for task in pl.range(pl.get_block_idx(), qr.shape[0] * NH, pl.get_block_num()):
            row = task // NH
            head = task % NH
            aa = a_g.current()
            sn = sn_g.current()
            cs = cs_g.current()
            rp = rope_g.current()
            cc = c_g.current()
            yy = y_g.current()
            pl.load(aa, qr, [row, head * (D + DR)])
            pl.load(sn, sin, [row, 0])
            pl.load(cs, cos, [row, 0])
            _rope_rows_f32(aa, cs, sn, rp, 1, D)
            pl.store(query_rope, rp, [row, head * DR])
            _uk_clear(cc)
            for k0 in pl.range(0, D, 32):
                ww = w_g.current()
                pl.load(ww, w, [head, k0, 0], order=[1, 2])
                _uk_chunk(aa, ww, cc, k0)
                w_g.next()
            _uk_pack(cc, yy)
            pl.store(query, yy, [row, head * HCKV])


# Per group: Mat 253952 B; each AIV Vec 172032 B; Left 4096 B,
# Right 57344 B, Acc 229376 B. Shared Mat slots alternate by logical K panel.
CP_KP = 64
CP_MB = 64
CP_NB = 896
CP_HALF = 448
CP_PRELOAD = 0


@pl.pipeline.stage
def _cp_load(ki, w, n0, width, nd_g, nz_g, shared):
    sid = pl.get_subblock_idx()
    part_cols = pl.min(CP_HALF, width - sid * CP_HALF)
    nd = nd_g.next()
    pl.set_validshape(nd, [CP_KP, part_cols])
    pl.load(nd, w, [ki * CP_KP, n0 + sid * CP_HALF])
    nz = nz_g.current()
    bm = shared.next()
    pl.set_validshape(nz, [CP_KP, part_cols])
    pl.move(nz, nd)
    pl.insert(bm, nz, [0, sid * CP_HALF])


@pl.pipeline.stage
def _cp_mm(
    ki, c_q0, c_q1, c_q2, n0, width, rows, a0_g, a1_g, a2_g, shared, l0a, l0b, acc_g
):
    a0 = a0_g.current()
    a1 = a1_g.current()
    a2 = a2_g.current()
    bm = shared.next()
    acc = acc_g.current()
    pl.set_validshape(a0, [rows, CP_KP])
    pl.set_validshape(a1, [rows, CP_KP])
    pl.set_validshape(a2, [rows, CP_KP])
    pl.set_validshape(bm, [CP_KP, width])
    pl.set_validshape(acc, [rows, width])
    pl.load(a0, c_q0, [0, ki * CP_KP])
    pl.load(a1, c_q1, [0, ki * CP_KP])
    pl.load(a2, c_q2, [0, ki * CP_KP])
    if ki == 0:
        pl.gemm(
            acc,
            a2,
            bm,
            l0a=l0a,
            l0b=l0b,
            m=rows,
            n=width,
            k=CP_KP,
            splitk=16,
            init=True,
        )
    else:
        pl.gemm(
            acc,
            a2,
            bm,
            l0a=l0a,
            l0b=l0b,
            m=rows,
            n=width,
            k=CP_KP,
            splitk=16,
            init=False,
        )
    pl.gemm(
        acc, a1, bm, l0a=l0a, l0b=l0b, m=rows, n=width, k=CP_KP, splitk=16, init=False
    )
    pl.gemm(
        acc, a0, bm, l0a=l0a, l0b=l0b, m=rows, n=width, k=CP_KP, splitk=16, init=False
    )


@pl.jit(
    arch="a5",
    auto_mutex=True,
    pipeline=pl.pipeline.PipelineConfig(preload=CP_PRELOAD),
    name="mla_prolog_cp_up",
)
def _cooperative_up(
    c_q0: pl.Tensor[[pl.DYNAMIC, HCQ], BF16],
    c_q1: pl.Tensor[[pl.DYNAMIC, HCQ], BF16],
    c_q2: pl.Tensor[[pl.DYNAMIC, HCQ], BF16],
    w: pl.Tensor[[HCQ, NH * (D + DR)], BF16],
    out: pl.Tensor[[pl.DYNAMIC, NH * (D + DR)], FP32],
):
    a0_g = pl.make_tile_group(
        type=pl.TileType(
            shape=[CP_MB, CP_KP], dtype=BF16, target_memory=pl.MemorySpace.Mat
        ),
        addrs=0,
        mutex_ids="auto",
        depth=1,
    )
    a1_g = pl.make_tile_group(
        type=pl.TileType(
            shape=[CP_MB, CP_KP], dtype=BF16, target_memory=pl.MemorySpace.Mat
        ),
        addrs=CP_MB * CP_KP * 2,
        mutex_ids="auto",
        depth=1,
    )
    a2_g = pl.make_tile_group(
        type=pl.TileType(
            shape=[CP_MB, CP_KP], dtype=BF16, target_memory=pl.MemorySpace.Mat
        ),
        addrs=CP_MB * CP_KP * 4,
        mutex_ids="auto",
        depth=1,
    )
    shared = pl.make_tile_group(
        type=pl.TileType(
            shape=[CP_KP, CP_NB], dtype=BF16, target_memory=pl.MemorySpace.Mat
        ),
        addrs=CP_MB * CP_KP * 6,
        mutex_ids="auto",
        depth=2,
        fwd_ids=[0, 1],
        bwd_ids=[2, 3],
    )
    nd_g = pl.make_tile_group(
        type=pl.TileType(
            shape=[CP_KP, CP_HALF], dtype=BF16, target_memory=pl.MemorySpace.Vec
        ),
        addrs=0,
        mutex_ids="auto",
        depth=2,
    )
    nz_g = pl.make_tile_group(
        type=pl.TileType(
            shape=[CP_KP, CP_HALF],
            dtype=BF16,
            target_memory=pl.MemorySpace.Vec,
            layout=pl.NZ,
        ),
        addrs=CP_KP * CP_HALF * 4,
        mutex_ids="auto",
        depth=1,
    )
    l0a = pl.make_tile_group(
        type=pl.TileType(
            shape=[CP_MB, 16], dtype=BF16, target_memory=pl.MemorySpace.Left, compact=1
        ),
        addrs=0,
        mutex_ids="auto",
        depth=2,
    )
    l0b = pl.make_tile_group(
        type=pl.TileType(
            shape=[16, CP_NB], dtype=BF16, target_memory=pl.MemorySpace.Right, compact=1
        ),
        addrs=0,
        mutex_ids="auto",
        depth=2,
    )
    acc_g = pl.make_tile_group(
        type=pl.TileType(
            shape=[CP_MB, CP_NB],
            dtype=FP32,
            target_memory=pl.MemorySpace.Acc,
            compact=1,
        ),
        addrs=0,
        mutex_ids="auto",
        depth=1,
    )
    gid = pl.get_block_idx() // pl.get_subblock_num()
    n0 = ((NH * (D + DR) * gid) // (CUBE_WORKERS * 64)) * 64
    n1 = ((NH * (D + DR) * (gid + 1)) // (CUBE_WORKERS * 64)) * 64
    width = n1 - n0
    rows = out.shape[0]
    for ki in pl.range(HCQ // CP_KP):
        with pl.section_vector():
            _cp_load(ki, w, n0, width, nd_g, nz_g, shared)
        with pl.section_cube():
            _cp_mm(
                ki,
                c_q0,
                c_q1,
                c_q2,
                n0,
                width,
                rows,
                a0_g,
                a1_g,
                a2_g,
                shared,
                l0a,
                l0b,
                acc_g,
            )
    with pl.section_cube():
        final_acc = acc_g.current()
        pl.set_validshape(final_acc, [rows, width])
        pl.store(out, final_acc, [0, n0])


@lru_cache(maxsize=None)
def _kernels(he, vector_up=False):
    rms_cq = _rmsnorm_full_vf(HCQ) if vector_up else _rmsnorm_vf(HCQ)
    cq_dtype = FP32 if vector_up else BF16
    rms_kv = _rmsnorm_vf(HCKV)

    # ------------------------------------------------------------------ down
    def make_down(nb, ks, bdepth, tag):
        @pl.jit(arch="a5", auto_mutex=True, name=f"mla_prolog_down_{tag}_{he}")
        def down(
            x: pl.Tensor[[pl.DYNAMIC, he], BF16],
            w_dq: pl.Tensor[[he, HCQ], BF16],
            w_dkv_kr: pl.Tensor[[he, NKR], BF16],
            cq_raw: pl.Tensor[[DOWN_NS, pl.DYNAMIC, HCQ], FP32],
            dkv_raw: pl.Tensor[[DOWN_NS, pl.DYNAMIC, NKR], FP32],
        ):
            a_g = pl.make_tile_group(
                type=pl.TileType(
                    shape=[DOWN_MB, DOWN_KP],
                    dtype=BF16,
                    target_memory=pl.MemorySpace.Mat,
                ),
                addrs=0,
                mutex_ids="auto",
                depth=2,
            )
            b_g = pl.make_tile_group(
                type=pl.TileType(
                    shape=[DOWN_KP, nb], dtype=BF16, target_memory=pl.MemorySpace.Mat
                ),
                addrs=2 * DOWN_ABYTES,
                mutex_ids="auto",
                depth=bdepth,
            )
            l0a = pl.make_tile_group(
                type=pl.TileType(
                    shape=[DOWN_MB, ks],
                    dtype=BF16,
                    target_memory=pl.MemorySpace.Left,
                    compact=1,
                ),
                addrs=0,
                mutex_ids="auto",
                depth=2,
            )
            l0b = pl.make_tile_group(
                type=pl.TileType(
                    shape=[ks, nb],
                    dtype=BF16,
                    target_memory=pl.MemorySpace.Right,
                    compact=1,
                ),
                addrs=0,
                mutex_ids="auto",
                depth=2,
            )
            acc_g = pl.make_tile_group(
                type=pl.TileType(
                    shape=[DOWN_MB, nb],
                    dtype=FP32,
                    target_memory=pl.MemorySpace.Acc,
                    compact=1,
                ),
                addrs=0,
                mutex_ids="auto",
                depth=1,
            )
            with pl.section_cube():
                rows_total = x.shape[0]
                m_tiles = (rows_total + DOWN_MB - 1) // DOWN_MB
                steps = he // DOWN_KP
                per_tile = m_tiles * DOWN_NS
                for task in pl.range(
                    pl.get_block_idx(),
                    ((HCQ + NKR) // nb) * per_tile,
                    pl.get_block_num(),
                ):
                    ti = task // per_tile
                    part = task % per_tile
                    ns = part // m_tiles
                    m0 = (part % m_tiles) * DOWN_MB
                    col = ti * nb
                    rows = pl.min(DOWN_MB, rows_total - m0)
                    k0 = ((steps * ns) // DOWN_NS) * DOWN_KP
                    k1 = ((steps * (ns + 1)) // DOWN_NS) * DOWN_KP
                    cur_acc = acc_g.current()
                    for kp in pl.range(k0, k1, DOWN_KP):
                        a = a_g.next()
                        b = b_g.next()
                        pl.set_validshape(a, [rows, DOWN_KP])
                        pl.set_validshape(b, [DOWN_KP, nb])
                        pl.load(a, x, [m0, kp])
                        if col < HCQ:
                            pl.load(b, w_dq, [kp, col])
                        else:
                            pl.load(b, w_dkv_kr, [kp, col - HCQ])
                        if kp == k0:
                            pl.gemm(
                                cur_acc,
                                a,
                                b,
                                l0a=l0a,
                                l0b=l0b,
                                m=rows,
                                n=nb,
                                k=DOWN_KP,
                                splitk=ks,
                                init=True,
                            )
                        else:
                            pl.gemm(
                                cur_acc,
                                a,
                                b,
                                l0a=l0a,
                                l0b=l0b,
                                m=rows,
                                n=nb,
                                k=DOWN_KP,
                                splitk=ks,
                                init=False,
                            )
                    if col < HCQ:
                        pl.store(cq_raw, cur_acc, [ns, m0, col], order=[1, 2])
                    else:
                        pl.store(dkv_raw, cur_acc, [ns, m0, col - HCQ], order=[1, 2])

        return down

    down_small = make_down(DOWN_NB_SMALL, 128, 5, "s")
    down_big = make_down(DOWN_NB_BIG, 64, 2, "b")

    # ------------------------------------------------------------------ norm
    @pl.jit(arch="a5", auto_mutex=True, name="mla_prolog_norm")
    def norm(
        cq_raw: pl.Tensor[[DOWN_NS, pl.DYNAMIC, HCQ], FP32],
        dkv_raw: pl.Tensor[[DOWN_NS, pl.DYNAMIC, NKR], FP32],
        gamma_cq: pl.Tensor[[1, HCQ], BF16],
        gamma_ckv: pl.Tensor[[1, HCKV], BF16],
        rope_sin: pl.Tensor[[pl.DYNAMIC, DR], BF16],
        rope_cos: pl.Tensor[[pl.DYNAMIC, DR], BF16],
        c_q0: pl.Tensor[[pl.DYNAMIC, HCQ], cq_dtype],
        c_q1: pl.Tensor[[pl.DYNAMIC, HCQ], BF16],
        c_q2: pl.Tensor[[pl.DYNAMIC, HCQ], BF16],
        c_kv: pl.Tensor[[pl.DYNAMIC, HCKV], BF16],
        k_rope: pl.Tensor[[pl.DYNAMIC, DR], BF16],
        eps_cq: pl.DT_FP32,
        eps_ckv: pl.DT_FP32,
    ):
        f32_1536 = pl.TileType(
            shape=[1, HCQ], dtype=FP32, target_memory=pl.MemorySpace.Vec
        )
        f32_576 = pl.TileType(
            shape=[1, NKR], dtype=FP32, target_memory=pl.MemorySpace.Vec
        )
        bf16_1536 = pl.TileType(
            shape=[1, HCQ], dtype=BF16, target_memory=pl.MemorySpace.Vec
        )
        bf16_512 = pl.TileType(
            shape=[1, HCKV], dtype=BF16, target_memory=pl.MemorySpace.Vec
        )
        bf16_64 = pl.TileType(
            shape=[1, DR], dtype=BF16, target_memory=pl.MemorySpace.Vec
        )

        # Explicit byte plan; reserve FP32 capacity for the first query component.
        a_cq = 0
        a_cq1 = a_cq + HCQ * 4
        a_cq2 = a_cq1 + HCQ * 4
        a_cq3 = a_cq2 + HCQ * 4
        a_c0 = a_cq3 + HCQ * 4
        a_c1 = a_c0 + HCQ * 4
        a_c2 = a_c1 + HCQ * 2
        a_gc = a_c2 + HCQ * 2
        a_dk = a_gc + HCQ * 2
        a_dk1 = a_dk + NKR * 4
        a_dk2 = a_dk1 + NKR * 4
        a_dk3 = a_dk2 + NKR * 4
        a_kv = a_dk3 + NKR * 4
        a_kv1 = a_kv + HCKV * 2
        a_kv2 = a_kv1 + HCKV * 2
        a_gk = a_kv2 + HCKV * 2
        a_sn = a_gk + HCKV * 2
        a_cs = a_sn + DR * 2
        a_kro = a_cs + DR * 2

        cq_g = pl.make_tile_group(type=f32_1536, addrs=a_cq, mutex_ids="auto", depth=1)
        cq1_g = pl.make_tile_group(
            type=f32_1536, addrs=a_cq1, mutex_ids="auto", depth=1
        )
        cq2_g = pl.make_tile_group(
            type=f32_1536, addrs=a_cq2, mutex_ids="auto", depth=1
        )
        cq3_g = pl.make_tile_group(
            type=f32_1536, addrs=a_cq3, mutex_ids="auto", depth=1
        )
        c0_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, HCQ], dtype=cq_dtype, target_memory=pl.MemorySpace.Vec
            ),
            addrs=a_c0,
            mutex_ids="auto",
            depth=1,
        )
        c1_g = pl.make_tile_group(type=bf16_1536, addrs=a_c1, mutex_ids="auto", depth=1)
        c2_g = pl.make_tile_group(type=bf16_1536, addrs=a_c2, mutex_ids="auto", depth=1)
        gc_g = pl.make_tile_group(type=bf16_1536, addrs=a_gc, mutex_ids="auto", depth=1)
        dk_g = pl.make_tile_group(type=f32_576, addrs=a_dk, mutex_ids="auto", depth=1)
        dk1_g = pl.make_tile_group(type=f32_576, addrs=a_dk1, mutex_ids="auto", depth=1)
        dk2_g = pl.make_tile_group(type=f32_576, addrs=a_dk2, mutex_ids="auto", depth=1)
        dk3_g = pl.make_tile_group(type=f32_576, addrs=a_dk3, mutex_ids="auto", depth=1)
        kv_g = pl.make_tile_group(type=bf16_512, addrs=a_kv, mutex_ids="auto", depth=1)
        kv1_g = pl.make_tile_group(
            type=bf16_512, addrs=a_kv1, mutex_ids="auto", depth=1
        )
        kv2_g = pl.make_tile_group(
            type=bf16_512, addrs=a_kv2, mutex_ids="auto", depth=1
        )
        gk_g = pl.make_tile_group(type=bf16_512, addrs=a_gk, mutex_ids="auto", depth=1)
        sn_g = pl.make_tile_group(type=bf16_64, addrs=a_sn, mutex_ids="auto", depth=1)
        cs_g = pl.make_tile_group(type=bf16_64, addrs=a_cs, mutex_ids="auto", depth=1)
        kro_g = pl.make_tile_group(type=bf16_64, addrs=a_kro, mutex_ids="auto", depth=1)

        with pl.section_vector():
            rows_total = cq_raw.shape[1]
            gc = gc_g.current()
            gk = gk_g.current()
            pl.load(gc, gamma_cq, [0, 0])
            pl.load(gk, gamma_ckv, [0, 0])
            for row in pl.range(pl.get_block_idx(), rows_total, pl.get_block_num()):
                cq = cq_g.current()
                cq1 = cq1_g.current()
                cq2 = cq2_g.current()
                cq3 = cq3_g.current()
                c0 = c0_g.current()
                c1 = c1_g.current()
                c2 = c2_g.current()
                pl.load(cq, cq_raw, [0, row, 0], order=[1, 2])
                pl.load(cq1, cq_raw, [1, row, 0], order=[1, 2])
                pl.load(cq2, cq_raw, [2, row, 0], order=[1, 2])
                pl.load(cq3, cq_raw, [3, row, 0], order=[1, 2])
                _sum_rows(cq, cq1, cq2, cq3, HCQ)
                rms_cq(cq, gc, c0, c1, c2, eps_cq)
                pl.store(c_q0, c0, [row, 0])
                if not vector_up:
                    pl.store(c_q1, c1, [row, 0])
                    pl.store(c_q2, c2, [row, 0])

                dk = dk_g.current()
                dk1 = dk1_g.current()
                dk2 = dk2_g.current()
                dk3 = dk3_g.current()
                kv = kv_g.current()
                kv1 = kv1_g.current()
                kv2 = kv2_g.current()
                pl.load(dk, dkv_raw, [0, row, 0], order=[1, 2])
                pl.load(dk1, dkv_raw, [1, row, 0], order=[1, 2])
                pl.load(dk2, dkv_raw, [2, row, 0], order=[1, 2])
                pl.load(dk3, dkv_raw, [3, row, 0], order=[1, 2])
                _sum_rows(dk, dk1, dk2, dk3, NKR)
                rms_kv(dk, gk, kv, kv1, kv2, eps_ckv)
                pl.store(c_kv, kv, [row, 0])

                sn = sn_g.current()
                cs = cs_g.current()
                kro = kro_g.current()
                pl.load(sn, rope_sin, [row, 0])
                pl.load(cs, rope_cos, [row, 0])
                _rope_rows_f32(dk, cs, sn, kro, 1, HCKV)
                pl.store(k_rope, kro, [row, 0])

    # -------------------------------------------------------------------- up
    @pl.jit(arch="a5", auto_mutex=True, name="mla_prolog_up")
    def up(
        c_q0: pl.Tensor[[pl.DYNAMIC, HCQ], BF16],
        c_q1: pl.Tensor[[pl.DYNAMIC, HCQ], BF16],
        c_q2: pl.Tensor[[pl.DYNAMIC, HCQ], BF16],
        w_uq_qr: pl.Tensor[[HCQ, NH * (D + DR)], BF16],
        qr: pl.Tensor[[pl.DYNAMIC, NH * (D + DR)], FP32],
    ):
        a0_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[UP_MB, UP_KP], dtype=BF16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=UP_A0,
            mutex_ids="auto",
            depth=UP_ADEPTH,
        )
        a1_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[UP_MB, UP_KP], dtype=BF16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=UP_A1,
            mutex_ids="auto",
            depth=UP_ADEPTH,
        )
        a2_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[UP_MB, UP_KP], dtype=BF16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=UP_A2,
            mutex_ids="auto",
            depth=UP_ADEPTH,
        )
        b_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[UP_KP, UP_NB], dtype=BF16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=UP_B,
            mutex_ids="auto",
            depth=UP_BDEPTH,
        )
        l0a = pl.make_tile_group(
            type=pl.TileType(
                shape=[UP_MB, UP_KS],
                dtype=BF16,
                target_memory=pl.MemorySpace.Left,
                compact=1,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=2,
        )
        l0b = pl.make_tile_group(
            type=pl.TileType(
                shape=[UP_KS, UP_NB],
                dtype=BF16,
                target_memory=pl.MemorySpace.Right,
                compact=1,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=2,
        )
        acc_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[UP_MB, UP_NB],
                dtype=FP32,
                target_memory=pl.MemorySpace.Acc,
                compact=1,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        with pl.section_cube():
            rows_total = c_q0.shape[0]
            m_tiles = (rows_total + UP_MB - 1) // UP_MB
            for task in pl.range(
                pl.get_block_idx(), m_tiles * UP_NT, pl.get_block_num()
            ):
                ti = task // m_tiles
                m0 = (task % m_tiles) * UP_MB
                n0 = ti * UP_NB
                rows = pl.min(UP_MB, rows_total - m0)
                cur_acc = acc_g.current()
                for kp in pl.range(0, HCQ, UP_KP):
                    a0 = a0_g.next()
                    a1 = a1_g.next()
                    a2 = a2_g.next()
                    b = b_g.next()
                    pl.set_validshape(a0, [rows, UP_KP])
                    pl.set_validshape(a1, [rows, UP_KP])
                    pl.set_validshape(a2, [rows, UP_KP])
                    pl.set_validshape(b, [UP_KP, UP_NB])
                    pl.load(a0, c_q0, [m0, kp])
                    pl.load(a1, c_q1, [m0, kp])
                    pl.load(a2, c_q2, [m0, kp])
                    pl.load(b, w_uq_qr, [kp, n0])
                    if kp == 0:
                        pl.gemm(
                            cur_acc,
                            a2,
                            b,
                            l0a=l0a,
                            l0b=l0b,
                            m=rows,
                            n=UP_NB,
                            k=UP_KP,
                            splitk=UP_KS,
                            init=True,
                        )
                    else:
                        pl.gemm(
                            cur_acc,
                            a2,
                            b,
                            l0a=l0a,
                            l0b=l0b,
                            m=rows,
                            n=UP_NB,
                            k=UP_KP,
                            splitk=UP_KS,
                            init=False,
                        )
                    pl.gemm(
                        cur_acc,
                        a1,
                        b,
                        l0a=l0a,
                        l0b=l0b,
                        m=rows,
                        n=UP_NB,
                        k=UP_KP,
                        splitk=UP_KS,
                        init=False,
                    )
                    pl.gemm(
                        cur_acc,
                        a0,
                        b,
                        l0a=l0a,
                        l0b=l0b,
                        m=rows,
                        n=UP_NB,
                        k=UP_KP,
                        splitk=UP_KS,
                        init=False,
                    )
                pl.store(qr, cur_acc, [m0, n0])

    # ------------------------------------------------------------- up (slab)

    # ----------------------------------------------------------------- split
    @pl.jit(arch="a5", auto_mutex=True, name="mla_prolog_split")
    def split(
        qr: pl.Tensor[[pl.DYNAMIC, NH * (D + DR)], FP32],
        rope_sin: pl.Tensor[[pl.DYNAMIC, DR], BF16],
        rope_cos: pl.Tensor[[pl.DYNAMIC, DR], BF16],
        qc0: pl.Tensor[[pl.DYNAMIC, NH * D], BF16],
        qc1: pl.Tensor[[pl.DYNAMIC, NH * D], BF16],
        qc2: pl.Tensor[[pl.DYNAMIC, NH * D], BF16],
        query_rope: pl.Tensor[[pl.DYNAMIC, NH * DR], BF16],
    ):
        f32_hb = pl.TileType(
            shape=[1, HB * (D + DR)], dtype=FP32, target_memory=pl.MemorySpace.Vec
        )
        bf16_hb = pl.TileType(
            shape=[1, HB * D], dtype=BF16, target_memory=pl.MemorySpace.Vec
        )
        bf16_rb = pl.TileType(
            shape=[1, HB * DR], dtype=BF16, target_memory=pl.MemorySpace.Vec
        )
        bf16_64 = pl.TileType(
            shape=[1, DR], dtype=BF16, target_memory=pl.MemorySpace.Vec
        )
        a_x = 0
        a_c0 = a_x + HB * (D + DR) * 4
        a_c1 = a_c0 + HB * D * 2
        a_c2 = a_c1 + HB * D * 2
        a_qr = a_c2 + HB * D * 2
        a_sn = a_qr + HB * DR * 2
        a_cs = a_sn + DR * 2
        x_g = pl.make_tile_group(type=f32_hb, addrs=a_x, mutex_ids="auto", depth=1)
        c0_g = pl.make_tile_group(type=bf16_hb, addrs=a_c0, mutex_ids="auto", depth=1)
        c1_g = pl.make_tile_group(type=bf16_hb, addrs=a_c1, mutex_ids="auto", depth=1)
        c2_g = pl.make_tile_group(type=bf16_hb, addrs=a_c2, mutex_ids="auto", depth=1)
        qr_g = pl.make_tile_group(type=bf16_rb, addrs=a_qr, mutex_ids="auto", depth=1)
        sn_g = pl.make_tile_group(type=bf16_64, addrs=a_sn, mutex_ids="auto", depth=1)
        cs_g = pl.make_tile_group(type=bf16_64, addrs=a_cs, mutex_ids="auto", depth=1)
        with pl.section_vector():
            rows_total = qr.shape[0]
            blocks = NH // HB
            for task in pl.range(
                pl.get_block_idx(), rows_total * blocks, pl.get_block_num()
            ):
                row = task // blocks
                h0 = (task % blocks) * HB
                x = x_g.current()
                c0 = c0_g.current()
                c1 = c1_g.current()
                c2 = c2_g.current()
                yb = qr_g.current()
                sn = sn_g.current()
                cs = cs_g.current()
                pl.load(x, qr, [row, h0 * (D + DR)])
                pl.load(sn, rope_sin, [row, 0])
                pl.load(cs, rope_cos, [row, 0])
                _split_rope(x, cs, sn, c0, c1, c2, yb)
                pl.store(qc0, c0, [row, h0 * D])
                pl.store(qc1, c1, [row, h0 * D])
                pl.store(qc2, c2, [row, h0 * D])
                pl.store(query_rope, yb, [row, h0 * DR])

    # -------------------------------------------------------------------- uk
    @pl.jit(arch="a5", auto_mutex=True, name="mla_prolog_uk")
    def uk(
        qc0: pl.Tensor[[pl.DYNAMIC, NH * D], BF16],
        qc1: pl.Tensor[[pl.DYNAMIC, NH * D], BF16],
        qc2: pl.Tensor[[pl.DYNAMIC, NH * D], BF16],
        w_uk: pl.Tensor[[NH, D, HCKV], BF16],
        query: pl.Tensor[[pl.DYNAMIC, NH * HCKV], BF16],
    ):
        a0_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[UK_MB, D], dtype=BF16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=UK_A0,
            mutex_ids="auto",
            depth=UK_ADEPTH,
        )
        a1_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[UK_MB, D], dtype=BF16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=UK_A1,
            mutex_ids="auto",
            depth=UK_ADEPTH,
        )
        a2_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[UK_MB, D], dtype=BF16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=UK_A2,
            mutex_ids="auto",
            depth=UK_ADEPTH,
        )
        b_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[D, UK_NB], dtype=BF16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=UK_B,
            mutex_ids="auto",
            depth=UK_BDEPTH,
        )
        l0a = pl.make_tile_group(
            type=pl.TileType(
                shape=[UK_MB, UK_KS],
                dtype=BF16,
                target_memory=pl.MemorySpace.Left,
                compact=1,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=2,
        )
        l0b = pl.make_tile_group(
            type=pl.TileType(
                shape=[UK_KS, UK_NB],
                dtype=BF16,
                target_memory=pl.MemorySpace.Right,
                compact=1,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=2,
        )
        acc_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[UK_MB, UK_NB],
                dtype=FP32,
                target_memory=pl.MemorySpace.Acc,
                compact=1,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        with pl.section_cube():
            rows_total = qc0.shape[0]
            m_tiles = (rows_total + UK_MB - 1) // UK_MB
            per_head = m_tiles * UK_NS
            for task in pl.range(pl.get_block_idx(), NH * per_head, pl.get_block_num()):
                h = task // per_head
                part = task % per_head
                m0 = (part // UK_NS) * UK_MB
                n0 = (part % UK_NS) * UK_NB
                rows = pl.min(UK_MB, rows_total - m0)
                a0 = a0_g.next()
                a1 = a1_g.next()
                a2 = a2_g.next()
                b = b_g.next()
                cur_acc = acc_g.current()
                pl.set_validshape(a0, [rows, D])
                pl.set_validshape(a1, [rows, D])
                pl.set_validshape(a2, [rows, D])
                pl.set_validshape(b, [D, UK_NB])
                pl.load(a0, qc0, [m0, h * D])
                pl.load(a1, qc1, [m0, h * D])
                pl.load(a2, qc2, [m0, h * D])
                pl.load(b, w_uk, [h, 0, n0], order=[1, 2])
                pl.gemm(
                    cur_acc,
                    a2,
                    b,
                    l0a=l0a,
                    l0b=l0b,
                    m=rows,
                    n=UK_NB,
                    k=D,
                    splitk=UK_KS,
                    init=True,
                )
                pl.gemm(
                    cur_acc,
                    a1,
                    b,
                    l0a=l0a,
                    l0b=l0b,
                    m=rows,
                    n=UK_NB,
                    k=D,
                    splitk=UK_KS,
                    init=False,
                )
                pl.gemm(
                    cur_acc,
                    a0,
                    b,
                    l0a=l0a,
                    l0b=l0b,
                    m=rows,
                    n=UK_NB,
                    k=D,
                    splitk=UK_KS,
                    init=False,
                )
                pl.store(query, cur_acc, [m0, h * HCKV + n0])

    return down_small, down_big, norm, up, split, uk


def mla_prolog(
    token_x: torch.Tensor,
    w_dq: torch.Tensor,
    w_uq_qr: torch.Tensor,
    w_uk: torch.Tensor,
    w_dkv_kr: torch.Tensor,
    rmsnorm_gamma_cq: torch.Tensor,
    rmsnorm_gamma_ckv: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    n_heads: int,
    rmsnorm_epsilon_cq: float = 1e-5,
    rmsnorm_epsilon_ckv: float = 1e-5,
):
    """CANN-Bench ``mla_prolog`` entry (see module docstring for the contract)."""
    b, s, he = token_x.shape
    if he not in HE_CHOICES:
        raise ValueError(f"unsupported hidden size {he}")
    if int(n_heads) != NH:
        raise ValueError(f"unsupported n_heads {n_heads}")
    if tuple(w_dq.shape) != (he, HCQ) or tuple(w_dkv_kr.shape) != (he, NKR):
        raise ValueError("weight shape mismatch")
    if tuple(w_uq_qr.shape) != (HCQ, NH * (D + DR)):
        raise ValueError("w_uq_qr shape mismatch")
    if tuple(w_uk.shape) != (NH, D, HCKV):
        raise ValueError("w_uk shape mismatch")

    m = b * s
    dev = token_x.device
    bf16 = token_x.dtype

    x2 = token_x.reshape(m, he)
    sin2 = rope_sin.reshape(m, DR)
    cos2 = rope_cos.reshape(m, DR)

    cq_raw = torch.empty((DOWN_NS, m, HCQ), dtype=torch.float32, device=dev)
    dkv_raw = torch.empty((DOWN_NS, m, NKR), dtype=torch.float32, device=dev)
    c_q0 = torch.empty(
        (m, HCQ), dtype=torch.float32 if m in (1, 4) else bf16, device=dev
    )
    c_q1 = torch.empty((m, HCQ), dtype=bf16, device=dev)
    c_q2 = torch.empty((m, HCQ), dtype=bf16, device=dev)
    qr = torch.empty((m, NH * (D + DR)), dtype=torch.float32, device=dev)
    qc0 = torch.empty((m, NH * D), dtype=bf16, device=dev)
    qc1 = torch.empty((m, NH * D), dtype=bf16, device=dev)
    qc2 = torch.empty((m, NH * D), dtype=bf16, device=dev)
    query = torch.empty((m, NH * HCKV), dtype=bf16, device=dev)
    query_rope = torch.empty((m, NH * DR), dtype=bf16, device=dev)
    c_kv = torch.empty((m, HCKV), dtype=bf16, device=dev)
    k_rope = torch.empty((m, DR), dtype=bf16, device=dev)

    down_small, down_big, norm, up, split, uk = _kernels(he, m in (1, 4))
    down = down_big if m > 32 else down_small  # wide tiles pay off past 32 tokens
    down[None, CUBE_WORKERS](x2, w_dq, w_dkv_kr, cq_raw, dkv_raw)
    norm[None, VEC_WORKERS](
        cq_raw,
        dkv_raw,
        rmsnorm_gamma_cq.reshape(1, HCQ),
        rmsnorm_gamma_ckv.reshape(1, HCKV),
        sin2,
        cos2,
        c_q0,
        c_q1,
        c_q2,
        c_kv,
        k_rope,
        float(rmsnorm_epsilon_cq),
        float(rmsnorm_epsilon_ckv),
    )
    if m == 1:
        _up_vector[None, VEC_WORKERS](c_q0, w_uq_qr, qr)
    elif m == 4:
        _up_vector4[None, VEC_WORKERS](c_q0, w_uq_qr, qr)
    elif m <= CP_MB:
        _cooperative_up[None, CUBE_WORKERS](c_q0, c_q1, c_q2, w_uq_qr, qr)
    else:
        up[None, CUBE_WORKERS](c_q0, c_q1, c_q2, w_uq_qr, qr)
    if m == 1:
        _split_uk_vector[None, VEC_WORKERS](qr, w_uk, sin2, cos2, query, query_rope)
    else:
        split[None, VEC_WORKERS](qr, sin2, cos2, qc0, qc1, qc2, query_rope)
        uk[None, CUBE_WORKERS](qc0, qc1, qc2, w_uk, query)

    return (
        query.reshape(b, s, NH, HCKV),
        query_rope.reshape(b, s, NH, DR),
        c_kv.reshape(b, s, HCKV),
        k_rope.reshape(b, s, DR),
    )
