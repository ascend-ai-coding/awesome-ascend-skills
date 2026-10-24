# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始（VF）。Select a BF16 sortable-key cache without GM scores."""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

U16 = pl.DT_UINT16
U32 = pl.DT_UINT32
I32 = pl.DT_INT32
BF = pl.DT_BF16


@pl.vector_function
def clear_cache(cache, hist, width, written):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U16)
    z = vf.full(0, p, dtype=U16)
    for r in pl.range(3):
        for i in pl.range(written, width, 128):
            vf.store_align(cache, z, p, r * width + i)
    initial = vf.full(width - written, p, dtype=U16)
    for i in pl.range(0, 3 * 256, 128):
        vf.store_align(hist, initial, p, i)


@pl.vector_function
def widen_histogram(src, dst, row):
    p16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U16)
    p32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U32)
    lo = vf.load_align(src, row * 256)
    hi = vf.load_align(src, row * 256 + 128)
    e0 = vf.astype(lo, p16, layout=pl.CastLayout.ZERO, dtype=U32)
    o0 = vf.astype(lo, p16, layout=pl.CastLayout.ONE, dtype=U32)
    e1 = vf.astype(hi, p16, layout=pl.CastLayout.ZERO, dtype=U32)
    o1 = vf.astype(hi, p16, layout=pl.CastLayout.ONE, dtype=U32)
    vf.store_align(dst, e0, o0, p32, dist=pl.StoreDist.INTLV_B32)
    vf.store_align(dst + 128, e1, o1, p32, dist=pl.StoreDist.INTLV_B32)


@pl.vector_function
def emit_cached(
    indices,
    cache,
    out_idx,
    out_val,
    topk,
    base,
    begin,
    mapped=False,
    mapping=0,
    permuted=False,
    offset=0,
    write_values=True,
):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U32)
    zero = vf.full(0, p, dtype=U32)
    sign = vf.full(0x8000, p, dtype=U32)
    allbits = vf.full(0xFFFF, p, dtype=U32)
    neginf = vf.full(0xFF80, p, dtype=U32)
    negone = vf.full(-1, p, dtype=I32)
    for c in pl.range(0, topk, 64):
        active = vf.update_mask(pl.min(64, topk - c), dtype=U32)
        idx = vf.load_align(indices, c, dist=pl.LoadDist.UNPK_B16, dtype=U32)
        gather_indices = vf.adds(idx, base, p)
        raw = vf.gather(cache, gather_indices, active)
        key = vf.bit_cast(raw, dtype=U32)
        positive = vf.ge(key, sign, p)
        flip = vf.select(sign, allbits, positive)
        bits = vf.xor(key, flip, p)
        valid = vf.gt(key, zero, active)
        value = vf.select(bits, neginf, valid)
        original = vf.adds(idx, 0, p)
        if permuted:
            lowmask = vf.full(15, p, dtype=U32)
            middlemask = vf.full(112, p, dtype=U32)
            planemask = vf.full(128, p, dtype=U32)
            tilemask = vf.full(0xFFFFFF00, p, dtype=U32)
            low = vf.and_(idx, lowmask, p)
            middle = vf.and_(idx, middlemask, p)
            middle = vf.shift_left(middle, 1, p)
            plane = vf.and_(idx, planemask, p)
            plane = vf.shift_right(plane, 3, p)
            tile = vf.and_(idx, tilemask, p)
            original = vf.or_(low, middle, p)
            original = vf.or_(original, plane, p)
            original = vf.or_(original, tile, p)
        global_idx = vf.adds(original, begin, p)
        signed_idx = vf.bit_cast(global_idx, dtype=I32)
        if mapped:
            signed_idx = vf.gather(mapping, idx, active)
            valid = vf.ne(signed_idx, negone, valid)
        shifted_idx = vf.adds(signed_idx, offset, p)
        oi = vf.select(shifted_idx, negone, valid)
        vf.store_align(out_idx, oi, active, c)
        if write_values:
            vf.store_align(out_val, value, active, c, dist=pl.StoreDist.PACK_B32)
