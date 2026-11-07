# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ruff: noqa: F821
"""写法：语法糖。Engram residual gate with dual RMS and weighted-dot VF reductions.

BF16 x/key have shape [t,hc,D], BF16 value [t,D] and FP32 weight [hc,D].
x/weight and the optional BOOL mask[t] are contiguous on one NPU; key/value
allow nonoverlapping token strides with contiguous feature rows. Supported
metadata: t>=0, 1<=hc<=56, 1<=D<=8512, eps in [1e-12,1], clamp in [0,1].
Finite input magnitudes are bounded by 8. Masked tokens copy x bit for bit;
otherwise the independently defined gate in reference.py produces one final
BF16 residual rounding. Empty token inputs launch no device task.

Width/Slots/Exact/Mask are internal tiling keys. A worker retains one head's
weights and rotates across tokens; x/key/value stay in UB and key storage
becomes output after reduction. Peak UB is padded_width*(4+6*Slots), bounded
by 187264 bytes on the public route. There is no GM intermediate workspace.
"""

import math

import torch
import pypto
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf
from pypto_pro.runtime.platform import get_platform_info
from pypto_pro.runtime.tilingkey import TilingKeyField


class EngramKey:
    Width = TilingKeyField(
        bits=4, values=[64, 128, 256, 512, 1024, 2048, 4096, 5120, 5184, 8512]
    )
    Slots = TilingKeyField(bits=2, values=[1, 2, 3, 5])
    Exact = TilingKeyField(bits=1, values=[0, 1])
    Mask = TilingKeyField(bits=1, values=[0, 1])

    def is_valid(self, key):
        width, slots, _, _ = key
        return slots != 5 or width <= 5184


@pl.vector_function
def _gate(
    ux,
    uk,
    uw,
    uv,
    dim: pl.DT_INT64,
    inv_dim: pl.DT_FP32,
    scale: pl.DT_FP32,
    eps: pl.DT_FP32,
    clamp: pl.DT_FP32,
):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    p32u = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    ax = vf.full(0.0, dtype=pl.DT_FP32)
    ak = vf.full(0.0, dtype=pl.DT_FP32)
    ad = vf.full(0.0, dtype=pl.DT_FP32)
    for c in pl.range(dim // 64):
        off = c * 64
        xb = vf.load_align(ux, off, dist=pl.LoadDist.UNPK_B16)
        kb = vf.load_align(uk, off, dist=pl.LoadDist.UNPK_B16)
        x = vf.astype(xb, p, dtype=pl.DT_FP32)
        k = vf.astype(kb, p, dtype=pl.DT_FP32)
        w = vf.load_align(uw, off)
        ax = vf.mul_add_dst(x, x, p)
        ak = vf.mul_add_dst(k, k, p)
        xw = vf.mul(x, w, p)
        ad = vf.mul_add_dst(xw, k, p)
    if dim % 64 != 0:
        tail = vf.update_mask(dim % 64, dtype=pl.DT_FP32)
        off = (dim // 64) * 64
        xbt = vf.load_align(ux, off, dist=pl.LoadDist.UNPK_B16)
        kbt = vf.load_align(uk, off, dist=pl.LoadDist.UNPK_B16)
        xt = vf.astype(xbt, p, dtype=pl.DT_FP32)
        kt = vf.astype(kbt, p, dtype=pl.DT_FP32)
        wt = vf.load_align(uw, off)
        xx = vf.mul(xt, xt, tail)
        kk = vf.mul(kt, kt, tail)
        xwt = vf.mul(xt, wt, tail)
        xd = vf.mul(xwt, kt, tail)
        ax = vf.add(ax, xx, p)
        ak = vf.add(ak, kk, p)
        ad = vf.add(ad, xd, p)
    sx = vf.reduce_sum(ax, p)
    sk = vf.reduce_sum(ak, p)
    sd = vf.reduce_sum(ad, p)
    mx = vf.muls(sx, inv_dim, p)
    mk = vf.muls(sk, inv_dim, p)
    mx = vf.adds(mx, eps, p)
    mk = vf.adds(mk, eps, p)
    rx = vf.sqrt(mx, p)
    rk = vf.sqrt(mk, p)
    denom = vf.mul(rx, rk, p)
    sd = vf.muls(sd, scale, p)
    nd = vf.div(sd, denom, p)
    ab = vf.abs(nd, p)
    ab = vf.maxs(ab, clamp, p)
    mag = vf.sqrt(ab, p)
    neg = vf.lt(nd, 0.0, p)
    nmag = vf.muls(mag, -1.0, p)
    signed = vf.select(nmag, mag, neg)
    signed = vf.muls(signed, -1.0, p)
    ex = vf.exp(signed, p)
    den = vf.adds(ex, 1.0, p)
    one = vf.full(1.0, dtype=pl.DT_FP32)
    gate = vf.div(one, den, p)
    gate = vf.full(gate, p)
    for c in pl.range((dim + 63) // 64):
        off = c * 64
        oxb = vf.load_align(ux, off, dist=pl.LoadDist.UNPK_B16)
        vb = vf.load_align(uv, c * 64, dist=pl.LoadDist.UNPK_B16)
        ox = vf.astype(oxb, p, dtype=pl.DT_FP32)
        val = vf.astype(vb, p, dtype=pl.DT_FP32)
        update = vf.mul(val, gate, p)
        result = vf.add(ox, update, p)
        ob = vf.astype(
            result, p16, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT
        )
        packed = vf.bit_cast(ob, dtype=pl.DT_UINT32)
        vf.store_align(uk, packed, p32u, off, dist=pl.StoreDist.PACK)


@pl.jit(auto_mutex=True, tiling_key=EngramKey)
def engram_gate_kernel(
    x_ptr: pl.Ptr[pl.DT_UINT8],
    k_ptr: pl.Ptr[pl.DT_UINT8],
    v_ptr: pl.Ptr[pl.DT_UINT8],
    w_ptr: pl.Ptr[pl.DT_UINT8],
    m_ptr: pl.Ptr[pl.DT_UINT8],
    y_ptr: pl.Ptr[pl.DT_UINT8],
    tokens: pl.DT_INT64,
    hc: pl.DT_INT64,
    dim: pl.DT_INT64,
    inv_dim: pl.DT_FP32,
    scale: pl.DT_FP32,
    eps: pl.DT_FP32,
    clamp: pl.DT_FP32,
    key_stride: pl.DT_INT64,
    value_stride: pl.DT_INT64,
):
    # Each worker owns one head and strides tokens; weight stays in UB.
    # Peak UB = Width * (4 + 6*Slots) bytes; no L1/L0/workspace allocation.
    # K is reused for Y only after the entire reduction; X remains resident.
    # No cross-worker dependency or GM intermediate is needed. Finite inputs in
    # [-8, 8], eps >= 1e-12: RMS product is >= eps; all sums fit in FP32.
    # BF16 element i occupies low 16 bits of b32 lane i after UNPK_B16;
    # FP32 predicates select that lane, and PACK of UINT32 emits dense BF16.
    dtype = pl.DT_BF16
    x = pl.make_tensor(x_ptr, [tokens, hc, dim], dtype=dtype)
    key = pl.make_tensor(k_ptr, [tokens, hc, dim], [key_stride, dim, 1], dtype=dtype)
    value = pl.make_tensor(v_ptr, [tokens, dim], [value_stride, 1], dtype=dtype)
    weight = pl.make_tensor(w_ptr, [hc, dim], dtype=pl.DT_FP32)
    mask = pl.make_tensor(m_ptr, [tokens], dtype=pl.DT_BOOL)
    y = pl.make_tensor(y_ptr, [tokens, hc, dim], dtype=dtype)
    wg_type = pl.TileType(
        shape=[1, Width], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
    )
    wg = pl.make_tile_group(type=wg_type, addrs=0, mutex_ids="auto", depth=1)
    xg_type = pl.TileType(
        shape=[1, Width], dtype=dtype, target_memory=pl.MemorySpace.Vec
    )
    xg = pl.make_tile_group(
        type=xg_type, addrs=4 * Width, mutex_ids="auto", depth=Slots
    )
    kg_type = pl.TileType(
        shape=[1, Width], dtype=dtype, target_memory=pl.MemorySpace.Vec
    )
    kg = pl.make_tile_group(
        type=kg_type, addrs=4 * Width + 2 * Slots * Width, mutex_ids="auto", depth=Slots
    )
    vg_type = pl.TileType(
        shape=[1, Width], dtype=dtype, target_memory=pl.MemorySpace.Vec
    )
    vg = pl.make_tile_group(
        type=vg_type, addrs=4 * Width + 4 * Slots * Width, mutex_ids="auto", depth=Slots
    )
    with pl.section_vector():
        groups = hc
        core = pl.get_block_idx()
        h0 = core % groups
        uw = wg[0]
        pl.set_validshape(uw, [1, dim])
        pl.load(uw, weight, [h0, 0])
        for token in pl.range(core // groups, tokens, pl.get_block_num() // groups):
            ux = xg.next()
            uk = kg.next()
            uv = vg.next()
            pl.set_validshape(ux, [1, dim])
            pl.load(ux, x, [token, h0, 0])
            masked = 0
            if Mask:
                masked = pl.getval(mask, token)
            if masked != 0:
                pl.store(y, ux, [token, h0, 0])
            else:
                pl.set_validshape(uk, [1, dim])
                pl.set_validshape(uv, [1, dim])
                pl.load(uk, key, [token, h0, 0])
                pl.load(uv, value, [token, 0])
                if Exact:
                    _gate(ux, uk, uw, uv, Width, inv_dim, scale, eps, clamp)
                else:
                    _gate(ux, uk, uw, uv, dim, inv_dim, scale, eps, clamp)
                pl.store(y, uk, [token, h0, 0])


def _launch(x, key, value, weight, image_mask, y, eps, clamp_value, slots=None):
    t, hc, dim = x.shape
    width = next(n for n in EngramKey.Width.values if n >= dim)
    nc = int(get_platform_info().vector_core_num)
    workers = min(nc, t * hc)
    workers -= workers % hc
    if slots is None:
        slots = 1 if t * hc <= nc else (5 if t >= 512 and width <= 5184 else 3)
    route = dict(
        Width=width,
        Slots=slots,
        Exact=int(dim == width),
        Mask=int(image_mask is not None),
    )
    with pypto.options(pass_options={"enable_slice": False}):
        engram_gate_kernel[None, workers, route](
            x,
            key,
            value,
            weight,
            x if image_mask is None else image_mask,
            y,
            t,
            hc,
            dim,
            1.0 / dim,
            dim**-0.5,
            eps,
            clamp_value,
            key.stride(0),
            value.stride(0),
        )


def engram_gate_wrapper(
    x, key, value, weight, image_mask=None, eps=1e-6, clamp_value=1e-6
):
    """Return Engram residual output, preserving the activation dtype."""
    if x.ndim != 3 or x.dtype != torch.bfloat16:
        raise ValueError("x must be a rank-3 BF16 tensor")
    t, hc, dim = x.shape
    if not (1 <= hc <= 56 and 1 <= dim <= 8512):
        raise ValueError("supported dimensions: 1 <= hc <= 56, 1 <= dim <= 8512")
    for tensor, shape, dtype in (
        (key, x.shape, x.dtype),
        (value, (t, dim), x.dtype),
        (weight, (hc, dim), torch.float32),
    ):
        if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
            raise ValueError("key/value/weight shape or dtype mismatch")
    tensors = [x, weight]
    if image_mask is not None:
        if tuple(image_mask.shape) != (t,) or image_mask.dtype != torch.bool:
            raise ValueError("image_mask must be a bool tensor of shape [t]")
        tensors.append(image_mask)
    if key.device != x.device or value.device != x.device:
        raise ValueError("key/value must be on the same NPU as x")
    if key.stride(2) != 1 or key.stride(1) != dim or key.stride(0) < hc * dim:
        raise ValueError(
            "key requires contiguous head rows and a non-overlapping token stride"
        )
    if value.stride(1) != 1 or value.stride(0) < dim:
        raise ValueError(
            "value requires contiguous rows and a non-overlapping token stride"
        )
    if x.device.type != "npu" or any(
        a.device != x.device or not a.is_contiguous() for a in tensors
    ):
        raise ValueError(
            "x, weight and mask must be contiguous tensors on the same NPU"
        )
    if not (
        math.isfinite(eps)
        and 1e-12 <= eps <= 1
        and math.isfinite(clamp_value)
        and 0 <= clamp_value <= 1
    ):
        raise ValueError("eps must be in [1e-12,1] and clamp_value in [0,1]")
    y = torch.empty_like(x)
    if t:
        _launch(x, key, value, weight, image_mask, y, eps, clamp_value)
    return y
