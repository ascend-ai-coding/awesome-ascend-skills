# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：语法糖。 Bounded on-chip front end for the ordinary 2-D matrix-chain reverse pass.

The mathematical/input/output contract and hi+lo representation are those of
kernel.py. M is a multiple of 64 in [64,1024]; K/N/D are multiples of 64 in
[64,256]. A query tile has 16 or 32 rows, divided equally between its AIVs.
The automatic launch count and preload=0 keep the serial C->V ownership clear.

One native task forms S=A@B and dP=G@C.T in Cube, transfers each FP32 Acc
result directly to its paired AIVs and computes FP32 P/dS in VF. Only the four
low-dtype component matrices reach GM (8*M*N bytes); the three unchanged
complete-grid Cube gradient owners then finish dA/dB/dC in three more tasks.
There are no gradient partials, atomics or merge tasks.

Every row of each GM component has one query-tile/AIV writer. Each final
gradient tile keeps the unique owner from kernel.py. The shared score/gradient
Vec groups are read-only in VF and have distinct forward/backward event pairs;
their final VF readers release them before the next query tile. The temporary
Acc is reused after each FIX transfer, with its last reader covered by the
ordinary automatic mutex. Local P and component Vec tiles are reused after
their final VF or MTE3 readers. All groups have one slot and explicit addresses.

For R=query rows, H=R/2, physical Mat bytes are
2*(R+N)*(K+D), at most 294912 bytes. Left is 128*R bytes, Right is 128*N
bytes, Acc is 4*R*N bytes and Vec is 20*H*N bytes (at most 81920).
These are separate spaces; no physical allocation aliases another live tile.
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

from .kernel import _make_product

FP32 = pl.DT_FP32


def make_onchip(rows, features, columns, channels, dtype_name="float16", query_rows=16):
    if type(rows) is not int or not 64 <= rows <= 1024 or rows % 64:
        raise ValueError("M must be a multiple of 64 in [64,1024]")
    if any(type(size) is not int or not 64 <= size <= 256 or size % 64 for size in (features, columns, channels)):
        raise ValueError("K, N and D must be multiples of 64 in [64,256]")
    if query_rows not in (16, 32) or type(query_rows) is not int:
        raise ValueError("query_rows must be 16 or 32")
    if dtype_name not in ("float16", "bfloat16"):
        raise ValueError("dtype_name must be float16 or bfloat16")
    dtype = pl.DT_FP16 if dtype_name == "float16" else pl.DT_BF16
    half = query_rows // 2

    @pl.vector_function
    def normalize_rows(score, p, high, low):
        mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        for row in pl.range(half):
            lane_max = vf.full(-1.0e30, mask, dtype=FP32)
            for chunk in pl.range(columns // 64):
                value = vf.load_align(score, [row, chunk * 64])
                lane_max = vf.max(lane_max, value, mask)
            maximum = vf.reduce_max(lane_max, mask)
            peak = vf.full(maximum, mask)
            lane_sum = vf.full(0.0, mask, dtype=FP32)
            for chunk in pl.range(columns // 64):
                value = vf.load_align(score, [row, chunk * 64])
                shifted = vf.sub(value, peak, mask)
                exponential = vf.exp(shifted, mask)
                lane_sum = vf.add(lane_sum, exponential, mask)
            total = vf.reduce_sum(lane_sum, mask)
            denominator = vf.full(total, mask)
            for chunk in pl.range(columns // 64):
                value = vf.load_align(score, [row, chunk * 64])
                shifted = vf.sub(value, peak, mask)
                exponential = vf.exp(shifted, mask)
                probability = vf.div(exponential, denominator, mask)
                vf.store_align(p, probability, mask, [row, chunk * 64])
                hi = vf.astype(probability, mask, dtype=dtype, layout=pl.CastLayout.ZERO)
                widened = vf.astype(hi, mask, dtype=FP32, layout=pl.CastLayout.ZERO)
                remainder = vf.sub(probability, widened, mask)
                lo = vf.astype(remainder, mask, dtype=dtype, layout=pl.CastLayout.ZERO)
                vf.store_align(high, hi, mask, [row, chunk * 64], dist=pl.StoreDist.PACK_B32)
                vf.store_align(low, lo, mask, [row, chunk * 64], dist=pl.StoreDist.PACK_B32)

    @pl.vector_function
    def reverse_rows(p, dp, high, low):
        mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        for row in pl.range(half):
            lane_sum = vf.full(0.0, mask, dtype=FP32)
            for chunk in pl.range(columns // 64):
                probability = vf.load_align(p, [row, chunk * 64])
                grad_probability = vf.load_align(dp, [row, chunk * 64])
                weighted = vf.mul(probability, grad_probability, mask)
                lane_sum = vf.add(lane_sum, weighted, mask)
            total = vf.reduce_sum(lane_sum, mask)
            row_dot = vf.full(total, mask)
            for chunk in pl.range(columns // 64):
                probability = vf.load_align(p, [row, chunk * 64])
                grad_probability = vf.load_align(dp, [row, chunk * 64])
                centered = vf.sub(grad_probability, row_dot, mask)
                grad_score = vf.mul(probability, centered, mask)
                hi = vf.astype(grad_score, mask, dtype=dtype, layout=pl.CastLayout.ZERO)
                widened = vf.astype(hi, mask, dtype=FP32, layout=pl.CastLayout.ZERO)
                remainder = vf.sub(grad_score, widened, mask)
                lo = vf.astype(remainder, mask, dtype=dtype, layout=pl.CastLayout.ZERO)
                vf.store_align(high, hi, mask, [row, chunk * 64], dist=pl.StoreDist.PACK_B32)
                vf.store_align(low, lo, mask, [row, chunk * 64], dist=pl.StoreDist.PACK_B32)

    @pl.pipeline.stage
    def products(tile, a, b, c, g, a_g, b_g, c_g, g_g, left, right, acc_g, shared_score, shared_dp):
        lhs = a_g.current()
        rhs = b_g.current()
        acc = acc_g.current()
        score = shared_score.current()
        pl.load(lhs, a, [tile * query_rows, 0])
        pl.load(rhs, b, [0, 0])
        pl.gemm(acc, lhs, rhs, l0a=left, l0b=right, splitk=64)
        pl.move(score, acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
        upstream = g_g.current()
        payload = c_g.current()
        dp = shared_dp.current()
        pl.load(upstream, g, [tile * query_rows, 0])
        pl.load(payload, c, [0, 0])
        pl.gemm(acc, upstream, payload, l0a=left, l0b=right, trans_b=True, splitk=64)
        pl.move(dp, acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)

    @pl.pipeline.stage
    def coefficients(tile, p_hi, p_lo, ds_hi, ds_lo, shared_score, shared_dp, p_g, ph_g, pl_g, dh_g, dl_g):
        score = shared_score.current()
        dp = shared_dp.current()
        p = p_g.current()
        ph = ph_g.current()
        low_p = pl_g.current()
        dh = dh_g.current()
        low_d = dl_g.current()
        normalize_rows(score, p, ph, low_p)
        reverse_rows(p, dp, dh, low_d)
        row = tile * query_rows + pl.get_subblock_idx() * half
        pl.store(p_hi, ph, [row, 0])
        pl.store(p_lo, low_p, [row, 0])
        pl.store(ds_hi, dh, [row, 0])
        pl.store(ds_lo, low_d, [row, 0])

    @pl.jit(
        arch="a5", auto_mutex=True, name=f"mm_softmax_mm_backward_front_r{query_rows}",
        pipeline=pl.pipeline.PipelineConfig(preload=0),
    )
    def front(
        a: pl.Tensor[[rows, features], dtype], b: pl.Tensor[[features, columns], dtype],
        c: pl.Tensor[[columns, channels], dtype], g: pl.Tensor[[rows, channels], dtype],
        p_hi: pl.Tensor[[rows, columns], dtype], p_lo: pl.Tensor[[rows, columns], dtype],
        ds_hi: pl.Tensor[[rows, columns], dtype], ds_lo: pl.Tensor[[rows, columns], dtype],
    ):
        a_g = pl.make_tile_group(
            type=pl.TileType(shape=[query_rows, features], dtype=dtype, target_memory=pl.MemorySpace.Mat), addrs=0,
            mutex_ids="auto", depth=1)
        b_g = pl.make_tile_group(
            type=pl.TileType(shape=[features, columns], dtype=dtype, target_memory=pl.MemorySpace.Mat),
            addrs=2 * query_rows * features, mutex_ids="auto", depth=1)
        g_g = pl.make_tile_group(
            type=pl.TileType(shape=[query_rows, channels], dtype=dtype, target_memory=pl.MemorySpace.Mat),
            addrs=2 * (query_rows * features + features * columns), mutex_ids="auto", depth=1)
        c_g = pl.make_tile_group(
            type=pl.TileType(shape=[columns, channels], dtype=dtype, target_memory=pl.MemorySpace.Mat),
            addrs=2 * (query_rows * features + features * columns + query_rows * channels), mutex_ids="auto", depth=1)
        left = pl.make_tile_group(
            type=pl.TileType(shape=[query_rows, 64], dtype=dtype, target_memory=pl.MemorySpace.Left), addrs=0,
            mutex_ids="auto", depth=1)
        right = pl.make_tile_group(
            type=pl.TileType(shape=[64, columns], dtype=dtype, target_memory=pl.MemorySpace.Right), addrs=0,
            mutex_ids="auto", depth=1)
        acc_g = pl.make_tile_group(
            type=pl.TileType(shape=[query_rows, columns], dtype=FP32, target_memory=pl.MemorySpace.Acc), addrs=0,
            mutex_ids="auto", depth=1)
        shared_score = pl.make_tile_group(
            type=pl.TileType(shape=[half, columns], dtype=FP32, target_memory=pl.MemorySpace.Vec), addrs=0,
            mutex_ids="auto", depth=1, fwd_ids=[0], bwd_ids=[1])
        shared_dp = pl.make_tile_group(
            type=pl.TileType(shape=[half, columns], dtype=FP32, target_memory=pl.MemorySpace.Vec),
            addrs=4 * half * columns, mutex_ids="auto", depth=1, fwd_ids=[2], bwd_ids=[3])
        p_g = pl.make_tile_group(
            type=pl.TileType(shape=[half, columns], dtype=FP32, target_memory=pl.MemorySpace.Vec),
            addrs=8 * half * columns, mutex_ids="auto", depth=1)
        ph_g = pl.make_tile_group(
            type=pl.TileType(shape=[half, columns], dtype=dtype, target_memory=pl.MemorySpace.Vec),
            addrs=12 * half * columns, mutex_ids="auto", depth=1)
        pl_g = pl.make_tile_group(
            type=pl.TileType(shape=[half, columns], dtype=dtype, target_memory=pl.MemorySpace.Vec),
            addrs=14 * half * columns, mutex_ids="auto", depth=1)
        dh_g = pl.make_tile_group(
            type=pl.TileType(shape=[half, columns], dtype=dtype, target_memory=pl.MemorySpace.Vec),
            addrs=16 * half * columns, mutex_ids="auto", depth=1)
        dl_g = pl.make_tile_group(
            type=pl.TileType(shape=[half, columns], dtype=dtype, target_memory=pl.MemorySpace.Vec),
            addrs=18 * half * columns, mutex_ids="auto", depth=1)
        group = pl.get_block_idx() // pl.get_subblock_num()
        for tile in pl.range(group, rows // query_rows, pl.get_block_num()):
            with pl.section_cube():
                products(tile, a, b, c, g, a_g, b_g, c_g, g_g, left, right, acc_g, shared_score, shared_dp)
            with pl.section_vector():
                coefficients(tile, p_hi, p_lo, ds_hi, ds_lo, shared_score, shared_dp, p_g, ph_g, pl_g, dh_g, dl_g)

    grad_a = _make_product(rows, columns, features, dtype, trans_b=True, residual="left")
    grad_b = _make_product(features, rows, columns, dtype, trans_a=True, residual="right")
    grad_c = _make_product(columns, rows, channels, dtype, trans_a=True, residual="left")
    return front, grad_a, grad_b, grad_c


def execute_onchip(kernels, a, b, c, upstream, outputs, workspace):
    """Enqueue four native tasks; workspace=(P_hi,P_lo,dS_hi,dS_lo)."""
    if any(value.ndim != 2 or not value.is_contiguous() for value in (a, b, c, upstream)):
        raise ValueError("inputs must be contiguous two-dimensional tensors")
    if (
        str(a.dtype) not in ("torch.float16", "torch.bfloat16")
        or any(value.dtype != a.dtype for value in (b, c, upstream))
    ):
        raise ValueError("inputs must have one common FP16 or BF16 dtype")
    if a.shape[1] != b.shape[0] or b.shape[1] != c.shape[0] or upstream.shape != (a.shape[0], c.shape[1]):
        raise ValueError("input matrix-chain dimensions do not match")
    ph, low_p, dh, low_d = workspace
    da, db, dc = outputs
    front, grad_a, grad_b, grad_c = kernels
    front(a, b, c, upstream, ph, low_p, dh, low_d)
    grad_a(dh, b, low_d, da)
    grad_b(a, dh, low_d, db)
    grad_c(ph, upstream, low_p, dc)
