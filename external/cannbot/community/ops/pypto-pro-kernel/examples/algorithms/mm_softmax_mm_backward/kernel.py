# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：语法糖。 Native materialized backward control for Y = softmax(A @ B, dim=1) @ C.

A[M,K], B[K,N], C[N,D] and the supplied upstream G[M,D] are contiguous
two-dimensional FP16 or BF16 tensors of the same dtype. All four dimensions
are multiples of 64 in [64,1024]. Inputs are finite and bounded by one;
the score range within each row is at most 16. This bounded example does not
implement arbitrary strides, partial tiles or exceptional floating values.

The ordinary matrix derivatives are dC=P.T@G, dP=G@C.T,
dS=P*(dP-sum(P*dP,dim=1)), dA=dS@B.T and dB=A.T@dS. Scores, P, dP,
dS and the three public gradients are FP32. Every matrix contraction uses
Cube pl.gemm; normalization, row reductions and coefficient splitting use VF.
Seven separate native launches on one stream establish the phase boundaries.
The caller owns and may inspect all GM intermediates.

Cube consumes each derived FP32 matrix as hi=RNE_low(x),
lo=RNE_low(x-FP32(hi)), with FP16/BF16 chosen from the original operands.
Both component products accumulate into one FP32 Acc before storage. This
retains the mathematical FP32 intermediate in the reference; component
rounding is an implementation error, not part of the mathematical contract.
Two components do not reconstruct every FP32 value exactly, and FP16 tiny
residuals may underflow. The independent tests retain the same criterion for
both input dtypes and include BF16 conversion and cancellation controls.

Each Cube worker uses 256*reduction bytes in Mat (two reused 64-wide panels),
8 KiB in each Left/Right space and 16 KiB in Acc. Each VF worker uses at
most 16*N bytes in Vec. Every group has one slot, an explicit address and
automatic mutex IDs. The default automatic launch count is retained.

example-query: {"aliases":["matrix softmax matrix backward","normalized matrix chain gradients","normalized contraction gradients","attention backward","二维归一化矩阵链反向"],"term_groups":[["softmax","normalized","attention","归一化"],["backward","gradient","gradients","grad","reverse","反向","梯度"]],"facets":{"source_dtype":["FP16","BF16"],"destination_dtype":["FP32"]},"exclude":[{"term_groups":[["causal mask","runtime mask","head mapping","TND","BSND","BNSD","arbitrary strides"]],"reason":"This bounded control implements contiguous 2-D unmasked matrices only: all extents are multiples of 64 in [64,1024], finite inputs have magnitude at most one, and each score row has range at most 16. It does not implement masks, head mapping or application layouts; use its matrix algebra through a separately verified adapter."}]}
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

FP32 = pl.DT_FP32


def _make_product(rows, reduction, columns, dtype, trans_a=False, trans_b=False, residual="none"):
    """Create one complete tile-owner GEMM, optionally with one residual operand."""
    a_shape = [reduction, rows] if trans_a else [rows, reduction]
    b_shape = [columns, reduction] if trans_b else [reduction, columns]
    a_tile_shape = [reduction, 64] if trans_a else [64, reduction]
    b_tile_shape = [64, reduction] if trans_b else [reduction, 64]
    low_shape = a_shape if residual == "left" else b_shape
    m_tiles, n_tiles = rows // 64, columns // 64

    def product_body(a, b, low, output):
        a_group = pl.make_tile_group(
            type=pl.TileType(shape=a_tile_shape, dtype=dtype, target_memory=pl.MemorySpace.Mat),
            addrs=0, mutex_ids="auto", depth=1,
        )
        b_group = pl.make_tile_group(
            type=pl.TileType(shape=b_tile_shape, dtype=dtype, target_memory=pl.MemorySpace.Mat),
            addrs=128 * reduction, mutex_ids="auto", depth=1,
        )
        left = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=dtype, target_memory=pl.MemorySpace.Left),
            addrs=0, mutex_ids="auto", depth=1,
        )
        right = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=dtype, target_memory=pl.MemorySpace.Right),
            addrs=0, mutex_ids="auto", depth=1,
        )
        accumulator = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=FP32, target_memory=pl.MemorySpace.Acc),
            addrs=0, mutex_ids="auto", depth=1,
        )
        with pl.section_cube():
            for task in pl.range(pl.get_block_idx(), m_tiles * n_tiles, pl.get_block_num()):
                m0 = (task // n_tiles) * 64
                n0 = (task % n_tiles) * 64
                lhs, rhs, acc = a_group.current(), b_group.current(), accumulator.current()
                if trans_a:
                    pl.load(lhs, a, [0, m0])
                else:
                    pl.load(lhs, a, [m0, 0])
                if trans_b:
                    pl.load(rhs, b, [n0, 0])
                else:
                    pl.load(rhs, b, [0, n0])
                pl.gemm(acc, lhs, rhs, l0a=left, l0b=right, trans_a=trans_a, trans_b=trans_b, splitk=64)
                if residual == "left":
                    if trans_a:
                        pl.load(lhs, low, [0, m0])
                    else:
                        pl.load(lhs, low, [m0, 0])
                    pl.gemm(acc, lhs, rhs, l0a=left, l0b=right, trans_a=trans_a, trans_b=trans_b, splitk=64, init=False)
                if residual == "right":
                    if trans_b:
                        pl.load(rhs, low, [n0, 0])
                    else:
                        pl.load(rhs, low, [0, n0])
                    pl.gemm(acc, lhs, rhs, l0a=left, l0b=right, trans_a=trans_a, trans_b=trans_b, splitk=64, init=False)
                pl.store(output, acc, [m0, n0])

    if residual == "none":
        @pl.jit(arch="a5", auto_mutex=True)
        def product(
            a: pl.Tensor[a_shape, dtype], b: pl.Tensor[b_shape, dtype], output: pl.Tensor[[rows, columns], FP32],
        ):
            product_body(a, b, b, output)
        return product

    @pl.jit(arch="a5", auto_mutex=True)
    def residual_product(
        a: pl.Tensor[a_shape, dtype], b: pl.Tensor[b_shape, dtype],
        low: pl.Tensor[low_shape, dtype], output: pl.Tensor[[rows, columns], FP32],
    ):
        product_body(a, b, low, output)
    return residual_product


def make_kernels(rows, features, columns, channels, dtype_name="float16"):
    """Return score, normalize, dP, softmax-backward, dA, dB and dC entries."""
    if any(
        type(size) is not int or size < 64 or size > 1024 or size % 64 for size in (rows, features, columns, channels)
    ):
        raise ValueError("M, K, N and D must be multiples of 64 in [64,1024]")
    if dtype_name not in ("float16", "bfloat16"):
        raise ValueError("dtype_name must be float16 or bfloat16")
    dtype = pl.DT_FP16 if dtype_name == "float16" else pl.DT_BF16

    @pl.vector_function
    def split_value(value, high, low, mask, offset):
        hi = vf.astype(value, mask, dtype=dtype, layout=pl.CastLayout.ZERO)
        widened = vf.astype(hi, mask, dtype=FP32, layout=pl.CastLayout.ZERO)
        remainder = vf.sub(value, widened, mask)
        lo = vf.astype(remainder, mask, dtype=dtype, layout=pl.CastLayout.ZERO)
        vf.store_align(high, hi, mask, [0, offset], dist=pl.StoreDist.PACK_B32)
        vf.store_align(low, lo, mask, [0, offset], dist=pl.StoreDist.PACK_B32)

    @pl.vector_function
    def normalize_row(score, probability, high, low):
        mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        lane_max = vf.full(-1.0e30, mask, dtype=FP32)
        for chunk in pl.range(columns // 64):
            value = vf.load_align(score, [0, chunk * 64])
            lane_max = vf.max(lane_max, value, mask)
        maximum = vf.reduce_max(lane_max, mask)
        peak = vf.full(maximum, mask)
        lane_sum = vf.full(0.0, mask, dtype=FP32)
        for chunk in pl.range(columns // 64):
            value = vf.load_align(score, [0, chunk * 64])
            shifted = vf.sub(value, peak, mask)
            exponential = vf.exp(shifted, mask)
            lane_sum = vf.add(lane_sum, exponential, mask)
        total = vf.reduce_sum(lane_sum, mask)
        denominator = vf.full(total, mask)
        for chunk in pl.range(columns // 64):
            value = vf.load_align(score, [0, chunk * 64])
            shifted = vf.sub(value, peak, mask)
            exponential = vf.exp(shifted, mask)
            p = vf.div(exponential, denominator, mask)
            vf.store_align(probability, p, mask, [0, chunk * 64])
            split_value(p, high, low, mask, chunk * 64)

    @pl.jit(arch="a5", auto_mutex=True)
    def normalize(
        scores: pl.Tensor[[rows, columns], FP32], probabilities: pl.Tensor[[rows, columns], FP32],
        high: pl.Tensor[[rows, columns], dtype], low: pl.Tensor[[rows, columns], dtype],
    ):
        score_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, columns], dtype=FP32, target_memory=pl.MemorySpace.Vec), addrs=0,
            mutex_ids="auto", depth=1)
        p_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, columns], dtype=FP32, target_memory=pl.MemorySpace.Vec), addrs=4 * columns,
            mutex_ids="auto", depth=1)
        high_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, columns], dtype=dtype, target_memory=pl.MemorySpace.Vec), addrs=8 * columns,
            mutex_ids="auto", depth=1)
        low_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, columns], dtype=dtype, target_memory=pl.MemorySpace.Vec), addrs=10 * columns,
            mutex_ids="auto", depth=1)
        with pl.section_vector():
            for row in pl.range(pl.get_block_idx(), rows, pl.get_block_num()):
                src, dst = score_g.current(), p_g.current()
                hi, lo = high_g.current(), low_g.current()
                pl.load(src, scores, [row, 0])
                normalize_row(src, dst, hi, lo)
                pl.store(probabilities, dst, [row, 0])
                pl.store(high, hi, [row, 0])
                pl.store(low, lo, [row, 0])

    @pl.vector_function
    def reverse_row(probability, grad_probability, grad_score, high, low):
        mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        lane_sum = vf.full(0.0, mask, dtype=FP32)
        for chunk in pl.range(columns // 64):
            p = vf.load_align(probability, [0, chunk * 64])
            dp = vf.load_align(grad_probability, [0, chunk * 64])
            weighted = vf.mul(p, dp, mask)
            lane_sum = vf.add(lane_sum, weighted, mask)
        total = vf.reduce_sum(lane_sum, mask)
        row_dot = vf.full(total, mask)
        for chunk in pl.range(columns // 64):
            p = vf.load_align(probability, [0, chunk * 64])
            dp = vf.load_align(grad_probability, [0, chunk * 64])
            centered = vf.sub(dp, row_dot, mask)
            ds = vf.mul(p, centered, mask)
            vf.store_align(grad_score, ds, mask, [0, chunk * 64])
            split_value(ds, high, low, mask, chunk * 64)

    @pl.jit(arch="a5", auto_mutex=True)
    def softmax_backward(
        probabilities: pl.Tensor[[rows, columns], FP32], grad_probabilities: pl.Tensor[[rows, columns], FP32],
        grad_scores: pl.Tensor[[rows, columns], FP32], high: pl.Tensor[[rows, columns], dtype],
        low: pl.Tensor[[rows, columns], dtype],
    ):
        p_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, columns], dtype=FP32, target_memory=pl.MemorySpace.Vec), addrs=0,
            mutex_ids="auto", depth=1)
        dp_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, columns], dtype=FP32, target_memory=pl.MemorySpace.Vec), addrs=4 * columns,
            mutex_ids="auto", depth=1)
        ds_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, columns], dtype=FP32, target_memory=pl.MemorySpace.Vec), addrs=8 * columns,
            mutex_ids="auto", depth=1)
        high_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, columns], dtype=dtype, target_memory=pl.MemorySpace.Vec), addrs=12 * columns,
            mutex_ids="auto", depth=1)
        low_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, columns], dtype=dtype, target_memory=pl.MemorySpace.Vec), addrs=14 * columns,
            mutex_ids="auto", depth=1)
        with pl.section_vector():
            for row in pl.range(pl.get_block_idx(), rows, pl.get_block_num()):
                p, dp, ds = p_g.current(), dp_g.current(), ds_g.current()
                hi, lo = high_g.current(), low_g.current()
                pl.load(p, probabilities, [row, 0])
                pl.load(dp, grad_probabilities, [row, 0])
                reverse_row(p, dp, ds, hi, lo)
                pl.store(grad_scores, ds, [row, 0])
                pl.store(high, hi, [row, 0])
                pl.store(low, lo, [row, 0])

    score = _make_product(rows, features, columns, dtype)
    grad_probability = _make_product(rows, channels, columns, dtype, trans_b=True)
    grad_a = _make_product(rows, columns, features, dtype, trans_b=True, residual="left")
    grad_b = _make_product(features, rows, columns, dtype, trans_a=True, residual="right")
    grad_c = _make_product(columns, rows, channels, dtype, trans_a=True, residual="left")
    return score, normalize, grad_probability, softmax_backward, grad_a, grad_b, grad_c


def execute(kernels, a, b, c, upstream, outputs, workspace):
    """Enqueue the seven native phases; allocation and synchronization belong to the caller.

    outputs=(dA,dB,dC); workspace=(scores,P,P_hi,P_lo,dP,dS,dS_hi,dS_lo).
    All tensors are contiguous with the shapes/dtypes declared above.
    """
    if any(value.ndim != 2 or not value.is_contiguous() for value in (a, b, c, upstream)):
        raise ValueError("inputs must be contiguous two-dimensional tensors")
    if any(value.dtype != a.dtype for value in (b, c, upstream)):
        raise ValueError("inputs must have one common dtype")
    if a.shape[1] != b.shape[0] or b.shape[1] != c.shape[0] or upstream.shape != (a.shape[0], c.shape[1]):
        raise ValueError("input matrix-chain dimensions do not match")
    scores, p, p_hi, p_lo, dp, ds, ds_hi, ds_lo = workspace
    da, db, dc = outputs
    score, normalize, grad_probability, reverse, grad_a, grad_b, grad_c = kernels
    score(a, b, scores)
    normalize(scores, p, p_hi, p_lo)
    grad_probability(upstream, c, dp)
    reverse(p, dp, ds, ds_hi, ds_lo)
    grad_a(ds_hi, b, ds_lo, da)
    grad_b(a, ds_hi, ds_lo, db)
    grad_c(p_hi, upstream, p_lo, dc)
