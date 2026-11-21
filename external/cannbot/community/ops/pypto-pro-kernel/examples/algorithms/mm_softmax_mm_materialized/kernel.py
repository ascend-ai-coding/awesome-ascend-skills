# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：语法糖。 Three native phases with independent grids and GM intermediates.

Contract: O = softmax(X @ Y.T / sqrt(K), valid prefix) @ V. X[B,M,K], Y[B,N,K], V[B,N,D] and O[B,M,D] use FP16 or BF16;
scores and accumulators use FP32. lengths[B,M] selects each row's key prefix, with exact zero output for an empty
prefix. Inputs are contiguous, finite and bounded by one in magnitude. M is positive, K is a multiple of 64 up to 1024,
N is a multiple of 128 up to 2048, and D is a positive multiple of 64.

Three native launches on the same stream establish complete phase boundaries. Score, normalization and product kernels
own independent grids over (batch,M tile,N tile), (batch,row) and (batch,M tile,D tile). Physical allocations remain
full-sized; valid shapes and compact Left/Acc tiles handle short and partial M.

All groups have one slot with explicit addresses and automatic mutex IDs. Score Mat X starts at 0 and Y at 128*K bytes;
Left/Right/Acc each start at 0. Product Mat probabilities start at 0 and values at 128*N bytes; Left/Right/Acc each
start at 0. Normalization Vec scores start at 0 and probabilities at 4*N bytes.

Per score worker, Mat uses 384*K bytes and Left/Right/Acc use 8/16/32 KiB. Per product worker, Mat uses 256*N bytes (512
KiB at N=2048) and each L0 space uses 16 KiB. Normalization Vec uses 6*N bytes. GM score/probability scratch uses
6*B*M*N bytes, in addition to the output; the caller allocates it.

This batched, scaled, prefix-validity chain differs from mm_softmax_mm's fixed unscaled FP16 contract. Align scaling,
layouts, dtypes and validity before comparing implementations. Tests use an independent FP64 reference and its existing
input-dependent error budget, plus exact uniform-weight/constant-payload checks, excluded-payload invariance, guards and
input immutability. No fusion performance conclusion follows from this example alone.
"""

import math

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

FP32 = pl.DT_FP32


def make_kernels(batch, rows, features, keys, channels, dtype_name="float16"):
    """Return three ordinary @pl.jit entries; allocate GM scratch in the caller."""
    if batch < 1 or rows < 1:
        raise ValueError("batch and rows must be positive")
    if not 64 <= features <= 1024 or features % 64:
        raise ValueError("features must be a multiple of 64 in [64, 1024]")
    if not 128 <= keys <= 2048 or keys % 128:
        raise ValueError("keys must be a multiple of 128 in [128, 2048]")
    if channels < 64 or channels % 64:
        raise ValueError("channels must be a positive multiple of 64")
    if dtype_name not in ("float16", "bfloat16"):
        raise ValueError("dtype_name must be float16 or bfloat16")
    dtype = pl.DT_FP16 if dtype_name == "float16" else pl.DT_BF16
    scale = 1.0 / math.sqrt(features)
    m_tiles, score_tiles, output_tiles = (rows + 63) // 64, keys // 128, channels // 64

    @pl.jit(arch="a5", auto_mutex=True)
    def score_kernel(
        x: pl.Tensor[[batch, rows, features], dtype],
        y: pl.Tensor[[batch, keys, features], dtype],
        scores: pl.Tensor[[batch, rows, keys], FP32],
    ):
        x_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[64, features], dtype=dtype, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        y_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[128, features], dtype=dtype, target_memory=pl.MemorySpace.Mat
            ),
            addrs=128 * features,
            mutex_ids="auto",
            depth=1,
        )
        left = pl.make_tile_group(
            type=pl.TileType(
                shape=[64, 64],
                dtype=dtype,
                target_memory=pl.MemorySpace.Left,
                compact=1,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        right = pl.make_tile_group(
            type=pl.TileType(
                shape=[64, 128], dtype=dtype, target_memory=pl.MemorySpace.Right
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        acc_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[64, 128], dtype=FP32, target_memory=pl.MemorySpace.Acc, compact=1
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        with pl.section_cube():
            for task in pl.range(
                pl.get_block_idx(), batch * m_tiles * score_tiles, pl.get_block_num()
            ):
                b = task // (m_tiles * score_tiles)
                m0 = ((task // score_tiles) % m_tiles) * 64
                m_valid = pl.min(64, rows - m0)
                n0 = (task % score_tiles) * 128
                a = x_g.current()
                k = y_g.current()
                acc = acc_g.current()
                pl.set_validshape(a, [m_valid, features])
                pl.set_validshape(acc, [m_valid, 128])
                pl.load(a, x, [b, m0, 0], order=[1, 2])
                pl.load(k, y, [b, n0, 0], order=[1, 2])
                pl.gemm(
                    acc, a, k, l0a=left, l0b=right, m=m_valid, trans_b=True, splitk=64
                )
                pl.store(scores, acc, [b, m0, n0], order=[1, 2])

    @pl.vector_function
    def normalize_row(src, dst, length):
        mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        neutral = vf.full(-1.0e30, mask, dtype=FP32)
        zero = vf.full(0.0, mask, dtype=FP32)
        one = vf.full(1.0, mask, dtype=FP32)
        lane_max = vf.full(-1.0e30, mask, dtype=FP32)
        for c in pl.range(keys // 64):
            active = pl.max(0, length - c * 64)
            prefix = vf.update_mask(active, dtype=FP32)
            x = vf.load_align(src, [0, c * 64])
            safe = vf.select(x, neutral, prefix)
            lane_max = vf.max(lane_max, safe, mask)
        maximum = vf.reduce_max(lane_max, mask)
        peak = vf.full(maximum, mask)
        lane_sum = vf.full(0.0, mask, dtype=FP32)
        for c in pl.range(keys // 64):
            active = pl.max(0, length - c * 64)
            prefix = vf.update_mask(active, dtype=FP32)
            x = vf.load_align(src, [0, c * 64])
            safe = vf.select(x, peak, prefix)
            shifted = vf.sub(safe, peak, mask)
            scaled = vf.muls(shifted, scale, mask)
            e = vf.exp(scaled, mask)
            masked = vf.select(e, zero, prefix)
            lane_sum = vf.add(lane_sum, masked, mask)
        total = vf.reduce_sum(lane_sum, mask)
        denominator = vf.full(total, mask)
        safe_denominator = vf.max(denominator, one, mask)
        for c in pl.range(keys // 64):
            active = pl.max(0, length - c * 64)
            prefix = vf.update_mask(active, dtype=FP32)
            x = vf.load_align(src, [0, c * 64])
            safe = vf.select(x, peak, prefix)
            shifted = vf.sub(safe, peak, mask)
            scaled = vf.muls(shifted, scale, mask)
            e = vf.exp(scaled, mask)
            masked = vf.select(e, zero, prefix)
            p = vf.div(masked, safe_denominator, mask)
            narrowed = vf.astype(p, mask, dtype=dtype, layout=pl.CastLayout.ZERO)
            vf.store_align(dst, narrowed, mask, [0, c * 64], dist=pl.StoreDist.PACK_B32)

    @pl.jit(arch="a5", auto_mutex=True)
    def normalize_kernel(
        scores: pl.Tensor[[batch, rows, keys], FP32],
        lengths: pl.Tensor[[batch, rows], pl.DT_INT32],
        probabilities: pl.Tensor[[batch, rows, keys], dtype],
    ):
        score_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, keys], dtype=FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        probability_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, keys], dtype=dtype, target_memory=pl.MemorySpace.Vec
            ),
            addrs=4 * keys,
            mutex_ids="auto",
            depth=1,
        )
        with pl.section_vector():
            for task in pl.range(pl.get_block_idx(), batch * rows, pl.get_block_num()):
                b = task // rows
                row = task % rows
                src = score_g.current()
                dst = probability_g.current()
                pl.load(src, scores, [b, row, 0], order=[1, 2])
                length = pl.getval(lengths, task)
                normalize_row(src, dst, length)
                pl.store(probabilities, dst, [b, row, 0], order=[1, 2])

    @pl.jit(arch="a5", auto_mutex=True)
    def product_kernel(
        probabilities: pl.Tensor[[batch, rows, keys], dtype],
        values: pl.Tensor[[batch, keys, channels], dtype],
        output: pl.Tensor[[batch, rows, channels], dtype],
    ):
        probability_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[64, keys], dtype=dtype, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        value_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[keys, 64], dtype=dtype, target_memory=pl.MemorySpace.Mat
            ),
            addrs=128 * keys,
            mutex_ids="auto",
            depth=1,
        )
        left = pl.make_tile_group(
            type=pl.TileType(
                shape=[64, 128],
                dtype=dtype,
                target_memory=pl.MemorySpace.Left,
                compact=1,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        right = pl.make_tile_group(
            type=pl.TileType(
                shape=[128, 64], dtype=dtype, target_memory=pl.MemorySpace.Right
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        acc_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[64, 64], dtype=FP32, target_memory=pl.MemorySpace.Acc, compact=1
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        with pl.section_cube():
            for task in pl.range(
                pl.get_block_idx(), batch * m_tiles * output_tiles, pl.get_block_num()
            ):
                b = task // (m_tiles * output_tiles)
                m0 = ((task // output_tiles) % m_tiles) * 64
                m_valid = pl.min(64, rows - m0)
                d0 = (task % output_tiles) * 64
                p = probability_g.current()
                v = value_g.current()
                acc = acc_g.current()
                pl.set_validshape(p, [m_valid, keys])
                pl.set_validshape(acc, [m_valid, 64])
                pl.load(p, probabilities, [b, m0, 0], order=[1, 2])
                pl.load(v, values, [b, 0, d0], order=[1, 2])
                pl.gemm(acc, p, v, l0a=left, l0b=right, m=m_valid, splitk=128)
                pl.store(output, acc, [b, m0, d0], order=[1, 2])

    return score_kernel, normalize_kernel, product_kernel
