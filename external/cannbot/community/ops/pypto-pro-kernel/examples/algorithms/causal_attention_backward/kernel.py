# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。 Streamed causal-attention gradients with explicit single-slot Vec resources.

Contract: contiguous q/k/v/dout[B,H,S,D] are FP16, dq/dk/dv have the same shape in FP32, and inputs/outputs do not
overlap. P=softmax(QK.T/sqrt(D)+causal_mask), O=PV; given dO, compute the three analytic gradients. Supported B=1..2,
H=1..4, S=1..1024, D=32/64/128, workers=1/4/16; finite inputs in [-1,1] are a precondition. The independent reference
uses the actual FP16 inputs in FP64, and all three outputs retain rtol=atol=3e-3.

Three native launches on the same stream compute completed row statistics, query-owned dQ, then key-block-owned dK/dV.
K/V transfers use 16-row blocks; dK/dV share each loaded Q/dO/statistics row across a key block. Each output element has
one final writer. All vector arithmetic stays in native VF; this implementation retains horizontal reductions and
repeated dot products rather than claiming Cube efficiency or balanced work for every shape.

Physical feature width is PAD_D=128. For block_size rows, block_bytes=block_size*PAD_D*2; its default is 4096 bytes.
Each phase declares its own single-slot groups with automatic intra-core mutex IDs. Explicit Vec byte ranges (ends
exclusive):

| Group | Range in every phase | Default bytes |
| --- | --- | --- |
| qg / dog | [0,0x100) / [0x100,0x200) | 256 each |
| kg / vg | [0x200,0x200+block_bytes) / [0x200+block_bytes,0x200+2*block_bytes) | 4096 each |
| pairg | [0x200+2*block_bytes,0x400+2*block_bytes) | 512 |
| stateg or statg | [0x400+2*block_bytes,0x600+2*block_bytes) | 512 |
| outg (stats) or dqg (dQ) | [0x600+2*block_bytes,0x800+2*block_bytes) | 512 |
| dkg/dvg | [0x600+2*block_bytes,0x600+4*block_bytes)/[0x600+4*block_bytes,0x600+6*block_bytes) | 8192 each |

Default footprints are 10/10/25.5 KiB. Phase boundaries are ordered GM dependencies; there is no Cube/Vector handover.
GM stats[B,H,S,128] stores replicated denominator and delta in its two 64-column halves, using 512*B*H*S bytes. At D=32
this equals the stated 16*B*H*S*D-byte workspace budget; no S-by-S GM matrix is materialized.

All intermediate arithmetic uses FP32. With bounded inputs, |score|<=sqrt(D). The fixed exponent reference sqrt(D) puts
weights in [exp(-2*sqrt(D)),1], so each nonempty causal row has a positive denominator and |dO dot V|<=D. Metadata
dispatch/allocation is the only host work. Tests cover tails, the joint upper bounds, workers, concentrated weights,
cancellation, input bit preservation and offset guards, with two output poison values. No current performance claim is
made.
"""

from functools import lru_cache
import math

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP16 = pl.DT_FP16
FP32 = pl.DT_FP32
VL32 = 64
PAD_D = 128
STATS_WIDTH = 128
BLOCK_SIZE = 16


@pl.vector_function
def _zero_pair(tile):
    all32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    zero = vf.full(0.0, all32, dtype=FP32)
    vf.store_align(tile, zero, all32, [0, 0])
    vf.store_align(tile, zero, all32, [0, VL32])


@pl.vector_function
def _zero_row(tile, dimension: pl.DT_INT64, row: pl.DT_INT64):
    all32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    for col in pl.range(0, dimension, VL32):
        active = pl.min(VL32, dimension - col)
        pred32 = vf.update_mask(active, dtype=FP32)
        zero = vf.full(0.0, all32, dtype=FP32)
        vf.store_align(tile, zero, pred32, [row, col])


@pl.vector_function
def _pair_dot(
    q, dout, k, v, pair, dimension: pl.DT_INT64, scale: pl.DT_FP32, key_row: pl.DT_INT64
):
    """One regular 64-value VF reduction block per contraction chunk.

    UNPK expands 64 FP16 values into 64 b32 containers before widening; the
    predicate spans 2*active FP16 positions as required by the source layout.
    """
    all32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    score = vf.full(0.0, all32, dtype=FP32)
    grad = vf.full(0.0, all32, dtype=FP32)
    for col in pl.range(0, dimension, VL32):
        active = pl.min(VL32, dimension - col)
        pred16 = vf.update_mask(2 * active, dtype=FP16)
        pred32 = vf.update_mask(active, dtype=FP32)

        q16 = vf.load_align(q, [0, col], dist=pl.LoadDist.UNPK)
        k16 = vf.load_align(k, [key_row, col], dist=pl.LoadDist.UNPK)
        do16 = vf.load_align(dout, [0, col], dist=pl.LoadDist.UNPK)
        v16 = vf.load_align(v, [key_row, col], dist=pl.LoadDist.UNPK)
        q32 = vf.astype(q16, pred16, dtype=FP32, layout=pl.CastLayout.ZERO)
        k32 = vf.astype(k16, pred16, dtype=FP32, layout=pl.CastLayout.ZERO)
        do32 = vf.astype(do16, pred16, dtype=FP32, layout=pl.CastLayout.ZERO)
        v32 = vf.astype(v16, pred16, dtype=FP32, layout=pl.CastLayout.ZERO)

        qk = vf.mul(q32, k32, pred32)
        qk_sum = vf.reduce_sum(qk, pred32)
        qk_full = vf.full(qk_sum, all32, dtype=FP32)
        score = vf.add(score, qk_full, all32)

        dov = vf.mul(do32, v32, pred32)
        dov_sum = vf.reduce_sum(dov, pred32)
        dov_full = vf.full(dov_sum, all32, dtype=FP32)
        grad = vf.add(grad, dov_full, all32)

    score = vf.muls(score, scale, all32)
    vf.store_align(pair, score, all32, [0, 0])
    vf.store_align(pair, grad, all32, [0, VL32])


@pl.vector_function
def _stats_add(pair, state, reference: pl.DT_FP32):
    all32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    score = vf.load_align(pair, [0, 0])
    grad = vf.load_align(pair, [0, VL32])
    shifted = vf.adds(score, -reference, all32)
    weight = vf.exp(shifted, all32)
    weighted_grad = vf.mul(weight, grad, all32)
    den = vf.load_align(state, [0, 0])
    num = vf.load_align(state, [0, VL32])
    den_next = vf.add(den, weight, all32)
    num_next = vf.add(num, weighted_grad, all32)
    vf.store_align(state, den_next, all32, [0, 0])
    vf.store_align(state, num_next, all32, [0, VL32])


@pl.vector_function
def _stats_finish(state, stats_out):
    all32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    den = vf.load_align(state, [0, 0])
    num = vf.load_align(state, [0, VL32])
    delta = vf.div(num, den, all32)
    vf.store_align(stats_out, den, all32, [0, 0])
    vf.store_align(stats_out, delta, all32, [0, VL32])


@pl.vector_function
def _dq_add(
    pair,
    stats,
    k,
    dq_acc,
    dimension: pl.DT_INT64,
    scale: pl.DT_FP32,
    reference: pl.DT_FP32,
    key_row: pl.DT_INT64,
):
    all32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    score = vf.load_align(pair, [0, 0])
    grad = vf.load_align(pair, [0, VL32])
    den = vf.load_align(stats, [0, 0])
    delta = vf.load_align(stats, [0, VL32])
    shifted = vf.adds(score, -reference, all32)
    weight = vf.exp(shifted, all32)
    prob = vf.div(weight, den, all32)
    centered_grad = vf.sub(grad, delta, all32)
    ds = vf.mul(prob, centered_grad, all32)
    ds_scaled = vf.muls(ds, scale, all32)
    for col in pl.range(0, dimension, VL32):
        active = pl.min(VL32, dimension - col)
        pred16 = vf.update_mask(2 * active, dtype=FP16)
        pred32 = vf.update_mask(active, dtype=FP32)
        k16 = vf.load_align(k, [key_row, col], dist=pl.LoadDist.UNPK)
        k32 = vf.astype(k16, pred16, dtype=FP32, layout=pl.CastLayout.ZERO)
        old = vf.load_align(dq_acc, [0, col])
        contribution = vf.mul(k32, ds_scaled, pred32)
        updated = vf.add(old, contribution, pred32)
        vf.store_align(dq_acc, updated, pred32, [0, col])


@pl.vector_function
def _dkv_add(
    pair,
    stats,
    q,
    dout,
    dk_acc,
    dv_acc,
    dimension: pl.DT_INT64,
    scale: pl.DT_FP32,
    reference: pl.DT_FP32,
    key_row: pl.DT_INT64,
):
    all32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    score = vf.load_align(pair, [0, 0])
    grad = vf.load_align(pair, [0, VL32])
    den = vf.load_align(stats, [0, 0])
    delta = vf.load_align(stats, [0, VL32])
    shifted = vf.adds(score, -reference, all32)
    weight = vf.exp(shifted, all32)
    prob = vf.div(weight, den, all32)
    centered_grad = vf.sub(grad, delta, all32)
    ds = vf.mul(prob, centered_grad, all32)
    ds_scaled = vf.muls(ds, scale, all32)
    for col in pl.range(0, dimension, VL32):
        active = pl.min(VL32, dimension - col)
        pred16 = vf.update_mask(2 * active, dtype=FP16)
        pred32 = vf.update_mask(active, dtype=FP32)

        q16 = vf.load_align(q, [0, col], dist=pl.LoadDist.UNPK)
        do16 = vf.load_align(dout, [0, col], dist=pl.LoadDist.UNPK)
        q32 = vf.astype(q16, pred16, dtype=FP32, layout=pl.CastLayout.ZERO)
        do32 = vf.astype(do16, pred16, dtype=FP32, layout=pl.CastLayout.ZERO)

        old_k = vf.load_align(dk_acc, [key_row, col])
        old_v = vf.load_align(dv_acc, [key_row, col])
        dk_term = vf.mul(q32, ds_scaled, pred32)
        dv_term = vf.mul(do32, prob, pred32)
        new_k = vf.add(old_k, dk_term, pred32)
        new_v = vf.add(old_v, dv_term, pred32)
        vf.store_align(dk_acc, new_k, pred32, [key_row, col])
        vf.store_align(dv_acc, new_v, pred32, [key_row, col])


def _make_kernels(batch: int, heads: int, seq: int, dim: int, block_size: int):
    reference = math.sqrt(dim)
    scale = 1.0 / reference
    block_bytes = block_size * PAD_D * 2

    @pl.jit(
        auto_mutex=True,
        name=f"attention_backward_stats_{batch}_{heads}_{seq}_{dim}_bk{block_size}",
    )
    def stats_kernel(
        q: pl.Tensor[[batch, heads, seq, dim], FP16],
        k: pl.Tensor[[batch, heads, seq, dim], FP16],
        v: pl.Tensor[[batch, heads, seq, dim], FP16],
        dout: pl.Tensor[[batch, heads, seq, dim], FP16],
        stats: pl.Tensor[[batch, heads, seq, STATS_WIDTH], FP32],
    ):
        with pl.section_vector():
            qg = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, PAD_D], dtype=FP16, target_memory=pl.MemorySpace.Vec
                ),
                addrs=0,
                mutex_ids="auto",
                depth=1,
            )
            dog = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, PAD_D], dtype=FP16, target_memory=pl.MemorySpace.Vec
                ),
                addrs=0x100,
                mutex_ids="auto",
                depth=1,
            )
            kg = pl.make_tile_group(
                type=pl.TileType(
                    shape=[block_size, PAD_D],
                    dtype=FP16,
                    target_memory=pl.MemorySpace.Vec,
                ),
                addrs=0x200,
                mutex_ids="auto",
                depth=1,
            )
            vg = pl.make_tile_group(
                type=pl.TileType(
                    shape=[block_size, PAD_D],
                    dtype=FP16,
                    target_memory=pl.MemorySpace.Vec,
                ),
                addrs=0x200 + block_bytes,
                mutex_ids="auto",
                depth=1,
            )
            pairg = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, STATS_WIDTH], dtype=FP32, target_memory=pl.MemorySpace.Vec
                ),
                addrs=0x200 + 2 * block_bytes,
                mutex_ids="auto",
                depth=1,
            )
            stateg = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, STATS_WIDTH], dtype=FP32, target_memory=pl.MemorySpace.Vec
                ),
                addrs=0x400 + 2 * block_bytes,
                mutex_ids="auto",
                depth=1,
            )
            outg = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, STATS_WIDTH], dtype=FP32, target_memory=pl.MemorySpace.Vec
                ),
                addrs=0x600 + 2 * block_bytes,
                mutex_ids="auto",
                depth=1,
            )

            rows = batch * heads * seq
            for row in pl.range(pl.get_block_idx(), rows, pl.get_block_num()):
                b = row // (heads * seq)
                h = (row // seq) % heads
                i = row % seq
                qt = qg.next()
                dot = dog.next()
                pl.set_validshape(qt, [1, dim])
                pl.set_validshape(dot, [1, dim])
                pl.load(qt, q, [b, h, i, 0], order=[2, 3])
                pl.load(dot, dout, [b, h, i, 0], order=[2, 3])
                state = stateg.next()
                _zero_pair(state)
                for key_base in pl.range(0, i + 1, block_size):
                    kt = kg.next()
                    vt = vg.next()
                    key_rows = pl.min(block_size, i + 1 - key_base)
                    pl.set_validshape(kt, [key_rows, dim])
                    pl.set_validshape(vt, [key_rows, dim])
                    pl.load(kt, k, [b, h, key_base, 0], order=[2, 3])
                    pl.load(vt, v, [b, h, key_base, 0], order=[2, 3])
                    for key_row in pl.range(0, key_rows):
                        pair = pairg.next()
                        _pair_dot(qt, dot, kt, vt, pair, dim, scale, key_row)
                        _stats_add(pair, state, reference)
                stats_row = outg.next()
                _stats_finish(state, stats_row)
                pl.set_validshape(stats_row, [1, STATS_WIDTH])
                pl.store(stats, stats_row, [b, h, i, 0], order=[2, 3])

    @pl.jit(
        auto_mutex=True,
        name=f"attention_backward_dq_{batch}_{heads}_{seq}_{dim}_bk{block_size}",
    )
    def dq_kernel(
        q: pl.Tensor[[batch, heads, seq, dim], FP16],
        k: pl.Tensor[[batch, heads, seq, dim], FP16],
        v: pl.Tensor[[batch, heads, seq, dim], FP16],
        dout: pl.Tensor[[batch, heads, seq, dim], FP16],
        stats: pl.Tensor[[batch, heads, seq, STATS_WIDTH], FP32],
        dq: pl.Tensor[[batch, heads, seq, dim], FP32],
    ):
        with pl.section_vector():
            qg = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, PAD_D], dtype=FP16, target_memory=pl.MemorySpace.Vec
                ),
                addrs=0,
                mutex_ids="auto",
                depth=1,
            )
            dog = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, PAD_D], dtype=FP16, target_memory=pl.MemorySpace.Vec
                ),
                addrs=0x100,
                mutex_ids="auto",
                depth=1,
            )
            kg = pl.make_tile_group(
                type=pl.TileType(
                    shape=[block_size, PAD_D],
                    dtype=FP16,
                    target_memory=pl.MemorySpace.Vec,
                ),
                addrs=0x200,
                mutex_ids="auto",
                depth=1,
            )
            vg = pl.make_tile_group(
                type=pl.TileType(
                    shape=[block_size, PAD_D],
                    dtype=FP16,
                    target_memory=pl.MemorySpace.Vec,
                ),
                addrs=0x200 + block_bytes,
                mutex_ids="auto",
                depth=1,
            )
            pairg = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, STATS_WIDTH], dtype=FP32, target_memory=pl.MemorySpace.Vec
                ),
                addrs=0x200 + 2 * block_bytes,
                mutex_ids="auto",
                depth=1,
            )
            statg = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, STATS_WIDTH], dtype=FP32, target_memory=pl.MemorySpace.Vec
                ),
                addrs=0x400 + 2 * block_bytes,
                mutex_ids="auto",
                depth=1,
            )
            dqg = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, PAD_D], dtype=FP32, target_memory=pl.MemorySpace.Vec
                ),
                addrs=0x600 + 2 * block_bytes,
                mutex_ids="auto",
                depth=1,
            )

            rows = batch * heads * seq
            for row in pl.range(pl.get_block_idx(), rows, pl.get_block_num()):
                b = row // (heads * seq)
                h = (row // seq) % heads
                i = row % seq
                qt = qg.next()
                dot = dog.next()
                pl.set_validshape(qt, [1, dim])
                pl.set_validshape(dot, [1, dim])
                pl.load(qt, q, [b, h, i, 0], order=[2, 3])
                pl.load(dot, dout, [b, h, i, 0], order=[2, 3])
                stat = statg.next()
                pl.set_validshape(stat, [1, STATS_WIDTH])
                pl.load(stat, stats, [b, h, i, 0], order=[2, 3])
                dq_row = dqg.next()
                _zero_row(dq_row, dim, 0)
                for key_base in pl.range(0, i + 1, block_size):
                    kt = kg.next()
                    vt = vg.next()
                    key_rows = pl.min(block_size, i + 1 - key_base)
                    pl.set_validshape(kt, [key_rows, dim])
                    pl.set_validshape(vt, [key_rows, dim])
                    pl.load(kt, k, [b, h, key_base, 0], order=[2, 3])
                    pl.load(vt, v, [b, h, key_base, 0], order=[2, 3])
                    for key_row in pl.range(0, key_rows):
                        pair = pairg.next()
                        _pair_dot(qt, dot, kt, vt, pair, dim, scale, key_row)
                        _dq_add(pair, stat, kt, dq_row, dim, scale, reference, key_row)
                pl.set_validshape(dq_row, [1, dim])
                pl.store(dq, dq_row, [b, h, i, 0], order=[2, 3])

    @pl.jit(
        auto_mutex=True,
        name=f"attention_backward_dkv_{batch}_{heads}_{seq}_{dim}_bk{block_size}",
    )
    def dkv_kernel(
        q: pl.Tensor[[batch, heads, seq, dim], FP16],
        k: pl.Tensor[[batch, heads, seq, dim], FP16],
        v: pl.Tensor[[batch, heads, seq, dim], FP16],
        dout: pl.Tensor[[batch, heads, seq, dim], FP16],
        stats: pl.Tensor[[batch, heads, seq, STATS_WIDTH], FP32],
        dk: pl.Tensor[[batch, heads, seq, dim], FP32],
        dv: pl.Tensor[[batch, heads, seq, dim], FP32],
    ):
        with pl.section_vector():
            qg = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, PAD_D], dtype=FP16, target_memory=pl.MemorySpace.Vec
                ),
                addrs=0,
                mutex_ids="auto",
                depth=1,
            )
            dog = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, PAD_D], dtype=FP16, target_memory=pl.MemorySpace.Vec
                ),
                addrs=0x100,
                mutex_ids="auto",
                depth=1,
            )
            kg = pl.make_tile_group(
                type=pl.TileType(
                    shape=[block_size, PAD_D],
                    dtype=FP16,
                    target_memory=pl.MemorySpace.Vec,
                ),
                addrs=0x200,
                mutex_ids="auto",
                depth=1,
            )
            vg = pl.make_tile_group(
                type=pl.TileType(
                    shape=[block_size, PAD_D],
                    dtype=FP16,
                    target_memory=pl.MemorySpace.Vec,
                ),
                addrs=0x200 + block_bytes,
                mutex_ids="auto",
                depth=1,
            )
            pairg = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, STATS_WIDTH], dtype=FP32, target_memory=pl.MemorySpace.Vec
                ),
                addrs=0x200 + 2 * block_bytes,
                mutex_ids="auto",
                depth=1,
            )
            statg = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, STATS_WIDTH], dtype=FP32, target_memory=pl.MemorySpace.Vec
                ),
                addrs=0x400 + 2 * block_bytes,
                mutex_ids="auto",
                depth=1,
            )
            dkg = pl.make_tile_group(
                type=pl.TileType(
                    shape=[block_size, PAD_D],
                    dtype=FP32,
                    target_memory=pl.MemorySpace.Vec,
                ),
                addrs=0x600 + 2 * block_bytes,
                mutex_ids="auto",
                depth=1,
            )
            dvg = pl.make_tile_group(
                type=pl.TileType(
                    shape=[block_size, PAD_D],
                    dtype=FP32,
                    target_memory=pl.MemorySpace.Vec,
                ),
                addrs=0x600 + 4 * block_bytes,
                mutex_ids="auto",
                depth=1,
            )

            blocks_per_head = (seq + block_size - 1) // block_size
            blocks = batch * heads * blocks_per_head
            for block in pl.range(pl.get_block_idx(), blocks, pl.get_block_num()):
                b = block // (heads * blocks_per_head)
                h = (block // blocks_per_head) % heads
                j_block = block % blocks_per_head
                key_base = j_block * block_size
                key_rows = pl.min(block_size, seq - key_base)
                kt = kg.next()
                vt = vg.next()
                pl.set_validshape(kt, [key_rows, dim])
                pl.set_validshape(vt, [key_rows, dim])
                pl.load(kt, k, [b, h, key_base, 0], order=[2, 3])
                pl.load(vt, v, [b, h, key_base, 0], order=[2, 3])
                dk_row = dkg.next()
                dv_row = dvg.next()
                for key_row in pl.range(0, key_rows):
                    _zero_row(dk_row, dim, key_row)
                    _zero_row(dv_row, dim, key_row)
                for i in pl.range(key_base, seq):
                    qt = qg.next()
                    dot = dog.next()
                    st = statg.next()
                    pl.set_validshape(qt, [1, dim])
                    pl.set_validshape(dot, [1, dim])
                    pl.set_validshape(st, [1, STATS_WIDTH])
                    pl.load(qt, q, [b, h, i, 0], order=[2, 3])
                    pl.load(dot, dout, [b, h, i, 0], order=[2, 3])
                    pl.load(st, stats, [b, h, i, 0], order=[2, 3])
                    active_keys = pl.min(key_rows, i + 1 - key_base)
                    for key_row in pl.range(0, active_keys):
                        pair = pairg.next()
                        _pair_dot(qt, dot, kt, vt, pair, dim, scale, key_row)
                        _dkv_add(
                            pair,
                            st,
                            qt,
                            dot,
                            dk_row,
                            dv_row,
                            dim,
                            scale,
                            reference,
                            key_row,
                        )
                pl.set_validshape(dk_row, [key_rows, dim])
                pl.set_validshape(dv_row, [key_rows, dim])
                pl.store(dk, dk_row, [b, h, key_base, 0], order=[2, 3])
                pl.store(dv, dv_row, [b, h, key_base, 0], order=[2, 3])

    return stats_kernel, dq_kernel, dkv_kernel


@lru_cache(maxsize=32)
def _kernels(batch: int, heads: int, seq: int, dim: int, block_size: int):
    return _make_kernels(batch, heads, seq, dim, block_size)


def _run_config(q, k, v, dout, dq, dk, dv, workers, block_size):
    if workers not in (1, 4, 16):
        raise ValueError("workers must be one of 1, 4, or 16")
    shape = tuple(q.shape)
    if len(shape) != 4:
        raise ValueError(f"expected [B,H,S,D], got {shape}")
    batch, heads, seq, dim = map(int, shape)
    if not (
        1 <= batch <= 2
        and 1 <= heads <= 4
        and 1 <= seq <= 1024
        and dim in (32, 64, 128)
    ):
        raise ValueError(f"unsupported shape {shape}")
    if any(tuple(t.shape) != shape for t in (k, v, dout, dq, dk, dv)):
        raise ValueError("all input and output tensors must have the same shape")
    import torch

    if any(t.dtype != torch.float16 for t in (q, k, v, dout)):
        raise TypeError("q, k, v, and dout must be FP16")
    if any(t.dtype != torch.float32 for t in (dq, dk, dv)):
        raise TypeError("dq, dk, and dv must be FP32")
    if any(not t.is_contiguous() for t in (q, k, v, dout, dq, dk, dv)):
        raise ValueError("all input and output tensors must be contiguous")
    if any(t.device != q.device for t in (k, v, dout, dq, dk, dv)):
        raise ValueError("all input and output tensors must be on the same device")

    stats = torch.empty(
        (batch, heads, seq, STATS_WIDTH), device=q.device, dtype=torch.float32
    )
    stats_kernel, dq_kernel, dkv_kernel = _kernels(batch, heads, seq, dim, block_size)
    stats_kernel[None, workers](q, k, v, dout, stats)
    dq_kernel[None, workers](q, k, v, dout, stats, dq)
    dkv_kernel[None, workers](q, k, v, dout, stats, dk, dv)


def run(q, k, v, dout, dq, dk, dv, workers=4):
    """Run the blocked streamed phases; host work is shape dispatch/allocation."""
    return _run_config(q, k, v, dout, dq, dk, dv, workers, block_size=BLOCK_SIZE)
