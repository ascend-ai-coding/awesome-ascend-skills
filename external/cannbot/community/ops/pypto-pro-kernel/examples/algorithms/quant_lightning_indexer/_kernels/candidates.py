# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始（VF）。Eight-token maxima and the mandatory partial block."""

from functools import lru_cache
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

BF = pl.DT_BF16
F32 = pl.DT_FP32
I32 = pl.DT_INT32
U32 = pl.DT_UINT32
NEG_INF = float("-inf")
POS_INF = float("inf")


@pl.vector_function
def block_max(src, dst, active, start, partial):
    p = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=F32)
    ninf = vf.full(NEG_INF, p, dtype=F32)
    pinf = vf.full(POS_INF, p, dtype=F32)
    idx = vf.arange(0, dtype=U32)
    eight = vf.full(8, p, dtype=U32)
    shifted = vf.sub(idx, eight, p)
    first = vf.lt(idx, eight, p)
    out_mask = vf.update_mask(16, dtype=F32)
    for c in pl.range(0, active, 128):
        x0 = vf.load_align(src, c, dist=pl.LoadDist.UNPK_B16)
        x1 = vf.load_align(src, c + 64, dist=pl.LoadDist.UNPK_B16)
        f0 = vf.astype(x0, p, dtype=F32, layout=pl.CastLayout.ZERO)
        f1 = vf.astype(x1, p, dtype=F32, layout=pl.CastLayout.ZERO)
        p0 = vf.update_mask(pl.min(64, active - c), dtype=F32)
        p1 = vf.update_mask(pl.max(0, pl.min(64, active - c - 64)), dtype=F32)
        a0 = vf.select(f0, ninf, p0)
        a1 = vf.select(f1, ninf, p1)
        m0 = vf.reduce_max(a0, p, datablock=True)
        m1 = vf.reduce_max(a1, p, datablock=True)
        hi = vf.gather(m1, shifted)
        joined = vf.select(m0, hi, first)
        indices = vf.adds(idx, (start + c) // 8, p)
        si = vf.bit_cast(indices, dtype=I32)
        tail = vf.eq(si, partial, p)
        value = vf.select(pinf, joined, tail)
        bf = vf.astype(value, out_mask, dtype=BF, layout=pl.CastLayout.ZERO)
        bits = vf.bit_cast(bf, dtype=U32)
        vf.store_align(dst, bits, out_mask, c // 8, dist=pl.StoreDist.PACK_B32)


@pl.vector_function
def write_count(dst, count):
    p = vf.update_mask(8, dtype=I32)
    value = vf.full(count, p, dtype=I32)
    vf.store_align(dst, value, p, 0)


@lru_cache(None)
def make_blocks(
    capacity,
    mask,
    ratio,
    has_cuq,
    has_cuk,
    has_usedq,
    has_usedk,
    has_residual,
):
    width = 2048

    @pl.jit(auto_mutex=True)
    def kernel(
        scores: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], BF],
        cuq: pl.Tensor[[pl.DYNAMIC], I32],
        cuk: pl.Tensor[[pl.DYNAMIC], I32],
        usedq: pl.Tensor[[pl.DYNAMIC], I32],
        usedk: pl.Tensor[[pl.DYNAMIC], I32],
        residual: pl.Tensor[[pl.DYNAMIC], I32],
        maxima: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], BF],
        lengths: pl.Tensor[[pl.DYNAMIC, 1], I32],
    ):
        rows = scores.shape[0]
        cols = scores.shape[1]
        batches = 1
        if has_cuq:
            batches = cuq.shape[0] - 1
        sg_type = pl.TileType(
            shape=[1, width],
            dtype=BF,
            target_memory=pl.MemorySpace.Vec,
        )
        sg = pl.make_tile_group(
            type=sg_type,
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        og_type = pl.TileType(
            shape=[1, width // 8],
            dtype=BF,
            target_memory=pl.MemorySpace.Vec,
        )
        og = pl.make_tile_group(
            type=og_type,
            addrs=2 * width,
            mutex_ids="auto",
            depth=1,
        )
        lg_type = pl.TileType(
            shape=[1, 8],
            dtype=I32,
            target_memory=pl.MemorySpace.Vec,
        )
        lg = pl.make_tile_group(
            type=lg_type,
            addrs=2 * width + 2 * (width // 8),
            mutex_ids="auto",
            depth=1,
        )
        with pl.section_vector():
            for row in pl.range(pl.get_block_idx(), rows, pl.get_block_num()):
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
                kl = cols
                if has_cuk:
                    kl = pl.getval(cuk, batch + 1) - pl.getval(cuk, batch)
                if has_usedk:
                    kl = pl.getval(usedk, batch)
                rem = 0
                if has_residual:
                    rem = pl.getval(residual, batch)
                visible = kl
                if mask == 3:
                    visible = pl.min(
                        kl,
                        pl.max(0, (ratio * kl + rem - uq + row - qstart + 1) // ratio),
                    )
                if row - qstart >= uq:
                    visible = 0
                partial = -1
                if visible % 8 != 0:
                    partial = visible // 8
                lt = lg.current()
                write_count(lt, pl.min(capacity, (visible + 7) // 8))
                pl.set_validshape(lt, [1, 1])
                pl.store(lengths, lt, [row, 0])
                for start in pl.range(0, cols, width):
                    active = pl.min(width, cols - start)
                    src = sg.current()
                    out = og.current()
                    pl.set_validshape(src, [1, active])
                    pl.load(src, scores, [row, start])
                    block_max(src, out, active, start, partial)
                    pl.set_validshape(out, [1, (active + 7) // 8])
                    pl.store(maxima, out, [row, start // 8])

    return kernel
