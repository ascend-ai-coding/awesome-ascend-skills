# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：语法糖（``pl.gemm`` + B 层流水 + VF）。M=1000 的 RMSNorm → GEMM，全链路有效行数收口。

Contract: X[1000,256], gamma[256] and W[256,256] are FP16; O[1000,256] is FP32. Arithmetic, row+1-pitch NZ packing,
explicit byte ranges, automatic mutex IDs and two-slot Mat handover follow rmsnorm_gemm/kernel.py. The gamma parameter
is one-dimensional and loads into a [1,K] Vec tile with offset [0].

Eight execution groups each own 125 rows. Their two bands contain 64 and 61 rows; AIV0 handles 32 rows and AIV1 handles
32 or 29. AIV insert offsets remain 0/32, aligned to the 16-row NZ fractal. The final band's windows are [939,971) and
[971,1000).

Each stage computes its runtime band_rows and AIV rows. Effective extents propagate through set_validshape(x_tile) →
load → VF's rows loop → set_validshape(nz_lo/nz_hi) → insert → gemm(m=band_rows) → store. l0a and acc_g use compact=1
because GEMM narrows them. Declared capacity and slot spacing remain unchanged; unused P rows are not consumed. The
implementation does not read or write padding past the logical tensors.

The tests combine an independent CPU reference, the existing FP16-rounding classifier, preload/repeated-call checks,
row-owner diagnostics, output guards, poisoned input padding and sanitizer instrumentation. Guards detect writes,
poisoning detects invalid values reaching useful outputs, and sanitizer checks the logical access windows; none alone
covers all errors. See references/diagnostics/check-coverage.md for their limits.

The sanitizer variant uses name=None so the program and entry-function names match. Each variant is built and run
separately; this example makes no performance claim.
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32
FP16 = pl.DT_FP16

M, K, N = 1000, 256, 256  # X[M,K], gamma[K], W[K,N], O[M,N]
NUM_CORES = 8
LAUNCH_BLOCK_DIM = NUM_CORES  # mix kernel 的 block_dim 是执行组数
ROWS_PER_CORE = M // NUM_CORES  # 125：均分，每核同量
BAND_ROWS = 64
NUM_BANDS_PER_CORE = -(-ROWS_PER_CORE // BAND_ROWS)  # 2（向上取整）
HALF_BAND = BAND_ROWS // 2  # 32：AIV0 在一条带里的行数上限
TAIL_BAND_ROWS = ROWS_PER_CORE - (NUM_BANDS_PER_CORE - 1) * BAND_ROWS  # 61
SK = 64  # splitk 步长：K/SK = 4 步

# 静态前提：均分成立；尾带比半带长（AIV1 在每条带里至少 1 行——0 行的 DMA/VF 循环未验过）。
assert M % NUM_CORES == 0
assert 0 < TAIL_BAND_ROWS <= BAND_ROWS
assert TAIL_BAND_ROWS > HALF_BAND

WIN = 128  # 一个 FP16 原生寄存器 = 128 元素
VL = 64  # 一个 FP32 寄存器 = 64 元素；UNPK 一次展开 64 个 FP16
NZ_PITCH = HALF_BAND + 1  # 33：UB NZ 物理列高 = rows+1

EPS = 1e-6
INV_K = 1.0 / K


@pl.vector_function
def _rmsnorm_cast_band(x_src, gamma_src, nz_lo, nz_hi, rows):
    """逐行 RMSNorm（全 FP32）+ 转 FP16 + 散写 row+1 pitch NZ；处理 ``rows`` 行（运行期）。"""
    preg16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP16)
    preg32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)

    g0r = vf.load_align(gamma_src, 0, dist=pl.LoadDist.UNPK)
    g0 = vf.astype(g0r, preg16, dtype=FP32, layout=pl.CastLayout.ZERO)
    g1r = vf.load_align(gamma_src, VL, dist=pl.LoadDist.UNPK)
    g1 = vf.astype(g1r, preg16, dtype=FP32, layout=pl.CastLayout.ZERO)
    g2r = vf.load_align(gamma_src, 2 * VL, dist=pl.LoadDist.UNPK)
    g2 = vf.astype(g2r, preg16, dtype=FP32, layout=pl.CastLayout.ZERO)
    g3r = vf.load_align(gamma_src, 3 * VL, dist=pl.LoadDist.UNPK)
    g3 = vf.astype(g3r, preg16, dtype=FP32, layout=pl.CastLayout.ZERO)

    for r in pl.range(rows):
        x0r = vf.load_align(x_src, r * K, dist=pl.LoadDist.UNPK)
        x0 = vf.astype(x0r, preg16, dtype=FP32, layout=pl.CastLayout.ZERO)
        x1r = vf.load_align(x_src, r * K + VL, dist=pl.LoadDist.UNPK)
        x1 = vf.astype(x1r, preg16, dtype=FP32, layout=pl.CastLayout.ZERO)
        x2r = vf.load_align(x_src, r * K + 2 * VL, dist=pl.LoadDist.UNPK)
        x2 = vf.astype(x2r, preg16, dtype=FP32, layout=pl.CastLayout.ZERO)
        x3r = vf.load_align(x_src, r * K + 3 * VL, dist=pl.LoadDist.UNPK)
        x3 = vf.astype(x3r, preg16, dtype=FP32, layout=pl.CastLayout.ZERO)

        sq0 = vf.mul(x0, x0, preg32)
        sq1 = vf.mul(x1, x1, preg32)
        sq2 = vf.mul(x2, x2, preg32)
        sq3 = vf.mul(x3, x3, preg32)
        s0 = vf.reduce_sum(sq0, preg32)
        s1 = vf.reduce_sum(sq1, preg32)
        s2 = vf.reduce_sum(sq2, preg32)
        s3 = vf.reduce_sum(sq3, preg32)
        s01 = vf.add(s0, s1, preg32)
        s23 = vf.add(s2, s3, preg32)
        total = vf.add(s01, s23, preg32)
        mean_sq = vf.muls(total, INV_K, preg32)
        mean_sq_eps = vf.adds(mean_sq, EPS, preg32)
        rms = vf.sqrt(mean_sq_eps, preg32)
        rms_b = vf.full(rms, preg32)

        n0 = vf.div(x0, rms_b, preg32)
        n1 = vf.div(x1, rms_b, preg32)
        n2 = vf.div(x2, rms_b, preg32)
        n3 = vf.div(x3, rms_b, preg32)
        sc0 = vf.mul(n0, g0, preg32)
        sc1 = vf.mul(n1, g1, preg32)
        sc2 = vf.mul(n2, g2, preg32)
        sc3 = vf.mul(n3, g3, preg32)

        h0 = vf.astype(sc0, preg32, dtype=FP16, layout=pl.CastLayout.ZERO)
        h1 = vf.astype(sc1, preg32, dtype=FP16, layout=pl.CastLayout.ZERO)
        packed_lo, _ = vf.de_interleave(h0, h1)
        vf.store_align(
            nz_lo,
            packed_lo,
            preg16,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=NZ_PITCH,
            repeat_stride=1,
            post_update=True,
        )

        h2 = vf.astype(sc2, preg32, dtype=FP16, layout=pl.CastLayout.ZERO)
        h3 = vf.astype(sc3, preg32, dtype=FP16, layout=pl.CastLayout.ZERO)
        packed_hi, _ = vf.de_interleave(h2, h3)
        vf.store_align(
            nz_hi,
            packed_hi,
            preg16,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=NZ_PITCH,
            repeat_stride=1,
            post_update=True,
        )


@pl.pipeline.stage
def rmsnorm_band(band, x, gamma_g, x_g, nz_lo_g, nz_hi_g, shared_p):
    """vector stage：本带内 AIV0 取前 ``HALF_BAND`` 行、AIV1 取余下的行（运行期行数）。"""
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    band_off = band * BAND_ROWS
    band_rows = pl.min(BAND_ROWS, ROWS_PER_CORE - band_off)  # 64 / 61
    off = pl.get_subblock_idx() * HALF_BAND  # 0 / 32（16 行对齐）
    rows = pl.min(HALF_BAND, band_rows - off)  # 本 AIV 的有效行数
    x_tile = x_g.next()
    pl.set_validshape(x_tile, [rows, K])  # 收窄 GM->UB 的搬运量
    pl.load(x_tile, x, [core_id * ROWS_PER_CORE + band_off + off, 0])
    gamma_tile = gamma_g.current()
    nz_lo = nz_lo_g.next()
    nz_hi = nz_hi_g.next()
    _rmsnorm_cast_band(x_tile, gamma_tile, nz_lo, nz_hi, rows)
    pl.set_validshape(nz_lo, [rows, WIN])
    pl.set_validshape(nz_hi, [rows, WIN])
    p_tile = shared_p.next()
    pl.insert(p_tile, nz_lo, [off, 0])
    pl.insert(p_tile, nz_hi, [off, WIN])


@pl.pipeline.stage
def gemm_band(band, w_g, out, l0a, l0b, acc_g, shared_p):
    """cube stage：整带 P @ W，``m=band_rows`` 收窄，累加器按有效行 store 到 out。"""
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    band_off = band * BAND_ROWS
    band_rows = pl.min(BAND_ROWS, ROWS_PER_CORE - band_off)
    p_tile = shared_p.next()
    w_tile = w_g.current()
    acc = acc_g.next()
    pl.gemm(acc, p_tile, w_tile, l0a=l0a, l0b=l0b, splitk=SK, m=band_rows)
    pl.store(out, acc, [core_id * ROWS_PER_CORE + band_off, 0])


def _make_kernel(preload: int, name: str, sanitizer: bool = False):
    """Build each preload variant separately; sanitizer requires name=None."""

    @pl.jit(
        arch="a5",
        auto_mutex=True,
        name=name,
        sanitizer=sanitizer,
        pipeline=pl.pipeline.PipelineConfig(preload=preload),
    )
    def kernel(
        x: pl.Tensor[[M, K], FP16],
        gamma: pl.Tensor[[K], FP16],
        w: pl.Tensor[[K, N], FP16],
        out: pl.Tensor[[M, N], FP32],
    ):
        x_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[HALF_BAND, K], dtype=FP16, target_memory=pl.MemorySpace.Vec
            ),
            addrs=0x00000,
            mutex_ids="auto",
            depth=2,
        )
        gamma_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, K], dtype=FP16, target_memory=pl.MemorySpace.Vec
            ),
            addrs=0x08000,
            mutex_ids="auto",
            depth=1,
        )
        nz_lo_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[NZ_PITCH, WIN],
                dtype=FP16,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.NZ,
            ),
            addrs=0x08200,
            mutex_ids="auto",
            depth=2,
        )
        nz_hi_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[NZ_PITCH, WIN],
                dtype=FP16,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.NZ,
            ),
            addrs=0x0C400,
            mutex_ids="auto",
            depth=2,
        )
        shared_p = pl.make_tile_group(
            type=pl.TileType(
                shape=[BAND_ROWS, K],
                dtype=FP16,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=2,
            fwd_ids=[0, 1],
            bwd_ids=[2, 3],
        )
        w_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[K, N], dtype=FP16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0x10000,
            mutex_ids="auto",
            depth=1,
        )
        # GEMM narrows l0a and acc through m=band_rows.
        l0a = pl.make_tile_group(
            type=pl.TileType(
                shape=[BAND_ROWS, SK],
                dtype=FP16,
                target_memory=pl.MemorySpace.Left,
                compact=1,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=2,
        )
        l0b = pl.make_tile_group(
            type=pl.TileType(
                shape=[SK, N], dtype=FP16, target_memory=pl.MemorySpace.Right
            ),
            addrs=0,
            mutex_ids="auto",
            depth=2,
        )
        acc_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[BAND_ROWS, N],
                dtype=FP32,
                target_memory=pl.MemorySpace.Acc,
                compact=1,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=2,
        )

        with pl.section_vector():
            gamma_tile = gamma_g.current()
            pl.load(gamma_tile, gamma, [0])
        with pl.section_cube():
            w_tile = w_g.current()
            pl.load(w_tile, w, [0, 0])

        for band in pl.range(NUM_BANDS_PER_CORE):
            with pl.section_vector():
                rmsnorm_band(band, x, gamma_g, x_g, nz_lo_g, nz_hi_g, shared_p)
            with pl.section_cube():
                gemm_band(band, w_g, out, l0a, l0b, acc_g, shared_p)

    return kernel


rmsnorm_gemm_tail = _make_kernel(preload=0, name="rmsnorm_gemm_tail_p0")
rmsnorm_gemm_tail_p2 = _make_kernel(preload=2, name="rmsnorm_gemm_tail_p2")
# Keep the sanitizer program name equal to its entry-function name.
rmsnorm_gemm_tail_san = _make_kernel(preload=0, name=None, sanitizer=True)
