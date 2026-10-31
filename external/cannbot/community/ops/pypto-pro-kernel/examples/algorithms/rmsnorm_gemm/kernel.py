# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：语法糖（``pl.gemm`` + B 层流水 + VF）。整带 RMSNorm → GEMM，8 组各两带。

Contract: X[1024,256], gamma[1,256] and W[256,256] are FP16; O[1024,256] is FP32. RMSNorm computes x / sqrt(mean(x²) +
1e-6) * gamma in FP32, narrows once to FP16, then feeds Cube GEMM with FP32 accumulation.

Each group owns 128 rows. A band contains 64 rows, split into two 32-row AIV regions. VF uses UNPK loads to widen
consecutive FP16 values in order, then packs its output directly into two row+1-pitch NZ tiles. Both AIVs insert their
halves into shared Mat storage; Cube consumes the complete band and stores O directly.

Explicit per-group placement (end addresses are exclusive):

| Space | Group | Byte range | Bytes/slot × depth |
| --- | --- | --- | --- |
| Vec | x_g | [0x00000, 0x08000) | 16384 × 2 |
| Vec | gamma_g | [0x08000, 0x08200) | 512 × 1 |
| Vec | nz_lo_g | [0x08200, 0x0c400) | 8448 × 2 |
| Vec | nz_hi_g | [0x0c400, 0x10600) | 8448 × 2 |
| Mat | shared_p | [0x00000, 0x10000) | 32768 × 2 |
| Mat | w_g | [0x10000, 0x30000) | 131072 × 1 |
| Left | l0a | [0, 0x04000) | 8192 × 2 |
| Right | l0b | [0, 0x10000) | 32768 × 2 |
| Acc | acc_g | [0, 0x20000) | 65536 × 2 |

The 33-row NZ pitch is physical storage, not an effective row count; it is not rounded to 48. W and gamma load once per
group. Independent local buffers use automatic mutex IDs. shared_p has two slots with forward events [0,1] (MTE3 → MTE1)
and reverse events [2,3] (MTE1 → MTE3), protecting each slot until Cube finishes reading it.

preload=0 and preload=2 retain the same geometry and arithmetic. The positive-preload schedule delays the sole Cube
stage by one beat, requiring two handover slots. test.py checks numerical agreement, preload/repeated-call consistency
and selected group/band boundaries. The reference preserves the FP16 intermediate quantization. The non-aligned row
variant is rmsnorm_gemm_tail.
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32
FP16 = pl.DT_FP16

M, K, N = 1024, 256, 256  # X[M,K], gamma[1,K], W[K,N], O[M,N]
NUM_CORES = 8
LAUNCH_BLOCK_DIM = NUM_CORES  # mix kernel 的 block_dim 是执行组数，见 docstring
M_PER_CORE = M // NUM_CORES  # 128
BAND_ROWS = 64
NUM_BANDS_PER_CORE = M_PER_CORE // BAND_ROWS  # 2
HALF_BAND = BAND_ROWS // 2  # 32：一个 AIV 在一条带里的份额
SK = 64  # splitk 步长：K/SK = 4 步

WIN = 128  # 一个 FP16 原生寄存器 = 256 字节 / 2 字节 = 128 元素
VL = 64  # 一个 FP32 寄存器 = 64 元素；UNPK 一次展开 64 个 FP16
NZ_PITCH = HALF_BAND + 1  # 33：UB NZ 物理列高 = rows+1，奇数错开 bank

EPS = 1e-6
INV_K = 1.0 / K


@pl.vector_function
def _rmsnorm_cast_band(x_src, gamma_src, nz_lo, nz_hi):
    """Compute 32 rows in FP32 and pack FP16 directly into two [33,128] NZ tiles.

    UNPK preserves the order of 64 FP16 values per load. Adjacent narrowed chunks
    are de-interleaved into 128 dense FP16 values before cursor-based NZ stores.
    """
    preg16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP16)
    preg32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)

    # gamma：UNPK 保序宽化到 FP32，4 个 64 宽 chunk，每带取一次常驻寄存器。
    g0r = vf.load_align(gamma_src, 0, dist=pl.LoadDist.UNPK)
    g0 = vf.astype(g0r, preg16, dtype=FP32, layout=pl.CastLayout.ZERO)
    g1r = vf.load_align(gamma_src, VL, dist=pl.LoadDist.UNPK)
    g1 = vf.astype(g1r, preg16, dtype=FP32, layout=pl.CastLayout.ZERO)
    g2r = vf.load_align(gamma_src, 2 * VL, dist=pl.LoadDist.UNPK)
    g2 = vf.astype(g2r, preg16, dtype=FP32, layout=pl.CastLayout.ZERO)
    g3r = vf.load_align(gamma_src, 3 * VL, dist=pl.LoadDist.UNPK)
    g3 = vf.astype(g3r, preg16, dtype=FP32, layout=pl.CastLayout.ZERO)

    for r in pl.range(HALF_BAND):
        x0r = vf.load_align(x_src, r * K, dist=pl.LoadDist.UNPK)
        x0 = vf.astype(x0r, preg16, dtype=FP32, layout=pl.CastLayout.ZERO)
        x1r = vf.load_align(x_src, r * K + VL, dist=pl.LoadDist.UNPK)
        x1 = vf.astype(x1r, preg16, dtype=FP32, layout=pl.CastLayout.ZERO)
        x2r = vf.load_align(x_src, r * K + 2 * VL, dist=pl.LoadDist.UNPK)
        x2 = vf.astype(x2r, preg16, dtype=FP32, layout=pl.CastLayout.ZERO)
        x3r = vf.load_align(x_src, r * K + 3 * VL, dist=pl.LoadDist.UNPK)
        x3 = vf.astype(x3r, preg16, dtype=FP32, layout=pl.CastLayout.ZERO)

        # --- 归约：sum of squares，FP32 ---
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

        # --- 逐元素应用：FP32，最后一次性窄化 ---
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
    """vector stage：本带内每个 AIV 直接从 GM load 自己的半区，VF RMSNorm + 转
    FP16 并直接散写 row+1 pitch NZ，``pl.insert`` 按半区偏移写回共享 Mat 组。
    """
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    row_base = core_id * M_PER_CORE + band * BAND_ROWS
    off = pl.get_subblock_idx() * HALF_BAND
    x_tile = x_g.next()
    pl.load(x_tile, x, [row_base + off, 0])
    gamma_tile = gamma_g.current()
    nz_lo = nz_lo_g.next()
    nz_hi = nz_hi_g.next()
    _rmsnorm_cast_band(x_tile, gamma_tile, nz_lo, nz_hi)
    pl.set_validshape(nz_lo, [HALF_BAND, WIN])
    pl.set_validshape(nz_hi, [HALF_BAND, WIN])
    p_tile = shared_p.next()
    pl.insert(p_tile, nz_lo, [off, 0])
    pl.insert(p_tile, nz_hi, [off, WIN])


@pl.pipeline.stage
def gemm_band(band, w_g, out, l0a, l0b, acc_g, shared_p):
    """cube stage：两个 AIV 都写完后的整带 P @ W，累加器直接 store 到 out。"""
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    row_base = core_id * M_PER_CORE + band * BAND_ROWS
    p_tile = shared_p.next()
    w_tile = w_g.current()
    acc = acc_g.next()
    pl.gemm(acc, p_tile, w_tile, l0a=l0a, l0b=l0b, splitk=SK)
    pl.store(out, acc, [row_base, 0])


def _make_kernel(preload: int, name: str):
    """Build separately named entries for the two preload configurations."""

    @pl.jit(
        arch="a5",
        auto_mutex=True,
        name=name,
        pipeline=pl.pipeline.PipelineConfig(preload=preload),
    )
    def kernel(
        x: pl.Tensor[[M, K], FP16],
        gamma: pl.Tensor[[1, K], FP16],
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
        l0a = pl.make_tile_group(
            type=pl.TileType(
                shape=[BAND_ROWS, SK], dtype=FP16, target_memory=pl.MemorySpace.Left
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
                shape=[BAND_ROWS, N], dtype=FP32, target_memory=pl.MemorySpace.Acc
            ),
            addrs=0,
            mutex_ids="auto",
            depth=2,
        )

        # 一次性设置：gamma 与 W 在本核所有带之间复用，不放进带循环。
        with pl.section_vector():
            gamma_tile = gamma_g.current()
            pl.load(gamma_tile, gamma, [0, 0])
        with pl.section_cube():
            w_tile = w_g.current()
            pl.load(w_tile, w, [0, 0])

        for band in pl.range(NUM_BANDS_PER_CORE):
            with pl.section_vector():
                rmsnorm_band(band, x, gamma_g, x_g, nz_lo_g, nz_hi_g, shared_p)
            with pl.section_cube():
                gemm_band(band, w_g, out, l0a, l0b, acc_g, shared_p)

    return kernel


# Serial handover control and positive-preload consistency variant.
rmsnorm_gemm_multicore = _make_kernel(preload=0, name="rmsnorm_gemm_p0")
rmsnorm_gemm_multicore_p2 = _make_kernel(preload=2, name="rmsnorm_gemm_p2")
