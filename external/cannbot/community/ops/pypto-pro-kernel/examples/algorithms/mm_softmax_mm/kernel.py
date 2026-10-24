# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：语法糖。 Fused softmax matrix chain with native GEMM, VF and a three-stage pipeline.

Contract: O = softmax(A @ B, dim=-1) @ C. A[1024,256], B[256,256] and C[256,256] are FP16; O[1024,256] is FP32. Scores
and normalization use FP32; probabilities narrow to FP16 before the second GEMM. This is a fixed-shape, unscaled chain
with all keys valid, distinct from the batched/scaled/prefix-validity materialized example.

Eight execution groups own 128 rows each, in two 64-row bands. Each band's Cube -> Vector -> Cube pipeline computes
scores, performs stable row softmax and NZ packing on two 32-row AIV halves, then consumes the shared FP16 probability
tile. Both handovers stay on chip.

Explicit byte ranges (ends exclusive):

| Space | Groups and ranges | Bytes/slot x depth |
| --- | --- | --- |
| Mat | a_g [0,0x10000) | 32768 x 2 |
| Mat | b_g [0x10000,0x30000), c_g [0x30000,0x50000) | 131072 x 1 each |
| Mat | shared_p [0x50000,0x60000) | 32768 x 2 |
| Left | l0a1 [0,0x2000), l0a2 [0x2000,0x4000) | 8192 x 1 each |
| Right | l0b1 [0,0x8000), l0b2 [0x8000,0x10000) | 32768 x 1 each |
| Acc | acc1_g [0,0x20000), acc2_g [0x20000,0x30000) | 65536 x 2 / 1 |
| Vec | shared_v [0,0x10000) | 32768 x 2 |
| Vec | nz_lo_g [0x10000,0x14200), nz_hi_g [0x14200,0x18400) | 8448 x 2 each |

All groups use automatic intra-core mutex IDs. shared_v declares forward [0,1] (FIX -> V) and reverse [2,3] (V -> FIX);
shared_p declares forward [4,5] (MTE3 -> MTE1) and reverse [6,7] (MTE1 -> MTE3). VF reads shared_v without overwriting
it. The NZ tiles have a physical 33-row pitch and 32 valid rows; cursor stores pack two 128-column halves before insert.

The main entry uses preload=2, with preload=0 as the serial control. Stage delays at preload=2 are 0/1/2, so each
handover needs two slots. Separate L0 sets serve the two Cube stages; their two Right slots total 64 KiB. Existing tests
require rtol=atol=2e-2 against the quantization-aware CPU reference and zero-tolerance preload/repeated-call
consistency. This example makes no current performance claim.
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32
FP16 = pl.DT_FP16

M, K, N = 1024, 256, 256  # A[M,K], B[K,N], C[N,N], O[M,N]
NUM_CORES = 8
LAUNCH_BLOCK_DIM = NUM_CORES  # mix kernel 的 block_dim 是执行组数，见 docstring
M_PER_CORE = M // NUM_CORES  # 128
BAND_ROWS = 64
NUM_BANDS_PER_CORE = M_PER_CORE // BAND_ROWS  # 2
HALF_BAND = BAND_ROWS // 2  # 32：一个 AIV 在一条带里的份额
SK = 64  # splitk 步长：K/SK = N/SK = 4 步，两个 gemm 共用

VL = 64  # 一个 FP32 寄存器 = 64 个元素；N=256 整除，无尾块
NCHUNK = N // VL  # 一行拆成的寄存器块数 = 4
NZ_PITCH = HALF_BAND + 1  # 33：UB NZ 物理列高 = rows+1，奇数错开 bank
HALF_N = N // 2  # 128：一条 FP16 寄存器的列数；每列组一个 NZ tile


@pl.vector_function
def _softmax_cast_band(src, nz_lo, nz_hi):
    """Compute stable row softmax in FP32, narrow to FP16 and pack two NZ tiles.

    Three passes find the row maximum, sum shifted exponentials, then normalize.
    Adjacent 64-lane narrowed chunks are de-interleaved into dense 128-value
    registers. Cursor-based DATA_BLOCK_COPY stores use the exact 33-row pitch."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    preg16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP16)
    for r in pl.range(HALF_BAND):
        # 第一遍：行最大值，跨 4 个列块合并。
        c0 = vf.load_align(src, [r, 0])
        m0 = vf.reduce_max(c0, preg)
        row_max = vf.full(m0, preg)
        for c in pl.range(1, NCHUNK):
            chunk = vf.load_align(src, [r, c * VL])
            mc = vf.reduce_max(chunk, preg)
            mcb = vf.full(mc, preg)
            row_max = vf.max(row_max, mcb, preg)
        # 第二遍：行指数和，基于行最大值。
        c0b = vf.load_align(src, [r, 0])
        s0 = vf.sub(c0b, row_max, preg)
        e0 = vf.exp(s0, preg)
        rsum0 = vf.reduce_sum(e0, preg)
        row_sum = vf.full(rsum0, preg)
        for c in pl.range(1, NCHUNK):
            chunk = vf.load_align(src, [r, c * VL])
            sc = vf.sub(chunk, row_max, preg)
            ec = vf.exp(sc, preg)
            rsc = vf.reduce_sum(ec, preg)
            rscb = vf.full(rsc, preg)
            row_sum = vf.add(row_sum, rscb, preg)
        # 第三遍：归一化，两个相邻 64 元素 chunk 一组 astype(layout=ZERO)+de_interleave
        # 拼成稠密 128 元素 FP16 寄存器，vsstb 游标散写（见 docstring）。列组必须
        # 展开成两段（store 目标是不同 tile，tile 选择是编译期结构，进不了 pl.range）。
        # —— 列 0..127 -> nz_lo ——
        chunk_lo = vf.load_align(src, [r, 0])
        s_lo = vf.sub(chunk_lo, row_max, preg)
        e_lo = vf.exp(s_lo, preg)
        p_lo = vf.div(e_lo, row_sum, preg)
        h_lo = vf.astype(p_lo, preg, dtype=FP16, layout=pl.CastLayout.ZERO)
        chunk_hi = vf.load_align(src, [r, VL])
        s_hi = vf.sub(chunk_hi, row_max, preg)
        e_hi = vf.exp(s_hi, preg)
        p_hi = vf.div(e_hi, row_sum, preg)
        h_hi = vf.astype(p_hi, preg, dtype=FP16, layout=pl.CastLayout.ZERO)
        packed, _ = vf.de_interleave(h_lo, h_hi)
        vf.store_align(
            nz_lo,
            packed,
            preg16,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=NZ_PITCH,
            repeat_stride=1,
            post_update=True,
        )
        # —— 列 128..255 -> nz_hi ——（同配方，变量重赋值同类型）
        chunk_lo = vf.load_align(src, [r, 2 * VL])
        s_lo = vf.sub(chunk_lo, row_max, preg)
        e_lo = vf.exp(s_lo, preg)
        p_lo = vf.div(e_lo, row_sum, preg)
        h_lo = vf.astype(p_lo, preg, dtype=FP16, layout=pl.CastLayout.ZERO)
        chunk_hi = vf.load_align(src, [r, 3 * VL])
        s_hi = vf.sub(chunk_hi, row_max, preg)
        e_hi = vf.exp(s_hi, preg)
        p_hi = vf.div(e_hi, row_sum, preg)
        h_hi = vf.astype(p_hi, preg, dtype=FP16, layout=pl.CastLayout.ZERO)
        packed, _ = vf.de_interleave(h_lo, h_hi)
        vf.store_align(
            nz_hi,
            packed,
            preg16,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=NZ_PITCH,
            repeat_stride=1,
            post_update=True,
        )


@pl.pipeline.stage
def gemm1_band(band, a, b_g, a_g, l0a1, l0b1, acc1_g, shared_v):
    """cube stage 1：本带 A 行切片 @ B，累加器 DualModeSplitM 直传给两个 AIV。"""
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    row_base = core_id * M_PER_CORE + band * BAND_ROWS
    a_tile = a_g.next()
    pl.load(a_tile, a, [row_base, 0])
    b_tile = b_g.current()  # B 循环外只加载一次，这里只是取引用
    acc1 = acc1_g.next()
    pl.gemm(acc1, a_tile, b_tile, l0a=l0a1, l0b=l0b1, splitk=SK)
    slot = shared_v.next()
    pl.move(slot, acc1, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)


@pl.pipeline.stage
def softmax_band(band, shared_v, nz_lo_g, nz_hi_g, shared_p):
    """vector stage：消费本带（本 AIV 半区），VF softmax + 转 FP16 并直接散写
    row+1 pitch NZ（vsstb 游标形态，见 ``_softmax_cast_band``），按半区偏移插回
    共享 Mat 组，供下一个 cube stage 当 gemm 的左操作数。
    """
    slot = shared_v.next()
    nz_lo = nz_lo_g.next()
    nz_hi = nz_hi_g.next()
    _softmax_cast_band(slot, nz_lo, nz_hi)
    pl.set_validshape(nz_lo, [HALF_BAND, HALF_N])  # 物理 33 行，逻辑 32 行
    pl.set_validshape(nz_hi, [HALF_BAND, HALF_N])
    off = pl.get_subblock_idx() * HALF_BAND
    p_tile = shared_p.next()
    pl.insert(p_tile, nz_lo, [off, 0])  # Vec NZ -> Mat NZ，MTE3，各写一半
    pl.insert(p_tile, nz_hi, [off, HALF_N])  # 两列组两次 insert（burst 总数不变）


@pl.pipeline.stage
def gemm2_band(band, c_g, out, l0a2, l0b2, acc2_g, shared_p):
    """cube stage 2：两个 AIV 都写完后的整带 P @ C，累加器直接 store 到 out。"""
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    row_base = core_id * M_PER_CORE + band * BAND_ROWS
    p_tile = shared_p.next()
    c_tile = c_g.current()
    acc2 = acc2_g.next()
    pl.gemm(acc2, p_tile, c_tile, l0a=l0a2, l0b=l0b2, splitk=SK)
    pl.store(out, acc2, [row_base, 0])


def _make_kernel(preload: int, name: str):
    """Build separately named entries for serial and positive-preload schedules."""

    @pl.jit(
        arch="a5",
        auto_mutex=True,
        name=name,
        pipeline=pl.pipeline.PipelineConfig(preload=preload),
    )
    def kernel(
        a: pl.Tensor[[M, K], FP16],
        b: pl.Tensor[[K, N], FP16],
        c: pl.Tensor[[N, N], FP16],
        out: pl.Tensor[[M, N], FP32],
    ):
        a_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[BAND_ROWS, K], dtype=FP16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0,
            mutex_ids="auto",
            depth=2,
        )
        b_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[K, N], dtype=FP16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0x10000,
            mutex_ids="auto",
            depth=1,
        )
        c_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[N, N], dtype=FP16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0x30000,
            mutex_ids="auto",
            depth=1,
        )
        l0a1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[BAND_ROWS, SK], dtype=FP16, target_memory=pl.MemorySpace.Left
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        l0b1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[SK, N], dtype=FP16, target_memory=pl.MemorySpace.Right
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        acc1_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[BAND_ROWS, N], dtype=FP32, target_memory=pl.MemorySpace.Acc
            ),
            addrs=0,
            mutex_ids="auto",
            depth=2,
        )
        l0a2 = pl.make_tile_group(
            type=pl.TileType(
                shape=[BAND_ROWS, SK], dtype=FP16, target_memory=pl.MemorySpace.Left
            ),
            addrs=0x02000,
            mutex_ids="auto",
            depth=1,
        )
        l0b2 = pl.make_tile_group(
            type=pl.TileType(
                shape=[SK, N], dtype=FP16, target_memory=pl.MemorySpace.Right
            ),
            addrs=0x08000,
            mutex_ids="auto",
            depth=1,
        )
        acc2_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[BAND_ROWS, N], dtype=FP32, target_memory=pl.MemorySpace.Acc
            ),
            addrs=0x20000,
            mutex_ids="auto",
            depth=1,
        )
        shared_v = pl.make_tile_group(
            type=pl.TileType(
                shape=[HALF_BAND, N],
                dtype=FP32,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.ND,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=2,
            fwd_ids=[0, 1],
            bwd_ids=[2, 3],
        )
        nz_lo_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[NZ_PITCH, HALF_N],
                dtype=FP16,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.NZ,
            ),
            addrs=0x10000,
            mutex_ids="auto",
            depth=2,
        )
        nz_hi_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[NZ_PITCH, HALF_N],
                dtype=FP16,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.NZ,
            ),
            addrs=0x14200,
            mutex_ids="auto",
            depth=2,
        )
        shared_p = pl.make_tile_group(
            type=pl.TileType(
                shape=[BAND_ROWS, N],
                dtype=FP16,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
            ),
            addrs=0x50000,
            mutex_ids="auto",
            depth=2,
            fwd_ids=[4, 5],
            bwd_ids=[6, 7],
        )

        # 一次性设置：B 与 C 在本核所有带之间复用，不放进带循环。
        with pl.section_cube():
            b_tile = b_g.current()
            pl.load(b_tile, b, [0, 0])
            c_tile = c_g.current()
            pl.load(c_tile, c, [0, 0])

        for band in pl.range(NUM_BANDS_PER_CORE):
            with pl.section_cube():
                gemm1_band(band, a, b_g, a_g, l0a1, l0b1, acc1_g, shared_v)
            with pl.section_vector():
                softmax_band(band, shared_v, nz_lo_g, nz_hi_g, shared_p)
            with pl.section_cube():
                gemm2_band(band, c_g, out, l0a2, l0b2, acc2_g, shared_p)

    return kernel


# Main positive-preload entry and serial consistency control.
mm_softmax_mm_multicore = _make_kernel(preload=2, name="mm_softmax_mm_p2")
mm_softmax_mm_multicore_p0 = _make_kernel(preload=0, name="mm_softmax_mm_p0")
