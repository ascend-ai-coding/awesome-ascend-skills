# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：语法糖。 Fused SwiGLU FFN with native GEMM, VF and explicit cross-core handovers.

Contract: O = fp16(silu(X @ W1) * (X @ W3)) @ W2. X[1024,256] and all three weights[256,256] are FP16; O[1024,256] is
FP32. Projections and SiLU/product arithmetic use FP32; the intermediate narrows once to FP16 before the final GEMM.

Eight execution groups each own two 64-row bands. The first Cube stage loads X once per band and reuses the same L0A/L0B
pair for its two projections. Two separate Acc -> Vec handovers feed VF; the VF product is packed into two row+1-pitch
NZ tiles and inserted into shared Mat for the final Cube stage.

Explicit byte ranges (ends exclusive):

| Space | Groups and ranges | Bytes/slot x depth |
| --- | --- | --- |
| Mat | x_g [0,0x10000) | 32768 x 2 |
| Mat | w1_g [0x10000,0x30000), w3_g [0x30000,0x50000), w2_g [0x50000,0x70000) | 131072 x 1 each |
| Mat | shared_p [0x70000,0x80000) | 32768 x 2 |
| Left | l0a1 [0,0x2000), l0a2 [0x2000,0x4000) | 8192 x 1 each |
| Right | l0b1 [0,0x8000), l0b2 [0x8000,0x10000) | 32768 x 1 each |
| Acc | acc_y1_g [0,0x10000), acc_y3_g [0x10000,0x20000), acc_o_g [0x20000,0x30000) | 65536 x 1 each |
| Vec | shared_y1 [0,0x10000), shared_y3 [0x10000,0x20000) | 32768 x 2 each |
| Vec | nz_lo_g [0x20000,0x24200), nz_hi_g [0x24200,0x28400) | 8448 x 2 each |

All groups use automatic intra-core mutex IDs. shared_y1 uses forward [0,1], reverse [2,3]; shared_y3 uses forward
[4,5], reverse [6,7]. Both are FIX -> V with V -> FIX release. shared_p uses forward [8,9] (MTE3 -> MTE1), reverse
[10,11] (MTE1 -> MTE3). The VF projection inputs remain read-only until their reverse release.

The main entry uses preload=2, with preload=0 as the serial control. Stage delays 0/1/2 at positive preload require two
slots on each cross-core edge. Mat reaches 512 KiB, the two Right sets total 64 KiB and the three accumulators total 192
KiB; generated declarations check their capacities.

The independent reference preserves the FP16 intermediate. Numerical acceptance requires finite output and
abs(out-golden) <= 2e-2*(abs(Z16) @ abs(W2)) + 2e-2. Preload and repeated calls use zero tolerance. A CPU diagnostic
compares precision and reduction orders on its fixed seed; it does not diagnose the cause of a device mismatch or prove
synchronization correctness. This example makes no current performance claim.
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32
FP16 = pl.DT_FP16

M, K, N = 1024, 256, 256  # X[M,K], W1/W3/W2[K,N], O[M,N]
NUM_CORES = 8
LAUNCH_BLOCK_DIM = NUM_CORES  # mix kernel 的 block_dim 是执行组数，见 docstring
M_PER_CORE = M // NUM_CORES  # 128
BAND_ROWS = 64
NUM_BANDS_PER_CORE = M_PER_CORE // BAND_ROWS  # 2
HALF_BAND = BAND_ROWS // 2  # 32：一个 AIV 在一条带里的份额
SK = 64  # splitk 步长：K/SK = N/SK = 4 步，三个 gemm 共用

VL = 64  # 一个 FP32 寄存器 = 64 个元素；N=256 整除，无尾块
NCHUNK = N // VL  # 一行拆成的寄存器块数 = 4
NZ_PITCH = HALF_BAND + 1  # 33：UB NZ 物理列高 = rows+1，奇数错开 bank
HALF_N = N // 2  # 128：一条 FP16 寄存器的列数；每列组一个 NZ tile


@pl.vector_function
def _swiglu_mul_cast_band(y1, y3, nz_lo, nz_hi):
    """Apply silu(y1)*y3 in FP32 and pack FP16 into two [33,128] NZ tiles.

    Adjacent 64-value chunks narrow with CastLayout.ZERO and de-interleave
    into dense 128-value registers. Cursor stores use the physical 33-row pitch;
    the stage inserts only the 32 valid rows from each AIV."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    preg16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP16)
    for r in pl.range(HALF_BAND):
        # 列组必须展开成两段（store 目标是不同 tile，tile 选择是编译期结构，
        # 进不了 pl.range）。—— 列 0..127 -> nz_lo ——
        a_lo = vf.load_align(y1, [r, 0])
        b_lo = vf.load_align(y3, [r, 0])
        neg_lo = vf.neg(a_lo, preg)
        exp_lo = vf.exp(neg_lo, preg)
        denom_lo = vf.adds(exp_lo, pl.const(1.0, FP32), preg)
        silu_lo = vf.div(a_lo, denom_lo, preg)
        z_lo = vf.mul(silu_lo, b_lo, preg)
        h_lo = vf.astype(z_lo, preg, dtype=FP16, layout=pl.CastLayout.ZERO)

        a_hi = vf.load_align(y1, [r, VL])
        b_hi = vf.load_align(y3, [r, VL])
        neg_hi = vf.neg(a_hi, preg)
        exp_hi = vf.exp(neg_hi, preg)
        denom_hi = vf.adds(exp_hi, pl.const(1.0, FP32), preg)
        silu_hi = vf.div(a_hi, denom_hi, preg)
        z_hi = vf.mul(silu_hi, b_hi, preg)
        h_hi = vf.astype(z_hi, preg, dtype=FP16, layout=pl.CastLayout.ZERO)

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
        a_lo = vf.load_align(y1, [r, 2 * VL])
        b_lo = vf.load_align(y3, [r, 2 * VL])
        neg_lo = vf.neg(a_lo, preg)
        exp_lo = vf.exp(neg_lo, preg)
        denom_lo = vf.adds(exp_lo, pl.const(1.0, FP32), preg)
        silu_lo = vf.div(a_lo, denom_lo, preg)
        z_lo = vf.mul(silu_lo, b_lo, preg)
        h_lo = vf.astype(z_lo, preg, dtype=FP16, layout=pl.CastLayout.ZERO)

        a_hi = vf.load_align(y1, [r, 3 * VL])
        b_hi = vf.load_align(y3, [r, 3 * VL])
        neg_hi = vf.neg(a_hi, preg)
        exp_hi = vf.exp(neg_hi, preg)
        denom_hi = vf.adds(exp_hi, pl.const(1.0, FP32), preg)
        silu_hi = vf.div(a_hi, denom_hi, preg)
        z_hi = vf.mul(silu_hi, b_hi, preg)
        h_hi = vf.astype(z_hi, preg, dtype=FP16, layout=pl.CastLayout.ZERO)

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
def gemm1_band(
    band, x, w1_g, w3_g, x_g, l0a1, l0b1, acc_y1_g, acc_y3_g, shared_y1, shared_y3
):
    """cube stage 1：本带 ``X`` 行切片分别 @ ``W1``、@ ``W3``，两个累加器各自
    ``DualModeSplitM`` 直传给两个 AIV。两个 gemm 共用同一块 ``X`` L1 tile 与同一套
    L0A/L0B（容量取舍见模块 docstring）。
    """
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    row_base = core_id * M_PER_CORE + band * BAND_ROWS
    x_tile = x_g.next()
    pl.load(x_tile, x, [row_base, 0])
    w1_tile = w1_g.current()  # 权重循环外只加载一次，这里只是取引用
    w3_tile = w3_g.current()

    acc_y1 = acc_y1_g.next()
    pl.gemm(acc_y1, x_tile, w1_tile, l0a=l0a1, l0b=l0b1, splitk=SK)
    slot_y1 = shared_y1.next()
    pl.move(slot_y1, acc_y1, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)

    acc_y3 = acc_y3_g.next()
    pl.gemm(acc_y3, x_tile, w3_tile, l0a=l0a1, l0b=l0b1, splitk=SK)
    slot_y3 = shared_y3.next()
    pl.move(slot_y3, acc_y3, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)


@pl.pipeline.stage
def fused_band(band, shared_y1, shared_y3, nz_lo_g, nz_hi_g, shared_p):
    """vector stage：消费本带（本 AIV 半区）的两个 gemm1 结果，VF 逐元素
    ``silu(y1) * y3`` 并直接散写 row+1 pitch NZ（vsstb 游标形态），按半区偏移
    插回共享 Mat 组，供下一个 cube stage 当 gemm 的左操作数。
    """
    slot_y1 = shared_y1.next()
    slot_y3 = shared_y3.next()
    nz_lo = nz_lo_g.next()
    nz_hi = nz_hi_g.next()
    _swiglu_mul_cast_band(slot_y1, slot_y3, nz_lo, nz_hi)
    pl.set_validshape(nz_lo, [HALF_BAND, HALF_N])  # 物理 33 行，逻辑 32 行
    pl.set_validshape(nz_hi, [HALF_BAND, HALF_N])
    off = pl.get_subblock_idx() * HALF_BAND
    p_tile = shared_p.next()
    pl.insert(p_tile, nz_lo, [off, 0])  # Vec NZ -> Mat NZ，MTE3，各写一半
    pl.insert(p_tile, nz_hi, [off, HALF_N])  # 两列组两次 insert（burst 总数不变）


@pl.pipeline.stage
def gemm2_band(band, w2_g, out, l0a2, l0b2, acc_o_g, shared_p):
    """cube stage 2：两个 AIV 都写完后的整带 Z @ W2，累加器直接 store 到 out。"""
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    row_base = core_id * M_PER_CORE + band * BAND_ROWS
    p_tile = shared_p.next()
    w2_tile = w2_g.current()
    acc_o = acc_o_g.next()
    pl.gemm(acc_o, p_tile, w2_tile, l0a=l0a2, l0b=l0b2, splitk=SK)
    pl.store(out, acc_o, [row_base, 0])


def _make_kernel(preload: int, name: str):
    """Build separately named entries for serial and positive-preload schedules."""

    @pl.jit(
        arch="a5",
        auto_mutex=True,
        name=name,
        pipeline=pl.pipeline.PipelineConfig(preload=preload),
    )
    def kernel(
        x: pl.Tensor[[M, K], FP16],
        w1: pl.Tensor[[K, N], FP16],
        w3: pl.Tensor[[K, N], FP16],
        w2: pl.Tensor[[K, N], FP16],
        out: pl.Tensor[[M, N], FP32],
    ):
        x_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[BAND_ROWS, K], dtype=FP16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0,
            mutex_ids="auto",
            depth=2,
        )
        w1_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[K, N], dtype=FP16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0x10000,
            mutex_ids="auto",
            depth=1,
        )
        w3_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[K, N], dtype=FP16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0x30000,
            mutex_ids="auto",
            depth=1,
        )
        w2_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[K, N], dtype=FP16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0x50000,
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
        acc_y1_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[BAND_ROWS, N], dtype=FP32, target_memory=pl.MemorySpace.Acc
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        acc_y3_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[BAND_ROWS, N], dtype=FP32, target_memory=pl.MemorySpace.Acc
            ),
            addrs=0x10000,
            mutex_ids="auto",
            depth=1,
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
        acc_o_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[BAND_ROWS, N], dtype=FP32, target_memory=pl.MemorySpace.Acc
            ),
            addrs=0x20000,
            mutex_ids="auto",
            depth=1,
        )
        shared_y1 = pl.make_tile_group(
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
        shared_y3 = pl.make_tile_group(
            type=pl.TileType(
                shape=[HALF_BAND, N],
                dtype=FP32,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.ND,
            ),
            addrs=0x10000,
            mutex_ids="auto",
            depth=2,
            fwd_ids=[4, 5],
            bwd_ids=[6, 7],
        )
        nz_lo_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[NZ_PITCH, HALF_N],
                dtype=FP16,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.NZ,
            ),
            addrs=0x20000,
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
            addrs=0x24200,
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
            addrs=0x70000,
            mutex_ids="auto",
            depth=2,
            fwd_ids=[8, 9],
            bwd_ids=[10, 11],
        )

        # 一次性设置：三个权重在本核所有带之间复用，不放进带循环。
        with pl.section_cube():
            w1_tile = w1_g.current()
            pl.load(w1_tile, w1, [0, 0])
            w3_tile = w3_g.current()
            pl.load(w3_tile, w3, [0, 0])
            w2_tile = w2_g.current()
            pl.load(w2_tile, w2, [0, 0])

        for band in pl.range(NUM_BANDS_PER_CORE):
            with pl.section_cube():
                gemm1_band(
                    band,
                    x,
                    w1_g,
                    w3_g,
                    x_g,
                    l0a1,
                    l0b1,
                    acc_y1_g,
                    acc_y3_g,
                    shared_y1,
                    shared_y3,
                )
            with pl.section_vector():
                fused_band(band, shared_y1, shared_y3, nz_lo_g, nz_hi_g, shared_p)
            with pl.section_cube():
                gemm2_band(band, w2_g, out, l0a2, l0b2, acc_o_g, shared_p)

    return kernel


# Main positive-preload entry and serial consistency control.
swiglu_ffn_multicore = _make_kernel(preload=2, name="swiglu_ffn_p2")
swiglu_ffn_multicore_p0 = _make_kernel(preload=0, name="swiglu_ffn_p0")
