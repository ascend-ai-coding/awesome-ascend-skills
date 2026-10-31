# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：混合。cube 侧 ``pl.gemm`` 糖，W 的 v -> c 交接走 B 层流水（``@pl.pipeline.stage``），scale 载荷的一次性交接手写 C 层事件；向量侧全 VF。

Contract: X[1024,256] FP16, W_q[256,256] INT8, scale[256] FP32 and O[1024,256] FP32. Each of eight execution groups owns
128 output rows; launch with kernel[None,8]. Four groups intentionally leave the trailing rows unwritten, as checked by
test.py. Other launch sizes are outside this example's full-output contract.

VF widens INT8 exactly to FP16, packs each 128-column strip into row+1-pitch NZ storage and inserts it into shared Mat.
Cube uses splitk=64 and scales the FP32 accumulator during its direct GM store. There is no floating-point weight matrix
in GM.

Explicit byte ranges (ends exclusive):

| Space | Group | Range | Bytes/slot x depth |
| --- | --- | --- | --- |
| Vec | wq_g | [0,0x08000) | 32768 x 1 |
| Vec | nz_g | [0x08000,0x18200) | 33024 x 2 |
| Vec | sc_g | [0x18200,0x18400) | 512 x 1 |
| Vec | pw_g | [0x18400,0x18800) | 1024 x 1 |
| Mat | sw_g | [0,0x00800) | 1024 x 2 |
| Mat | shared_w | [0x00800,0x20800) | 65536 x 2 |
| Mat | x_g | [0x20800,0x30800) | 65536 x 1 |
| Left | l0a | [0,0x08000) | 16384 x 2 |
| Right | l0b | [0,0x08000) | 16384 x 2 |
| Acc | acc_g | [0,0x20000) | 65536 x 2 |
| Scaling | fb_g | [0,0x00800) | 1024 x 2 |

Every group uses automatic intra-core mutex IDs. The W handover declares forward events [0,1] (MTE3 -> MTE1) and reverse
events [2,3] (MTE1 -> MTE3). Its two slots serve both preload=0 and preload=2 without changing geometry or arithmetic.

The one-shot scale payload uses a separate manual event 4, MTE3 -> FIX. VF rounds each scale to fp19 with RNE,
interleaves its FP32 bits with zero, and inserts two FP32 carrier slots into Mat. Cube waits on FIX, reinterprets the
whole sw_g group as INT64 [1,128], then moves both slots into Scaling. The group alias preserves addresses, slot spacing
and mutex ownership; it is not a pipeline handover declaration.

fp19 precision, payload bit layout and the Mat -> Scaling FIX pipe are toolchain-dependent behavior. test.py validates
the own-layer reference (X @ W_q) * rne19(scale), a separate mathematical error bound, wrong-layer reference rejection,
fresh scale payloads, INT8 endpoints and exact coordinate patterns. The numerical gates use TOL_S=K*2^-24 and
SCALE_REL=2^-11. NaN/Inf/subnormal scales are outside the tested contract. This example makes no current performance
claim.
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32
FP16 = pl.DT_FP16
INT8 = pl.DT_INT8
INT64 = pl.DT_INT64
UINT32 = pl.DT_UINT32

M, K, N = 1024, 256, 256
NUM_CORES = 8
LAUNCH_BLOCK_DIM = NUM_CORES
ROWS_PER_CORE = M // NUM_CORES  # 128
STRIP = 128  # 一个 strip = 一个 FP16 原生寄存器宽度
NUM_STRIPS = N // STRIP  # 2
K_HALF = K // 2  # 每个 AIV 拓宽 strip 的一半 K 行
SK = 64  # splitk 步长
NZ_PITCH = K_HALF + 1  # row+1 pitch
VL = 64  # 一个 FP32 寄存器 = 64 元素
PAYLOAD_EVENT = 4  # Payload event; W handover uses 0..3.


@pl.vector_function
def _widen_strip(wq_src, nz, col0):
    """INT8 -> FP16（精确），逐行游标 vsstb 散写 row+1 pitch NZ。``col0`` = 本 strip 在整宽 tile 里的起始列。"""
    p8 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=INT8)
    p16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP16)
    for r in pl.range(K_HALF):
        raw = vf.load_align(wq_src, r * N + col0, dist=pl.LoadDist.UNPK_B8)
        h = vf.astype(raw, p8, dtype=FP16, layout=pl.CastLayout.ZERO)
        vf.store_align(
            nz,
            h,
            p16,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=NZ_PITCH,
            repeat_stride=1,
            post_update=True,
        )


@pl.vector_function
def _build_payload(sc, words):
    """一个 strip 的 FP32 scale ``[1, STRIP]`` -> INT64 载荷字（以 FP32 类型承载，``[1, 2 * STRIP]``）。

    每个 64 通道 chunk：位型 -> RNE 到 fp19（``bits + 0x0FFF + ((bits >> 13) & 1)`` 再清低 13 位）-> 与全零寄存器 interleave
    （偶数下标 = 低 32 位 = scale 位型，奇数下标 = 高 32 位 = 0）。VF 调用不能嵌套作参数，中间量都先绑名。
    """
    p32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    one = vf.full(1, p32, dtype=UINT32)
    for c in pl.range(STRIP // VL):
        s = vf.load_align(sc, c * VL)
        bits = vf.bit_cast(s, dtype=UINT32)
        lsb = vf.shift_right(bits, 13, p32)
        lsb1 = vf.and_(lsb, one, p32)
        bias = vf.adds(lsb1, 0x0FFF, p32)
        rounded = vf.add(bits, bias, p32)
        hi13 = vf.shift_right(rounded, 13, p32)
        m = vf.shift_left(hi13, 13, p32)
        z = vf.xor(m, m, p32)
        lo, hi = vf.interleave(m, z)
        lo_f = vf.bit_cast(lo, dtype=FP32)
        hi_f = vf.bit_cast(hi, dtype=FP32)
        vf.store_align(words, lo_f, p32, c * 2 * VL)
        vf.store_align(words, hi_f, p32, c * 2 * VL + VL)


@pl.pipeline.stage
def convert_strip(j, wq_g, nz_g, shared_w):
    sub = pl.get_subblock_idx()
    wq_tile = wq_g.current()
    nz = nz_g.next()
    _widen_strip(wq_tile, nz, j * STRIP)
    pl.set_validshape(nz, [K_HALF, STRIP])
    w_tile = shared_w.next()
    pl.insert(w_tile, nz, [sub * K_HALF, 0])


@pl.pipeline.stage
def gemm_strip(j, x_g, out, l0a, l0b, acc_g, shared_w, fb_g):
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    x_tile = x_g.current()
    w_tile = shared_w.next()
    acc = acc_g.next()
    fp_tile = fb_g.next()
    pl.gemm(acc, x_tile, w_tile, l0a=l0a, l0b=l0b, splitk=SK)
    pl.store(out, acc, [core_id * ROWS_PER_CORE, j * STRIP], scale=fp_tile)


def _make_kernel(preload: int, name: str):
    @pl.jit(
        arch="a5",
        auto_mutex=True,
        name=name,
        pipeline=pl.pipeline.PipelineConfig(preload=preload),
    )
    def kernel(
        x: pl.Tensor[[M, K], FP16],
        w_q: pl.Tensor[[K, N], INT8],
        scale: pl.Tensor[[N], FP32],
        out: pl.Tensor[[M, N], FP32],
    ):
        wq_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[K_HALF, N], dtype=INT8, target_memory=pl.MemorySpace.Vec
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        nz_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[NZ_PITCH, STRIP],
                dtype=FP16,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.NZ,
            ),
            addrs=0x08000,
            mutex_ids="auto",
            depth=2,
        )
        sc_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, STRIP], dtype=FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=0x18200,
            mutex_ids="auto",
            depth=1,
        )
        pw_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, 2 * STRIP],
                dtype=FP32,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.TensorLayout.ND,
            ),
            addrs=0x18400,
            mutex_ids="auto",
            depth=1,
        )
        sw_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, 2 * STRIP],
                dtype=FP32,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.TensorLayout.ND,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=2,
        )
        shared_w = pl.make_tile_group(
            type=pl.TileType(
                shape=[K, STRIP],
                dtype=FP16,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
            ),
            addrs=0x00800,
            mutex_ids="auto",
            depth=2,
            fwd_ids=[0, 1],
            bwd_ids=[2, 3],
        )
        x_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[ROWS_PER_CORE, K], dtype=FP16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0x20800,
            mutex_ids="auto",
            depth=1,
        )
        l0a = pl.make_tile_group(
            type=pl.TileType(
                shape=[ROWS_PER_CORE, SK], dtype=FP16, target_memory=pl.MemorySpace.Left
            ),
            addrs=0,
            mutex_ids="auto",
            depth=2,
        )
        l0b = pl.make_tile_group(
            type=pl.TileType(
                shape=[SK, STRIP], dtype=FP16, target_memory=pl.MemorySpace.Right
            ),
            addrs=0,
            mutex_ids="auto",
            depth=2,
        )
        acc_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[ROWS_PER_CORE, STRIP],
                dtype=FP32,
                target_memory=pl.MemorySpace.Acc,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=2,
        )
        fb_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, STRIP], dtype=INT64, target_memory=pl.MemorySpace.Scaling
            ),
            addrs=0,
            mutex_ids="auto",
            depth=2,
        )

        # 序幕（vector）：第一件事是把本 AIV 那半 K 行的整宽 W_q [128, 256] 一次载入（GM 连续 32 KB，两个 strip 共用）；
        # 然后每个 AIV 构造自己那个 strip（下标 = subblock）的载荷字，insert 进 Mat 的对应槽并通知。
        # 两个 AIV 都要 set（cube 侧 INTRA_BLOCK 等两路）。
        with pl.section_vector():
            sub = pl.get_subblock_idx()
            wq_full = wq_g.current()
            pl.load(wq_full, w_q, [sub * K_HALF, 0])
            sc_tile = sc_g.current()
            pl.load(sc_tile, scale, [sub * STRIP])
            words = pw_g.current()
            _build_payload(sc_tile, words)
            slot = sw_g[sub]
            pl.insert(slot, words, [0, 0])
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=PAYLOAD_EVENT)

        # 序幕（cube）：X 载入；等载荷；组级 reinterpret 成 INT64 视图搬进两个 Scaling 槽。
        with pl.section_cube():
            core_id = pl.get_block_idx() // pl.get_subblock_num()
            x_tile = x_g.current()
            pl.load(x_tile, x, [core_id * ROWS_PER_CORE, 0])
            pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=PAYLOAD_EVENT)
            sw64_g = pl.reinterpret(sw_g, dtype=INT64, shape=[1, STRIP])
            v0 = sw64_g[0]
            v1 = sw64_g[1]
            f0 = fb_g.next()
            f1 = fb_g.next()
            pl.move(f0, v0)
            pl.move(f1, v1)

        for j in pl.range(NUM_STRIPS):
            with pl.section_vector():
                convert_strip(j, wq_g, nz_g, shared_w)
            with pl.section_cube():
                gemm_strip(j, x_g, out, l0a, l0b, acc_g, shared_w, fb_g)

    return kernel


quant_gemm_w8a16 = _make_kernel(preload=0, name="quant_gemm_w8a16_p0")
quant_gemm_w8a16_p2 = _make_kernel(preload=2, name="quant_gemm_w8a16_p2")
