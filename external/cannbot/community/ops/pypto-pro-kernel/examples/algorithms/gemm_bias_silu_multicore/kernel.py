# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：语法糖。八个执行组分担 ``C = silu(A @ B + bias)``。

A 为 [2048,256] FP16，B 为 [256,256] FP16，bias 为 [1,256] FP32，输出为 [2048,256] FP32。每组处理连续 256 行，再切成四条 64 行的带。

``core_id = get_block_idx() // get_subblock_num()`` 在 Cube 与 Vector 两侧得到同一组号。Vector 的 block_idx 是 AIV 展平序号，需要除以每组的 AIV
数；Cube 侧除数为 1。两侧的带起点都是 ``core_id * M_PER_CORE + band * BAND_ROWS``，AIV 再加上 ``get_subblock_idx() * HALF_BAND``。各组及两个 AIV
的输出区域不重叠。

以 ``kernel[None, NUM_CORES]`` 显式启动八个执行组；mixed kernel 的 block_dim 表示组数（每组一个 AIC、两个 AIV）。

每组的资源布局、双缓冲与 Cube→Vector 交接同 [单组带流水](../gemm_bias_silu_banded/kernel.py)：L1 的 A/B/bias 从 0x00000/0x10000/0x30000 开始；
Left、Right、Bias、Acc、Vec 各自从 0 开始。局部 L0A/L0B、A 和 Acc 均为 depth=2，跨核 Vec 为 depth=2，使用 ``fwd_ids=[0,1]`` 与
``bwd_ids=[2,3]``。事件计数器属于各执行组，可在不同组内使用相同编号。核内 mutex ID 由 ``mutex_ids="auto"`` 分配。

核内两个 stage 从同一带循环调用，B/bias 在循环外各搬运一次；``DualModeSplitM`` 直接把本带 Acc 交给两个 AIV。1+1 拓扑的 preload=0/2 对照检查同步一致性。

``reference.py`` 复用未分块的 FP32 数学定义。``test.py`` 保持 rtol=atol=2e-2，覆盖全量输出、preload 与重复调用的逐位一致，以及首/中/末组的首末带边界。
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32
FP16 = pl.DT_FP16

K = N = 256
SK = 64  # splitk 步长：K/SK=4 步，与基线/banded 一致
NUM_CORES = 8
LAUNCH_BLOCK_DIM = (
    NUM_CORES  # mix kernel 的 block_dim 是执行组数（1c+2v/组），见 docstring
)
BAND_ROWS = 64  # 一条带的行数，与 banded 一致（容量论证不变）
HALF_BAND = BAND_ROWS // 2  # 32：一个 AIV 在一条带里的份额
NUM_BANDS_PER_CORE = 4
M_PER_CORE = NUM_BANDS_PER_CORE * BAND_ROWS  # 256
M = NUM_CORES * M_PER_CORE  # 2048

VL = 64  # 一个 FP32 寄存器 = 64 个元素；N=256 整除，无尾块


@pl.vector_function
def _silu_half_band(tile):
    """原地 silu：``tile`` 是 ``[HALF_BAND, N]``，一个 AIV 在一条带里的份额。"""
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    for r in pl.range(HALF_BAND):
        for off in pl.range(0, N, VL):
            x = vf.load_align(tile, [r, off])
            neg_x = vf.neg(x, all_lanes)
            exp_neg_x = vf.exp(neg_x, all_lanes)
            denom = vf.adds(exp_neg_x, pl.const(1.0, FP32), all_lanes)
            silu = vf.div(x, denom, all_lanes)
            vf.store_align(tile, silu, all_lanes, [r, off])


@pl.pipeline.stage
def cube_band(band, a, b_g, bias_g, a_g, l0a, l0b, acc_g, shared_v):
    """cube stage：取本核本带的 A 切片，整带 ``pl.gemm``，片上直传给两个 AIV。"""
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    row_base = core_id * M_PER_CORE + band * BAND_ROWS
    a_tile = a_g.next()
    pl.load(a_tile, a, [row_base, 0])
    b_tile = b_g.current()  # B 循环外只加载一次，这里只是取引用
    bias_tile = bias_g.current()
    acc = acc_g.next()
    pl.gemm(acc, a_tile, b_tile, l0a=l0a, l0b=l0b, splitk=SK, bias=bias_tile)
    slot = shared_v.next()
    pl.move(slot, acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)


@pl.pipeline.stage
def vector_band(band, out, shared_v):
    """vector stage：消费本带（本 AIV 的半区），VF silu，写回 out 对应偏移。"""
    slot = shared_v.next()
    _silu_half_band(slot)
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    row_base = core_id * M_PER_CORE + band * BAND_ROWS
    off_m = row_base + pl.get_subblock_idx() * HALF_BAND
    pl.store(out, slot, [off_m, 0])


def _make_kernel(preload: int, name: str):
    """kernel 工厂：编译产物按 kernel 名 + arch + tilingkey 缓存、不含 preload，
    每个 preload 档必须是全新的 ``@pl.jit`` 对象（banded 样例的同一条注意事项）。
    """

    @pl.jit(
        arch="a5",
        auto_mutex=True,
        name=name,
        pipeline=pl.pipeline.PipelineConfig(preload=preload),
    )
    def kernel(
        a: pl.Tensor[[M, K], FP16],
        b: pl.Tensor[[K, N], FP16],
        bias: pl.Tensor[[1, N], FP32],
        out: pl.Tensor[[M, N], FP32],
    ):
        a_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[BAND_ROWS, K], dtype=FP16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0x00000,
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
        bias_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, N], dtype=FP32, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0x30000,
            mutex_ids="auto",
            depth=1,
        )
        bias_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, N], dtype=FP32, target_memory=pl.MemorySpace.Bias
            ),
            addrs=0,
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
        shared_v = pl.make_tile_group(
            type=pl.TileType(
                shape=[HALF_BAND, N],
                dtype=FP32,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.TensorLayout.ND,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=2,
            fwd_ids=[0, 1],
            bwd_ids=[2, 3],
        )

        # 一次性设置：B 与 bias 在本核所有带之间复用，不放进带循环。
        with pl.section_cube():
            b_tile = b_g.current()
            pl.load(b_tile, b, [0, 0])
            bias_l1_tile = bias_l1.current()
            pl.load(bias_l1_tile, bias, [0, 0])
            bias_tile = bias_g.current()
            pl.move(bias_tile, bias_l1_tile)

        for band in pl.range(NUM_BANDS_PER_CORE):
            with pl.section_cube():
                cube_band(band, a, b_g, bias_g, a_g, l0a, l0b, acc_g, shared_v)
            with pl.section_vector():
                vector_band(band, out, shared_v)

    return kernel


# 出货主形态：preload=0，局部 buffer 全部双缓冲。
gemm_bias_silu_multicore = _make_kernel(preload=0, name="gemm_bias_silu_mc_p0")

# Layer B 验证纪律要求的第二档：只改 preload，其余全同，用于「结果逐位一致」对照。
gemm_bias_silu_multicore_p2 = _make_kernel(preload=2, name="gemm_bias_silu_mc_p2")
