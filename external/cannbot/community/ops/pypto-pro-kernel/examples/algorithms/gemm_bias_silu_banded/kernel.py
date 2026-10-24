# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：语法糖。四条带的流水 ``C = silu(A @ B + bias)``。

契约与 [单次交接样例](../gemm_bias_silu/kernel.py) 相同：A、B 为 [256,256] FP16，bias 为 [1,256] FP32，输出为 [256,256] FP32。M 按
BAND_ROWS=64 切成四条带，每带用 splitk=64 的 GEMM 和 VF SiLU。

Cube、Vector 各有一个 ``@pl.pipeline.stage``，从同一个带循环调用。Cube 用 ``DualModeSplitM`` 将 [64,256] FP32 Acc 直接送给两个 AIV，各消费 [32,256]
FP32 UB tile，不经过 GM。

资源地址与深度：

- L1：A 带缓冲从 0x00000 开始，32768B/槽、depth=2；B 从 0x10000 开始，131072B、depth=1；bias 从 0x30000 开始，1024B、depth=1。B/bias 在循环外只搬运一次。
- Left、Right、Bias、Acc、Vec 分属独立地址空间，均从 0 开始。L0A 为 8192B/槽，L0B 为 32768B/槽，深度由 l0_depth 指定；depth=2 时 L0B 恰好占满 65536B。Acc 为
65536B/槽、depth=2。
- 跨核 Vec 为 32768B/槽、depth=2，覆盖生产/消费 stage 的一拍距离。``fwd_ids=[0,1]`` 表示每槽的数据就绪，``bwd_ids=[2,3]`` 表示消费完成，防止覆盖仍在读取的槽。

所有组使用显式地址与 ``mutex_ids="auto"``；自动 mutex 负责核内依赖，流水根据显式 fwd/bwd 事件生成跨核同步。两个 AIV 均参与交接，各写本带一半的输出。

该 1+1 拓扑没有同核后续 stage，所以 preload=0/2 的对照检查跨核同步一致性；局部 L0 深度 1/2 的对照检查缓冲配置。显式启动一个执行组，带间重叠由双槽交接提供。

``test.py`` 使用与单次交接样例相同的 FP32 参考及 rtol=atol=2e-2，并检查 preload 逐位一致、重复调用逐位一致和每条带的输出。完整调用的性能使用 skill 的原生任务计时工具在当前工具链与设备上单独测
量。
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32
FP16 = pl.DT_FP16

M = K = N = 256
LAUNCH_BLOCK_DIM = (
    1  # 单组工作量；mix kernel 的 block_dim 是执行组数，eager 会 28 组同工（慢且浪费）
)
BAND_ROWS = 64  # 一条带的行数
NUM_BANDS = M // BAND_ROWS  # 4 条带，M 整除，无尾块
HALF_BAND = BAND_ROWS // 2  # 32：一个 AIV 在一条带里分到的行数
SK = 64  # splitk 步长：K/SK = 4 步（与基线一致）

VL = 64  # 一个 FP32 寄存器 = 64 个元素；N=256 整除，无尾块


@pl.vector_function
def _silu_half_band(tile):
    """原地 silu：``tile`` 是 ``[HALF_BAND, N]``，一个 AIV 在一条带里的份额。

    与基线 ``_silu_rows`` 逐字同构（只是行数从 ``HALF_M=128`` 换成 ``HALF_BAND=32``）——
    这一段已经上板验证过，本文件不重新论证它，只论证"分带 + 流水"这个新加的维度。
    """
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
    """cube stage：取本带的 A 切片，整带 ``pl.gemm``（bias 随 matmul 融合），片上直传给两个 AIV。"""
    a_tile = a_g.next()
    pl.load(a_tile, a, [band * BAND_ROWS, 0])
    b_tile = b_g.current()  # B 循环外只加载一次，这里只是取引用，不重复搬运
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
    off_m = band * BAND_ROWS + pl.get_subblock_idx() * HALF_BAND
    pl.store(out, slot, [off_m, 0])


def _make_kernel(preload: int, l0_depth: int, name: str):
    """kernel 工厂：每次调用产出一个**全新**的 ``@pl.jit`` 函数对象。

    "每个 preload/depth 组合一个独立 kernel"——编译产物按 kernel 名 + arch + tilingkey
    缓存，不含 preload/depth，复用同一个对象改参数重调有喂到旧二进制的风险。
    stage 函数（``cube_band``/``vector_band``）在模块级只定义
    一次，三个变体共用——stage 的同步信息挂在函数名上，共用没有问题；变的只是
    调用它们的 kernel 外壳里 tile group 的 depth 和 ``PipelineConfig.preload``。
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
            depth=l0_depth,
        )
        l0b = pl.make_tile_group(
            type=pl.TileType(
                shape=[SK, N], dtype=FP16, target_memory=pl.MemorySpace.Right
            ),
            addrs=0,
            mutex_ids="auto",
            depth=l0_depth,
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

        # B 与 bias 在所有带之间复用，只在带循环外搬运一次。
        with pl.section_cube():
            b_tile = b_g.current()
            pl.load(b_tile, b, [0, 0])
            bias_l1_tile = bias_l1.current()
            pl.load(bias_l1_tile, bias, [0, 0])
            bias_tile = bias_g.current()
            pl.move(bias_tile, bias_l1_tile)

        for band in pl.range(NUM_BANDS):
            with pl.section_cube():
                cube_band(band, a, b_g, bias_g, a_g, l0a, l0b, acc_g, shared_v)
            with pl.section_vector():
                vector_band(band, out, shared_v)

    return kernel


# 出货主形态：preload=0（"从 preload=0 起步"），局部 buffer 全部双缓冲。
gemm_bias_silu_banded = _make_kernel(
    preload=0, l0_depth=2, name="gemm_bias_silu_banded_p0"
)

# Layer B 验证纪律要求的第二档：只改 preload，其余全同，用于"结果逐位一致"对照。
gemm_bias_silu_banded_p2 = _make_kernel(
    preload=2, l0_depth=2, name="gemm_bias_silu_banded_p2"
)

# 局部 buffer 深度杠杆对照：只把 L0A/L0B 从双缓冲改单缓冲，其余全同（含 preload=0）。
gemm_bias_silu_banded_l0depth1 = _make_kernel(
    preload=0, l0_depth=1, name="gemm_bias_silu_banded_l0depth1"
)
