# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：混合。单次片上交接的 ``C = silu(A @ B + bias)``。

契约：A、B 为 [256,256] FP16，bias 为 [1,256] FP32，沿 M 广播；输出为 [256,256] FP32。VF 按 ``x / (1 + exp(-x))`` 计算 SiLU。

Cube 将 A、B 搬到 L1，bias 经 L1 搬到 BiasTable，``pl.gemm(splitk=64, bias=...)`` 用 FP32 累加。L1 的 A、B、bias 分别从 0x00000、0x20000、
0x40000 开始；Left、Right、Bias、Acc、Vec 是独立地址空间，均从 0 开始。所有组显式给出地址、depth=1 和 ``mutex_ids="auto"``。

整块 Acc 占 262144B；``DualModeSplitM`` 将它直接送到两个 AIV 各自的 [128,256] FP32 UB tile（每个 131072B）。不使用 GM 中转。

交接只发生一次，没有槽位复用：Cube 在最后一次 Acc→Vec 搬运后用 FIX 设置事件 0；AIV 在第一次 VF 读取前用 V 等待同一事件。核内同步由 auto_mutex 负责；跨核事件由这里的手写 set/wait 负责。

显式以 ``kernel[None, LAUNCH_BLOCK_DIM]`` 启动一个执行组。两个 AIV 按 ``get_subblock_idx() * HALF_M`` 写入不重叠的输出半区。

``reference.py`` 从未分块的 FP32 数学定义计算参考；``test.py`` 覆盖随机 bias 和全零 bias，固定 rtol=atol=2e-2。带流水与多核切分分别由相邻两个样例覆盖。
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32
FP16 = pl.DT_FP16

M = K = N = 256
LAUNCH_BLOCK_DIM = (
    1  # 单组工作量；mix kernel 的 block_dim 是执行组数，eager 会 28 组同工（慢且浪费）
)
SK = 64  # splitk 步长：K/SK = 4 步，L0A/L0B 每槽 32768B（Left/Right 各 65536B 的一半）
HALF_M = M // 2  # 128：一个 AIV 处理的行数；半区 tile 字节数与容量分析见文件头

VL = 64  # 一个 FP32 寄存器 = 256 字节 = 64 个元素；N=256 恰好整除，没有尾块


@pl.vector_function
def _silu_rows(tile):
    """原地 silu：``tile`` 既是输入又是输出，``HALF_M`` 行、每行 ``N/VL`` 个满寄存器块。

    ``silu(x) = x / (1 + exp(-x))``。四步都是纯函数（不读 dst 前值），掩码全程
    ``MaskPattern.ALL``——列数整除寄存器宽度，不需要尾块掩码。列的循环必须也是
    ``pl.range``：kernel 体是 DSL，对静态 Python 列表 ``for off in [...]`` 在解析期
    直接拒绝（"For loop must use pl.range()"），哪怕列表内容是编译期常量；嵌套
    ``pl.range`` 本身是board 证过的写法（见 ``examples/matmul/gemm_mx/kernel.py``
    的三层 mi/ni/ki 循环）。
    """
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    for r in pl.range(HALF_M):
        for off in pl.range(0, N, VL):
            x = vf.load_align(tile, [r, off])
            neg_x = vf.neg(x, all_lanes)
            exp_neg_x = vf.exp(neg_x, all_lanes)
            denom = vf.adds(exp_neg_x, pl.const(1.0, FP32), all_lanes)
            silu = vf.div(x, denom, all_lanes)
            vf.store_align(tile, silu, all_lanes, [r, off])


@pl.jit(auto_mutex=True)
def gemm_bias_silu(
    a: pl.Tensor[[M, K], FP16],
    b: pl.Tensor[[K, N], FP16],
    bias: pl.Tensor[[1, N], FP32],
    out: pl.Tensor[[M, N], FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids="auto",
        depth=1,
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[K, N], dtype=FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000,
        mutex_ids="auto",
        depth=1,
    )
    bias_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[1, N], dtype=FP32, target_memory=pl.MemorySpace.Mat),
        addrs=0x40000,
        mutex_ids="auto",
        depth=1,
    )
    l0a = pl.make_tile_group(
        type=pl.TileType(shape=[M, SK], dtype=FP16, target_memory=pl.MemorySpace.Left),
        addrs=0,
        mutex_ids="auto",
        depth=1,
    )
    l0b = pl.make_tile_group(
        type=pl.TileType(shape=[SK, N], dtype=FP16, target_memory=pl.MemorySpace.Right),
        addrs=0,
        mutex_ids="auto",
        depth=1,
    )
    bias_l0 = pl.make_tile_group(
        type=pl.TileType(shape=[1, N], dtype=FP32, target_memory=pl.MemorySpace.Bias),
        addrs=0,
        mutex_ids="auto",
        depth=1,
    )
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[M, N], dtype=FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0,
        mutex_ids="auto",
        depth=1,
    )
    work = pl.make_tile_group(
        type=pl.TileType(
            shape=[HALF_M, N],
            dtype=FP32,
            target_memory=pl.MemorySpace.Vec,
            layout=pl.TensorLayout.ND,
        ),
        addrs=0,
        mutex_ids="auto",
        depth=1,
    )

    with pl.section_cube():
        cur_a, cur_b, cur_bias_l1 = a_l1.current(), b_l1.current(), bias_l1.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.load(cur_bias_l1, bias, [0, 0])
        bl = bias_l0.current()
        pl.move(bl, cur_bias_l1)

        ac = acc.current()
        pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b, splitk=SK, bias=bl)
        # 片上直传给两个 AIV（不经 GM）：DualModeSplitM 一步把 [256,256] 累加器按 M
        # 劈成两个 [128,256] 半区，各写进两个 AIV 各自的 work tile。
        w = work.current()
        pl.move(w, ac, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    with pl.section_vector():
        # 一次性交接，只需正向：消费侧第一次碰这块数据的 op 是 _silu_rows 里的
        # vf 读（V 流水），不是 pl.load——数据已经直接落在 UB，不再经过 GM。
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=0)
        off_m = pl.get_subblock_idx() * HALF_M
        w = work.current()
        _silu_rows(w)
        pl.store(out, w, [off_m, 0])
