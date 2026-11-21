# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：语法糖（`pl.gemm`）。pl.gemm 的 trans_a / trans_b：转置作为对字节的重解释。"""

import pypto_pro.language as pl

M, K, N = 64, 128, 32


@pl.jit(auto_mutex=True)
def gemm_trans_both(
    a_t: pl.Tensor[[K, M], pl.DT_FP16],
    b_t: pl.Tensor[[N, K], pl.DT_FP16],
    out: pl.Tensor[[M, N], pl.DT_FP32],
):
    """两个操作数都转置存放，用 trans_a / trans_b 告诉 gemm。

    注意 L1 Tile 的 **shape 是 GM 里的形状**（``[K, M]`` / ``[N, K]``），layout 是
    常规的 NZ，load 也不带 ``order``——搬运照着 GM 的样子原样搬进来，转置发生在
    gemm 里面。这和 ``l1_layout`` 样例的 TT 正好相反，那里转置发生在 load。
    """
    a_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[K, M], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[N, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat
        ),
        addrs=0x4000,
        mutex_ids="auto",
        depth=1,
    )
    l0a = pl.make_tile_group(
        type=pl.TileType(
            shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    l0b = pl.make_tile_group(
        type=pl.TileType(
            shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    acc = pl.make_tile_group(
        type=pl.TileType(
            shape=[M, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )

    with pl.section_cube():
        # 必须先绑到变量：gemm 降级时要读操作数的声明形状，再求值一次表达式会让
        # TileGroup 多转一格。
        cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc.current()
        pl.load(cur_a, a_t, [0, 0])
        pl.load(cur_b, b_t, [0, 0])
        pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b, trans_a=True, trans_b=True)
        pl.store(out, ac, [0, 0])


@pl.jit(auto_mutex=True)
def gemm_trans_a_only(
    a_t: pl.Tensor[[K, M], pl.DT_FP16],
    b: pl.Tensor[[K, N], pl.DT_FP16],
    out: pl.Tensor[[M, N], pl.DT_FP32],
):
    """只有左操作数转置存放。右边照常，不受影响。"""
    a_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[K, M], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat
        ),
        addrs=0x4000,
        mutex_ids="auto",
        depth=1,
    )
    l0a = pl.make_tile_group(
        type=pl.TileType(
            shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    l0b = pl.make_tile_group(
        type=pl.TileType(
            shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    acc = pl.make_tile_group(
        type=pl.TileType(
            shape=[M, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )

    with pl.section_cube():
        cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc.current()
        pl.load(cur_a, a_t, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b, trans_a=True)
        pl.store(out, ac, [0, 0])


def make_rectangular_handover(m, dtype):
    """两个 AIV 各用 NZ [K, M/2] 源块装配 Mat ZN [M, K]，由 Cube 消费。

    每个源块覆盖完整 K，一次 insert 的行偏移为 0。B 层 preload=0 负责
    MTE3→MTE1 交接，auto_mutex 负责各核内的依赖。固定 K=128、N=64；
    test.py 验证 M=32/64 与 FP16/BF16，分离物理装配形状与逻辑矩阵形状。
    单执行组、所有 group 单槽：M=32/64 时，Mat 分别为 24/32 KiB，
    每 AIV 的 Vec 为 8/16 KiB，Left 为 4/8 KiB，Right 为 8 KiB，Acc 为 8/16 KiB。
    """
    k, n = 128, 64
    half = m // 2

    @pl.pipeline.stage
    def produce(src, nd_group, nz_group, shared):
        sub_id = pl.get_subblock_idx()
        nd = nd_group.current()
        nz = nz_group.current()
        dst = shared.current()
        pl.load(nd, src, [0, sub_id * half])
        pl.move(nz, nd)
        pl.insert(dst, nz, [0, sub_id * half])

    @pl.pipeline.stage
    def consume(rhs, out, shared, rhs_group, left, right, acc_group):
        a = shared.current()
        b = rhs_group.current()
        acc = acc_group.current()
        pl.load(b, rhs, [0, 0])
        pl.gemm(acc, a, b, l0a=left, l0b=right, splitk=64)
        pl.store(out, acc, [0, 0])

    @pl.jit(arch="a5", auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=0))
    def rectangular_handover(
        src: pl.Tensor[[k, m], dtype],
        rhs: pl.Tensor[[k, n], dtype],
        out: pl.Tensor[[m, n], pl.DT_FP32],
    ):
        shared = pl.make_tile_group(
            type=pl.TileType(
                shape=[m, k],
                dtype=dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.ZN,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
            fwd_ids=[0],
            bwd_ids=[1],
        )
        rhs_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[k, n], dtype=dtype, target_memory=pl.MemorySpace.Mat
            ),
            addrs=m * k * 2,
            mutex_ids="auto",
            depth=1,
        )
        left = pl.make_tile_group(
            type=pl.TileType(
                shape=[m, 64], dtype=dtype, target_memory=pl.MemorySpace.Left
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        right = pl.make_tile_group(
            type=pl.TileType(
                shape=[64, n], dtype=dtype, target_memory=pl.MemorySpace.Right
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        acc_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[m, n], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        nd_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[k, half], dtype=dtype, target_memory=pl.MemorySpace.Vec
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        nz_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[k, half],
                dtype=dtype,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.NZ,
            ),
            addrs=k * half * 2,
            mutex_ids="auto",
            depth=1,
        )
        for _ in pl.range(1):
            with pl.section_vector():
                produce(src, nd_group, nz_group, shared)
            with pl.section_cube():
                consume(rhs, out, shared, rhs_group, left, right, acc_group)

    return rectangular_handover
