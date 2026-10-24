# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TT：两个操作数都转置存放，两块 L1 Tile 都是 ZN。

结果始终是 ``C[M, N] = A[M, K] @ B[K, N]``——L1 的 layout 与 GM 的存放方向配套变化，
**L0A / L0B / L0C 的 layout 一个都不变**，因为那三个由 cube 单元决定。
"""

import pypto_pro.language as pl

M, K, N = 64, 128, 32


@pl.jit(auto_mutex=True)
def matmul_tt(
    a: pl.Tensor[[K, M], pl.DT_FP16],
    b: pl.Tensor[[N, K], pl.DT_FP16],
    out: pl.Tensor[[M, N], pl.DT_FP32],
):
    # Tile 的形状始终是逻辑操作数的形状：变的是 GM 的存放方向和 layout。
    a_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[M, K],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZN,
        ),
        addrs=0x00000,
        mutex_ids="auto",
        depth=1,
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[K, N],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZN,
        ),
        addrs=0x10000,
        mutex_ids="auto",
        depth=1,
    )
    # 以下三块与 GM 存放方向无关。
    l0a = pl.make_tile_group(
        type=pl.TileType(
            shape=[M, K],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Left,
            layout=pl.NZ,
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    l0b = pl.make_tile_group(
        type=pl.TileType(
            shape=[K, N],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Right,
            layout=pl.ZN,
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    acc = pl.make_tile_group(
        type=pl.TileType(
            shape=[M, N],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            fractal=1024,
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )

    with pl.section_cube():
        cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc.current()
        pl.load(cur_a, a, [0, 0], order=[1, 0])
        pl.load(cur_b, b, [0, 0], order=[1, 0])
        pl.move(l0a.current(), cur_a)
        pl.move(l0b.current(), cur_b)
        pl.matmul(ac, l0a.current(), l0b.current())
        pl.store(out, ac, [0, 0])
