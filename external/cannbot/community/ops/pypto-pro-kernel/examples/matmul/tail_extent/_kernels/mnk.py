# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""三个一起。三块都被收窄，三块都要 compact。

``compact=1`` 出现在哪几块由上面这句话决定，不是随手都加。
"""

import pypto_pro.language as pl

M = N = K = 128
EM, EN, EK = 72, 88, 104  # 规整一些的有效范围
OM, ON, OK = 37, 23, 19  # 一个都不在分形边界上


@pl.jit(auto_mutex=True)
def tail_mnk(
    a: pl.Tensor[[M, K], pl.DT_FP16],
    b: pl.Tensor[[K, N], pl.DT_FP16],
    out: pl.Tensor[[M, N], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat
        ),
        addrs=0x8000,
        mutex_ids="auto",
        depth=1,
    )
    l0a = pl.make_tile_group(
        type=pl.TileType(
            shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, compact=1
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
            compact=1,
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    acc = pl.make_tile_group(
        type=pl.TileType(
            shape=[M, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, compact=1
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )

    with pl.section_cube():
        # 声明与搬运都是整块的：有效范围只作用在 gemm 这一行上，
        # 既不改 Tile 的 shape，也不改 pl.load 搬多少。
        cur_a, cur_b, cur_acc = a_l1.current(), b_l1.current(), acc.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.gemm(cur_acc, cur_a, cur_b, l0a=l0a, l0b=l0b, m=EM, n=EN, k=EK)
        pl.store(out, cur_acc, [0, 0])
