# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：语法糖（`pl.gemm`）。N 维切条带：条带是累加器内部的视图，不是新的缓冲区。"""

import pypto_pro.language as pl

M, K, N = 128, 128, 128
SN = 64  # 整除的切法：128 = 64 + 64
SN_TAIL = 48  # 不整除的切法，配合下面的 N_EXTENT
N_EXTENT = 104  # 104 = 48 + 48 + 8，最后那 8 列会被向上取整到 16

# 两个 kernel 的声明故意写全：差别在 l0b 的宽度，以及尾条带的 compact。


@pl.jit(auto_mutex=True)
def split_n_even(
    a: pl.Tensor[[M, K], pl.DT_FP16],
    b: pl.Tensor[[K, N], pl.DT_FP16],
    out: pl.Tensor[[M, N], pl.DT_FP32],
):
    """128 列切成两条 64 列，正好整除，没有尾条带。"""
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
            shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    # 宽度是一条带的宽度；整除时不会收窄，所以不需要 compact。
    l0b = pl.make_tile_group(
        type=pl.TileType(
            shape=[K, SN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=2,
    )
    # 条带由 gemm 从累加器内部切出，共用地址和 mutex ID。
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
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b, splitn=SN)
        pl.store(out, ac, [0, 0])


@pl.jit(auto_mutex=True)
def split_n_tail(
    a: pl.Tensor[[M, K], pl.DT_FP16],
    b: pl.Tensor[[K, N], pl.DT_FP16],
    out: pl.Tensor[[M, N], pl.DT_FP32],
):
    """104 列切成 48 + 48 + 8：最后那条不足一整条。

    ``n`` 在 ``splitn`` 之下是**循环上界**，所以必须是编译期常量。
    """
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
            shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    # n 收窄 l0b 和 acc，这两块都必须 compact=1。
    l0b = pl.make_tile_group(
        type=pl.TileType(
            shape=[K, SN_TAIL],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Right,
            compact=1,
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=2,
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
        cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b, n=N_EXTENT, splitn=SN_TAIL)
        pl.store(out, ac, [0, 0])
