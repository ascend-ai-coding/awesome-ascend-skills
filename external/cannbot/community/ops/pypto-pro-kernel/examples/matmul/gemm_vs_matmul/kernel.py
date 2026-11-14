# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：混合。同一个乘法写两遍：手写，以及经 pl.gemm。"""

import pypto_pro.language as pl

M = K = N = 128


def _groups():
    """两个 kernel 共用的五条声明。

    普通 Python 函数，由 parser 内联。放在这里是为了让两个 kernel **只有函数体
    不同**——那正是要对照的东西。
    """
    a_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat
        ),
        addrs=0,
        mutex_ids="auto",
        depth=1,
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat
        ),
        addrs=M * K * 2,
        mutex_ids="auto",
        depth=1,
    )
    l0a = pl.make_tile_group(
        type=pl.TileType(
            shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left
        ),
        addrs=0,
        mutex_ids="auto",
        depth=1,
    )
    l0b = pl.make_tile_group(
        type=pl.TileType(
            shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right
        ),
        addrs=0,
        mutex_ids="auto",
        depth=1,
    )
    acc = pl.make_tile_group(
        type=pl.TileType(
            shape=[M, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc
        ),
        addrs=0,
        mutex_ids="auto",
        depth=1,
    )
    return a_l1, b_l1, l0a, l0b, acc


@pl.jit(auto_mutex=True)
def matmul_by_hand(
    a: pl.Tensor[[M, K], pl.DT_FP16],
    b: pl.Tensor[[K, N], pl.DT_FP16],
    out: pl.Tensor[[M, N], pl.DT_FP32],
):
    a_l1, b_l1, l0a, l0b, acc = _groups()
    with pl.section_cube():
        cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.move(l0a.current(), cur_a)  # 这三行
        pl.move(l0b.current(), cur_b)  # 就是 pl.gemm
        pl.matmul(ac, l0a.current(), l0b.current())  # 所代表的全部
        pl.store(out, ac, [0, 0])


@pl.jit(auto_mutex=True)
def matmul_by_gemm(
    a: pl.Tensor[[M, K], pl.DT_FP16],
    b: pl.Tensor[[K, N], pl.DT_FP16],
    out: pl.Tensor[[M, N], pl.DT_FP32],
):
    a_l1, b_l1, l0a, l0b, acc = _groups()
    with pl.section_cube():
        cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        # L0 是以 TileGroup 而不是 Tile 的形式传进去的：gemm 每个 split 步骤从里面
        # 取一次 next()，所以缓冲深度仍然由调用方决定。
        pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b)
        pl.store(out, ac, [0, 0])
