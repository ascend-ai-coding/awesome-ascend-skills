# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：混合（手写与 `pl.gemm` 各写一版）。K 维分块累加：首块写入，其余累加。"""

import pypto_pro.language as pl

M, N = 128, 128
K = 256  # 归约维
SK = 128  # 每步归约多少


def _groups():
    """两个 kernel 共用的五条声明。

    L0A / L0B 是双槽的：K 的两步交替使用，前一步还在算时后一步就能搬进来。
    累加器只有一槽——K 的每一步都往同一块 L0C 上累加，这正是「分块归约」的定义。
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
        addrs=0x10000,
        mutex_ids="auto",
        depth=1,
    )
    l0a = pl.make_tile_group(
        type=pl.TileType(
            shape=[M, SK], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=2,
    )
    l0b = pl.make_tile_group(
        type=pl.TileType(
            shape=[SK, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=2,
    )
    acc = pl.make_tile_group(
        type=pl.TileType(
            shape=[M, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    return a_l1, b_l1, l0a, l0b, acc


@pl.jit(auto_mutex=True)
def split_k_by_hand(
    a: pl.Tensor[[M, K], pl.DT_FP16],
    b: pl.Tensor[[K, N], pl.DT_FP16],
    out: pl.Tensor[[M, N], pl.DT_FP32],
):
    """手写 K 循环：第一步 matmul，之后 matmul_acc。"""
    a_l1, b_l1, l0a, l0b, acc = _groups()
    with pl.section_cube():
        cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        for ki in pl.range(0, K, SK):
            al, br = l0a.next(), l0b.next()
            # 从 L1 上按 K 的偏移取这一步的切片。
            pl.move(al, cur_a, offset=[0, ki])
            pl.move(br, cur_b, offset=[ki, 0])
            if ki == 0:
                pl.matmul(ac, al, br)  # 首块：写入 L0C
            else:
                pl.matmul_acc(ac, ac, al, br)  # 其余：累加进 L0C
        pl.store(out, ac, [0, 0])


@pl.jit(auto_mutex=True)
def split_k_by_gemm(
    a: pl.Tensor[[M, K], pl.DT_FP16],
    b: pl.Tensor[[K, N], pl.DT_FP16],
    out: pl.Tensor[[M, N], pl.DT_FP32],
):
    """``splitk=SK``：上面那个循环由 gemm 展开。"""
    a_l1, b_l1, l0a, l0b, acc = _groups()
    with pl.section_cube():
        cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b, splitk=SK)
        pl.store(out, ac, [0, 0])
