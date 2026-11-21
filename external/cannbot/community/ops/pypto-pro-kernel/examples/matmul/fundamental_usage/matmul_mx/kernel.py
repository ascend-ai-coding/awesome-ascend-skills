# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始（教学 MX plane 逐槽约束——糖接管了它但 debug 要懂；**出货先走 `pl.gemm_mx`**）。MX 矩阵乘：FP8 尾数 + E8M0 指数面，手写 `pl.matmul_mx`。

九个 tile group，其中**两个的地址不是自由的**——ScaleLeft / ScaleRight 里那块必须坐在它
配对的 L0 数据 tile 地址 `>> 4`。框架查这条，但在 op 上求值时一个多槽操作数只带着**槽 0**
的地址，于是它验的是基址、放过槽间距（`a5c48ce63` 起 parser 侧补了逐槽检查）。这里手写的
地址在 `test.py` 里仍有一条专门的断言——样例自己钉住不变量，不依赖框架版本。

两个槽是刻意的：一个槽时 plane 地址恒为 0，那条规则看不出来。
"""

import pypto_pro.language as pl

M, N, K = 128, 128, 128
TILE = 64  # M/N/K 三个方向都是 2 轮
GROUP_SIZE = 32  # 一个 E8M0 管 32 个元素
SCALE_K = TILE // GROUP_SIZE  # 一个 K tile 里几个组 = 2
PAIR = GROUP_SIZE * 2  # GM 上 scale 的 K 轴单位：一个 [.., 2] 装两个组

FP8 = pl.DT_FP8E4M3FN
E8M0 = pl.DT_FP8E8M0

# L0 数据 tile 的字节数：FP8 一字节一个元素。
_L0_BYTES = TILE * TILE  # 4096 = 0x1000
# plane 的槽间距 = 数据的槽间距 >> 4。**不等于 plane 自己的大小**（它只有 128 字节）,
# 所以这里必须写成地址表，不能给一个标量让框架按自己的 size 递推。
_PLANE_SLOT = _L0_BYTES >> 4  # 256 = 0x100


@pl.jit(arch="a5", auto_mutex=True)
def matmul_mx(
    a: pl.Tensor[[M, K], FP8],
    b: pl.Tensor[[K, N], FP8],
    scale_a: pl.Tensor[[M, K // PAIR, 2], E8M0],
    scale_b: pl.Tensor[[K // PAIR, N, 2], E8M0],
    out: pl.Tensor[[M, N], pl.DT_FP32],
):
    """FP8 在 torch 里没有 dtype，操作数以 uint8 裸码传进来。"""
    a_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE],
            dtype=FP8,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
        ),
        addrs=0x00000,
        mutex_ids="auto",
        depth=2,
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE],
            dtype=FP8,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
        ),
        addrs=0x10000,
        mutex_ids="auto",
        depth=2,
    )
    # L1 上的 scale 也是 Mat，但布局是固定的：A 侧 ZZ，B 侧 NN。
    sa_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, SCALE_K],
            dtype=E8M0,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZZ,
        ),
        addrs=0x20000,
        mutex_ids="auto",
        depth=2,
    )
    sb_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[SCALE_K, TILE],
            dtype=E8M0,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NN,
        ),
        addrs=0x21000,
        mutex_ids="auto",
        depth=2,
    )

    l0a = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE],
            dtype=FP8,
            target_memory=pl.MemorySpace.Left,
            layout=pl.NZ,
            compact=1,
        ),
        addrs=0x00000,
        mutex_ids="auto",
        depth=2,
    )
    l0b = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE],
            dtype=FP8,
            target_memory=pl.MemorySpace.Right,
            layout=pl.ZN,
            compact=1,
        ),
        addrs=0x00000,
        mutex_ids="auto",
        depth=2,
    )
    # 两块 plane。形状是数据 tile 的函数（[M, K/32] 和 [K/32, N]），地址是数据地址 >> 4。
    plane_a = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, SCALE_K],
            dtype=E8M0,
            target_memory=pl.MemorySpace.ScaleLeft,
            layout=pl.ZZ,
            compact=1,
        ),
        addrs=[0x00000, _PLANE_SLOT],
        mutex_ids="auto",
        depth=2,
    )
    plane_b = pl.make_tile_group(
        type=pl.TileType(
            shape=[SCALE_K, TILE],
            dtype=E8M0,
            target_memory=pl.MemorySpace.ScaleRight,
            layout=pl.NN,
            compact=1,
        ),
        addrs=[0x00000, _PLANE_SLOT],
        mutex_ids="auto",
        depth=2,
    )

    acc = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            fractal=1024,
            compact=1,
        ),
        addrs=0x00000,
        mutex_ids="auto",
        depth=1,
    )

    with pl.section_cube():
        for mi in pl.range(0, M, TILE):
            for ni in pl.range(0, N, TILE):
                cur_acc = acc.current()
                for ki in pl.range(0, K, TILE):
                    cur_a = a_l1.next()
                    cur_b = b_l1.next()
                    cur_sa = sa_l1.next()
                    cur_sb = sb_l1.next()
                    left = l0a.next()
                    right = l0b.next()
                    pa = plane_a.next()
                    pb = plane_b.next()
                    pl.load(cur_a, a, [mi, ki])
                    pl.load(cur_b, b, [ki, ni])
                    # scale 的 K 轴单位是 PAIR=64，不是 GROUP_SIZE=32——GM 上一个
                    # [.., 2] 里装了两个组。
                    pl.load(cur_sa, scale_a, [mi, ki // PAIR, 0])
                    pl.load(cur_sb, scale_b, [ki // PAIR, ni, 0])
                    # 四条 move。数据那条和它 plane 那条在硬件上是同一条
                    # dma.l1_to_l0.mx，所以成对写。
                    pl.move(left, cur_a)
                    pl.move(pa, cur_sa)
                    pl.move(right, cur_b)
                    pl.move(pb, cur_sb)
                    if ki == 0:
                        pl.matmul_mx(cur_acc, left, right, pa, pb)
                    else:
                        pl.matmul_mx_acc(cur_acc, cur_acc, left, right, pa, pb)
                pl.store(out, cur_acc, [mi, ni])
