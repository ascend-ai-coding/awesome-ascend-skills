# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始（手写链——教学 bias 通路与 K 分段语义；**出货 matmul 先走 `pl.gemm`**，本样例勿作常规范式照抄）。matmul 融合 bias：单次，以及 K 分段时 bias 该放哪一段。

bias 是 ``[1, N]``，沿 M 广播到每一行。它走的通路和 a / b 不一样：
GM -> L1(Mat) -> **L0(Bias)**，而且 L0 那块必须是 FP32——搬运顺带做类型转换。
"""

import pypto_pro.language as pl

M = N = 128
K = 384  # 三段
TILE = 128


@pl.jit(auto_mutex=True)
def matmul_bias(
    a: pl.Tensor[[M, TILE], pl.DT_FP16],
    b: pl.Tensor[[TILE, N], pl.DT_FP16],
    bias: pl.Tensor[[1, N], pl.DT_FP16],
    out: pl.Tensor[[M, N], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[M, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=2,
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat
        ),
        addrs=0x10000,
        mutex_ids="auto",
        depth=2,
    )
    # L1 上的 bias 跟着输入的 dtype 走（FP16）。
    bias_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat
        ),
        addrs=0x20000,
        mutex_ids="auto",
        depth=1,
    )
    l0a = pl.make_tile_group(
        type=pl.TileType(
            shape=[M, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=2,
    )
    l0b = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=2,
    )
    # L0 上的 bias 必须是 Bias 空间 + FP32。move 做 FP16 -> FP32 转换。
    bias_l0 = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Bias
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
        cur_a, cur_b = a_l1.next(), b_l1.next()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        # bias 不能直接 load 进 Bias 空间：pl.load 只支持 Vec / Mat。
        cur_bias_l1 = bias_l1.next()
        pl.load(cur_bias_l1, bias, [0, 0])
        al, br = l0a.next(), l0b.next()
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        bl = bias_l0.next()
        pl.move(bl, cur_bias_l1)
        ac = acc.current()
        pl.matmul(ac, al, br, bl)
        pl.store(out, ac, [0, 0])


@pl.jit(auto_mutex=True)
def matmul_bias_k_split(
    a: pl.Tensor[[M, K], pl.DT_FP16],
    b: pl.Tensor[[K, N], pl.DT_FP16],
    bias: pl.Tensor[[1, N], pl.DT_FP16],
    out: pl.Tensor[[M, N], pl.DT_FP32],
):
    """K 分三段。bias 只能挂在**第一段**上。

    ``pl.matmul(ac, a, b, bias)`` 是**覆盖写**累加器，不是累加。挂到中间段或最后
    一段会把前面几段的部分和冲掉——框架直接拒绝这个写法，不会静默算错。
    """
    a_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[M, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=2,
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat
        ),
        addrs=0x10000,
        mutex_ids="auto",
        depth=2,
    )
    bias_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat
        ),
        addrs=0x20000,
        mutex_ids="auto",
        depth=1,
    )
    l0a = pl.make_tile_group(
        type=pl.TileType(
            shape=[M, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=2,
    )
    l0b = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=2,
    )
    bias_l0 = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Bias
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
        ac = acc.current()
        for k in pl.range(0, K, TILE):
            cur_a, cur_b = a_l1.next(), b_l1.next()
            al, br = l0a.next(), l0b.next()
            pl.load(cur_a, a, [0, k])
            pl.load(cur_b, b, [k, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if k == 0:
                cur_bias_l1 = bias_l1.next()
                pl.load(cur_bias_l1, bias, [0, 0])
                bl = bias_l0.next()
                pl.move(bl, cur_bias_l1)
                # 第一段：覆盖写，顺带把 bias 铺进去。
                pl.matmul(ac, al, br, bl)
            else:
                # 其余各段：累加，不再提 bias——它已经在累加器里了。
                pl.matmul_acc(ac, ac, al, br)
        pl.store(out, ac, [0, 0])
