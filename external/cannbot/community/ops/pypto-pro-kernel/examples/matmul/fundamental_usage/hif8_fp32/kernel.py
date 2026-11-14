# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始（教学点在低精度参考怎么写，不在 matmul 链；**出货 matmul 先走 `pl.gemm`**）。HF8 操作数、FP32 累加、FP32 输出。

`pl.load` 和 `pl.move` **从不做类型转换**，所以从 GM 到 L1 到 L0A/L0B 整条操作数通路
都是 HF8；只有累加器是 FP32。这条是这个样例的主张，有一条测试直接在 IR 上验。
"""

import pypto_pro.language as pl

M, N, K = 128, 128, 128
TILE = 64  # M/N/K 三个方向都是 2 轮，都整除，不牵扯尾块


@pl.jit(arch="a5", auto_mutex=True)
def matmul_hf8(
    a: pl.Tensor[[M, K], pl.DT_HF8],
    b: pl.Tensor[[K, N], pl.DT_HF8],
    out: pl.Tensor[[M, N], pl.DT_FP32],
):
    """HF8 在 torch 里没有对应 dtype，调用方传的是 `uint8`——一字节一个元素的裸码。"""
    # `addrs` 给一个标量 = 「从这里开始连续摆 depth 块」，第二块的地址由框架算。
    # 本库别处多是手写地址表，这里用标量形式：块大小是 dtype 决定的，手算容易错。
    a_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE],
            dtype=pl.DT_HF8,
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
            dtype=pl.DT_HF8,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
        ),
        addrs=0x40000,
        mutex_ids="auto",
        depth=2,
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE],
            dtype=pl.DT_HF8,
            target_memory=pl.MemorySpace.Left,
            layout=pl.NZ,
        ),
        addrs=0x00000,
        mutex_ids="auto",
        depth=2,
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE],
            dtype=pl.DT_HF8,
            target_memory=pl.MemorySpace.Right,
            layout=pl.ZN,
        ),
        addrs=0x00000,
        mutex_ids="auto",
        depth=2,
    )
    # 唯一一块 FP32：K 方向的累加在这里做，这也是结果能被紧容差比较的原因。
    c_l0c = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            fractal=1024,
        ),
        addrs=0x00000,
        mutex_ids="auto",
        depth=1,
    )

    with pl.section_cube():
        for mi in pl.range(0, M, TILE):
            for ni in pl.range(0, N, TILE):
                acc = c_l0c.current()
                for ki in pl.range(0, K, TILE):
                    cur_a = a_l1.next()
                    cur_b = b_l1.next()
                    left = a_l0a.next()
                    right = b_l0b.next()
                    pl.load(cur_a, a, [mi, ki])
                    pl.load(cur_b, b, [ki, ni])
                    pl.move(left, cur_a)  # HF8 -> HF8，不转换
                    pl.move(right, cur_b)
                    if ki == 0:
                        pl.matmul(acc, left, right, phase=pl.AccPhase.Partial)
                    else:
                        pl.matmul_acc(acc, acc, left, right, phase=pl.AccPhase.Final)
                pl.store(out, acc, [mi, ni], phase=pl.STPhase.Final)
