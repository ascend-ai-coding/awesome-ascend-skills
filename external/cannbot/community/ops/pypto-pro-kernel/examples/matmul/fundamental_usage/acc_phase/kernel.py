# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始（手写链——教学 unitFlag 性能机制，糖里不可见；**出货 matmul 先走 `pl.gemm`**）。K 三段累加，一个开 unitFlag，一个不开。两者数值必须完全一样。

``AccPhase`` / ``STPhase`` 是**性能机制**（硬件 unitFlag），不改变算什么。累加本身
由 ``pl.matmul`` / ``pl.matmul_acc`` 决定，与 phase 无关——两件事凑在同一个循环里，
很容易被当成一件。
"""

import pypto_pro.language as pl

M = N = 128
K = 384  # 三段，每段 TILE
TILE = 128


@pl.jit(auto_mutex=True)
def k_split_with_phase(
    a: pl.Tensor[[M, K], pl.DT_FP16],
    b: pl.Tensor[[K, N], pl.DT_FP16],
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
                # 第一段用 matmul（init，写入），之后用 matmul_acc（累加）。
                # 这一维是 AccPhase 管不着的。
                pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
            elif k < K - TILE:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
            else:
                # 最后一次写 L0C 必须是 Final：它负责把 unit-flag 置 1。
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
        # 配了 AccPhase 就必须配 STPhase，最后一次搬出用 Final。漏掉会卡死。
        pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)


@pl.jit(auto_mutex=True)
def k_split_no_phase(
    a: pl.Tensor[[M, K], pl.DT_FP16],
    b: pl.Tensor[[K, N], pl.DT_FP16],
    out: pl.Tensor[[M, N], pl.DT_FP32],
):
    """与上面逐行相同，只是不带 phase——即 ``Unspecified``，不启用 unitFlag。"""
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
                pl.matmul(ac, al, br)
            else:
                pl.matmul_acc(ac, ac, al, br)
        pl.store(out, ac, [0, 0])
