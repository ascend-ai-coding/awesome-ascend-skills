# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：语法糖（`pl.gemm`）。任意 M/N/K：形状从张量读出来，一个编译产物服务所有形状。

和 `tail_extent` 的区别只有一条：它的形状是**解析期常量**，
这里是运行期从 `out.shape[0]` 读出来的。Tile 的**容量**仍然是编译期常量——cube 指令要求
如此，动态的只有 GM 的尺寸。
"""

import pypto_pro.language as pl

TILE = 64  # Tile 容量，编译期常量；GM 的 M/N/K 三个方向都可以不是它的倍数


@pl.jit(arch="a5", auto_mutex=True)
def matmul_dynamic(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE],
            dtype=pl.DT_FP16,
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
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
        ),
        addrs=0x40000,
        mutex_ids="auto",
        depth=2,
    )
    # 会被 gemm 收窄的三块都要 compact=1，少了 gemm 会拒（`_gemm_require_compact`）。
    # 拒的理由是它挡掉的那个静默错误：没有 compact 时 tile 保持整块的物理布局，而乘法和
    # 搬出按收窄后的范围走，元素落错位置且**边界仍然正确**，所以后面没有检查能抓到。
    l0a = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE],
            dtype=pl.DT_FP16,
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
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Right,
            layout=pl.ZN,
            compact=1,
        ),
        addrs=0x00000,
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
        # 三个上界都是运行期值。
        mt = out.shape[0]
        nt = out.shape[1]
        kt = a.shape[1]
        for mi in pl.range(0, mt, TILE):
            m_valid = pl.min(TILE, mt - mi)
            for ni in pl.range(0, nt, TILE):
                n_valid = pl.min(TILE, nt - ni)
                cur_acc = acc.current()
                for ki in pl.range(0, kt, TILE):
                    k_valid = pl.min(TILE, kt - ki)
                    cur_a = a_l1.next()
                    cur_b = b_l1.next()
                    # L1 的收窄是调用方的事：它决定从 GM 搬进来多少。gemm 故意不碰 L1。
                    pl.set_validshape(cur_a, [m_valid, k_valid])
                    pl.set_validshape(cur_b, [k_valid, n_valid])
                    pl.load(cur_a, a, [mi, ki])
                    pl.load(cur_b, b, [ki, ni])
                    # l0a / l0b / acc 的收窄由 gemm 按 m/n/k 自己设，三块不会各说各话。
                    # init 必须是解析期的 bool，所以这里是两臂 if，不是 init=(ki == 0)。
                    if ki == 0:
                        pl.gemm(
                            cur_acc,
                            cur_a,
                            cur_b,
                            l0a=l0a,
                            l0b=l0b,
                            m=m_valid,
                            n=n_valid,
                            k=k_valid,
                            init=True,
                        )
                    else:
                        pl.gemm(
                            cur_acc,
                            cur_a,
                            cur_b,
                            l0a=l0a,
                            l0b=l0b,
                            m=m_valid,
                            n=n_valid,
                            k=k_valid,
                            init=False,
                        )
                pl.store(out, cur_acc, [mi, ni])
