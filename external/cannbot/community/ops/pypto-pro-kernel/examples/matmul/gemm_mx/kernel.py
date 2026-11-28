# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：混合。`pl.gemm_mx`：MX 矩阵乘的语法糖，和它的手写展开并排。

两版均显式声明物理地址；糖版本使用自动 mutex ID，手写展开版保留手动 ID 作对照：

| kernel | 每个 K 步写什么 |
| --- | --- |
| `mx_sugar` | 一条 `pl.gemm_mx` |
| `mx_manual` | 4 条 `pl.next()` + 4 条 `pl.move` + 一条 `matmul_mx` |
| `mx_sugar_m_tail` | 一条 `pl.gemm_mx(m=...)`——收窄走 plane 的行轴 |

`mx_manual` 的语句顺序**照糖展开的顺序写**（数据、它的 plane、另一侧数据、它的 plane），
不是照 Pro 手写惯例那个「Left 数据、Right 数据、Left plane、Right plane」。`test.py` 里
那条 IR 比对因此是逐条对齐的比较，不是「大致一样」。

`mx_manual` 的九个 tile group 的形状和物理地址与基础 API 样例一致，plane 使用显式地址表。
糖版本的 plane 同样逐槽写成配对 L0 数据地址 `>> 4`，深度与数据组一致。
"""

import pypto_pro.language as pl

M, N, K = 128, 128, 128
TILE = 64  # M/N/K 三个方向都是 2 轮
GROUP_SIZE = 32  # 一个 E8M0 管 32 个元素
SCALE_K = TILE // GROUP_SIZE  # 一个 K tile 里几个组 = 2
PAIR = GROUP_SIZE * 2  # GM 上 scale 的 K 轴单位

M_TAIL = 48  # 收窄用：48 行，plane 的 2 个组一个不少

FP8 = pl.DT_FP8E4M3FN
E8M0 = pl.DT_FP8E8M0

_L0_BYTES = TILE * TILE  # 4096
_PLANE_SLOT = _L0_BYTES >> 4  # 256——数据槽间距 >> 4，不是 plane 自己的大小


def _declare():
    """手写展开版的九个显式声明，供资源排布对照。"""
    a_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE],
            dtype=FP8,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
        ),
        addrs=0x00000,
        mutex_ids=[0, 1],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE],
            dtype=FP8,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
        ),
        addrs=0x10000,
        mutex_ids=[2, 3],
    )
    sa_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, SCALE_K],
            dtype=E8M0,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZZ,
        ),
        addrs=0x20000,
        mutex_ids=[4, 5],
    )
    sb_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[SCALE_K, TILE],
            dtype=E8M0,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NN,
        ),
        addrs=0x21000,
        mutex_ids=[6, 7],
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
        mutex_ids=[8, 9],
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
        mutex_ids=[10, 11],
    )
    plane_a = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, SCALE_K],
            dtype=E8M0,
            target_memory=pl.MemorySpace.ScaleLeft,
            layout=pl.ZZ,
            compact=1,
        ),
        addrs=[0x00000, _PLANE_SLOT],
        mutex_ids=[12, 13],
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
        mutex_ids=[14, 15],
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
        mutex_ids=[16],
    )
    return a_l1, b_l1, sa_l1, sb_l1, l0a, l0b, plane_a, plane_b, acc


def _groups():
    """糖版本的显式资源声明与自动 mutex ID；plane 地址逐槽配对。"""
    a_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=FP8, target_memory=pl.MemorySpace.Mat
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=2,
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=FP8, target_memory=pl.MemorySpace.Mat
        ),
        addrs=0x2000,
        mutex_ids="auto",
        depth=2,
    )
    sa_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, SCALE_K],
            dtype=E8M0,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZZ,
        ),
        addrs=0x4000,
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
        addrs=0x4100,
        mutex_ids="auto",
        depth=2,
    )
    l0a = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=FP8, target_memory=pl.MemorySpace.Left, compact=1
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=2,
    )
    l0b = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=FP8, target_memory=pl.MemorySpace.Right, compact=1
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=2,
    )
    plane_a = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, SCALE_K],
            dtype=E8M0,
            target_memory=pl.MemorySpace.ScaleLeft,
            compact=1,
        ),
        addrs=[0x0, 0x100],
        mutex_ids="auto",
        depth=2,
    )
    plane_b = pl.make_tile_group(
        type=pl.TileType(
            shape=[SCALE_K, TILE],
            dtype=E8M0,
            target_memory=pl.MemorySpace.ScaleRight,
            compact=1,
        ),
        addrs=[0x0, 0x100],
        mutex_ids="auto",
        depth=2,
    )
    acc = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            compact=1,
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    return a_l1, b_l1, sa_l1, sb_l1, l0a, l0b, plane_a, plane_b, acc


@pl.jit(arch="a5", auto_mutex=True)
def mx_sugar(
    a: pl.Tensor[[M, K], FP8],
    b: pl.Tensor[[K, N], FP8],
    scale_a: pl.Tensor[[M, K // PAIR, 2], E8M0],
    scale_b: pl.Tensor[[K // PAIR, N, 2], E8M0],
    out: pl.Tensor[[M, N], pl.DT_FP32],
):
    """一条 `pl.gemm_mx` 顶四条 move 加一条 matmul_mx。

    `init` 必须是编译期 bool，所以 K 的首步/后续写成两臂——和 `pl.gemm` 一样，
    两臂各展开一份 move，只有一臂会执行。
    """
    a_l1, b_l1, sa_l1, sb_l1, l0a, l0b, plane_a, plane_b, acc = _groups()
    with pl.section_cube():
        for mi in pl.range(0, M, TILE):
            for ni in pl.range(0, N, TILE):
                cur_acc = acc.current()
                for ki in pl.range(0, K, TILE):
                    cur_a = a_l1.next()
                    cur_b = b_l1.next()
                    cur_sa = sa_l1.next()
                    cur_sb = sb_l1.next()
                    pl.load(cur_a, a, [mi, ki])
                    pl.load(cur_b, b, [ki, ni])
                    pl.load(cur_sa, scale_a, [mi, ki // PAIR, 0])
                    pl.load(cur_sb, scale_b, [ki // PAIR, ni, 0])
                    if ki == 0:
                        pl.gemm_mx(
                            cur_acc,
                            cur_a,
                            cur_b,
                            cur_sa,
                            cur_sb,
                            l0a=l0a,
                            l0b=l0b,
                            scale_a=plane_a,
                            scale_b=plane_b,
                            init=True,
                        )
                    else:
                        pl.gemm_mx(
                            cur_acc,
                            cur_a,
                            cur_b,
                            cur_sa,
                            cur_sb,
                            l0a=l0a,
                            l0b=l0b,
                            scale_a=plane_a,
                            scale_b=plane_b,
                            init=False,
                        )
                pl.store(out, cur_acc, [mi, ni])


@pl.jit(arch="a5", auto_mutex=True)
def mx_manual(
    a: pl.Tensor[[M, K], FP8],
    b: pl.Tensor[[K, N], FP8],
    scale_a: pl.Tensor[[M, K // PAIR, 2], E8M0],
    scale_b: pl.Tensor[[K // PAIR, N, 2], E8M0],
    out: pl.Tensor[[M, N], pl.DT_FP32],
):
    """同一件事手写。语句顺序照糖展开的顺序，好让 IR 比对逐条对齐。"""
    a_l1, b_l1, sa_l1, sb_l1, l0a, l0b, plane_a, plane_b, acc = _declare()
    with pl.section_cube():
        for mi in pl.range(0, M, TILE):
            for ni in pl.range(0, N, TILE):
                cur_acc = acc.current()
                for ki in pl.range(0, K, TILE):
                    cur_a = a_l1.next()
                    cur_b = b_l1.next()
                    cur_sa = sa_l1.next()
                    cur_sb = sb_l1.next()
                    pl.load(cur_a, a, [mi, ki])
                    pl.load(cur_b, b, [ki, ni])
                    pl.load(cur_sa, scale_a, [mi, ki // PAIR, 0])
                    pl.load(cur_sb, scale_b, [ki // PAIR, ni, 0])
                    if ki == 0:
                        left = l0a.next()
                        pa = plane_a.next()
                        right = l0b.next()
                        pb = plane_b.next()
                        pl.move(left, cur_a)
                        pl.move(pa, cur_sa)
                        pl.move(right, cur_b)
                        pl.move(pb, cur_sb)
                        pl.matmul_mx(cur_acc, left, right, pa, pb)
                    else:
                        left = l0a.next()
                        pa = plane_a.next()
                        right = l0b.next()
                        pb = plane_b.next()
                        pl.move(left, cur_a)
                        pl.move(pa, cur_sa)
                        pl.move(right, cur_b)
                        pl.move(pb, cur_sb)
                        pl.matmul_mx_acc(cur_acc, cur_acc, left, right, pa, pb)
                pl.store(out, cur_acc, [mi, ni])


@pl.jit(arch="a5", auto_mutex=True)
def mx_sugar_m_tail(
    a: pl.Tensor[[M, K], FP8],
    b: pl.Tensor[[K, N], FP8],
    scale_a: pl.Tensor[[M, K // PAIR, 2], E8M0],
    scale_b: pl.Tensor[[K // PAIR, N, 2], E8M0],
    out: pl.Tensor[[M, N], pl.DT_FP32],
):
    """只算 M 的头 48 行：`m=M_TAIL` 让糖收窄 l0a、acc 和 **plane 的行轴**。

    声明和搬运都是整块的，有效范围只作用在这一次调用上——沿用
    [`../tail_extent/`](../tail_extent/) 的写法。plane 的组轴不动，组不可分。
    这条分工是糖自己推的，所以要上板验数值。

    只做第一个 M 块，`test.py` 里比的也只有那 48 行。
    """
    a_l1, b_l1, sa_l1, sb_l1, l0a, l0b, plane_a, plane_b, acc = _groups()
    with pl.section_cube():
        for ni in pl.range(0, N, TILE):
            cur_acc = acc.current()
            for ki in pl.range(0, K, TILE):
                cur_a = a_l1.next()
                cur_b = b_l1.next()
                cur_sa = sa_l1.next()
                cur_sb = sb_l1.next()
                pl.load(cur_a, a, [0, ki])
                pl.load(cur_b, b, [ki, ni])
                pl.load(cur_sa, scale_a, [0, ki // PAIR, 0])
                pl.load(cur_sb, scale_b, [ki // PAIR, ni, 0])
                if ki == 0:
                    pl.gemm_mx(
                        cur_acc,
                        cur_a,
                        cur_b,
                        cur_sa,
                        cur_sb,
                        l0a=l0a,
                        l0b=l0b,
                        scale_a=plane_a,
                        scale_b=plane_b,
                        m=M_TAIL,
                        init=True,
                    )
                else:
                    pl.gemm_mx(
                        cur_acc,
                        cur_a,
                        cur_b,
                        cur_sa,
                        cur_sb,
                        l0a=l0a,
                        l0b=l0b,
                        scale_a=plane_a,
                        scale_b=plane_b,
                        m=M_TAIL,
                        init=False,
                    )
            pl.store(out, cur_acc, [0, ni])


K_HALF = K // 2  # 收窄用：128 声明里只算前 64（= 2 个组）


@pl.jit(arch="a5", auto_mutex=True)
def mx_sugar_k_half(
    a: pl.Tensor[[TILE, K], FP8],
    b: pl.Tensor[[K, TILE], FP8],
    scale_a: pl.Tensor[[TILE, K // PAIR, 2], E8M0],
    scale_b: pl.Tensor[[K // PAIR, TILE, 2], E8M0],
    out: pl.Tensor[[TILE, TILE], pl.DT_FP32],
):
    """整块声明 K=128，一次调用只算前 `k=64`——**这条路曾经静默算错**。

    `k` 管的是组轴，所以它必须收到 plane 上：保持四个组不动，实测 max|diff| 183.2；收到两个组，
    逐位相等。`test.py` 里有一条上板断言钉住它。

    `k` 必须是 64 的倍数：实测 32 和 96（都是整数个组）各自把设备打成 AICore trap 507015。
    """
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, K], dtype=FP8, target_memory=pl.MemorySpace.Mat),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[K, TILE], dtype=FP8, target_memory=pl.MemorySpace.Mat),
        addrs=0x2000,
        mutex_ids="auto",
        depth=1,
    )
    sa_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, K // GROUP_SIZE],
            dtype=E8M0,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZZ,
        ),
        addrs=0x4000,
        mutex_ids="auto",
        depth=1,
    )
    sb_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[K // GROUP_SIZE, TILE],
            dtype=E8M0,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NN,
        ),
        addrs=0x4100,
        mutex_ids="auto",
        depth=1,
    )
    l0a = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, K], dtype=FP8, target_memory=pl.MemorySpace.Left, compact=1
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    l0b = pl.make_tile_group(
        type=pl.TileType(
            shape=[K, TILE], dtype=FP8, target_memory=pl.MemorySpace.Right, compact=1
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    plane_a = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, K // GROUP_SIZE],
            dtype=E8M0,
            target_memory=pl.MemorySpace.ScaleLeft,
            compact=1,
        ),
        addrs=[0x0],
        mutex_ids="auto",
        depth=1,
    )
    plane_b = pl.make_tile_group(
        type=pl.TileType(
            shape=[K // GROUP_SIZE, TILE],
            dtype=E8M0,
            target_memory=pl.MemorySpace.ScaleRight,
            compact=1,
        ),
        addrs=[0x0],
        mutex_ids="auto",
        depth=1,
    )
    acc = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            compact=1,
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )

    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        cur_sa = sa_l1.current()
        cur_sb = sb_l1.current()
        cur_acc = acc.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.load(cur_sa, scale_a, [0, 0, 0])
        pl.load(cur_sb, scale_b, [0, 0, 0])
        pl.gemm_mx(
            cur_acc,
            cur_a,
            cur_b,
            cur_sa,
            cur_sb,
            l0a=l0a,
            l0b=l0b,
            scale_a=plane_a,
            scale_b=plane_b,
            k=K_HALF,
        )
        pl.store(out, cur_acc, [0, 0])


FP4 = pl.DT_FP4E2M1
FP4_B = pl.DT_FP4E1M2  # 混合配对用：官方 ST 测了 e2m1/e1m2 四种组合
# FP4 的 tile 形状是**逻辑元素数**（ST 注释原文 "FP4 x2 Tile shapes use logical columns"），
# 所以 plane 的形状推导和 FP8 一模一样；差别只在字节数，也就是槽间距：
# FP8 的 64x64 数据块是 4096 字节 -> plane 槽间距 0x100；FP4 是 2048 -> 0x80。
_FP4_PLANE_SLOT = (TILE * TILE // 2) >> 4  # 0x80


@pl.jit(arch="a5", auto_mutex=True)
def mx_placed_fp4(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], FP4],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], FP4],
    scale_a: pl.Tensor[[TILE, TILE // PAIR, 2], E8M0],
    scale_b: pl.Tensor[[TILE // PAIR, TILE, 2], E8M0],
    out: pl.Tensor[[TILE, TILE], pl.DT_FP32],
):
    """FP4 使用相同的逻辑 Tile 形状，数据每槽 2048 字节、plane 槽间距 128 字节。

    GM 注解沿用动态维度，Host 提供 `[TILE, TILE//2]` 的打包字节。当前框架也支持逻辑形状为
    `[TILE, TILE]` 的静态注解；静态与动态的打包契约见 dtype_carriers 样例。
    """
    a_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=FP4, target_memory=pl.MemorySpace.Mat
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=2,
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=FP4, target_memory=pl.MemorySpace.Mat
        ),
        addrs=0x1000,
        mutex_ids="auto",
        depth=2,
    )
    sa_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, SCALE_K],
            dtype=E8M0,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZZ,
        ),
        addrs=0x2000,
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
        addrs=0x2100,
        mutex_ids="auto",
        depth=2,
    )
    l0a = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=FP4, target_memory=pl.MemorySpace.Left, compact=1
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=2,
    )
    l0b = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=FP4, target_memory=pl.MemorySpace.Right, compact=1
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=2,
    )
    plane_a = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, SCALE_K],
            dtype=E8M0,
            target_memory=pl.MemorySpace.ScaleLeft,
            compact=1,
        ),
        addrs=[0x0, 0x80],
        mutex_ids="auto",
        depth=2,
    )
    plane_b = pl.make_tile_group(
        type=pl.TileType(
            shape=[SCALE_K, TILE],
            dtype=E8M0,
            target_memory=pl.MemorySpace.ScaleRight,
            compact=1,
        ),
        addrs=[0x0, 0x80],
        mutex_ids="auto",
        depth=2,
    )
    acc = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            compact=1,
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )

    with pl.section_cube():
        cur_acc = acc.current()
        cur_a = a_l1.next()
        cur_b = b_l1.next()
        cur_sa = sa_l1.next()
        cur_sb = sb_l1.next()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.load(cur_sa, scale_a, [0, 0, 0])
        pl.load(cur_sb, scale_b, [0, 0, 0])
        pl.gemm_mx(
            cur_acc,
            cur_a,
            cur_b,
            cur_sa,
            cur_sb,
            l0a=l0a,
            l0b=l0b,
            scale_a=plane_a,
            scale_b=plane_b,
        )
        pl.store(out, cur_acc, [0, 0])


@pl.jit(arch="a5", auto_mutex=True)
def mx_placed_fp4_mixed(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], FP4],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], FP4_B],
    scale_a: pl.Tensor[[TILE, TILE // PAIR, 2], E8M0],
    scale_b: pl.Tensor[[TILE // PAIR, TILE, 2], E8M0],
    out: pl.Tensor[[TILE, TILE], pl.DT_FP32],
):
    """A 侧 E2M1、B 侧 E1M2——两边的 4 位格式不必相同。

    和 `mx_placed_fp4` 只差 B 的 dtype：两种格式都是 4 位，字节数一样，所以 plane 的地址派生、
    组数、槽间距全都不变。这条用例存在的理由是把 `reference.py` 推出来的**第二张**值表也跑上
    板——推出来没跑过的表不算验过。
    """
    a_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=FP4, target_memory=pl.MemorySpace.Mat
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=2,
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=FP4_B, target_memory=pl.MemorySpace.Mat
        ),
        addrs=0x1000,
        mutex_ids="auto",
        depth=2,
    )
    sa_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, SCALE_K],
            dtype=E8M0,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZZ,
        ),
        addrs=0x2000,
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
        addrs=0x2100,
        mutex_ids="auto",
        depth=2,
    )
    l0a = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=FP4, target_memory=pl.MemorySpace.Left, compact=1
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=2,
    )
    l0b = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE],
            dtype=FP4_B,
            target_memory=pl.MemorySpace.Right,
            compact=1,
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=2,
    )
    plane_a = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, SCALE_K],
            dtype=E8M0,
            target_memory=pl.MemorySpace.ScaleLeft,
            compact=1,
        ),
        addrs=[0x0, 0x80],
        mutex_ids="auto",
        depth=2,
    )
    plane_b = pl.make_tile_group(
        type=pl.TileType(
            shape=[SCALE_K, TILE],
            dtype=E8M0,
            target_memory=pl.MemorySpace.ScaleRight,
            compact=1,
        ),
        addrs=[0x0, 0x80],
        mutex_ids="auto",
        depth=2,
    )
    acc = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            compact=1,
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )

    with pl.section_cube():
        cur_acc = acc.current()
        cur_a = a_l1.next()
        cur_b = b_l1.next()
        cur_sa = sa_l1.next()
        cur_sb = sb_l1.next()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.load(cur_sa, scale_a, [0, 0, 0])
        pl.load(cur_sb, scale_b, [0, 0, 0])
        pl.gemm_mx(
            cur_acc,
            cur_a,
            cur_b,
            cur_sa,
            cur_sb,
            l0a=l0a,
            l0b=l0b,
            scale_a=plane_a,
            scale_b=plane_b,
        )
        pl.store(out, cur_acc, [0, 0])
