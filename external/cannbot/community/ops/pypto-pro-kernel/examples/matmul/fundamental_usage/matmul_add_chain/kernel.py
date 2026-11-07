# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始（手写跨核事件——C 层同步的唯一样例；**出货常规链先走 `pl.gemm` + B 层流水**）。out = ((Q @ K) + X1) @ V + X2 —— 数据在 cube 与 vector 之间往返两趟。

核内同步交给 ``auto_mutex`` 与自动 mutex ID；手写三个跨核事件的 set/wait。上游 ST 是不开 auto_mutex 的
手工同步版，那里还要成对写 ``sync_src`` / ``sync_dst``，噪声盖过了主题。
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

SIZE = 64
HALF = SIZE // 2  # 一个 Block 两个 AIV，各分一半 M 行


@pl.vector_function
def _add(dst, lhs, rhs):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for row in pl.range(HALF):
        a = vf.load_align(lhs, [row, 0])
        b = vf.load_align(rhs, [row, 0])
        result = vf.add(a, b, mask)
        vf.store_align(dst, result, mask, [row, 0])


@pl.jit(auto_mutex=True)
def matmul_add_chain(
    q: pl.Tensor[[SIZE, SIZE], pl.DT_FP32],
    k: pl.Tensor[[SIZE, SIZE], pl.DT_FP32],
    v: pl.Tensor[[SIZE, SIZE], pl.DT_FP32],
    x1: pl.Tensor[[SIZE, SIZE], pl.DT_FP32],
    x2: pl.Tensor[[SIZE, SIZE], pl.DT_FP32],
    out: pl.Tensor[[SIZE, SIZE], pl.DT_FP32],
):
    # ---- 跨核共享的三块，声明在两个 section 之外 ----
    tt_half_ub = pl.TileType(
        shape=[HALF, SIZE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
    )
    mm1_ub = pl.make_tile_group(
        type=tt_half_ub, addrs=0x0000, mutex_ids="auto", depth=1
    )  # cube -> vec
    mm2_ub = pl.make_tile_group(
        type=tt_half_ub, addrs=0x2000, mutex_ids="auto", depth=1
    )  # cube -> vec
    v1_l1 = pl.make_tile_group(  # vec -> cube
        type=pl.TileType(
            shape=[SIZE, SIZE],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
        ),
        addrs=0x10000,
        mutex_ids="auto",
        depth=1,
    )

    with pl.section_cube():
        tt_mat = pl.TileType(
            shape=[SIZE, SIZE],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
        )
        q_l1 = pl.make_tile_group(type=tt_mat, addrs=0x00000, mutex_ids="auto", depth=1)
        k_l1 = pl.make_tile_group(type=tt_mat, addrs=0x04000, mutex_ids="auto", depth=1)
        v_l1 = pl.make_tile_group(type=tt_mat, addrs=0x08000, mutex_ids="auto", depth=1)
        l0a = pl.make_tile_group(
            type=pl.TileType(
                shape=[SIZE, SIZE],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Left,
                layout=pl.NZ,
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        tt_right = pl.TileType(
            shape=[SIZE, SIZE],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Right,
            layout=pl.ZN,
        )
        k_l0b = pl.make_tile_group(
            type=tt_right, addrs=0x0000, mutex_ids="auto", depth=1
        )
        v_l0b = pl.make_tile_group(
            type=tt_right, addrs=0x4000, mutex_ids="auto", depth=1
        )
        tt_acc = pl.TileType(
            shape=[SIZE, SIZE],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            fractal=1024,
        )
        c1 = pl.make_tile_group(type=tt_acc, addrs=0x0000, mutex_ids="auto", depth=1)
        c2 = pl.make_tile_group(type=tt_acc, addrs=0x8000, mutex_ids="auto", depth=1)

        # 第一趟：mm1 = Q @ K，交给 vector
        pl.load(q_l1.current(), q, [0, 0])
        pl.load(k_l1.current(), k, [0, 0])
        pl.move(l0a.current(), q_l1.current())
        pl.move(k_l0b.current(), k_l1.current())
        pl.matmul(c1.current(), l0a.current(), k_l0b.current())
        # DualModeSplitM：把 [64,64] 的累加器按 M 劈成两半，各发给一个 AIV
        pl.move(
            mm1_ub.current(),
            c1.current(),
            acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM,
        )
        pl.system.set_cross_core(
            pipe=pl.PipeType.FIX, event_id=0
        )  # move 源在 Acc -> FIX

        pl.load(v_l1.current(), v, [0, 0])
        pl.move(v_l0b.current(), v_l1.current())

        # 第二趟：等 vector 把 v1 放进 L1，再算 mm2 = v1 @ V
        # 本侧第一个碰 v1_l1 的 op 是下面这条 move（Mat -> Left），所以等 MTE1
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=2)
        pl.move(l0a.current(), v1_l1.current())
        pl.matmul(c2.current(), l0a.current(), v_l0b.current())
        pl.move(
            mm2_ub.current(),
            c2.current(),
            acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM,
        )
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=1)

    with pl.section_vector():
        tt_ub = pl.TileType(
            shape=[HALF, SIZE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
        )
        x1_ub = pl.make_tile_group(type=tt_ub, addrs=0x4000, mutex_ids="auto", depth=1)
        x2_ub = pl.make_tile_group(type=tt_ub, addrs=0x6000, mutex_ids="auto", depth=1)
        res_ub = pl.make_tile_group(type=tt_ub, addrs=0x8000, mutex_ids="auto", depth=1)
        nz_ub = pl.make_tile_group(  # 送回 L1 前要先转成 NZ
            type=pl.TileType(
                shape=[HALF, SIZE],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.NZ,
            ),
            addrs=0xA000,
            mutex_ids="auto",
            depth=1,
        )

        off = pl.get_subblock_idx() * HALF

        # 第一趟：加 X1，再把结果送回 L1 给 cube
        pl.load(x1_ub.current(), x1, [off, 0])
        # 本侧第一个碰 mm1_ub 的 op 是下面这条 add，跑在 V 上 —— 必须等 V，不是 MTE3
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=0)
        _add(res_ub.current(), mm1_ub.current(), x1_ub.current())
        pl.move(nz_ub.current(), res_ub.current())  # ND -> NZ
        pl.insert(v1_l1.current(), nz_ub.current(), [off, 0])  # UB -> L1，各写一半
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=2)  # insert 走 MTE3

        # 第二趟：加 X2，写出去
        pl.load(x2_ub.current(), x2, [off, 0])
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=1)
        _add(res_ub.current(), mm2_ub.current(), x2_ub.current())
        pl.store(out, res_ub.current(), [off, 0])
