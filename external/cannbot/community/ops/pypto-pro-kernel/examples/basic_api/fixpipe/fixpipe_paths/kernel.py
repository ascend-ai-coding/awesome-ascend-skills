# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。Fixpipe 的三条输出路径及目的布局。

Acc→GM 用 ``pl.store`` 直接出片；Acc→Vec 用 ``pl.move`` 后由 Vector 存储；Acc→Mat 用 ``pl.move`` 留在 L1，作为下一次 GEMM 的操作数。源在 Acc 时，这些调用使
用 FIX 流水。目的布局由 Tensor/Tile 的 layout 指定；ND 与 NZ 有数值及物理排布对照，二维 DN 的落盘限制由独立编译反例检查。

本例 M=N=K=128，FP16 输入、FP32 累加。原生 ``pl.gemm`` 只负责产生 Acc；``acc_to_mat_chain`` 的契约是 ``fp16(a @ b) @ c``，中间降位由 Fixpipe 完成。

所有组使用显式地址、depth=1 和 ``mutex_ids="auto"``。L1 两个输入分别从 0、0x8000 开始；链式 GEMM 的中间值和 C 分别从 0x10000、0x18000 开始。两个 Acc 分别位于 0、
0x10000；Left、Right 和 Vec 在各自独立空间从 0 开始。

Acc→Vec 是一次性交接：FIX 写完后设置事件 0，Vector 在 MTE3 存储前等待它；只让 subblock 0 写完整结果。Acc→Mat 与 Acc→GM 均在 Cube 侧完成，核内依赖由自动 mutex 保护。

``reference.py`` 使用 NumPy 数学参考，并分别编码 ND/NZ 的物理字节布局。小整数输入保证中间 FP16 可精确表示，测试保留 rtol=atol=0；三条路径与布局各有独立验证目的。
"""

import pypto_pro.language as pl

FP16 = pl.DT_FP16
FP32 = pl.DT_FP32

M = N = K = 128

ND = pl.TensorLayout.ND
DN = pl.TensorLayout.DN
NZ = pl.TensorLayout.NZ


def _cube_groups():
    """L1 两个操作数 + L0A/L0B + Acc，本例三条输出路径共用。"""
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[K, N], dtype=FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x8000,
        mutex_ids="auto",
        depth=1,
    )
    l0a = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    l0b = pl.make_tile_group(
        type=pl.TileType(shape=[K, N], dtype=FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[M, N], dtype=FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    return a_l1, b_l1, l0a, l0b, acc


# ---------------------------------------------------------------------------
# 一、Acc → GM：最短的一条，pl.store 直接从 L0C 出片
# ---------------------------------------------------------------------------


@pl.jit(auto_mutex=True)
def acc_to_gm(
    a: pl.Tensor[[M, K], FP16], b: pl.Tensor[[K, N], FP16], out: pl.Tensor[[M, N], FP32]
):
    """`pl.store(out, acc, [0, 0])`——**这一行就是一次 fixpipe**。

    没有中间 Tile，L0C 的分形排布在搬出时被转成 GM Tensor 声明的排布（这里是 ND）。
    `pl.store` 看起来和 UB→GM 的 store 是同一个接口，实际走的是完全不同的硬件单元：
    源在 Acc 时发的是 `TSTORE`（fix 流水），源在 Vec 时发的是 MTE3。

    **判据**：与 numpy 的 `a @ b` 比。FP16 输入、FP32 累加，所以参考也用 FP32 累加。
    """
    a_l1, b_l1, l0a, l0b, acc = _cube_groups()
    with pl.section_cube():
        cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b)
        pl.store(out, ac, [0, 0])


# ---------------------------------------------------------------------------
# 二、Acc → Vec：三种布局，只差目的 Tile 的 layout=
# ---------------------------------------------------------------------------


def make_acc_to_vec(dst_layout):
    """`pl.move(ub_tile, acc_tile)` + `pl.store`。三种布局各造一个 kernel。

    **`layout=` 不是装饰**：它决定 fixpipe 发的是 `NZ2ND` / `NZ2DN` / `NZ2NZ`,
    三者写出来的字节排布不同。Python 侧没有 `layout_mode` 参数，
    这个枚举只能通过目的 Tile 的声明表达。

    工厂函数的写法：枚举值从闭包传进 kernel 体是可以的（`vf_spr_misc` 实测过），
    所以三个只差一个枚举的 kernel 用一个工厂造，不抄三遍。

    **GM 侧的 Tensor 必须声明成同一个 layout。** Vec→GM 的组合表只有
    `ND→ND` / `DN→DN` / `NZ→NZ` 三对——**跨布局的 store 不存在**，
    布局转换是 fixpipe 那一步做的，MTE3 不做。写 `ND` 的 Tensor 接 DN 的 Tile 会被
    `block_ops.py:3127` 拦。注解带 layout 的写法是 `pl.Tensor[[M, N], FP32, layout]`。

    ## 这条路要跨核交接，Acc→GM 不用

    **fixpipe 在 cube 侧，而 UB→GM 的 store 是 AIV 的指令。** 两件事在两个核上：

    - `pl.move(ub, acc)` 必须在 `pl.section_cube()` 里（fix 流水属 cube）；
    - `pl.store(out, ub, ...)` 必须在 `pl.section_vector()` 里。

    把 store 写在 cube 段里能过解析，**到 bisheng 才炸**：
    `'copy_ubuf_to_gm_align_v2' does not support the given target feature`
    ——那条 intrinsic 在 AIC 的目标特性里不存在。这个报错不提「段放错了」,
    第一次见很难联想。

    所以要三件事：UB 的 Tile 组**声明在两段之外**（两个核看同一个地址）、
    cube 段末尾 `set_cross_core(pipe=FIX)`、vector 段开头 `wait_cross_core(pipe=MTE3)`。
    再加 `if pl.get_subblock_idx() == 0`——两个 AIV 子块共用这一个 UB Tile，
    不加就是两个核写同一片 GM。收尾 `bar_all()`。

    对照 `acc_to_gm`：那条 `pl.store(out, acc, ...)` 整个在 cube 段内完成，
    **一次跨核交接都不需要**。这是直出 GM 的实际优势，不只是少写一个 Tile。

    形状抄自 `pypto/python/tests/st/pypto_pro/frontend/datacopy/test_scale_move_dtype.py`。
    """

    @pl.jit(auto_mutex=True)
    def kernel(
        a: pl.Tensor[[M, K], FP16],
        b: pl.Tensor[[K, N], FP16],
        out: pl.Tensor[[M, N], FP32, dst_layout],
    ):
        # UB 的 Tile 组声明在两段之外：cube 段写它，vector 段读它
        ub = pl.make_tile_group(
            type=pl.TileType(
                shape=[M, N],
                dtype=FP32,
                target_memory=pl.MemorySpace.Vec,
                layout=dst_layout,
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        a_l1, b_l1, l0a, l0b, acc = _cube_groups()
        t = ub.current()
        with pl.section_cube():
            cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc.current()
            pl.load(cur_a, a, [0, 0])
            pl.load(cur_b, b, [0, 0])
            pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b)
            pl.move(t, ac)  # ← fixpipe：Acc → Vec
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)
        with pl.section_vector():
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
            if pl.get_subblock_idx() == 0:  # 两个 AIV 共用这个 Tile
                pl.store(out, t, [0, 0])  # ← MTE3：Vec → GM
            pl.system.bar_all()

    return kernel


acc_to_vec_nd = make_acc_to_vec(ND)
acc_to_vec_nz = make_acc_to_vec(NZ)

# **2-D 的 DN 在这一版跑不通**，留着是为了把边界钉成测试（见 test.py §四）：
# Python 侧的组合表认 `Acc(NZ)→Vec(DN)` 和 `Vec(DN)→GM(DN)` 两步，
# 但 codegen 发出的 `GlobalTensor` 是 `Layout::ND`，被 A5 的
# `TStore.hpp:169` 的 static_assert 拦住（"Src and dst layout must be same"）。
acc_to_vec_dn = make_acc_to_vec(DN)


# ---------------------------------------------------------------------------
# 三、Acc → Mat：结果留在片上当下一次乘法的操作数
# ---------------------------------------------------------------------------


@pl.jit(auto_mutex=True)
def acc_to_mat_chain(
    a: pl.Tensor[[M, K], FP16],
    b: pl.Tensor[[K, N], FP16],
    c: pl.Tensor[[N, N], FP16],
    out: pl.Tensor[[M, N], FP32],
):
    """`pl.move(l1_tile, acc_tile)`：L0C → L1，**结果不出片直接当下一次乘法的操作数**。

    算 `fp16(a @ b) @ c`。这不是为了演示而演示——**Acc→Mat 的结果没有别的出路**：

    | 想走 | 结果 |
    | --- | --- |
    | `pl.move(ub, l1)` | **不支持。** `block_ops.py:707` 的合法路径里没有 `Mat→Vec` |
    | `pl.store(gm, l1)` | **不支持。** store 的组合表里 `Mat→GM` 一行都没有 |
    | `pl.move(l0a, l1)` | 支持——这就是它唯一的用途 |

    **`move` 的 docstring 里那张空间路径表写着 `Mat (L1) → Vec (UB)` 走 v 流水,
    与实现不符**（已报到 pypto 侧）。照表写会在解析期被拦。

    `move` 那一步顺带做 `FP32 → FP16` 降位：cube 的操作数要 FP16，而 L0C 是 FP32。
    官方 `move.md` 的表说 Acc→Mat 不配 scale 时支持 `FP32 → FP32/FP16/BF16`。
    **降位在 fixpipe 里免费做掉**，不需要单独一趟向量转换——这是这条路的第二个收益。

    小整数输入下 `a @ b` 最大 1024，FP16 精确表示整数到 2048，所以降位不丢位,
    判据仍是 `atol=0`。
    """
    a_l1, b_l1, l0a, l0b, acc = _cube_groups()
    mid = pl.make_tile_group(
        type=pl.TileType(
            shape=[M, N], dtype=FP16, target_memory=pl.MemorySpace.Mat, layout=NZ
        ),
        addrs=0x10000,
        mutex_ids="auto",
        depth=1,
    )
    c_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[N, N], dtype=FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x18000,
        mutex_ids="auto",
        depth=1,
    )
    acc2 = pl.make_tile_group(
        type=pl.TileType(shape=[M, N], dtype=FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x10000,
        mutex_ids="auto",
        depth=1,
    )
    with pl.section_cube():
        cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b)

        m = mid.current()
        pl.move(m, ac)  # ← fixpipe：Acc → Mat，顺带 FP32→FP16

        cc, ac2 = c_l1.current(), acc2.current()
        pl.load(cc, c, [0, 0])
        pl.gemm(ac2, m, cc, l0a=l0a, l0b=l0b)  # ← 中间结果直接当左操作数
        pl.store(out, ac2, [0, 0])
