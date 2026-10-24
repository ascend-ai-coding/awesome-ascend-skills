# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。Fixpipe 的 Single/Dual 目的选择与两个 AIV 的输出所有权。

M=N=K=64，输入为 FP16、Acc 为 FP32。SingleModeVec0/Vec1 把整块矩阵送到指定 AIV；DualModeSplitM/N 分别按行/列拆给两个 AIV。UB 物理容量与各自消费的半块形状一致，测试检查两
半组合后的完整输出。

Single 模式探针先给两个 UB 写入不同的已知值（2×acc 与 1×acc），再分别读回，避免依赖未初始化内容。默认模式探针同样先初始化两个目的缓冲。

L1 的 A/B 位于 0、0x2000；Left、Right、Vec 各自从 0 开始。两个 Acc 使用独立的 0、0x4000 地址，保留两个生产者，避免为了测试目的选择而引入同一 Acc 的读写复用。组深度均为 1，核内 mutex
ID 自动分配。跨核交接保留显式 FIX→MTE3 事件，两个 AIV 按 subblock 索引写各自的输出区域。

Dual 与 scale、ReLU、降位 dtype、DN 目的布局的非法组合均只检查解析期拒绝，不执行危险组合。phase 不在本例的运行范围内。

混合与纯 Vector 两个探针分别读回 block/subblock 数，测试依据当前执行预算检查关系，不固定核数。``reference.py`` 与 ``test.py`` 保留逐位数值对照、目的区分和参数覆盖检查。
"""

import pypto_pro.language as pl

FP16 = pl.DT_FP16
FP32 = pl.DT_FP32

M = N = K = 64

#: dual 模式下每个 UB 只拿一半。
HALF_M = M // 2
HALF_N = N // 2


def _cube_groups():
    """L1 两个操作数 + L0A/L0B + **两个** Acc。

    两个 Acc 是为了让两条 move 读的是两个不同的 tile（见模块 docstring §二）。
    """
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[K, N], dtype=FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x2000,
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
    acc0 = pl.make_tile_group(
        type=pl.TileType(shape=[M, N], dtype=FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    acc1 = pl.make_tile_group(
        type=pl.TileType(shape=[M, N], dtype=FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x4000,
        mutex_ids="auto",
        depth=1,
    )
    return a_l1, b_l1, l0a, l0b, acc0, acc1


# ---------------------------------------------------------------------------
# 一、机器形状：混合 kernel 里有几个 vector 子块
# ---------------------------------------------------------------------------


@pl.jit()
def probe_subblock_shape(out: pl.Tensor[[2], pl.DT_INT32]):
    """`get_block_num()` 与 `get_subblock_num()`——**双目标的前提**。

    `AccToVecMode` 的 Vec0 / Vec1 要存在，`get_subblock_num()` 就得是 2。
    这一条把前提量出来，不靠「ST 里写了 `sub_id == 0/1` 所以一定有两个」推。

    注意这个 kernel **只有 vector 段**，与本例其他 kernel（cube + vector 混合）
    不是同一种形状；[`support/debug_probe`](../../support/debug_probe/kernel.py)
    里量到的 `get_block_num() = 56 / get_subblock_num() = 1` 就是这一种。
    两种形状的数不一样，所以两处都量。
    """
    with pl.section_vector():
        out[0] = pl.get_block_num()
        out[1] = pl.get_subblock_num()


@pl.jit(auto_mutex=True)
def probe_mixed_subblock_shape(
    a: pl.Tensor[[M, K], FP16],
    b: pl.Tensor[[K, N], FP16],
    out: pl.Tensor[[2], pl.DT_INT32],
):
    """同样两个数，但在**带 cube 段**的 kernel 里量。

    `a` / `b` 没被用到——它们只是为了让这个 kernel 的签名和形状与本例其余
    kernel 一致（同样有 cube 段和 vector 段）。**形状变了数就可能变**,
    所以不能拿上一条的结果替这一条。
    """
    a_l1, b_l1, l0a, l0b, acc0, _ = _cube_groups()
    with pl.section_cube():
        cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc0.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b)
    with pl.section_vector():
        out[0] = pl.get_block_num()
        out[1] = pl.get_subblock_num()


# ---------------------------------------------------------------------------
# 二、SingleModeVec0 / SingleModeVec1：两个 UB 写进不同的值
# ---------------------------------------------------------------------------


@pl.jit(auto_mutex=True)
def single_vec0_and_vec1(
    a: pl.Tensor[[M, K], FP16],
    b: pl.Tensor[[K, N], FP16],
    out0: pl.Tensor[[M, N], FP32],
    out1: pl.Tensor[[M, N], FP32],
):
    """`SingleModeVec0` ← `2 × acc`，`SingleModeVec1` ← `1 × acc`，两个子块各存自己那份。

    期望 `out0 == 2*ref` 且 `out1 == ref`——见模块 docstring §二。
    一条测试同时验了两个模式，而且**两个模式互为对照组**。
    """
    ub = pl.make_tile_group(
        type=pl.TileType(
            shape=[M, N],
            dtype=FP32,
            target_memory=pl.MemorySpace.Vec,
            layout=pl.TensorLayout.ND,
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    a_l1, b_l1, l0a, l0b, acc0, acc1 = _cube_groups()
    t = ub.current()
    with pl.section_cube():
        cur_a, cur_b = a_l1.current(), b_l1.current()
        ac0, ac1 = acc0.current(), acc1.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        la, lb = l0a.current(), l0b.current()
        pl.move(la, cur_a)
        pl.move(lb, cur_b)
        pl.matmul(ac0, la, lb)
        pl.matmul(ac1, la, lb)
        pl.matmul_acc(ac1, ac1, la, lb)
        pl.move(t, ac1, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
        pl.move(t, ac0, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec1)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)
    with pl.section_vector():
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
        if pl.get_subblock_idx() == 0:
            pl.store(out0, t, [0, 0])
        else:
            pl.store(out1, t, [0, 0])
        pl.system.bar_all()


@pl.jit(auto_mutex=True)
def default_mode(
    a: pl.Tensor[[M, K], FP16],
    b: pl.Tensor[[K, N], FP16],
    out0: pl.Tensor[[M, N], FP32],
    out1: pl.Tensor[[M, N], FP32],
):
    """**不带 `acc_to_vec_mode` 的 move 写哪个 UB？**

    先把两个 UB 都种成 `1 × acc`，再不带 mode 地写 `2 × acc`。
    四种结果都可分辨，不需要任何「未初始化内容」的假设：

    | out0 | out1 | 说明 |
    | --- | --- | --- |
    | 2× | 1× | 默认 = `SingleModeVec0` |
    | 1× | 2× | 默认 = `SingleModeVec1` |
    | 2× | 2× | 默认广播到两个 UB |
    | 1× | 1× | 默认那条 move 什么也没写 |

    **这个问题不是学术的**：`fixpipe_paths` / `fixpipe_quant_matrix` 那 39 条
    全都是不带 mode 的 move + `if sub_id == 0` 的存，也就是说它们一直依赖
    这一格的答案，只是没写下来。
    """
    ub = pl.make_tile_group(
        type=pl.TileType(
            shape=[M, N],
            dtype=FP32,
            target_memory=pl.MemorySpace.Vec,
            layout=pl.TensorLayout.ND,
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    a_l1, b_l1, l0a, l0b, acc0, acc1 = _cube_groups()
    t = ub.current()
    with pl.section_cube():
        cur_a, cur_b = a_l1.current(), b_l1.current()
        ac0, ac1 = acc0.current(), acc1.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        la, lb = l0a.current(), l0b.current()
        pl.move(la, cur_a)
        pl.move(lb, cur_b)
        pl.matmul(ac0, la, lb)
        pl.matmul(ac1, la, lb)
        pl.matmul_acc(ac1, ac1, la, lb)
        pl.move(t, ac0, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
        pl.move(t, ac0, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec1)
        pl.move(t, ac1)  # ← 不带 mode
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)
    with pl.section_vector():
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
        if pl.get_subblock_idx() == 0:
            pl.store(out0, t, [0, 0])
        else:
            pl.store(out1, t, [0, 0])
        pl.system.bar_all()


# ---------------------------------------------------------------------------
# 三、DualModeSplitM / DualModeSplitN：一份 L0C 拆给两个 UB
# ---------------------------------------------------------------------------


@pl.jit(auto_mutex=True)
def dual_split_m(
    a: pl.Tensor[[M, K], FP16], b: pl.Tensor[[K, N], FP16], out: pl.Tensor[[M, N], FP32]
):
    """按 M 拆：UB tile 声明成**半高** `[HALF_M, N]`，两个子块拼出完整结果。

    Vec0 拿前 `HALF_M` 行，Vec1 拿后 `HALF_M` 行，所以存的时候行偏移不同。
    两半写进同一个 `[M, N]` 张量，判据就是「拼起来等于参考」。

    **这条也是 SplitM / SplitN 的判别式**：如果 SplitN 的行为和 SplitM 一样，
    下一个 kernel 的两个子块会把行切片写进列切片，数值就不对了。
    """
    ub = pl.make_tile_group(
        type=pl.TileType(
            shape=[HALF_M, N],
            dtype=FP32,
            target_memory=pl.MemorySpace.Vec,
            layout=pl.TensorLayout.ND,
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    a_l1, b_l1, l0a, l0b, acc0, _ = _cube_groups()
    t = ub.current()
    with pl.section_cube():
        cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc0.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b)
        pl.move(t, ac, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)
    with pl.section_vector():
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
        if pl.get_subblock_idx() == 0:
            pl.store(out, t, [0, 0])
        else:
            pl.store(out, t, [HALF_M, 0])
        pl.system.bar_all()


@pl.jit(auto_mutex=True)
def dual_split_n(
    a: pl.Tensor[[M, K], FP16], b: pl.Tensor[[K, N], FP16], out: pl.Tensor[[M, N], FP32]
):
    """按 N 拆：UB tile 声明成**半宽** `[M, HALF_N]`，两个子块各拿一半列。

    存的时候列偏移不同——`[0, HALF_N]` 那一半是带行间距的条带存
    （行长 `HALF_N`，行距 `N`）。
    """
    ub = pl.make_tile_group(
        type=pl.TileType(
            shape=[M, HALF_N],
            dtype=FP32,
            target_memory=pl.MemorySpace.Vec,
            layout=pl.TensorLayout.ND,
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    a_l1, b_l1, l0a, l0b, acc0, _ = _cube_groups()
    t = ub.current()
    with pl.section_cube():
        cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc0.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b)
        pl.move(t, ac, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)
    with pl.section_vector():
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
        if pl.get_subblock_idx() == 0:
            pl.store(out, t, [0, 0])
        else:
            pl.store(out, t, [0, HALF_N])
        pl.system.bar_all()


# ---------------------------------------------------------------------------
# 四、dual + rider：一条拒、两条要量
# ---------------------------------------------------------------------------


@pl.jit(auto_mutex=True)
def dual_with_relu(
    a: pl.Tensor[[M, K], FP16], b: pl.Tensor[[K, N], FP16], out: pl.Tensor[[M, N], FP32]
):
    """`DualModeSplitM` + `relu_pre_mode=NormalRelu`——**pypto_pro 没拦，效果待量**。

    预测（写在测出来之前）：**静默丢弃**。依据是 C++ 的 `TExtractL0CToUBDualDst`
    只收 `AccToVecMode` 模板参数，不收 relu；而 Python 侧 `block_ops.py:754`
    那条检查只查了 `scale`。

    预测错了也留着——`fixpipe_quant_matrix` 那次静态读 C++ 谓词的预测
    11 格错了 8 格（设计文档 §十四点三），这一条同样只是预测。
    """
    ub = pl.make_tile_group(
        type=pl.TileType(
            shape=[HALF_M, N],
            dtype=FP32,
            target_memory=pl.MemorySpace.Vec,
            layout=pl.TensorLayout.ND,
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    a_l1, b_l1, l0a, l0b, acc0, _ = _cube_groups()
    t = ub.current()
    with pl.section_cube():
        cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc0.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b)
        pl.move(
            t,
            ac,
            acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM,
            relu_pre_mode=pl.ReluPreMode.NormalRelu,
        )
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)
    with pl.section_vector():
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
        if pl.get_subblock_idx() == 0:
            pl.store(out, t, [0, 0])
        else:
            pl.store(out, t, [HALF_M, 0])
        pl.system.bar_all()


@pl.jit(auto_mutex=True)
def dual_with_scale(
    a: pl.Tensor[[M, K], FP16], b: pl.Tensor[[K, N], FP16], out: pl.Tensor[[M, N], FP32]
):
    """`DualModeSplitM` + `scale`——**解析期拒**，这个 kernel 只为钉住那条报错。

    `block_ops.py:754`：「the fixpipe dual-destination control word does not
    support quantization (hardware limit)」。
    """
    ub = pl.make_tile_group(
        type=pl.TileType(
            shape=[HALF_M, N],
            dtype=FP32,
            target_memory=pl.MemorySpace.Vec,
            layout=pl.TensorLayout.ND,
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    a_l1, b_l1, l0a, l0b, acc0, _ = _cube_groups()
    t = ub.current()
    with pl.section_cube():
        cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc0.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b)
        pl.move(t, ac, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM, scale=0.5)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)
    with pl.section_vector():
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
        if pl.get_subblock_idx() == 0:
            pl.store(out, t, [0, 0])
        pl.system.bar_all()


@pl.jit(auto_mutex=True)
def dual_dn_destination(
    a: pl.Tensor[[M, K], FP16], b: pl.Tensor[[K, N], FP16], out: pl.Tensor[[M, N], FP32]
):
    """`DualModeSplitM` + **`DN` 目的布局**——同类型、不带 scale、不带 relu。

    前三条排除规则（scale / relu / dtype 转换）都避开了，只剩布局这一项,
    所以这个 kernel 只钉一件事：**`GetDualDstCtl` 的第二条 static_assert**
    `"Dual Dst Mode is not support in nz2dn."`。

    这一格是把 ISA 的八条规则逐条对账时找出来的，不是撞出来的——
    本章原来所有 kernel 都用 `ND`，从没碰到它。
    """
    ub = pl.make_tile_group(
        type=pl.TileType(
            shape=[HALF_M, N],
            dtype=FP32,
            target_memory=pl.MemorySpace.Vec,
            layout=pl.TensorLayout.DN,
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    a_l1, b_l1, l0a, l0b, acc0, _ = _cube_groups()
    t = ub.current()
    with pl.section_cube():
        cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc0.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b)
        pl.move(t, ac, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)
    with pl.section_vector():
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
        if pl.get_subblock_idx() == 0:
            pl.store(out, t, [0, 0])
        pl.system.bar_all()


@pl.jit(auto_mutex=True)
def dual_narrowing_dtype(
    a: pl.Tensor[[M, K], FP16], b: pl.Tensor[[K, N], FP16], out: pl.Tensor[[M, N], FP16]
):
    """`DualModeSplitM` + **降位**（L0C 是 FP32，UB 是 FP16）——不带 scale。

    `FP32 → FP16` 在不带 scale 的组合表里是合法的（`fixpipe_quant_matrix` 量过），
    所以 Python 侧放行；ascriptor 的规则说双目标**连不带 deqScalar 的浮点降位
    也不支持**。这一条量的是「拦在哪一层、报什么」。
    """
    ub = pl.make_tile_group(
        type=pl.TileType(
            shape=[HALF_M, N],
            dtype=FP16,
            target_memory=pl.MemorySpace.Vec,
            layout=pl.TensorLayout.ND,
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    a_l1, b_l1, l0a, l0b, acc0, _ = _cube_groups()
    t = ub.current()
    with pl.section_cube():
        cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc0.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b)
        pl.move(t, ac, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)
    with pl.section_vector():
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
        if pl.get_subblock_idx() == 0:
            pl.store(out, t, [0, 0])
        else:
            pl.store(out, t, [HALF_M, 0])
        pl.system.bar_all()
