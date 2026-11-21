# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始（`vf.*` 只有原始写法，没有语法糖）。

进 vf 的机制。前三个 kernel 只差进入方式和掩码来源，算的都是 `x + 1.0`——运算故意选最无聊的，
这样看得见的东西只有机制本身。后四个换个角度：**同一件事的两套接口**（`load`/`store` 与
`load_align`/`store_align`）、广播，以及 `vf.move` 的偏掩码。

第一手来源：`pypto/python/tests/st/pypto_pro/frontend/vf_api/test_vf_scope.py`（`auto_mutex` +
运行期 `count` 那条）和 `test_vf_basic_ops.py`（手工 `sync_src` / `sync_dst` 那条）。
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32

# 一个 FP32 寄存器装 VL = 256 字节 = 64 个元素。循环以它为步长，不是随便取的数。
VL_FP32 = 64
N = 128  # 两个整寄存器，好让「第二次迭代」真的存在
TILE_BYTES = N * 4  # 512

VA_IN = 0
VA_OUT = VA_IN + TILE_BYTES

# 后四个 kernel 只处理一个寄存器的量（`vf.load` / `vf.store` 没有 offset 参数，只能从 tile
# 基址起搬一个寄存器）。地址另起，不和上面两块重叠，省得读代码时要算。
# 一个 [1, VL] 的 FP32 tile 是 VL × 4 = 256 字节 = 0x100，所以这四块每块间隔 0x100，
# 只有三行那块占 0x300。`vf_move_merging` 同时用到前三块，它们必须互不重叠。
VA_REG_IN = 0x400  # 0x400..0x500
VA_REG_OUT = 0x500  # 0x500..0x600
VA_KEEP_IN = 0x600  # 0x600..0x700 —— `vf_move_merging` 的第二个输入
VA_ROWS_OUT = 0x700  # 0x700..0xA00 —— 3 × VL × 4 = 768 字节

BROADCAST_SCALAR = 7.0
ACTIVE_LANES = VL_FP32 // 2  # 广播那一例的偏掩码宽度


# ---------------------------------------------------------------------------
# vf 体：三个，对应三种掩码来源
# ---------------------------------------------------------------------------


@pl.vector_function
def _add_one_whole(src, dst):
    """整寄存器，掩码是编译期的 `MaskPattern.ALL`。"""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    for offset in pl.range(0, N, VL_FP32):
        reg = vf.load_align(src, offset)
        reg = vf.adds(reg, pl.const(1.0, FP32), preg)
        vf.store_align(dst, reg, preg, offset)


@pl.vector_function
def _add_one_tail(src, dst, count):
    """运行期长度，掩码每次迭代由 `update_mask` 现算。

    `active` 是本次迭代真正有效的元素个数：`min(count - offset, VL)` 夹到 0 以上。最后一次
    迭代的 `active` 通常小于 VL——那就是尾块。`count` 之外的位置**不被写**，保留目的地原值，
    这是掩码起作用的唯一可观测证据。
    """
    for offset in pl.range(0, N, VL_FP32):
        active = pl.max(0, pl.min(count - offset, VL_FP32))
        preg = vf.update_mask(active, dtype=FP32)
        reg = vf.load_align(src, offset)
        reg = vf.adds(reg, pl.const(1.0, FP32), preg)
        vf.store_align(dst, reg, preg, offset)


# ---------------------------------------------------------------------------
# 三个 kernel：auto_mutex / 手工同步 / 运行期尾块
# ---------------------------------------------------------------------------


@pl.jit(auto_mutex=True, name="vf_auto")
def vf_auto(x: pl.Tensor[[1, N], FP32], out: pl.Tensor[[1, N], FP32]):
    """`auto_mutex=True`：搬运与 vf 之间的同步由框架插。

    vf 体必须写在 `@pl.vector_function` 里。`pl.section_vector()` **不是**入口——直接在它
    里面写 `vf.*` 会被解析期拒绝（`test.py` 里有断言）。section 是放 vf 体**调用**的地方。
    """
    tile_type = pl.TileType(shape=[1, N], dtype=FP32, target_memory=pl.MemorySpace.Vec)
    src_group = pl.make_tile_group(type=tile_type, addrs=[VA_IN], mutex_ids=[0])
    dst_group = pl.make_tile_group(type=tile_type, addrs=[VA_OUT], mutex_ids=[1])
    with pl.section_vector():
        src = src_group.next()
        dst = dst_group.next()
        pl.load(src, x, [0, 0])
        _add_one_whole(src, dst)
        pl.store(out, dst, [0, 0])


@pl.jit(name="vf_manual_sync")
def vf_manual_sync(x: pl.Tensor[[1, N], FP32], out: pl.Tensor[[1, N], FP32]):
    """同样的数，同步自己插。

    三段流水要缝起来：MTE2 把数据搬进 UB，V 算，MTE3 搬出去。每一处交接是一对
    `sync_src` / `sync_dst`，`event_id` 自己分配、自己保证不撞。

    这个版本和 `vf_auto` 的数值**逐位相同**——同步形式不改变算什么。`test.py` 里有断言。
    """
    tile_type = pl.TileType(shape=[1, N], dtype=FP32, target_memory=pl.MemorySpace.Vec)
    src = pl.make_tile(tile_type, addr=VA_IN)
    dst = pl.make_tile(tile_type, addr=VA_OUT)
    with pl.section_vector():
        pl.load(src, x, [0, 0])
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
        )
        _add_one_whole(src, dst)
        pl.system.sync_src(
            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
        )
        pl.store(out, dst, [0, 0])


@pl.jit(auto_mutex=True, name="vf_tail")
def vf_tail(
    x: pl.Tensor[[1, N], FP32], out: pl.Tensor[[1, N], FP32], count: pl.DT_INT64
):
    """`count` 是运行期标量，尾块靠 `update_mask` 收。

    `out` 先被主机填成哨兵值，kernel 只写前 `count` 个——所以「掩码有没有生效」是可观测的，
    不是靠信任。注意 `out` 同时是输入：先 `load` 进来，写一部分，再 `store` 回去。
    """
    tile_type = pl.TileType(shape=[1, N], dtype=FP32, target_memory=pl.MemorySpace.Vec)
    src_group = pl.make_tile_group(type=tile_type, addrs=[VA_IN], mutex_ids=[0])
    dst_group = pl.make_tile_group(type=tile_type, addrs=[VA_OUT], mutex_ids=[1])
    with pl.section_vector():
        src = src_group.next()
        dst = dst_group.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, out, [0, 0])
        _add_one_tail(src, dst, count)
        pl.store(out, dst, [0, 0])


# ---------------------------------------------------------------------------
# 四、同一件事的两套接口：`load`/`store` 与 `load_align`/`store_align`
#
# 不是一套的简写，是两个接口。差别在**有没有 offset 和模式面**：
#
# | | 定位 | 一次搬多少 | 模式面 |
# | --- | --- | --- | --- |
# | `vf.load(tile[, stride])` / `vf.store(tile, src[, count])` | tile 基址 | 一个寄存器 | 只有 post-update |
# | `vf.load_align(tile, offset)`/`vf.store_align(tile, src, preg, offset)` | 基址+偏移 | 一个寄存器 | 双搬入、DataBlock、dist= |
#
# `vf.store` 的 `count` **可以不传**——不传就是一个整寄存器。写这一例时三份说法不一致（官方
# `store.md` 的原型和参数说明把它写成必填、Python 签名也没默认值，而那一页的**调用示例里就没传**、
# 框架 docstring 写着「optional」），实测站在调用示例这边，**错的两份已经修了**
# （`pypto` 的 `f8f172bdc`）。
# ---------------------------------------------------------------------------


@pl.vector_function
def _simple_pair(src, dst):
    """`vf.load` → `vf.move` → `vf.store`：一个寄存器的平凡拷贝。

    这里掩码给满（`MaskPattern.ALL`），因为 `copy` 是个**新名字**——见 `_move_merging`：
    偏掩码要配合复用已有寄存器才有定义。
    """
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    reg = vf.load(src)  # 没有 offset 参数：从 tile 基址读一个寄存器
    copy = vf.move(reg, all_lanes)
    vf.store(dst, copy)  # 不传 count：一个整寄存器


@pl.vector_function
def _aligned_pair(src, dst):
    """同一件事用对齐接口写一遍。和 `_simple_pair` 的输出必须逐位相同（`test.py` 有断言）。"""
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    reg = vf.load_align(src, 0)
    vf.store_align(dst, reg, all_lanes, 0)


@pl.vector_function
def _move_merging(src, keep, dst):
    """`vf.move` 的偏掩码用法：未选中的 lane 保留目的寄存器的原值。

    **关键在 LHS 是不是新名字。** `_assignment_parser.py` 的 Step 2 只在**第一次定义**时
    声明寄存器，「For existing variables, reuse them directly -- do NOT create a new register,
    as that would lose the accumulated value in self-referential ops」。所以：

    | LHS | 目的寄存器 | 未选中的 lane |
    | --- | --- | --- |
    | 已赋过值的名字（本函数） | **复用同一个** | 保留它上一次的内容，**有定义** |
    | 新名字（`_simple_pair` 的 `copy`） | 新分配 | 原值未定义 |

    实测（Ascend950PR，2026-09-27）：`keep` 取 1001..1064、`src` 取 1..64、半掩码，
    输出是 `[1..32, 1033..1064]`——后半正是 `keep`。这就是 `MERGING` 作为默认值的用途。

    **新名字那一支为什么危险**：同一个实测里它给出 `[1..32, 33..64]`，未选中的 lane 拿到的是
    **源寄存器自己的值**，于是输出看起来就是一次完整拷贝。一条「move 拷贝了整块」的断言会以
    错误的理由通过（见 `references/verification/evidence.md` #7 / #14）。所以本例**只验复用那一支**，
    新名字那一支没有测试——未定义行为没有判据。
    """
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    half = vf.update_mask(ACTIVE_LANES, dtype=FP32)
    merged = vf.load_align(keep, 0)  # merged 先装 keep 的数据
    incoming = vf.load_align(src, 0)
    merged = vf.move(incoming, half)  # 同一个名字 → 复用，后半保留 keep
    vf.store_align(dst, merged, all_lanes, 0)


@pl.vector_function
def _broadcast_rows(src, dst):
    """`vf.full` 的两种模式，三行输出。

    - 第 0 行：**Scalar 模式** + 偏掩码。`vf.full` 的 MergeMode 默认 **`ZEROING`**
      （`MERGING` 本设备不支持），所以未选中的 lane **置 0**——和上面 `vf.move` 的默认
      **正好相反**，而且两者各自的另一种模式都不支持。这是这一节真正要记的一条。
    - 第 1 行：**Tensor 模式** `pos=LOWEST`，把源寄存器的**元素 0** 广播到所有 lane。
    - 第 2 行：`pos=HIGHEST`，广播**最后一个元素**（元素 63）。「最高位」说的是元素序，
      不是位序——探针实测（输入 `1..64`）拿到的是 `64.0`，即最后一个元素。`test.py` 里那条
      测试用的是 `arange(0..63)`，所以它比的是 `63.0`；参考按 `lanes[vl - 1]` 算，与输入无关。

    Scalar 模式的 `preg` 可省（实测 `vf.full(2.5, dtype=FP32)` 解析通过），这里显式传是为了
    让偏掩码这件事看得见。
    """
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    half = vf.update_mask(ACTIVE_LANES, dtype=FP32)
    reg = vf.load_align(src, 0)

    filled = vf.full(BROADCAST_SCALAR, half, dtype=FP32)
    vf.store_align(dst, filled, all_lanes, 0)

    lowest = vf.full(reg, half, pos=pl.DuplicatePos.LOWEST)
    vf.store_align(dst, lowest, all_lanes, VL_FP32)

    highest = vf.full(reg, half, pos=pl.DuplicatePos.HIGHEST)
    vf.store_align(dst, highest, all_lanes, 2 * VL_FP32)


# ---------------------------------------------------------------------------
# 后四个 kernel
# ---------------------------------------------------------------------------


@pl.jit(auto_mutex=True, name="vf_simple_pair")
def vf_simple_pair(
    x: pl.Tensor[[1, VL_FP32], FP32], out: pl.Tensor[[1, VL_FP32], FP32]
):
    """`vf.load` / `vf.move` / `vf.store` 三件套，一个寄存器进出。"""
    tile_type = pl.TileType(
        shape=[1, VL_FP32], dtype=FP32, target_memory=pl.MemorySpace.Vec
    )
    src_group = pl.make_tile_group(type=tile_type, addrs=[VA_REG_IN], mutex_ids=[0])
    dst_group = pl.make_tile_group(type=tile_type, addrs=[VA_REG_OUT], mutex_ids=[1])
    with pl.section_vector():
        src = src_group.next()
        dst = dst_group.next()
        pl.load(src, x, [0, 0])
        _simple_pair(src, dst)
        pl.store(out, dst, [0, 0])


@pl.jit(auto_mutex=True, name="vf_aligned_pair")
def vf_aligned_pair(
    x: pl.Tensor[[1, VL_FP32], FP32], out: pl.Tensor[[1, VL_FP32], FP32]
):
    """同一个拷贝，换对齐接口。存在的意义是给 `vf_simple_pair` 一个对照。"""
    tile_type = pl.TileType(
        shape=[1, VL_FP32], dtype=FP32, target_memory=pl.MemorySpace.Vec
    )
    src_group = pl.make_tile_group(type=tile_type, addrs=[VA_REG_IN], mutex_ids=[0])
    dst_group = pl.make_tile_group(type=tile_type, addrs=[VA_REG_OUT], mutex_ids=[1])
    with pl.section_vector():
        src = src_group.next()
        dst = dst_group.next()
        pl.load(src, x, [0, 0])
        _aligned_pair(src, dst)
        pl.store(out, dst, [0, 0])


@pl.jit(auto_mutex=True, name="vf_broadcast")
def vf_broadcast(x: pl.Tensor[[1, VL_FP32], FP32], out: pl.Tensor[[3, VL_FP32], FP32]):
    """`vf.full` 的 Scalar / Tensor 两种模式，三行输出。行含义见 `_broadcast_rows`。"""
    src_type = pl.TileType(
        shape=[1, VL_FP32], dtype=FP32, target_memory=pl.MemorySpace.Vec
    )
    dst_type = pl.TileType(
        shape=[3, VL_FP32], dtype=FP32, target_memory=pl.MemorySpace.Vec
    )
    src_group = pl.make_tile_group(type=src_type, addrs=[VA_REG_IN], mutex_ids=[0])
    dst_group = pl.make_tile_group(type=dst_type, addrs=[VA_ROWS_OUT], mutex_ids=[1])
    with pl.section_vector():
        src = src_group.next()
        dst = dst_group.next()
        pl.load(src, x, [0, 0])
        _broadcast_rows(src, dst)
        pl.store(out, dst, [0, 0])


@pl.jit(auto_mutex=True, name="vf_move_merging")
def vf_move_merging(
    x: pl.Tensor[[1, VL_FP32], FP32],
    keep: pl.Tensor[[1, VL_FP32], FP32],
    out: pl.Tensor[[1, VL_FP32], FP32],
):
    """`vf.move` 配偏掩码 + 复用寄存器：前半来自 `x`，后半保留 `keep`。"""
    tile_type = pl.TileType(
        shape=[1, VL_FP32], dtype=FP32, target_memory=pl.MemorySpace.Vec
    )
    src_group = pl.make_tile_group(type=tile_type, addrs=[VA_REG_IN], mutex_ids=[0])
    keep_group = pl.make_tile_group(type=tile_type, addrs=[VA_KEEP_IN], mutex_ids=[1])
    dst_group = pl.make_tile_group(type=tile_type, addrs=[VA_REG_OUT], mutex_ids=[2])
    with pl.section_vector():
        src = src_group.next()
        kept = keep_group.next()
        dst = dst_group.next()
        pl.load(src, x, [0, 0])
        pl.load(kept, keep, [0, 0])
        _move_merging(src, kept, dst)
        pl.store(out, dst, [0, 0])
