# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。五个位运算：`and_` `or_` `xor` `shift_left` `shift_right`。

**名字带下划线的三个是躲 Python 关键字**：`and` / `or` 是关键字，`not` 也是（`vf.not_` 在
[`vf_predicate`](../vf_predicate/kernel.py)）。`xor` 不是关键字，所以它没有下划线——
**这一组的命名不统一，按关键字冲突来的，不是按风格**。查接口时按 `and_` 搜不到就试 `and`，
反之亦然（[skill constraints](../../../../SKILL.md#constraints-and-workspace-policy) 的搜索纪律）。

移位量的形式：**裸字面量和 `pl.const(...)` 都可以**，两者实测结果相同（2026-09-27）。
本例两种都用一次，免得读者以为只有一种写法。

第一手来源：`_vf_api.py` 的签名；ST 调用次数 `or_`:14、`and_`:8、`xor`:6、
`shift_right`:3、`shift_left`:2。
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

I32 = pl.DT_INT32

VL_I32 = 64  # 一个 INT32 寄存器 = 256 字节 = 64 个元素
N = 64
SHIFT = 2

_IN = [0x0000, 0x0400]
_OUT = [0x1000 + 0x400 * i for i in range(5)]


def _tile_type():
    return pl.TileType(shape=[1, N], dtype=I32, target_memory=pl.MemorySpace.Vec)


@pl.vector_function
def _bitwise(src_a, src_b, o_and, o_or, o_xor, o_shl, o_shr):
    """五个算子。掩码用 `dtype=I32` 建——**掩码的 dtype 要跟数据的位宽对上**。

    `xor` 的签名比另外两个多一个 `dtype=`（`xor(src0, src1, preg, mode=, dtype=)`）。
    本例不传它，走默认；它的作用留给 [`vf_lane_layout`](../vf_lane_layout/kernel.py)
    那边和 `bit_cast` 一起讲——那里才是「换个 dtype 看同一批位」的主题。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=I32)
    a = vf.load_align(src_a, 0)
    b = vf.load_align(src_b, 0)

    r = vf.and_(a, b, preg)
    vf.store_align(o_and, r, preg, 0)

    r = vf.or_(a, b, preg)
    vf.store_align(o_or, r, preg, 0)

    r = vf.xor(a, b, preg)
    vf.store_align(o_xor, r, preg, 0)

    r = vf.shift_left(a, SHIFT, preg)  # 裸字面量：模块级常量名可以
    vf.store_align(o_shl, r, preg, 0)

    # `pl.const` 的第一个参数**必须是字面量**——写 `pl.const(SHIFT, I32)` 会被拒：
    # `pl.const() first argument must be a numeric literal`（`scalar_ops.py:214`）。
    # 注意这和上面 `shift_left` 的第二个参数不同：那里收 `SHIFT` 这个名字没问题。
    r = vf.shift_right(a, pl.const(2, I32), preg)
    vf.store_align(o_shr, r, preg, 0)


@pl.jit(auto_mutex=True)
def bitwise_ops(
    a: pl.Tensor[[1, N], I32],
    b: pl.Tensor[[1, N], I32],
    o_and: pl.Tensor[[1, N], I32],
    o_or: pl.Tensor[[1, N], I32],
    o_xor: pl.Tensor[[1, N], I32],
    o_shl: pl.Tensor[[1, N], I32],
    o_shr: pl.Tensor[[1, N], I32],
):
    ga = pl.make_tile_group(type=_tile_type(), addrs=_IN[0], mutex_ids=[0])
    gb = pl.make_tile_group(type=_tile_type(), addrs=_IN[1], mutex_ids=[1])
    g0 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[0], mutex_ids=[2])
    g1 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[1], mutex_ids=[3])
    g2 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[2], mutex_ids=[4])
    g3 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[3], mutex_ids=[5])
    g4 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[4], mutex_ids=[6])
    with pl.section_vector():
        ta, tb = ga.current(), gb.current()
        t0, t1, t2 = g0.current(), g1.current(), g2.current()
        t3, t4 = g3.current(), g4.current()
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        _bitwise(ta, tb, t0, t1, t2, t3, t4)
        pl.store(o_and, t0, [0, 0])
        pl.store(o_or, t1, [0, 0])
        pl.store(o_xor, t2, [0, 0])
        pl.store(o_shl, t3, [0, 0])
        pl.store(o_shr, t4, [0, 0])


def make_mask_logic(active: int):
    """MaskReg truth tables with a full or partial b32 participation mask.

    Verified on behavior version 872df763f, CANN 9.2.0, Ascend950PR.
    Mask results are observed with select, not a partial MERGING move.
    """

    @pl.vector_function
    def logical(tile):
        all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=I32)
        gate = vf.update_mask(active, dtype=I32)
        lane = vf.arange(0, dtype=I32)
        ones = vf.full(1, all_lanes, dtype=I32)
        twos = vf.full(2, all_lanes, dtype=I32)
        zeros = vf.full(0, all_lanes, dtype=I32)
        bit0 = vf.and_(lane, ones, all_lanes)
        bit1 = vf.and_(lane, twos, all_lanes)
        a = vf.eq(bit0, 0, all_lanes)
        b = vf.eq(bit1, 0, all_lanes)

        both = vf.and_(a, b, gate)
        either = vf.or_(a, b, gate)
        different = vf.xor(a, b, gate)
        inverse = vf.not_(a, gate)
        r = vf.select(ones, zeros, both)
        vf.store_align(tile, r, all_lanes, [0, 0])
        r = vf.select(ones, zeros, either)
        vf.store_align(tile, r, all_lanes, [1, 0])
        r = vf.select(ones, zeros, different)
        vf.store_align(tile, r, all_lanes, [2, 0])
        r = vf.select(ones, zeros, inverse)
        vf.store_align(tile, r, all_lanes, [3, 0])

    @pl.jit(auto_mutex=True)
    def kernel(out: pl.Tensor[[4, N], I32]):
        kind = pl.TileType(
            shape=[4, N],
            dtype=I32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[4, N],
        )
        group = pl.make_tile_group(type=kind, addrs=[0], mutex_ids=[0])
        with pl.section_vector():
            tile = group.next()
            logical(tile)
            pl.store(out, tile, [0, 0])

    return kernel


def make_mask_pack_roundtrip(active: int):
    """MaskReg packing is not RegTensor dtype rebinding (872df763f, CANN 9.2)."""

    @pl.vector_function
    def roundtrip(tile):
        mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=I32)
        gate = vf.update_mask(active, dtype=pl.DT_UINT32)
        packed = vf.pack(gate, part=pl.PackPart.LOWER, dtype=pl.DT_UINT16)
        packed = vf.unpack(packed, part=pl.PackPart.LOWER, dtype=pl.DT_UINT32)
        one = vf.full(7, mask, dtype=I32)
        zero = vf.full(0, mask, dtype=I32)
        result = vf.select(one, zero, packed)
        vf.store_align(tile, result, mask, [0, 0])

    @pl.jit(auto_mutex=True)
    def kernel(out: pl.Tensor[[1, N], I32]):
        group = pl.make_tile_group(type=_tile_type(), addrs=[0], mutex_ids=[0])
        with pl.section_vector():
            tile = group.current()
            roundtrip(tile)
            pl.store(out, tile, [0, 0])

    return kernel
