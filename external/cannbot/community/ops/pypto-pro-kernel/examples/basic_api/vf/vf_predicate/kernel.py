# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。十个谓词相关 API：六个比较 + `select` + `not_` + 两个掩码来源。

## 掩码不是「每 lane 一位」

官方 `basic_data_structures/mask_reg.md`：**mask_reg 总宽固定 256 bit**，每个元素占几位由
dtype 决定——FP32 是 **4 bit/元素**（64 个元素 × 4 = 256），FP16 是 2 bit，INT8 是 1 bit。
这决定了 `mask_gen_with_reg_tensor` 的 broadcast 倍数，也解释了为什么掩码的 dtype 要和数据对上。

## 怎么把掩码「看出来」

掩码寄存器不能直接存到 Tile。本例一律用 **`select(ones, zeros, mask)`** 把它变成 0/1 向量再存
——`select` 的语义是文档写明的（掩码真取 `src0`、假取 `src1`），没有歧义。

**不要用 `vf.move(x, mask)` 去观察掩码。** 写这一例时我这么试过，得到的非活跃 lane 是另一个
输入的值而不是 0——那取决于编译器把目的寄存器分配到了哪里，不是可依赖的语义。

## 链式比较与掩码逻辑

比较算子**自己带 `preg`**（`gt(src0, src1, preg) -> dst_mask`），所以两个条件取交集就是**链式**：
把第一个掩码当第二个比较的 `preg`。无效位置在 `dst_mask` 里置零，正好是 AND。

当前 MaskReg 也有逻辑运算重载；版本化契约归 [skill 的掩码组合说明
](../../../../references/api/vf.md#25-combining-masks-with-existing-comparisons-and-selection)，真值表测试在
[vf_bitwise](../vf_bitwise/test.py)。本例的 `vf.not_` 使用 RegTensor 重载，处理数据位型。

## 两个掩码来源

- `get_mask_spr(width)`：读 **kernel 作用域** `pl.set_vec_mask(high, low)` 设的那对 SPR。
  FP32 下取 MASK0 的 64 bit，每 bit 复制成 4 bit。这是 kernel 体与 vf 体之间的桥，
  完整的设/复位面在 [`vf_mask_state`](../vf_mask_state/kernel.py)。
- `mask_gen_with_reg_tensor(src, offset)`：从 reg_tensor 的第 `offset` 个 **DataBlock** 的裸比特
  生成掩码。FP32 下 256B 分成 32 个 8B 块，每块 64 bit → 每 bit 复制 4 倍 → 64 个元素的掩码。

第一手来源：官方 `reg_computation/` 下 `mask_reg.md`、`select.md`、`gt.md`、
`mask_operations/mask_gen_with_reg_tensor.md`、`data_movement/get_mask_spr.md`。
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32

VL_FP32 = 64  # 一个 FP32 寄存器 = 256 字节 = 64 个元素
N = 64

LO = 16.0  # 区间下界，2 的幂，比较结果精确
HI = 48.0  # 区间上界
SCALAR = 32.0  # 给标量比较用

_IN = [0x0000, 0x0400]
_OUT = [0x1000 + 0x400 * i for i in range(8)]


def _tile_type(dtype=FP32):
    return pl.TileType(shape=[1, N], dtype=dtype, target_memory=pl.MemorySpace.Vec)


# ---------------------------------------------------------------------------
# 一、六个比较算子：结果是 mask_reg，用 select 看出来
# ---------------------------------------------------------------------------


@pl.vector_function
def _compare(src_a, src_b, o_eq, o_ne, o_lt, o_gt, o_le, o_ge):
    """六个比较，各自把掩码用 `select(ones, zeros, mask)` 变成 0/1 存出去。

    `ones` / `zeros` 用 `vf.full` 造，不占额外的 Tile。
    """
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    a = vf.load_align(src_a, 0)
    b = vf.load_align(src_b, 0)
    ones = vf.full(pl.const(1.0, FP32), dtype=FP32)
    zeros = vf.full(pl.const(0.0, FP32), dtype=FP32)

    m = vf.eq(a, b, all_lanes)
    r = vf.select(ones, zeros, m)
    vf.store_align(o_eq, r, all_lanes, 0)

    m = vf.ne(a, b, all_lanes)
    r = vf.select(ones, zeros, m)
    vf.store_align(o_ne, r, all_lanes, 0)

    m = vf.lt(a, b, all_lanes)
    r = vf.select(ones, zeros, m)
    vf.store_align(o_lt, r, all_lanes, 0)

    m = vf.gt(a, b, all_lanes)
    r = vf.select(ones, zeros, m)
    vf.store_align(o_gt, r, all_lanes, 0)

    m = vf.le(a, b, all_lanes)
    r = vf.select(ones, zeros, m)
    vf.store_align(o_le, r, all_lanes, 0)

    m = vf.ge(a, b, all_lanes)
    r = vf.select(ones, zeros, m)
    vf.store_align(o_ge, r, all_lanes, 0)


@pl.jit(auto_mutex=True)
def compare_all(
    a: pl.Tensor[[1, N], FP32],
    b: pl.Tensor[[1, N], FP32],
    o_eq: pl.Tensor[[1, N], FP32],
    o_ne: pl.Tensor[[1, N], FP32],
    o_lt: pl.Tensor[[1, N], FP32],
    o_gt: pl.Tensor[[1, N], FP32],
    o_le: pl.Tensor[[1, N], FP32],
    o_ge: pl.Tensor[[1, N], FP32],
):
    ga = pl.make_tile_group(type=_tile_type(), addrs=_IN[0], mutex_ids=[0])
    gb = pl.make_tile_group(type=_tile_type(), addrs=_IN[1], mutex_ids=[1])
    g0 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[0], mutex_ids=[2])
    g1 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[1], mutex_ids=[3])
    g2 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[2], mutex_ids=[4])
    g3 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[3], mutex_ids=[5])
    g4 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[4], mutex_ids=[6])
    g5 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[5], mutex_ids=[7])
    with pl.section_vector():
        ta, tb = ga.current(), gb.current()
        t0, t1, t2 = g0.current(), g1.current(), g2.current()
        t3, t4, t5 = g3.current(), g4.current(), g5.current()
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        _compare(ta, tb, t0, t1, t2, t3, t4, t5)
        pl.store(o_eq, t0, [0, 0])
        pl.store(o_ne, t1, [0, 0])
        pl.store(o_lt, t2, [0, 0])
        pl.store(o_gt, t3, [0, 0])
        pl.store(o_le, t4, [0, 0])
        pl.store(o_ge, t5, [0, 0])


# ---------------------------------------------------------------------------
# 二、条件取交集 = 链式；取反 = select 换源；标量比较
# ---------------------------------------------------------------------------


@pl.vector_function
def _combine(src_a, o_band, o_inverted, o_scalar, o_not_data):
    """三种组合手法 + `not_` 的真实用途。

    - **交集**：`hi_mask = vf.lt(a, HI, lo_mask)`——第二个比较拿第一个掩码当 `preg`，
      无效位置在结果掩码里置零，于是等于 `(a > LO) and (a < HI)`。
    - **取反**：`select(zeros, ones, m)`，把两个源调个位置。不需要掩码取反算子。
    - **标量比较**：比较算子的第二个操作数可以直接是标量（官方 `gt.md`：
      *"第二个参数可以是标量或reg_tensor，接口自动识别并分发到对应的硬件指令"*）。
    - **`not_`**：RegTensor 的按位取反。这里作用在**数据**上，不是掩码上。
    """
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    a = vf.load_align(src_a, 0)
    ones = vf.full(pl.const(1.0, FP32), dtype=FP32)
    zeros = vf.full(pl.const(0.0, FP32), dtype=FP32)

    lo_mask = vf.gt(a, pl.const(16.0, FP32), all_lanes)  # a > LO
    band = vf.lt(a, pl.const(48.0, FP32), lo_mask)  # ← 链式：再取 a < HI
    r = vf.select(ones, zeros, band)
    vf.store_align(o_band, r, all_lanes, 0)

    r = vf.select(zeros, ones, band)  # ← 换源 = 取反
    vf.store_align(o_inverted, r, all_lanes, 0)

    scalar_mask = vf.ge(a, pl.const(32.0, FP32), all_lanes)  # 标量比较
    r = vf.select(ones, zeros, scalar_mask)
    vf.store_align(o_scalar, r, all_lanes, 0)

    r = vf.not_(a, all_lanes)  # 数据按位取反
    vf.store_align(o_not_data, r, all_lanes, 0)


@pl.jit(auto_mutex=True)
def combine(
    a: pl.Tensor[[1, N], FP32],
    o_band: pl.Tensor[[1, N], FP32],
    o_inverted: pl.Tensor[[1, N], FP32],
    o_scalar: pl.Tensor[[1, N], FP32],
    o_not_data: pl.Tensor[[1, N], FP32],
):
    ga = pl.make_tile_group(type=_tile_type(), addrs=_IN[0], mutex_ids=[0])
    g0 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[0], mutex_ids=[1])
    g1 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[1], mutex_ids=[2])
    g2 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[2], mutex_ids=[3])
    g3 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[3], mutex_ids=[4])
    with pl.section_vector():
        ta = ga.current()
        t0, t1, t2, t3 = g0.current(), g1.current(), g2.current(), g3.current()
        pl.load(ta, a, [0, 0])
        _combine(ta, t0, t1, t2, t3)
        pl.store(o_band, t0, [0, 0])
        pl.store(o_inverted, t1, [0, 0])
        pl.store(o_scalar, t2, [0, 0])
        pl.store(o_not_data, t3, [0, 0])


# ---------------------------------------------------------------------------
# 三、两个掩码来源：SPR 与 reg_tensor 的裸比特
# ---------------------------------------------------------------------------


@pl.vector_function
def _from_spr(src_a, dst):
    """`get_mask_spr()` 读 kernel 体里 `pl.set_vec_mask` 设的那对 SPR。

    FP32 下取 MASK0 的低 64 bit，每 bit 复制成 4 bit。所以 kernel 侧给的 64 位整数里
    第 i 位控制第 i 个 lane。
    """
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    ones = vf.full(pl.const(1.0, FP32), dtype=FP32)
    zeros = vf.full(pl.const(0.0, FP32), dtype=FP32)

    from_spr = vf.get_mask_spr()
    r = vf.select(ones, zeros, from_spr)
    vf.store_align(dst, r, all_lanes, 0)


@pl.jit(auto_mutex=True)
def mask_from_spr(a: pl.Tensor[[1, N], FP32], out: pl.Tensor[[1, N], FP32]):
    """`pl.set_vec_mask(high, low)` 在 **kernel 体**里设，vf 体里 `get_mask_spr()` 读。

    位型写成**字面量** `0xFFFFFFFF`——低 32 位为 1，所以 lane 0..31 有效、32..63 无效。
    不用运行期参数，因为 `pl.*` 的这类编译期参数多半只收字面量（`pl.const` 就是这样）。

    `pl.reset_mask()` 收尾：mask SPR 是 kernel 作用域的状态，不复位会影响后面的代码。
    """
    ga = pl.make_tile_group(type=_tile_type(), addrs=_IN[0], mutex_ids=[0])
    g0 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[0], mutex_ids=[1])
    with pl.section_vector():
        ta, t0 = ga.current(), g0.current()
        pl.load(ta, a, [0, 0])
        pl.set_vec_mask(0, 0xFFFFFFFF)  # 低 32 个 lane 有效
        _from_spr(ta, t0)
        pl.reset_mask()
        pl.store(out, t0, [0, 0])


@pl.vector_function
def _from_reg_bits(src_bits, o_block0, o_block1):
    """`mask_gen_with_reg_tensor(src, offset)`：从第 `offset` 个 8 字节 DataBlock 的裸比特生成掩码。

    FP32 下 256B 分 32 个 8B 块。`offset=0` 取 lane 0..1 的位，`offset=1` 取 lane 2..3 的位
    ——**按字节算，不是按元素算**。
    """
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    ones = vf.full(pl.const(1.0, FP32), dtype=FP32)
    zeros = vf.full(pl.const(0.0, FP32), dtype=FP32)
    bits = vf.load_align(src_bits, 0)

    m0 = vf.mask_gen_with_reg_tensor(bits, offset=0)
    r = vf.select(ones, zeros, m0)
    vf.store_align(o_block0, r, all_lanes, 0)

    m1 = vf.mask_gen_with_reg_tensor(bits, offset=1)
    r = vf.select(ones, zeros, m1)
    vf.store_align(o_block1, r, all_lanes, 0)


@pl.jit(auto_mutex=True)
def mask_from_reg_bits(
    bits: pl.Tensor[[1, N], FP32],
    o_block0: pl.Tensor[[1, N], FP32],
    o_block1: pl.Tensor[[1, N], FP32],
):
    """输入声明成 **FP32**，但装的是位型——host 侧用 `int32 数组 .view(float32)` 造。

    为什么不直接声明 INT32：`mask_reg` 的 dtype 要和配套的 reg_tensor 对上
    （`mask_reg.md`：「不一致的情况下需要自行判断结果行为」）。这个 vf 体里 `select` 用的是
    FP32 寄存器，所以源也用 FP32，整条链只有一种位宽，不用讨论混用。
    """
    gb = pl.make_tile_group(type=_tile_type(), addrs=_IN[0], mutex_ids=[0])
    g0 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[0], mutex_ids=[1])
    g1 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[1], mutex_ids=[2])
    with pl.section_vector():
        tb = gb.current()
        t0, t1 = g0.current(), g1.current()
        pl.load(tb, bits, [0, 0])
        _from_reg_bits(tb, t0, t1)
        pl.store(o_block0, t0, [0, 0])
        pl.store(o_block1, t1, [0, 0])
