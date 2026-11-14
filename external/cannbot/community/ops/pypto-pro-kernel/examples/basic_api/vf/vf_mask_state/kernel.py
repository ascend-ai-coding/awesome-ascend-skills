# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。四个 **kernel 作用域** 的 mask 状态 API，全在 `pl.` 侧：

`pl.set_vec_mask` / `pl.set_mask_norm` / `pl.set_mask_count` / `pl.reset_mask`。

## 为什么这四个归进 vf

它们本身不是 vf 算子——写在 `pl.section_vector()` 里、不在 `@pl.vector_function` 里。但它们设的
状态**跨 `@vf` 边界可见**：vf 体内 `vf.get_mask_spr()` 读到的就是这里设的值。框架自己的
`mask_state_prefix` 样例测的就是这个（*"Mask state read across a @vf boundary"*）。

## 没有官方 API 文档

这四个在 `docs/zh/api/pro_api/` 下**没有自己的页面**——只有 `get_mask_spr.md` 顺带提到
`set_vec_mask`。唯一的第一手来源是
`pypto/python/tests/st/pypto_pro/frontend/element_wise/test_vector_mask.py`。
所以下面每条行为都是 2026-09-27 实测的，不是照文档抄的。

## 两种模式，`set_vec_mask` 的含义不同

| 模式 | 由谁切 | `set_vec_mask(high, low)` 的含义 |
| --- | --- | --- |
| **norm**（位图） | `pl.set_mask_norm()`（默认） | `low` 的第 i 位控制第 i 个元素 |
| **count**（计数） | `pl.set_mask_count()` | `low` 是**有效元素个数**，不是位图 |

**`vf.get_mask_spr()` 只按位图读。** 计数模式下它会把那个计数当成位图——设计数 16 得到的是
「只有 lane 4 有效」（16 = `0b10000`）。**不报错**。见 test.py §三。

所以：计数模式是给 **tile 级** 向量算子（`pl.add` 那类）用的；vf 路径要按个数掩码，用
`vf.update_mask(count, dtype=)`（见 [`vf_entry`](../vf_entry/kernel.py) 的尾块处理）。
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32

VL_FP32 = 64  # 一个 FP32 寄存器 = 256 字节 = 64 个元素
N = 64

BITMAP = 0x0F0F0F0F  # 每 4 个有效、4 个无效
NARROW = 0x0000FFFF  # 低 16 个有效
COUNT = 16  # 计数模式用的个数

_OUT = [0x0000 + 0x400 * i for i in range(3)]


def _tile_type():
    return pl.TileType(shape=[1, N], dtype=FP32, target_memory=pl.MemorySpace.Vec)


@pl.vector_function
def _show_spr(dst):
    """把当前 SPR 掩码变成 0/1 存出去。

    用 `select(ones, zeros, m)` 而不是 `vf.move`——后者非活跃 lane 的内容取决于寄存器分配，
    不是可依赖的语义（见 [`vf_predicate`](../vf_predicate/test.py) §二）。
    """
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    ones = vf.full(pl.const(1.0, FP32), dtype=FP32)
    zeros = vf.full(pl.const(0.0, FP32), dtype=FP32)
    m = vf.get_mask_spr()
    r = vf.select(ones, zeros, m)
    vf.store_align(dst, r, all_lanes, 0)


# ---------------------------------------------------------------------------
# 一、norm 模式：set_vec_mask 的 low 是位图
# ---------------------------------------------------------------------------


@pl.jit(auto_mutex=True)
def norm_bitmap(out: pl.Tensor[[1, N], FP32]):
    """`set_mask_norm()` + 位图。第 i 位控制第 i 个 lane。

    位型写字面量。`pl.set_mask_norm()` 是默认模式，这里显式写出来是为了让这一例自成一体
    ——不依赖「进来时是什么模式」。
    """
    g = pl.make_tile_group(type=_tile_type(), addrs=_OUT[0], mutex_ids=[0])
    with pl.section_vector():
        t = g.current()
        pl.set_mask_norm()
        pl.set_vec_mask(0, 0x0F0F0F0F)
        _show_spr(t)
        pl.reset_mask()  # 收尾复位，见下面 reset_restores_all_lanes
        pl.store(out, t, [0, 0])


# ---------------------------------------------------------------------------
# 二、计数模式：get_mask_spr 读不懂它
# ---------------------------------------------------------------------------


@pl.jit(auto_mutex=True)
def count_mode_seen_from_vf(out: pl.Tensor[[1, N], FP32]):
    """**这是一个坑的现场，不是推荐写法。**

    `set_mask_count()` 之后 `set_vec_mask(0, 16)` 的意思是「前 16 个元素有效」，
    但 `vf.get_mask_spr()` 仍按位图读，于是读到 `16 = 0b10000` → 只有 lane 4 有效。
    **不报错。**

    vf 路径要按个数掩码，用 `vf.update_mask(16, dtype=FP32)`。
    """
    g = pl.make_tile_group(type=_tile_type(), addrs=_OUT[0], mutex_ids=[0])
    with pl.section_vector():
        t = g.current()
        pl.set_mask_count()
        pl.set_vec_mask(0, 16)
        _show_spr(t)
        pl.set_mask_norm()  # 切回位图模式
        pl.reset_mask()
        pl.store(out, t, [0, 0])


# ---------------------------------------------------------------------------
# 三、reset_mask：复位成全有效
# ---------------------------------------------------------------------------


@pl.jit(auto_mutex=True)
def reset_restores_all_lanes(out: pl.Tensor[[1, N], FP32]):
    """先设一个窄掩码，再 `reset_mask()`，读出来应该全有效。

    顺序很重要：如果 `reset_mask` 什么都没做，读到的会是那个窄掩码。
    """
    g = pl.make_tile_group(type=_tile_type(), addrs=_OUT[0], mutex_ids=[0])
    with pl.section_vector():
        t = g.current()
        pl.set_vec_mask(0, 0x000000FF)  # 只有低 8 个有效
        pl.reset_mask()  # 再复位
        _show_spr(t)
        pl.store(out, t, [0, 0])


# ---------------------------------------------------------------------------
# 四、状态跨 @vf 边界持续
# ---------------------------------------------------------------------------


@pl.jit(auto_mutex=True)
def state_persists_across_vf_calls(
    o1: pl.Tensor[[1, N], FP32], o2: pl.Tensor[[1, N], FP32]
):
    """设一次，**两次** vf 调用都看得到——这就是「跨 @vf 边界」的含义。

    反过来说：**不复位就会漏给后面的代码**。这是 `pl.reset_mask()` 存在的理由，
    也是本例其他 kernel 都在收尾复位的理由。
    """
    g1 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[0], mutex_ids=[0])
    g2 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[1], mutex_ids=[1])
    with pl.section_vector():
        t1, t2 = g1.current(), g2.current()
        pl.set_vec_mask(0, 0x0000FFFF)
        _show_spr(t1)
        _show_spr(t2)  # 没有重新设，照样看得到
        pl.reset_mask()
        pl.store(o1, t1, [0, 0])
        pl.store(o2, t2, [0, 0])
