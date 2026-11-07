# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。四个归约：`reduce_sum` `reduce_max` `reduce_min` `pair_reduce_sum`。

**三个 `reduce_*` 各有两种粒度，由 `datablock=` 选**：

| 写法 | 硬件指令 | 结果 |
| --- | --- | --- |
| `reduce_sum(r, preg)` | `vcadd` | 整个寄存器归约成**一个**值，放 lane 0，其余 lane 清零 |
| `reduce_sum(r, preg, datablock=True)` | `vcgadd` | **每 8 个 FP32 lane（32 字节）一组**各归约一个值，压在 lane 0..7 |

**`datablock=True` 就是组内归约。** 设计文档 §八点四 曾把 ascriptor 的 `cgadd` / `cgmax` /
`cgmin` 列成「我们可能缺的一组功能」——它们不是缺失的接口，是 `reduce_*` 的一个参数。
这和 `load_align` 的 `dist=` 是同一个坑：**把一个接口的一个参数当成缺失的接口**。

组大小 8 是实测的（2026-09-27）：输入 1..64，`datablock=True` 得到
`[36, 100, 164, 228, 292, 356, 420, 484, 0, ...]`，正是 1..8、9..16、…各自求和。

`pair_reduce_sum` 是另一回事，不带 `datablock`：`dst[i] = src[2i] + src[2i+1]`，相邻两两相加。

**`reduce_max` / `reduce_min` 还会把极值下标写进 lane 1**，见 `_whole` 的 docstring。

第一手来源：`_vf_api.py`；ST 调用次数 `reduce_sum`:23、`reduce_max`:21、`reduce_min`:2、
`pair_reduce_sum`:1。
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32

VL_FP32 = 64  # 一个 FP32 寄存器 = 256 字节 = 64 个元素
N = 64
DATABLOCK_LANES = 8  # 32 字节 / 4 字节 —— 实测出来的组大小

_IN = 0x0000
_OUT = [0x1000 + 0x400 * i for i in range(4)]


def _tile_type():
    return pl.TileType(shape=[1, N], dtype=FP32, target_memory=pl.MemorySpace.Vec)


# ---------------------------------------------------------------------------
# 一、整寄存器归约：结果落 lane 0
# ---------------------------------------------------------------------------


@pl.vector_function
def _whole(src, o_sum, o_max, o_min, o_pair):
    """三个整寄存器归约 + 一个成对归约。

    **`reduce_max` / `reduce_min` 写两个 lane**：lane 0 是极值，**lane 1 是极值的下标**
    （第一个极值；有 NaN 时是第一个 NaN 的下标）。下标写的是整数，而寄存器是 FP32，
    所以当浮点读出来是个非规格化小数——下标 63 读成 `8.83e-44`。要当下标看得按位重解释。

    `reduce_sum` 只写 lane 0，其余清零。

    依据：官方文档 `reduction/reduce_max.md` / `reduce_min.md` 的「返回值说明」。
    **`_vf_api.py` 的 docstring 对这两个算子写的 "The remaining lanes are zeroed" 是错的**
    ——写这一例时参考照那句写，被测试打回来才发现。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    a = vf.load_align(src, 0)

    r = vf.reduce_sum(a, preg)
    vf.store_align(o_sum, r, preg, 0)

    r = vf.reduce_max(a, preg)
    vf.store_align(o_max, r, preg, 0)

    r = vf.reduce_min(a, preg)
    vf.store_align(o_min, r, preg, 0)

    r = vf.pair_reduce_sum(a, preg)  # dst[i] = src[2i] + src[2i+1]
    vf.store_align(o_pair, r, preg, 0)


@pl.jit(auto_mutex=True)
def reduce_whole(
    a: pl.Tensor[[1, N], FP32],
    o_sum: pl.Tensor[[1, N], FP32],
    o_max: pl.Tensor[[1, N], FP32],
    o_min: pl.Tensor[[1, N], FP32],
    o_pair: pl.Tensor[[1, N], FP32],
):
    ga = pl.make_tile_group(type=_tile_type(), addrs=_IN, mutex_ids=[0])
    g0 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[0], mutex_ids=[1])
    g1 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[1], mutex_ids=[2])
    g2 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[2], mutex_ids=[3])
    g3 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[3], mutex_ids=[4])
    with pl.section_vector():
        ta = ga.current()
        t0, t1, t2, t3 = g0.current(), g1.current(), g2.current(), g3.current()
        pl.load(ta, a, [0, 0])
        _whole(ta, t0, t1, t2, t3)
        pl.store(o_sum, t0, [0, 0])
        pl.store(o_max, t1, [0, 0])
        pl.store(o_min, t2, [0, 0])
        pl.store(o_pair, t3, [0, 0])


# ---------------------------------------------------------------------------
# 二、组内归约：datablock=True
# ---------------------------------------------------------------------------


@pl.vector_function
def _by_block(src, o_sum, o_max, o_min):
    """同样三个算子，加 `datablock=True`——每 8 个 FP32 一组，结果压在 lane 0..7。

    这就是 ascriptor 的 `cgadd` / `cgmax` / `cgmin`。**不需要绕 `squeeze`。**
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    a = vf.load_align(src, 0)

    r = vf.reduce_sum(a, preg, datablock=True)
    vf.store_align(o_sum, r, preg, 0)

    r = vf.reduce_max(a, preg, datablock=True)
    vf.store_align(o_max, r, preg, 0)

    r = vf.reduce_min(a, preg, datablock=True)
    vf.store_align(o_min, r, preg, 0)


@pl.jit(auto_mutex=True)
def reduce_by_block(
    a: pl.Tensor[[1, N], FP32],
    o_sum: pl.Tensor[[1, N], FP32],
    o_max: pl.Tensor[[1, N], FP32],
    o_min: pl.Tensor[[1, N], FP32],
):
    ga = pl.make_tile_group(type=_tile_type(), addrs=_IN, mutex_ids=[0])
    g0 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[0], mutex_ids=[1])
    g1 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[1], mutex_ids=[2])
    g2 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[2], mutex_ids=[3])
    with pl.section_vector():
        ta = ga.current()
        t0, t1, t2 = g0.current(), g1.current(), g2.current()
        pl.load(ta, a, [0, 0])
        _by_block(ta, t0, t1, t2)
        pl.store(o_sum, t0, [0, 0])
        pl.store(o_max, t1, [0, 0])
        pl.store(o_min, t2, [0, 0])
