# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始（`vf.*` 没有语法糖）。

`vf.load_align` / `vf.store_align` 的模式面。**只有两个 API 名，但行为面比一般接口大得多**——
`LoadDist` 有 23 个取值、`StoreDist` 有 18 个（`__members__` 取，不是 `dir()`——后者会多算
`name` / `value` 两个描述符），外加 `DataCopyMode` 的 3 个取值里的非连续搬运，和双搬入/双搬出两种
形式。多数取值是同一机制的位宽变体，所以下面按**机制**分组，不按枚举值分组。

每个机制的行为都是 2026-09-27 在 A5 上逐个试出来的，不是从文档推的——文档说得对，但「哪些在 FP32
下可用」它没说，而用错位宽是**静默错**。

第一手来源：官方文档 `load_align.md` / `store_align.md`，以及
`pypto/python/tests/st/pypto_pro/frontend/vf_api/test_vf_basic_ops.py`（`dist=` 的零散用法）。
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32
I32 = pl.DT_INT32

VL_FP32 = 64  # 一个 FP32 寄存器 = 256 字节 = 64 个元素
N = 128  # 两个 VL：双搬入要读 2*VL，双搬出要写 2*VL
TILE_BYTES = N * 4

_A = 0
_O = [_A + TILE_BYTES * (i + 1) for i in range(5)]


def _tile_type(dtype):
    return pl.TileType(shape=[1, N], dtype=dtype, target_memory=pl.MemorySpace.Vec)


# ---------------------------------------------------------------------------
# 一、load 的分布模式：同一条指令，dist= 改变数据怎么铺进寄存器
# ---------------------------------------------------------------------------


@pl.vector_function
def _load_dist(src, o_norm, o_brc, o_unpk, o_blk):
    """四种在 FP32 下可用的分布。实测结果写在各行注释里。

    `dist=` 省略即 `LoadDist.NORM`。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)

    r = vf.load_align(src, 0, dist=pl.LoadDist.NORM)
    vf.store_align(o_norm, r, preg, 0)  # 1,2,3,...,64  逐元素

    r = vf.load_align(src, 0, dist=pl.LoadDist.BRC)
    vf.store_align(o_brc, r, preg, 0)  # 1,1,1,...     lane 0 广播到整个寄存器

    r = vf.load_align(src, 0, dist=pl.LoadDist.UNPK)
    vf.store_align(o_unpk, r, preg, 0)  # 1,0,2,0,3,0,...  读 VL/2 个，零填展开

    r = vf.load_align(src, 0, dist=pl.LoadDist.BLK)
    vf.store_align(o_blk, r, preg, 0)  # 1..8,1..8,...  一个 32B block 复制铺满


# ---------------------------------------------------------------------------
# 二、双搬入：一条指令读 2*VL，拆成奇偶两个寄存器
# ---------------------------------------------------------------------------


@pl.vector_function
def _load_dintlv(src, o_even, o_odd):
    """`dist=DINTLV_B32` 返回**两个**寄存器：偶数下标一个、奇数下标一个。

    这是 `load_align` 的**四参数形式**（两个 dst + ptr + offset）。codegen 按参数个数分派
    （`backend_cce_vf_ops.cpp:1016`），所以这个形式**不接受** `data_copy_mode` / `block_stride` /
    `repeat_stride`——传了会报 `INVALID_ARGUMENT`，消息里点名 "4-arg (de-interleave) form"。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    even, odd = vf.load_align(src, 0, dist=pl.LoadDist.DINTLV_B32)
    vf.store_align(o_even, even, preg, 0)  # 1,3,5,7,...
    vf.store_align(o_odd, odd, preg, 0)  # 2,4,6,8,...


# ---------------------------------------------------------------------------
# 三、store 的分布模式
# ---------------------------------------------------------------------------


@pl.vector_function
def _store_dist(src, o_norm, o_first):
    """`NORM` 按 preg 逐元素写；`FIRST_ELEMENT` 只写 lane 0。

    **preg 为 0 的位置不被写，Tile 保留原值**（官方文档原文）。所以测试用哨兵值就能看出写了哪些
    ——这和算子的 `MergeMode.ZEROING`（目的**寄存器**里置 0）是两件不同的事。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.VL32, dtype=FP32)  # 只有低 32 个有效
    r = vf.load_align(src, 0)
    vf.store_align(o_norm, r, preg, 0, dist=pl.StoreDist.NORM)
    vf.store_align(o_first, r, preg, 0, dist=pl.StoreDist.FIRST_ELEMENT)


# ---------------------------------------------------------------------------
# 四、双搬出：两个寄存器交错写成 2*VL
# ---------------------------------------------------------------------------


@pl.vector_function
def _store_intlv(src, dst):
    """`dist=INTLV_B32` 要**两个 src**（src_even, src_odd），写出长度 2*VL。

    它是双搬入的逆：`[a0, b0, a1, b1, ...]`。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    a = vf.load_align(src, 0)  # 1..64
    b = vf.load_align(src, VL_FP32)  # 65..128
    vf.store_align(dst, a, b, preg, dist=pl.StoreDist.INTLV_B32)  # 1,65,2,66,...


# ---------------------------------------------------------------------------
# 五、非连续：DataBlock 模式
# ---------------------------------------------------------------------------


@pl.vector_function
def _datablock(src, dst):
    """一条指令搬 8 个 32B DataBlock，`block_stride` 是相邻 block 的间隔（单位 block）。

    **这个模式改变两个参数位置的含义**：
    - `offset` 的位置换成一个**选择用的 mask_reg**——某个 DataBlock 在 mask 里对应的 32 bit
      有任意一位为 1 就搬，全 0 则不搬且目的位置不更新（即使越界也不报错）。
    - `block_stride` **必须作关键字传**。位置传会让参数个数变成 4，撞上双搬入那个形式并报
      `INVALID_ARGUMENT`（实测）。

    `block_stride=2` 即隔一个 block 取一个：输出的第 k 个 block 来自源的第 2k 个 block。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    select = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)  # 全选
    r = vf.load_align(
        src, select, data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, block_stride=2
    )
    vf.store_align(dst, r, preg, 0)


# ---------------------------------------------------------------------------
# 六、位宽陷阱：DS 在 32 位下静默给零
# ---------------------------------------------------------------------------


@pl.vector_function
def _width_trap(src, dst):
    """`LoadDist.DS` 用在 FP32 上**不报错，给全零**（实测 2026-09-27）。

    枚举里只有 `DS` / `DS_B8` / `DS_B16`，**没有 `DS_B32`**；`US` 同理。也就是说下采样/上采样
    只在 8 位和 16 位宽存在。在 32 位上用它，工具链一路放过，拿到的是一块零。

    这个 kernel 存在的目的是把这件事变成一条会失败的测试——哪天行为变了（无论是改成报错还是
    改成可用），测试会告诉我们。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    r = vf.load_align(src, 0, dist=pl.LoadDist.DS)
    vf.store_align(dst, r, preg, 0)


# ---------------------------------------------------------------------------
# kernel
# ---------------------------------------------------------------------------


@pl.jit(auto_mutex=True, name="align_load_dist")
def align_load_dist(
    a: pl.Tensor[[1, N], FP32],
    o0: pl.Tensor[[1, N], FP32],
    o1: pl.Tensor[[1, N], FP32],
    o2: pl.Tensor[[1, N], FP32],
    o3: pl.Tensor[[1, N], FP32],
):
    tt = _tile_type(FP32)
    ga = pl.make_tile_group(type=tt, addrs=[_A], mutex_ids=[0])
    g0 = pl.make_tile_group(type=tt, addrs=[_O[0]], mutex_ids=[1])
    g1 = pl.make_tile_group(type=tt, addrs=[_O[1]], mutex_ids=[2])
    g2 = pl.make_tile_group(type=tt, addrs=[_O[2]], mutex_ids=[3])
    g3 = pl.make_tile_group(type=tt, addrs=[_O[3]], mutex_ids=[4])
    with pl.section_vector():
        ta = ga.next()
        t0 = g0.next()
        t1 = g1.next()
        t2 = g2.next()
        t3 = g3.next()
        pl.load(ta, a, [0, 0])
        _load_dist(ta, t0, t1, t2, t3)
        pl.store(o0, t0, [0, 0])
        pl.store(o1, t1, [0, 0])
        pl.store(o2, t2, [0, 0])
        pl.store(o3, t3, [0, 0])


@pl.jit(auto_mutex=True, name="align_load_dintlv")
def align_load_dintlv(
    a: pl.Tensor[[1, N], FP32],
    even: pl.Tensor[[1, N], FP32],
    odd: pl.Tensor[[1, N], FP32],
):
    tt = _tile_type(FP32)
    ga = pl.make_tile_group(type=tt, addrs=[_A], mutex_ids=[0])
    g0 = pl.make_tile_group(type=tt, addrs=[_O[0]], mutex_ids=[1])
    g1 = pl.make_tile_group(type=tt, addrs=[_O[1]], mutex_ids=[2])
    with pl.section_vector():
        ta = ga.next()
        t0 = g0.next()
        t1 = g1.next()
        pl.load(ta, a, [0, 0])
        _load_dintlv(ta, t0, t1)
        pl.store(even, t0, [0, 0])
        pl.store(odd, t1, [0, 0])


@pl.jit(auto_mutex=True, name="align_store_dist")
def align_store_dist(
    a: pl.Tensor[[1, N], FP32],
    o_norm: pl.Tensor[[1, N], FP32],
    o_first: pl.Tensor[[1, N], FP32],
):
    """两个输出先 `load` 进来，这样「preg 为 0 的位置保留原值」才可观测。"""
    tt = _tile_type(FP32)
    ga = pl.make_tile_group(type=tt, addrs=[_A], mutex_ids=[0])
    g0 = pl.make_tile_group(type=tt, addrs=[_O[0]], mutex_ids=[1])
    g1 = pl.make_tile_group(type=tt, addrs=[_O[1]], mutex_ids=[2])
    with pl.section_vector():
        ta = ga.next()
        t0 = g0.next()
        t1 = g1.next()
        pl.load(ta, a, [0, 0])
        pl.load(t0, o_norm, [0, 0])
        pl.load(t1, o_first, [0, 0])
        _store_dist(ta, t0, t1)
        pl.store(o_norm, t0, [0, 0])
        pl.store(o_first, t1, [0, 0])


@pl.jit(auto_mutex=True, name="align_store_intlv")
def align_store_intlv(a: pl.Tensor[[1, N], FP32], out: pl.Tensor[[1, N], FP32]):
    tt = _tile_type(FP32)
    ga = pl.make_tile_group(type=tt, addrs=[_A], mutex_ids=[0])
    g0 = pl.make_tile_group(type=tt, addrs=[_O[0]], mutex_ids=[1])
    with pl.section_vector():
        ta = ga.next()
        t0 = g0.next()
        pl.load(ta, a, [0, 0])
        _store_intlv(ta, t0)
        pl.store(out, t0, [0, 0])


@pl.jit(auto_mutex=True, name="align_datablock")
def align_datablock(a: pl.Tensor[[1, N], FP32], out: pl.Tensor[[1, N], FP32]):
    tt = _tile_type(FP32)
    ga = pl.make_tile_group(type=tt, addrs=[_A], mutex_ids=[0])
    g0 = pl.make_tile_group(type=tt, addrs=[_O[0]], mutex_ids=[1])
    with pl.section_vector():
        ta = ga.next()
        t0 = g0.next()
        pl.load(ta, a, [0, 0])
        _datablock(ta, t0)
        pl.store(out, t0, [0, 0])


@pl.jit(auto_mutex=True, name="align_width_trap")
def align_width_trap(a: pl.Tensor[[1, N], FP32], out: pl.Tensor[[1, N], FP32]):
    tt = _tile_type(FP32)
    ga = pl.make_tile_group(type=tt, addrs=[_A], mutex_ids=[0])
    g0 = pl.make_tile_group(type=tt, addrs=[_O[0]], mutex_ids=[1])
    with pl.section_vector():
        ta = ga.next()
        t0 = g0.next()
        pl.load(ta, a, [0, 0])
        _width_trap(ta, t0)
        pl.store(out, t0, [0, 0])
