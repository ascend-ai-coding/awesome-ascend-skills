# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。22 个 `DT_*` 怎么从 host 传进来、哪些传不进来。

这一例没有计算，全是**管道**：一条 GM→Vec→GM 直通，换 dtype 反复走。要回答的是写 vf 之前
必须知道的两件事——`vf.astype` 的两端各用什么载体，以及为什么某个 dtype 一上手就报错。

张量参数与 tile 都按逻辑元素计数；4 bit host 载体的末轴按字节计数。
编译绑定和启动检查都会把 host 末轴乘 pack 后比较，因此固定 `[1, 64]` 接收 `(1, 32)` 字节。
动态末轴不限制某个固定宽度，调用方仍须按算法检查 host 输入。

第一手来源：上面那条 ST，以及 `runtime/jit.py` 的 `_PL_DTYPE_TO_TORCH`、
`ir/op/block_ops.py` 的 `_A5_LOAD_COMBOS`。
"""

import pypto_pro.language as pl

N = 64  # 逻辑元素数。4 bit 时 host 末轴给 N // 2
_A = 0
_B = 0x800


def _tile_type(dtype, n=N):
    return pl.TileType(shape=[1, n], dtype=dtype, target_memory=pl.MemorySpace.Vec)


# ---------------------------------------------------------------------------
# 一、整字节 dtype：静态末轴的直通
# ---------------------------------------------------------------------------


def make_static_copy(dtype, n=N):
    """GM→Vec→GM，不做任何计算。换 dtype 就是换载体的问题。

    工厂而不是 22 个手写 kernel：注解在 `def` 时求值，`dtype` 是闭包变量，
    解析器支持这个（`ut/.../test_closure_var_resolution.py`）。
    """

    @pl.jit(auto_mutex=True)
    def copy_kernel(a: pl.Tensor[[1, n], dtype], out: pl.Tensor[[1, n], dtype]):
        g = pl.make_tile_group(type=_tile_type(dtype, n), addrs=_A, mutex_ids=[0])
        with pl.section_vector():
            t = g.current()
            pl.load(t, a, [0, 0])
            pl.store(out, t, [0, 0])

    return copy_kernel


# ---------------------------------------------------------------------------
# 二、4 bit：动态末轴的调用方约束
# ---------------------------------------------------------------------------


def make_dynamic_copy(dtype, n=N):
    """末轴 `pl.DYNAMIC` 演示动态形状。Tile 仍按逻辑元素数 `n` 声明；固定末轴也支持。

    host 末轴给 `n // 2`（字节数）。给成 `n` 不报错，但**只搬前一半，数据静默错**。
    """

    @pl.jit(auto_mutex=True)
    def copy_kernel(
        a: pl.Tensor[[1, pl.DYNAMIC], dtype], out: pl.Tensor[[1, pl.DYNAMIC], dtype]
    ):
        g = pl.make_tile_group(type=_tile_type(dtype, n), addrs=_A, mutex_ids=[0])
        with pl.section_vector():
            t = g.current()
            pl.load(t, a, [0, 0])
            pl.store(out, t, [0, 0])

    return copy_kernel


# ---------------------------------------------------------------------------
# 三、不在搬运表里的 dtype：声明成同位宽的受支持类型，再 reinterpret
# ---------------------------------------------------------------------------


@pl.jit(auto_mutex=True)
def bool_via_reinterpret(
    a: pl.Tensor[[1, N], pl.DT_UINT8], out: pl.Tensor[[1, N], pl.DT_UINT8]
):
    """`DT_BOOL` 不在搬运表里，但它是 8 位——声明 `DT_UINT8` 搬进来，在片上换视图。

    `pl.reinterpret` 是**编译期的视图声明**，不生成任何搬运/转换指令（官方 docstring 原话：
    *"a compile-time view declaration, not a conversion"*）。

    **换 `dtype` 时必须同时给 `shape=`**，否则报
    `pl.reinterpret() 'shape' is required when 'dtype' changes`（`block_ops.py:3835`）——
    即使同位宽、形状不变也要给。
    """
    g = pl.make_tile_group(type=_tile_type(pl.DT_UINT8), addrs=_A, mutex_ids=[0])
    with pl.section_vector():
        t = g.current()
        pl.load(t, a, [0, 0])
        as_bool = pl.reinterpret(t, dtype=pl.DT_BOOL, shape=[1, N])
        back = pl.reinterpret(as_bool, dtype=pl.DT_UINT8, shape=[1, N])
        pl.store(out, back, [0, 0])


@pl.jit(auto_mutex=True)
def int4_via_reinterpret(
    a: pl.Tensor[[1, pl.DYNAMIC], pl.DT_FP4E2M1],
    out: pl.Tensor[[1, pl.DYNAMIC], pl.DT_FP4E2M1],
):
    """同样的路子对 4 位的 `DT_INT4` **走不通**：编译失败（`jit.py:2465`）。

    **原因是硬件没有 int4**，不是框架少了一条路。三处独立印证：搬运表里 4 位只有
    `FP4E2M1` / `FP4E1M2`；`DT_INT4` 在整个 `pypto_pro.language` 面一次都没出现；
    ascriptor（同一块硬件的上一代编译器）的 a5 dtype 表同样只有 `fp4_e2m1` / `fp4_e1m2`,
    它的 `int4` 只是塞在 int32 里的位负载（`pack_int4_to_int32` / `unpack_int32_to_int4`）。

    所以 `INT4` / `UINT4` / `FP4` / `HF4` 在枚举里是占位。A5 上要用 4 位数据就用
    `FP4E2M1` / `FP4E1M2`。
    """
    g = pl.make_tile_group(type=_tile_type(pl.DT_FP4E2M1), addrs=_A, mutex_ids=[0])
    with pl.section_vector():
        t = g.current()
        pl.load(t, a, [0, 0])
        as_int4 = pl.reinterpret(t, dtype=pl.DT_INT4, shape=[1, N])
        back = pl.reinterpret(as_int4, dtype=pl.DT_FP4E2M1, shape=[1, N])
        pl.store(out, back, [0, 0])


# ---------------------------------------------------------------------------
# 四、同位宽跨 dtype 搬运：搬运表允许，而且很有用
# ---------------------------------------------------------------------------


@pl.jit(auto_mutex=True)
def gm_int8_to_tile_hf8(
    a: pl.Tensor[[1, N], pl.DT_INT8], out: pl.Tensor[[1, N], pl.DT_INT8]
):
    """GM 声明 `DT_INT8`，Tile 声明 `DT_HF8`。搬运表里这一对是合法的。

    GM(ND)→Vec(ND) 的 69 条组合 = 2² + 6² + 4² + 3² + 2²，正好是**按位宽分组后组内任意配对**。
    所以「host 用 int8 装 hif8 字节」不需要 reinterpret——直接让两端 dtype 不同就行。
    """
    g = pl.make_tile_group(type=_tile_type(pl.DT_HF8), addrs=_A, mutex_ids=[0])
    with pl.section_vector():
        t = g.current()
        pl.load(t, a, [0, 0])
        pl.store(out, t, [0, 0])
