# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。八个超越函数：`ln` `log` `log2` `log10` `exp` `exp_sub` `sqrt` `abs`。

**这是第一例不能 `rtol=0, atol=0` 的样例**，所以判据本身是它的主题之一。前面几例都靠挑
2 的幂做到逐位相同；超越函数没有这个办法，只能量出误差再定阈值。

三件要分清的事：

1. **`log` 是 `ln` 的别名**（`_vf_api.py` 的 docstring 原话：*"Natural logarithm (alias for
   :func:`vf.ln`)"*）。不是「log 默认以 e 为底」这种约定，是同一个函数两个名字。
   `test_log_is_an_alias_of_ln` 在 IR 上验它们发射同一个 op。
2. **`exp_sub(src0, src1)` = `exp(src0 - src1)`**，一条指令。它的签名和其余七个不同：
   没有 `mode=`，多一个 `layout=`。ST 里调了 93 次——softmax 的 `exp(x - max)` 就是它。
3. **`precision=` 是六个函数共有的形参**（`abs` 和 `exp_sub` 没有）。两件事：
   它**进了 IR**、但在 FP32 上结果逐位相同（test.py §三）；同名连续调用的临时量必须独立，回归见 test.py §六。

第一手来源：`_vf_api.py` 的签名；ST 调用次数 `exp_sub`:93、`abs`:12、`sqrt`:3、`exp`:3，
`ln` / `log` / `log2` / `log10` 各 1。
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32

VL_FP32 = 64  # 一个 FP32 寄存器 = 256 字节 = 64 个元素
N = 64

_IN = [0x0000, 0x0400]
_OUT = [0x1000 + 0x400 * i for i in range(8)]


def _tile_type():
    return pl.TileType(shape=[1, N], dtype=FP32, target_memory=pl.MemorySpace.Vec)


# ---------------------------------------------------------------------------
# 一、四个对数 + exp + sqrt + abs
# ---------------------------------------------------------------------------


@pl.vector_function
def _elementwise(src, o_ln, o_log, o_log2, o_log10, o_exp, o_sqrt, o_abs):
    """七个逐元素函数。`ln` 与 `log` 是同一个函数的两个名字，这里都调一次。

    `abs` 的签名里没有 `precision=`——它不做近似，没有精度可选。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    a = vf.load_align(src, 0)

    r = vf.ln(a, preg)
    vf.store_align(o_ln, r, preg, 0)

    r = vf.log(a, preg)  # 与上面同一个函数
    vf.store_align(o_log, r, preg, 0)

    r = vf.log2(a, preg)
    vf.store_align(o_log2, r, preg, 0)

    r = vf.log10(a, preg)
    vf.store_align(o_log10, r, preg, 0)

    r = vf.exp(a, preg)
    vf.store_align(o_exp, r, preg, 0)

    r = vf.sqrt(a, preg)
    vf.store_align(o_sqrt, r, preg, 0)

    r = vf.abs(a, preg)
    vf.store_align(o_abs, r, preg, 0)


@pl.jit(auto_mutex=True)
def transcendental(
    a: pl.Tensor[[1, N], FP32],
    o_ln: pl.Tensor[[1, N], FP32],
    o_log: pl.Tensor[[1, N], FP32],
    o_log2: pl.Tensor[[1, N], FP32],
    o_log10: pl.Tensor[[1, N], FP32],
    o_exp: pl.Tensor[[1, N], FP32],
    o_sqrt: pl.Tensor[[1, N], FP32],
    o_abs: pl.Tensor[[1, N], FP32],
):
    ga = pl.make_tile_group(type=_tile_type(), addrs=_IN[0], mutex_ids=[0])
    g0 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[0], mutex_ids=[1])
    g1 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[1], mutex_ids=[2])
    g2 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[2], mutex_ids=[3])
    g3 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[3], mutex_ids=[4])
    g4 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[4], mutex_ids=[5])
    g5 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[5], mutex_ids=[6])
    g6 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[6], mutex_ids=[7])
    with pl.section_vector():
        ta = ga.current()
        t0, t1, t2, t3 = g0.current(), g1.current(), g2.current(), g3.current()
        t4, t5, t6 = g4.current(), g5.current(), g6.current()
        pl.load(ta, a, [0, 0])
        _elementwise(ta, t0, t1, t2, t3, t4, t5, t6)
        pl.store(o_ln, t0, [0, 0])
        pl.store(o_log, t1, [0, 0])
        pl.store(o_log2, t2, [0, 0])
        pl.store(o_log10, t3, [0, 0])
        pl.store(o_exp, t4, [0, 0])
        pl.store(o_sqrt, t5, [0, 0])
        pl.store(o_abs, t6, [0, 0])


# ---------------------------------------------------------------------------
# 二、exp_sub：一条指令做 exp(a - b)
# ---------------------------------------------------------------------------


@pl.vector_function
def _exp_sub(src_a, src_b, o_fused, o_manual):
    """融合版与手写版并排算，用来验「它真的是 exp(a-b)，而且不是先减再 exp 的两条指令」。

    手写版是 `sub` 再 `exp`；两者在数学上相同，数值上**可以有差**（融合指令的中间精度不同）。
    这一例断言两者在 `rtol` 内一致，并把实测的最大相对差记在 test.py 里。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    a = vf.load_align(src_a, 0)
    b = vf.load_align(src_b, 0)

    r = vf.exp_sub(a, b, preg)  # 一条指令
    vf.store_align(o_fused, r, preg, 0)

    d = vf.sub(a, b, preg)  # 两条指令
    r = vf.exp(d, preg)
    vf.store_align(o_manual, r, preg, 0)


@pl.jit(auto_mutex=True)
def exp_sub_vs_manual(
    a: pl.Tensor[[1, N], FP32],
    b: pl.Tensor[[1, N], FP32],
    o_fused: pl.Tensor[[1, N], FP32],
    o_manual: pl.Tensor[[1, N], FP32],
):
    ga = pl.make_tile_group(type=_tile_type(), addrs=_IN[0], mutex_ids=[0])
    gb = pl.make_tile_group(type=_tile_type(), addrs=_IN[1], mutex_ids=[1])
    g0 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[0], mutex_ids=[2])
    g1 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[1], mutex_ids=[3])
    with pl.section_vector():
        ta, tb = ga.current(), gb.current()
        t0, t1 = g0.current(), g1.current()
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        _exp_sub(ta, tb, t0, t1)
        pl.store(o_fused, t0, [0, 0])
        pl.store(o_manual, t1, [0, 0])


# ---------------------------------------------------------------------------
# 三、precision= ：六个函数共有的形参
# ---------------------------------------------------------------------------


@pl.vector_function
def _precision_pairs(src, o_ln_d, o_ln_p, o_sqrt_d, o_sqrt_p, o_log2_d, o_log2_p):
    """同一个输入，同一个函数，只差 `precision=True`。成对输出供逐位比较。

    每个输出用独立名字便于阅读；框架也支持复用名字，test.py 有专门回归。
    FP32 的所选输入域中两种模式数值相同，但不能据此推断参数被忽略。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    a = vf.load_align(src, 0)

    ln_default = vf.ln(a, preg)
    vf.store_align(o_ln_d, ln_default, preg, 0)
    ln_precise = vf.ln(a, preg, precision=True)  # ← 独立名字
    vf.store_align(o_ln_p, ln_precise, preg, 0)

    sqrt_default = vf.sqrt(a, preg)
    vf.store_align(o_sqrt_d, sqrt_default, preg, 0)
    sqrt_precise = vf.sqrt(a, preg, precision=True)  # ← 独立名字
    vf.store_align(o_sqrt_p, sqrt_precise, preg, 0)

    log2_default = vf.log2(a, preg)
    vf.store_align(o_log2_d, log2_default, preg, 0)
    log2_precise = vf.log2(a, preg, precision=True)  # ← 独立名字
    vf.store_align(o_log2_p, log2_precise, preg, 0)


@pl.jit(auto_mutex=True)
def precision_pairs(
    a: pl.Tensor[[1, N], FP32],
    o_ln_d: pl.Tensor[[1, N], FP32],
    o_ln_p: pl.Tensor[[1, N], FP32],
    o_sqrt_d: pl.Tensor[[1, N], FP32],
    o_sqrt_p: pl.Tensor[[1, N], FP32],
    o_log2_d: pl.Tensor[[1, N], FP32],
    o_log2_p: pl.Tensor[[1, N], FP32],
):
    ga = pl.make_tile_group(type=_tile_type(), addrs=_IN[0], mutex_ids=[0])
    g0 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[0], mutex_ids=[1])
    g1 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[1], mutex_ids=[2])
    g2 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[2], mutex_ids=[3])
    g3 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[3], mutex_ids=[4])
    g4 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[4], mutex_ids=[5])
    g5 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[5], mutex_ids=[6])
    with pl.section_vector():
        ta = ga.current()
        t0, t1, t2 = g0.current(), g1.current(), g2.current()
        t3, t4, t5 = g3.current(), g4.current(), g5.current()
        pl.load(ta, a, [0, 0])
        _precision_pairs(ta, t0, t1, t2, t3, t4, t5)
        pl.store(o_ln_d, t0, [0, 0])
        pl.store(o_ln_p, t1, [0, 0])
        pl.store(o_sqrt_d, t2, [0, 0])
        pl.store(o_sqrt_p, t3, [0, 0])
        pl.store(o_log2_d, t4, [0, 0])
        pl.store(o_log2_p, t5, [0, 0])
