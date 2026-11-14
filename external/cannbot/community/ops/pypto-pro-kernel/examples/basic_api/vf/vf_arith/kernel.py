# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始（`vf.*` 没有语法糖）。

寄存器算术。四个 kernel 按**调用形状**分组，不是按运算分组——形状才是会写错的地方：

| kernel | 形状 | 算子 |
| --- | --- | --- |
| ``arith_binary`` | `f(src0, src1, preg)` | add / sub / mul / div / min / max |
| ``arith_scalar`` | `f(src, scalar, preg)` | adds / muls / mins / maxs |
| ``arith_composite`` | 同上两种，但 `axpy` 读 dst 前值 | abs_sub / axpy |
| ``arith_wide`` | 返回**元组** | mull |
| ``arith_carry`` | 返回元组，且**目标变量要预声明** | addc / subc |

第一手来源：`pypto/python/tests/st/pypto_pro/frontend/vf_api/test_vf_basic_ops.py`（十个基础算子）
和官方文档 `addc.md` 的调用示例（带进位那两个——框架 ST 里没有它们的用例）。
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32
I32 = pl.DT_INT32
U32 = pl.DT_UINT32

VL_FP32 = 64  # 一个 FP32 寄存器 = 256 字节 = 64 个元素
N = 64  # 正好一个寄存器：这一例讲算术，不讲循环
TILE_BYTES = N * 4

# 标量变体统一用 2.0，好让四个输出互相可比。**注意不能把它提成模块常量再传给
# `pl.const`**：`pl.const()` 的第一个参数必须是字面量，闭包变量会被拒
# （`scalar_ops.py:214`：*"first argument must be a numeric literal"*）。所以下面几处是
# 写死的 `2.0`，这个常量只给 host 侧的参考实现和测试用。
SCALAR = 2.0

_ADDR = {
    name: i * TILE_BYTES
    for i, name in enumerate(("a", "b", "o0", "o1", "o2", "o3", "o4", "o5"))
}


def _tile_type(dtype):
    return pl.TileType(shape=[1, N], dtype=dtype, target_memory=pl.MemorySpace.Vec)


# ---------------------------------------------------------------------------
# 一、reg ⊕ reg：六个算子同一个形状
# ---------------------------------------------------------------------------


@pl.vector_function
def _binary(src_a, src_b, o0, o1, o2, o3, o4, o5):
    """`f(src0, src1, preg)` —— 两个源都是寄存器。

    `mode=` 省略即 `MergeMode.ZEROING`：**掩码没选中的车道在目的寄存器里置 0**，不是保留原值。
    MERGING（保留原值）在 A5 上编译期就被拒，见 `test.py` 的验证 5。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    a = vf.load_align(src_a, 0)
    b = vf.load_align(src_b, 0)
    # **每个 vf 结果都必须先赋给一个变量。** 把调用嵌进 store_align 的参数里会被拒：
    # "Use: dst = vf.add(...)"（`_call_parser.py:1647`）。解析器要在 IR 里给结果一个名字。
    r = vf.add(a, b, preg)
    vf.store_align(o0, r, preg, 0)
    r = vf.sub(a, b, preg)
    vf.store_align(o1, r, preg, 0)
    r = vf.mul(a, b, preg)
    vf.store_align(o2, r, preg, 0)
    r = vf.div(a, b, preg)
    vf.store_align(o3, r, preg, 0)
    r = vf.min(a, b, preg)
    vf.store_align(o4, r, preg, 0)
    r = vf.max(a, b, preg)
    vf.store_align(o5, r, preg, 0)


# ---------------------------------------------------------------------------
# 二、reg ⊕ 标量：第二个操作数不是寄存器
# ---------------------------------------------------------------------------


@pl.vector_function
def _scalar(src_a, o0, o1, o2, o3):
    """`f(src, scalar, preg)` —— 标量走 `pl.const(value, dtype)`。

    `pl.const` 把字面量固定成一个带 dtype 的编译期常量；直接传 `2.0` 在 ST 里也见得到
    （`vf.axpy(reg_a, 2.0, preg)`），但显式给 dtype 在混精度时更不容易出错。

    **`pl.const()` 的第一个参数必须是字面量**——写 `pl.const(SCALAR, FP32)` 会被拒，哪怕
    `SCALAR` 是个模块级的 `2.0`。解析期从 AST 上读这个值，闭包变量读不到。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    s = pl.const(2.0, FP32)
    a = vf.load_align(src_a, 0)
    r = vf.adds(a, s, preg)
    vf.store_align(o0, r, preg, 0)
    r = vf.muls(a, s, preg)
    vf.store_align(o1, r, preg, 0)
    r = vf.mins(a, s, preg)
    vf.store_align(o2, r, preg, 0)
    r = vf.maxs(a, s, preg)
    vf.store_align(o3, r, preg, 0)


# ---------------------------------------------------------------------------
# 三、复合：abs_sub 普通，axpy 读 dst 前值
# ---------------------------------------------------------------------------


@pl.vector_function
def _composite(src_a, src_b, o0, o1):
    """`axpy` 是这一例最容易写错的一个：**它累加进 dst，而 dst 是赋值左边那个变量的前值**。

    `dst_i = scalar * src_i + dst_i`。所以下面 `acc` 必须先装上初值（这里装 b），再交给 `axpy`；
    `acc = vf.axpy(a, s, preg)` 读的是**上一行那个 acc**。少了装载那行，累加的初值就是未定义的。

    这个写法照的是 `test_vf_basic_ops.py:486-487`：先 `reg_dst = vf.load_align(in_b, 0)`，
    紧接着 `reg_dst = vf.axpy(reg_a, 2.0, preg)`。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    s = pl.const(2.0, FP32)
    a = vf.load_align(src_a, 0)
    b = vf.load_align(src_b, 0)
    d = vf.abs_sub(a, b, preg)
    vf.store_align(o0, d, preg, 0)

    acc = vf.load_align(src_b, 0)  # ← 累加初值。删掉这行 axpy 就没有 dst 可读
    acc = vf.axpy(a, s, preg)  # ← acc = 2.0 * a + acc(前值) = 2a + b
    vf.store_align(o1, acc, preg, 0)


# ---------------------------------------------------------------------------
# 四、宽整数：返回元组
# ---------------------------------------------------------------------------


@pl.vector_function
def _wide(src_a, src_b, o_lo, o_hi):
    """`mull` 返回 `(dst_lo, dst_hi)`：乘积的低位和高位，各一个寄存器。

    同族的 `addc` / `subc` 见下面的 ``_carry``——它们也返回元组，但多一条要求。

    `dst_lo = (a*b) mod 2^bit`，`dst_hi = (a*b) // 2^bit`，`bit` 是元素位宽（这里 32）。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=I32)
    a = vf.load_align(src_a, 0)
    b = vf.load_align(src_b, 0)
    lo, hi = vf.mull(a, b, preg)
    vf.store_align(o_lo, lo, preg, 0)
    vf.store_align(o_hi, hi, preg, 0)


@pl.vector_function
def _carry(src_a, src_b, o_add0, o_add1, o_sub0, o_sub1):
    """带进位的加减。**两条规则，缺一条就错。**

    **规则一：元组的目标变量必须预声明成对的类型。** `carry` / `borrow` 是 mask_reg，不是
    reg_tensor，所以调用之前要先 `vf.create_mask(...)` 把那个名字绑成 mask。解析器按赋值目标
    **已有的绑定**定类型——和 `axpy` 读 dst 前值是同一条机制。少了预声明，`carry` 会被当成
    reg_tensor，codegen 出 `RegTensor<uint32_t>` 喂给要 `vector_bool&` 的 intrinsic，编译报
    "no matching function for call to 'vaddcs'"。这个写法照的是官方 `addc.md` 的调用示例。

    **规则二：mask 里存的是硬件的进位位，不是语义上的「借位」。** 减法在底层是 `a + ~b + carry`，
    所以同一个 `ALL` 在两边读法相反（实测 2026-09-27）：

    | 调用 | 结果 |
    | --- | --- |
    | `addc(a, b, ALLF)` | `a + b`（进位 0） |
    | `addc(a, b, ALL)` | `a + b + 1`（进位 1） |
    | `subc(a, b, ALL)` | `a - b`（**不借位**） |
    | `subc(a, b, ALLF)` | `a - b - 1`（借位 1） |

    参数名叫 `borrow_src` 会让人以为「全 0 = 不借位」，实测是反的。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U32)
    a = vf.load_align(src_a, 0, dtype=U32)
    b = vf.load_align(src_b, 0, dtype=U32)

    carry_zero = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=U32)
    carry_one = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U32)
    carry = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=U32)  # ← 规则一：预声明

    carry, r = vf.addc(a, b, carry_zero, preg)
    vf.store_align(o_add0, r, preg, 0)
    carry, r = vf.addc(a, b, carry_one, preg)
    vf.store_align(o_add1, r, preg, 0)

    borrow = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=U32)  # ← 同样要预声明
    borrow, r = vf.subc(a, b, carry_one, preg)  # ALL → 不借位 → a - b
    vf.store_align(o_sub0, r, preg, 0)
    borrow, r = vf.subc(a, b, carry_zero, preg)  # ALLF → 借位 1 → a - b - 1
    vf.store_align(o_sub1, r, preg, 0)


# ---------------------------------------------------------------------------
# kernel：每个 vf 体一个，输出个数不同
# ---------------------------------------------------------------------------


@pl.jit(auto_mutex=True, name="arith_binary")
def arith_binary(
    a: pl.Tensor[[1, N], FP32],
    b: pl.Tensor[[1, N], FP32],
    o0: pl.Tensor[[1, N], FP32],
    o1: pl.Tensor[[1, N], FP32],
    o2: pl.Tensor[[1, N], FP32],
    o3: pl.Tensor[[1, N], FP32],
    o4: pl.Tensor[[1, N], FP32],
    o5: pl.Tensor[[1, N], FP32],
):
    tt = _tile_type(FP32)
    ga = pl.make_tile_group(type=tt, addrs=[_ADDR["a"]], mutex_ids=[0])
    gb = pl.make_tile_group(type=tt, addrs=[_ADDR["b"]], mutex_ids=[1])
    # 六个输出组逐个写开。**kernel 体内不能用列表推导**——
    # `[pl.make_tile_group(...) for i in range(6)]` 会被解析期拒绝
    # （`_expression_parser.py:267`）。这里的重复是语言约束，不是没想到。
    g0 = pl.make_tile_group(type=tt, addrs=[_ADDR["o0"]], mutex_ids=[2])
    g1 = pl.make_tile_group(type=tt, addrs=[_ADDR["o1"]], mutex_ids=[3])
    g2 = pl.make_tile_group(type=tt, addrs=[_ADDR["o2"]], mutex_ids=[4])
    g3 = pl.make_tile_group(type=tt, addrs=[_ADDR["o3"]], mutex_ids=[5])
    g4 = pl.make_tile_group(type=tt, addrs=[_ADDR["o4"]], mutex_ids=[6])
    g5 = pl.make_tile_group(type=tt, addrs=[_ADDR["o5"]], mutex_ids=[7])
    with pl.section_vector():
        ta = ga.next()
        tb = gb.next()
        t0 = g0.next()
        t1 = g1.next()
        t2 = g2.next()
        t3 = g3.next()
        t4 = g4.next()
        t5 = g5.next()
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        _binary(ta, tb, t0, t1, t2, t3, t4, t5)
        pl.store(o0, t0, [0, 0])
        pl.store(o1, t1, [0, 0])
        pl.store(o2, t2, [0, 0])
        pl.store(o3, t3, [0, 0])
        pl.store(o4, t4, [0, 0])
        pl.store(o5, t5, [0, 0])


@pl.jit(auto_mutex=True, name="arith_scalar")
def arith_scalar(
    a: pl.Tensor[[1, N], FP32],
    o0: pl.Tensor[[1, N], FP32],
    o1: pl.Tensor[[1, N], FP32],
    o2: pl.Tensor[[1, N], FP32],
    o3: pl.Tensor[[1, N], FP32],
):
    tt = _tile_type(FP32)
    ga = pl.make_tile_group(type=tt, addrs=[_ADDR["a"]], mutex_ids=[0])
    g0 = pl.make_tile_group(type=tt, addrs=[_ADDR["o0"]], mutex_ids=[2])
    g1 = pl.make_tile_group(type=tt, addrs=[_ADDR["o1"]], mutex_ids=[3])
    g2 = pl.make_tile_group(type=tt, addrs=[_ADDR["o2"]], mutex_ids=[4])
    g3 = pl.make_tile_group(type=tt, addrs=[_ADDR["o3"]], mutex_ids=[5])
    with pl.section_vector():
        ta = ga.next()
        t0 = g0.next()
        t1 = g1.next()
        t2 = g2.next()
        t3 = g3.next()
        pl.load(ta, a, [0, 0])
        _scalar(ta, t0, t1, t2, t3)
        pl.store(o0, t0, [0, 0])
        pl.store(o1, t1, [0, 0])
        pl.store(o2, t2, [0, 0])
        pl.store(o3, t3, [0, 0])


@pl.jit(auto_mutex=True, name="arith_composite")
def arith_composite(
    a: pl.Tensor[[1, N], FP32],
    b: pl.Tensor[[1, N], FP32],
    o0: pl.Tensor[[1, N], FP32],
    o1: pl.Tensor[[1, N], FP32],
):
    tt = _tile_type(FP32)
    ga = pl.make_tile_group(type=tt, addrs=[_ADDR["a"]], mutex_ids=[0])
    gb = pl.make_tile_group(type=tt, addrs=[_ADDR["b"]], mutex_ids=[1])
    g0 = pl.make_tile_group(type=tt, addrs=[_ADDR["o0"]], mutex_ids=[2])
    g1 = pl.make_tile_group(type=tt, addrs=[_ADDR["o1"]], mutex_ids=[3])
    with pl.section_vector():
        ta, tb, t0, t1 = ga.next(), gb.next(), g0.next(), g1.next()
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        _composite(ta, tb, t0, t1)
        pl.store(o0, t0, [0, 0])
        pl.store(o1, t1, [0, 0])


@pl.jit(auto_mutex=True, name="arith_wide")
def arith_wide(
    a: pl.Tensor[[1, N], I32],
    b: pl.Tensor[[1, N], I32],
    lo: pl.Tensor[[1, N], I32],
    hi: pl.Tensor[[1, N], I32],
):
    tt = _tile_type(I32)
    ga = pl.make_tile_group(type=tt, addrs=[_ADDR["a"]], mutex_ids=[0])
    gb = pl.make_tile_group(type=tt, addrs=[_ADDR["b"]], mutex_ids=[1])
    g0 = pl.make_tile_group(type=tt, addrs=[_ADDR["o0"]], mutex_ids=[2])
    g1 = pl.make_tile_group(type=tt, addrs=[_ADDR["o1"]], mutex_ids=[3])
    with pl.section_vector():
        ta, tb, t0, t1 = ga.next(), gb.next(), g0.next(), g1.next()
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        _wide(ta, tb, t0, t1)
        pl.store(lo, t0, [0, 0])
        pl.store(hi, t1, [0, 0])


@pl.jit(auto_mutex=True, name="arith_carry")
def arith_carry(
    a: pl.Tensor[[1, N], U32],
    b: pl.Tensor[[1, N], U32],
    add0: pl.Tensor[[1, N], U32],
    add1: pl.Tensor[[1, N], U32],
    sub0: pl.Tensor[[1, N], U32],
    sub1: pl.Tensor[[1, N], U32],
):
    tt = _tile_type(U32)
    ga = pl.make_tile_group(type=tt, addrs=[_ADDR["a"]], mutex_ids=[0])
    gb = pl.make_tile_group(type=tt, addrs=[_ADDR["b"]], mutex_ids=[1])
    g0 = pl.make_tile_group(type=tt, addrs=[_ADDR["o0"]], mutex_ids=[2])
    g1 = pl.make_tile_group(type=tt, addrs=[_ADDR["o1"]], mutex_ids=[3])
    g2 = pl.make_tile_group(type=tt, addrs=[_ADDR["o2"]], mutex_ids=[4])
    g3 = pl.make_tile_group(type=tt, addrs=[_ADDR["o3"]], mutex_ids=[5])
    with pl.section_vector():
        ta = ga.next()
        tb = gb.next()
        t0 = g0.next()
        t1 = g1.next()
        t2 = g2.next()
        t3 = g3.next()
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        _carry(ta, tb, t0, t1, t2, t3)
        pl.store(add0, t0, [0, 0])
        pl.store(add1, t1, [0, 0])
        pl.store(sub0, t2, [0, 0])
        pl.store(sub1, t3, [0, 0])
