# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。六个激活/融合乘加算子。

**两组，分界线是「读不读 dst 的前值」**：

- `relu` / `neg` / `prelu` / `leaky_relu`：纯函数，只读 src。
- `mul_add_dst` / `mul_dst_add`：**读 dst 的前值**，所以 dst 必须先装好。这是
  [`vf_arith`](../vf_arith/kernel.py) 里 `axpy` 那条规则的第二、第三个实例——解析器按
  **已有绑定**给赋值目标定型，所以 `acc = vf.mul_add_dst(...)` 里的 `acc` 必须先被赋过值。

`prelu` 与 `leaky_relu` 的区别不是写法，是斜率的来源：**`prelu` 的斜率是逐元素的寄存器，
`leaky_relu` 的是一个标量**。本例让两者的斜率不同，否则这一例就看不出为什么有两个 API。

第一手来源：`_vf_api.py` 的签名与 docstring；`pypto/python/tests/st/pypto_pro/frontend/` 下
`relu` / `neg` / `prelu` / `leaky_relu` 各 1 次、`mul_add_dst` 6 次、`mul_dst_add` 1 次调用。
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32

VL_FP32 = 64  # 一个 FP32 寄存器 = 256 字节 = 64 个元素
N = 64
TILE_BYTES = N * 4

_IN = [0x0000 + 0x400 * i for i in range(3)]
_OUT = [0x1000 + 0x400 * i for i in range(4)]


def _tile_type():
    return pl.TileType(shape=[1, N], dtype=FP32, target_memory=pl.MemorySpace.Vec)


# ---------------------------------------------------------------------------
# 一、纯函数：只读 src
# ---------------------------------------------------------------------------


@pl.vector_function
def _unary(src, slope, o_relu, o_neg, o_prelu, o_leaky):
    """四个只读 src 的算子。

    `prelu(src, slope, preg)` 的 `slope` 是**寄存器**，逐元素取；
    `leaky_relu(src, scalar, preg)` 的第二个参数是**标量**，全部 lane 共用一个值。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    a = vf.load_align(src, 0)
    s = vf.load_align(slope, 0)

    r = vf.relu(a, preg)  # max(0, a)
    vf.store_align(o_relu, r, preg, 0)

    r = vf.neg(a, preg)  # -a
    vf.store_align(o_neg, r, preg, 0)

    r = vf.prelu(a, s, preg)  # a >= 0 ? a : a * s[i]
    vf.store_align(o_prelu, r, preg, 0)

    r = vf.leaky_relu(a, pl.const(0.5, FP32), preg)  # a >= 0 ? a : 0.5 * a
    vf.store_align(o_leaky, r, preg, 0)


@pl.jit(auto_mutex=True)
def act_unary(
    a: pl.Tensor[[1, N], FP32],
    slope: pl.Tensor[[1, N], FP32],
    o_relu: pl.Tensor[[1, N], FP32],
    o_neg: pl.Tensor[[1, N], FP32],
    o_prelu: pl.Tensor[[1, N], FP32],
    o_leaky: pl.Tensor[[1, N], FP32],
):
    ga = pl.make_tile_group(type=_tile_type(), addrs=_IN[0], mutex_ids=[0])
    gs = pl.make_tile_group(type=_tile_type(), addrs=_IN[1], mutex_ids=[1])
    g0 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[0], mutex_ids=[2])
    g1 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[1], mutex_ids=[3])
    g2 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[2], mutex_ids=[4])
    g3 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[3], mutex_ids=[5])
    with pl.section_vector():
        ta, ts = ga.current(), gs.current()
        t0, t1, t2, t3 = g0.current(), g1.current(), g2.current(), g3.current()
        pl.load(ta, a, [0, 0])
        pl.load(ts, slope, [0, 0])
        _unary(ta, ts, t0, t1, t2, t3)
        pl.store(o_relu, t0, [0, 0])
        pl.store(o_neg, t1, [0, 0])
        pl.store(o_prelu, t2, [0, 0])
        pl.store(o_leaky, t3, [0, 0])


# ---------------------------------------------------------------------------
# 二、融合乘加：读 dst 的前值
# ---------------------------------------------------------------------------


@pl.vector_function
def _fused(src_a, src_b, src_c, o_mula, o_madd):
    """两个都读 dst，但**读它的位置不同**——这是最容易记混的地方。

    | 算子 | 硬件指令 | 语义 |
    | --- | --- | --- |
    | `mul_add_dst(src0, src1)` | `vmula` | `dst = src0 * src1 + dst`（dst 当**加数**） |
    | `mul_dst_add(src0, src1)` | `vmadd` | `dst = dst * src0 + src1`（dst 当**被乘数**） |

    两行 `acc = vf.load_align(...)` 不是多余的：去掉它，`acc` 没有已有绑定，解析器无法给
    赋值目标定型——和 `axpy` 同一条规则。装进去的值就是公式里的 `dst` 前值。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    a = vf.load_align(src_a, 0)
    b = vf.load_align(src_b, 0)
    c = vf.load_align(src_c, 0)

    acc = vf.load_align(src_c, 0)  # ← dst 前值 = c
    acc = vf.mul_add_dst(a, b, preg)  # c + a*b
    vf.store_align(o_mula, acc, preg, 0)

    acc2 = vf.load_align(src_a, 0)  # ← dst 前值 = a
    acc2 = vf.mul_dst_add(b, c, preg)  # a*b + c
    vf.store_align(o_madd, acc2, preg, 0)


@pl.jit(auto_mutex=True)
def act_fused(
    a: pl.Tensor[[1, N], FP32],
    b: pl.Tensor[[1, N], FP32],
    c: pl.Tensor[[1, N], FP32],
    o_mula: pl.Tensor[[1, N], FP32],
    o_madd: pl.Tensor[[1, N], FP32],
):
    ga = pl.make_tile_group(type=_tile_type(), addrs=_IN[0], mutex_ids=[0])
    gb = pl.make_tile_group(type=_tile_type(), addrs=_IN[1], mutex_ids=[1])
    gc = pl.make_tile_group(type=_tile_type(), addrs=_IN[2], mutex_ids=[2])
    g0 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[0], mutex_ids=[3])
    g1 = pl.make_tile_group(type=_tile_type(), addrs=_OUT[1], mutex_ids=[4])
    with pl.section_vector():
        ta, tb, tc = ga.current(), gb.current(), gc.current()
        t0, t1 = g0.current(), g1.current()
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.load(tc, c, [0, 0])
        _fused(ta, tb, tc, t0, t1)
        pl.store(o_mula, t0, [0, 0])
        pl.store(o_madd, t1, [0, 0])
