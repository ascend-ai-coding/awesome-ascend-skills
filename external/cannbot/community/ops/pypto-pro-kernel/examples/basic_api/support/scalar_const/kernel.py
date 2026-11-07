# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。标量边界、字面量构造与 VF 比较/归约。

pl.min/max 处理标量控制，pl.const 构造字面量，pl.astype 转换标量。
逐元素比较、夹逼和行/列归约全部在 VF 内实现。行归约的两个输出布局
分别使用 DN 紧凑列和 ND 补齐行，公开输出均为 [M,1]。
原生启动、搬运与 tile group 同步保持显式；数值参考不依赖编译器。
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

FP32 = pl.DT_FP32
I32 = pl.DT_INT32
I64 = pl.DT_INT64
VEC = pl.MemorySpace.Vec

M = 16
K = 64

_A = 0x0000
_B = 0x4000
_OUT = 0x8000

CLAMP_LOW = -0.5
CLAMP_HIGH = 0.5
"""夹逼用的两个界。**只给参考和判据用，kernel 里必须写成字面量**——
`pl.const` 的检查是纯语法的（`isinstance(node, ast.Constant)`），常量名会被拒。
所以这两个值在 `kernel.py` 里出现两次，一次是这里的名字、一次是 `elementwise_tile_and_scalar`
里的字面量。**这是框架的约束，不是重复代码没清理**，见 `test.py` §六。"""

LOOP_LIMIT = 4
"""`pl.min(block_num, LOOP_LIMIT)` 的上限。取 4 是因为判据会用 `blocks=1` 和 `blocks=8`
各跑一次——一次让 `min` 取到 `block_num`，一次取到 `LOOP_LIMIT`，两条分支都走到。"""


# ---------------------------------------------------------------------------
# 一、标量 min / max
# ---------------------------------------------------------------------------


@pl.vector_function
def _pair_max(dst, lhs, rhs):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    for row in pl.range(M):
        a = vf.load_align(lhs, [row, 0])
        b = vf.load_align(rhs, [row, 0])
        result = vf.max(a, b, mask)
        vf.store_align(dst, result, mask, [row, 0])


@pl.vector_function
def _clamp(dst, src):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    for row in pl.range(M):
        a = vf.load_align(src, [row, 0])
        low = vf.maxs(a, pl.const(-0.5, FP32), mask)
        result = vf.mins(low, pl.const(0.5, FP32), mask)
        vf.store_align(dst, result, mask, [row, 0])


@pl.vector_function
def _row_max(dst, src, pitch):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    for row in pl.range(M):
        a = vf.load_align(src, [row, 0])
        maximum = vf.reduce_max(a, mask)
        vf.store_align(
            dst, maximum, mask, row * pitch, dist=pl.StoreDist.FIRST_ELEMENT_B32
        )


@pl.vector_function
def _col_min(dst, src):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    minimum = vf.load_align(src, [0, 0])
    for row in pl.range(1, M):
        a = vf.load_align(src, [row, 0])
        minimum = vf.min(minimum, a, mask)
    vf.store_align(dst, minimum, mask, [0, 0])


@pl.jit()
def scalar_bounds(out: pl.Tensor[[4], I64]):
    """`pl.min` / `pl.max` 作用在标量上，典型用途是算循环边界。

    四个量：`block_num`、`min(block_num, 4)`、`max(block_num, 4)`、以及
    `min` 和 `max` 之和——**和必须等于 `block_num + 4`**，这是个不依赖具体核数的恒等式。
    """
    num = pl.get_block_num()
    lo = pl.min(num, LOOP_LIMIT)
    hi = pl.max(num, LOOP_LIMIT)
    with pl.section_vector():
        pl.setval(out, 0, num)
        pl.setval(out, 1, lo)
        pl.setval(out, 2, hi)
        pl.setval(out, 3, lo + hi)


@pl.jit()
def const_and_astype(out: pl.Tensor[[4], I64]):
    """`pl.const` 造编译期常量，`pl.astype` 转运行期表达式。

    第 3、4 个量是同一个数走两条路：`pl.const(7, I32)` 与 `pl.astype(7, I32)`。
    **`astype` 的 docstring 说它转 "runtime scalar expression"，但字面量也吃**（实测）。
    """
    typed = pl.const(7, I32)
    converted = pl.astype(7, I32)
    from_runtime = pl.astype(pl.get_block_num(), I32)
    with pl.section_vector():
        pl.setval(out, 0, typed)
        pl.setval(out, 1, converted)
        pl.setval(out, 2, from_runtime)
        pl.setval(out, 3, pl.const(-3, I64))


# ---------------------------------------------------------------------------
# 二、逐元素
# ---------------------------------------------------------------------------


@pl.jit(auto_mutex=True)
def elementwise_tile_and_scalar(
    a: pl.Tensor[[M, K], FP32],
    b: pl.Tensor[[M, K], FP32],
    o_pair: pl.Tensor[[M, K], FP32],
    o_clamped: pl.Tensor[[M, K], FP32],
):
    """VF 逐元素最大值与标量夹逼，两个输出独立验证。

    第二个输出是一次**夹逼**：先 `maximum` 到下界，再 `minimum` 到上界。
    VF 使用 `max` / `maxs` / `mins`，
    比分别测两个算子更能说明它们怎么配合。
    """
    ga = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=FP32, target_memory=VEC),
        addrs=_A,
        mutex_ids=[0],
    )
    gb = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=FP32, target_memory=VEC),
        addrs=_B,
        mutex_ids=[1],
    )
    go = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=FP32, target_memory=VEC),
        addrs=_OUT,
        mutex_ids=[2],
    )
    with pl.section_vector():
        ta, tb, to = ga.current(), gb.current(), go.current()
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        _pair_max(to, ta, tb)
        pl.store(o_pair, to, [0, 0])
        # 字面量，不是 `pl.const(CLAMP_LOW, FP32)`：常量名会被解析期拒（见 test.py §六）。
        # 这两个数必须和模块顶上的 `CLAMP_LOW` / `CLAMP_HIGH` 一致。
        _clamp(to, ta)
        pl.store(o_clamped, to, [0, 0])


# ---------------------------------------------------------------------------
# 三、归约：两个 dim
# ---------------------------------------------------------------------------


@pl.jit(auto_mutex=True)
def row_max_dim0(a: pl.Tensor[[M, K], FP32], out: pl.Tensor[[M, 1], FP32]):
    """VF 沿末轴取最大值，输出公开形状 [M,1]；物理列步长为 1。"""
    ga = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=FP32, target_memory=VEC),
        addrs=_A,
        mutex_ids=[0],
    )
    go = pl.make_tile_group(
        type=pl.TileType(
            shape=[M, 1],
            dtype=FP32,
            target_memory=VEC,
            layout=pl.DN,
            valid_shape=[M, 1],
        ),
        addrs=_OUT,
        mutex_ids=[1],
    )
    with pl.section_vector():
        ta, to = ga.current(), go.current()
        pl.load(ta, a, [0, 0])
        _row_max(to, ta, 1)
        pl.store(out, to, [0, 0])


@pl.jit(auto_mutex=True)
def row_max_dim0_without_dn(a: pl.Tensor[[M, K], FP32], out: pl.Tensor[[M, 1], FP32]):
    """VF 沿末轴取最大值，输出公开形状 [M,1]；物理列步长为 8。"""
    ga = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=FP32, target_memory=VEC),
        addrs=_A,
        mutex_ids=[0],
    )
    go = pl.make_tile_group(
        type=pl.TileType(
            shape=[M, 8],
            dtype=FP32,
            target_memory=VEC,
            layout=pl.ND,
            valid_shape=[M, 1],
        ),
        addrs=_OUT,
        mutex_ids=[1],
    )
    with pl.section_vector():
        ta, to = ga.current(), go.current()
        pl.load(ta, a, [0, 0])
        _row_max(to, ta, 8)
        pl.store(out, to, [0, 0])


@pl.jit(auto_mutex=True)
def col_min_dim1(a: pl.Tensor[[M, K], FP32], out: pl.Tensor[[1, K], FP32]):
    """VF 沿行循环，每个 lane 保留独立列，输出 [1,K]。"""
    ga = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=FP32, target_memory=VEC),
        addrs=_A,
        mutex_ids=[0],
    )
    go = pl.make_tile_group(
        type=pl.TileType(shape=[1, K], dtype=FP32, target_memory=VEC),
        addrs=_OUT,
        mutex_ids=[1],
    )
    with pl.section_vector():
        ta, to = ga.current(), go.current()
        pl.load(ta, a, [0, 0])
        _col_min(to, ta)
        pl.store(out, to, [0, 0])


# ---------------------------------------------------------------------------
# 四、被拒的写法
# ---------------------------------------------------------------------------


@pl.jit(auto_mutex=True)
def scalar_min_on_tiles(a: pl.Tensor[[M, K], FP32], out: pl.Tensor[[M, K], FP32]):
    """反例：`pl.min` 是 scalar-only，给它 tile。逐元素计算应在 VF 中使用 `vf.min`。"""
    ga = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=FP32, target_memory=VEC),
        addrs=_A,
        mutex_ids=[0],
    )
    go = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=FP32, target_memory=VEC),
        addrs=_OUT,
        mutex_ids=[1],
    )
    with pl.section_vector():
        ta, to = ga.current(), go.current()
        pl.load(ta, a, [0, 0])
        pl.min(ta, to)
        pl.store(out, to, [0, 0])


@pl.jit()
def const_of_a_named_constant(out: pl.Tensor[[1], I64]):
    """反例：`pl.const` 连**模块级常量名**都拒。

    `CLAMP_LOW` 在任何意义上都是编译期常量，但检查看的是 AST 节点类型
    （`isinstance(value_node, ast.Constant)`，`scalar_ops.py:214`），
    `ast.Name` 过不去。

    `pl.const` 的字面量限制见[写法约束](../../../../references/api/constraints.md)；GEMM MX 的当前 K 约束见[GEMM MX 契约
    ](../../../../references/api/gemm-mx.md)。
    """
    value = pl.const(CLAMP_LOW, FP32)
    with pl.section_vector():
        pl.setval(out, 0, pl.astype(value, I64))


@pl.jit()
def const_of_a_runtime_value(out: pl.Tensor[[1], I64]):
    """反例：`pl.const` 的第一个参数必须是数字字面量，给运行期标量会被拒。

    **要转运行期标量用 `pl.astype`**——这是两个接口的实际分界,
    而从名字（const / astype）看不出「一个只吃字面量」。
    """
    value = pl.const(pl.get_block_idx(), I32)
    with pl.section_vector():
        pl.setval(out, 0, value)
