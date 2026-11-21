# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。kernel 入口、分块循环、Cube section 与自动流水。

``@pl.jit`` 定义入口，kernel 循环使用 ``pl.range``，``pl.section_cube`` 和
``pl.section_vector`` 指定执行侧。流水由 ``@pl.pipeline.stage`` 与
``PipelineConfig(preload=...)`` 共同声明；preload=0 保留串行顺序并插入核间同步。

样例覆盖逐块缩放、带步长的部分写出、混合 section，以及 preload=0/1/2/3 的
相同数学结果。跨核缓冲的 MemorySpace 可直接声明，也可使用解析器能解析的闭包枚举别名。
``@pl.inline`` 是已废弃的 no-op；普通辅助函数由 JIT 解析器内联。
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

FP32 = pl.DT_FP32
FP16 = pl.DT_FP16
I64 = pl.DT_INT64

TM, TN = 16, 64
NT = 4
FULL_M = TM * NT

STEP = 2
"""`pl.range(0, NT, 2)` 的步长。取 2 而不是 1：步长 1 和不给步长无法区分。"""

PRELOADS = (0, 1, 2, 3)
"""测哪些 preload。**0 必须在里面**——它是「串行 + 跨核同步」那一档,
也是 docstring 说的「先确认串行版对，再往上加」的起点。
3 也必须在：它大于 NT/2，让填充和排空阶段占掉大半个循环。"""


# ---------------------------------------------------------------------------
# 一、pl.range
# ---------------------------------------------------------------------------


@pl.vector_function
def _scale_fp32(tile, scale):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    for offset in pl.range(0, TM * TN, 64):
        value = vf.load_align(tile, offset)
        scaled = vf.muls(value, scale, mask)
        vf.store_align(tile, scaled, mask, offset)


@pl.vector_function
def _scale_fp16(tile, scale):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP16)
    for offset in pl.range(0, TM * TN, 128):
        value = vf.load_align(tile, offset)
        scaled = vf.muls(value, scale, mask)
        vf.store_align(tile, scaled, mask, offset)


@pl.jit(auto_mutex=True)
def loop_over_blocks(
    a: pl.Tensor[[FULL_M, TN], FP32], out: pl.Tensor[[FULL_M, TN], FP32]
):
    """`pl.range(NT)`：一轮搬一块，乘上 `k + 1`。

    **乘的是轮次号而不是常数**：常数的话循环体跑错次数、跑错顺序都看不出来。
    用 `tile_group` + `auto_mutex` 而不是单个 `make_tile`：一个 tile 跨轮次复用
    需要 store 和下一轮 load 之间有同步，而那是 `move_sync` 的题目。
    """
    with pl.section_vector():
        g = pl.make_tile_group(
            type=pl.TileType(
                shape=[TM, TN], dtype=FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[0x0000, 0x4000],
            mutex_ids=[0, 1],
        )
        for k in pl.range(NT):
            t = g.next()
            pl.load(t, a, [k * TM, 0])
            _scale_fp32(t, pl.astype(k + 1, FP32))
            pl.store(out, t, [k * TM, 0])


@pl.jit(auto_mutex=True)
def loop_with_a_step(
    a: pl.Tensor[[FULL_M, TN], FP32], out: pl.Tensor[[FULL_M, TN], FP32]
):
    """`pl.range(0, NT, STEP)` 的三参形式：只碰偶数号块，奇数号保持输出张量的初值。

    **判据靠「没碰的地方没变」**，所以主机侧要先填哨兵。
    """
    with pl.section_vector():
        g = pl.make_tile_group(
            type=pl.TileType(
                shape=[TM, TN], dtype=FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[0x0000, 0x4000],
            mutex_ids=[0, 1],
        )
        for k in pl.range(0, NT, STEP):
            t = g.next()
            pl.load(t, a, [k * TM, 0])
            _scale_fp32(t, pl.astype(k + 1, FP32))
            pl.store(out, t, [k * TM, 0])


@pl.jit(auto_mutex=True)
def loop_with_python_range(
    a: pl.Tensor[[FULL_M, TN], FP32], out: pl.Tensor[[FULL_M, TN], FP32]
):
    """反例：内置 `range`。和 `loop_over_blocks` **只差这一个词**。

    反例：native-loop-iterator —— 原生解析器必须拒绝这里刻意保留的内置 range。
    """
    with pl.section_vector():
        g = pl.make_tile_group(
            type=pl.TileType(
                shape=[TM, TN], dtype=FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[0x0000, 0x4000],
            mutex_ids=[0, 1],
        )
        for k in range(NT):
            t = g.next()
            pl.load(t, a, [k * TM, 0])
            _scale_fp32(t, pl.astype(k + 1, FP32))
            pl.store(out, t, [k * TM, 0])


# ---------------------------------------------------------------------------
# 二、section_cube
# ---------------------------------------------------------------------------


@pl.jit(auto_mutex=True)
def cube_section_moves_through_l1(
    a: pl.Tensor[[TM, TN], FP16], out: pl.Tensor[[TM, TN], FP16]
):
    """`section_cube` 里搬一趟 L1（`MemorySpace.Mat`），再在 vector 段把原值送出去。

    **cube 段那一趟的结果不出去**：它的作用是让这个 kernel 成为混合 kernel。
    输出走 vector 段，所以判据能确认这个 kernel 真跑了
    （而 `support/block_grid` 验证 4 会告诉你混合 kernel 的 `subblock_num` 变成 2）。
    """
    l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TM, TN], dtype=FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x0000,
        mutex_ids=[0],
    )
    ub = pl.make_tile_group(
        type=pl.TileType(shape=[TM, TN], dtype=FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids=[1],
    )
    with pl.section_cube():
        pl.load(l1.current(), a, [0, 0])
    with pl.section_vector():
        t = ub.current()
        pl.load(t, a, [0, 0])
        pl.store(out, t, [0, 0])


# ---------------------------------------------------------------------------
# 三、pl.pipeline
# ---------------------------------------------------------------------------


@pl.pipeline.stage
def _produce(k, scale, src, out, shared_group, ub_group):
    """vector 侧：搬入、乘标量、存回，再把 tile 交给 cube 侧。

    `scale` 从调用处传进来（kernel 体里按轮次算的），流水变换要保证第 k 轮看到的是
    **为第 k 轮算的那个 scale**，不是别的节拍的——这是判据 4 真正在验的东西。
    """
    tile = ub_group.next()
    pl.load(tile, src, [k * TM, 0])
    _scale_fp16(tile, scale)
    pl.store(out, tile, [k * TM, 0])
    handover = shared_group.next()
    pl.move(handover, tile)


@pl.pipeline.stage
def _consume(k, shared_group, left_group):
    """cube 侧：交接的另一半。结果不出去，作用是让这条流水真有两级。"""
    handover = shared_group.next()
    left = left_group.next()
    pl.move(left, handover)


def make_pipelined(preload):
    """`preload` 是这一组唯一的自变量。

    **四个 `target_memory` 全写成字面量 `pl.MemorySpace.X`**：跨核扫描器读 AST,
    别名过不去（见模块 docstring）。
    """

    @pl.jit(auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=preload))
    def kernel(src: pl.Tensor[[FULL_M, TN], FP16], out: pl.Tensor[[FULL_M, TN], FP16]):
        shared = pl.make_tile_group(
            type=pl.TileType(
                shape=[TM, TN],
                dtype=FP16,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.ZN,
            ),
            addrs=[0x0000, 0x8000],
            mutex_ids=[0, 1],
            fwd_ids=[0, 1],
            bwd_ids=[2, 3],
        )
        with pl.section_cube():
            left = pl.make_tile_group(
                type=pl.TileType(
                    shape=[TM, TN],
                    dtype=FP16,
                    target_memory=pl.MemorySpace.Left,
                    layout=pl.NZ,
                ),
                addrs=[0x0000, 0x2000],
                mutex_ids=[4, 5],
            )
        with pl.section_vector():
            ub = pl.make_tile_group(
                type=pl.TileType(
                    shape=[TM, TN], dtype=FP16, target_memory=pl.MemorySpace.Vec
                ),
                addrs=[0x0000, 0x2000],
                mutex_ids=[6, 7],
            )
        for ki in pl.range(0, NT):
            scale = ki + 2
            with pl.section_vector():
                _produce(ki, scale, src, out, shared, ub)
            with pl.section_cube():
                _consume(ki, shared, left)

    return kernel


def make_pipelined_with_an_aliased_memory_space(preload=1):
    """跨核缓冲通过可解析的 MemorySpace 闭包别名声明。"""
    mat = pl.MemorySpace.Mat

    @pl.jit(auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=preload))
    def kernel(src: pl.Tensor[[FULL_M, TN], FP16], out: pl.Tensor[[FULL_M, TN], FP16]):
        shared = pl.make_tile_group(
            type=pl.TileType(
                shape=[TM, TN], dtype=FP16, target_memory=mat, layout=pl.ZN
            ),
            addrs=[0x0000, 0x8000],
            mutex_ids=[0, 1],
            fwd_ids=[0, 1],
            bwd_ids=[2, 3],
        )
        with pl.section_cube():
            left = pl.make_tile_group(
                type=pl.TileType(
                    shape=[TM, TN],
                    dtype=FP16,
                    target_memory=pl.MemorySpace.Left,
                    layout=pl.NZ,
                ),
                addrs=[0x0000, 0x2000],
                mutex_ids=[4, 5],
            )
        with pl.section_vector():
            ub = pl.make_tile_group(
                type=pl.TileType(
                    shape=[TM, TN], dtype=FP16, target_memory=pl.MemorySpace.Vec
                ),
                addrs=[0x0000, 0x2000],
                mutex_ids=[6, 7],
            )
        for ki in pl.range(0, NT):
            scale = ki + 2
            with pl.section_vector():
                _produce(ki, scale, src, out, shared, ub)
            with pl.section_cube():
                _consume(ki, shared, left)

    return kernel
