# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。VF 逐行稳定 softmax：最大值、广播减、指数、行和、广播除。

每行64个FP32值，恰好一个寄存器。归约的 lane0 必须显式广播，其他 lane
不是同一个统计量。输入/输出使用 tile group，由 auto_mutex 排序搬运与 VF。
更长行和尾部见 algorithms/masked_softmax 与 streaming_masked_softmax。
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32

R = 8  # 行数
C = 64  # 每行元素数 = 一个 FP32 寄存器

_A = 0x0000
_O = [0x1000 + 0x800 * i for i in range(4)]


def _tile(rows, cols):
    return pl.TileType(shape=[rows, cols], dtype=FP32, target_memory=pl.MemorySpace.Vec)


# ---------------------------------------------------------------------------
# 一、纯 vf：一行一个寄存器
# ---------------------------------------------------------------------------


@pl.vector_function
def _softmax_vf(src, dst):
    """五步链，外面套一个行循环。

    **归约结果要 `vf.full` 广播一次才能参与逐元素运算。**
    `reduce_max` / `reduce_sum` 把结果写在 `dst[0]`，其余 lane 是别的东西
    （`reduce_max` 的 `dst[1]` 装极值下标，见 `vf_reduce`）——所以不能直接拿去 `sub`。

    `vf.full(m, preg)` 走的是 **Tensor 模式**：广播源寄存器的一个元素到所有 lane，
    默认 `pos=DuplicatePos.LOWEST`，正好是 `dst[0]`。
    Tensor 模式**必须传 `preg`**（Scalar 模式可以不传）。

    `load_align(src, [i, 0])` 的 `[row, col]` 形式线性偏移是 `row * shape[1] + col`,
    **单位是元素，不是字节**——`offset` 的单位这一点文档写在参数表里，容易看漏。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    for i in pl.range(0, R, 1):
        row = vf.load_align(src, [i, 0])
        m = vf.reduce_max(row, preg)  # m[0] = 行最大；m[1] 是下标，别碰
        mb = vf.full(m, preg)  # Tensor 模式：广播 m[0] 到所有 lane
        shifted = vf.sub(row, mb, preg)
        e = vf.exp(shifted, preg)
        s = vf.reduce_sum(e, preg)  # s[0] = 行和
        sb = vf.full(s, preg)
        out = vf.div(e, sb, preg)
        vf.store_align(dst, out, preg, [i, 0])


@pl.jit(auto_mutex=True)
def softmax_vf(a: pl.Tensor[[R, C], FP32], out: pl.Tensor[[R, C], FP32]):
    ga = pl.make_tile_group(type=_tile(R, C), addrs=_A, mutex_ids=[0])
    g0 = pl.make_tile_group(type=_tile(R, C), addrs=_O[0], mutex_ids=[1])
    with pl.section_vector():
        ta, t0 = ga.current(), g0.current()
        pl.load(ta, a, [0, 0])
        _softmax_vf(ta, t0)
        pl.store(out, t0, [0, 0])


def make_vf_chain(reps):
    """`x <- softmax(x)` 迭代 `reps` 次，纯 vf，就地改写 `work`。"""

    @pl.vector_function
    def _body(work):
        preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        for _ in pl.range(0, reps, 1):
            for i in pl.range(0, R, 1):
                row = vf.load_align(work, [i, 0])
                m = vf.reduce_max(row, preg)
                mb = vf.full(m, preg)
                sh = vf.sub(row, mb, preg)
                e = vf.exp(sh, preg)
                s = vf.reduce_sum(e, preg)
                sb = vf.full(s, preg)
                prob = vf.div(e, sb, preg)
                vf.store_align(work, prob, preg, [i, 0])
            vf.mem_bar(mode=pl.MemBarMode.VST_VLD)  # 写 Tile → 下一轮读同一个 Tile

    @pl.jit(auto_mutex=True)
    def kernel(a: pl.Tensor[[R, C], FP32], out: pl.Tensor[[R, C], FP32]):
        ga = pl.make_tile_group(type=_tile(R, C), addrs=_A, mutex_ids=[0])
        with pl.section_vector():
            ta = ga.current()
            pl.load(ta, a, [0, 0])
            _body(ta)
            pl.store(out, ta, [0, 0])

    return kernel
