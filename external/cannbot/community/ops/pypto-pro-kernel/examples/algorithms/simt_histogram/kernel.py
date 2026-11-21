# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。UINT8 直方图：直接 GM 原子与每 block 的 UB 局部聚合。

输入是连续 UINT8[N]，输出是连续 INT32[256]，二者不重叠，要求 N <= INT32_MAX。
每次调用先用独立原生 kernel 清零输出，同一 Stream 后续更新无需全网格屏障。
local=True 时，每 block 初始化自己的 256 个 UB 计数，再累加并汇总到 GM。
两个线程块屏障均位于线程控制流之外；没有输入的线程也必须到达。
local=False 保留直接 GM 原子对照，combine_runs=True 则合并同一线程遍历中的连续相同目标。
三条路径都使用两个原生调用，空输入只清零；线程内合并不引入共享状态或额外屏障。
线程数和 worker 数为显式配置；默认值不是性能最优声明。
来源机制：Ascend 950 SIMT 原子优化指南；接口事实和数值验收以当前 PyPTO 安装为准。
原子更新显式接收容器与逐轴元素坐标；数值验收使用 test.py 的独立频数参考、输入保持和输出守卫。
example-query: {"aliases":["SIMT local histogram","SIMT local counters"],"term_groups":[["simt","ub","block-local"],["histogram","counters","categorical","直方图"]]}
"""

from functools import lru_cache

import pypto_pro.language as pl

I32 = pl.DT_INT32
U8 = pl.DT_UINT8
BINS = 256


@lru_cache(None)
def make_histogram(n, threads=256, local=True, combine_runs=False):
    """Return native initialization and update kernels for one physical shape."""
    if not 0 <= n <= 2**31 - 1 or threads not in (64, 128, 256, 512):
        raise ValueError("require 0 <= N <= INT32_MAX and threads in 64/128/256/512")
    if local and combine_runs:
        raise ValueError("run combining is a direct-GM alternative; use local=False")

    @pl.vector_function(mode="simt", max_threads=threads)
    def clear_body(out):
        first = pl.simt.block_idx().x * threads + pl.simt.linear_thread_idx()
        step = pl.simt.grid_dim().x * threads
        for j in pl.range(first, BINS, step):
            out[j] = pl.simt.cast(0, I32)

    @pl.jit()
    def initialize(out: pl.Tensor[[BINS], I32]):
        with pl.section_vector():
            clear_body[threads](out)

    if n == 0:
        return initialize, None

    @pl.vector_function(mode="simt", max_threads=threads)
    def global_body(x, out):
        first = pl.simt.block_idx().x * threads + pl.simt.linear_thread_idx()
        step = pl.simt.grid_dim().x * threads
        one = pl.simt.cast(1, I32)
        for i in pl.range(first, n, step):
            pl.simt.atomic_add(out, [x[i]], one)

    @pl.vector_function(mode="simt", max_threads=threads)
    def runs_body(x, out):
        first = pl.simt.block_idx().x * threads + pl.simt.linear_thread_idx()
        step = pl.simt.grid_dim().x * threads
        one = pl.simt.cast(1, I32)
        if first < n:
            key = x[first]
            count = one
            for i in pl.range(first + step, n, step):
                next_key = x[i]
                if next_key == key:
                    count = count + one
                else:
                    pl.simt.atomic_add(out, [key], count)
                    key = next_key
                    count = one
            pl.simt.atomic_add(out, [key], count)

    @pl.vector_function(mode="simt", max_threads=threads)
    def local_body(x, counts, out):
        tid = pl.simt.linear_thread_idx()
        for j in pl.range(tid, BINS, threads):
            counts[0, j] = pl.simt.cast(0, I32)
        pl.simt.syncthreads()
        first = pl.simt.block_idx().x * threads + tid
        step = pl.simt.grid_dim().x * threads
        one = pl.simt.cast(1, I32)
        for i in pl.range(first, n, step):
            pl.simt.atomic_add(counts, [0, x[i]], one)
        pl.simt.syncthreads()
        for j in pl.range(tid, BINS, threads):
            pl.simt.atomic_add(out, [j], counts[0, j])

    if local:

        @pl.jit(auto_mutex=False)
        def accumulate(x: pl.Tensor[[n], U8], out: pl.Tensor[[BINS], I32]):
            counts = pl.make_tile(
                pl.TileType(
                    shape=[1, BINS], dtype=I32, target_memory=pl.MemorySpace.Vec
                ),
                addr=0,
            )
            with pl.section_vector():
                local_body[threads](x, counts, out)
    elif combine_runs:

        @pl.jit()
        def accumulate(x: pl.Tensor[[n], U8], out: pl.Tensor[[BINS], I32]):
            with pl.section_vector():
                runs_body[threads](x, out)
    else:

        @pl.jit()
        def accumulate(x: pl.Tensor[[n], U8], out: pl.Tensor[[BINS], I32]):
            with pl.section_vector():
                global_body[threads](x, out)

    return initialize, accumulate


def run(x, out, *, workers=4, threads=256, local=True, combine_runs=False):
    """Overwrite every bin; Host reads only shape, dtype and layout metadata."""
    if str(x.dtype) != "torch.uint8" or str(out.dtype) != "torch.int32":
        raise TypeError("require UINT8 input and INT32 output")
    if (
        x.ndim != 1
        or tuple(out.shape) != (BINS,)
        or not x.is_contiguous()
        or not out.is_contiguous()
    ):
        raise ValueError("require contiguous input[N] and output[256]")
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be a positive legal Stream block count")
    initialize, accumulate = make_histogram(
        int(x.shape[0]), threads, local, combine_runs
    )
    initialize[None, workers](out)
    if accumulate is not None:
        accumulate[None, workers](x, out)
