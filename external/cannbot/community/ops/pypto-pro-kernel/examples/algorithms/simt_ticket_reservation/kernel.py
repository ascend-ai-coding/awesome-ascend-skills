# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。SIMT 单元素预留与线程内批量 ticket 预留。

契约：连续 INT32 keys[N]，0<=keys[i]<B，1<=B<=256，0<=N<=INT32_MAX。
counts[B] 和 tickets[N] 均覆盖旧内容，输入与两个输出互不重叠。
每个 key 的 tickets 是 0..counts[key]-1 的排列，不承诺输入顺序或跨线程顺序。
先以独立原生调用清零 counts，同一 Stream 后续预留；N=0 时只清零。
combine_runs=True 合并同一线程实际 grid-stride 遍历中的相同 key 连续段。
atomic_add 返回该段的起点，再回访段内位置写 base+rank；不需要线程私有数组。
每个元素的 ticket 语义仍被保留；不支持观察逐项中间状态或更强顺序的调用者。
atomic_add 显式接收 counts 与单轴元素坐标，返回更新前的 INT32 计数；数值验收见 test.py。
example-query: {"aliases":["SIMT ticket ranges","SIMT batched reservation"],"term_groups":[["simt","atomic"],["ticket","reservation","票据"]]}
"""

from functools import lru_cache

import pypto_pro.language as pl

I32 = pl.DT_INT32


@lru_cache(None)
def make_reservation(n, bins, threads=128, combine_runs=True):
    if (
        not 0 <= n <= 2**31 - 1
        or not 1 <= bins <= 256
        or threads not in (64, 128, 256, 512)
    ):
        raise ValueError("require bounded N, 1<=B<=256 and threads 64/128/256/512")

    @pl.vector_function(mode="simt", max_threads=threads)
    def clear_body(counts):
        first = pl.simt.block_idx().x * threads + pl.simt.linear_thread_idx()
        step = pl.simt.grid_dim().x * threads
        for key in pl.range(first, bins, step):
            counts[key] = pl.simt.cast(0, I32)

    @pl.jit()
    def initialize(counts: pl.Tensor[[bins], I32]):
        with pl.section_vector():
            clear_body[threads](counts)

    if n == 0:
        return initialize, None

    @pl.vector_function(mode="simt", max_threads=threads)
    def direct_body(keys, counts, tickets):
        first = pl.simt.block_idx().x * threads + pl.simt.linear_thread_idx()
        step = pl.simt.grid_dim().x * threads
        one = pl.simt.cast(1, I32)
        for i in pl.range(first, n, step):
            tickets[i] = pl.simt.atomic_add(counts, [keys[i]], one)

    @pl.vector_function(mode="simt", max_threads=threads)
    def range_body(keys, counts, tickets):
        first = pl.simt.block_idx().x * threads + pl.simt.linear_thread_idx()
        step = pl.simt.grid_dim().x * threads
        one = pl.simt.cast(1, I32)
        if first < n:
            run_key = keys[first]
            run_start = first
            run_count = one
            for i in pl.range(first + step, n, step):
                key = keys[i]
                if key == run_key:
                    run_count = run_count + one
                else:
                    base = pl.simt.atomic_add(counts, [run_key], run_count)
                    pos = run_start
                    ticket = base
                    while pos < i:
                        tickets[pos] = ticket
                        pos = pos + step
                        ticket = ticket + one
                    run_key = key
                    run_start = i
                    run_count = one
            base = pl.simt.atomic_add(counts, [run_key], run_count)
            pos = run_start
            ticket = base
            while pos < n:
                tickets[pos] = ticket
                pos = pos + step
                ticket = ticket + one

    if combine_runs:

        @pl.jit()
        def reserve(
            keys: pl.Tensor[[n], I32],
            counts: pl.Tensor[[bins], I32],
            tickets: pl.Tensor[[n], I32],
        ):
            with pl.section_vector():
                range_body[threads](keys, counts, tickets)
    else:

        @pl.jit()
        def reserve(
            keys: pl.Tensor[[n], I32],
            counts: pl.Tensor[[bins], I32],
            tickets: pl.Tensor[[n], I32],
        ):
            with pl.section_vector():
                direct_body[threads](keys, counts, tickets)

    return initialize, reserve


def run(keys, counts, tickets, *, workers=4, threads=128, combine_runs=True):
    """Use metadata only on Host; every reservation remains a native SIMT update."""
    if any(str(x.dtype) != "torch.int32" for x in (keys, counts, tickets)):
        raise TypeError("require INT32 input and outputs")
    if (
        keys.ndim != 1
        or counts.ndim != 1
        or tuple(tickets.shape) != tuple(keys.shape)
        or not all(x.is_contiguous() for x in (keys, counts, tickets))
    ):
        raise ValueError("require contiguous keys[N], counts[B] and tickets[N]")
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be a positive legal Stream block count")
    initialize, reserve = make_reservation(
        keys.shape[0], counts.shape[0], threads, combine_runs
    )
    initialize[None, workers](counts)
    if reserve is not None:
        reserve[None, workers](keys, counts, tickets)
