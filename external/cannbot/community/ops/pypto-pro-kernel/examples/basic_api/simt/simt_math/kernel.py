# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。SIMT 的标量数学：十个超越函数 + `fma` + `abs` / `min` / `max`。

**每个线程算一个元素**，一行一个算子。`pl.simt.*` 这些函数收标量、返标量——和 `vf.*`（整个向量
寄存器）、`pl.*`（整个 tile）是三条不同的面，算同一个数学函数时**精度不保证一致**。

两个输入行分工：

| | 取值 | 给谁用 |
| --- | --- | --- |
| `a` | `(0, 8]` 严格正 | `sqrt` / `rsqrt` / `log` / `log2` / `log1p`（定义域要求） |
| `b` | `[-4, 4]` 含负数和 0 | `exp` / `exp2` / `sin` / `cos` / `tanh` / `abs` / `min` / `max` |

**`a` 不含 0**：`rsqrt(0)` 和 `log(0)` 是 inf/-inf，那属于边界行为，不是这一例的题目
（谓词那一套在 `simt_round_pred`）。**`b` 含 0 和负数**：不含负数的话 `abs` 和 `min` 什么都验不到。

第一手来源：`pypto/python/tests/st/pypto_pro/frontend/simt/scalar/`（15 个文件，
`test_simt_unary.py` 一份就覆盖了十个超越函数）。
"""

import pypto_pro.language as pl

FP32 = pl.DT_FP32

THREADS = 32

# 一行一个算子。顺序和 `reference.py` 的 `OPS` 必须一致——**那张表是唯一的顺序来源**。
ROWS = 14
(
    ROW_SQRT,
    ROW_RSQRT,
    ROW_EXP,
    ROW_EXP2,
    ROW_LOG,
    ROW_LOG2,
    ROW_LOG1P,
    ROW_SIN,
    ROW_COS,
    ROW_TANH,
    ROW_FMA,
    ROW_ABS,
    ROW_MIN,
    ROW_MAX,
) = range(ROWS)

VA_A = 0x0000  # 1 × 32 FP32 = 0x80
VA_B = 0x0080
VA_OUT = 0x0100  # 14 × 32 FP32 = 0x700


@pl.vector_function(mode="simt", max_threads=THREADS)
def scalar_math(a, b, out):
    """十四个算子，每个线程各算一遍自己那一列。"""
    tid = pl.simt.linear_thread_idx()
    av = a[0, tid]
    bv = b[0, tid]

    out[ROW_SQRT, tid] = pl.simt.sqrt(av)
    out[ROW_RSQRT, tid] = pl.simt.rsqrt(av)
    out[ROW_EXP, tid] = pl.simt.exp(bv)
    out[ROW_EXP2, tid] = pl.simt.exp2(bv)
    out[ROW_LOG, tid] = pl.simt.log(av)
    out[ROW_LOG2, tid] = pl.simt.log2(av)
    out[ROW_LOG1P, tid] = pl.simt.log1p(av)
    out[ROW_SIN, tid] = pl.simt.sin(bv)
    out[ROW_COS, tid] = pl.simt.cos(bv)
    out[ROW_TANH, tid] = pl.simt.tanh(bv)
    out[ROW_FMA, tid] = pl.simt.fma(av, bv, av)
    out[ROW_ABS, tid] = pl.simt.abs(bv)
    out[ROW_MIN, tid] = pl.simt.min(av, bv)
    out[ROW_MAX, tid] = pl.simt.max(av, bv)


@pl.jit()
def simt_scalar_math(
    a: pl.Tensor[[1, THREADS], FP32],
    b: pl.Tensor[[1, THREADS], FP32],
    out: pl.Tensor[[ROWS, THREADS], FP32],
):
    """**`[None, 1]` 启动**（调用处），否则 56 个 block 会往同一片 GM 各写一遍。

    这一例的结果是覆盖写而不是累加，所以多 block 不会算错——但 `simt_launch` §四 的教训是
    「一个从没影响过结果的事实，不会在需要它的时候被想起来」，所以这里也照写。
    """
    ta = pl.make_tile(
        pl.TileType(shape=[1, THREADS], dtype=FP32, target_memory=pl.MemorySpace.Vec),
        addr=VA_A,
    )
    tb = pl.make_tile(
        pl.TileType(shape=[1, THREADS], dtype=FP32, target_memory=pl.MemorySpace.Vec),
        addr=VA_B,
    )
    tout = pl.make_tile(
        pl.TileType(
            shape=[ROWS, THREADS], dtype=FP32, target_memory=pl.MemorySpace.Vec
        ),
        addr=VA_OUT,
    )
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
        )
        scalar_math[THREADS](ta, tb, tout)
        pl.system.sync_src(
            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
        )
        pl.store(out, tout, [0, 0])
