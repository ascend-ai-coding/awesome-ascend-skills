# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。进 SIMT 的入口：`@pl.vector_function(mode="simt")` 与启动括号。

**SIMT 是第三条执行面**，和 vf（显式向量寄存器）、block op（整 tile 一条指令）并列。
一个 SIMT 函数里写的是**一个线程**做什么，`tid` 从 `pl.simt.linear_thread_idx()` 拿。

```python
@pl.vector_function(mode="simt", max_threads=16)     # 声明线程上限
def body(dst):
    tid = pl.simt.linear_thread_idx()
    dst[0, tid] = tid                                # tile 按元素下标访问

@pl.jit()
def kernel(out):
    ...
    with pl.section_vector():
        body[4, 2, 2](dst)                           # 启动：block 形状 4×2×2
```

**两层"启动括号"，容易混：**

| 括号 | 管什么 | 写在哪 |
| --- | --- | --- |
| `body[x, y, z](...)` | 一个 block 里线程怎么排（thread block 形状） | kernel 体内 |
| `kernel[None, 1](...)` | **几个 block（AI 核）在跑** | 主机侧调用处 |

**第二层不写的话默认不是 1。** 直接 `kernel(out)` 走 `_AUTO_BLOCK_DIM = 0`，native 侧解成当前
stream 的全量核数（这台机器 56）。这些 block 都跑同一份代码、写同一片 GM，于是
`block_idx` 读回来是**哪个 block 最后写赢**，不可复现。实测见 `test.py` §四。

`warp_size()` 在这台机器上是 **32**（实测）。

第一手来源：`pypto/python/tests/st/pypto_pro/frontend/simt/thread/test_thread_context.py`
（整个形状照它抄，包括 `[None, 1]` 那个启动括号）与 `launch/`、`synchronization/` 两组。
"""

import pypto_pro.language as pl

I32 = pl.DT_INT32

# block 形状取 4 × 2 × 2：三个轴都 > 1 且**互不相等**。
# 取 [16, 1, 1] 的话线性化规则怎么错都看不出来；取 [2, 2, 2] 的话三个轴的系数分不开。
THREADS_X = 4
THREADS_Y = 2
THREADS_Z = 2
THREADS = THREADS_X * THREADS_Y * THREADS_Z  # 16

# 八行，一行一个量：**不要把它们打包成一个数**。ST 那样打包能省行数，但对不上的时候
# 分不清是哪个量错了。
ROW_THREAD_X = 0
ROW_THREAD_Y = 1
ROW_THREAD_Z = 2
ROW_BLOCK_DIM_X = 3
ROW_LINEAR_TID = 4
ROW_WARP_SIZE = 5
ROW_BLOCK_IDX_X = 6
ROW_GRID_DIM_X = 7
ROWS = 8

VA_CTX = 0x0000
VA_SCRATCH = 0x0000
VA_RESULT = 0x0100


@pl.vector_function(mode="simt", max_threads=THREADS)
def write_thread_context(dst):
    """每个线程把自己看到的 7 个上下文量写进自己那一列。

    `pl.simt.warp_size()` 要过一次 `pl.simt.cast`：它的类型不是 INT32，直接存进 INT32 tile
    会类型不匹配（ST 里也是先 cast 再存）。
    """
    thread = pl.simt.thread_idx()
    bdim = pl.simt.block_dim()
    bidx = pl.simt.block_idx()
    gdim = pl.simt.grid_dim()
    tid = pl.simt.linear_thread_idx()

    dst[ROW_THREAD_X, tid] = thread.x
    dst[ROW_THREAD_Y, tid] = thread.y
    dst[ROW_THREAD_Z, tid] = thread.z
    dst[ROW_BLOCK_DIM_X, tid] = bdim.x
    dst[ROW_LINEAR_TID, tid] = tid
    dst[ROW_WARP_SIZE, tid] = pl.simt.cast(pl.simt.warp_size(), I32)
    dst[ROW_BLOCK_IDX_X, tid] = bidx.x
    dst[ROW_GRID_DIM_X, tid] = gdim.x


@pl.jit()
def thread_context(out: pl.Tensor[[ROWS, THREADS], I32]):
    """**这个 kernel 被两种启动方式各测一次**，见 `test.py` §四。

    kernel 体本身不知道有几个 block 在跑——那是调用处的事。
    """
    dst = pl.make_tile(
        pl.TileType(shape=[ROWS, THREADS], dtype=I32, target_memory=pl.MemorySpace.Vec),
        addr=VA_CTX,
    )
    with pl.section_vector():
        write_thread_context[THREADS_X, THREADS_Y, THREADS_Z](dst)
        pl.system.sync_src(
            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0
        )
        pl.store(out, dst, [0, 0])


@pl.vector_function(mode="simt", max_threads=THREADS)
def rotate_after_sync(scratch, result):
    """每个线程先写自己的槽，`syncthreads()` 之后读**邻居**的槽。

    没有那条 `syncthreads()` 的话，读邻居时邻居可能还没写——结果取决于线程的推进步调。
    有了它，`result[tid] == scratch[(tid + 1) % THREADS]` 是确定的。

    **这是能给 `syncthreads` 写判据的最小形状**：必须让一个线程去读**别的线程**写的东西,
    只读自己写的东西根本用不着同步。
    """
    tid = pl.simt.linear_thread_idx()
    scratch[0, tid] = tid * 10
    pl.simt.syncthreads()
    neighbour = (tid + 1) % THREADS
    result[0, tid] = scratch[0, neighbour]


@pl.jit()
def rotate_through_shared_ub(out: pl.Tensor[[1, THREADS], I32]):
    """两块 UB：一块给线程之间交换用，一块放结果。

    分成两块而不是原地转：原地转的话「读到的是旧值还是新值」本身就是要验的东西，
    判据会和被验的对象缠在一起。
    """
    scratch = pl.make_tile(
        pl.TileType(shape=[1, THREADS], dtype=I32, target_memory=pl.MemorySpace.Vec),
        addr=VA_SCRATCH,
    )
    result = pl.make_tile(
        pl.TileType(shape=[1, THREADS], dtype=I32, target_memory=pl.MemorySpace.Vec),
        addr=VA_RESULT,
    )
    with pl.section_vector():
        rotate_after_sync[THREADS](scratch, result)
        pl.system.sync_src(
            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0
        )
        pl.store(out, result, [0, 0])
