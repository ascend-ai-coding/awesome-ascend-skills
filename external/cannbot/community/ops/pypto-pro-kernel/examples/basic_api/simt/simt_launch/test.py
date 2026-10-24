# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

r"""进 SIMT 的入口：`@pl.vector_function(mode="simt")`、线程上下文、`syncthreads`。

**写法：原始。** 这是 simt 一章的第一例——后面四例都假定这里的形状已经懂了。

## 一、SIMT 是第三条执行面

| 面 | 一条语句做什么 | 入口 |
| --- | --- | --- |
| block op（`pl.add` 等） | 整个 tile | `with pl.section_vector():` |
| vf（`vf.*`） | 一个向量寄存器 | ``@pl.vector_function`` |
| **SIMT** | **一个线程** | ``@pl.vector_function(mode="simt", max_threads=N)`` |

SIMT 函数体里写的是**一个线程**做什么，线程号从 `pl.simt.linear_thread_idx()` 拿，
tile 用**元素下标**访问（`dst[row, tid]`），不是整块操作。

## 二、两层启动括号，别混

| 括号 | 管什么 | 写在哪 |
| --- | --- | --- |
| ``body[x, y, z](...)`` | 一个 block 里线程怎么排 | kernel 体内 |
| ``kernel[None, 1](...)`` | **几个 block（AI 核）在跑** | 主机侧调用处 |

**第二层不写的话默认不是 1。** 见 §四——这一例专门用同一个 kernel 的两种启动方式把它钉住。

## 三、如何独立验证

1. ``thread_context`` 用 ``[None, 1]`` 启动：
   - `thread.x / y / z` 分别等于 `tid % 4`、`(tid // 4) % 2`、`tid // 8`；
   - `block_dim.x == 4`；
   - `linear_thread_idx` 等于 `0..15`；
   - `warp_size == 32`；
   - `block_idx.x == 0`、`grid_dim.x == 1`（**因为带了启动括号**）。
2. **线性化恒等式**：`linear_tid == x + y * bdim.x + z * bdim.x * bdim.y`。
   这条不比任何预期值，**只比实测的三个轴和实测的线性号**——所以它验的是框架内部一致，
   而不是"我猜的公式对不对"。
3. ``thread_context`` **不带**启动括号：`grid_dim.x > 1`（这台机器 56），
   而 `block_idx.x` **不写判据**（见 §四）。
4. ``rotate_through_shared_ub``：每个线程读到邻居写的值 → `result[tid] == ((tid+1) % 16) * 10`。

block 形状取 `4 × 2 × 2`——**三个轴都 > 1 且互不相等**。取 `[16, 1, 1]` 的话线性化怎么错都看不
出来；取 `[2, 2, 2]` 的话三个轴的系数分不开。八个量分八行存，**不打包成一个数**：ST 那样打包省
行数，但对不上时分不清是哪个量错了。

## 四、不带启动括号时有 56 个 block 在跑（实测）

同一个 `thread_context`，只改调用方式：

| 调用 | `grid_dim.x` | `block_idx.x` |
| --- | ---: | --- |
| ``kernel[None, 1](out)`` | **1** | **0** |
| ``kernel(out)`` | **56**（= 这台机器的核数） | 每次不一样 |

直接调用走 `_AUTO_BLOCK_DIM = 0`，native 侧解成当前 stream 的全量核数（`jit.py:1883`）。
这 56 个 block 跑同一份代码、写同一片 GM，所以 `block_idx` 读回来是**哪个 block 最后写赢**
——探针里量到过 `9`，那个数字没有任何意义。**所以第 3 条判据只断言 `> 1`，不断言具体是谁。**

`grid_dim.x == 56` 这个数是**机器相关**的，所以也不写进判据（基线数字会漂，
`references/verification/board-runs.md` §六）。判据只说"大于 1"，那是和机器无关的那一半事实。

**这条在本仓栽过一次**（`references/repeated-mistakes.md` #9）：`fixpipe_riders` 的 atomic 测试
直接 `kernel(...)` 调用，于是 56 个 block 各往同一片 GM 累加一次，结果是期望值的 28 倍，
而我第一反应是去查 atomic 的语义。**这一例把它变成一条会红的判据**，而不是一段说明。

## 五、`syncthreads` 的判据必须让线程读别人写的东西

验证 4 的形状是刻意的：每个线程先写自己的槽，`syncthreads()` 之后读**邻居**的槽。
**只读自己写的东西根本用不着同步**，那样的"判据"在去掉 `syncthreads()` 之后照样绿。

用两块 UB（一块交换、一块放结果）而不是原地转：原地转的话「读到的是旧值还是新值」本身就是要验
的东西，判据会和被验对象缠在一起。

**没有测"去掉 syncthreads 会怎样"**：那是时序相关的，可能碰巧对（本章 `move_sync` §五 量过
同一类现象：冷进程里连完全不同步都对）。**这是明说跳过**，不是"验过没问题"。

## 六、这一例不讲什么

- **标量运算**——`pl.simt.sqrt` / `exp` / `fma` 等在 ``simt_math``。
- **取整与谓词**——五种取整、`isnan` 系在 ``simt_round_pred``。
- **原子操作**——九个原子族在 ``simt_atomic``；环形增减与 `threadfence` 在 ``simt_ring_fence``。
- **核／子核编号**——`pl.get_block_idx()` 那一套在 support 一章的 ``block_grid``。

  **2026-09-28 更正**：这里原来写的是「两套编号不是一回事」，**那句话不对**。
  `block_grid` 验证 5 实测两套报**同一个数**（`pl.get_block_idx() == pl.simt.block_idx().x`,
  `pl.get_block_num() == pl.simt.grid_dim().x`，`blocks=1` 和 `blocks=4` 都对得上）。
  真正的区别是**能在哪儿调**：`pl.get_block_idx()` 在 SIMT 体里被解析期拒
  （`_call_parser.py:1101`）。作用域不同，语义相同。

设计依据见 [tooling/example-coverage.md](../../../../references/tooling/example-coverage.md) §二 三节。
"""

import numpy as np
import torch

from .kernel import (
    ROW_BLOCK_DIM_X,
    ROW_BLOCK_IDX_X,
    ROW_GRID_DIM_X,
    ROW_LINEAR_TID,
    ROW_THREAD_X,
    ROW_THREAD_Y,
    ROW_THREAD_Z,
    ROW_WARP_SIZE,
    ROWS,
    THREADS,
    THREADS_X,
    THREADS_Y,
    rotate_through_shared_ub,
    thread_context,
)
from .reference import WARP_SIZE, linearised, rotated, thread_x, thread_y, thread_z


def _run_context(device, *, one_block):
    """`one_block=True` 走 `kernel[None, 1]`，否则直接调用。**这是这一例唯一的自变量。**"""
    out = torch.full((ROWS, THREADS), -1, dtype=torch.int32, device=device)
    if one_block:
        thread_context[None, 1](out)
    else:
        thread_context(out)
    torch.npu.synchronize()
    return out.cpu().numpy()


def test_every_thread_sees_its_own_coordinates(device):
    """验证 1。七个量逐个断言，不打包。"""
    got = _run_context(device, one_block=True)
    tid = np.arange(THREADS, dtype=np.int32)

    np.testing.assert_array_equal(got[ROW_THREAD_X], thread_x(tid, THREADS_X))
    np.testing.assert_array_equal(
        got[ROW_THREAD_Y], thread_y(tid, THREADS_X, THREADS_Y)
    )
    np.testing.assert_array_equal(
        got[ROW_THREAD_Z], thread_z(tid, THREADS_X, THREADS_Y)
    )
    np.testing.assert_array_equal(got[ROW_BLOCK_DIM_X], np.full(THREADS, THREADS_X))
    np.testing.assert_array_equal(got[ROW_LINEAR_TID], tid)
    np.testing.assert_array_equal(got[ROW_WARP_SIZE], np.full(THREADS, WARP_SIZE))


def test_the_launch_bracket_pins_the_grid_to_one_block(device):
    """验证 1 的后两个量。**带了 `[None, 1]` 才有 `block_idx == 0`、`grid_dim == 1`。**"""
    got = _run_context(device, one_block=True)
    np.testing.assert_array_equal(
        got[ROW_BLOCK_IDX_X], np.zeros(THREADS, dtype=np.int32)
    )
    np.testing.assert_array_equal(got[ROW_GRID_DIM_X], np.ones(THREADS, dtype=np.int32))


def test_the_linear_index_is_the_three_axes_folded_together(device):
    """验证 2。`linear_tid == x + y * bdim.x + z * bdim.x * bdim.y`。

    **这条不比任何预期值**：它拿实测的三个轴折起来，和实测的线性号比。所以它验的是框架内部
    一致，而不是"我猜的公式对不对"——公式猜错了验证 1 会红，这条不会。
    """
    got = _run_context(device, one_block=True)
    folded = linearised(
        got[ROW_THREAD_X], got[ROW_THREAD_Y], got[ROW_THREAD_Z], THREADS_X, THREADS_Y
    )
    np.testing.assert_array_equal(got[ROW_LINEAR_TID], folded)


def test_without_a_launch_bracket_the_grid_is_the_whole_card(device):
    """验证 3。直接调用 → `grid_dim.x > 1`（这台机器 56）。

    **只断言 `> 1`**：具体的核数是机器相关的（基线数字会漂），而 `block_idx` 更是不可复现——
    56 个 block 写同一片 GM，读回来的是谁最后写赢。这条判据只取和机器无关的那一半事实。
    """
    got = _run_context(device, one_block=False)
    grid = got[ROW_GRID_DIM_X]
    assert (grid == grid[0]).all(), f"同一次启动里 grid_dim 不一致：{grid.tolist()}"
    assert grid[0] > 1, (
        f"grid_dim.x = {grid[0]}，期望 > 1：不带启动括号应该解成当前 stream 的全量核数"
    )
    # block_idx 刻意不断言：它是"哪个 block 最后写赢"，不是一个确定的值。


def test_syncthreads_makes_a_neighbour_read_deterministic(device):
    """验证 4。每个线程读到邻居写的值。

    形状是刻意的：**只读自己写的东西根本用不着同步**，那样的判据去掉 `syncthreads()` 照样绿。
    """
    out = torch.full((1, THREADS), -1, dtype=torch.int32, device=device)
    rotate_through_shared_ub[None, 1](out)
    torch.npu.synchronize()
    np.testing.assert_array_equal(out.cpu().numpy()[0], rotated(THREADS))
