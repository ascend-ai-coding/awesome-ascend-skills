# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。核／子核的编号与总数：四个 `pl.get_*` 与它们和 SIMT 那套的关系。

```python
idx  = pl.get_block_idx()       # 当前 AI 核（block）的号
num  = pl.get_block_num()       # 实际启动了几个 block
sidx = pl.get_subblock_idx()    # 子核号，0 或 1
snum = pl.get_subblock_num()    # 每个 AI 核几个子核
```

四个都是 `@pl.jit` 体里的标量 API，**进不了 SIMT 函数**（实测解析期拒）。

## `subblock_num` 取决于 kernel 有没有 cube 段（实测）

| kernel 形状 | `subblock_num` | `subblock_idx` |
| --- | ---: | --- |
| 只有 `section_vector` | **1** | 恒 0 |
| 只有 `section_cube` | **1** | 恒 0 |
| **两段都有** | **2** | 按 block 交替 **0 / 1** |

docstring 说「Returns 1 on AIC binaries, `get_subblockdim()` on AIV binaries」——实测印证了
它，但**读原话读不出「加一段 cube 就会变」**。任何用 `subblock_idx` 分工的代码,
在 kernel 多出一个 cube 段的那天行为就变了。

## 当前样例的 GM `pl.setval` 写回诊断

本例通过 GM 标量 `pl.setval` 写回核编号，并对比不同 block 间距；STRIDE 是诊断变量，
不是所有 GM 输出的布局要求。不要把这里的 64 字节间距迁移到 `pl.store` 的 DMA 输出，
也不要据此修改调用方的紧密张量或守卫布局。标量/DMA 对照、工具链边界及已知限制统一见
[GM 标量小写入](../../../../references/api/support.md#7-small-gm-scalar-writes-from-multiple-blocks)。
`test.py` 只维护本例的验证矩阵；精确硬件机制尚未证明。

第一手来源：`fa`（76 次）、`matmul`（43 次）、`datacopy`（30 次）三组 ST 里这四个 API 的用法;
`_api.py` 的 docstring（`get_block_num` 那条讲 stream 核数限制的话）。
"""

import pypto_pro.language as pl

I32 = pl.DT_INT32
FP32 = pl.DT_FP32

N = 32
_UB = 0x0
_L1 = 0x0
_SIMT = 0x200

# 每个 block 占多少个 INT32：16 是本诊断已测稳定的标量写间距。
# 与 4 的对照不构成 DMA 布局要求，也不证明所有可能跨距的精确下限。
STRIDE = 16
TIGHT_STRIDE = 4
SLOTS = 64
"""留 64 个槽：这台机器 56 核，不带启动括号时 `block_num` 就是 56，槽要够。"""

ROW_BLOCK_IDX = 0
ROW_BLOCK_NUM = 1
ROW_SUBBLOCK_IDX = 2
ROW_SUBBLOCK_NUM = 3
QUANTITIES = 4

SIMT_THREADS = 8
SIMT_ROW_BLOCK_IDX = 0
SIMT_ROW_GRID_DIM = 1


def _ub(dtype=FP32, n=N):
    return pl.TileType(shape=[1, n], dtype=dtype, target_memory=pl.MemorySpace.Vec)


def _l1(dtype=FP32, n=N):
    return pl.TileType(shape=[1, n], dtype=dtype, target_memory=pl.MemorySpace.Mat)


# ---------------------------------------------------------------------------
# 一、每个 block 报自己的四个量
# ---------------------------------------------------------------------------


def make_vector_reporter(stride):
    """只有 vector 段。`stride` 是每个 block 占的 INT32 个数——**这一例唯一的自变量**。

    工厂而不是两个手写 kernel：紧凑和宽两种跨距必须是同一份代码，否则「变的是跨距」
    这句话就不成立了。
    """

    @pl.jit()
    def report(out: pl.Tensor[[SLOTS * stride], I32]):
        idx = pl.get_block_idx()
        num = pl.get_block_num()
        sidx = pl.get_subblock_idx()
        snum = pl.get_subblock_num()
        with pl.section_vector():
            pl.setval(out, idx * stride + ROW_BLOCK_IDX, idx)
            pl.setval(out, idx * stride + ROW_BLOCK_NUM, num)
            pl.setval(out, idx * stride + ROW_SUBBLOCK_IDX, sidx)
            pl.setval(out, idx * stride + ROW_SUBBLOCK_NUM, snum)

    return report


@pl.jit()
def cube_only_reporter(out: pl.Tensor[[SLOTS * STRIDE], I32]):
    """只有 cube 段。四个量在 kernel 体里读，写出去放在 cube 段里。

    和 `make_vector_reporter(STRIDE)` 对照：两个单段 kernel 的 `subblock_num` 都是 1。
    """
    idx = pl.get_block_idx()
    num = pl.get_block_num()
    sidx = pl.get_subblock_idx()
    snum = pl.get_subblock_num()
    with pl.section_cube():
        pl.setval(out, idx * STRIDE + ROW_BLOCK_IDX, idx)
        pl.setval(out, idx * STRIDE + ROW_BLOCK_NUM, num)
        pl.setval(out, idx * STRIDE + ROW_SUBBLOCK_IDX, sidx)
        pl.setval(out, idx * STRIDE + ROW_SUBBLOCK_NUM, snum)


@pl.jit(auto_mutex=True)
def mixed_reporter(
    a: pl.Tensor[[1, N], FP32],
    out: pl.Tensor[[1, N], FP32],
    ids: pl.Tensor[[SLOTS * STRIDE], I32],
):
    """**两段都有**：cube 段搬一趟 L1，vector 段搬一趟 UB 并报编号。

    cube 段那一趟搬运的结果没用上——**它在这里的作用是让这个 kernel 成为混合 kernel**,
    而那正是 `subblock_num` 变成 2 的条件。`out` 给的是 vector 段那一趟,
    判据要靠它确认这个 kernel 真在干活（不然「编号是 2」可能只是编译期的巧合）。
    """
    ub = pl.make_tile_group(type=_ub(), addrs=_UB, mutex_ids=[0])
    l1 = pl.make_tile_group(type=_l1(), addrs=_L1, mutex_ids=[1])
    idx = pl.get_block_idx()
    num = pl.get_block_num()
    sidx = pl.get_subblock_idx()
    snum = pl.get_subblock_num()
    with pl.section_cube():
        pl.load(l1.current(), a, [0, 0])
    with pl.section_vector():
        t = ub.current()
        pl.load(t, a, [0, 0])
        pl.store(out, t, [0, 0])
        pl.setval(ids, idx * STRIDE + ROW_BLOCK_IDX, idx)
        pl.setval(ids, idx * STRIDE + ROW_BLOCK_NUM, num)
        pl.setval(ids, idx * STRIDE + ROW_SUBBLOCK_IDX, sidx)
        pl.setval(ids, idx * STRIDE + ROW_SUBBLOCK_NUM, snum)


# ---------------------------------------------------------------------------
# 二、和 SIMT 那套编号的关系
# ---------------------------------------------------------------------------


@pl.vector_function(mode="simt", max_threads=SIMT_THREADS)
def _read_simt_numbers(dst):
    """SIMT 侧的块号与网格大小。每个线程写自己那一列。"""
    tid = pl.simt.linear_thread_idx()
    bidx = pl.simt.block_idx()
    gdim = pl.simt.grid_dim()
    dst[SIMT_ROW_BLOCK_IDX, tid] = bidx.x
    dst[SIMT_ROW_GRID_DIM, tid] = gdim.x


@pl.jit()
def both_numbering_schemes(
    simt_out: pl.Tensor[[2, SIMT_THREADS], I32],
    kernel_out: pl.Tensor[[SLOTS * STRIDE], I32],
):
    """同一次启动里把两套编号都读出来。

    kernel 侧的两个量按 block 分段写（跨距 `STRIDE`），SIMT 侧的走 tile → `pl.store`。
    **`simt_out` 只在单 block 时可判**：多 block 都 store 到 `[0, 0]`，谁最后写赢。
    """
    idx = pl.get_block_idx()
    num = pl.get_block_num()
    dst = pl.make_tile(
        pl.TileType(
            shape=[2, SIMT_THREADS], dtype=I32, target_memory=pl.MemorySpace.Vec
        ),
        addr=_SIMT,
    )
    with pl.section_vector():
        _read_simt_numbers[SIMT_THREADS](dst)
        pl.system.sync_src(
            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0
        )
        pl.store(simt_out, dst, [0, 0])
        pl.setval(kernel_out, idx * STRIDE + ROW_BLOCK_IDX, idx)
        pl.setval(kernel_out, idx * STRIDE + ROW_BLOCK_NUM, num)


# ---------------------------------------------------------------------------
# 三、dcci 救不救得回紧凑跨距
# ---------------------------------------------------------------------------


def make_tight_reporter_with_dcci():
    """紧凑跨距 + 每个 block 写完清自己那段的 GM cache。

    **实测救不回来**，所以这个 kernel 的判据只能是「没有垃圾值」。收它是为了钉住
    「dcci 不是这个问题的解」——那是很自然的第一反应。
    """

    @pl.jit()
    def report(out: pl.Tensor[[SLOTS * TIGHT_STRIDE], I32]):
        idx = pl.get_block_idx()
        num = pl.get_block_num()
        with pl.section_vector():
            pl.setval(out, idx * TIGHT_STRIDE + ROW_BLOCK_IDX, idx)
            pl.setval(out, idx * TIGHT_STRIDE + ROW_BLOCK_NUM, num)
            pl.setval(out, idx * TIGHT_STRIDE + ROW_SUBBLOCK_IDX, idx * 10)
            pl.setval(out, idx * TIGHT_STRIDE + ROW_SUBBLOCK_NUM, idx * 100)
            pl.system.dcci(out, idx * TIGHT_STRIDE)

    return report


# ---------------------------------------------------------------------------
# 四、被拒的写法
# ---------------------------------------------------------------------------


@pl.vector_function(mode="simt", max_threads=SIMT_THREADS)
def _kernel_api_in_simt(dst):
    """反例：kernel 体的 `pl.get_block_idx()` 放进 SIMT 体。SIMT 侧要用 `pl.simt.block_idx()`。"""
    tid = pl.simt.linear_thread_idx()
    dst[0, tid] = pl.get_block_idx()


@pl.jit()
def kernel_api_inside_a_simt_function(out: pl.Tensor[[2, SIMT_THREADS], I32]):
    """反例：两套编号的 API 不能互换位置，即使它们报的是同一个数。"""
    dst = pl.make_tile(
        pl.TileType(
            shape=[2, SIMT_THREADS], dtype=I32, target_memory=pl.MemorySpace.Vec
        ),
        addr=_SIMT,
    )
    with pl.section_vector():
        _kernel_api_in_simt[SIMT_THREADS](dst)
        pl.system.sync_src(
            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0
        )
        pl.store(out, dst, [0, 0])
