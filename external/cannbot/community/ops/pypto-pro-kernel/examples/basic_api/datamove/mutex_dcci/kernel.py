# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。手工上锁（`mutex_lock` / `mutex_unlock`）与数据 cache 清理（`dcci`）。

**`mutex_lock` / `mutex_unlock` 是 `auto_mutex=True` 帮你插的那两条 op 的手写形式。**
`auto_mutex` + `make_tile_group(mutex_ids=...)` 在解析后生成的正是
`system.mutex_lock_dyn` / `system.mutex_unlock_dyn`；这一例把同样的 op 自己写出来。

| | 池子 | 定义处 |
| --- | --- | --- |
| flag 的 `event_id`（`sync_src` / `sync_dst`） | `[0, 7]` | `MAX_FLAG_EVENT_ID` |
| 跨核的 `event_id`（`set_cross_core`） | `[0, 15]` | `MAX_EVENT_ID` |
| mutex 的 `mutex_id` | `[0, 31]` | `MAX_MUTEX_ID` |

三个池子互不相干（`pypto/python/pypto_pro/ir/op/system_ops.py:38-41`，注释里写明这是
「the single definition of each limit」）。

`dcci` 是 Data Cache Clean and Invalid，目标可以是 GM 张量或 UB tile。
**这一例给不出 `dcci` 的数值判据**，理由见 `test.py` §五——不是懒，是单核单进程里它的效果
不可观测，硬凑一个判据只会得到一条"以错误的理由通过"的测试。

第一手来源：`pypto/python/tests/st/pypto_pro/frontend/system/`（`system`:29 处 mutex 用法）
与 `sanitizer/`（6 处）。
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

FP32 = pl.DT_FP32

ROWS = 8
COLS = 16

VA_SRC = 0x0000  # 8 × 16 FP32 = 0x200 字节
VA_DST = 0x0200

MUTEX_CHAIN = 0
"""整条链共用一个 `mutex_id`，于是三段严格串行。

真实流水会给每块 buffer 一个自己的号（`make_tile_group(mutex_ids=[0, 1])` 就是这么做的），
让不同 buffer 上的段能重叠。这一例要的是**最容易核对的形状**，不是最快的形状。
"""


@pl.vector_function
def _double(dst, src):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    for offset in pl.range(0, ROWS * COLS, 64):
        value = vf.load_align(src, offset)
        doubled = vf.add(value, value, mask)
        vf.store_align(dst, doubled, mask, offset)


def _tile(addr):
    return pl.make_tile(
        pl.TileType(shape=[ROWS, COLS], dtype=FP32, target_memory=pl.MemorySpace.Vec),
        addr=addr,
    )


@pl.jit()
def manual_mutex_chain(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[ROWS, COLS], FP32]
):
    """手工上锁版的 MTE2 → V → MTE3。输出 `x * 2`。

    `mutex_lock(pipe=P, mutex_id=n)` 让流水 `P` 的指令队列停下来，直到上一个持有 `n` 的人
    `mutex_unlock`。**方向不写在参数里**，是靠「同一个 `mutex_id` 被谁先持有」表达的——
    这和 flag 用 `set_pipe` / `wait_pipe` 把方向写出来不一样，也是它更容易写错的地方。

    `pipe` 是**要上锁的那条流水**，不是要等的那条：`pl.load` 在 MTE2，所以第一段锁 MTE2。
    一锁一放必须同号，三段共用 `MUTEX_CHAIN` 于是严格串行。
    """
    src = _tile(VA_SRC)
    dst = _tile(VA_DST)
    with pl.section_vector():
        pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=MUTEX_CHAIN)
        pl.load(src, x, [0, 0])
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=MUTEX_CHAIN)

        pl.system.mutex_lock(pipe=pl.PipeType.V, mutex_id=MUTEX_CHAIN)
        _double(dst, src)
        pl.system.mutex_unlock(pipe=pl.PipeType.V, mutex_id=MUTEX_CHAIN)

        pl.system.mutex_lock(pipe=pl.PipeType.MTE3, mutex_id=MUTEX_CHAIN)
        pl.store(out, dst, [0, 0])
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE3, mutex_id=MUTEX_CHAIN)


# ---------------------------------------------------------------------------
# dcci：四种目标 / 参数形态
#
# 每个都在一条已经正确同步的链上多加一条 dcci，判据是「结果不变」。
# **这不验证 dcci 做了什么**，只验证它不破坏正确的结果，见 test.py §五。
# ---------------------------------------------------------------------------


@pl.jit()
def dcci_on_tile(x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[ROWS, COLS], FP32]):
    """`dcci(tile)`：目标是 UB tile，`DcciDst.AUTO` 会把它映射到 `CACHELINE_UB`。"""
    src = _tile(VA_SRC)
    dst = _tile(VA_DST)
    with pl.section_vector():
        pl.load(src, x, [0, 0])
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
        )
        _double(dst, src)
        pl.system.dcci(dst)
        pl.system.sync_src(
            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
        )
        pl.store(out, dst, [0, 0])


@pl.jit()
def dcci_on_tensor(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[ROWS, COLS], FP32]
):
    """`dcci(tensor, [r, c])`：目标是 GM 张量，偏移是**逐维**的；`AUTO` 映射到 `CACHELINE_OUT`。

    张量目标的 `offset` 可以是逐维列表，也可以是一个标量元素偏移；tile 目标只接受标量。
    这条 dcci 放在 `pl.store` **之后**——清的是刚写出去的那块 GM。
    """
    src = _tile(VA_SRC)
    dst = _tile(VA_DST)
    with pl.section_vector():
        pl.load(src, x, [0, 0])
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
        )
        _double(dst, src)
        pl.system.sync_src(
            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
        )
        pl.store(out, dst, [0, 0])
        pl.system.dcci(out, [0, 0])


@pl.jit()
def dcci_single_cache_line(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[ROWS, COLS], FP32]
):
    """`cache_line=SINGLE_CACHE_LINE`：只清一条 cache line，不是整个 data cache。

    缺省是 `ENTIRE_DATA_CACHE`。**两个值都只是解析期通过**，代价差别在这一例里量不出来。
    """
    src = _tile(VA_SRC)
    dst = _tile(VA_DST)
    with pl.section_vector():
        pl.load(src, x, [0, 0])
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
        )
        _double(dst, src)
        pl.system.dcci(dst, cache_line=pl.CacheLine.SINGLE_CACHE_LINE)
        pl.system.sync_src(
            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
        )
        pl.store(out, dst, [0, 0])


@pl.jit()
def dcci_explicit_dst(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[ROWS, COLS], FP32]
):
    """`dst=DcciDst.CACHELINE_UB`：把 `AUTO` 对 tile 的推断显式写出来。

    行为上和 `dcci_on_tile` 等价（`AUTO` 对 tile 就是映射到 `CACHELINE_UB`），留着它是为了
    **把 `dst=` 这个参数也覆盖掉**——一个参数从来没被任何样例传过，就等于没有被验证过。
    """
    src = _tile(VA_SRC)
    dst = _tile(VA_DST)
    with pl.section_vector():
        pl.load(src, x, [0, 0])
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
        )
        _double(dst, src)
        pl.system.dcci(dst, dst=pl.DcciDst.CACHELINE_UB)
        pl.system.sync_src(
            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
        )
        pl.store(out, dst, [0, 0])


# ---------------------------------------------------------------------------
# 只解析、不上板
# ---------------------------------------------------------------------------


@pl.jit()
def mutex_id_out_of_range(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[ROWS, COLS], FP32]
):
    """`mutex_id=32`。池子是 `[0, 31]`，和 flag 的 `[0, 7]` 是两回事。"""
    src = _tile(VA_SRC)
    dst = _tile(VA_DST)
    with pl.section_vector():
        pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=32)
        pl.load(src, x, [0, 0])
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=32)
        _double(dst, src)
        pl.store(out, dst, [0, 0])


@pl.jit()
def lock_without_unlock(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[ROWS, COLS], FP32]
):
    """**`mutex_lock` 没有配对的 `mutex_unlock`，解析期照样通过。** `mutex_lock` 的 docstring 写着
    「Must be paired with `mutex_unlock`」，但没有任何一层检查它。

    **这个 kernel 不上板**，理由见 `test.py` §六：一个没释放的 buffer-id token 可能把流水
    队列留在阻塞态，在共享机上试这个不值当。只钉住「解析期不拦」这件事。
    """
    src = _tile(VA_SRC)
    dst = _tile(VA_DST)
    with pl.section_vector():
        pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=MUTEX_CHAIN)
        pl.load(src, x, [0, 0])
        _double(dst, src)
        pl.store(out, dst, [0, 0])
