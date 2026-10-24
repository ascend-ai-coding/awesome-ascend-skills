# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。环形计数（`atomic_inc` / `atomic_dec`）与两个内存栅栏。

**环形计数的周期是 `limit + 1`，不是 `limit`**（实测）。计数器在 `0 .. limit` 之间转：

| | 规则 | 第 k 次（0 起）拿到的旧值 |
| --- | --- | --- |
| `atomic_inc(t, limit)` | `old >= limit ? 0 : old + 1` | `k mod (limit + 1)` |
| `atomic_dec(t, limit)` | `old == 0 ? limit : old - 1` | `(-k) mod (limit + 1)` |

所以 32 个线程、`limit=6`（周期 7）之后：`inc` 的最终值是 `32 mod 7 = 4`，`dec` 是 `3`。
**两者的返回值多重集也是确定的**，判据落在那上面——见 `test.py` §三。

**`limit` 的 dtype 必须和目标一致。** 目标是 UINT32 计数器时，直接写 Python 字面量会被拒
（字面量是 INT64）：

```
simt.atomic_inc operand 0 dtype must match target dtype uint32, but got int64
```

所以 `limit` 从一个 UINT32 张量里读，不写成字面量。

**两个栅栏只排序、不等待**（docstring：「does not wait for other threads」）。它们和
`syncthreads` 不是一回事：后者是**屏障**，所有线程都得到。

第一手来源：`pypto/python/tests/st/pypto_pro/frontend/simt/atomic/test_atomic_inc.py`、
`test_atomic_dec.py` 与 `synchronization/` 三个文件。
"""

import pypto_pro.language as pl

U32 = pl.DT_UINT32
I32 = pl.DT_INT32

THREADS = 32

# `limit=6` → 周期 7。**刻意取成和线程数不互相整除**：32 % 7 = 4 ≠ 0，
# 整除的话最终值会回到初值，"转了几圈"就验不出来。
LIMIT = 6

COUNTERS = 2
COUNTER_INC = 0
COUNTER_DEC = 1

RET_ROWS = 2
RET_INC = 0
RET_DEC = 1


@pl.vector_function(mode="simt", max_threads=THREADS)
def ring_counters(
    counter: pl.Tensor[[COUNTERS, 1], U32],
    limits: pl.Tensor[[1, 1], U32],
    ret: pl.Tensor[[RET_ROWS, THREADS], I32],
):
    """两个环形计数器各被 32 个线程打一遍，旧值记下来。

    `limit` 从 `limits` 张量读，**不写字面量**：字面量是 INT64，和 UINT32 的目标不匹配会被拒。
    返回的旧值是 UINT32，存进 INT32 行之前过一次 `cast`。
    """
    tid = pl.simt.linear_thread_idx()
    lim = limits[0, 0]
    ret[RET_INC, tid] = pl.simt.cast(
        pl.simt.atomic_inc(counter[COUNTER_INC, 0], lim), I32
    )
    ret[RET_DEC, tid] = pl.simt.cast(
        pl.simt.atomic_dec(counter[COUNTER_DEC, 0], lim), I32
    )


@pl.jit()
def simt_ring_counters(
    counter: pl.Tensor[[COUNTERS, 1], U32],
    limits: pl.Tensor[[1, 1], U32],
    ret: pl.Tensor[[RET_ROWS, THREADS], I32],
):
    """**必须 `[None, 1]`**：累加类的东西，多 block 会各转一遍。"""
    with pl.section_vector():
        ring_counters[THREADS](counter, limits, ret)


@pl.vector_function(mode="simt", max_threads=THREADS)
def fences_around_a_shared_write(scratch, result):
    """两个栅栏各来一条，中间的计算必须还是对的。

    形状和 ``simt_launch`` 的 `rotate_after_sync` 一样（写自己的槽、读邻居的槽），
    但**把定序交给 `syncthreads`**，栅栏只是额外加的——

    **栅栏本身在这里验不到**：它排序的是"这个线程自己的内存操作对别人的可见顺序"，
    要验得构造一个生产者／消费者、让消费者自旋等标志。那在 SIMT 里有挂死风险，
    `test.py` §五 明说跳过。这个 kernel 能证明的只有"加了栅栏不会把对的弄坏"。
    """
    tid = pl.simt.linear_thread_idx()
    scratch[0, tid] = tid * 10
    pl.simt.threadfence_block()
    pl.simt.syncthreads()
    neighbour = (tid + 1) % THREADS
    value = scratch[0, neighbour]
    pl.simt.threadfence()
    result[0, tid] = value


@pl.jit()
def simt_fences(out: pl.Tensor[[1, THREADS], I32]):
    scratch = pl.make_tile(
        pl.TileType(shape=[1, THREADS], dtype=I32, target_memory=pl.MemorySpace.Vec),
        addr=0x0000,
    )
    result = pl.make_tile(
        pl.TileType(shape=[1, THREADS], dtype=I32, target_memory=pl.MemorySpace.Vec),
        addr=0x0100,
    )
    with pl.section_vector():
        fences_around_a_shared_write[THREADS](scratch, result)
        pl.system.sync_src(
            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0
        )
        pl.store(out, result, [0, 0])


# ---------------------------------------------------------------------------
# 只解析、不上板
# ---------------------------------------------------------------------------


@pl.vector_function(mode="simt", max_threads=THREADS)
def _limit_as_a_literal(
    counter: pl.Tensor[[COUNTERS, 1], U32], ret: pl.Tensor[[RET_ROWS, THREADS], I32]
):
    """`limit` 写成 Python 字面量：字面量是 INT64，和 UINT32 的目标不匹配。"""
    tid = pl.simt.linear_thread_idx()
    ret[RET_INC, tid] = pl.simt.cast(
        pl.simt.atomic_inc(counter[COUNTER_INC, 0], LIMIT), I32
    )


@pl.jit()
def limit_dtype_mismatch(
    counter: pl.Tensor[[COUNTERS, 1], U32], ret: pl.Tensor[[RET_ROWS, THREADS], I32]
):
    with pl.section_vector():
        _limit_as_a_literal[THREADS](counter, ret)


@pl.vector_function(mode="simt", max_threads=THREADS)
def _syncthreads_in_a_branch(out):
    """反例：syncthreads-must-be-uniform —— 这一例的内容就是这条违规本身，只解析不上板

    `syncthreads` 放在运行期 `if` 里：docstring 说不行，解析期不拦。非一致地到达屏障会挂死
    （一部分线程在等永远不会来的另一部分）。**这个 kernel 只解析，绝不上板**——在共享机上试一个
    会挂死的东西不值当。
    """
    tid = pl.simt.linear_thread_idx()
    if tid > 0:
        pl.simt.syncthreads()
    out[0, tid] = tid


@pl.jit()
def syncthreads_inside_a_branch(out: pl.Tensor[[1, THREADS], I32]):
    with pl.section_vector():
        _syncthreads_in_a_branch[THREADS](out)


@pl.vector_function(mode="simt", max_threads=THREADS)
def _fence_in_a_branch(out):
    """栅栏放在运行期 `if` 里——docstring 说**这是允许的**，实测通过。

    和上面那个放在一起，才说明 docstring 区分的是"屏障"和"栅栏"，不是笼统地禁控制流。
    """
    tid = pl.simt.linear_thread_idx()
    if tid > 0:
        pl.simt.threadfence_block()
    out[0, tid] = tid


@pl.jit()
def fence_inside_a_branch(out: pl.Tensor[[1, THREADS], I32]):
    with pl.section_vector():
        _fence_in_a_branch[THREADS](out)
