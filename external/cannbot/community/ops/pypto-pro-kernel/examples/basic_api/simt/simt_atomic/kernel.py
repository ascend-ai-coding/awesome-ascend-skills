# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。INT32 九个原子族：七个顺序无关，两个不是。

**所有线程打同一个地址**，这是原子操作唯一值得验的形状——各打各的地址的话，普通写也"对"。

| 算子 | 最终值顺序无关吗 | 为什么 |
| --- | --- | --- |
| `atomic_add` / `atomic_sub` | **是** | 加法可交换 |
| `atomic_max` / `atomic_min` | **是** | 取极值可交换 |
| `atomic_and` / `atomic_or` / `atomic_xor` | **是** | 位运算可交换 |
| `atomic_exch` | **否** | 最终值是最后一个换进去的 |
| `atomic_cas` | **否** | 谁先看到 compare 值谁赢 |

后两个的判据只能断言**与顺序无关的性质**（最终值落在候选集合里、恰好一个赢家），
不能断言具体是谁——那是时序。

**`atomic_add` 的返回值是"加之前的旧值"**，这一例用它证明原子性：N 个线程各加 1，
拿回来的 N 个旧值必须**恰好是 `0..N-1` 的一个排列**。少一个就是丢了更新，重一个就是两个线程
读到了同一个旧值。**这比看最终值有力**——最终值对了也可能是别的原因。

**操作数的 dtype 必须和目标完全一致。** `tid + 1` 和 Python 字面量都是 INT64，目标是 INT32 时
解析期直接拒：

```
simt.atomic_add operand 0 dtype must match target dtype int32, but got int64
```

所以这一例里每个标量操作数都过一次 `pl.simt.cast`。从 INT32 张量读出来的值不用再 cast。

第一手来源：`pypto/python/tests/st/pypto_pro/frontend/simt/atomic/` 的 11 个文件。
"""

import pypto_pro.language as pl

I32 = pl.DT_INT32

THREADS = 32

# 一个槽一个算子。**十个槽，九个算子**：`add` 占两个（一个验最终值、一个验返回值）。
SLOTS = 10
(
    SLOT_ADD,
    SLOT_SUB,
    SLOT_MAX,
    SLOT_MIN,
    SLOT_AND,
    SLOT_OR,
    SLOT_XOR,
    SLOT_EXCH,
    SLOT_CAS,
    SLOT_ADD_RET,
) = range(SLOTS)

CAS_COMPARE = 0
"""`atomic_cas` 的比较值。初值也是它，所以**恰好一个线程**能看到它并赢下这次交换。"""


@pl.vector_function(mode="simt", max_threads=THREADS)
def nine_atomic_families(
    state: pl.Tensor[[SLOTS, 1], I32],
    vals: pl.Tensor[[2, THREADS], I32],
    returns: pl.Tensor[[1, THREADS], I32],
):
    """九个原子族各打一个槽。`vals` 的两行分别是"乱序的数值"和"位掩码"。

    数值刻意**不用 `tid + 1`** 给 `max` / `min`：单调序列下"取最大"和"取最后一个"分不开。
    """
    tid = pl.simt.linear_thread_idx()
    value = vals[0, tid]
    mask = vals[1, tid]

    # **每个操作数都要显式 cast 到目标的 dtype。** `tid + 1` 和 Python 字面量都是 INT64，
    # 而目标是 INT32：不 cast 的话解析期就被拒（消息见 `test.py` §七）。
    # `value` / `mask` 从 INT32 张量读出来，已经是 INT32，不用再 cast。
    contribution = pl.simt.cast(tid + 1, I32)
    one = pl.simt.cast(1, I32)
    compare = pl.simt.cast(CAS_COMPARE, I32)

    pl.simt.atomic_add(state[SLOT_ADD, 0], contribution)
    pl.simt.atomic_sub(state[SLOT_SUB, 0], contribution)
    pl.simt.atomic_max(state[SLOT_MAX, 0], value)
    pl.simt.atomic_min(state[SLOT_MIN, 0], value)
    pl.simt.atomic_and(state[SLOT_AND, 0], mask)
    pl.simt.atomic_or(state[SLOT_OR, 0], mask)
    pl.simt.atomic_xor(state[SLOT_XOR, 0], mask)
    pl.simt.atomic_exch(state[SLOT_EXCH, 0], contribution)
    pl.simt.atomic_cas(state[SLOT_CAS, 0], compare, contribution)

    # 每个线程加 1，把"加之前的旧值"记下来——N 个旧值必须是 0..N-1 的一个排列。
    returns[0, tid] = pl.simt.atomic_add(state[SLOT_ADD_RET, 0], one)


@pl.jit()
def simt_nine_atomic_families(
    state: pl.Tensor[[SLOTS, 1], I32],
    vals: pl.Tensor[[2, THREADS], I32],
    returns: pl.Tensor[[1, THREADS], I32],
):
    """**必须 `[None, 1]` 启动。**

    不带启动括号的话 56 个 block 各跑一遍，每个槽被累加 56 倍——`repeated-mistakes.md` #9
    记的正是这个坑（当时 `fixpipe_riders` 的 atomic 测试拿到了期望值的 28 倍，
    而我第一反应是去查 atomic 的语义）。**累加类的东西，启动方式是语义的一部分。**
    """
    with pl.section_vector():
        nine_atomic_families[THREADS](state, vals, returns)


@pl.vector_function(mode="simt", max_threads=THREADS)
def _int64_operand(state: pl.Tensor[[SLOTS, 1], I32]):
    """操作数不 cast：`tid + 1` 是 INT64，目标是 INT32。**解析期拒。**"""
    tid = pl.simt.linear_thread_idx()
    pl.simt.atomic_add(state[SLOT_ADD, 0], tid + 1)


@pl.jit()
def operand_dtype_mismatch(state: pl.Tensor[[SLOTS, 1], I32]):
    with pl.section_vector():
        _int64_operand[THREADS](state)
