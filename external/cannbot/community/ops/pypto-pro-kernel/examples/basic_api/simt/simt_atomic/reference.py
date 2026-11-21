# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""九个原子族的初值与期望值。

**为什么可以 `rtol=0, atol=0`。** 全是整数。

**每个槽的初值是算子的单位元**（或一个能让判据有意义的值），写在 `initial_state()` 里——
初值挑错会让判据失去意义：比如 `and` 的初值不取全 1，最终值就带着初值的零位，
看不出是哪些线程贡献的。
"""

import numpy as np

THREADS = 32
CAS_COMPARE = 0

# 乱序的数值，给 max / min 用。`(tid * 7) % 32 + 1` 是 32 的一个置换（7 与 32 互质），
# **所以它覆盖 1..32 但顺序不单调**——单调的话"取最大"和"取最后一个"分不开。
_PERM = (np.arange(THREADS, dtype=np.int64) * 7) % THREADS + 1

# 位掩码：低 5 位轮转，外加一个所有线程都有的公共位 0x100。
# **公共位不可省**：没有它 `and` 的结果是 0，而 0 可能来自任何地方。
_MASKS = (1 << (np.arange(THREADS, dtype=np.int64) % 5)) | 0x100

INT32_LARGE = 1 << 30
"""`min` 的初值。取一个比所有候选都大的数，否则最终值可能就是初值本身。"""


def make_values():
    """两行：乱序数值、位掩码。"""
    return np.stack([_PERM, _MASKS]).astype(np.int32)


def initial_state(slots):
    """每个槽的初值。**这张表和 `kernel.py` 的 SLOT_* 一一对应。**"""
    state = np.zeros((slots, 1), dtype=np.int32)
    state[4, 0] = -1  # SLOT_AND：全 1（INT32 的 -1）
    state[3, 0] = INT32_LARGE  # SLOT_MIN
    return state


def expected_order_independent(threads=THREADS):
    """七个顺序无关的算子的最终值。"""
    contributions = np.arange(1, threads + 1, dtype=np.int64)
    masks = _MASKS
    return {
        "add": int(contributions.sum()),
        "sub": int(-contributions.sum()),
        "max": int(_PERM.max()),
        "min": int(_PERM.min()),
        "and": int(np.bitwise_and.reduce(masks)),
        "or": int(np.bitwise_or.reduce(masks)),
        "xor": int(np.bitwise_xor.reduce(masks)),
    }


def exch_candidates(threads=THREADS):
    """`atomic_exch` 的最终值必须是其中之一。**初值 0 不在里面**：每个线程都换进了自己的值,
    所以最后一定不是 0。
    """
    return set(range(1, threads + 1))


def cas_candidates(threads=THREADS):
    """`atomic_cas(compare=0, value=tid+1)`，初值 0：**恰好一个线程**看到 0 并换成自己的值。

    所以最终值是 `1..N` 里的某一个，而不是 0。
    """
    return set(range(1, threads + 1))


def add_return_permutation(threads=THREADS):
    """N 个线程各加 1 时，返回的旧值排序后必须正好是 `0..N-1`。

    **这是原子性的判据**：少一个说明丢了更新，重一个说明两个线程读到了同一个旧值。
    """
    return np.arange(threads, dtype=np.int32)
