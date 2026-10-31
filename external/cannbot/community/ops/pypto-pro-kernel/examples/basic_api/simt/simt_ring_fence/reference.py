# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""环形计数的周期、最终值和返回值多重集。

**为什么可以 `rtol=0, atol=0`。** 全是整数计数。

**周期是 `limit + 1`，这是实测结论**：计数器在 `0 .. limit` 之间转（含两端）。用 `limit=6`
和 32 个线程能把两种候选规则分开——周期 7 给最终值 4，周期 6 给 2，实测是 4。
"""

import numpy as np

THREADS = 32
LIMIT = 6


def period(limit=LIMIT):
    """**`limit + 1`**，不是 `limit`。这一行是这一例最容易记错的地方。"""
    return limit + 1


def inc_returns(threads=THREADS, limit=LIMIT):
    """第 k 次 `atomic_inc`（k 从 0 起）拿到的旧值是 `k mod P`。

    返回的是**排序后的多重集**：线程完成的顺序不确定，但"哪些旧值各出现几次"是确定的。
    """
    p = period(limit)
    return np.sort(np.array([k % p for k in range(threads)], dtype=np.int32))


def dec_returns(threads=THREADS, limit=LIMIT):
    """第 k 次 `atomic_dec` 拿到的旧值是 `(-k) mod P`：从 0 开始，下一个是 `limit`。"""
    p = period(limit)
    return np.sort(np.array([(-k) % p for k in range(threads)], dtype=np.int32))


def inc_final(threads=THREADS, limit=LIMIT):
    return threads % period(limit)


def dec_final(threads=THREADS, limit=LIMIT):
    p = period(limit)
    return (-threads) % p


def rotated(threads=THREADS, step=10):
    """栅栏那一例的期望：和 ``simt_launch`` 的 `rotate` 一样。"""
    tid = np.arange(threads, dtype=np.int32)
    return (((tid + 1) % threads) * step).astype(np.int32)
