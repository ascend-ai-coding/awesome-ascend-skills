# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""线程上下文的每一个量该是什么，以及它们之间的那条恒等式。

**为什么可以 `rtol=0, atol=0` 比。** 全是整数编号，没有算术。

参考值写成 `tid` 的函数（`tid % TX` 这种）而不是写死一串数：**那个表达式本身就是被验证的规则**，
读参考的人一眼看得出线性化是怎么定义的。
"""

import numpy as np

WARP_SIZE = 32
"""`pl.simt.warp_size()` 的实测值（Ascend950PR，2026-09-28）。

**这是硬件事实，不是约定**，所以写成常量而不是"随便什么值都行"。它变了这一例会红，
那正是我们想收到的通知。
"""


def thread_x(tid, threads_x):
    return tid % threads_x


def thread_y(tid, threads_x, threads_y):
    return (tid // threads_x) % threads_y


def thread_z(tid, threads_x, threads_y):
    return tid // (threads_x * threads_y)


def linearised(tx, ty, tz, threads_x, threads_y):
    """`linear_thread_idx` 的定义：x 最快、z 最慢。

    这个函数就是 `test_the_linear_index_is_the_three_axes_folded_together` 的全部内容——
    它不比 golden，它比**实测的三个轴**折起来等不等于实测的线性号。所以它不依赖任何预期值。
    """
    return tx + ty * threads_x + tz * threads_x * threads_y


def rotated(threads, step=10):
    """`syncthreads` 之后每个线程读到邻居写的值：`result[tid] == ((tid + 1) % N) * step`。"""
    tid = np.arange(threads, dtype=np.int32)
    return (((tid + 1) % threads) * step).astype(np.int32)
