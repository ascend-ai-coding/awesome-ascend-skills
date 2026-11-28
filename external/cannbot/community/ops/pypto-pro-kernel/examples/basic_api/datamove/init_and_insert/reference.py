# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""初始化后的 GM 该长什么样，插入后的大块该长什么样。

**为什么可以 `rtol=0, atol=0` 比。** 两个 API 都不做算术：`init_output` 写一个常量，
`insert` 搬字节。填充值取 `7.5` 和 `-1.0`——**FP32 里都是精确值**（2 的幂的有限和），
所以"填对了"的定义就是逐位相等。

主机侧的哨兵是 `SENTINEL`。**哨兵是这一例唯一能判"不该动的没动"的手段**：不填哨兵就只能证明
该填的那段对了，证不了别的地方没被顺手写掉——而偏移算错最典型的表现正是多写或少写一截。
"""

import numpy as np

SENTINEL = np.float32(-999.0)
"""主机侧预填的值。取一个既不可能是输入、也不可能是任何填充值的数。"""


def make_input(rows, cols):
    """`arange + 1`：第 i 个元素的值是 i+1，搬错位置一眼能看出错了多少。不含 0。"""
    return (np.arange(rows * cols, dtype=np.float32) + 1.0).reshape(rows, cols)


def filled_whole(rows, cols, value):
    return np.full((rows, cols), value, dtype=np.float32)


def filled_region(rows, cols, offset, size, value, sentinel=SENTINEL):
    """`[offset, offset+size)` 填 `value`，其余保持哨兵。**区间按行优先拉平后算。**

    这个函数本身就是「offset / size 是元素数」这条判据的形式化——它用 `ravel()` 而不是按行切，
    所以区间跨行时也对。跨行是刻意的：落在整行边界上的区间，「按元素」和「按行」两种理解
    都会给出同样的期望值，判据就分不出来。
    """
    out = np.full(rows * cols, sentinel, dtype=np.float32)
    out[slice(offset, offset + size)] = value
    return out.reshape(rows, cols)


def inserted(
    x, small_src_row, small_src_col, small_rows, small_cols, insert_row, insert_col
):
    """大块是 `x`，其中 `[insert_row, insert_col]` 起的一块被 `x` 右下角那块覆盖。

    源窗口和目的窗口都不在原点：两处都在原点的话，行列写反也会给出正确结果。
    """
    out = x.copy()
    patch = x[
        slice(small_src_row, small_src_row + small_rows),
        slice(small_src_col, small_src_col + small_cols),
    ]
    out[
        slice(insert_row, insert_row + small_rows),
        slice(insert_col, insert_col + small_cols),
    ] = patch
    return out
