# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""填充区该长什么样。

**为什么可以 `rtol=0, atol=0` 比。** 整例不做算术：有效区是原样搬运，填充区是一个常量。
用 INT32 就更彻底——整型没有舍入这回事。

`TilePad` 的三个值就是字面意思，实测与声明一致（`zero`→`0`、`max`→`2147483647`、
`min`→`-2147483648`），冷进程、热进程、任意调用顺序都一致。

**这张表曾经被我写成「错位一位」并当成框架缺陷报出去。** 根因是我自己的 kernel 少了
MTE2→V→MTE3 的同步，填充和搬运在赛跑，读到的是别的样例留在 UB 里的残留。
经过见 `test.py` §七 与 `references/verification/evidence.md` #16。
"""

import numpy as np

PAD_VALUE = {
    "zero": np.int32(0),
    "max": np.int32(2147483647),
    "min": np.int32(-2147483648),
}
"""`null` 不在表里：它配 `fillpad` 会被解析期拒，见 `test.py` 最后一条。"""


def make_input(rows, cols):
    """1 起的连续整数。**不含 0**，这样「填充区是 0」不会和「有效区的数据」撞上。"""
    return (np.arange(rows * cols, dtype=np.int32) + 1).reshape(rows, cols)


def padded(x, valid_rows, valid_cols, pad_value):
    """有效区保留 `x`，其余填 `pad_value`。用于 `NORMAL` 与 `INPLACE`。"""
    out = np.full_like(x, pad_value, dtype=np.int32)
    out[:valid_rows, :valid_cols] = x[:valid_rows, :valid_cols]
    return out


def expanded(x, valid_rows, valid_cols, rows, cols, pad_value=np.int32(0)):
    """`EXPAND`：目的比源宽，有效区之外（含多出来的那些列）全是填充值。

    注意有效区的**取法和 `NORMAL` 完全一样**——`EXPAND` 改的是目的的形状，不是收窄的方式。
    所以这个函数和 `padded` 唯一的差别就是输出形状。
    """
    out = np.full((rows, cols), pad_value, dtype=np.int32)
    out[:valid_rows, :valid_cols] = x[:valid_rows, :valid_cols]
    return out
