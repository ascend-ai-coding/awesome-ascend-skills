# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""块索引与元素坐标该算出同一块数据。

**为什么可以 `rtol=0, atol=0` 比。** 整例不做任何算术：数据从 GM 搬进 UB 再搬回去，
每个字节都是原样。搬运正确的定义就是逐位相等，容差在这里没有意义。
"""

import numpy as np

SENTINEL = -1.0


def make_input(rows, cols):
    """`arange`：第 i 个元素的值就是 i，所以搬错位置一眼能看出错了多少。"""
    return np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)


def block_written(x, block_row, block_col, rows, cols, sentinel=SENTINEL):
    """只有 `[block_row, block_col]` 号块被写回，其余保持哨兵值。

    哨兵是这一例唯一的判据来源：不填哨兵就只能证明那一块对，证明不了**别的块没被碰**——
    而偏移算错最典型的表现正是写到了邻块。
    """
    out = np.full_like(x, sentinel, dtype=np.float32)
    r, c = block_row * rows, block_col * cols
    out[slice(r, r + rows), slice(c, c + cols)] = x[
        slice(r, r + rows), slice(c, c + cols)
    ]
    return out
