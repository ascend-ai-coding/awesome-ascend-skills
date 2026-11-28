# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""三层形状各自该产生什么结果。

**为什么可以 `rtol=0, atol=0` 比。** 切片和重解释只改变搬运范围或查看方式；FIX 消费端探针使用小整数矩阵乘，FP32 累加也能精确表示。

哨兵是这一例的主要判据手段：「少搬了多少」只能靠"还剩几个哨兵"来数。
"""

import numpy as np

SENTINEL = np.float32(-999.0)


def make_input(rows, cols):
    """`arange + 1`：元素值就是下标 +1。不含 0，免得"没被写"和"写了个 0"混在一起。"""
    return (np.arange(rows * cols, dtype=np.float32) + 1.0).reshape(rows, cols)


def valid_region_only(x, valid_rows, valid_cols, sentinel=SENTINEL):
    """只有左上角 `valid_rows × valid_cols` 被写，其余保持哨兵。

    收窄两个轴（而不是只收一个）是刻意的：只收一个轴的话，行列写反也会得到同样的期望值。
    """
    out = np.full_like(x, sentinel, dtype=np.float32)
    out[:valid_rows, :valid_cols] = x[:valid_rows, :valid_cols]
    return out


def reshaped(x, rows, cols):
    """同一批字节换个形状看。行优先下 `reshape` 就是这件事的定义。"""
    return x.reshape(rows, cols)


def vf_view_window(x):
    """声明列数为 16 时，VF 的 [2, 0] 连续加载从元素 32 开始。"""
    return x.reshape(-1)[32:96].reshape(1, 64)


def accumulated_product(a, b):
    """两次同一矩阵乘累加；FP64 独立参考不复制 tile 布局或同步。"""
    return 2 * (a.astype(np.float64) @ b.astype(np.float64))
