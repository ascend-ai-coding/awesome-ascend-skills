# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""`x + 1.0`，以及尾块该长什么样。

**为什么可以 `rtol=0, atol=0` 比。** 输入取 `arange(0, 128)`，都是 FP32 精确可表示的整数；
加 1.0 之后还是精确可表示的整数（远小于 2^24）。所以参考在 float64 上算、再降到 float32，和
设备上的 FP32 加法**逐位相同**。这不是「误差小」，是没有误差。
"""

import numpy as np


def make_input(n, seed=0):
    """`arange`，不是随机数：值精确可表示，而且一眼能看出第 i 个元素是谁。"""
    del seed  # 保持和别的样例同签名，但这里刻意不用随机
    return np.arange(n, dtype=np.float32).reshape(1, n)


SENTINEL = -10.0


def add_one(x):
    """整块都加一。"""
    return (x.astype(np.float64) + 1.0).astype(np.float32)


def add_one_prefix(x, count, sentinel=SENTINEL):
    """只前 `count` 个加一，其余保持哨兵值。

    这是 `vf.update_mask` 该产生的结果：掩码之外的位置**不被写**，所以目的地保留原值。
    `count=0` 时整块都是哨兵；`count=n` 时退化成 `add_one`。
    """
    out = np.full_like(x, sentinel, dtype=np.float32)
    if count > 0:
        out[:, :count] = add_one(x)[:, :count]
    return out


def one_register(x, vl):
    """前 `vl` 个元素——`vf.load` / `vf.store` 一次只搬一个寄存器的量。"""
    return x[:, :vl].copy()


def broadcast_rows(x, scalar, active, vl):
    """`vf.full` 该产生的三行。

    三行分别是 Scalar 模式配偏掩码、Tensor 模式 `LOWEST`、Tensor 模式 `HIGHEST`。

    **`vf.full` 的 MergeMode 默认是 `ZEROING`**（`MERGING` 本设备不支持），所以三行里未被
    掩码选中的 lane 都是 `0.0`，不是保留原值——这和 `vf.move` 的默认恰好相反。
    """
    row = np.zeros(vl, dtype=np.float32)
    lanes = x.reshape(-1)[:vl]

    scalar_row = row.copy()
    scalar_row[:active] = np.float32(scalar)

    lowest_row = row.copy()
    lowest_row[:active] = lanes[0]

    highest_row = row.copy()
    highest_row[:active] = lanes[vl - 1]

    return np.stack([scalar_row, lowest_row, highest_row])


def move_merging(x, keep, active, vl):
    """`vf.move` 配偏掩码 + 复用寄存器：前 `active` 个来自 `x`，其余保留 `keep`。

    这是 `MERGING`（`vf.move` 的默认，且唯一支持的模式）在**目的寄存器已有已知内容**时的语义。
    LHS 是新名字时目的寄存器是新分配的，原值未定义，那种形状没有参考可写。
    """
    out = keep.reshape(-1)[:vl].copy()
    out[:active] = x.reshape(-1)[:active]
    return out.reshape(1, vl)
