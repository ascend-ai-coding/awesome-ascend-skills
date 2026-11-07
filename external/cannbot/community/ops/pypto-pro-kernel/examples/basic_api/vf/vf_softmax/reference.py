# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""numpy 参考与操作数构造。

**这一例的判据不是 `atol=0`**，理由写在 `softmax` 的 docstring 里——`exp` 的实现误差
没有文档保证。阈值是实测出来的，不是挑一个好看的数。
"""

import numpy as np


def make_operands(rows, cols, seed=0):
    """`[-8, 8)` 的随机 FP32，行与行之间的量级不同。

    三个考虑：

    - **要有负数也要有正数**，否则减去行最大之后全是 0，`exp` 一律给 1，验不到东西。
    - **每行的最大值位置不同**（随机），否则「广播的是 `dst[0]` 还是恰好对」分不清。
    - **行间量级不同**：第 `r` 行整体乘 `2^r`，让行 0 的 softmax 接近均匀、
      最后一行接近 one-hot。两个极端都覆盖到。
    """
    rng = np.random.default_rng(seed)
    base = rng.uniform(-8.0, 8.0, size=(rows, cols))
    scale = (2.0 ** np.arange(rows)).reshape(rows, 1)
    return (base * scale).astype(np.float32)


def softmax(a):
    """行 softmax，**用 float64 算**，作为「数学上对」的基准。

    为什么不用 float32 参考：两条设备链（vf 与 tile）的差异本身是要测的量，
    参考如果也带 FP32 误差，就分不清「设备错了」和「参考错了」。
    float64 参考 + 实测阈值，把两件事分开。

    **判据不能是 `atol=0`**：`exp` 的 FP32 实现误差没有文档保证
    （官方 `exp.md` 只有一个 `precision` 开关，没给误差界），归约的折叠顺序
    也未必和 numpy 相同。阈值由第一次上板实测的最大相对误差决定，见 test.py。
    """
    x = a.astype(np.float64)
    shifted = x - x.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


def row_sums_are_one(got, tol):
    """softmax 的定义性质：每行和为 1。返回每行和与 1 的最大偏差。

    这条比「和参考逐元素一致」更根本——它不依赖参考实现，**参考写错也抓得住**。
    """
    return float(np.abs(got.astype(np.float64).sum(axis=1) - 1.0).max()), tol
