# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""标量、夹逼和行/列归约的独立 NumPy 参考，不导入编译器。"""

from __future__ import annotations

import numpy as np

CLAMP_LOW = -0.5
CLAMP_HIGH = 0.5


def make_input(rows: int, cols: int, seed: int = 7) -> np.ndarray:
    """固定 seed。判据要跨 kernel 比同一份输入，随机每次不同就没法比。"""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rows, cols)).astype(np.float32)


def elementwise_max(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(a, b)


def clamped(
    a: np.ndarray, low: float = CLAMP_LOW, high: float = CLAMP_HIGH
) -> np.ndarray:
    """先 `maximum` 到下界再 `minimum` 到上界，和 kernel 里的顺序一致。

    **顺序在这里不影响结果**（low < high），但写成和 kernel 相同的顺序:
    参考代码要能逐行对上被测代码，不然出入的时候不知道差在哪。
    """
    return np.minimum(np.maximum(a, np.float32(low)), np.float32(high))


def row_max_dim0(a: np.ndarray) -> np.ndarray:
    """沿末轴求最大值，NumPy 使用 axis=1。

    `dim=0` 保留轴 0：`out[i, 0] = max_j(a[i, j])`，形状 `[M, 1]`。
    """
    return a.max(axis=1, keepdims=True)


def col_min_dim1(a: np.ndarray) -> np.ndarray:
    """沿行轴求最小值，NumPy 使用 axis=0。

    `dim=1` 保留轴 1：`out[0, j] = min_i(a[i, j])`，形状 `[1, K]`。
    """
    return a.min(axis=0, keepdims=True)


def the_two_dims_disagree(a: np.ndarray) -> bool:
    """两个 `dim` 的结果形状不同，所以不可能互相冒充。

    判据用它确认输入的形状选得有区分力：`M == K` 的方阵上形状一样,
    行列写反也能「对」。
    """
    return row_max_dim0(a).shape != col_min_dim1(a).shape
