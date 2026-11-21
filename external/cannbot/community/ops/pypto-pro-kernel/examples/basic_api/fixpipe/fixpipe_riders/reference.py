# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""numpy 参考与操作数构造。

判据 `rtol=0, atol=0`：目的类型是 FP16，`scale` 取 2 的幂，输入是小整数,
`scale * acc` 在 FP16 里精确（最大量级 50，FP16 精确表示整数到 2048）。
**选 FP16 而不是 FP32 是为了避开 `FP32→FP32` 丢 scale 那个缺陷**，见 kernel.py。
"""

import numpy as np


def make_operands(m, k, n, seed=0):
    """FP16 小整数，**acc 里必须有负数**——relu 那一条的前提。

    `a ∈ [-4, 4)`、`b ∈ [-2, 2)`，K=64 下 acc 落在 `[-57, 100]`（实测），
    负数约占四成。`test_the_accumulator_has_negatives` 把这个前提钉成测试。
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(-4, 4, size=(m, k)).astype(np.float16)
    b = rng.integers(-2, 2, size=(k, n)).astype(np.float16)
    return a, b


def accumulate(a, b):
    """cube 的累加结果，用 float64 算。"""
    return a.astype(np.float64) @ b.astype(np.float64)


def scaled(a, b, scale):
    """不带 rider 的期望：`scale * acc`。"""
    return accumulate(a, b) * scale


def scaled_relu(a, b, scale):
    """带 `relu_pre_mode=NormalRelu` 的期望：负数钳到 0。

    `scale > 0` 时「先 relu 再乘」和「先乘再 relu」结果相同，所以这个参考
    对两种顺序都成立——本例因此验不出 `pre` 到底 pre 在哪（见 kernel.py）。
    """
    return np.maximum(scaled(a, b, scale), 0.0)


def accumulate_twice(a, b):
    """`matmul` + `matmul_acc` 的期望：`2 * (a @ b)`。"""
    return accumulate(a, b) * 2.0
