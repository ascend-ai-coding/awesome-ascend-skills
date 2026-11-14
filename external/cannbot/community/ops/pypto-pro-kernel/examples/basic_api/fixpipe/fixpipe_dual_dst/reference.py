# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""numpy 参考与操作数构造。

目的类型是 FP32、不带 scale，所以 `acc` 本身就是期望值；判据 `rtol=0, atol=0`
（输入是小整数，`a @ b` 在 FP32 里精确）。

**输入必须含负数**——`dual_with_relu` 那条靠「负数还在不在」判断 relu 是否被丢。
"""

import numpy as np


def make_operands(m, k, n, seed=0):
    """FP16 小整数，`a @ b` 里负数约占四成。

    和 [`fixpipe_riders`](../fixpipe_riders/reference.py) 用的是同一组参数，
    那边有一条测试专门钉「acc 里有负数」这个前提，这里再钉一次——
    两个样例各自独立，不互相依赖。
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(-4, 4, size=(m, k)).astype(np.float16)
    b = rng.integers(-2, 2, size=(k, n)).astype(np.float16)
    return a, b


def accumulate(a, b):
    """cube 的累加结果，用 float64 算。"""
    return a.astype(np.float64) @ b.astype(np.float64)


def relu(x):
    """`relu_pre_mode=NormalRelu` 生效时的期望：负数钳到 0。"""
    return np.maximum(x, 0.0)
