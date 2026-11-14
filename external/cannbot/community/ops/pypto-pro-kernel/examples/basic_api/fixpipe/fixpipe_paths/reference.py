# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""numpy 参考与操作数构造。

判据：`rtol=0, atol=0`。FP16 输入取小整数，乘积和累加全在 FP32 能精确表示的范围内,
所以设备和参考应当逐位相同——**这一例不需要容差**，出现差异就是搬运或布局错了。
"""

import numpy as np


def make_operands(m, k, n, seed=0):
    """FP16 的小整数矩阵。

    为什么不用随机浮点：这一例验的是**搬运与布局**，不是数值精度。整数输入下
    `a @ b` 在 FP32 里精确，判据可以是 `atol=0`——一有差异就只可能是排布错了,
    不用去分辨「是精度还是 bug」。

    取值范围：`a` 在 `[-4, 4)`，`b` 在 `[-2, 2)`。K=128 时最大累加量级
    `128 * 4 * 2 = 1024`，FP32 精确表示整数到 2^24，差得很远。

    **`a` 里必须有负数**：后面 `fixpipe_riders` 那一例要验随路 relu，
    全正的输入下 relu 是恒等映射，验不到。同一个构造函数两例共用。
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(-4, 4, size=(m, k)).astype(np.float16)
    b = rng.integers(-2, 2, size=(k, n)).astype(np.float16)
    return a, b


def make_third(n, seed=1):
    """`acc_to_mat_chain` 的第三个操作数 `c`，同样是 FP16 小整数。"""
    rng = np.random.default_rng(seed)
    return rng.integers(-2, 2, size=(n, n)).astype(np.float16)


def matmul(a, b):
    """`a @ b`，FP32 累加——和 cube 的 FP16×FP16→FP32 一致。"""
    return (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float32)


def matmul_chain(a, b, c):
    """`fp16(a @ b) @ c`——中间结果经 fixpipe 降位到 FP16 才进第二次乘法。

    **参考必须照做这个降位**，否则比的不是同一件事。小整数下 `a @ b` 最大 1024,
    FP16 精确表示整数到 2048，所以这一步实际不丢位——判据仍是 `atol=0`。
    """
    mid = matmul(a, b).astype(np.float16)
    return matmul(mid, c)


def has_both_signs(x):
    """输入是否同时有正负。relu 那一例的前提，在这一例先断言一次。"""
    return bool((x > 0).any() and (x < 0).any())
