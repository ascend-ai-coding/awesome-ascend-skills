# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""六个算子的参考实现，以及为什么全都能 `rtol=0, atol=0` 比。

**操作数是挑过的**：`a` 跨正负（激活函数才有意义），`slope` 与标量 alpha 都取 **2 的幂**
（0.5 / 0.25），所以乘法只改阶码、尾数一位不动。`b` / `c` 同样取 2 的幂。

| 运算 | 结果 | 为什么精确 |
| --- | --- | --- |
| `relu` / `neg` | 原值或 0 或取反 | 不做算术 |
| `prelu` / `leaky_relu` | 负半边乘 2 的幂 | 只改阶码 |
| `mul_add_dst` = `c + a*b` | 小整数 | FP32 精确表示整数到 2^24 |
| `mul_dst_add` = `a*b + c` | 同上 | 同上 |

两个融合算子的**数值结果相同**（都是 `a*b + c`），区别在 dst 出现在公式的哪个位置——
所以参考里只有一个 `fused` 函数，而 kernel 侧用两条不同的路径算它。这是故意的：
如果两者算出不同的数，说明我把 dst 的预装值搞错了。
"""

import numpy as np


def make_operands(n):
    """`a` 从 -32 到 31（含负数）；`slope` / `b` / `c` 都是 2 的幂循环。"""
    a = (np.arange(n, dtype=np.float32) - n // 2).reshape(1, n)  # -32..31
    slope = np.tile(np.array([0.5, 0.25], dtype=np.float32), n // 2).reshape(1, n)
    b = np.tile(np.array([2.0, 4.0], dtype=np.float32), n // 2).reshape(1, n)
    c = np.tile(np.array([8.0, 16.0], dtype=np.float32), n // 2).reshape(1, n)
    return a, slope, b, c


ALPHA = 0.5


def unary(a, slope, alpha=ALPHA):
    """`relu` / `neg` / `prelu` / `leaky_relu`，顺序与 kernel 的输出顺序一致。"""
    a64 = a.astype(np.float64)
    s64 = slope.astype(np.float64)
    return [
        np.maximum(a64, 0.0).astype(np.float32),
        (-a64).astype(np.float32),
        np.where(a64 >= 0, a64, a64 * s64).astype(np.float32),  # 逐元素斜率
        np.where(a64 >= 0, a64, a64 * alpha).astype(np.float32),  # 标量斜率
    ]


def fused(a, b, c):
    """`mul_add_dst` 与 `mul_dst_add` 都等于 `a*b + c`。

    - `mul_add_dst(a, b)`，dst 预装 `c` → `a*b + c`
    - `mul_dst_add(b, c)`，dst 预装 `a` → `a*b + c`

    两条路径算同一个数。**这是检验 dst 预装值对不对的办法**：如果 kernel 里预装错了，
    两个输出就会不一样，而参考只给一个值，必然有一个对不上。
    """
    a64, b64, c64 = a.astype(np.float64), b.astype(np.float64), c.astype(np.float64)
    both = (a64 * b64 + c64).astype(np.float32)
    return [both, both]
