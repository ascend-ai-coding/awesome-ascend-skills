# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""八个超越函数的参考，以及**这一例为什么不能逐位比**。

前面几例（`vf_arith` / `vf_activation` / `vf_bitwise` / `vf_reduce`）都能 `rtol=0, atol=0`：
操作数挑成 2 的幂或小整数，硬件和 numpy 走的是同一条精确路径。**超越函数没有这个办法**
——`exp` / `ln` 在硬件上是查表加多项式，在 numpy 里是 libm 的实现，两者不可能逐位相同。

所以判据必须是量出来的。实测（2026-09-27，A5，FP32，输入见下）最大相对误差：

| 函数 | 最大相对误差 |
| --- | --- |
| `ln` / `log` | 5.4e-08 |
| `exp` | 7.2e-08 |

FP32 的 1 ulp 约 1.2e-07，所以这两个都在 1 ulp 之内。阈值取 **`rtol=1e-6`**——比实测大一个
数量级出头，留住「换输入区间也够用」的余量，同时远小于「算错一个数量级」那种错误。

**阈值不是拍的，是实测值的十几倍。** 具体测出多少写在 test.py 的表里，这样下次有人把阈值
往上调时，能看出他是在容忍比当初大多少倍的误差。

## 输入区间

- 对数和 `sqrt` 要 **正数**：取 `[0.5, 8.0]` 线性铺开。负数进 `ln` 是 NaN，那是另一个主题
  （未测，本例不声称）。
- `exp` 要避免溢出：同一个 `[0.5, 8.0]` 区间，`exp(8) ≈ 2981`，安全。
- `abs` 要 **跨正负** 才验得到，所以它单独用一份含负数的输入。
"""

import numpy as np

RTOL = 1e-6  # 实测最大相对误差 7.2e-08 的十几倍
LO, HI = 0.5, 8.0


def make_operands(n):
    """`[0.5, 8.0]` 线性铺开。全正——对数和 sqrt 的定义域。"""
    return np.linspace(LO, HI, n, dtype=np.float32).reshape(1, n)


def make_signed_operands(n):
    """跨正负，给 `abs` 用。"""
    return (np.linspace(-HI, HI, n, dtype=np.float32)).reshape(1, n)


def elementwise(a):
    """七个逐元素函数，顺序与 kernel 的输出顺序一致：ln / log / log2 / log10 / exp / sqrt / abs。

    `log` 是 `ln` 的别名，所以第二项与第一项相同——这是故意的，参考里也写成同一个值。
    """
    a64 = a.astype(np.float64)
    ln = np.log(a64).astype(np.float32)
    return [
        ln,
        ln,  # log 是 ln 的别名
        np.log2(a64).astype(np.float32),
        np.log10(a64).astype(np.float32),
        np.exp(a64).astype(np.float32),
        np.sqrt(a64).astype(np.float32),
        np.abs(a64).astype(np.float32),
    ]


def exp_sub(a, b):
    """`exp_sub(a, b) = exp(a - b)`。实测语义，不是从名字猜的。"""
    return np.exp(a.astype(np.float64) - b.astype(np.float64)).astype(np.float32)


def make_exp_sub_operands(n):
    """`a - b` 落在 `[-4, 4]`，exp 不溢出也不下溢到 0。"""
    a = np.linspace(0.0, 4.0, n, dtype=np.float32).reshape(1, n)
    b = np.full((1, n), 2.0, dtype=np.float32)
    return a, b
