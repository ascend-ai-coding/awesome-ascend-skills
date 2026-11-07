# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""十三个算子的参考实现，以及为什么全都能 `rtol=0, atol=0` 比。

**操作数是挑过的，不是随机的。** `a = 1..64`（小整数），`b` 是 2 的幂循环（2/4/8/16）。于是：

| 运算 | 结果 | 为什么精确 |
| --- | --- | --- |
| `a + b`、`a - b` | 整数，绝对值 ≤ 80 | FP32 精确表示整数到 2^24 |
| `a * b` | 整数 ≤ 1024 | 同上 |
| `a / b` | `a` 除以 2 的幂 | 只改阶码，尾数一位不动 |
| `min` / `max` | 原值之一 | 不做算术 |
| `adds` / `muls`（标量 2.0） | 整数 | 同上 |
| `mins` / `maxs` | 原值或 2.0 | 不做算术 |
| `abs_sub` | 整数 | 同上 |
| `axpy` = `2a + b` | 整数 ≤ 208 | 同上 |

所以参考在 float64 上算、降到 float32，和设备上的 FP32 运算**逐位相同**。这不是「误差小」，是没有误差
——除法那条特别值得留意：换成非 2 的幂（比如 3）就必须改成带容差比，那时这一例检验的东西就弱了。

`mull` 走整数，没有浮点问题；但要注意 numpy 的 int32 乘法会**回绕**，参考必须先升到 int64 再拆。
"""

import numpy as np

SCALAR = 2.0


def make_operands(n):
    """`a` 是 1..n，`b` 是 2 的幂循环。两者都不含 0——除法要用 `b` 当除数。"""
    a = np.arange(1, n + 1, dtype=np.float32).reshape(1, n)
    powers = np.array([2.0, 4.0, 8.0, 16.0], dtype=np.float32)
    b = np.tile(powers, n // len(powers) + 1)[:n].reshape(1, n)
    return a, b


def binary(a, b):
    """六个 reg ⊕ reg 算子，顺序与 kernel 的输出顺序一致。"""
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    return [
        (a64 + b64).astype(np.float32),
        (a64 - b64).astype(np.float32),
        (a64 * b64).astype(np.float32),
        (a64 / b64).astype(np.float32),
        np.minimum(a64, b64).astype(np.float32),
        np.maximum(a64, b64).astype(np.float32),
    ]


def scalar(a, s=SCALAR):
    """四个 reg ⊕ 标量算子。"""
    a64 = a.astype(np.float64)
    return [
        (a64 + s).astype(np.float32),
        (a64 * s).astype(np.float32),
        np.minimum(a64, s).astype(np.float32),
        np.maximum(a64, s).astype(np.float32),
    ]


def composite(a, b, s=SCALAR):
    """`abs_sub` 与 `axpy`。

    `axpy` 的定义是 `dst = s * src + dst`，而 kernel 把 `dst` 的初值装成了 `b`，
    所以结果是 `s * a + b`——**不是** `s * a`。这条是 axpy 最容易记错的地方。
    """
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    return [
        np.abs(a64 - b64).astype(np.float32),
        (s * a64 + b64).astype(np.float32),
    ]


def make_wide_operands(n):
    """让乘积一定溢出 32 位，否则 `dst_hi` 全是 0，测不出高位那一半。"""
    a = np.full((1, n), 1 << 20, dtype=np.int32)
    b = np.tile(
        np.array([1 << 15, 1 << 16, 1 << 12, 1 << 20], dtype=np.int32), n // 4 + 1
    )[:n].reshape(1, n)
    return a, b


def wide(a, b):
    """`mull` 的 `(lo, hi)`。

    先升 int64 再拆，因为 numpy 的 int32 乘法会静默回绕——直接 `a * b` 得到的正是被截断的
    `lo`，而 `hi` 就拿不到了。
    """
    product = a.astype(np.int64) * b.astype(np.int64)
    lo = (product & 0xFFFFFFFF).astype(np.uint32).astype(np.int32)  # 低 32 位，按位搬
    hi = (product >> 32).astype(np.int32)
    return lo, hi


def make_carry_operands(n):
    """`addc` / `subc` 用无符号 32 位；取值小，好让 `a - b` 不下溢到需要讨论回绕。"""
    a = np.arange(100, 100 + n, dtype=np.uint32).reshape(1, n)
    b = np.tile(np.array([1, 2, 3, 4], dtype=np.uint32), n // 4 + 1)[:n].reshape(1, n)
    return a, b


def carry(a, b):
    """四个进位组合，顺序与 kernel 的输出顺序一致。

    **`subc` 的 mask 是硬件进位，不是语义借位**：`ALL` 给 `a - b`，`ALLF` 给 `a - b - 1`。
    参考这里按实测写，不按参数名的字面读法写。
    """
    a64, b64 = a.astype(np.int64), b.astype(np.int64)
    as_u32 = lambda x: (x & 0xFFFFFFFF).astype(np.uint32).astype(np.int32)  # noqa: E731
    return [
        as_u32(a64 + b64),  # addc, carry_in = 0
        as_u32(a64 + b64 + 1),  # addc, carry_in = 1
        as_u32(a64 - b64),  # subc, ALL  → 不借位
        as_u32(a64 - b64 - 1),  # subc, ALLF → 借位 1
    ]
