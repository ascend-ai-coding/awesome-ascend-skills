# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""MX（microscaling）操作数怎么在 host 侧造出来，以及参考实现。

**MX 是什么**：每 32 个连续元素（沿归约轴）共享一个 **E8M0** 指数字节，这 32 个元素各自
存成一个 **FP8** 尾数。一个元素的值是 `fp8_value * 2^(e8m0 - 127)`。

三条编码约定都对着框架 ST（`test_matmul_mx.py`）逐元素核过（2026-09-26）：

| | 约定 | 核对方式 |
| --- | --- | --- |
| E4M3 | `(1 + M/8) * 2^(E - 7)`，bias 7，3 位尾数 | 用 ST 的 `_decode_fp8` 验了 16 个码，全符 |
| E8M0 | `2^(byte - 127)`，所以 127 就是 ×1 | 同上 |
| GM 上 scale 的物理形状 | 组轴拆成 `[G/2, 2]`，再把那个 2 挪到**最后** | 与 ST 的 `_pack_mx_scale_axis` 逐元素相同 |

**这个样例只用 E4M3 的「单位阶梯」**，即 `E=7`（偏置后 2^0）那一档：

```
bits  s 1 1 1 M M M     code = s*128 + 56 + M     value = ±(1 + M/8)
```

16 个字节、16 个值，覆盖 ±1.0 .. ±1.875，全部精确可表示，所以参考不需要量化步骤。
（这和 `hif8_fp32` 是同一个手法，理由见那里：参考必须吃 kernel 真正吃到的值。）

**由此得到一个很强的性质**：scale 取 `{0.5, 1, 2}`，于是每个乘积项是 `2^-8` 的倍数、
绝对值不超过 14；K=128 的和不超过约 1800，而 `1800 * 256 < 2^24`。**整条和在 FP32 里
精确可表示**，所以结果可以按**逐位相等**验，不给任何容差。
"""

import numpy as np

GROUP_SIZE = 32  # 一个 E8M0 管 32 个元素
PAIR = GROUP_SIZE * 2  # GM 上一个 [.., 2] 里装两个组，所以 K 轴的单位是 64
E8M0_CHOICES = (126, 127, 128)  # -> 0.5, 1.0, 2.0


def _unit_ladder():
    """E4M3 里 E=7 那一档：16 个码，16 个精确值，按值排序。"""
    codes, values = [], []
    for sign_bit in (0, 1):
        sign = -1.0 if sign_bit else 1.0
        for m in range(8):
            codes.append(sign_bit * 128 + 56 + m)
            values.append((1.0 + m / 8) * sign)
    order = np.argsort(values)
    return np.asarray(values, np.float32)[order], np.asarray(codes, np.uint8)[order]


LADDER_VALUES, LADDER_CODES = _unit_ladder()


def _codes_and_values(shape, seed):
    """从阶梯里抽码，并给出它们的精确值。"""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, LADDER_VALUES.size, size=shape)
    return LADDER_CODES[idx], LADDER_VALUES[idx]


def _scales(shape, seed):
    """E8M0 字节（逻辑形状）和它们代表的倍数。"""
    rng = np.random.default_rng(seed)
    e8m0 = np.asarray(E8M0_CHOICES, np.uint8)[
        rng.integers(0, len(E8M0_CHOICES), size=shape)
    ]
    return e8m0, np.exp2(e8m0.astype(np.int16) - 127).astype(np.float32)


def pack_scale_a(e8m0):
    """A 侧：逻辑 `[M, G]` -> 物理 `[M, G/2, 2]`。组轴在后，拆开即可。"""
    rows, groups = e8m0.shape
    return np.ascontiguousarray(e8m0.reshape(rows, groups // 2, 2))


def pack_scale_b(e8m0):
    """B 侧：逻辑 `[G, N]` -> 物理 `[G/2, N, 2]`。组轴在前，拆开后要把那个 2 挪到最后。"""
    groups, cols = e8m0.shape
    return np.ascontiguousarray(e8m0.reshape(groups // 2, 2, cols).transpose(0, 2, 1))


def make_operands(m, k, n, seed=0):
    """返回喂给 kernel 的字节，和参考要用的精确值。

    A 的组轴是 K（最后一维），B 的组轴也是 K（第一维）——归约轴两侧都是它。
    """
    a_codes, a_mantissa = _codes_and_values((m, k), seed)
    b_codes, b_mantissa = _codes_and_values((k, n), seed + 1)
    a_e8m0, a_scale = _scales((m, k // GROUP_SIZE), seed + 2)
    b_e8m0, b_scale = _scales((k // GROUP_SIZE, n), seed + 3)

    a_values = a_mantissa * np.repeat(a_scale, GROUP_SIZE, axis=1)
    b_values = b_mantissa * np.repeat(b_scale, GROUP_SIZE, axis=0)
    return dict(
        a_codes=a_codes,
        b_codes=b_codes,
        scale_a=pack_scale_a(a_e8m0),
        scale_b=pack_scale_b(b_e8m0),
        a_values=a_values.astype(np.float32),
        b_values=b_values.astype(np.float32),
    )


def matmul_reference(a_values, b_values):
    """在 float64 上算再降到 float32。

    和 kernel 的 FP32 累加不是同一个顺序，但**这组操作数下整条和精确可表示**，所以两者
    应当逐位相同（见模块 docstring）。
    """
    return (a_values.astype(np.float64) @ b_values.astype(np.float64)).astype(
        np.float32
    )
