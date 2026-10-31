# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""五个位运算的参考。整数运算没有浮点问题，全部 `rtol=0, atol=0`。

**操作数是挑过的**，为了让每个算子都真的被验到：

- `a` 含**负数**（从 -32 到 31）——右移必须验到负数，否则分不清算术右移和逻辑右移。
- `b` 是 `0x0F0F0F0F` / `0xF0F0F0F0` / `0x00FF00FF` 循环——位模式互补，`and`/`or`/`xor`
  三者的结果两两不同。用小整数当 `b` 的话，`or` 和 `xor` 在很多位置会相等，测不出区别。

实测（2026-09-27，A5）：`shift_right` 对负数是**算术右移**（补符号位），见
`test_shift_right_is_arithmetic_on_negative_values`。
"""

import numpy as np

SHIFT = 2

_MASKS = [
    0x0F0F0F0F,
    -0x0F0F0F10,
    0x00FF00FF,
    -0x00FF0100,
]  # 后两个是 0xF0F0F0F0 / 0xFF00FF00 的有符号形式


def make_operands(n):
    """`a` 跨正负；`b` 是互补的位模式循环。两者都用 int32。"""
    a = (np.arange(n, dtype=np.int32) - n // 2).reshape(1, n)
    b = np.tile(np.array(_MASKS, dtype=np.int32), n // len(_MASKS)).reshape(1, n)
    return a, b


def bitwise(a, b, shift=SHIFT):
    """顺序与 kernel 的输出顺序一致：and / or / xor / shl / shr。

    numpy 的 `>>` 对 int32 是算术右移，与实测的硬件行为一致；`<<` 会回绕，
    和硬件一样截断到 32 位，所以两边都不需要特殊处理。
    """
    return [
        (a & b).astype(np.int32),
        (a | b).astype(np.int32),
        (a ^ b).astype(np.int32),
        (a << shift).astype(np.int32),
        (a >> shift).astype(np.int32),
    ]


def mask_logic(n, active):
    lane = np.arange(n)
    a, b = (lane % 2 == 0), (lane // 2 % 2 == 0)
    gate = lane < active
    return np.stack(
        [(a & b) & gate, (a | b) & gate, (a ^ b) & gate, (~a) & gate]
    ).astype(np.int32)
