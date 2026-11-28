# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""FP32 → INT16 的两种结果：饱和与截断。不 import 编译器。

判据要区分的就是这两个函数的输出。**它们在越界输入上必须不同**——
`probe_values()` 的前四个值就是为此选的，后两个在范围内，两种模式下应当相同（控制项）。
"""

from __future__ import annotations

import numpy as np

INT16_MIN = -32768
INT16_MAX = 32767

PROBE = np.array(
    [50000.0, -50000.0, 100000.0, -100000.0, 100.0, -100.0], dtype=np.float32
)
"""前四个远超 INT16 范围，后两个在范围内。

**每个值都精确可表示**，所以参考里不用考虑 `round_mode`（缺省 `CAST_RINT`）：
50000 / 100000 在 FP32 里是精确的，转换只做位宽处理，不做舍入。
选 50000 而不是 40000：`40000 & 0xFFFF = 40000`，当成 int16 是 -25536,
和 32767 差得开；但 50000 更不容易和「乘 2 溢出」这类别的解释混。
"""

IN_RANGE_FROM = 4
"""`PROBE[IN_RANGE_FROM:]` 是范围内的那两个。两种模式下它们必须相同——
**这是这一组的控制项**：如果它们也不同，说明变的不是「越界怎么办」而是别的东西。
"""


def saturated() -> np.ndarray:
    """饱和：越界值贴到目标类型的边界。"""
    return np.clip(PROBE, INT16_MIN, INT16_MAX).astype(np.int16).astype(np.float32)


def truncated() -> np.ndarray:
    """截断：取低 16 位，高位丢掉，再按 int16 解释。

    `50000 = 0xC350` → int16 是 `-15536`。符号在这里是**位型决定的**,
    不是「保留原来的符号」——这一点是截断模式最容易想错的地方。
    """
    low16 = (PROBE.astype(np.int64) & 0xFFFF).astype(np.uint16)
    return low16.view(np.int16).astype(np.float32)


def modes_differ_on() -> np.ndarray:
    """两种模式给出不同结果的下标。判据用它确认探针输入确实有区分力。"""
    return np.flatnonzero(saturated() != truncated())
