# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""四个 mask 状态 API 的参考。全部 `rtol=0, atol=0`——掩码是位，没有浮点误差。

`_show_spr` 用 `select(ones, zeros, mask)` 把掩码变成 1.0 / 0.0，所以参考就是位展开。
"""

import numpy as np

BITMAP = 0x0F0F0F0F
NARROW = 0x0000FFFF
COUNT = 16


def bits_to_lanes(value, n):
    """第 i 位 → 第 i 个 lane。`get_mask_spr` 在 FP32 下读 MASK0 的 64 bit，每 bit 复制 4 份。"""
    return np.array([(value >> i) & 1 for i in range(n)], dtype=np.float32).reshape(
        1, n
    )


def all_lanes(n):
    return np.ones((1, n), dtype=np.float32)


def count_mode_misread(n, count=COUNT):
    """**计数模式下 `get_mask_spr` 把计数当位图读**——这是实测的错误行为，不是期望行为。

    设计数 16，读成 `0b10000` → 只有 lane 4 有效。参考按这个「错」写，因为测试要钉住现状。
    """
    return bits_to_lanes(count, n)
