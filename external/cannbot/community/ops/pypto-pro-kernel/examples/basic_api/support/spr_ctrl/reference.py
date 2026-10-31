# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CTRL 位域的参考计算。不 import 编译器。

这一份很薄，因为 CTRL 的判据大多是**实测值本身**——没有第三方实现可以对照
（`saturation` / `spr_ctrl` 在 ascriptor 里对应 `saturation_flags` / `cast_saturation`,
但 CTRL 的位布局是 A5 特有的）。所以这里只放两样东西：

1. 从一个 64 位值里取位区间（判据要能把「整寄存器」和「单个位」对起来）;
2. 入口值与文档默认值的**差集**——这一例的核心结论就是这个集合非空。
"""

from __future__ import annotations

# 官方 `reset_ctrl_spr.md` 写的 CTRL 默认值，也是 C++ `defaultCtrl` 的值。
DOCUMENTED_DEFAULT = 0x1000000000000008

# 2026-09-28 在 Ascend950PR 上实测的 kernel 入口值。**和上面那个不同**。
MEASURED_ENTRY = 0x1001000000000008

AR_SEEN = (0, 128)
"""`pl.get_spr()` 实测出现过的两个值。**它不是一个常量**：

| 进程 | 读到 |
| --- | ---: |
| 只跑 `spr_ctrl` 自己的 kernel | **128** |
| 跑完整个 `support` 一章再跑到这里 | **0** |

两个进程里它都**不随有效元素个数变**（0/1/7/8/16/30/31/32/63/64 全同）、
section 前后相同、连不带任何 vf 指令的 kernel 也相同。

所以能站住的结论是「**同一个进程内它不随 squeeze 变**」，不是「它等于 128」。
判据断言的是这些读数彼此相等，**不钉具体值**——
`references/verification/board-runs.md` §五点十：判据的严格程度不该超过被测量的可重复性。

差一点就把 `== 128` 写进判据了：单独跑 15 passed，整章跑 5 条红。
"""


def bits(value: int, start: int, end: int) -> int:
    """取 `value` 的 `[start, end]` 位区间，和 `pl.get_ctrl_spr(start, end)` 同义。

    判据用它把「整寄存器读数」和「单个位读数」互相验证：两边都是实测量,
    所以这一条验的是框架内部一致，不是「我猜的默认值对不对」。
    """
    if not 0 <= start <= end <= 63:
        raise ValueError(f"位区间越界：[{start}, {end}]")
    width = end - start + 1
    return (value >> start) & ((1 << width) - 1)


def differing_bits(lhs: int, rhs: int) -> list[int]:
    """两个 64 位值不同的位号，从低到高。"""
    diff = (lhs ^ rhs) & ((1 << 64) - 1)
    return [bit for bit in range(64) if (diff >> bit) & 1]


def set_bit(value: int, bit: int, to: int) -> int:
    """把 `value` 的第 `bit` 位写成 `to`，其余不变——`set_ctrl_spr` 的单比特形式。"""
    cleared = value & ~(1 << bit)
    return cleared | ((to & 1) << bit)
