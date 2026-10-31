# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""核／子核编号的参考。不 import 编译器。

编号本身没什么可算的——判据大多是「这一批数是 `0..n-1` 的一个排列」这类结构性质。
这一份就是那些性质的形式化。
"""

from __future__ import annotations

import numpy as np

PURE_SECTION_SUBBLOCKS = 1
"""只有一段（纯 vector 或纯 cube）时 `get_subblock_num()` 的实测值。"""

MIXED_SECTION_SUBBLOCKS = 2
"""**两段都有**时的实测值。这个差别是这一例的核心结论之一。"""

TIGHT_SURVIVORS = (2, 3)
"""紧凑跨距（16 字节）、8 个 block 时，实测留下来的 block 个数落在这个范围。

10 次观测：2、3、2、2、2、2、2、2、2、2。**判据不断言这个数**——
`references/verification/board-runs.md` §五点十：判据的严格程度不该超过被测量的可重复性。
它在这里的作用是记录量到了什么。
"""


def reported_rows(
    flat: np.ndarray, stride: int, blocks: int, quantities: int
) -> np.ndarray:
    """把「每个 block 一段」的扁平输出切成 `blocks × quantities`。"""
    return flat.reshape(-1, stride)[:blocks, :quantities]


def surviving_blocks(flat: np.ndarray, stride: int, blocks: int) -> list[int]:
    """哪些 block 的数据完整留下来了：它那一段的第一个量等于自己的编号。"""
    rows = flat.reshape(-1, stride)
    return [block for block in range(blocks) if rows[block][0] == block]


def is_self_consistent(
    row: np.ndarray, block: int, blocks: int, subblocks: int
) -> bool:
    """一个 block 那一段的四个量是否互相自洽。

    **这条是紧凑跨距那一组唯一能断言的东西**：留下来的数据必须是某个 block 真写的,
    不能是拼接出来的。自洽靠三件事——编号对得上、总数对得上、子核号在范围内。
    """
    idx, num, sidx, snum = (int(v) for v in row)
    return (
        idx == block and num == blocks and snum == subblocks and 0 <= sidx < subblocks
    )


def is_a_permutation(values: np.ndarray, n: int) -> bool:
    """`values` 是否恰好是 `0..n-1` 的一个排列。

    比「每个值都在范围内」强：它还排除了重号和缺号。
    """
    return sorted(int(v) for v in values) == list(range(n))
