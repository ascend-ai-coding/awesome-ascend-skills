# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from ..scan_tiling_data import ScanStrategy

STATUS = "DESIGN_ONLY"
STRATEGY = ScanStrategy(
    "oneway_sklansky",
    "one core owns one resident row",
    True,
    "ceil(log2(R))",
    "static R; each level broadcasts the group anchor to the upper half",
)


def level_pairs(length: int, level: int):
    """Return (anchor, target) pairs for one forward Sklansky level."""
    half = 1 << level
    group = half << 1
    return [
        (start + half - 1, target)
        for start in range(0, length, group)
        for target in range(start + half, min(start + group, length))
    ]
