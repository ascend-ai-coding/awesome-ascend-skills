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
    "row_kogge_stone",
    "one core owns one resident row",
    True,
    "ceil(log2(R))",
    "static R; every level combines element i with i-2^level",
)


def level_pairs(length: int, level: int):
    distance = 1 << level
    return [(i - distance, i) for i in range(distance, length)]
