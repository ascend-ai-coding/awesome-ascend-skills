# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from .scan_base import build as _baseline
from ..scan_tiling_data import ScanStrategy

STATUS = "EXECUTABLE_BASELINE"
STRATEGY = ScanStrategy(
    "tile_resident",
    "one core owns one complete row",
    True,
    "O(R) baseline",
    "the complete row and fp32 output fit UB",
)


def build(rows: int, cols: int, num_cores: int | None = None):
    return _baseline(rows, cols, cols, num_cores)
