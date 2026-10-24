# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from .scan_tiling_data import ScanTiling


def select_tiling(rows: int, cols: int, num_cores: int = 64) -> ScanTiling:
    if cols <= 256:
        strategy, tile_cols = "tile_resident", cols
    elif rows >= num_cores:
        strategy, tile_cols = "streaming", 256
    else:
        strategy, tile_cols = "core_partition", 256
    return ScanTiling(rows, cols, tile_cols, num_cores, strategy)
