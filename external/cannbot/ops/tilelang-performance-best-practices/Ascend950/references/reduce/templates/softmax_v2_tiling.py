# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from .softmax_v2_tiling_data import SoftmaxTiling


def select_tiling(rows: int, cols: int, num_cores: int = 64) -> SoftmaxTiling:
    if cols <= 4096:
        strategy, tile_cols = "full_load", cols
    elif cols <= 16384:
        strategy, tile_cols = "recompute", 4096
    else:
        strategy, tile_cols = "online", 4096
    return SoftmaxTiling(rows, cols, tile_cols, num_cores, strategy)
