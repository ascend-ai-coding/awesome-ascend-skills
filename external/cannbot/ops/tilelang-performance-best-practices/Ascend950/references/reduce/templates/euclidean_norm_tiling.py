# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from .euclidean_norm_tiling_data import EuclideanNormTiling


def select_tiling(rows: int, cols: int, num_cores: int = 64) -> EuclideanNormTiling:
    if rows <= 0 or cols <= 0 or num_cores <= 0:
        raise ValueError("rows, cols, and num_cores must be positive")
    padded_cols = 1 << (cols - 1).bit_length()
    tile_cols = min(padded_cols, 4096)
    strategy = "padded_full_load" if padded_cols <= 4096 else "split_r_design_only"
    return EuclideanNormTiling(rows, cols, tile_cols, num_cores, strategy)
