# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from .broadcast_common import BroadcastTiling


def select_tiling(
    rows: int, cols: int, dtype_bytes: int, num_cores: int = 64
) -> BroadcastTiling:
    ub_budget = 96 * 1024
    vectors = max(1, ub_budget // (3 * dtype_bytes * 256))
    # PTO SIMT layout inference requires a complete vector/thread footprint.
    # Keep the GM extent in ``valid`` while padding the UB/fragment extent.
    padded_cols = ((cols + 255) // 256) * 256
    tile_cols = min(padded_cols, vectors * 256)
    cfg = BroadcastTiling(rows, cols, tile_cols, num_cores)
    cfg.validate()
    return cfg
