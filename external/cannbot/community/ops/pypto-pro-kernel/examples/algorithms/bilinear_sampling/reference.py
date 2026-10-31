# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""FP64 floor 与四邻点独立数学参考，每个越界邻点单独取零。"""

import numpy as np


def bilinear_reference(x, grid):
    batches, height, width, channels = x.shape
    points = grid.shape[1]
    result = np.zeros((batches, points, channels), dtype=np.float64)
    for batch in range(batches):
        gx, gy = (
            grid[batch, :, 0].astype(np.float64),
            grid[batch, :, 1].astype(np.float64),
        )
        x0, y0 = np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64)
        wx, wy = gx - x0, gy - y0
        for dx, dy in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            xx, yy = x0 + dx, y0 + dy
            valid = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
            weight = (wx if dx else 1 - wx) * (wy if dy else 1 - wy)
            result[batch, valid] += weight[valid, None] * x[
                batch, yy[valid], xx[valid]
            ].astype(np.float64)
    return result
