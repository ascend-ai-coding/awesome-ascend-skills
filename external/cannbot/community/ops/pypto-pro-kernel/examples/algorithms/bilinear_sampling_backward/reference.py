# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""独立 NumPy 数学参考，低精度输入按契约给定的实际值消费。"""

import numpy as np


def reference(x, grid, grad):
    b, h, w, c = x.shape
    px = grid[..., 0].astype(np.float64)
    py = grid[..., 1].astype(np.float64)
    x0 = np.floor(px).astype(np.int64)
    y0 = np.floor(py).astype(np.int64)
    fx = px - x0
    fy = py - y0
    dx = np.zeros(x.shape, dtype=np.float64)
    dg = np.zeros(grid.shape, dtype=np.float64)
    gf = grad.astype(np.float64)
    for oy in (0, 1):
        for ox in (0, 1):
            xx = x0 + ox
            yy = y0 + oy
            valid = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
            wx = fx if ox else 1 - fx
            wy = fy if oy else 1 - fy
            weight = wx * wy
            gradx = (1 if ox else -1) * wy
            grady = (1 if oy else -1) * wx
            for batch in range(b):
                live = valid[batch]
                values = x[batch, yy[batch, live], xx[batch, live]].astype(np.float64)
                channel_dot = (values * gf[batch, live]).sum(axis=-1)
                dg[batch, live, 0] += channel_dot * gradx[batch, live]
                dg[batch, live, 1] += channel_dot * grady[batch, live]
                np.add.at(
                    dx[batch],
                    (yy[batch, live], xx[batch, live]),
                    gf[batch, live] * weight[batch, live, None],
                )
    return dx, dg
