# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent FP64 definition, consuming the supplied FP32 statistics."""

import numpy as np


def layernorm_backward_reference(x, dy, gamma, mean, rstd):
    x, dy, gamma, mean, rstd = [
        np.asarray(value, dtype=np.float64) for value in (x, dy, gamma, mean, rstd)
    ]
    xhat = (x - mean[:, None]) * rstd[:, None]
    u = dy * gamma
    dx = rstd[:, None] * (
        u
        - u.mean(axis=1, keepdims=True)
        - xhat * (u * xhat).mean(axis=1, keepdims=True)
    )
    return dx, (dy * xhat).sum(axis=0), dy.sum(axis=0)
