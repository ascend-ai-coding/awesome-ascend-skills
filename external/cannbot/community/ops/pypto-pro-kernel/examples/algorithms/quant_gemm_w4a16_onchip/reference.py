# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""按公开打包格式解码，执行指定 FP32→FP16 舍入后在 FP64 中计算矩阵积。"""

import numpy as np


def reference(a, packed, scale, zero, bias, g):
    q = np.empty((packed.shape[0] * 2, packed.shape[1]), np.float32)
    q[0::2] = packed & np.uint8(15)
    q[1::2] = packed >> np.uint8(4)
    w = (
        (
            (q - np.repeat(zero, g, axis=0).astype(np.float32))
            * np.repeat(scale, g, axis=0)
        )
        .astype(np.float16)
        .astype(np.float64)
    )
    a64 = a.astype(np.float64)
    return a64 @ w + bias.astype(np.float64), np.abs(a64) @ np.abs(w)
