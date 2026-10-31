# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""按数学定义计算 FP64 参考，保留契约要求的 FP32 相加与 FP16 RNE。"""

import numpy as np


def reference(x, residual, gamma):
    rounded = np.add(
        x.astype(np.float32), residual.astype(np.float32), dtype=np.float32
    ).astype(np.float16)
    r64 = rounded.astype(np.float64)
    normalized = r64 / np.sqrt(np.mean(r64 * r64, axis=1, keepdims=True) + 1e-5)
    u = normalized * gamma.astype(np.float64)[None, :]
    maximum = np.max(np.abs(u), axis=1)
    scale = np.where(maximum == 0, 1.0, maximum / 127.0)
    z = u / scale[:, None]
    quantized = np.clip(np.rint(z), -127, 127).astype(np.int8)
    return rounded, scale, z, quantized
