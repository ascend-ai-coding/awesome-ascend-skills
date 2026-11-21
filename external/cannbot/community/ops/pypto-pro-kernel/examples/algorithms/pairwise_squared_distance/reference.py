# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""从实际 FP16 输入直接求 FP64 差平方和，不使用范数展开或 kernel 的分块顺序。"""

import numpy as np


def reference(x, z):
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    result = np.empty((x.shape[0], z.shape[0]), dtype=np.float64)
    for begin in range(0, len(x), 16):
        result[begin: begin + 16] = (
            (x[begin: begin + 16, None, :] - z[None, :, :]) ** 2
        ).sum(-1)
    return result
