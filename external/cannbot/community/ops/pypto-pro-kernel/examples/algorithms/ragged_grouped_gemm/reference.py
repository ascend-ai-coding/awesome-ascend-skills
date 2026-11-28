# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""独立 FP64 分组矩阵乘与逐元素绝对项和。"""

import numpy as np


def reference(a, w, offsets):
    left = np.asarray(a, dtype=np.float16).astype(np.float64)
    weights = np.asarray(w, dtype=np.float16).astype(np.float64)
    output = np.empty((left.shape[0], weights.shape[2]), np.float64)
    term_scale = np.empty_like(output)
    for group in range(weights.shape[0]):
        begin, end = map(int, offsets[slice(group, group + 2)])
        output[begin:end] = left[begin:end] @ weights[group]
        term_scale[begin:end] = np.abs(left[begin:end]) @ np.abs(weights[group])
    return output, term_scale
