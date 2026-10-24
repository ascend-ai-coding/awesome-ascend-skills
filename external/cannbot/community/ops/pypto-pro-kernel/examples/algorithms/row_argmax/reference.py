# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent scalar reference for first-occurrence row argmax."""

import numpy as np


def row_argmax_reference(x):
    """Return exact FP32 maxima and INT32 first indices, row by row."""
    source = np.asarray(x, dtype=np.float32)
    rows, cols = source.shape
    values = np.empty((rows,), dtype=np.float32)
    indices = np.empty((rows,), dtype=np.int32)
    for row in range(rows):
        best_value = float(source[row, 0])
        best_index = 0
        for col in range(1, cols):
            value = float(source[row, col])
            if value > best_value:
                best_value = value
                best_index = col
        values[row] = np.float32(best_value)
        indices[row] = best_index
    return values, indices
