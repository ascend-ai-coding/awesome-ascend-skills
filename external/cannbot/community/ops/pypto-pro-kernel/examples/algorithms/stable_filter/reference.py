# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent stable-filter definition using NumPy selection."""

import numpy as np


def filter_reference(x):
    values = np.zeros_like(x)
    indices = np.full(x.shape, -1, dtype=np.int32)
    counts = np.count_nonzero(x > 0, axis=1).astype(np.int32)
    for row in range(x.shape[0]):
        positions = np.flatnonzero(x[row] > 0)
        values[row, : positions.size] = x[row, positions]
        indices[row, : positions.size] = positions
    return values, indices, counts
