# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent math-definition reference for masked softmax."""

import numpy as np


def masked_softmax(x: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Compute each valid prefix in float64; leave the suffix exactly zero."""
    x64 = np.asarray(x, dtype=np.float64)
    lens = np.asarray(lengths, dtype=np.int64)
    result = np.zeros(x64.shape, dtype=np.float64)
    for row, raw_length in enumerate(lens):
        length = int(raw_length)
        if length == 0:
            continue
        valid = x64[row, :length]
        shifted = valid - np.max(valid)
        weights = np.exp(shifted)
        result[row, :length] = weights / np.sum(weights)
    return result


def boundary_lengths(n_cols: int) -> list[int]:
    values = {0, 1, n_cols - 1, n_cols}
    for edge in range(64, n_cols + 1, 64):
        values.update((edge - 1, edge, edge + 1))
    return sorted(v for v in values if 0 <= v <= n_cols)
