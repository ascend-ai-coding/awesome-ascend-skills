# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent NumPy definition of the row-wise gather contract."""

import numpy as np


def row_gather(x: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Return x[row, index[row, col]] with negative indices translated by N."""
    if x.ndim != 2 or index.ndim != 2 or x.shape[0] != index.shape[0]:
        raise ValueError("x and index must be rank-2 with equal row counts")
    if x.dtype != np.float32 or index.dtype != np.int32:
        raise TypeError("x must be float32 and index must be int32")
    if not np.isfinite(x).all():
        raise ValueError("x must contain only finite values")
    n = x.shape[1]
    if np.any(index < -n) or np.any(index >= n):
        raise ValueError("indices must be in [-N, N)")
    normalized = np.where(index < 0, index + n, index)
    return np.take_along_axis(x, normalized, axis=1)
