# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent float64 references for masked softmax and logsumexp."""

import numpy as np


def reference(x, lengths):
    x64 = np.asarray(x, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.int64)
    y = np.zeros(x64.shape, dtype=np.float64)
    lse = np.full((x64.shape[0], 1), -np.inf, dtype=np.float64)
    for row, length in enumerate(lengths):
        if length == 0:
            continue
        values = x64[row, :length]
        maximum = np.max(values)
        exponents = np.exp(values - maximum)
        denominator = np.sum(exponents, dtype=np.float64)
        y[row, :length] = exponents / denominator
        lse[row, 0] = maximum + np.log(denominator)
    return y, lse
