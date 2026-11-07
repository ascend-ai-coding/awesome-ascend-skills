# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""独立 FP64 CSR softmax 加权聚合参考，不复现 kernel 的分块或平移常数。"""

import numpy as np


def reference(logits, offsets, indices, table):
    scores = np.asarray(logits, np.float32).astype(np.float64)
    payload = np.asarray(table, np.float32).astype(np.float64)
    rows = len(offsets) - 1
    out = np.zeros((rows, payload.shape[1]), np.float64)
    lse = np.full(rows, -np.inf, np.float64)
    for row in range(rows):
        begin, end = map(int, offsets[slice(row, row + 2)])
        if begin == end:
            continue
        peak = scores[begin:end].max()
        weights = np.exp(scores[begin:end] - peak)
        denominator = weights.sum()
        out[row] = (weights / denominator) @ payload[indices[begin:end]]
        lse[row] = peak + np.log(denominator)
    return out, lse
