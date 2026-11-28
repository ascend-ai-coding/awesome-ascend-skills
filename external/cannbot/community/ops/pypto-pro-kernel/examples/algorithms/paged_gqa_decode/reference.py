# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""按逻辑页表展开的独立 FP64 attention 数学参考。"""

import numpy as np


def reference(query, keys, values, table, lengths):
    query = np.asarray(query, dtype=np.float16).astype(np.float64)
    keys = np.asarray(keys, dtype=np.float16).astype(np.float64)
    values = np.asarray(values, dtype=np.float16).astype(np.float64)
    batch, heads, depth = query.shape
    page_size, kv_heads = keys.shape[1:3]
    ratio = heads // kv_heads
    output = np.zeros((batch, heads, values.shape[-1]), np.float64)
    lse = np.full((batch, heads), -np.inf, np.float64)
    for row, length in enumerate(lengths):
        if not length:
            continue
        logical = np.arange(int(length))
        physical = table[row, logical // page_size]
        slots = logical % page_size
        for head in range(kv_heads):
            lo, hi = head * ratio, (head + 1) * ratio
            scores = query[row, lo:hi] @ keys[physical, slots, head].T / np.sqrt(depth)
            peak = scores.max(axis=1, keepdims=True)
            weights = np.exp(scores - peak)
            denominator = weights.sum(axis=1, keepdims=True)
            output[row, lo:hi] = (weights / denominator) @ values[physical, slots, head]
            lse[row, lo:hi] = (peak + np.log(denominator))[:, 0]
    return output, lse
