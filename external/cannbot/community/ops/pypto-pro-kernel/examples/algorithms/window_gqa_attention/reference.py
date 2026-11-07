# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""独立 NumPy 数学参考，低精度输入按契约给定的实际值消费。"""

import numpy as np


def reference(q, k, v, lengths, positions, window):
    b, hq, m, d = q.shape
    hkv = k.shape[1]
    n = k.shape[2]
    head_map = np.arange(hq) // (hq // hkv)
    scores = np.einsum(
        "bhmd,bhnd->bhmn", q.astype(np.float64), k[:, head_map].astype(np.float64)
    ) / np.sqrt(float(d))
    key = np.arange(n)[None, None, None, :]
    pos = positions[:, None, :, None]
    valid = (key < lengths[:, None, None, None]) & (key <= pos) & (key > pos - window)
    valid = np.broadcast_to(valid, scores.shape)
    nonempty = valid.any(axis=-1)
    masked = np.where(valid, scores, -np.inf)
    maximum = masked.max(axis=-1)
    origin = np.where(nonempty, maximum, 0)
    weights = np.exp(masked - origin[..., None])
    denom = weights.sum(axis=-1)
    normal = np.divide(
        weights,
        denom[..., None],
        out=np.zeros_like(weights),
        where=denom[..., None] > 0,
    )
    out = np.einsum("bhmn,bhnd->bhmd", normal, v[:, head_map].astype(np.float64))
    lse = np.full(maximum.shape, -np.inf, dtype=np.float64)
    lse[nonempty] = maximum[nonempty] + np.log(denom[nonempty])
    return out, lse
