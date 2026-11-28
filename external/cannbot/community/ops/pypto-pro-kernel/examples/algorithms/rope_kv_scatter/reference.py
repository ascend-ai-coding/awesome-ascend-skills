# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""相邻偶奇对 RoPE 与唯一槽位写入的独立 FP64 数学参考。"""

import numpy as np


def rotate(value, cos, sin, positions):
    value = np.asarray(value, dtype=np.float16)
    rotated = value.copy()
    rotary = cos.shape[1] * 2
    pairs = value[..., :rotary].astype(np.float64).reshape(*value.shape[:-1], -1, 2)
    c = np.asarray(cos, dtype=np.float32)[positions].astype(np.float64)[:, :, None, :]
    s = np.asarray(sin, dtype=np.float32)[positions].astype(np.float64)[:, :, None, :]
    rotated[..., :rotary:2] = (pairs[..., 0] * c - pairs[..., 1] * s).astype(np.float16)
    rotated[..., 1:rotary:2] = (pairs[..., 0] * s + pairs[..., 1] * c).astype(
        np.float16
    )
    return rotated


def reference(q, k, v, cos, sin, positions, slots, cache_k, cache_v):
    out = rotate(q, cos, sin, positions)
    rotated_k = rotate(k, cos, sin, positions)
    next_k, next_v = cache_k.copy(), cache_v.copy()
    for batch in range(q.shape[0]):
        for token in range(q.shape[1]):
            slot = int(slots[batch, token])
            if slot >= 0:
                next_k[slot] = rotated_k[batch, token]
                next_v[slot] = v[batch, token]
    return out, next_k, next_v
