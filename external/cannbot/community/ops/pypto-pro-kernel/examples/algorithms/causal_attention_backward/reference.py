# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""按实际 FP16 输入在 FP64 中计算 causal attention 的三个解析梯度。"""

import numpy as np


def probabilities(q, k):
    scores = (
        q.astype(np.float64)
        @ np.swapaxes(k.astype(np.float64), -1, -2)
        / np.sqrt(q.shape[-1])
    )
    mask = np.tri(q.shape[-2], dtype=bool)
    scores = np.where(mask, scores, -np.inf)
    exps = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)


def reference(q, k, v, dout):
    q, k, v, dout = [a.astype(np.float64) for a in (q, k, v, dout)]
    p = probabilities(q, k)
    dp = dout @ np.swapaxes(v, -1, -2)
    ds = p * (dp - np.sum(p * dp, axis=-1, keepdims=True))
    dq = ds @ k / np.sqrt(q.shape[-1])
    dk = np.swapaxes(ds, -1, -2) @ q / np.sqrt(q.shape[-1])
    dv = np.swapaxes(p, -1, -2) @ dout
    return dq, dk, dv
