# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Stable FP64 mathematical reference; no kernel blocking or summation order."""

import numpy as np


def cross_entropy_reference(x, labels, lengths, smoothing):
    loss = np.zeros(x.shape[0], dtype=np.float64)
    grad = np.zeros(x.shape, dtype=np.float64)
    for row, (label, length) in enumerate(zip(labels, lengths)):
        if label < 0 or length == 0:
            continue
        values = x[row, :length].astype(np.float64)
        shifted = values - values.max()
        log_prob = shifted - np.log(np.exp(shifted).sum())
        target = np.full(length, smoothing / length, dtype=np.float64)
        target[label] += 1 - smoothing
        loss[row] = -np.dot(target, log_prob)
        grad[row, :length] = np.exp(log_prob) - target
    return loss, grad
