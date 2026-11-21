# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""独立数学参考：INT8积和由FP64精确表示，激活与量化不复现分块顺序。"""

import numpy as np


def make_inputs(counts, inner, cols, seed=19):
    """非恒定分组数据，供精度和明确标注分布的性能对照复用。"""
    rng = np.random.default_rng(seed)
    rows, groups = sum(counts), len(counts)
    return [
        rng.integers(-128, 128, (rows, inner), dtype=np.int8),
        rng.integers(-128, 128, (groups, inner, cols), dtype=np.int8),
        rng.uniform(0.001, 0.01, (groups, cols)).astype(np.float32),
        rng.uniform(0.001, 0.01, rows).astype(np.float32),
        np.r_[0, np.cumsum(counts)].astype(np.int32),
    ]


def reference(x, weight, weight_scale, x_scale, offsets):
    x = np.asarray(x, dtype=np.int8).astype(np.float64)
    weight = np.asarray(weight, dtype=np.int8).astype(np.float64)
    ws = np.asarray(weight_scale, dtype=np.float32).astype(np.float64)
    xs = np.asarray(x_scale, dtype=np.float32).astype(np.float64)
    mm = np.empty((x.shape[0], weight.shape[2]), dtype=np.int32)
    dequant = np.empty(mm.shape, dtype=np.float64)
    for group, (begin, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        mm[begin:end] = x[begin:end] @ weight[group]
        dequant[begin:end] = mm[begin:end] * xs[begin:end, None] * ws[group]
    left, right = np.split(dequant, 2, axis=1)
    exponent = np.exp(-np.abs(left))
    sigmoid = np.where(left >= 0, 1.0 / (1.0 + exponent), exponent / (1.0 + exponent))
    act = left * sigmoid * right
    scale = np.maximum(np.max(np.abs(act), axis=1), np.finfo(np.float32).tiny) / 127.0
    y = np.clip(np.rint(act / scale[:, None]), -128, 127).astype(np.int8)
    return mm, y, scale
