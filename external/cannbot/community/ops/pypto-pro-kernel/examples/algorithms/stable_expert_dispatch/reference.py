# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""独立 CPU 稳定排序与前缀计数，不复用 kernel 的区间分工。"""

import numpy as np


def stable_dispatch_reference(x, experts, count):
    valid = np.flatnonzero(experts >= 0)
    order = valid[np.argsort(experts[valid], kind="stable")]
    y = np.zeros_like(x)
    y[: len(order)] = x[order]
    permutation = np.full(len(experts), -1, dtype=np.int32)
    permutation[: len(order)] = order
    inverse = np.full(len(experts), -1, dtype=np.int32)
    inverse[order] = np.arange(len(order), dtype=np.int32)
    sizes = np.bincount(experts[valid], minlength=count)
    offsets = np.concatenate(([0], np.cumsum(sizes))).astype(np.int32)
    return y, permutation, inverse, offsets
