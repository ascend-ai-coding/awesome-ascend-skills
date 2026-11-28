# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""用实际量化输入求 FP64 状态递推，不复制 kernel 的分块或累加顺序。"""

import numpy as np


def reference(q, k, v, g, s0):
    q, k, v, g, state = [np.asarray(value, dtype=np.float64) for value in (q, k, v, g, s0)]
    state = state.copy()
    result = np.empty(q.shape, dtype=np.float64)
    for t in range(q.shape[1]):
        state = g[:, t, None, None] * state + k[:, t, :, None] * v[:, t, None, :]
        result[:, t] = np.einsum("ri,rij->rj", q[:, t], state)
    return result, state
