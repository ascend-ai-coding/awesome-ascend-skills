# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""按有效路由在FP64中直接汇总，不复用设备量化或分区顺序。"""

import numpy as np


def reference(values, ids, weights, t):
    valid = ids >= 0
    contributions = (
        values[valid].astype(np.float64) * weights[valid].astype(np.float64)[:, None]
    )
    result = np.zeros((t, values.shape[1]), dtype=np.float64)
    scale = np.zeros_like(result)
    np.add.at(result, ids[valid], contributions)
    np.add.at(scale, ids[valid], np.abs(contributions))
    return result, scale
