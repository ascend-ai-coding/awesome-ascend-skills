# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""FP64 数学参考，输入已经是实际传入设备的 FP16 量化值。"""

import numpy as np


def gemm_rmsnorm_reference(a, b, bias, gamma, epsilon):
    t = a.astype(np.float64) @ b.astype(np.float64) + bias.astype(np.float64)
    return (
        t
        / np.sqrt(np.mean(t * t, axis=1, keepdims=True) + epsilon)
        * gamma.astype(np.float64)
    )
