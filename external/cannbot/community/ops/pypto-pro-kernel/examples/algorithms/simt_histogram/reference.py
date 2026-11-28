# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""独立数学参考：每个字节值出现的次数，不复用设备分块顺序。"""

import numpy as np


def reference(x):
    values = np.asarray(x)
    if values.dtype != np.uint8 or values.ndim != 1 or values.size > 2**31 - 1:
        raise ValueError("require UINT8[N] with N <= INT32_MAX")
    return np.bincount(values, minlength=256).astype(np.int32)
