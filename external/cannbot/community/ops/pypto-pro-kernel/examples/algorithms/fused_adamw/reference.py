# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""从每次调用前的实际状态求 FP64 更新，当前公式内部不加入额外存储舍入。"""

import numpy as np


def reference(p, g, m, v, lr, b1, b2, eps, wd, step):
    p, g, m, v = [a.astype(np.float64) for a in (p, g, m, v)]
    new_m = b1 * m + (1 - b1) * g
    new_v = b2 * v + (1 - b2) * g * g
    new_p = p * (1 - lr * wd) - lr * (new_m / (1 - b1**step)) / (
        np.sqrt(new_v / (1 - b2**step)) + eps
    )
    return new_p, new_m, new_v
