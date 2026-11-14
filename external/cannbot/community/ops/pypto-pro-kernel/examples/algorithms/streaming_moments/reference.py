# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""独立 CPU FP64 参考，直接按数学定义计算行均值与总体方差。"""

import numpy as np


def reference(x):
    wide = np.asarray(x, dtype=np.float32).astype(np.float64)
    mean = wide.mean(axis=1)
    centered = wide - mean[:, None]
    return mean, np.mean(centered * centered, axis=1)
