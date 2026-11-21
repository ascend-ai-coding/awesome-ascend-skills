# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""独立 FP64 列均值与总体方差参考，输入先解释为实际 FP32 值。"""

import numpy as np


def column_stats_reference(x):
    """Return column means and population variances in FP64."""
    x64 = np.asarray(x, dtype=np.float32).astype(np.float64)
    mean = x64.mean(axis=0, dtype=np.float64)
    centered = x64 - mean[None, :]
    variance = (centered * centered).mean(axis=0, dtype=np.float64)
    return mean, variance
