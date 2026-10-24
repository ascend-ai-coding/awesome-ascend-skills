# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent mathematical reference for finite FP32 clipped windows."""

import numpy as np


def sliding_max_reference(x, radius):
    values = np.asarray(x)
    padded = np.pad(values, ((0, 0), (radius, radius)), constant_values=-np.inf)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * radius + 1, axis=1)
    return windows.max(axis=-1)
