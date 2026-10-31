# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent CPU reference: split segments, then accumulate each in INT64."""

import numpy as np


def segmented_scan_reference(values, heads):
    result = np.empty_like(values)
    for row in range(len(values)):
        starts = np.unique(np.r_[0, np.flatnonzero(heads[row])])
        stops = np.r_[starts[1:], values.shape[1]]
        for start, stop in zip(starts, stops):
            result[row, start:stop] = np.cumsum(values[row, start:stop], dtype=np.int64)
    return result
