# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent scalar reference for the scatter-add contract."""

import numpy as np


def scatter_add_reference(base, index, value):
    """Compute y[i] = base[i] + sum(value[j] for normalized index[j] == i)."""
    result = [int(item) for item in base]
    n = len(result)
    for raw_index, contribution in zip(index, value):
        dst = int(raw_index)
        if dst < 0:
            dst += n
        result[dst] += int(contribution)
    return np.asarray(result, dtype=np.int32)
