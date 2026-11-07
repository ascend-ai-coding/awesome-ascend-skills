# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU stable ordering with a source gather to preserve the chosen bits."""

import numpy as np


def top2_reference(x):
    indices = np.array(
        [
            sorted(range(x.shape[1]), key=lambda col: (-float(row[col]), col))[:2]
            for row in x
        ],
        dtype=np.int32,
    )
    values = np.take_along_axis(x, indices, axis=1)
    return values, indices
