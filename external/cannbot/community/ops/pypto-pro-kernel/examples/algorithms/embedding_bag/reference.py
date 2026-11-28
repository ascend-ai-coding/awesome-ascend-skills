# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent FP64 reduction over each logical segment."""

import numpy as np


def embedding_bag_reference(table, indices, offsets):
    result = np.zeros((offsets.size - 1, table.shape[1]), dtype=np.float64)
    for bag in range(offsets.size - 1):
        selected = indices[slice(offsets[bag], offsets[bag + 1])]
        selected = selected[selected >= 0]
        result[bag] = table[selected].astype(np.float64).sum(axis=0)
    return result
