# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""独立 NumPy 数学参考，低精度输入按契约给定的实际值消费。"""

import numpy as np


def unpack(packed):
    result = np.empty((packed.shape[0] * 2, packed.shape[1]), dtype=np.uint8)
    result[0::2] = packed & np.uint8(15)
    result[1::2] = packed >> np.uint8(4)
    return result


def reference(a, packed, scale, zero, bias, group_size):
    q = unpack(packed).astype(np.float32)
    centered = q - np.repeat(zero, group_size, axis=0).astype(np.float32)
    value = centered * np.repeat(scale, group_size, axis=0)
    weight = value.astype(np.float16)
    return a.astype(np.float64) @ weight.astype(np.float64) + bias.astype(np.float64)
