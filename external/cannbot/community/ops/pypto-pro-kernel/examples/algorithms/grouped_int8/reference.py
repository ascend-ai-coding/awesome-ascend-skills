# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Contract reference: FP32 product, ties-to-even, then symmetric clipping."""

import numpy as np


def quantize_reference(x, inverse_scales, group_size):
    scales = inverse_scales[:, np.arange(x.shape[1]) // group_size]
    product = np.multiply(x, scales, dtype=np.float32)
    return np.clip(np.rint(product), -127, 127).astype(np.int8)
