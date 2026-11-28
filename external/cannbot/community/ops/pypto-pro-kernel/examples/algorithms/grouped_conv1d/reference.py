# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent CPU FP64 reference for grouped 1-D dilated cross-correlation."""

import numpy as np


def reference(x, w, bias, stride=1, dilation=1):
    """Compute the contract in FP64 from the actual quantized input arrays."""
    x64 = np.asarray(x, dtype=np.float64)
    w64 = np.asarray(w, dtype=np.float64)
    bias64 = np.asarray(bias, dtype=np.float64)
    batch, length, cin = x64.shape
    groups, taps, cin_group, cout_group = w64.shape
    if taps != 3 or cin != groups * cin_group:
        raise ValueError("incompatible input and weight shapes")
    cout = groups * cout_group
    out_length = (length + stride - 1) // stride
    result = np.broadcast_to(bias64, (batch, out_length, cout)).copy()
    output_positions = np.arange(out_length, dtype=np.int64) * stride

    for group in range(groups):
        in_start = group * cin_group
        in_stop = in_start + cin_group
        out_start = group * cout_group
        out_stop = out_start + cout_group
        for tap in range(3):
            source_positions = output_positions + (tap - 1) * dilation
            valid = (source_positions >= 0) & (source_positions < length)
            patch = np.zeros((batch, out_length, cin_group), dtype=np.float64)
            patch[:, valid, :] = x64[:, source_positions[valid], in_start:in_stop]
            result[:, :, out_start:out_stop] += np.einsum(
                "btk,ko->bto", patch, w64[group, tap], optimize=False
            )
    return result
