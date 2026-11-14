# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""FP64 mathematical reference and an a-priori, input-dependent error budget."""

import math

import torch


def reference_and_bound(x, y, values, lengths):
    # Reference has no candidate-dependent probability rounding or tiling.
    x64, y64, v64 = x.double(), y.double(), values.double()
    k, n = x.shape[-1], y.shape[-2]
    scale = 1.0 / math.sqrt(k)
    score = (x64 @ y64.transpose(-1, -2)) * scale
    valid = torch.arange(n)[None, None, :] < lengths[..., None]
    probabilities = torch.softmax(score.masked_fill(~valid, -float("inf")), dim=-1)
    probabilities = torch.where(
        lengths[..., None] > 0, probabilities, torch.zeros_like(probabilities)
    )
    expected = probabilities @ v64
    u32 = torch.finfo(torch.float32).eps / 2
    storage = torch.finfo(x.dtype)
    u = storage.eps / 2
    subnormal_half = storage.tiny * storage.eps / 2

    def gamma(count):
        return count * u32 / (1 - count * u32)

    score_error = gamma(k) * (x64.abs() @ y64.abs().transpose(-1, -2)) * scale
    score_error += u32 * score.abs()
    row_error = score_error.amax(-1, keepdim=True)
    # Softmax perturbation plus FP32 exp/reduction/division budget. The input
    # domain keeps shifted exponential arguments in [-64, 0]. This is a test
    # budget, not a hardware guarantee for exp outside that domain.
    weight_error = torch.expm1(2 * row_error) + 4 * gamma(n) + 16 * u32
    weight_error += u * (1 + weight_error) + n * subnormal_half
    vmax = v64.abs().amax(-2, keepdim=True)
    magnitude = probabilities @ v64.abs()
    error = weight_error * vmax + gamma(n) * (magnitude + weight_error * vmax)
    bound = error + u * (expected.abs() + error) + subnormal_half
    return expected, bound
