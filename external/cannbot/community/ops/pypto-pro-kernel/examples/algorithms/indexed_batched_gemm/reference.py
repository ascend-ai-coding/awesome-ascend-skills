# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent FP64 matrix product from the original table and selectors."""

import torch


def reference_and_bound(x, table, indices):
    b, _, k = x.shape
    selected = table[torch.arange(b)[:, None], indices.long()].double()
    query = x.double()
    expected = query @ selected.transpose(-1, -2)
    magnitude = query.abs() @ selected.abs().transpose(-1, -2)
    # FP16/BF16 products fit FP32 significands. Bound additions and partial
    # accumulation with gamma(2K); include the FP32 underflow/flush allowance.
    u = torch.finfo(torch.float32).eps / 2
    gamma = (2 * k * u) / (1 - 2 * k * u)
    bound = gamma * magnitude + 2 * k * torch.finfo(torch.float32).tiny
    return expected, bound
