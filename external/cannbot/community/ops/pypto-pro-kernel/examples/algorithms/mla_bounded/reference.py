# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""独立 FP64 参考：先恢复 head/query 轴，再求完整数学归一化矩阵链。"""

import torch

from ..mm_softmax_mm_materialized.reference import (
    reference_and_bound as chain_reference,
)


def reference_and_bound(qn, qr, kn, kr, values, layout, causal):
    query = torch.cat((qn, qr), dim=-1)
    if layout == "BSND":
        query = query.permute(0, 2, 1, 3)
    batch, heads, queries, features = query.shape
    keys = kn.shape[1 if layout == "BSND" else 2]
    key = torch.cat((kn, kr), dim=-1).reshape(batch, keys, features)
    value = values.reshape(batch, keys, values.shape[-1])
    lengths = torch.full((batch, heads * queries), keys, dtype=torch.int32)
    if causal:
        lengths[:] = torch.arange(queries).repeat(heads) + keys - queries + 1
    expected, bound = chain_reference(
        query.reshape(batch, heads * queries, features), key, value, lengths
    )
    expected = expected.reshape(batch, heads, queries, -1)
    bound = bound.reshape(batch, heads, queries, -1)
    if layout == "BSND":
        expected = expected.permute(0, 2, 1, 3)
        bound = bound.permute(0, 2, 1, 3)
    return expected, bound
