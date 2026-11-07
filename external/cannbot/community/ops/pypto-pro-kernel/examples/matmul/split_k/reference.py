# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""C = A @ B，从定义算——归约怎么分块与结果无关。"""

import torch


def matmul_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``C[i, j] = sum_k A[i, k] * B[k, j]``，一次性归约完整个 K。

    **故意不分块**：分块是被测实现的做法，参考实现照着分一遍就只能验出分块被抄
    对了。这里让 torch 一次算完，分块与否对数学结果没有影响。
    """
    return torch.matmul(a.float(), b.float())
