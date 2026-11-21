# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""NN 的参考实现：从矩阵乘的定义算，再把 GM 的存放方向还原。"""

import torch


def reference_nn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``C[i, j] = sum_k A[i, k] * B[k, j]``。a 与 b 就是 A 与 B。"""
    return torch.matmul(a.float(), b.float())
