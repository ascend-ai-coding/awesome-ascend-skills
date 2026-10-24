# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""参考实现：按定义算 ((Q@K) + X1) @ V + X2。"""

import torch


def chain_reference(q, k, v, x1, x2) -> torch.Tensor:
    mm1 = torch.matmul(q.float(), k.float())
    v1 = mm1 + x1.float()
    return torch.matmul(v1, v.float()) + x2.float()


def mm1_reference(q, k, x1) -> torch.Tensor:
    """第一趟交接之后、第二次矩阵乘之前的中间量。"""
    return torch.matmul(q.float(), k.float()) + x1.float()
