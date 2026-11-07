# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""参考实现：按定义算 ``a @ b + bias``，bias 沿 M 广播。"""

import torch


def matmul_bias_reference(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """``[M, K] @ [K, N] + [1, N]``。

    广播交给 torch，不手写循环——手写会把 kernel 的广播方式抄一遍，
    那样只能验出自己抄得对不对。
    """
    return torch.matmul(a.float(), b.float()) + bias.float()
