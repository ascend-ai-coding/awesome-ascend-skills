# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""三个一起的参考实现：按有效范围截取操作数，再按定义算。"""

import torch

EM, EN, EK = 72, 88, 104


def reference_mnk(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """返回**有效范围内**那片结果，形状 ``[EM, EN]``。

    范围之外的那部分由 ``test.py`` 单独断言（要求一个非零都没有），不在这里表达。
    """
    return torch.matmul(a[:EM, :EK].float(), b[:EK, :EN].float())
