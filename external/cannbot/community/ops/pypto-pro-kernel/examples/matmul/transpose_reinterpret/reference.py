# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""转置存放的操作数还原之后，按定义算矩阵乘。"""

import torch


def reference_trans_both(a_t: torch.Tensor, b_t: torch.Tensor) -> torch.Tensor:
    """``a_t`` 存的是 A 的转置，``b_t`` 存的是 B 的转置，各自转回来再乘。"""
    return torch.matmul(a_t.float().t(), b_t.float().t())


def reference_trans_a(a_t: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """只有 ``a_t`` 是转置存放的。"""
    return torch.matmul(a_t.float().t(), b.float())
