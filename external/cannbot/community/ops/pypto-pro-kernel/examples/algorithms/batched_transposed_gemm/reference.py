# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""独立的 batched transposed GEMM 数学参考。"""

import torch


def reference(a, w):
    """按 FP16 已量化输入升 FP64 后计算 A @ W^T。"""
    a64 = a.detach().cpu().to(torch.float64)
    w64 = w.detach().cpu().to(torch.float64)
    return torch.matmul(a64, w64.transpose(-1, -2))
