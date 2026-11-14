# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""参考实现：按定义算，不复现分块。

两个 dtype 的容差不同，原因在输入的舍入而不在算法：BF16 的尾数是 7 位显式位，
FP16 是 10 位，所以同一个数转过去的相对误差差约 8 倍。累加都在 FP32 上做。
"""

import torch

TOLERANCE = {
    torch.float16: dict(rtol=1e-2, atol=1e-2),
    torch.bfloat16: dict(rtol=5e-2, atol=5e-2),
}


def matmul_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.float() @ b.float()
