# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""参考实现：``silu(A @ B + bias)``，CPU、FP32，从数学定义写起。

不复现 kernel 的 splitk/分块顺序——那样只能验出"抄对了分块"，验不出分块归约等于
不分块归约。``A``/``B`` 本来就是 FP16（算子的真实输入精度，不是待量化的高精度值）：
这里只是上库到 FP32 做乘加，match 硬件"FP16 输入、FP32 累加"的计算精度，不是又一次
量化步骤。``bias`` 已经是 FP32，直接用。
"""

import torch


def gemm_bias_silu_reference(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """``silu(A @ B + bias)``：``[M,K]@[K,N] + [1,N]``（沿 M 广播），逐元素 ``x*sigmoid(x)``。

    广播交给 torch，不手写循环。``silu`` 按任务给定的公式显式展开（``x * sigmoid(x)``），
    不借 ``torch.nn.functional.silu`` 的黑盒实现，保持这份参考自解释。
    """
    a_cpu = a.cpu().float()
    b_cpu = b.cpu().float()
    bias_cpu = bias.cpu().float()
    pre_act = torch.matmul(a_cpu, b_cpu) + bias_cpu
    return pre_act * torch.sigmoid(pre_act)
