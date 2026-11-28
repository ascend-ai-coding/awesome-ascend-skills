# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""参考实现：``O = softmax(A @ B, dim=-1) @ C``，CPU、FP32，从数学定义写起。

不复现 kernel 的分核/分带/splitk/DualModeSplitM 顺序——那样只能验出"抄对了分块"，
验不出分块归约等于不分块归约。``A``/``B``/``C`` 本来就是 FP16（算子的真实输入精度），
这里上库到 FP32 做乘加，匹配硬件"FP16 输入、FP32 累加"的计算精度，不是又一次量化步骤。

``softmax`` 的 P（概率）在 kernel 里被转成 FP16 才喂给第二个 matmul（cube 的输入精度
约束，见 kernel.py 的 ``_softmax_cast_band``）——这是一处真实的精度损失，参考必须吃同一
个量化后的值，不能吃未量化的 FP32 概率，否则参考和 kernel 算的不是同一件事，误差会被
错误地记成"kernel 算错"（references/tooling/examples.md「低精度 dtype 的参考必须吃量化后的值」）。
"""

import torch


def mm_softmax_mm_reference(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """``softmax(A @ B, dim=-1) @ C``：``[M,K]@[K,N]`` -> softmax 沿最后一维 -> ``@[N,N]``。

    softmax 数值稳定形式（减行最大值）由 ``torch.softmax`` 内部保证，不手写。
    """
    s = torch.matmul(a.float(), b.float())
    p = torch.softmax(s, dim=-1)
    p16 = p.to(torch.float16).float()  # kernel 实际喂给第二个 gemm 的精度
    return torch.matmul(p16, c.float())
