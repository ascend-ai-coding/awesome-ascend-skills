# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent CPU FP32 reference for the SwiGLU chain.

Two projections form Z=silu(X@W1)*(X@W3), which narrows to FP16 before the final GEMM. The reference does not reproduce
group/band/split-K scheduling. cancellation_scale returns abs(Z16) @ abs(W2), the term-sum scale used by the numerical
comparator.
"""

import torch


def swiglu_ffn_reference(
    x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor, w2: torch.Tensor
) -> torch.Tensor:
    """``(silu(X @ W1) * (X @ W3)) @ W2``：两个 ``[M,K]@[K,N]`` -> 逐元素 silu 与乘 ->
    量化到 FP16（kernel 实际喂给第三个 gemm 的精度）-> ``@[N,N]``。

    ``silu(x) = x / (1 + exp(-x))``，与 kernel 的 VF 配方（``neg -> exp -> adds(1) ->
    div``）算的是同一个数学定义，这里直接用 torch 算子表达，不必手写同样的四步。
    """
    y1 = torch.matmul(x.float(), w1.float())
    y3 = torch.matmul(x.float(), w3.float())
    silu = y1 / (1.0 + torch.exp(-y1))
    z = silu * y3
    z16 = z.to(torch.float16).float()  # kernel 实际喂给第三个 gemm 的精度
    return torch.matmul(z16, w2.float())


def cancellation_scale(
    x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor, w2: torch.Tensor
) -> torch.Tensor:
    """Return abs(Z16) @ abs(W2) for the term-sum error criterion."""
    y1 = torch.matmul(x.float(), w1.float())
    y3 = torch.matmul(x.float(), w3.float())
    silu = y1 / (1.0 + torch.exp(-y1))
    z16 = (silu * y3).to(torch.float16).float()
    return torch.matmul(z16.abs(), w2.float().abs())
