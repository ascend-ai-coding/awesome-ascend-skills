# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent CPU reference for O = fp16(rmsnorm(X, gamma)) @ W.

RMSNorm uses float64 intermediates and one explicit FP16 conversion before GEMM, matching the kernel's quantization
boundary. The CPU matmul consumes those FP16 values in FP32. The reference does not reproduce core/band/split-K
partitioning or VF packing; the tail example uses a separate float64 matmul reference.
"""

import torch

EPS = 1e-6  # 必须与 kernel.py 的编译期常量一致


def rmsnorm_gemm_reference(
    x: torch.Tensor, gamma: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    """``x``: ``[rows, K]``；``gamma``: ``[K]`` 或可广播到 ``[1, K]``；``w``: ``[K, N]``。

    返回 ``[rows, N]`` float32：``rmsnorm(x, gamma) @ w``，rmsnorm 的量化到 FP16
    的结果就是 kernel 实际喂给 gemm 的值。
    """
    x64 = x.double()
    gamma64 = gamma.double().reshape(1, -1)
    mean_sq = (x64 * x64).mean(dim=-1, keepdim=True)
    rms = torch.sqrt(mean_sq + EPS)
    normed = (x64 / rms) * gamma64
    normed_fp16 = normed.to(torch.float16)  # kernel 实际喂给 gemm 的精度
    out = torch.matmul(normed_fp16.float(), w.float())
    return out.float()
