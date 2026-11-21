# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""验证 GEMM 后行归一化，覆盖形状尾块、抵消和 epsilon 边界。

连续 GM 输入和偏移输出使用前后守卫；两个任务的同 Stream 顺序保护中间 out，毒化后重复调用检查完整覆盖。
独立 FP64 参考从实际 FP16 输入计算，固定逐元素容差，并逐位检查输入未改。
"""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import gemm_rmsnorm_reference


def data(rows, inner, cols, kind):
    rng = np.random.default_rng(rows * cols + inner)
    a = rng.uniform(-1, 1, (rows, inner)).astype(np.float16)
    b = rng.uniform(-1, 1, (inner, cols)).astype(np.float16)
    bias = rng.uniform(-1, 1, cols).astype(np.float32)
    gamma = rng.uniform(-1, 1, cols).astype(np.float32)
    if kind == "zero":
        a[:] = 0
        b[:] = 0
        bias[:] = 0
    elif kind == "cancel":
        a[:, 1::2] = -a[:, ::2]
        b[1::2] = b[::2]
        bias[:] = np.linspace(-0.01, 0.01, cols, dtype=np.float32)
    elif kind == "structured":
        a[:] = (np.arange(rows)[:, None] % 3 - 1).astype(np.float16)
        b[:] = (np.arange(cols)[None, :] % 5 - 2).astype(np.float16) / np.float16(2)
        gamma[::3] = 0
    return a, b, bias, gamma


def guarded(value, device, head=64):
    dtype = torch.float16 if value.dtype == np.float16 else torch.float32
    backing = torch.full((value.size + head + 64,), 8192, dtype=dtype, device=device)
    tensor = backing[slice(head, head + value.size)].view(*value.shape)
    tensor.copy_(torch.from_numpy(value).to(device))
    return tensor, backing


@pytest.mark.parametrize(
    "rows,inner,cols",
    [
        (17, 96, 80),
        (65, 160, 272),
        (129, 512, 1040),
        (1, 16, 16),
        (31, 48, 144),
        (257, 32, 2048),
        (257, 512, 2048),
    ],
)
@pytest.mark.parametrize("kind", ["random", "zero", "cancel", "structured"])
@pytest.mark.parametrize("epsilon,workers", [(1e-5, 1), (1e-2, 3)])
def test_gemm_rmsnorm(device, rows, inner, cols, kind, epsilon, workers):
    host = data(rows, inner, cols, kind)
    expected = gemm_rmsnorm_reference(*host, epsilon)
    inputs = [guarded(value, device) for value in host]
    output = guarded(np.empty((rows, cols), dtype=np.float32), device, head=8)
    for poison in [float("nan"), 12345.0]:
        output[0].fill_(poison)
        run(*(tensor for tensor, _ in inputs), output[0], epsilon, workers=workers)
        torch.npu.synchronize()
        np.testing.assert_allclose(
            output[0].cpu().numpy(), expected, atol=2e-3, rtol=2e-3
        )
        for (tensor, _), value in zip(inputs, host):
            bits = np.uint16 if value.dtype == np.float16 else np.uint32
            np.testing.assert_array_equal(
                tensor.cpu().numpy().view(bits), value.view(bits)
            )
        for tensor, backing in inputs + [output]:
            np.testing.assert_array_equal(
                backing[: tensor.storage_offset()].cpu().numpy(), 8192
            )
            np.testing.assert_array_equal(backing[-64:].cpu().numpy(), 8192)
