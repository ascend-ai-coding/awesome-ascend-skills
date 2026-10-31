# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Broadcast axes, M tails, epilogue semantics, repeated calls and guards."""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import gemm_relu_reference


def guarded(array, device):
    dtype = torch.float16 if array.dtype == np.float16 else torch.float32
    backing = torch.full((array.size + 128,), 1234, device=device, dtype=dtype)
    tensor = backing[64:-64].view(*array.shape)
    tensor.copy_(torch.from_numpy(array).to(device))
    return tensor, backing


@pytest.mark.parametrize("b,m,k,n", [(3, 17, 64, 48), (5, 33, 96, 80), (2, 19, 32, 32)])
@pytest.mark.parametrize("kind", ["random", "batch", "zero", "cancel"])
@pytest.mark.parametrize("workers", [1, 3])
def test_gemm_relu(device, b, m, k, n, kind, workers):
    rng = np.random.default_rng(b * m * k * n)
    a = rng.uniform(-1, 1, (b, m, k)).astype(np.float16)
    w = rng.uniform(-1, 1, (1, k, n)).astype(np.float16)
    bias = np.linspace(-0.5, 0.5, n, dtype=np.float32)
    if kind == "batch":
        for batch in range(b):
            a[batch].fill((batch + 1) / 8)
    elif kind == "zero":
        a.fill(0)
    elif kind == "cancel":
        a.fill(1)
        w[:, ::2, :] = 1
        w[:, 1::2, :] = -1
    expected = gemm_relu_reference(a, w, bias)
    inputs = [guarded(value, device) for value in [a, w, bias]]
    out, backing = guarded(np.zeros((b, m, n), dtype=np.float32), device)
    for _ in range(2):
        out.fill_(float("nan"))
        run(*(pair[0] for pair in inputs), out, workers=workers)
        torch.npu.synchronize()
        np.testing.assert_allclose(out.cpu().numpy(), expected, atol=3e-4, rtol=3e-4)
        for buffer in [pair[1] for pair in inputs] + [backing]:
            np.testing.assert_array_equal(buffer[:64].cpu().numpy(), 1234)
            np.testing.assert_array_equal(buffer[-64:].cpu().numpy(), 1234)
    for (tensor, _), original in zip(inputs, [a, w, bias]):
        np.testing.assert_array_equal(tensor.cpu().numpy(), original)
