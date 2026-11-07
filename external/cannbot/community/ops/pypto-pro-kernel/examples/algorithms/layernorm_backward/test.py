# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Both reduction axes, supplied statistics, tails, repeated calls and guards."""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import layernorm_backward_reference


def guarded(array, device):
    backing = torch.full((array.size + 128,), 1234, device=device, dtype=torch.float32)
    tensor = backing[64:-64].view(*array.shape)
    tensor.copy_(torch.from_numpy(array).to(device))
    return tensor, backing


@pytest.mark.parametrize("rows,cols", [(7, 65), (33, 257), (257, 1024), (5, 129)])
@pytest.mark.parametrize(
    "kind", ["random", "constant", "zero_dy", "zero_gamma", "epsilon"]
)
@pytest.mark.parametrize("workers", [1, 3])
def test_layernorm_backward(device, rows, cols, kind, workers):
    rng = np.random.default_rng(rows * cols + 4721)
    x = rng.uniform(-2, 2, (rows, cols)).astype(np.float32)
    dy = rng.uniform(-2, 2, (rows, cols)).astype(np.float32)
    gamma = rng.uniform(-1.5, 1.5, cols).astype(np.float32)
    if kind == "constant":
        x[:] = np.linspace(-1.75, 1.75, rows, dtype=np.float32)[:, None]
    elif kind == "zero_dy":
        dy.fill(0)
    elif kind == "zero_gamma":
        gamma.fill(0)
    mean = x.astype(np.float64).mean(axis=1).astype(np.float32)
    # A different valid forward epsilon detects silently recomputed statistics.
    epsilon = 0.125 if kind == "epsilon" else 1e-5
    rstd = (1 / np.sqrt(x.astype(np.float64).var(axis=1) + epsilon)).astype(np.float32)
    originals = [x, dy, gamma, mean, rstd]
    expected = layernorm_backward_reference(*originals)
    inputs = [guarded(value, device) for value in originals]
    outputs = [
        guarded(np.zeros_like(value, dtype=np.float32), device) for value in expected
    ]
    for _ in range(2):
        for tensor, _ in outputs:
            tensor.fill_(float("nan"))
        run(*(pair[0] for pair in inputs + outputs), workers=workers)
        torch.npu.synchronize()
        for name, (tensor, _), ref in zip(["dx", "dgamma", "dbeta"], outputs, expected):
            np.testing.assert_allclose(
                tensor.cpu().numpy(), ref, rtol=2e-4, atol=2e-4, err_msg=name
            )
        for _, backing in inputs + outputs:
            np.testing.assert_array_equal(backing[:64].cpu().numpy(), 1234)
            np.testing.assert_array_equal(backing[-64:].cpu().numpy(), 1234)
    for (tensor, _), original in zip(inputs, originals):
        np.testing.assert_array_equal(
            tensor.cpu().numpy().view(np.uint32), original.view(np.uint32)
        )
