# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Window coordinates, halo boundaries, extrema, scalar reuse and guards."""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import sliding_max_reference

SHAPES = [
    (7, 65, 1),
    (33, 257, 3),
    (3, 65537, 7),
    (1, 1, 3),
    (5, 4097, 17),
    (2, 130, 0),
    (1, 131072, 31),
    (64, 1, 31),
    (2, 31, 31),
    (2, 64, 31),
    (7, 65, 31),
    (7, 65, 0),
]


def guarded(value, device):
    backing = torch.full((value.size + 128,), 8192, dtype=torch.float32, device=device)
    tensor = backing[64:-64].view(*value.shape)
    tensor.copy_(torch.from_numpy(value).to(device))
    return tensor, backing


@pytest.mark.parametrize("rows,cols,radius", SHAPES)
@pytest.mark.parametrize(
    "kind", ["random", "negative", "ascending", "impulse", "constant", "extreme"]
)
@pytest.mark.parametrize("workers", [1, 3])
def test_sliding_max(device, rows, cols, radius, kind, workers):
    rng = np.random.default_rng(rows * cols + radius)
    x = rng.uniform(-4, 4, (rows, cols)).astype(np.float32)
    if kind == "negative":
        x[:] = -np.abs(x) - np.float32(0.1)
    elif kind == "ascending":
        x[:] = np.linspace(-3, 3, cols, dtype=np.float32)
        x[::2] = x[::2, ::-1]
    elif kind == "impulse":
        x.fill(-3)
        for col in [0, 31, 32, 63, 64, 65, 127, 128, 255, 256, 1023, 1024, cols - 1]:
            if col < cols:
                x[:, col] = 3
    elif kind == "constant":
        x[:] = np.linspace(-2, 2, rows, dtype=np.float32)[:, None]
    elif kind == "extreme":
        x.fill(np.finfo(np.float32).min)
        x[:, cols // 2] = np.finfo(np.float32).max
    expected = sliding_max_reference(x, radius)
    inputs = guarded(x, device)
    output = guarded(np.zeros_like(x), device)
    for _ in range(2):
        output[0].fill_(float("nan"))
        run(inputs[0], output[0], radius, workers=workers)
        torch.npu.synchronize()
        np.testing.assert_array_equal(output[0].cpu().numpy(), expected)
        for _, backing in [inputs, output]:
            np.testing.assert_array_equal(backing[:64].cpu().numpy(), 8192)
            np.testing.assert_array_equal(backing[-64:].cpu().numpy(), 8192)
    np.testing.assert_array_equal(
        inputs[0].cpu().numpy().view(np.uint32), x.view(np.uint32)
    )
