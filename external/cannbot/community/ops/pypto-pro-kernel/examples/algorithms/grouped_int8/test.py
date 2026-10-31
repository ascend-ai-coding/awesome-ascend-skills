# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Numerics, arbitrary group alignment, tails, bounds, repeated calls and guards."""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import quantize_reference

SHAPES = [
    (7, 65, 32),
    (33, 257, 96),
    (3, 4097, 128),
    (5, 193, 48),
    (2, 513, 160),
    (4, 97, 7),
    (3, 129, 1),
    (2, 8192, 8192),
    (1, 8191, 8191),
]


def guarded(value, device):
    dtype, sentinel = (
        (torch.int8, -103) if value.dtype == np.int8 else (torch.float32, 8192.0)
    )
    backing = torch.full((value.size + 128,), sentinel, dtype=dtype, device=device)
    tensor = backing[64:-64].view(*value.shape)
    tensor.copy_(torch.from_numpy(value).to(device))
    return tensor, backing, sentinel


@pytest.mark.parametrize("rows,cols,group", SHAPES)
@pytest.mark.parametrize("kind", ["random", "ties", "saturation", "zero", "boundary"])
@pytest.mark.parametrize("workers", [1, 3])
def test_grouped_quantize(device, rows, cols, group, kind, workers):
    groups = (cols + group - 1) // group
    rng = np.random.default_rng(rows * cols + group)
    x = rng.uniform(-4, 4, (rows, cols)).astype(np.float32)
    scales = rng.uniform(1, 64, (rows, groups)).astype(np.float32)
    if kind == "ties":
        scales[:] = 32
        x[:] = np.resize(np.arange(-126.5, 127, 1, dtype=np.float32) / 32, x.shape)
    elif kind == "saturation":
        scales[:] = 64
        x[:] = np.resize(np.array([-4, 4, -2, 2], dtype=np.float32), x.shape)
    elif kind == "zero":
        x.fill(0)
    elif kind == "boundary":
        scales[:] = 2 ** (np.arange(rows * groups).reshape(rows, groups) % 7)
        x[:] = np.resize(
            np.array([-1.75, -0.25, 0.25, 1.75], dtype=np.float32), x.shape
        )
    expected = quantize_reference(x, scales, group)
    inputs = [guarded(x, device), guarded(scales, device)]
    output = guarded(np.zeros((rows, cols), dtype=np.int8), device)
    for _ in range(2):
        output[0].fill_(-103)
        run(inputs[0][0], inputs[1][0], output[0], group, workers=workers)
        torch.npu.synchronize()
        np.testing.assert_array_equal(output[0].cpu().numpy(), expected)
        for _, backing, sentinel in inputs + [output]:
            np.testing.assert_array_equal(backing[:64].cpu().numpy(), sentinel)
            np.testing.assert_array_equal(backing[-64:].cpu().numpy(), sentinel)
    for (tensor, _, _), original in zip(inputs, [x, scales]):
        np.testing.assert_array_equal(
            tensor.cpu().numpy().view(np.uint32), original.view(np.uint32)
        )
