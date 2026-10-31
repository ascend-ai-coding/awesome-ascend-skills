# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exact counts, concentrated bins, overflow boundaries, empty input and guards."""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import histogram_reference

CASES = [(0, "random")] + [
    (size, kind)
    for size in [65, 4097, 131101, 65537]
    for kind in ["random", "zero", "high", "edges", "cycle"]
]


def guarded(value, device):
    dtype, sentinel = (
        (torch.uint8, 201) if value.dtype == np.uint8 else (torch.int32, 9876543)
    )
    backing = torch.full((value.size + 128,), sentinel, dtype=dtype, device=device)
    tensor = backing[64:-64].view(*value.shape)
    tensor.copy_(torch.from_numpy(value).to(device))
    return tensor, backing, sentinel


@pytest.mark.parametrize("size,kind", CASES)
@pytest.mark.parametrize("workers", [1, 3])
def test_byte_histogram(device, size, kind, workers):
    rng = np.random.default_rng(size + 2719)
    x = rng.integers(0, 256, size, dtype=np.uint8)
    if kind == "zero":
        x.fill(0)
    elif kind == "high":
        x.fill(255)
    elif kind == "edges":
        x[:] = np.resize(np.array([0, 127, 128, 255], dtype=np.uint8), size)
    elif kind == "cycle":
        x[:] = (np.arange(size, dtype=np.int64) % 256).astype(np.uint8)
    expected = histogram_reference(x)
    inputs = guarded(x, device)
    outputs = guarded(np.zeros(256, dtype=np.int32), device)
    for _ in range(2):
        outputs[0].fill_(-123)
        run(inputs[0], outputs[0], workers=workers)
        torch.npu.synchronize()
        np.testing.assert_array_equal(outputs[0].cpu().numpy(), expected)
        for _, backing, sentinel in [inputs, outputs]:
            np.testing.assert_array_equal(backing[:64].cpu().numpy(), sentinel)
            np.testing.assert_array_equal(backing[-64:].cpu().numpy(), sentinel)
    np.testing.assert_array_equal(inputs[0].cpu().numpy(), x)
