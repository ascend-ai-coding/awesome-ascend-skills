# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exact state propagation, input immutability, repeated calls and guard tests."""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import segmented_scan_reference


@pytest.mark.parametrize("rows,cols", [(3, 65), (67, 257), (7, 4097), (5, 129)])
@pytest.mark.parametrize("kind", ["none", "all", "sparse", "boundaries"])
@pytest.mark.parametrize("workers", [1, 3])
def test_scan(device, rows, cols, kind, workers):
    rng = np.random.default_rng(29381 + rows * cols)
    values = rng.integers(-4, 5, size=(rows, cols), dtype=np.int32)
    heads = np.zeros_like(values)
    if kind == "none":
        values.fill(1)
    elif kind == "all":
        heads.fill(1)
    elif kind == "sparse":
        heads[:] = (rng.random((rows, cols)) < 0.02).astype(np.int32)
    else:
        for position in [63, 64, 65, 127, 129, 255, 256, 4095, 4096]:
            if position < cols:
                heads[:, position] = 1
    expected = segmented_scan_reference(values, heads)
    backings = [
        torch.full((rows * cols + 128,), 314159, dtype=torch.int32, device=device)
        for _ in range(3)
    ]
    dx, dh, out = [buffer[64:-64].view(rows, cols) for buffer in backings]
    dx.copy_(torch.from_numpy(values).to(device))
    dh.copy_(torch.from_numpy(heads).to(device))
    for _ in range(2):
        out.fill_(-999999)
        run(dx, dh, out, workers=workers)
        torch.npu.synchronize()
        np.testing.assert_array_equal(out.cpu().numpy(), expected)
        for buffer in backings:
            np.testing.assert_array_equal(buffer[:64].cpu().numpy(), 314159)
            np.testing.assert_array_equal(buffer[-64:].cpu().numpy(), 314159)
    np.testing.assert_array_equal(dx.cpu().numpy(), values)
    np.testing.assert_array_equal(dh.cpu().numpy(), heads)
