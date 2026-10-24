# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Stable field pairing, exact output bits, cardinality boundaries and guards."""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import filter_reference


def guarded(value, device):
    dtype = torch.float32 if value.dtype == np.float32 else torch.int32
    backing = torch.full((value.size + 128,), 1234, dtype=dtype, device=device)
    tensor = backing[64:-64].view(*value.shape)
    tensor.copy_(torch.from_numpy(value).to(device))
    return tensor, backing


@pytest.mark.parametrize("rows,cols", [(3, 65), (33, 130), (7, 4097), (5, 257)])
@pytest.mark.parametrize(
    "kind", ["all", "none", "alternate", "edges", "random", "counts"]
)
@pytest.mark.parametrize("workers", [1, 3])
def test_stable_filter(device, rows, cols, kind, workers):
    rng = np.random.default_rng(4927 + rows * cols)
    x = rng.uniform(-3, 3, (rows, cols)).astype(np.float32)
    if kind == "all":
        x[:] = np.maximum(np.abs(x), np.float32(1e-5))
    elif kind == "none":
        x[:] = -np.abs(x)
        x[:, ::2], x[:, 1::4] = 0.0, -0.0
    elif kind == "alternate":
        x[:] = -1
        x[:, ::2] = 2
    elif kind == "edges":
        x[:] = -1
        selected = sorted(
            {0, min(31, cols - 1), min(63, cols - 1), min(64, cols - 1), cols - 1}
        )
        x[:, selected] = np.linspace(0.25, 2.5, len(selected), dtype=np.float32)
    elif kind == "counts":
        x[:] = -1
        choices = [0, 1, 7, 8, 9, 63, 64, 65, 127, 128]
        for row in range(rows):
            count = min(cols, choices[row % len(choices)])
            positions = rng.choice(cols, size=count, replace=False)
            x[row, positions] = rng.uniform(0.1, 2.9, count).astype(np.float32)
    expected = filter_reference(x)
    inputs = guarded(x, device)
    outputs = [guarded(np.zeros_like(value), device) for value in expected]
    for _ in range(2):
        outputs[0][0].fill_(float("nan"))
        outputs[1][0].fill_(-99)
        outputs[2][0].fill_(-99)
        run(inputs[0], *(pair[0] for pair in outputs), workers=workers)
        torch.npu.synchronize()
        for (tensor, _), ref in zip(outputs, expected):
            np.testing.assert_array_equal(
                tensor.cpu().numpy().view(np.uint32), ref.view(np.uint32)
            )
        for _, backing in [inputs] + outputs:
            np.testing.assert_array_equal(backing[:64].cpu().numpy(), 1234)
            np.testing.assert_array_equal(backing[-64:].cpu().numpy(), 1234)
    np.testing.assert_array_equal(
        inputs[0].cpu().numpy().view(np.uint32), x.view(np.uint32)
    )
