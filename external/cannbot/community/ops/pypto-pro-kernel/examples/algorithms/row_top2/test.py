# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Stable indices, selected source bits, repeated calls and storage guards."""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import top2_reference


def make_input(rows, cols, kind):
    rng = np.random.default_rng(3271 + rows * cols)
    x = rng.integers(-100, 0, (rows, cols)).astype(np.float32)
    if kind == "equal":
        x.fill(-7.25)
    elif kind == "ties":
        for row in range(rows):
            x[row, [row % cols, min(64, cols - 1), cols - 1]] = 50
    elif kind == "zeros":
        x.fill(-1)
        for row in range(rows):
            x[row, 0] = -0.0 if row % 2 else 0.0
            x[row, 1] = 0.0 if row % 2 else -0.0
            x[row, -1] = -0.0
    elif kind == "extreme":
        x.fill(-np.finfo(np.float32).max)
        x[:, -1] = np.finfo(np.float32).max
    elif kind == "tail":
        x[:, -2:] = np.float32(1234567)
    elif kind != "negative":
        raise ValueError(kind)
    return x


@pytest.mark.parametrize("rows,cols", [(3, 2), (7, 65), (33, 130), (3, 4097), (5, 129)])
@pytest.mark.parametrize(
    "kind", ["equal", "negative", "ties", "zeros", "extreme", "tail"]
)
@pytest.mark.parametrize("workers", [1, 3])
def test_top2(device, rows, cols, kind, workers):
    x = make_input(rows, cols, kind)
    values, indices = top2_reference(x)
    xb = torch.full((x.size + 128,), 12345.0, device=device, dtype=torch.float32)
    vb = torch.full((rows * 2 + 128,), 12345.0, device=device, dtype=torch.float32)
    ib = torch.full((rows * 2 + 128,), 314159, device=device, dtype=torch.int32)
    dx = xb[64:-64].view(rows, cols)
    dv, di = vb[64:-64].view(rows, 2), ib[64:-64].view(rows, 2)
    dx.copy_(torch.from_numpy(x).to(device))
    for _ in range(2):
        dv.fill_(float("nan"))
        di.fill_(-9)
        run(dx, dv, di, workers=workers)
        torch.npu.synchronize()
        np.testing.assert_array_equal(di.cpu().numpy(), indices)
        np.testing.assert_array_equal(
            dv.cpu().numpy().view(np.uint32), values.view(np.uint32)
        )
        for backing, guard in [(xb, 12345.0), (vb, 12345.0), (ib, 314159)]:
            np.testing.assert_array_equal(backing[:64].cpu().numpy(), guard)
            np.testing.assert_array_equal(backing[-64:].cpu().numpy(), guard)
    np.testing.assert_array_equal(dx.cpu().numpy().view(np.uint32), x.view(np.uint32))
