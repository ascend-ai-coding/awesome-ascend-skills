# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Stride, storage offset, source immutability and exact copy tests."""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import transpose_reference


@pytest.mark.parametrize("rows,cols", [(5, 7), (17, 65), (33, 129), (9, 66)])
@pytest.mark.parametrize("col_stride", [1, 2, 3])
@pytest.mark.parametrize("workers", [1, 3])
def test_strided_transpose(device, rows, cols, col_stride, workers):
    row_stride, offset = cols * col_stride + 11, 67
    size = offset + (rows - 1) * row_stride + (cols - 1) * col_stride + 65
    backing = np.full(size, 12345, dtype=np.float32)
    x = np.random.default_rng(rows * cols).normal(size=(rows, cols)).astype(np.float32)
    x[:, 0] = -0.0
    x[:, 1] = 0.0
    x[:, 2:4] = np.array([1, 0x80000001], dtype=np.uint32).view(np.float32)
    x[:, -2] = -np.finfo(np.float32).max
    x[:, -1] = np.finfo(np.float32).max
    for row in range(rows):
        for col in range(cols):
            backing[offset + row * row_stride + col * col_stride] = x[row, col]
    expected = transpose_reference(x)
    db = torch.from_numpy(backing).to(device)
    dx = torch.as_strided(
        db, (rows, cols), (row_stride, col_stride), storage_offset=offset
    )
    ob = torch.full((rows * cols + 128,), 8765, device=device, dtype=torch.float32)
    out = ob[64:-64].view(cols, rows)
    for _ in range(2):
        out.fill_(float("nan"))
        run(dx, out, workers=workers)
        torch.npu.synchronize()
        np.testing.assert_array_equal(
            out.cpu().numpy().view(np.uint32), expected.view(np.uint32)
        )
        np.testing.assert_array_equal(
            db.cpu().numpy().view(np.uint32), backing.view(np.uint32)
        )
        np.testing.assert_array_equal(ob[:64].cpu().numpy(), 8765)
        np.testing.assert_array_equal(ob[-64:].cpu().numpy(), 8765)
