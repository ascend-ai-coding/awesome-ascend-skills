# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compact output views, streaming tails, ignored rows, smoothing and guards."""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import cross_entropy_reference


def data(rows, cols, kind):
    rng = np.random.default_rng(rows * cols + 9041)
    x = rng.uniform(-4, 4, (rows, cols)).astype(np.float32)
    lengths = np.array(
        [cols, max(1, cols - 1), min(64, cols), 1, 0, cols // 2], dtype=np.int32
    )
    lengths = np.resize(np.roll(lengths, kind), rows)
    labels = np.array([max(0, int(n) - 1) for n in lengths], dtype=np.int32)
    labels[lengths == 0] = -1
    if kind == 1:
        x[:] = np.linspace(-70, 70, cols, dtype=np.float32)
        labels[1::3] = -1
    elif kind == 2:
        x[:] = np.linspace(-75, 75, rows, dtype=np.float32)[:, None]
        labels[2::3] = -1
    elif kind == 3:
        x.fill(-80)
        for row, length in enumerate(lengths):
            if length:
                x[row, 0 if row % 2 else length - 1] = 80
                labels[row] = length - 1 if row % 2 else 0
        labels[1::3] = -1
    for row, length in enumerate(lengths):
        x[row, length:] = 80
    return x, labels, lengths


def guarded(value, device, head=64):
    dtype = torch.float32 if value.dtype == np.float32 else torch.int32
    backing = torch.full((value.size + head + 64,), 8192, dtype=dtype, device=device)
    tensor = backing[slice(head, head + value.size)].view(*value.shape)
    tensor.copy_(torch.from_numpy(value).to(device))
    return tensor, backing


@pytest.mark.parametrize(
    "rows,cols",
    [(7, 65), (33, 1025), (3, 131073), (1, 1), (5, 4097), (1, 262145), (64, 7)],
)
@pytest.mark.parametrize("kind", range(4))
@pytest.mark.parametrize("smoothing", [0.0, 0.2, 1.0])
@pytest.mark.parametrize("workers", [1, 3])
def test_prefix_cross_entropy(device, rows, cols, kind, smoothing, workers):
    originals = data(rows, cols, kind)
    x, labels, lengths = originals
    inputs = [guarded(value, device) for value in originals]
    loss = guarded(np.zeros(rows, dtype=np.float32), device, head=8 if kind % 2 else 64)
    grad = guarded(np.zeros((rows, cols), dtype=np.float32), device)
    expected_loss, expected_grad = cross_entropy_reference(*originals, smoothing)
    ignored = (labels < 0) | (lengths == 0)
    zero_grad = (np.arange(cols)[None, :] >= lengths[:, None]) | ignored[:, None]
    for poison in [float("nan"), -123.0]:
        loss[0].fill_(poison)
        grad[0].fill_(poison)
        run(*(pair[0] for pair in inputs), loss[0], grad[0], smoothing, workers=workers)
        torch.npu.synchronize()
        got_loss, got_grad = loss[0].cpu().numpy(), grad[0].cpu().numpy()
        np.testing.assert_allclose(got_loss, expected_loss, atol=2e-4, rtol=2e-4)
        np.testing.assert_allclose(got_grad, expected_grad, atol=2e-6, rtol=2e-4)
        np.testing.assert_array_equal(got_loss[ignored], 0)
        np.testing.assert_array_equal(got_grad[zero_grad], 0)
        for tensor, backing in inputs + [loss, grad]:
            np.testing.assert_array_equal(
                backing[: tensor.storage_offset()].cpu().numpy(), 8192
            )
            np.testing.assert_array_equal(backing[-64:].cpu().numpy(), 8192)
        for (tensor, _), original in zip(inputs, originals):
            np.testing.assert_array_equal(
                tensor.cpu().numpy().view(np.uint32), original.view(np.uint32)
            )
