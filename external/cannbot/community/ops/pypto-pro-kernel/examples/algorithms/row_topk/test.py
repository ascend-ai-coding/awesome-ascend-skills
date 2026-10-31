# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Full finite FP32 order, exact payload bits, tails, repeated calls and guards."""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import stable_topk_reference


def data(rows, cols, kind):
    rng = np.random.default_rng(rows * cols + 5327)
    x = rng.uniform(-100, 100, (rows, cols)).astype(np.float32)
    if kind == "negative":
        x = -np.abs(x) - np.float32(0.25)
    elif kind == "constant":
        x[:] = np.linspace(-7, 7, rows, dtype=np.float32)[:, None]
    elif kind == "ties":
        x = rng.integers(-4, 5, (rows, cols)).astype(np.float32)
        for col in [
            0,
            63,
            64,
            127,
            128,
            4095,
            4096,
            8191,
            8192,
            32767,
            32768,
            cols - 1,
        ]:
            if col < cols:
                x[:, col] = 16
    elif kind == "zeros":
        bits = np.resize(np.array([0x80000000, 0], dtype=np.uint32), (rows, cols))
        x = bits.view(np.float32).copy()
    elif kind == "ulp":
        patterns = np.array(
            [
                0x3F7FFFFF,
                0x3F800000,
                0x3F800001,
                0x3F800002,
                0xBF7FFFFF,
                0xBF800000,
                0xBF800001,
            ],
            dtype=np.uint32,
        )
        x = np.resize(patterns, (rows, cols)).view(np.float32).copy()
    elif kind == "subnormal":
        patterns = np.array(
            [
                1,
                2,
                0x7FFFFF,
                0x800000,
                0x80000001,
                0x807FFFFF,
                0,
                0x80000000,
                3,
                0x80000002,
            ],
            dtype=np.uint32,
        )
        x = np.resize(patterns, (rows, cols)).view(np.float32).copy()
    elif kind == "extreme":
        patterns = np.array(
            [
                0xFF7FFFFF,
                0xFF7FFFFE,
                0x7F7FFFFF,
                0x7F7FFFFE,
                0x80800000,
                0x800000,
                0x3F800000,
                0xBF800000,
            ],
            dtype=np.uint32,
        )
        x = np.resize(patterns, (rows, cols)).view(np.float32).copy()
    return x


def guarded(value, device, head=64):
    dtype = torch.float32 if value.dtype == np.float32 else torch.int32
    backing = torch.full((value.size + head + 64,), 8192, dtype=dtype, device=device)
    tensor = backing[slice(head, head + value.size)].view(*value.shape)
    tensor.copy_(torch.from_numpy(value).to(device))
    return tensor, backing


@pytest.mark.parametrize(
    "rows,cols,k",
    [
        (7, 65, 7),
        (33, 1025, 16),
        (3, 65537, 9),
        (1, 1, 1),
        (5, 129, 3),
        (2, 131073, 16),
        (64, 17, 16),
    ],
)
@pytest.mark.parametrize(
    "kind",
    ["random", "negative", "constant", "ties", "zeros", "ulp", "subnormal", "extreme"],
)
@pytest.mark.parametrize("workers", [1, 3])
def test_stable_topk(device, rows, cols, k, kind, workers):
    x = data(rows, cols, kind)
    expected_values, expected_indices = stable_topk_reference(x, k)
    inputs = guarded(x, device)
    values = guarded(
        np.zeros((rows, k), dtype=np.float32), device, head=8 if workers == 1 else 64
    )
    indices = guarded(
        np.zeros((rows, k), dtype=np.int32), device, head=64 if workers == 1 else 8
    )
    for poison in [float("nan"), 12345.0]:
        values[0].fill_(poison)
        indices[0].fill_(-777)
        run(inputs[0], values[0], indices[0], k, workers=workers)
        torch.npu.synchronize()
        actual_values = values[0].cpu().numpy()
        actual_indices = indices[0].cpu().numpy()
        np.testing.assert_array_equal(actual_indices, expected_indices)
        np.testing.assert_array_equal(
            actual_values.view(np.uint32), expected_values.view(np.uint32)
        )
        for row in actual_indices:
            assert len(np.unique(row)) == k
        for tensor, backing in [inputs, values, indices]:
            np.testing.assert_array_equal(
                backing[: tensor.storage_offset()].cpu().numpy(), 8192
            )
            np.testing.assert_array_equal(backing[-64:].cpu().numpy(), 8192)
        np.testing.assert_array_equal(
            inputs[0].cpu().numpy().view(np.uint32), x.view(np.uint32)
        )


@pytest.mark.parametrize("k", [2, 8, 16])
@pytest.mark.parametrize("workers", [1, 3])
def test_displaced_tie_survives_local_pruning(device, k, workers):
    # Fill one lane's capacity with ties, then displace them with a stronger key.
    # Strict value-only insertion used to evict index 0 in favor of later ties.
    x = np.full((2, 64 * k + 1), -1, dtype=np.float32)
    x[:, slice(None, 64 * k, 64)] = 1
    x[:, 64 * k] = 2
    expected_values, expected_indices = stable_topk_reference(x, k)
    dx = torch.from_numpy(x).to(device)
    values = torch.empty((2, k), dtype=torch.float32, device=device)
    indices = torch.empty((2, k), dtype=torch.int32, device=device)
    run(dx, values, indices, k, workers=workers)
    torch.npu.synchronize()
    np.testing.assert_array_equal(indices.cpu().numpy(), expected_indices)
    np.testing.assert_array_equal(
        values.cpu().numpy().view(np.uint32), expected_values.view(np.uint32)
    )
