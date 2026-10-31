# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""固定数学契约下的边界、运行时元数据、重复写出、输入不变与守卫验证。"""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import reference


SHAPES = [
    (1, 4, 1, 9, 65, 32, 48, 33),
    (2, 4, 2, 17, 257, 64, 32, 129),
    (1, 8, 2, 33, 513, 32, 64, 257),
    (1, 1, 1, 1, 1, 16, 16, 1),
    (2, 2, 1, 7, 33, 16, 64, 1),
    (2, 8, 2, 33, 513, 64, 48, 513),
]
KINDS = ["random", "uniform", "head", "empty", "windows", "dominant"]


def data(b, hq, hkv, m, n, d, dv, window, kind):
    rng = np.random.default_rng(b * 701 + hq * 101 + m * 19 + n)
    q = rng.uniform(-2, 2, (b, hq, m, d)).astype(np.float16)
    k = rng.uniform(-2, 2, (b, hkv, n, d)).astype(np.float16)
    v = rng.uniform(-2, 2, (b, hkv, n, dv)).astype(np.float16)
    lengths = np.resize(np.array([n, max(1, n - 3)], dtype=np.int32), b).copy()
    positions = np.resize(
        np.array([-1, 0, 1, n // 2, n - 1, window - 1, window], dtype=np.int32), (b, m)
    ).copy()
    positions = np.clip(positions, -1, n - 1)
    if m == 1:
        positions[:, 0] = 0
    if kind == "uniform":
        q[:] = 0
        k[:] = 0
    elif kind == "head":
        q[:] = 0
        k[:] = 0
        for batch in range(b):
            for head in range(hkv):
                v[batch, head] = np.float16(0.25 * (1 + batch * hkv + head))
    elif kind == "empty":
        lengths[:] = 0
    elif kind == "windows":
        lengths[:] = np.resize(np.array([1, n // 2], dtype=np.int32), b)
        positions[:] = np.resize(
            np.array([n - 1, 0, -1, n // 2, window - 1], dtype=np.int32), (b, m)
        )
        positions[:] = np.clip(positions, -1, n - 1)
    elif kind == "dominant":
        q[:] = 2
        k[:] = -2
        k[:, :, 0, :] = 2
        k[:, :, -1, :] = 2
        v[:, :, 0, :] = 1.5
        v[:, :, -1, :] = -1.5
    return q, k, v, lengths, positions


def guarded(value, device, head=64):
    types = {
        np.dtype("float16"): torch.float16,
        np.dtype("float32"): torch.float32,
        np.dtype("int32"): torch.int32,
    }
    backing = torch.full(
        (head + value.size + 64,), 8192, dtype=types[value.dtype], device=device
    )
    tensor = backing[slice(head, head + value.size)].view(*value.shape)
    tensor.copy_(torch.from_numpy(value).to(device))
    return tensor, backing


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("workers", [1, 3, 28])
def test_contract(device, shape, kind, workers):
    b, hq, hkv, m, n, d, dv, window = shape
    host = data(*shape, kind)
    expected = reference(*host, window)
    inputs = [guarded(value, device) for value in host]
    outputs = [
        guarded(np.empty(value.shape, dtype=np.float32), device, head=8)
        for value in expected
    ]
    empty = np.isneginf(expected[1])
    for poison in (float("nan"), -12345.0):
        for tensor, _ in outputs:
            tensor.fill_(poison)
        run(
            *(item[0] for item in inputs),
            *(item[0] for item in outputs),
            window,
            workers=workers,
        )
        torch.npu.synchronize()
        actual = [tensor.cpu().numpy() for tensor, _ in outputs]
        for got, want in zip(actual, expected):
            np.testing.assert_allclose(got, want, atol=2e-3, rtol=2e-3)
        np.testing.assert_array_equal(actual[0][empty], 0)
        np.testing.assert_array_equal(np.isneginf(actual[1]), empty)
        for (tensor, _), value in zip(inputs, host):
            np.testing.assert_array_equal(
                tensor.cpu().numpy().view(np.uint8), value.view(np.uint8)
            )
        for tensor, backing in inputs + outputs:
            np.testing.assert_array_equal(
                backing[: tensor.storage_offset()].cpu().numpy(), 8192
            )
            np.testing.assert_array_equal(backing[-64:].cpu().numpy(), 8192)
