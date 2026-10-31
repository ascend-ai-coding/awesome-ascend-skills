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
    (1, 5, 7, 17, 37),
    (2, 17, 19, 65, 257),
    (2, 65, 97, 129, 1025),
    (1, 1, 1, 1, 1),
    (1, 3, 5, 3, 33),
    (4, 129, 193, 129, 4097),
]
KINDS = ["random", "interior", "integer", "edge", "outside", "repeated"]


def data(b, h, w, c, p, kind):
    rng = np.random.default_rng(b * 101 + h * 907 + w * 31 + c + p)
    x = rng.uniform(-1, 1, (b, h, w, c)).astype(np.float32)
    grid = rng.uniform((-2 * w, -2 * h), (2 * w, 2 * h), (b, p, 2)).astype(np.float32)
    grad = rng.uniform(-1, 1, (b, p, c)).astype(np.float32)
    if kind == "interior":
        grid[..., 0] = rng.uniform(0, max(0.1, w - 1), (b, p))
        grid[..., 1] = rng.uniform(0, max(0.1, h - 1), (b, p))
    elif kind == "integer":
        grid[..., 0] = rng.integers(-1, w + 1, (b, p))
        grid[..., 1] = rng.integers(-1, h + 1, (b, p))
    elif kind == "edge":
        points = np.array(
            [
                [-0.25, -0.25],
                [0, 0],
                [w - 1, h - 1],
                [w - 0.75, h - 0.75],
                [-1, 0],
                [0, -1],
                [w, 0],
                [0, h],
                [w - 1.25, 0.25],
            ],
            dtype=np.float32,
        )
        grid[:] = np.resize(points, grid.shape)
    elif kind == "outside":
        grid[..., 0] = -1.5
        grid[..., 1] = -1.5
    elif kind == "repeated":
        grid[..., 0] = max(0.0, w // 2 - 0.375)
        grid[..., 1] = max(0.0, h // 2 - 0.25)
    return x, grid, grad


def guarded(value, device, head=64):
    backing = torch.full(
        (head + value.size + 64,), 8192, dtype=torch.float32, device=device
    )
    tensor = backing[slice(head, head + value.size)].view(*value.shape)
    tensor.copy_(torch.from_numpy(value).to(device))
    return tensor, backing


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("workers", [1, 3])
def test_contract(device, shape, kind, workers):
    host = data(*shape, kind)
    expected = reference(*host)
    inputs = [guarded(value, device) for value in host]
    outputs = [
        guarded(np.empty(value.shape, dtype=np.float32), device, head=8)
        for value in expected
    ]
    for poison in (float("nan"), -12345.0):
        for tensor, _ in outputs:
            tensor.fill_(poison)
        run(
            *(item[0] for item in inputs),
            *(item[0] for item in outputs),
            workers=workers,
        )
        torch.npu.synchronize()
        actual = [tensor.cpu().numpy() for tensor, _ in outputs]
        for got, want in zip(actual, expected):
            np.testing.assert_allclose(got, want, atol=2e-4, rtol=2e-4)
        x0 = np.floor(host[1][..., 0])
        y0 = np.floor(host[1][..., 1])
        invalid = (x0 < -1) | (x0 >= shape[2]) | (y0 < -1) | (y0 >= shape[1])
        np.testing.assert_array_equal(actual[1][invalid], 0)
        if kind == "outside":
            for got in actual:
                np.testing.assert_array_equal(got, 0)
        for (tensor, _), value in zip(inputs, host):
            np.testing.assert_array_equal(
                tensor.cpu().numpy().view(np.uint8), value.view(np.uint8)
            )
        for tensor, backing in inputs + outputs:
            np.testing.assert_array_equal(
                backing[: tensor.storage_offset()].cpu().numpy(), 8192
            )
            np.testing.assert_array_equal(backing[-64:].cpu().numpy(), 8192)
