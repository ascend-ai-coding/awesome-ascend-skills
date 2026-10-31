# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""验证像素坐标双线性采样与逐邻点零延拓。

连续 NHWC 输入和偏移输出带物理守卫；唯一输出 owner 应完整覆写每个通道。
FP64 floor 参考覆盖负分数、边角、相邻 ULP、退化空间轴、重复坐标和通道尾部。
"""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import bilinear_reference


def data(batches, height, width, channels, points, kind):
    rng = np.random.default_rng(batches * height * width * channels + points)
    x = rng.uniform(-1, 1, (batches, height, width, channels)).astype(np.float32)
    grid = np.empty((batches, points, 2), dtype=np.float32)
    grid[:, :, 0] = rng.uniform(-1.5, width + 0.5, (batches, points))
    grid[:, :, 1] = rng.uniform(-1.5, height + 0.5, (batches, points))
    if kind == "integer":
        special = np.array(
            [1, 0x80000001, 0x7FFFFF, 0x800000, 0x80000000, 0], dtype=np.uint32
        )
        payload = x.view(np.uint32).reshape(-1)[::13]
        payload[:] = np.resize(special, payload.shape)
        grid[:, :, 0] = rng.integers(0, width, (batches, points))
        grid[:, :, 1] = rng.integers(0, height, (batches, points))
    elif kind == "edge":
        coordinates = np.array(
            [
                [-0.25, 0],
                [0, -0.25],
                [-0.25, -0.75],
                [width - 0.25, height - 1],
                [width - 1, height - 0.25],
                [width - 0.5, height - 0.5],
                [-1, 0],
                [width, 0],
                [0, height],
                [0, -1],
                [0, 0],
                [width - 1, height - 1],
            ],
            dtype=np.float32,
        )
        grid[:] = np.resize(coordinates, (batches, points, 2))
    elif kind == "outside":
        grid[:, :, 0] = np.resize(
            np.array([-2.0, float(width + 1)], dtype=np.float32), (batches, points)
        )
        grid[:, :, 1] = np.float32(height / 2)
    elif kind == "repeated":
        grid[:, :, 0] = np.float32((width - 1) * 0.37)
        grid[:, :, 1] = np.float32((height - 1) * 0.61)
        x[:] = np.linspace(-1, 1, channels, dtype=np.float32)
    elif kind == "ulp":
        positions = np.array([0, 1, width - 1, width], dtype=np.float32)
        near = np.stack(
            [
                np.nextafter(positions, np.float32(-np.inf)),
                positions,
                np.nextafter(positions, np.float32(np.inf)),
            ],
            axis=1,
        ).reshape(-1)
        grid[:, :, 0] = np.resize(near, (batches, points))
        grid[:, :, 1] = np.float32((height - 1) * 0.5)
    return x, grid


def guarded(value, device, head=64):
    backing = torch.full(
        (value.size + head + 64,), 8192, dtype=torch.float32, device=device
    )
    tensor = backing[slice(head, head + value.size)].view(*value.shape)
    tensor.copy_(torch.from_numpy(value).to(device))
    return tensor, backing


@pytest.mark.parametrize(
    "batches,height,width,channels,points",
    [
        (2, 7, 11, 33, 65),
        (1, 33, 65, 65, 257),
        (2, 65, 97, 129, 1025),
        (1, 1, 1, 1, 17),
        (3, 1, 17, 7, 1),
        (1, 129, 1, 17, 129),
        (4, 3, 5, 3, 4097),
        (4, 129, 193, 129, 4097),
    ],
)
@pytest.mark.parametrize(
    "kind", ["random", "integer", "edge", "outside", "repeated", "ulp"]
)
@pytest.mark.parametrize("workers", [1, 3])
def test_bilinear(device, batches, height, width, channels, points, kind, workers):
    x, grid = data(batches, height, width, channels, points, kind)
    expected = bilinear_reference(x, grid)
    inputs = [guarded(x, device), guarded(grid, device)]
    output = guarded(np.zeros_like(expected, dtype=np.float32), device, head=8)
    for poison in [float("nan"), 12345.0]:
        output[0].fill_(poison)
        run(inputs[0][0], inputs[1][0], output[0], workers=workers)
        torch.npu.synchronize()
        actual = output[0].cpu().numpy()
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
        if kind in ["outside", "integer"]:
            np.testing.assert_array_equal(actual, expected.astype(np.float32))
        for tensor, backing in inputs + [output]:
            np.testing.assert_array_equal(
                backing[: tensor.storage_offset()].cpu().numpy(), 8192
            )
            np.testing.assert_array_equal(backing[-64:].cpu().numpy(), 8192)
        for (tensor, _), value in zip(inputs, [x, grid]):
            np.testing.assert_array_equal(
                tensor.cpu().numpy().view(np.uint32), value.view(np.uint32)
            )
