# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""长行均值与总体方差的独立验收。

语义：连续 FP32 x[M,N]，M=1..64、N=1..262145、有限值在 [-16384,16384]，输出 FP32 mean[M] 与 var[M]；常数行和 N=1 的方差精确零。
物理访问：输入输出均来自有前后守卫的连续切片，起点偏移 67 个元素；覆盖寄存器与传输边界、极长行、短行和不整除的多 worker 分工。
同步：每次调用前换用不同输出毒值，检查输出全部定义、输入位模式不变及所有守卫。kernel.py 说明聚合状态与输入搬运的同步。
独立性：reference.py 从实际输入的数学定义求 FP64 结果，不复现设备的分块顺序。均值 atol=1e-3/rtol=2e-6；方差 atol=1e-5/rtol=2e-4。除随机全范围输入，还包括非二进制常数、大偏置小方差、
抵消、扰动抵消与首尾异常值。
"""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import reference


SHAPES = ((1, 1), (3, 65), (7, 4097), (2, 65537), (5, 262145), (64, 8192))
CASES = [
    (rows, cols, kind, workers)
    for rows, cols in SHAPES
    for kind in ("full_range", "constant", "offset", "cancel")
    for workers in (1, 4)
] + [
    (1, 1, "random", 4),
    (3, 63, "offset", 1),
    (5, 64, "constant", 4),
    (2, 65, "tail_peak", 4),
    (3, 4095, "random", 4),
    (3, 4096, "offset", 1),
    (1, 65535, "offset", 4),
    (2, 65536, "constant", 1),
    (3, 65537, "tail_peak", 4),
    (2, 262143, "offset", 4),
    (1, 262144, "cancel", 1),
    (3, 262145, "offset", 4),
    (7, 262145, "constant", 4),
    (64, 1, "random", 4),
    (64, 262145, "random", 4),
    (8, 3, "constant", 4),
    (3, 262145, "head_peak", 4),
    (2, 65537, "perturbed_cancel", 1),
    (3, 262145, "perturbed_cancel", 4),
]


def make_input(rows, cols, kind):
    seed = 113 * rows + cols
    if kind == "full_range":
        seed = 20261007 + SHAPES.index((rows, cols))
    rng = np.random.default_rng(seed)
    extent = 16384 if kind == "full_range" else 2
    x = rng.uniform(-extent, extent, (rows, cols)).astype(np.float32)
    if kind == "constant":
        values = np.resize(
            np.array(
                [1.500000238418579, -17.3, 0.1, 16383.99, -8191.875, 2**-23, 1e-20, 0],
                np.float32,
            ),
            rows,
        )
        x[:] = values[:, None]
    elif kind == "offset":
        x[:] = 8192 + rng.integers(-8, 9, x.shape) * 2**-5
    elif kind in ("head_peak", "tail_peak"):
        x.fill(0)
        x[:, 0 if kind == "head_peak" else -1] = 16384
    elif kind in ("cancel", "perturbed_cancel"):
        x[:, ::2] = 16384
        x[:, 1::2] = -16384 + (2**-6 if kind == "perturbed_cancel" else 0)
    return x


def guarded(array, device):
    backing = torch.full((array.size + 138,), 123.0, dtype=torch.float32, device=device)
    value = backing[slice(67, 67 + array.size)].view(array.shape)
    value.copy_(torch.from_numpy(array).to(device))
    return value, backing


@pytest.mark.parametrize("rows,cols,kind,workers", CASES)
def test_streaming_moments(device, rows, cols, kind, workers):
    original = make_input(rows, cols, kind)
    expected_mean, expected_var = reference(original)
    source = guarded(original, device)
    outputs = [guarded(np.zeros(rows, np.float32), device) for _ in range(2)]
    for poison in (123.0, -99.0):
        for value, _ in outputs:
            value.fill_(poison)
        run(source[0], outputs[0][0], outputs[1][0], workers=workers)
        torch.npu.synchronize()
        mean, variance = (item[0].cpu().numpy() for item in outputs)
        np.testing.assert_allclose(mean, expected_mean, atol=1e-3, rtol=2e-6)
        np.testing.assert_allclose(variance, expected_var, atol=1e-5, rtol=2e-4)
        assert np.all(variance >= 0)
        if kind == "constant" or cols == 1:
            assert np.all(variance == 0)
        for value, backing in [source] + outputs:
            host = backing.cpu().numpy()
            assert np.all(host[:67] == 123.0)
            assert np.all(host[slice(67 + value.numel(), None)] == 123.0)
    np.testing.assert_array_equal(
        source[0].cpu().numpy().view(np.uint32), original.view(np.uint32)
    )
