# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""验证原地状态更新的独立 FP64 参考、连续调用、输入不变及偏移守卫。

每个 worker 独占其元素，四个缓冲组使用独立地址与 ID，自动同步延续到写回完成。连续偏移张量的前后守卫检查越界写，grad 按位检查不变。

微小梯度与零二阶状态覆盖 epsilon 的作用；近乎抵消的参数更新防止大幅输出的相对容差掩盖误差。测试按实际状态逐次计算参考，只在下一次调用的输入中承接 FP32 存储舍入。
"""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import reference


def test_changing_gradient_and_scalars(device):
    rng = np.random.default_rng(64379)
    n = 129
    initial = (
        rng.uniform(-1, 1, n).astype(np.float32),
        rng.uniform(-1, 1, n).astype(np.float16),
        rng.uniform(-0.1, 0.1, n).astype(np.float32),
        rng.uniform(0.1, 1, n).astype(np.float32),
    )
    tensors = [torch.from_numpy(x).to(device) for x in initial]
    settings = (
        (1e-5, 1e-8, 0.0, 1),
        (0.01, 1e-6, 0.1, 10000),
        (0.001, 1e-8, 0.05, 2),
        (0.003, 1e-6, 0.0, 17),
    )
    for lr, eps, wd, step in settings * 2:
        gradient = rng.uniform(-10, 10, n).astype(np.float16)
        tensors[1].copy_(torch.from_numpy(gradient))
        before = [tensor.cpu().numpy().copy() for tensor in tensors]
        wanted = reference(*before, lr, 0.9, 0.999, eps, wd, step)
        run(*tensors, lr, 0.9, 0.999, eps, wd, step, workers=4)
        torch.npu.synchronize()
        for actual, expected in zip((tensors[0], tensors[2], tensors[3]), wanted):
            np.testing.assert_allclose(
                actual.cpu().numpy(), expected, rtol=2e-5, atol=2e-5
            )
        np.testing.assert_array_equal(
            tensors[1].cpu().numpy().view(np.uint16), gradient.view(np.uint16)
        )


CASES = [
    (n, 4, kind, step, eps)
    for n in (1, 7, 63, 64, 65, 127, 128, 129, 4097)
    for kind, step, eps in (("random", 1, 1e-8), ("tiny", 9998, 1e-6))
]
CASES += [
    (n, w, kind, step, eps)
    for n in (65537, 1048576)
    for w in (1, 16)
    for kind, step, eps in (("random", 99, 1e-8), ("zero", 1, 1e-6))
]
CASES += [
    (n, w, "extrema", step, eps)
    for n, w in ((129, 1), (4097, 16))
    for step, eps in ((1, 1e-8), (10000, 1e-6))
]
CASES += [
    (257, 4, "cancellation", step, eps) for step in (1, 10000) for eps in (1e-8, 1e-6)
]
CASES += [(129, 16, "extrema", 1, 1e-8, 1e-5)]
CASES += [
    (8191, 1, "random", 1, 1e-8),
    (8192, 4, "random", 1, 1e-8),
    (8193, 16, "random", 1, 1e-8),
    (1048575, 16, "random", 99, 1e-6),
]
CASES += [
    (16383, 1, "random", 1, 1e-8),
    (16384, 4, "random", 1, 1e-8),
    (16385, 1, "random", 1, 1e-8),
]
CASES = [case if len(case) == 6 else (*case, None) for case in CASES]


@pytest.mark.parametrize("n,workers,kind,step,eps,lr_override", CASES)
def test_adamw(device, n, workers, kind, step, eps, lr_override):
    rng = np.random.default_rng(181213 + n)
    p = rng.uniform(-1, 1, n).astype(np.float32)
    g = rng.uniform(-1, 1, n).astype(np.float16)
    m = rng.uniform(-0.05, 0.05, n).astype(np.float32)
    v = rng.uniform(0.1, 1, n).astype(np.float32)
    if kind == "tiny":
        g = np.resize(
            np.array(
                [0, 2 ** (-24), -(2 ** (-24)), 2 ** (-14), -(2 ** (-14))], np.float16
            ),
            n,
        )
        m.fill(0)
        v.fill(0)
    elif kind == "zero":
        g.fill(0)
        m.fill(0)
        v.fill(0)
    elif kind == "extrema":
        p = np.resize(np.array([-10, 10, 0, 1], np.float32), n)
        g = np.resize(np.array([-10, 10, 0, 2 ** (-24)], np.float16), n)
        m = np.resize(np.array([-10, 10, 10, -10], np.float32), n)
        v = np.resize(np.array([100, 0, 0, 2 ** (-126)], np.float32), n)
    elif kind == "cancellation":
        g = np.resize(
            np.array(
                [
                    0,
                    2 ** (-24),
                    -(2 ** (-24)),
                    2 ** (-14),
                    -(2 ** (-14)),
                    1 / 64,
                    -1 / 64,
                    1 / 16,
                ],
                np.float16,
            ),
            n,
        )
        v.fill(0)
        target = np.resize(np.array([-9, 9], np.float64), n)
        g64 = g.astype(np.float64)
        denominator = np.sqrt((1 - 0.999) * g64 * g64 / (1 - 0.999**step)) + eps
        m = (
            (target / 0.01 * (1 - 0.9**step) * denominator - (1 - 0.9) * g64) / 0.9
        ).astype(np.float32)
        p = -reference(
            np.zeros(n, np.float32), g, m, v, 0.01, 0.9, 0.999, eps, 0, step
        )[0].astype(np.float32)
        assert np.max(np.abs(m)) <= 10 and np.max(np.abs(p)) <= 10
    host = [p, g, m, v]
    tensors = []
    backings = []
    for a in host:
        dtype = torch.float16 if a.dtype == np.float16 else torch.float32
        backing = torch.full((n + 81,), 128, dtype=dtype, device=device)
        tensor = backing[slice(17, 17 + n)]
        tensor.copy_(torch.from_numpy(a))
        tensors.append(tensor)
        backings.append(backing)
    lr, b1, b2, wd = (
        0.01 if kind in ("extrema", "cancellation") else 0.001,
        0.9,
        0.999,
        0 if kind in ("extrema", "cancellation") else 0.1,
    )
    if lr_override is not None:
        lr = lr_override
    iterations = 1 if kind in ("extrema", "cancellation") else 3
    for iteration in range(iterations):
        current = [tensor.cpu().numpy().copy() for tensor in tensors]
        expected = reference(*current, lr, b1, b2, eps, wd, step + iteration)
        run(*tensors, lr, b1, b2, eps, wd, step + iteration, workers=workers)
        torch.npu.synchronize()
        for tensor, wanted in zip((tensors[0], tensors[2], tensors[3]), expected):
            np.testing.assert_allclose(
                tensor.cpu().numpy(), wanted, rtol=2e-05, atol=2e-05
            )
        np.testing.assert_array_equal(
            tensors[1].cpu().numpy().view(np.uint16), g.view(np.uint16)
        )
        for backing in backings:
            assert bool(torch.all(backing[:17].cpu() == 128))
            assert bool(torch.all(backing[slice(17 + n, None)].cpu() == 128))
