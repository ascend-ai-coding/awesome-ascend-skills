# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""验证片上矩阵状态的跨时间传播、门控重置、轴方向和改变输入后的完整调用。

每个 worker 独占一个状态序列；独立的 Vec 缓冲组由 auto_mutex 管理 VF 调用之间的读写依赖。状态只在开始读入、结束写出，输出按时间步写出。连续偏移输入的前后守卫检查越界写，独立 FP64 数学参考使用实际 FP16/FP32 输入，不模拟分块顺序。此处验证正确性与状态语义，不声称性能优势。
"""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import reference


CASES = [
    (1, 1, 16, "coordinates"),
    (2, 2, 32, "zero"),
    (3, 5, 64, "one"),
    (5, 3, 128, "mixed"),
    (33, 1, 16, "coordinates"),
    (1, 257, 128, "upper"),
    (2, 65, 64, "random"),
    (3, 129, 16, "mixed"),
    (33, 257, 128, "random"),
]


def arrays(rows, steps, dim, pattern):
    rng = np.random.default_rng(rows * 100003 + steps * 131 + dim)
    q, k, v = [
        rng.uniform(-0.125, 0.125, (rows, steps, dim)).astype(np.float16)
        for _ in range(3)
    ]
    g = rng.uniform(0, 1, (rows, steps)).astype(np.float32)
    s0 = rng.uniform(-0.125, 0.125, (rows, dim, dim)).astype(np.float32)
    if pattern == "zero":
        g.fill(0)
    elif pattern in ("one", "upper"):
        g.fill(1)
        if pattern == "upper":
            q.fill(0.125)
            k.fill(0.125)
            v.fill(0.125)
            s0.fill(-0.125)
    elif pattern == "mixed":
        g[:] = (np.arange(rows)[:, None] + np.arange(steps)) % 3 != 0
        g[:, 1::4] = 0.5
    elif pattern == "coordinates":
        g.fill(1)
        q.fill(0)
        k.fill(0)
        v.fill(0)
        for r in range(rows):
            q[r, :, r % dim] = 0.125
        s0[:] = (
            (
                np.arange(dim)[:, None] * 5
                + np.arange(dim) * 3
                + np.arange(rows)[:, None, None]
            )
            % 17
            - 8
        ) / 64
    return q, k, v, g, s0


@pytest.mark.parametrize("case", CASES)
def test_recurrence_and_changed_inputs(device, case):
    host = arrays(*case)
    backings, tensors = [], []
    for value in host:
        dtype = torch.float16 if value.dtype == np.float16 else torch.float32
        backing = torch.full((value.size + 64,), -2, dtype=dtype, device=device)
        tensor = backing[17: 17 + value.size].view(value.shape)
        backings.append(backing)
        tensors.append(tensor)
    changed = tuple(
        (value * factor).astype(value.dtype)
        for value, factor in zip(host[:3], (-0.75, -0.5, 0.25))
    )
    changed += (1 - host[3], -host[4] * np.float32(0.75))
    for current in (host, changed):
        for tensor, value in zip(tensors, current):
            tensor.copy_(torch.from_numpy(value))
        expected_y, expected_state = reference(*current)
        y, state = run(*tensors)
        torch.npu.synchronize()
        for actual, expected, atol in (
            (y, expected_y, 2e-3),
            (state, expected_state, 1e-4),
        ):
            assert (
                actual.dtype == torch.float32 and tuple(actual.shape) == expected.shape
            )
            assert actual.is_contiguous() and actual.device == tensors[0].device
            result = actual.cpu().numpy()
            assert np.isfinite(result).all()
            np.testing.assert_allclose(result, expected, atol=atol, rtol=1e-4)
        for tensor, value, backing in zip(tensors, current, backings):
            np.testing.assert_array_equal(tensor.cpu().numpy(), value)
            assert (backing[:17] == -2).all().item()
            assert (backing[17 + value.size:] == -2).all().item()
