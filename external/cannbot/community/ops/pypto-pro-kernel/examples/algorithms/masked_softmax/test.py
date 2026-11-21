# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""四问：

1. 解决什么：有效前缀归一化、空集结果、动态行数与非整除 worker 分配。
2. 物理访问是什么：物理 UB 宽度按 FP32 寄存器填齐，GM 仅搬运真实列数。
3. 为何这样同步：每行唯一 owner，单槽 auto_mutex 保护搬运和 VF；无跨核交接。
4. 如何独立验证：float64 数学参考、块边界长度、失效区毒化、守卫行和输入未被改写。
"""

from functools import lru_cache

import numpy as np
import pytest
import torch

from .kernel import make_masked_softmax
from .reference import boundary_lengths, masked_softmax

# 在试验前固定的 FP32 exp / reduction 容差；屏蔽区另按精确零判定。
RTOL = 1e-4
ATOL = 1e-5
_make = lru_cache(maxsize=None)(make_masked_softmax)


@pytest.mark.parametrize("workers", [1, 3, 8])
@pytest.mark.parametrize("m,n", [(m, n) for m in (1, 7, 65) for n in (64, 130, 257)])
def test_masked_prefix_and_guards(device, m, n, workers):
    kernel = _make(n)
    rng = np.random.default_rng(831792 + m * n)
    edges = boundary_lengths(n)
    for phase in range(0, len(edges), m):
        lengths = np.resize(np.asarray(edges[slice(phase, phase + m)], np.int32), m)
        x = rng.uniform(-8, 8, (m, n)).astype(np.float32)
        for row, length in enumerate(lengths):
            if row % 3 == 0:
                x[row, :length] = -8.0
                x[row, length:] = 8.0
            elif row % 3 == 1:
                x[row] = 0.0
        expected = masked_softmax(x, lengths)
        tx = torch.from_numpy(x).to(device)
        tl = torch.from_numpy(lengths).to(device)
        backing = torch.full((m + 2, n), 777.0, dtype=torch.float32, device=device)
        out = backing[1:-1]
        out.fill_(float("nan"))
        kernel[None, workers](tx, tl, out)
        torch.npu.synchronize()
        got = out.cpu().numpy()
        np.testing.assert_allclose(got, expected, rtol=RTOL, atol=ATOL)
        for row, length in enumerate(lengths):
            np.testing.assert_array_equal(got[row, length:], 0.0)
        np.testing.assert_array_equal(backing[[0, m + 1]].cpu().numpy(), 777.0)
        np.testing.assert_array_equal(tx.cpu().numpy(), x)
        np.testing.assert_array_equal(tl.cpu().numpy(), lengths)
