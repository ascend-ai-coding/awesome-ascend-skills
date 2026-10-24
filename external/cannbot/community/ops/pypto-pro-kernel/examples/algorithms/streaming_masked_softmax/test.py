# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""四问：

1. 解决什么：超过 UB 容量的行归一化、空有效集和稳定的在线指数和。
2. 物理访问是什么：固定搬运块加两个 UB 统计槽，输出保持紧密连续布局。
3. 为何这样同步：每行单一 owner，tile group 排序 GM 搬运、VF 状态更新与输出。
4. 如何独立验证：FP64 数学参考、首/末峰值和递增最大值、重复调用、双输出守卫与输入不变。
"""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import reference

# 在实现前固定的任务容差，覆盖 FP32 exp、归约和输出舍入；屏蔽区另按精确零判定。
PROB_RTOL = 1e-4
PROB_ATOL = 1e-7
LSE_ATOL = 2e-4


@pytest.mark.parametrize("m,n", [(5, 65), (3, 4097), (3, 131073)])
@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize(
    "kind", ["random", "offset", "uniform", "first_peak", "last_peak", "rising_max"]
)
def test_streamed_normalization(device, m, n, workers, kind):
    rng = np.random.default_rng(5921 + m * n)
    if kind in ("random", "offset"):
        x = rng.uniform(-20, 20, (m, n)).astype(np.float32)
        if kind == "offset":
            x += np.where(np.arange(m) % 2, 1000.0, -1000.0).astype(np.float32)[:, None]
    elif kind == "uniform":
        x = np.full((m, n), -3.0, dtype=np.float32)
    elif kind == "rising_max":
        x = np.broadcast_to(np.linspace(-2, 2, n, dtype=np.float32), (m, n)).copy()
    else:
        x = np.full((m, n), -1.0, dtype=np.float32)
        x[:, 0 if kind == "first_peak" else -1] = 1.0

    for boundaries in ([0, 1, n, 64, n - 1], [n, n - 1, n // 2, 1, 0]):
        lengths = np.resize(np.asarray(boundaries, dtype=np.int32), m)
        values = x.copy()
        if kind in ("random", "offset"):
            for row, length in enumerate(lengths):
                values[row, length:] = 1020.0
        expected_y, expected_lse = reference(values, lengths)
        tx = torch.from_numpy(values).to(device)
        tl = torch.from_numpy(lengths).to(device)
        y_backing = torch.full(
            (m * n + 128,), 913.0, dtype=torch.float32, device=device
        )
        lse_backing = torch.full((m + 128,), 913.0, dtype=torch.float32, device=device)
        out = y_backing[64:-64].view(m, n)
        lse = lse_backing[64:-64].view(m, 1)
        out.fill_(float("nan"))
        lse.fill_(float("nan"))
        run(tx, tl, out, lse, workers=workers)
        torch.npu.synchronize()
        got, got_lse = out.cpu().numpy(), lse.cpu().numpy()
        np.testing.assert_allclose(got, expected_y, rtol=PROB_RTOL, atol=PROB_ATOL)
        np.testing.assert_allclose(got_lse, expected_lse, rtol=0, atol=LSE_ATOL)
        for row, length in enumerate(lengths):
            np.testing.assert_array_equal(got[row, length:], 0.0)
            if length:
                np.testing.assert_allclose(
                    got[row].sum(dtype=np.float64), 1.0, rtol=0, atol=2e-4
                )
            else:
                assert np.isneginf(got_lse[row, 0])
        for backing in (y_backing, lse_backing):
            assert torch.all(backing[:64].cpu() == 913.0)
            assert torch.all(backing[-64:].cpu() == 913.0)
        np.testing.assert_array_equal(tx.cpu().numpy(), values)
        np.testing.assert_array_equal(tl.cpu().numpy(), lengths)
