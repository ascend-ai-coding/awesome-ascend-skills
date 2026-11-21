# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""四问：外轴归约；紧密 GM 与按列分块；单槽同步；FP64 参考、极值分布与守卫。"""

from functools import lru_cache
import numpy as np
import pytest
import torch
from .kernel import make_column_stats
from .reference import column_stats_reference

_make = lru_cache(maxsize=None)(make_column_stats)


@pytest.mark.parametrize("m,n", [(m, n) for m in (1, 33, 257) for n in (31, 64, 129)])
@pytest.mark.parametrize("workers", [1, 3, 8])
@pytest.mark.parametrize("kind", [0, 1, 2])
def test_statistics_and_guards(device, m, n, workers, kind):
    rng = np.random.default_rng(97317 + m * n)
    if kind == 0:
        x = rng.uniform(-4, 4, (m, n)).astype(np.float32)
    elif kind == 1:
        x = (
            np.where(np.arange(n) % 2, -4096.0, 4096.0)[None, :]
            + rng.uniform(-1, 1, (m, n))
        ).astype(np.float32)
    else:
        x = np.broadcast_to(rng.uniform(-4, 4, n).astype(np.float32), (m, n)).copy()
    expected_mean, expected_var = column_stats_reference(x)
    tx = torch.from_numpy(x).to(device)
    mean_backing = torch.full((n + 128,), 6917.0, device=device, dtype=torch.float32)
    var_backing = torch.full_like(mean_backing, 6917.0)
    mean, var = mean_backing[64:-64], var_backing[64:-64]
    mean.fill_(float("nan"))
    var.fill_(float("nan"))
    _make(m, n)[None, workers](tx, mean, var)
    torch.npu.synchronize()
    np.testing.assert_allclose(mean.cpu().numpy(), expected_mean, rtol=1e-6, atol=1e-3)
    np.testing.assert_allclose(var.cpu().numpy(), expected_var, rtol=2e-4, atol=2e-4)
    assert torch.all(var.cpu() >= 0)
    for backing in (mean_backing, var_backing):
        assert torch.all(backing[:64].cpu() == 6917)
        assert torch.all(backing[-64:].cpu() == 6917)
    np.testing.assert_array_equal(tx.cpu().numpy(), x)
