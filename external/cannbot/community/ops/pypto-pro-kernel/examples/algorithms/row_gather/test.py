# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""四问：逐行间接读取；物理补齐与逻辑尾块分开；单槽自动同步；新种子、位模式、输入不变与输出守卫。"""

from functools import lru_cache

import numpy as np
import pytest
import torch

from .kernel import make_row_gather
from .reference import row_gather

_make = lru_cache(maxsize=None)(make_row_gather)


@pytest.mark.parametrize("workers", [1, 4, 8])
@pytest.mark.parametrize(
    "m,n,k", [(m, n, k) for m in (1, 7, 33) for n in (65, 257) for k in (31, 80)]
)
def test_indexing_bits_and_guards(device, m, n, k, workers):
    kernel = _make(n, k)
    for kind in (0, 1):
        rng = np.random.default_rng(19283 + m * n + k + kind)
        if kind == 0:
            x = (np.arange(m * n, dtype=np.float32).reshape(m, n) + 17) / 8
        else:
            bits = rng.integers(0, 2**32, size=(m, n), dtype=np.uint32) & np.uint32(
                0xFEFFFFFF
            )
            x = bits.view(np.float32)
            x[:, 0] = np.float32(-0.0)
        index = rng.integers(-n, n, size=(m, k), dtype=np.int32)
        index[:, :6] = np.array([-n, n - 1, -1, 0, -n, n - 1], np.int32)
        expected = row_gather(x, index)
        tx = torch.from_numpy(x).to(device)
        ti = torch.from_numpy(index).to(device)
        backing = torch.full((m + 2, k), 777.0, device=device, dtype=torch.float32)
        out = backing[1:-1]
        out.fill_(float("nan"))
        kernel[None, workers](tx, ti, out)
        torch.npu.synchronize()
        np.testing.assert_array_equal(
            out.cpu().numpy().view(np.uint32), expected.view(np.uint32)
        )
        np.testing.assert_array_equal(backing[[0, m + 1]].cpu().numpy(), 777.0)
        np.testing.assert_array_equal(
            tx.cpu().numpy().view(np.uint32), x.view(np.uint32)
        )
        np.testing.assert_array_equal(ti.cpu().numpy(), index)
