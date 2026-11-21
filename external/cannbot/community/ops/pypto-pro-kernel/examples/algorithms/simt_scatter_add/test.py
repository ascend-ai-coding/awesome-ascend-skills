# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""四问：重复索引累加；直接 GM；同一 Stream 分阶段；整数参考、重复调用与输入/输出守卫。"""

from functools import lru_cache
import numpy as np
import pytest
import torch
from .kernel import make_scatter_add
from .reference import scatter_add_reference

_make = lru_cache(None)(make_scatter_add)
SHAPES = [(n, u) for n in (1, 33, 257) for u in (0, 65, 4097)] + [
    (17, 1),
    (31, 129),
    (97, 2049),
]


@pytest.mark.parametrize("n,u", SHAPES)
@pytest.mark.parametrize("init_workers,update_workers", [(1, 8), (3, 3), (8, 1)])
def test_scatter_contract_and_guards(device, n, u, init_workers, update_workers):
    rng = np.random.default_rng(74197 + n * u)
    initialize, accumulate = _make(n, u)
    for kind in range(3):
        base = rng.integers(-100, 101, n, dtype=np.int32)
        values = rng.integers(-7, 8, u, dtype=np.int32)
        if kind == 0:
            index = rng.integers(-n, n, u, dtype=np.int32)
        elif kind == 1:
            index = np.where(np.arange(u) % 2, -n, 0).astype(np.int32)
        else:
            index = np.where(np.arange(u) % 2, -1, n - 1).astype(np.int32)
            values[:] = 7
        if u:
            values[-1] = 7
        expected = scatter_add_reference(base, index, values)
        tb, ti, tv = [torch.from_numpy(a).to(device) for a in (base, index, values)]
        raw = torch.full((n + 128,), 123456789, dtype=torch.int32, device=device)
        out = raw[64:-64]
        for repeat in range(2):
            out.fill_(-23456789 + repeat)
            initialize[None, init_workers](tb, out)
            if accumulate is not None:
                accumulate[None, update_workers](ti, tv, out)
            torch.npu.synchronize()
            np.testing.assert_array_equal(out.cpu().numpy(), expected)
            assert torch.all(raw[:64].cpu() == 123456789)
            assert torch.all(raw[-64:].cpu() == 123456789)
            for tensor, host in ((tb, base), (ti, index), (tv, values)):
                np.testing.assert_array_equal(tensor.cpu().numpy(), host)
