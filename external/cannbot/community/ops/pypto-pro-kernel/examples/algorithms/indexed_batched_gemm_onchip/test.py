# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Verify slot reuse, both subblocks, tails and changed inputs at fixed addresses."""

import pytest
import torch

from .kernel import make_kernel
from .reference import reference_and_bound


CASES = [
    ((1, 1, 37, 128, 64), 1, 1, 0),
    ((2, 17, 97, 256, 192), 3, 1, 0),
    ((3, 129, 193, 384, 320), 7, 1, 0),
    ((3, 65, 257, 128, 1024), 3, 1, 0),
    ((2, 17, 37, 128, 64), 1, 2, 1),
    ((3, 129, 193, 384, 320), 7, 2, 1),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape,workers,depth,preload", CASES)
def test_indexed_onchip(shape, workers, depth, preload, dtype, device):
    b, m, r, n, k = shape
    generator = torch.Generator().manual_seed(18371 + k + m)
    x = torch.rand((b, m, k), generator=generator).mul(2).sub(1).to(dtype)
    table = torch.rand((b, r, k), generator=generator).mul(2).sub(1).to(dtype)
    indices = torch.randint(r, (b, n), generator=generator, dtype=torch.int32)
    dx, dt, di = x.to(device), table.to(device), indices.to(device)
    backing = torch.full((b * m * n + 128,), 23, dtype=torch.float32, device=device)
    out = backing[64:-64].view(b, m, n)
    kernel = make_kernel(*shape, str(dtype).split(".")[-1], workers, depth, preload)
    pointers = [t.data_ptr() for t in (dx, dt, di, out)]
    for repeat in range(2):
        current = (indices + 7 * repeat) % r
        values = table if repeat == 0 else -table
        dt.copy_(values)
        di.copy_(current)
        out.fill_(float("nan"))
        kernel[None, workers](dx, dt, di, out)
        torch.npu.synchronize()
        expected, bound = reference_and_bound(x, values, current)
        got = out.cpu().double()
        assert torch.isfinite(got).all() and ((got - expected).abs() <= bound).all()
        assert torch.equal(dx.cpu(), x) and torch.equal(dt.cpu(), values)
        assert torch.equal(di.cpu(), current)
        assert (backing[:64] == 23).all() and (backing[-64:] == 23).all()
        assert [t.data_ptr() for t in (dx, dt, di, out)] == pointers
