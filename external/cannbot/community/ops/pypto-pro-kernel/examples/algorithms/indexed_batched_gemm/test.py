# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：语法糖。Verify indexed bytes and the actual dense consumer separately.

Short/partial M, rectangular operands, different batches/selectors, repeated
and endpoint indices, changing selectors at reused addresses, 1/7 workers,
poisoned panel/output and guards. The reference uses the original table and
FP64 products, not the kernel's transport grouping or accumulation order.
"""

import pytest
import torch

from .kernel import make_kernels
from .reference import reference_and_bound


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [
        (2, 1, 97, 128, 64),
        (2, 3, 257, 256, 192),
        (3, 17, 53, 128, 128),
        (1, 65, 193, 384, 1024),
        (2, 129, 37, 128, 64),
    ],
)
def test_panel_and_matrix_consumer(shape, dtype, device):
    b, m, r, n, k = shape
    generator = torch.Generator().manual_seed(91)
    x = torch.rand((b, m, k), generator=generator).mul(2).sub(1).to(dtype)
    table = torch.rand((b, r, k), generator=generator).mul(2).sub(1).to(dtype)
    indices = torch.randint(r, (b, n), generator=generator, dtype=torch.int32)
    indices[:, :3] = torch.tensor([0, r - 1, 0], dtype=torch.int32)
    dx, dt, di = x.to(device), table.to(device), indices.to(device)
    panel_backing = torch.full((b * n * k + 64,), 19, dtype=dtype, device=device)
    panel = panel_backing[32:-32].view(b, n, k)
    output_backing = torch.full(
        (b * m * n + 64,), 23, dtype=torch.float32, device=device
    )
    out = output_backing[32:-32].view(b, m, n)
    gather, product = make_kernels(*shape, str(dtype).split(".")[-1])
    for workers, shift in ((1, 0), (7, 7)):
        current = (indices + shift) % r
        di.copy_(current)
        panel.fill_(float("nan"))
        out.fill_(float("nan"))
        gather[None, workers](dt, di, panel)
        product[None, workers](dx, panel, out)
        torch.npu.synchronize()
        expected_panel = table[torch.arange(b)[:, None], current.long()]
        assert torch.equal(
            panel.cpu().view(torch.int16), expected_panel.view(torch.int16)
        )
        expected, bound = reference_and_bound(x, table, current)
        got = out.cpu().double()
        assert torch.isfinite(got).all()
        assert ((got - expected).abs() <= bound).all()
        assert (panel_backing[:32] == 19).all() and (panel_backing[-32:] == 19).all()
        assert (output_backing[:32] == 23).all() and (output_backing[-32:] == 23).all()
        assert torch.equal(dx.cpu(), x) and torch.equal(dt.cpu(), table)
        assert torch.equal(di.cpu(), current)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_coordinate_distinct_exact_product(dtype, device):
    b, m, r, n, k = 2, 17, 97, 128, 192
    batch = torch.arange(b)[:, None, None]
    row = torch.arange(r)[None, :, None]
    feature = torch.arange(k)[None, None, :]
    table = (((row * (feature + 1) + 13 * batch) % 31 - 15) / 16).to(dtype)
    x = torch.zeros((b, m, k), dtype=dtype)
    for bi in range(b):
        x[bi, torch.arange(m), (torch.arange(m) * 17 + bi) % k] = 1
    indices = (
        (torch.arange(n)[None, :] * 37 + torch.arange(b)[:, None] * 11) % r
    ).int()
    expected, _ = reference_and_bound(x, table, indices)
    panel = torch.empty((b, n, k), dtype=dtype, device=device)
    out = torch.full((b, m, n), float("nan"), dtype=torch.float32, device=device)
    gather, product = make_kernels(b, m, r, n, k, str(dtype).split(".")[-1])
    gather[None, 7](table.to(device), indices.to(device), panel)
    product[None, 7](x.to(device), panel, out)
    torch.npu.synchronize()
    assert torch.equal(out.cpu().double(), expected)
