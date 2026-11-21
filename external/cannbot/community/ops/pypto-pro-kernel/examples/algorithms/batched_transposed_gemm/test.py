# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""四问：batched A@W.T；高阶 GM 到二维 L1；单槽自动同步；量化参考、尾块、守卫与核数。"""

from functools import lru_cache
import pytest
import torch
from .kernel import make_batched_transposed_gemm
from .reference import reference

make = lru_cache(None)(make_batched_transposed_gemm)


@pytest.mark.parametrize(
    "b,m,n,k",
    [(b, m, n, k) for b in (1, 3) for m, n, k in ((16, 32, 64), (19, 80, 96))],
)
@pytest.mark.parametrize("kind", range(3))
@pytest.mark.parametrize("workers", [1, 3, 8])
def test_batches_tails_and_guards(device, b, m, n, k, kind, workers):
    g = torch.Generator().manual_seed(92571 + b * m * n + kind)
    if kind == 0:
        a = (torch.rand((b, m, k), generator=g) * 2 - 1).half()
        w = (torch.rand((b, n, k), generator=g) * 2 - 1).half()
    elif kind == 1:
        a = ((torch.arange(b * m * k).reshape(b, m, k) % 17 - 8) / 8).half()
        w = ((torch.arange(b * n * k).reshape(b, n, k) % 15 - 7) / 8).half()
    else:
        a = torch.zeros((b, m, k), dtype=torch.float16)
        w = (torch.rand((b, n, k), generator=g) * 2 - 1).half()
        for batch in range(b):
            a[batch, :, 0] = (batch + 1) / 4
            a[batch, :, k - 1] = -(batch + 1) / 4
    expected = reference(a, w)
    da, dw = a.to(device), w.to(device)
    raw = torch.full((b * m * n + 128,), 777.0, device=device, dtype=torch.float32)
    out = raw[64:-64].reshape(b, m, n)
    out.fill_(float("nan"))
    make(b, m, n, k)[None, workers](da, dw, out)
    torch.npu.synchronize()
    got = out.cpu().double()
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)
    assert torch.all(raw[:64].cpu() == 777) and torch.all(raw[-64:].cpu() == 777)
    assert torch.equal(da.cpu(), a) and torch.equal(dw.cpu(), w)
