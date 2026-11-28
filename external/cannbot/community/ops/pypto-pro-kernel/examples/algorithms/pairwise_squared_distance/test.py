# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""验证直接差平方、近邻、FP16 算术舍入陷阱、尾部和原地址输入变化。

worker 独占输出区域，输入和输出 tile 用独立缓冲组同步；FIRST_ELEMENT_B32 在 UB 内打包单值，GM 每次写入完整有效片段。输入从非零存储偏移开始，前后守卫检查越界写。FP64 参考独立于 VF 分块与搬运顺序；三条公开路径都受同一数学契约约束。
"""

import numpy as np
import pytest
import torch

from .kernel import run, run_rows, run_batched
from .reference import reference

CASES = [
    (1, 1, 1, "same"),
    (3, 5, 17, "random"),
    (7, 3, 63, "coordinates"),
    (2, 33, 65, "close"),
    (33, 2, 127, "same"),
    (1, 257, 1025, "close"),
    (129, 257, 512, "random"),
    (257, 129, 1025, "same"),
    (5, 7, 513, "extreme"),
    (257, 257, 1025, "random"),
    (3, 5, 1025, "subnormal"),
]


def arrays(p, q, d, pattern):
    rng = np.random.default_rng(p * 100003 + q * 131 + d)
    x = rng.uniform(-8, 8, (p, d)).astype(np.float16)
    z = rng.uniform(-8, 8, (q, d)).astype(np.float16)
    if pattern == "same":
        z[:] = x[np.arange(q) % p]
    elif pattern == "close":
        x[:] = rng.uniform(2, 6, (1, d)).astype(np.float16)
        z[:] = x[0]
        for j in range(1, q):
            cols = (j * 7 + np.arange(min(d, 33))) % d
            z[j, cols] = np.nextafter(z[j, cols], np.float16(np.inf))
    elif pattern == "coordinates":
        x[:] = ((np.arange(p)[:, None] * 7 + np.arange(d) * 3) % 31 - 15) / 2
        z[:] = ((np.arange(q)[:, None] * 5 + np.arange(d) * 2) % 29 - 14) / 2
    elif pattern == "extreme":
        x[:, ::2] = 8
        x[:, 1::2] = -8
        z[:] = x[0]
        z[1::2] *= -1
        x[-1].fill(1.1)
        z[-1].fill(0)
    elif pattern == "subnormal":
        x.fill(2**-14 - 2**-24)
        z[:] = -x[0]
        z[0] = x[0]
    return x, z


@pytest.mark.parametrize("case", CASES)
def test_contract_and_reused_addresses(device, case):
    host = arrays(*case)
    backings = [
        torch.full((a.size + 64,), -13, dtype=torch.float16, device=device)
        for a in host
    ]
    tensors = [b[16: 16 + a.size].view(a.shape) for a, b in zip(host, backings)]
    changed = tuple(
        (a * np.float16(-0.5) + np.float16(0.25)).astype(np.float16) for a in host
    )
    variants = (
        (run, run_rows, run_batched)
        if case[:3] in ((2, 33, 65), (129, 257, 512), (1, 257, 1025))
        else (run,)
    )
    for values in (host, changed):
        for tensor, value in zip(tensors, values):
            tensor.copy_(torch.from_numpy(value))
        golden = reference(*values)
        for invoke in variants:
            actual = invoke(*tensors)
            torch.npu.synchronize()
            assert actual.dtype == torch.float32 and tuple(actual.shape) == golden.shape
            assert actual.is_contiguous() and actual.device == tensors[0].device
            got = actual.cpu().numpy()
            assert np.isfinite(got).all() and (got >= 0).all()
            assert (got[golden == 0] == 0).all()
            np.testing.assert_allclose(got, golden, atol=1e-5, rtol=2e-5)
        for tensor, value, backing in zip(tensors, values, backings):
            np.testing.assert_array_equal(tensor.cpu().numpy(), value)
            assert (backing[:16] == -13).all().item()
            assert (backing[16 + value.size:] == -13).all().item()
