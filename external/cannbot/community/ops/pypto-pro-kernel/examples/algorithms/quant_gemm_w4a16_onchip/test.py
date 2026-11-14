# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""验证片上反量化→Cube 的完整契约，不复用 kernel 的分块与索引算法。

位置编码与 one-hot 区分分组/偶奇行；舍入中点两侧覆盖正负 RNE。
各轴尾部与多条带检查槽复用，重复毒值输出检查覆盖，连续偏移视图检查守卫。
"""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import reference


def data(m, k, n, g, kind):
    rng = np.random.default_rng(953903 + m * 1093 + 31 * k + 151 * n + g)
    a = rng.uniform(-1, 1, (m, k)).astype(np.float16)
    q = rng.integers(0, 16, (k, n), dtype=np.uint8)
    zero = rng.integers(0, 16, (k // g, n), dtype=np.int32)
    scale = rng.uniform(2**-8, 2**-3, (k // g, n)).astype(np.float32)
    bias = rng.uniform(-1, 1, n).astype(np.float32)
    if kind == "layout":
        q = ((3 * np.arange(k)[:, None] + 5 * np.arange(n)[None, :]) % 16).astype(
            np.uint8
        )
        scale = np.exp2(
            -8 + ((np.arange(k // g)[:, None] + np.arange(n)[None, :]) % 5)
        ).astype(np.float32)
    elif kind == "cancel":
        a[:, 1::2] = -a[:, 0::2]
        q[1::2] = q[0::2]
    if kind in ("onehot", "rounding"):
        positions = np.resize(np.array([k - 1, g, g - 1, 1, k - 2, 0, g + 1]), m) % k
        a.fill(0)
        a[np.arange(m), positions] = 1
        bias.fill(0)
        if kind == "rounding":
            lo = np.float16(1 / 32)
            hi = np.nextafter(lo, np.float16(np.inf))
            hi2 = np.nextafter(hi, np.float16(np.inf))
            values = []
            for left, right in ((lo, hi), (hi, hi2)):
                midpoint = np.float32((float(left) + float(right)) / 2)
                values.extend(
                    (
                        np.nextafter(midpoint, np.float32(-np.inf)),
                        midpoint,
                        np.nextafter(midpoint, np.float32(np.inf)),
                    )
                )
            scale[:] = np.resize(np.array(values, np.float32), scale.shape)
            zero.fill(7)
            q[0::2] = 8
            q[1::2] = 6
    packed = q[0::2] | (q[1::2] << np.uint8(4))
    return a, packed, scale, zero, bias


def guarded(value, device):
    types = {
        np.dtype("float16"): torch.float16,
        np.dtype("float32"): torch.float32,
        np.dtype("uint8"): torch.uint8,
        np.dtype("int32"): torch.int32,
    }
    sentinel = 197 if value.dtype == np.uint8 else 8192
    backing = torch.full(
        (value.size + 81,), sentinel, dtype=types[value.dtype], device=device
    )
    tensor = backing[slice(17, 17 + value.size)].reshape(value.shape)
    tensor.copy_(torch.from_numpy(value))
    return tensor, backing, sentinel


CASES = [
    (m, k, n, g, kind, w)
    for m, k, n, g in (
        (1, 64, 512, 64),
        (15, 320, 80, 64),
        (33, 576, 112, 32),
        (2, 128, 32, 32),
    )
    for kind, w in (("layout", 4), ("onehot", 1), ("rounding", 16))
]
CASES += [
    (m, k, n, g, kind, w)
    for m, k, n, g in (
        (1, 2048, 512, 64),
        (127, 1984, 496, 32),
        (49, 1152, 176, 64),
        (3, 576, 16, 64),
        (7, 1984, 48, 32),
        (129, 2048, 512, 64),
    )
    for kind, w in (("random", 16), ("cancel", 4), ("onehot", 1))
]
CASES += [
    (128, 2048, 512, g, kind, w)
    for g, kind, w in ((32, "random", 16), (64, "onehot", 1), (64, "cancel", 4))
]


@pytest.mark.parametrize("m,k,n,g,kind,workers", CASES)
def test_onchip_w4a16(device, m, k, n, g, kind, workers):
    host = data(m, k, n, g, kind)
    expected, term_sum = reference(*host, g)
    inputs = [guarded(a, device) for a in host]
    output = guarded(np.empty((m, n), np.float32), device)
    for poison in (float("nan"), -777):
        output[0].fill_(poison)
        run(*(a for a, _, _ in inputs), output[0], g, workers=workers)
        torch.npu.synchronize()
        actual = output[0].cpu().numpy()
        bound = 2e-3 + 3e-4 * term_sum
        error = np.abs(actual.astype(np.float64) - expected)
        assert np.isfinite(actual).all() and (error <= bound).all(), (
            m,
            k,
            n,
            g,
            kind,
            float(np.nanmax(error / bound)),
        )
        if kind in ("onehot", "rounding"):
            np.testing.assert_array_equal(actual, expected.astype(np.float32))
        for (tensor, _, _), value in zip(inputs, host):
            np.testing.assert_array_equal(
                tensor.cpu().numpy().view(np.uint8), value.view(np.uint8)
            )
        for tensor, backing, sentinel in inputs + [output]:
            assert bool(torch.all(backing[:17].cpu() == sentinel))
            assert bool(
                torch.all(backing[slice(17 + tensor.numel(), None)].cpu() == sentinel)
            )
