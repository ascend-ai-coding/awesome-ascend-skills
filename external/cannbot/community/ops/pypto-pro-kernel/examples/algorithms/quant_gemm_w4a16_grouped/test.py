# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""分组量化的舍入契约、逻辑轴映射、尾部、工作核分配、重复写出和守卫。

参考先执行规定的 FP32 反量化及 FP16 RNE，再用 FP64 计算 GEMM。
一般结果使用固定 atol=rtol=2e-3；one-hot 与舍入边界必须精确返回权重。
"""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import reference


SHAPES = [
    (9, 160, 144, 32),
    (41, 256, 528, 64),
    (97, 512, 1040, 32),
    (1, 32, 16, 32),
    (7, 96, 48, 32),
    (129, 512, 1040, 64),
]
KINDS = ["random", "zero", "layout", "cancel", "onehot", "rounding"]


def data(m, k, n, g, kind):
    rng = np.random.default_rng(m * 1009 + k * 13 + n + g)
    a = rng.uniform(-1, 1, (m, k)).astype(np.float16)
    q = rng.integers(0, 16, (k, n), dtype=np.uint8)
    scale = rng.uniform(2**-8, 2**-3, (k // g, n)).astype(np.float32)
    zero = rng.integers(0, 16, (k // g, n), dtype=np.int32)
    bias = rng.uniform(-1, 1, n).astype(np.float32)
    if kind == "zero":
        q = np.repeat(zero, g, axis=0).astype(np.uint8)
    elif kind == "layout":
        q = ((3 * np.arange(k)[:, None] + 5 * np.arange(n)[None, :]) % 16).astype(
            np.uint8
        )
        zero = ((2 * np.arange(k // g)[:, None] + np.arange(n)[None, :]) % 16).astype(
            np.int32
        )
        scale = np.exp2(
            -7 + ((np.arange(k // g)[:, None] + np.arange(n)[None, :]) % 4)
        ).astype(np.float32)
    elif kind == "cancel":
        a[:, 1::2] = -a[:, 0::2]
        q[1::2] = q[0::2]
    if kind in ("onehot", "rounding"):
        positions = (
            np.resize(
                np.array([0, 1, g - 2, g - 1, g, min(g + 1, k - 1), k - 2, k - 1]), m
            )
            % k
        )
        a[:] = 0
        a[np.arange(m), positions] = 1
        bias[:] = 0
        if kind == "rounding":
            lo = np.float16(1 / 32)
            hi = np.nextafter(lo, np.float16(np.inf), dtype=np.float16)
            hi2 = np.nextafter(hi, np.float16(np.inf), dtype=np.float16)
            mids = [
                np.float32((float(lo) + float(hi)) / 2),
                np.float32((float(hi) + float(hi2)) / 2),
            ]
            values = np.array(
                [
                    item
                    for mid in mids
                    for item in (
                        np.nextafter(mid, np.float32(-np.inf)),
                        mid,
                        np.nextafter(mid, np.float32(np.inf)),
                    )
                ],
                dtype=np.float32,
            )
            scale[:] = np.resize(values, scale.shape)
            zero[:] = 7
            q[0::2] = 8
            q[1::2] = 6
    packed = q[0::2] | (q[1::2] << np.uint8(4))
    return a, packed, scale, zero, bias


def guarded(value, device, head=64):
    types = {
        np.dtype("float16"): torch.float16,
        np.dtype("float32"): torch.float32,
        np.dtype("uint8"): torch.uint8,
        np.dtype("int32"): torch.int32,
    }
    sentinel = 197 if value.dtype == np.uint8 else 8192
    backing = torch.full(
        (value.size + head + 64,), sentinel, dtype=types[value.dtype], device=device
    )
    tensor = backing[slice(head, head + value.size)].view(*value.shape)
    tensor.copy_(torch.from_numpy(value).to(device))
    return tensor, backing, sentinel


@pytest.mark.parametrize("m,k,n,g", SHAPES)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("workers", [1, 3])
def test_grouped_int4_contract(device, m, k, n, g, kind, workers):
    host = data(m, k, n, g, kind)
    expected = reference(*host, g)
    inputs = [guarded(value, device) for value in host]
    output = guarded(np.empty((m, n), dtype=np.float32), device, head=8)
    for poison in (float("nan"), -12345.0):
        output[0].fill_(poison)
        run(*(item[0] for item in inputs), output[0], g, workers=workers)
        torch.npu.synchronize()
        actual = output[0].cpu().numpy()
        np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-3)
        if kind in ("onehot", "rounding"):
            np.testing.assert_array_equal(actual, expected.astype(np.float32))
        for (tensor, _, _), value in zip(inputs, host):
            np.testing.assert_array_equal(
                tensor.cpu().numpy().view(np.uint8), value.view(np.uint8)
            )
        for tensor, backing, sentinel in inputs + [output]:
            np.testing.assert_array_equal(
                backing[: tensor.storage_offset()].cpu().numpy(), sentinel
            )
            np.testing.assert_array_equal(backing[-64:].cpu().numpy(), sentinel)
