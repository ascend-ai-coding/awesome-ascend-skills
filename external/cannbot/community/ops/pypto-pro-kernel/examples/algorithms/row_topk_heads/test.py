# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""完整总序、实际剪枝边界与位模式保持的独立板测。

参考直接排序完整输入，不复用候选构建或归并顺序。集中获胜项并在后续插入
更强键，验证旧并列项不会在局部容量边界丢失。覆盖不同workers、寄存器尾部、
超长行、稠密输出的非零偏移、重复毒值初始化、输入不变和前后守卫。
"""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import reference


def data(m, n, kind):
    rng = np.random.default_rng(714853 + 137 * m + n)
    x = rng.standard_normal((m, n)).astype(np.float32)
    if kind == "ties":
        x = rng.integers(-3, 4, (m, n)).astype(np.float32)
    elif kind == "constant":
        x.fill(-7)
    elif kind == "bits":
        bits = np.array(
            [
                0,
                0x80000000,
                1,
                2,
                0x7FFFFF,
                0x80000001,
                0x807FFFFF,
                0x3F7FFFFF,
                0x3F800000,
                0x3F800001,
                0x7F7FFFFF,
                0xFF7FFFFF,
                0x00800000,
                0x80800000,
            ],
            dtype=np.uint32,
        )
        x = np.resize(bits, (m, n)).view(np.float32).copy()
    elif kind == "zeros":
        x = (
            np.resize(np.array([0x80000000, 0], dtype=np.uint32), (m, n))
            .view(np.float32)
            .copy()
        )
    elif kind == "subnormal":
        bits = rng.integers(1, 0x800000, (m, n), dtype=np.uint32)
        bits |= rng.integers(0, 2, (m, n), dtype=np.uint32) << np.uint32(31)
        x = bits.view(np.float32).copy()
    elif kind == "sorted":
        x = np.sort(x, axis=1)
    return x


def guarded(value, offset, device):
    dtype = torch.float32 if value.dtype == np.float32 else torch.int32
    backing = torch.full((value.size + offset + 64,), 8192, dtype=dtype, device=device)
    tensor = backing[slice(offset, offset + value.size)].reshape(value.shape)
    tensor.copy_(torch.from_numpy(value))
    return tensor, backing


CASES = [
    (m, n, k, "ties", 4)
    for m, n, k in (
        (1, 1, 1),
        (3, 7, 7),
        (2, 31, 17),
        (2, 32, 32),
        (5, 33, 33),
        (7, 65, 64),
        (3, 257, 256),
        (32, 513, 127),
    )
]
CASES += [
    (m, n, k, kind, w)
    for m, n, k in ((1, 4097, 256), (3, 65537, 128), (2, 32769, 256))
    for kind, w in (("random", 4), ("constant", 16), ("sorted", 1))
]
CASES += [
    (m, n, k, kind, w)
    for m, n, k in ((5, 129, 97), (3, 4097, 256), (1, 65537, 32))
    for kind, w in (("bits", 1), ("bits", 16), ("zeros", 4))
]
CASES += [
    (m, n, k, "subnormal", w)
    for m, n, k in ((2, 257, 128), (1, 8193, 256))
    for w in (1, 16)
]
CASES += [(1, 64 * k + 1, k, "pruning", 4) for k in (2, 8, 64, 256)]
CASES += [(32, 65537, 256, "random", w) for w in (1, 16)]


@pytest.mark.parametrize("m,n,k,kind,workers", CASES)
def test_topk(device, m, n, k, kind, workers):
    x = data(m, n, kind)
    if kind == "pruning":
        x.fill(-1)
        x[:, slice(None, 64 * k, 64)] = 1
        x[:, 64 * k] = 2
    values, indices = reference(x, k)
    dx, bx = guarded(x, 17, device)
    dv, bv = guarded(np.zeros((m, k), dtype=np.float32), 9, device)
    di, bi = guarded(np.zeros((m, k), dtype=np.int32), 13, device)
    for iteration in range(2):
        dv.fill_(float("nan") if iteration == 0 else -99)
        di.fill_(-989898)
        run(dx, dv, di, k, workers=workers)
        torch.npu.synchronize()
        np.testing.assert_array_equal(di.cpu().numpy(), indices)
        np.testing.assert_array_equal(
            dv.cpu().numpy().view(np.uint32), values.view(np.uint32)
        )
        np.testing.assert_array_equal(
            dx.cpu().numpy().view(np.uint32), x.view(np.uint32)
        )
        for tensor, backing in ((dx, bx), (dv, bv), (di, bi)):
            off = tensor.storage_offset()
            assert bool(torch.all(backing[:off].cpu() == 8192))
            assert bool(
                torch.all(backing[slice(off + tensor.numel(), None)].cpu() == 8192)
            )
