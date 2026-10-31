# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""独立FP64参考验证定点累加精度、可重复性及路由变化。

参考汇总真实乘积，不模拟设备定点数。输入、输出均有连续非零偏移的前后守卫；
重复毒值初始化后逐位比较，并在同一输入地址改为全无效路由。覆盖不同lane组、
分区数量边界、最坏临时空间、同号微小贡献、最大整数和及相消残差。
三次原生调用提供初始化/累加/归并的全局先后关系，线程块屏障不承担跨worker同步。
"""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import reference


def data(t, r, d, kind):
    rng = np.random.default_rng(7013 + 19 * t + 137 * r + d)
    values = rng.uniform(-1, 1, (r, d)).astype(np.float32)
    weights = rng.uniform(0, 1, r).astype(np.float32)
    ids = rng.integers(0, t, r, dtype=np.int32)
    if kind == "hot":
        ids.fill(t - 1)
    elif kind == "mixed":
        ids[::3] = -1
        ids[1::3] = t // 2
    elif kind == "invalid":
        ids.fill(-1)
    elif kind == "cancel":
        ids.fill(0)
        for pos in range(0, r - 1, 2):
            values[pos + 1] = -values[pos]
            weights[pos + 1] = weights[pos]
    elif kind == "tiny_coherent":
        ids.fill(t - 1)
        weights.fill(1)
        levels = np.array(
            [0.49, 0.51, 0.99, 1.01, -0.49, -0.51, -0.99, -1.01], np.float32
        ) * np.float32(2.0**-30)
        values[:] = np.resize(levels, d)
    elif kind == "extreme":
        ids.fill(0)
        weights.fill(1)
        values[:] = np.resize(np.array([1, -1], np.float32), d)
    elif kind == "residual":
        ids.fill(0)
        weights.fill(1)
        values.fill(1)
        values[1::2] = np.float32(-1 + 2.0**-24)
    return values, ids, weights


def guarded(array, offset, device):
    dtype = torch.float32 if array.dtype == np.float32 else torch.int32
    backing = torch.full((array.size + offset + 64,), 8192, dtype=dtype, device=device)
    tensor = backing[slice(offset, offset + array.size)].reshape(array.shape)
    tensor.copy_(torch.from_numpy(array))
    return tensor, backing


CASES = [
    (t, r, d, kind, w)
    for t, r, d in (
        (1, 0, 1),
        (7, 0, 65),
        (1, 1, 1),
        (9, 37, 7),
        (17, 129, 65),
        (5, 257, 129),
    )
    for kind, w in (("uniform", 1), ("mixed", 16))
]
CASES += [
    (t, r, d, kind, w)
    for t, r, d in ((257, 4097, 127), (4096, 32768, 512), (1, 32768, 65))
    for kind, w in (("uniform", 4), ("hot", 16), ("cancel", 4))
]
CASES += [
    (t, 32768, 8, kind, w)
    for t, w in ((1, 1), (1, 16), (17, 4))
    for kind in ("tiny_coherent", "extreme", "residual")
]
CASES += [(3, 65, d, "mixed", 4) for d in (2, 3, 4, 15, 16, 63, 64)]
CASES += [(t, 65, 64, "mixed", 4) for t in (1024, 1025, 1366, 2049)]


@pytest.mark.parametrize("t,r,d,kind,workers", CASES)
def test_combine(device, t, r, d, kind, workers):
    host = data(t, r, d, kind)
    expected, scale = reference(*host, t)
    guarded_inputs = [guarded(a, 17, device) for a in host]
    output = guarded(np.empty((t, d), dtype=np.float32), 9, device)
    previous = None
    for iteration in range(4):
        output[0].fill_(float("nan") if iteration % 2 else 999)
        run(*(a for a, _ in guarded_inputs), output[0], workers=workers)
        torch.npu.synchronize()
        actual = output[0].cpu().numpy()
        bound = 2e-5 + 2e-5 * scale
        error = np.abs(actual.astype(np.float64) - expected)
        assert np.isfinite(actual).all() and (error <= bound).all(), (
            t,
            r,
            d,
            kind,
            float(np.nanmax(error / bound)),
        )
        if previous is not None:
            np.testing.assert_array_equal(actual.view(np.uint32), previous)
        previous = actual.view(np.uint32).copy()
        np.testing.assert_array_equal(
            actual[scale == 0], np.zeros(np.count_nonzero(scale == 0), np.float32)
        )
    for (tensor, backing), value in zip(guarded_inputs, host):
        np.testing.assert_array_equal(
            tensor.cpu().numpy().view(np.uint8), value.view(np.uint8)
        )
    for tensor, backing in guarded_inputs + [output]:
        off = tensor.storage_offset()
        assert bool(torch.all(backing[:off].cpu() == 8192))
        assert bool(torch.all(backing[slice(off + tensor.numel(), None)].cpu() == 8192))
    # Reuse the same input allocations with different routing metadata.
    ids = np.full(r, -1, np.int32)
    guarded_inputs[1][0].copy_(torch.from_numpy(ids))
    output[0].fill_(91)
    run(*(a for a, _ in guarded_inputs), output[0], workers=workers)
    torch.npu.synchronize()
    np.testing.assert_array_equal(output[0].cpu().numpy(), np.zeros((t, d), np.float32))
    for (tensor, _), original in zip(guarded_inputs, (host[0], ids, host[2])):
        np.testing.assert_array_equal(
            tensor.cpu().numpy().view(np.uint8), original.view(np.uint8)
        )
    for tensor, backing in guarded_inputs + [output]:
        off = tensor.storage_offset()
        assert bool(torch.all(backing[:off].cpu() == 8192))
        assert bool(torch.all(backing[slice(off + tensor.numel(), None)].cpu() == 8192))
