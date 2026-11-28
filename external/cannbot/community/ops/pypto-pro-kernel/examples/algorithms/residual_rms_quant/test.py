# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""逐行残差、RMSNorm 和动态量化的独立验收。

语义：输入 FP16 x/residual 和 FP32 gamma，先 FP32 相加并以 FP16 RNE 形成 r，再按该 r 计算 RMSNorm、行 scale 和 INT8 q。范围 M=1..128、N=1..16385，
|x|/|residual|<=8、|gamma|<=2。
物理访问：连续张量使用非零且非 DMA 对齐的 GM 起点；每份输入输出都有前后守卫。覆盖寄存器边界、极短行、最大行宽、多 worker、全零约定和最小合法非零输入。
同步：同一组输入用两种不同输出毒值重复调用，核验输出全部定义、输入不变和守卫不变；kernel.py 解释片上依赖的处理。
独立性：reference.py 消费实际输入，从数学定义重新计算 r，后续参考不依赖设备输出。r 逐位相等；s atol=1e-6/rtol=2e-4；参考 z 距半整数大于 0.02 时 q 精确，其余整数差至多 1；全零行
s=1/q=0 精确。阈值来自任务契约。
"""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import reference


CASES = [
    (rows, cols, "random", workers)
    for rows, cols in ((1, 1), (3, 65), (17, 257), (33, 4097), (7, 16385))
    for workers in (1, 4)
] + [
    (2, 7, "mixed", 4),
    (3, 31, "small", 1),
    (5, 63, "random", 4),
    (3, 64, "cancel", 1),
    (4, 127, "sparse_peak", 4),
    (3, 128, "random", 1),
    (5, 129, "small", 4),
    (3, 255, "zero", 4),
    (2, 256, "random", 1),
    (7, 257, "zero_gamma", 4),
    (2, 4095, "mixed", 4),
    (3, 4096, "sparse_peak", 1),
    (1, 16383, "small", 4),
    (2, 16384, "random", 1),
    (128, 1, "mixed", 4),
    (128, 16385, "random", 4),
    (1, 1, "subnormal", 4),
    (3, 65, "min_subnormal", 4),
    (5, 4097, "subnormal", 1),
    (2, 16385, "min_subnormal", 4),
    (3, 65, "tiny_pair", 4),
    (2, 16385, "tiny_pair", 1),
    (3, 65, "disjoint_scales", 4),
    (3, 65, "extremes", 4),
    (32, 4096, "random", 4),
    (128, 8192, "random", 4),
]


def inputs(rows, cols, kind, workers):
    rng = np.random.default_rng(173 * rows + 101 * cols + workers)
    x = rng.uniform(-8, 8, (rows, cols)).astype(np.float16)
    residual = rng.uniform(-8, 8, (rows, cols)).astype(np.float16)
    gamma = rng.uniform(-2, 2, cols).astype(np.float32)
    if kind == "zero":
        x.fill(0)
        residual.fill(0)
    elif kind == "cancel":
        residual = -x
    elif kind == "zero_gamma":
        gamma.fill(0)
    elif kind == "mixed":
        residual[::2] = -x[::2]
        gamma[::3] = 0
    elif kind == "sparse_peak":
        x.fill(0)
        residual.fill(0)
        x[:, -1] = 8
        gamma.fill(1)
    elif kind == "small":
        x = (x * np.float16(2**-10)).astype(np.float16)
        residual = (residual * np.float16(2**-10)).astype(np.float16)
    elif kind == "subnormal":
        gamma.fill(np.float32(1e-40))
    elif kind == "min_subnormal":
        least = np.nextafter(np.float32(0), np.float32(1))
        gamma[:] = (
            np.resize(np.array([1, -1, 0, 2, -2, 16, -16], np.float32), gamma.shape)
            * least
        )
    elif kind == "tiny_pair":
        least = np.nextafter(np.float16(0), np.float16(1))
        x[:] = np.resize(np.array([1, -1, 0, 2, -2], np.float16), x.shape) * least
        residual.fill(0)
        gamma.fill(np.nextafter(np.float32(0), np.float32(1)))
    elif kind == "disjoint_scales":
        x.fill(4)
        x[:, 0] = 0
        residual.fill(0)
        least = np.nextafter(np.float32(0), np.float32(1))
        gamma[:] = (
            np.resize(np.array([1, 2, 4, -1, -2, -4], np.float32), gamma.shape) * least
        )
        gamma[0] = 2
    elif kind == "extremes":
        x[:] = np.resize(np.array([-8, 8, 8, -8], np.float16), x.shape)
        residual[:] = np.resize(np.array([-8, 8, -8, 8], np.float16), x.shape)
        gamma[:] = np.resize(np.array([-2, 2], np.float32), gamma.shape)
    return x, residual, gamma


def guarded(array, device, fill=None):
    host = torch.from_numpy(array)
    sentinel = -99 if host.dtype == torch.int8 else 123.0
    backing = torch.full(
        (host.numel() + 138,), sentinel, dtype=host.dtype, device=device
    )
    value = backing[slice(67, 67 + host.numel())].view(host.shape)
    value.copy_(host) if fill is None else value.fill_(fill)
    return value, backing, sentinel


@pytest.mark.parametrize("rows,cols,kind,workers", CASES)
def test_residual_rms_quant(device, rows, cols, kind, workers):
    originals = inputs(rows, cols, kind, workers)
    expected_r, expected_s, z, expected_q = reference(*originals)
    source = [guarded(array, device) for array in originals]
    outputs = [
        guarded(np.empty((rows, cols), np.float16), device, 123),
        guarded(np.empty(rows, np.float32), device, 123),
        guarded(np.empty((rows, cols), np.int8), device, -99),
    ]
    for repeat in range(2):
        for value, _, sentinel in outputs:
            value.fill_(sentinel if repeat == 0 else -sentinel)
        run(
            *(item[0] for item in source),
            *(item[0] for item in outputs),
            workers=workers,
        )
        torch.npu.synchronize()
        r, s, q = (item[0].cpu().numpy() for item in outputs)
        np.testing.assert_array_equal(r.view(np.uint16), expected_r.view(np.uint16))
        np.testing.assert_allclose(s, expected_s, atol=1e-6, rtol=2e-4)
        distance = np.abs(z - (np.floor(z) + 0.5))
        error = np.abs(q.astype(np.int16) - expected_q.astype(np.int16))
        assert np.all(error[distance > 0.02] == 0)
        assert np.all(error <= 1)
        zero_rows = np.all(z == 0, axis=1)
        assert np.all(s[zero_rows] == 1) and np.all(q[zero_rows] == 0)
        for value, backing, sentinel in source + outputs:
            host = backing.cpu().numpy()
            assert np.all(host[:67] == sentinel)
            assert np.all(host[slice(67 + value.numel(), None)] == sentinel)
    for item, original in zip(source, originals):
        np.testing.assert_array_equal(
            item[0].cpu().numpy().view(np.uint8), original.view(np.uint8)
        )
