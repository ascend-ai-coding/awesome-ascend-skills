# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：语法糖。验证独立网格、完整输出与数值预算，不验证注意力特有契约。

三个独立 launch 由同一 stream 排序；每个阶段拥有不同的完整输出网格。
测试 FP16/BF16、短 M 与 M 尾块、非方阵、多个 batch、最小/最大 K/N、空/短/整行前缀、1/7 worker、重复执行及
输入保持不变与输出 guard。参考从实际量化输入在 FP64 中计算，不复制 kernel 的分块或 P 舍入。
另用零分数、单位置载荷检验可精确表示的均匀权重，避免误差预算掩盖遗漏贡献。
排除位置的载荷扰动必须不影响输出；正负分数端点搭配常量载荷验证指数范围及最终消费。
"""

import pytest
import torch

from .kernel import make_kernels
from .reference import reference_and_bound


def execute(kernels, x, y, values, lengths, output, workers):
    b, m, _ = x.shape
    n = y.shape[1]
    scores = torch.empty((b, m, n), dtype=torch.float32, device=x.device)
    probabilities = torch.empty((b, m, n), dtype=x.dtype, device=x.device)
    score, normalize, product = kernels
    score[None, workers](x, y, scores)
    normalize[None, workers](scores, lengths, probabilities)
    product[None, workers](probabilities, values, output)
    torch.npu.synchronize()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 64, 64, 128, 64),
        (2, 192, 192, 384, 320),
        (2, 128, 1024, 2048, 512),
        (2, 1, 64, 128, 64),
        (2, 3, 192, 256, 128),
        (2, 17, 64, 128, 192),
        (2, 65, 128, 256, 64),
        (1, 129, 256, 128, 320),
    ],
)
def test_rectangular_chain(shape, dtype, device):
    b, m, k, n, d = shape
    generator = torch.Generator().manual_seed(43)
    cpu = [
        torch.rand(s, generator=generator).mul(2).sub(1).to(dtype)
        for s in [(b, m, k), (b, n, k), (b, n, d)]
    ]
    prefixes = torch.tensor([0, 1, 63, 64, 65, n - 1, n], dtype=torch.int32)
    lengths = prefixes[torch.arange(b * m) % len(prefixes)].reshape(b, m)
    expected, bound = reference_and_bound(*cpu, lengths)
    inputs = [t.to(device) for t in cpu]
    device_lengths = lengths.to(device)
    dtype_name = str(dtype).split(".")[-1]
    kernels = make_kernels(b, m, k, n, d, dtype_name)
    backing = torch.full((b * m * d + 64,), 19, dtype=dtype, device=device)
    output = backing[32:-32].view(b, m, d)
    for workers in (1, 7):
        output.fill_(float("nan"))
        execute(kernels, *inputs, device_lengths, output, workers)
        got = output.cpu().double()
        assert torch.isfinite(got).all()
        assert ((got - expected).abs() <= bound).all()
        assert (got[lengths == 0] == 0).all()
        assert torch.equal(backing[:32].cpu(), torch.full((32,), 19, dtype=dtype))
        assert torch.equal(backing[-32:].cpu(), torch.full((32,), 19, dtype=dtype))
        for actual, original in zip(inputs, cpu):
            assert torch.equal(actual.cpu(), original)
        assert torch.equal(device_lengths.cpu(), lengths)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("keys", [128, 2048])
def test_uniform_weights_preserve_every_payload_column(dtype, keys, device):
    b, m, k, n, d = 2, 192, 64, keys, 320
    x = torch.zeros((b, m, k), dtype=dtype, device=device)
    y = torch.zeros((b, n, k), dtype=dtype, device=device)
    values = torch.zeros((b, n, d), dtype=dtype)
    signs = torch.where(torch.arange(d) % 2 == 0, 1.0, -1.0).to(dtype)
    values[0, 0] = signs
    values[1, -1] = -signs / 2
    prefixes = torch.tensor([0, 1, 63, 64, 65, n - 1, n], dtype=torch.int32)
    lengths = prefixes[torch.arange(b * m) % len(prefixes)].reshape(b, m)
    valid = torch.arange(n)[None, None, :] < lengths[..., None]
    weights = valid.double() / lengths.clamp_min(1)[..., None]
    expected = (weights @ values.double()).to(dtype)
    output = torch.empty((b, m, d), dtype=dtype, device=device)
    kernels = make_kernels(b, m, k, n, d, str(dtype).split(".")[-1])
    execute(kernels, x, y, values.to(device), lengths.to(device), output, 7)
    assert torch.equal(output.cpu(), expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_excluded_payload_does_not_change_output(dtype, device):
    b, m, k, n, d = 2, 64, 64, 128, 64
    x = torch.zeros((b, m, k), dtype=dtype)
    y = torch.zeros((b, n, k), dtype=dtype)
    values = torch.full((b, n, d), 0.25, dtype=dtype)
    values[1] = -0.5
    prefixes = torch.tensor([0, 1, 63, 64, 65], dtype=torch.int32)
    lengths = prefixes[torch.arange(b * m) % len(prefixes)].reshape(b, m)
    expected, bound = reference_and_bound(x, y, values, lengths)
    inputs = [x.to(device), y.to(device), values.to(device)]
    device_lengths = lengths.to(device)
    kernels = make_kernels(b, m, k, n, d, str(dtype).split(".")[-1])
    backing = torch.full((b * m * d + 64,), 19, dtype=dtype, device=device)
    output = backing[32:-32].view(b, m, d)
    first = None
    for excluded_value in (1.0, -1.0):
        inputs[2][:, 65:, :] = excluded_value
        output.fill_(float("nan"))
        execute(kernels, *inputs, device_lengths, output, 2)
        got = output.cpu()
        assert torch.isfinite(got).all()
        assert ((got.double() - expected).abs() <= bound).all()
        assert (got[lengths == 0] == 0).all()
        assert (backing[:32] == 19).all() and (backing[-32:] == 19).all()
        if first is not None:
            assert torch.equal(got, first)
        first = got


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_score_endpoints_preserve_constant_payload(dtype, device):
    b, m, k, n, d = 2, 64, 1024, 128, 64
    # Attain scores +32/-32, where a guessed fixed shift can fail after narrowing.
    x = torch.ones((b, m, k), dtype=dtype, device=device)
    x[1] = -1
    y = torch.ones((b, n, k), dtype=dtype, device=device)
    values = torch.full((b, n, d), 0.25, dtype=dtype, device=device)
    values[1] = -0.5
    lengths = torch.full((b, m), n, dtype=torch.int32, device=device)
    lengths[:, 0] = 0
    expected = torch.full((b, m, d), 0.25, dtype=dtype)
    expected[1] = -0.5
    expected[:, 0] = 0
    output = torch.full((b, m, d), float("nan"), dtype=dtype, device=device)
    kernels = make_kernels(b, m, k, n, d, str(dtype).split(".")[-1])
    execute(kernels, x, y, values, lengths, output, 2)
    assert torch.isfinite(output).all()
    assert torch.equal(output.cpu(), expected)
