# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CSR softmax 加权聚合的独立数值与边界验收。

语义：FP32 logits[E]/table[V,D]，INT32 offsets[R+1]/indices[E]；段界单调、首0末E，索引位于[0,V)。R=1..257、E=0..65537、V=1..4097、D=1..129，
logits∈[-20,20]、table∈[-2,2]且有限。按段 softmax 后聚合索引载荷，输出 FP32 out[R,D]/lse[R]。
物理访问：保持公共布局连续，所有输入输出起点偏移67个元素，并有前后守卫。覆盖空输入、空段、极长单段、偏斜、重复索引、列尾和两种 worker 数。
同步：每次使用不同输出毒值重复调用，核验输出完全定义、输入不变及所有守卫；kernel 的分段状态交接独立验证完整调用。
独立性：reference.py 从实际 FP32 输入按数学定义计算 FP64 参考，不使用设备输出或候选的分块/平移选择。有限输出 atol=rtol=3e-4，空行 out 精确零、lse 精确负无穷。端点/常数分数验证归一化范围；接
近抵消的 FP32 载荷防止低精度转换被随机数据掩盖。
"""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import reference


CASES = [
    (1, 0, 1, 1, "uniform", "random", "random"),
    (3, 1, 7, 1, "one", "random", "random"),
    (7, 65, 17, 63, "uniform", "random", "random"),
    (3, 127, 129, 64, "one", "random", "random"),
    (17, 129, 33, 65, "skew", "random", "random"),
    (257, 65, 17, 129, "uniform", "random", "random"),
    (3, 4097, 129, 65, "skew", "random", "random"),
    (1, 65537, 4097, 129, "uniform", "random", "random"),
    (257, 65537, 4097, 129, "one", "random", "random"),
    (257, 65537, 4097, 129, "skew", "random", "random"),
    (3, 65, 17, 65, "uniform", "low", "random"),
    (1, 65537, 4097, 129, "uniform", "high", "random"),
    (257, 65, 17, 129, "uniform", "low", "random"),
    (3, 4097, 129, 63, "skew", "zero", "random"),
    (3, 65, 2, 65, "uniform", "zero", "cancel"),
    (257, 0, 1, 129, "uniform", "random", "random"),
    (256, 65536, 4096, 64, "uniform", "random", "random"),
    (256, 65536, 4096, 64, "skew", "random", "random"),
]


def make_inputs(rows, edges, vocab, width, distribution, score_kind, payload):
    rng = np.random.default_rng(rows * 73 + edges + width)
    counts = np.zeros(rows, np.int32)
    if distribution == "one":
        counts[-1] = edges
    elif distribution == "skew":
        counts[0] = edges * 9 // 10
        for index in range(edges - int(counts[0])):
            counts[index % rows] += 1
    else:
        counts[:] = edges // rows
        counts[: edges % rows] += 1
    offsets = np.concatenate((np.zeros(1, np.int32), np.cumsum(counts, dtype=np.int32)))
    indices = rng.integers(0, vocab, edges, dtype=np.int32)
    if distribution == "one":
        indices.fill(vocab - 1)
    logits = rng.uniform(-20, 20, edges).astype(np.float32)
    if score_kind != "random":
        logits.fill({"low": -20, "high": 20, "zero": 0}[score_kind])
    table = rng.uniform(-2, 2, (vocab, width)).astype(np.float32)
    if payload == "cancel":
        indices[:] = np.arange(edges) % 2
        table[0].fill(np.float32(1.0005))
        table[1].fill(np.float32(-1.0004))
    return logits, offsets, indices, table


def guarded(array, device):
    host = torch.from_numpy(array)
    backing = torch.full((host.numel() + 138,), -99, dtype=host.dtype, device=device)
    value = backing[slice(67, 67 + host.numel())].view(host.shape)
    value.copy_(host)
    return value, backing


@pytest.mark.parametrize("spec", CASES)
@pytest.mark.parametrize("workers", [1, 4])
def test_csr_softmax_aggregate(device, spec, workers):
    originals = make_inputs(*spec)
    expected, expected_lse = reference(*originals)
    source = [guarded(value, device) for value in originals]
    outputs = [
        guarded(np.zeros(expected.shape, np.float32), device),
        guarded(np.zeros(expected_lse.shape, np.float32), device),
    ]
    empty = np.diff(originals[1]) == 0
    for poison in (123, -77):
        for value, _ in outputs:
            value.fill_(poison)
        run(
            *(pair[0] for pair in source),
            *(pair[0] for pair in outputs),
            workers=workers,
        )
        torch.npu.synchronize()
        out, lse = (pair[0].cpu().numpy() for pair in outputs)
        np.testing.assert_allclose(out, expected, atol=3e-4, rtol=3e-4)
        np.testing.assert_allclose(lse, expected_lse, atol=3e-4, rtol=3e-4)
        assert np.all(out[empty] == 0)
        assert np.all(np.isneginf(lse[empty]))
        for value, backing in source + outputs:
            host = backing.cpu().numpy()
            assert np.all(host[:67] == -99)
            assert np.all(host[slice(67 + value.numel(), None)] == -99)
    for pair, original in zip(source, originals):
        np.testing.assert_array_equal(
            pair[0].cpu().numpy().view(np.uint32), original.view(np.uint32)
        )
