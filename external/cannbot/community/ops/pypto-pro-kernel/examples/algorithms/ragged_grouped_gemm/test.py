# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""运行期变长分组 GEMM 的独立数值、元数据和物理边界验收。

契约：连续 FP16 A[T,K]/W[G,K,N]，有限值在[-1,1]，INT32 offsets[G+1]单调且首0末T，输出 FP32 C[T,N]。T=1..4096、G=1..32，K/N各为16/32/64/128/256。
物理访问：输入输出均以非零存储偏移67个元素的连续视图传入，前后守卫；涵盖空组、集中、偏斜、组长1/15/16/17/65、K/N边界和最大T/G。workers=1/4分别验证。
同步与重跑：同一输出 buffer 用不同毒值重复调用，检查完整输出、所有输入逐位不变及守卫；另在同一输入 buffer 更新运行期 offsets，检查特化不能固化分组内容。
独立性：reference.py 直接按实际 FP16 输入计算 FP64 数学矩阵乘及绝对项和，不复现 kernel 的 tile、分组分工或累加顺序。每元素误差必须不超过1e-3+3e-4*sum(abs(A*W))。
"""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import reference


CASES = [
    (1, 1, 16, 16, "uniform"),
    (1, 32, 32, 64, "one"),
    (17, 3, 64, 32, "one"),
    (129, 7, 128, 256, "tails"),
    (65, 32, 256, 128, "uniform"),
    (257, 17, 16, 256, "skew"),
    (513, 7, 256, 16, "tails"),
    (4096, 1, 64, 64, "uniform"),
    (4096, 32, 256, 256, "uniform"),
    (4096, 32, 256, 256, "one"),
    (4096, 32, 256, 256, "skew"),
    (2048, 16, 256, 256, "uniform"),
    (2048, 16, 256, 256, "one"),
    (2048, 16, 256, 256, "skew"),
]
CASES += [
    (65, 7, inner, width, "tails")
    for inner in (16, 32, 64, 128, 256)
    for width in (16, 32, 64, 128, 256)
]


def make_offsets(rows, groups, distribution):
    counts = np.zeros(groups, np.int32)
    if distribution == "one":
        counts[-1] = rows
    elif distribution == "tails":
        remaining = rows
        for group, wanted in enumerate([0, 1, 15, 16, 17, 65][: groups - 1]):
            counts[group] = min(remaining, wanted)
            remaining -= int(counts[group])
        counts[-1] = remaining
    elif distribution == "skew":
        counts[0] = rows * 9 // 10
        for index in range(rows - int(counts[0])):
            counts[index % groups] += 1
    else:
        counts[:] = rows // groups
        counts[: rows % groups] += 1
    return np.concatenate((np.zeros(1, np.int32), np.cumsum(counts, dtype=np.int32)))


def guarded(array, device):
    value = torch.from_numpy(array)
    backing = torch.full((value.numel() + 138,), -99, dtype=value.dtype, device=device)
    view = backing[slice(67, 67 + value.numel())].view(value.shape)
    view.copy_(value)
    return view, backing


def verify(call_inputs, output, originals, workers):
    expected, term_scale = reference(*originals)
    run(*(value for value, _ in call_inputs), output[0], workers=workers)
    torch.npu.synchronize()
    actual = output[0].cpu().numpy().astype(np.float64)
    error = np.abs(actual - expected)
    tolerance = 1e-3 + 3e-4 * term_scale
    assert np.all(error <= tolerance), (
        float(np.max(error / tolerance)),
        float(np.max(error)),
    )
    for value, backing in call_inputs + [output]:
        host = backing.cpu().numpy()
        assert np.all(host[:67] == -99)
        assert np.all(host[slice(67 + value.numel(), None)] == -99)
    for (value, _), original in zip(call_inputs, originals):
        np.testing.assert_array_equal(
            value.cpu().numpy().view(np.uint8), original.view(np.uint8)
        )


@pytest.mark.parametrize("spec", CASES)
@pytest.mark.parametrize("workers", [1, 4])
def test_ragged_grouped_gemm(device, spec, workers):
    rows, groups, inner, width, distribution = spec
    rng = np.random.default_rng(rows * 71 + inner + width)
    originals = [
        rng.uniform(-1, 1, (rows, inner)).astype(np.float16),
        rng.uniform(-1, 1, (groups, inner, width)).astype(np.float16),
        make_offsets(rows, groups, distribution),
    ]
    data = [guarded(value, device) for value in originals]
    out = guarded(np.zeros((rows, width), np.float32), device)
    for poison in (123, -77):
        out[0].fill_(poison)
        verify(data, out, originals, workers)


@pytest.mark.parametrize("workers", [1, 4])
def test_runtime_offsets_change_with_fixed_shapes_and_storage(device, workers):
    rng = np.random.default_rng(7727)
    originals = [
        rng.uniform(-1, 1, (129, 64)).astype(np.float16),
        rng.uniform(-1, 1, (7, 64, 64)).astype(np.float16),
        make_offsets(129, 7, "uniform"),
    ]
    data = [guarded(value, device) for value in originals]
    out = guarded(np.zeros((129, 64), np.float32), device)
    for index, distribution in enumerate(("uniform", "one", "tails", "uniform")):
        originals[2] = make_offsets(129, 7, distribution)
        data[2][0].copy_(torch.from_numpy(originals[2]))
        out[0].fill_(123 if index % 2 == 0 else -77)
        verify(data, out, originals, workers)
