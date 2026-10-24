# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""VF softmax 对独立 FP64 参考、行和、极端输入与重复调用的验证。

计算始终由原生 VF 完成；输入量级逐行增加，区分近均匀与近 one-hot 情况。
浮点验收固定 atol/rtol；连续 softmax 的跨轮 VST/VLD 依赖也逐项验证。
"""

import pytest
import numpy as np
import torch

from . import reference as ref
from .kernel import C, R, softmax_vf, make_vf_chain

_RTOL = 1e-5
_ATOL = 1e-7
_SUM_TOL = 1e-6


def _run(device):
    source = ref.make_operands(R, C)
    x = torch.from_numpy(source).to(device)
    out = torch.full((R, C), -99.0, dtype=torch.float32, device=device)
    softmax_vf(x, out)
    torch.npu.synchronize()
    return source, out.cpu().numpy()


def test_matches_independent_float64_reference(device):
    source, got = _run(device)
    np.testing.assert_allclose(got, ref.softmax(source), rtol=_RTOL, atol=_ATOL)


def test_each_row_sums_to_one(device):
    _, got = _run(device)
    deviation, tolerance = ref.row_sums_are_one(got, _SUM_TOL)
    assert deviation <= tolerance


def test_extreme_rows_distinguish_stability(device):
    _, got = _run(device)
    assert got[0].max() < 0.5
    assert got[-1].max() > 0.9


def test_repeated_call_overwrites_poisoned_output(device):
    source = ref.make_operands(R, C)
    x = torch.from_numpy(source).to(device)
    out = torch.empty_like(x)
    for sentinel in (-99.0, 123.0):
        out.fill_(sentinel)
        softmax_vf(x, out)
        torch.npu.synchronize()
        np.testing.assert_allclose(
            out.cpu().numpy(), ref.softmax(source), rtol=_RTOL, atol=_ATOL
        )
    np.testing.assert_array_equal(x.cpu().numpy(), source)


@pytest.mark.parametrize("repeats", [1, 64, 256])
def test_dependent_softmax_rounds(device, repeats):
    source = np.random.default_rng(0).uniform(-4, 4, (R, C)).astype(np.float32)
    x = torch.from_numpy(source).to(device)
    out = torch.empty_like(x)
    expected = source.astype(np.float64)
    for _ in range(repeats):
        expected = ref.softmax(expected)
    make_vf_chain(repeats)(x, out)
    torch.npu.synchronize()
    np.testing.assert_allclose(out.cpu().numpy(), expected, atol=_ATOL, rtol=_RTOL)
