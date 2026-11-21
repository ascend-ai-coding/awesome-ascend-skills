# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Grouped channels, zero extension, FP32 products and changed inputs at reused addresses."""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import reference


CASES = [
    (1, 1, 16, 32, 8, 2, 2),
    (2, 3, 16, 16, 4, 1, 2),
    (3, 31, 16, 48, 8, 2, 1),
    (2, 64, 48, 32, 2, 1, 1),
    (1, 65, 128, 16, 1, 2, 2),
    (3, 129, 64, 128, 4, 2, 1),
    (1, 5, 80, 112, 4, 2, 2),
    (2, 4097, 64, 96, 2, 1, 1),
    (3, 4097, 128, 128, 1, 1, 2),
]


def assert_output(actual, expected):
    assert actual.dtype == np.float32 and actual.shape == expected.shape
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=3e-4)


def arrays(case):
    batch, length, cin, cout, groups, _, _ = case
    rng = np.random.default_rng(971 + length + cin * cout)
    x = rng.uniform(-1, 1, (batch, length, cin)).astype(np.float16)
    w = rng.uniform(-1, 1, (groups, 3, cin // groups, cout // groups)).astype(
        np.float16
    )
    bias = rng.uniform(-2, 2, cout).astype(np.float32)
    return x, w, bias


@pytest.mark.parametrize("case", CASES)
def test_contract_and_changed_storage(device, case):
    x, w, bias = arrays(case)
    stride, dilation = case[-2:]
    tensors = [torch.from_numpy(a).to(device) for a in (x, w, bias)]
    changed = (-x[:, ::-1, :].copy(), w[:, ::-1, :, :].copy(), -bias)
    for host in ((x, w, bias), changed):
        for tensor, value in zip(tensors, host):
            tensor.copy_(torch.from_numpy(np.ascontiguousarray(value)))
        result = run(*tensors, stride=stride, dilation=dilation)
        torch.npu.synchronize()
        assert result.is_contiguous() and result.device == tensors[0].device
        assert_output(result.cpu().numpy(), reference(*host, stride, dilation))
        for tensor, value in zip(tensors, host):
            np.testing.assert_array_equal(tensor.cpu().numpy(), value)


def test_group_isolation_and_bias(device):
    x = np.zeros((2, 7, 32), dtype=np.float16)
    x[:, :, 8:16] = np.float16(0.25)
    w = np.ones((4, 3, 8, 12), dtype=np.float16)
    bias = np.linspace(-0.5, 0.5, 48, dtype=np.float32)
    result = run(
        *(torch.from_numpy(a).to(device) for a in (x, w, bias)), stride=2, dilation=2
    )
    torch.npu.synchronize()
    actual = result.cpu().numpy()
    assert_output(actual, reference(x, w, bias, 2, 2))
    untouched = np.r_[0:12, 24:48]
    np.testing.assert_array_equal(
        actual[:, :, untouched],
        np.broadcast_to(bias[untouched], (2, 4, len(untouched))),
    )


def test_comparison_rejects_corruption():
    expected = reference(
        np.ones((1, 1, 16), dtype=np.float16),
        np.ones((8, 3, 2, 2), dtype=np.float16),
        np.zeros(16, dtype=np.float32),
        1,
        2,
    )
    assert_output(expected.astype(np.float32), expected)
    with pytest.raises(AssertionError):
        assert_output(expected.astype(np.float32) + 1, expected)
