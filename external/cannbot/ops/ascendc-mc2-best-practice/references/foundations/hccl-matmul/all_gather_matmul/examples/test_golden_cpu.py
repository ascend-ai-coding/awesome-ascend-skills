# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU self-test for golden.x_all_gather_matmul_golden — no NPU needed.
Run: python test_golden_cpu.py
"""
import logging

import torch

from golden import x_all_gather_matmul_golden

logging.basicConfig(level=logging.INFO, format='%(message)s')


def test_basic_gather_matmul_bias():
    # shard0 all 1.0 [2,4], shard1 all 2.0 [2,4] -> gathered [4,4]
    s0 = torch.full((2, 4), 1.0)
    s1 = torch.full((2, 4), 2.0)
    x2 = torch.full((4, 3), 0.5)        # [K=4, N=3]
    bias = torch.full((3,), 1.0)        # [N=3]
    out, gout = x_all_gather_matmul_golden([s0, s1], x2, bias=bias, is_gather_out=True)
    expected_gout = torch.cat([s0, s1], dim=0)
    assert gout.shape == (4, 4)
    assert torch.equal(gout, expected_gout)
    expected_out = torch.tensor(
        [[3., 3., 3.], [3., 3., 3.], [5., 5., 5.], [5., 5., 5.]])
    assert out.shape == (4, 3)
    assert torch.allclose(out, expected_out)


def test_no_bias_no_gather_out():
    s0 = torch.full((1, 4), 1.0)
    x2 = torch.full((4, 2), 1.0)
    out, gout = x_all_gather_matmul_golden([s0], x2, bias=None, is_gather_out=False)
    assert out.shape == (1, 2)
    assert gout is None
    assert torch.allclose(out, torch.full((1, 2), 4.0))


def test_trans_b():
    s0 = torch.full((2, 4), 1.0)
    s1 = torch.full((2, 4), 2.0)
    x2 = torch.full((3, 4), 0.5)
    out, gout = x_all_gather_matmul_golden(
        [s0, s1], x2, bias=None, is_trans_b=True, is_gather_out=True)
    assert out.shape == (4, 3)
    assert gout.shape == (4, 4)
    expected_out = torch.tensor(
        [[2., 2., 2.], [2., 2., 2.], [4., 4., 4.], [4., 4., 4.]])
    assert torch.allclose(out, expected_out)


if __name__ == "__main__":
    test_basic_gather_matmul_bias()
    test_no_bias_no_gather_out()
    test_trans_b()
    logging.info("golden self-test PASSED")
