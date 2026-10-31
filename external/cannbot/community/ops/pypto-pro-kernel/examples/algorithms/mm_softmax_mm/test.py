# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Checks for the fixed unscaled FP16 softmax matrix chain.

The independent CPU reference preserves FP16 probability quantization before the second GEMM. The full output and
selected group/band boundaries use rtol=atol=2e-2; preload=0/2 and repeated calls use zero tolerance. All eight
execution groups are launched explicitly. These checks do not establish performance or the
batched/scaled/prefix-validity contract of mm_softmax_mm_materialized.
"""

from __future__ import annotations

import pytest
import torch

from .kernel import (
    BAND_ROWS,
    K,
    LAUNCH_BLOCK_DIM,
    M,
    M_PER_CORE,
    N,
    NUM_BANDS_PER_CORE,
    NUM_CORES,
    mm_softmax_mm_multicore,
    mm_softmax_mm_multicore_p0,
)
from .reference import mm_softmax_mm_reference


@pytest.fixture(scope="session")
def inputs(device: str):
    """session 级：随机输入只生成一次。"""
    torch.manual_seed(0)
    a = torch.randn([M, K], device=device, dtype=torch.float16)
    b = torch.randn([K, N], device=device, dtype=torch.float16)
    c = torch.randn([N, N], device=device, dtype=torch.float16)
    return a, b, c


@pytest.fixture(scope="session")
def golden(inputs):
    """session 级：CPU FP32 参考只算一次，全部正确性测试共用。"""
    a, b, c = inputs
    return mm_softmax_mm_reference(a.cpu(), b.cpu(), c.cpu())


def _run(kernel, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """显式 ``kernel[None, LAUNCH_BLOCK_DIM]`` 启动——mix kernel 的方括号数是执行组数
    （1c+2v/组），不是 AIV 线程数；不依赖 eager 默认（auto 哨兵用满 stream 预算）。
    """
    out = torch.zeros([M, N], device=a.device, dtype=torch.float32)
    kernel[None, LAUNCH_BLOCK_DIM](a, b, c, out)
    torch.npu.synchronize()
    return out


def test_matches_reference(device, inputs, golden):
    a, b, c = inputs
    out = _run(mm_softmax_mm_multicore, a, b, c)
    torch.testing.assert_close(out.cpu(), golden, rtol=2e-2, atol=2e-2)


def test_preload0_and_preload2_agree_bit_for_bit(device, inputs):
    """cube 侧两个 stage：preload 预期真的改变调度（不同于 1+1 拓扑的既有前例），
    这条测的是跨核同步本身对不对——同步不足的表现是数值错误或不稳定，调度差异
    不应该改变结果。``mm_softmax_mm_multicore``（出货形态）是 preload=2；
    ``mm_softmax_mm_multicore_p0`` 是 preload=0 对照档。
    """
    a, b, c = inputs
    p0 = _run(mm_softmax_mm_multicore_p0, a, b, c)
    p2 = _run(mm_softmax_mm_multicore, a, b, c)
    torch.testing.assert_close(p0, p2, rtol=0, atol=0)


def test_two_runs_agree_bit_for_bit(device, inputs):
    a, b, c = inputs
    first = _run(mm_softmax_mm_multicore, a, b, c)
    second = _run(mm_softmax_mm_multicore, a, b, c)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize(
    "core_idx",
    [0, NUM_CORES // 2, NUM_CORES - 1],
    ids=["core0", "core_mid", "core_last"],
)
def test_core_boundary_bands_match_reference(device, inputs, golden, core_idx):
    """定位性测试：抽样核（首/中/末）的首带与末带各自对参考切片。

    全量比较已覆盖正确性；这一条的价值是失败时直接指出"哪个核、哪条带"——
    ``core_id * M_PER_CORE`` 的偏移算式若有 off-by-one 或除法漏做，全量失败信息
    分不清是核切分错、带内 splitk 错，还是 softmax->第二个 gemm 的半区插入错。
    """
    a, b, c = inputs
    out = _run(mm_softmax_mm_multicore, a, b, c).cpu()
    core_base = core_idx * M_PER_CORE
    for band in (0, NUM_BANDS_PER_CORE - 1):
        row0 = core_base + band * BAND_ROWS
        row1 = row0 + BAND_ROWS
        torch.testing.assert_close(
            out[row0:row1], golden[row0:row1], rtol=2e-2, atol=2e-2
        )
