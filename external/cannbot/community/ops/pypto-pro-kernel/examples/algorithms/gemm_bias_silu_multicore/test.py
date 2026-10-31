# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""上板对照：mix kernel 多核切分（8 核 × 4 带流水）的 ``silu(A@B+bias)``。

**解决什么.** (1) 真实 8 核切分（``LAUNCH_BLOCK_DIM=8``，即 8 个执行组）、``M=2048`` 下数值对；
(2) Layer B 验证纪律——``preload=0``/``preload=2`` 逐位一致（测跨核同步本身，
1+1 拓扑下 preload 是死参数）；(3) 重复运行逐位一致（同步不足的典型表现是不稳定）。

**物理访问.** 见 ``kernel.py``：8 核各按 ``core_id = get_block_idx() //
get_subblock_num()`` 分到不重叠的 256 行，核内 4 带流水，带内两个 AIV 各 32 行
（``DualModeSplitM`` 直传）。

**容差依据.** ``rtol=atol=2e-2``：与基线/banded 相同——FP16 输入、FP32 累加、
K=256 的既有 board 证取值，不因规模变大而放宽。逐位比较用 ``rtol=atol=0``。

**如何独立验证.** ``reference.py``（CPU FP32）；核边界定位测试单独覆盖「分核偏移
算对了没有」——失败时直接指认哪个核哪条带，不必在 2048 行里反推。
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
    gemm_bias_silu_multicore,
    gemm_bias_silu_multicore_p2,
)
from .reference import gemm_bias_silu_reference


@pytest.fixture(scope="session")
def inputs(device: str):
    """session 级：随机输入只生成一次（M=2048，重复生成没必要）。"""
    torch.manual_seed(0)
    a = torch.randn([M, K], device=device, dtype=torch.float16)
    b = torch.randn([K, N], device=device, dtype=torch.float16)
    bias = torch.randn([N], device=device, dtype=torch.float32)
    return a, b, bias


@pytest.fixture(scope="session")
def golden(inputs):
    """session 级：CPU FP32 参考只算一次，全部正确性测试共用。"""
    a, b, bias = inputs
    return gemm_bias_silu_reference(a, b, bias.reshape(1, N)).cpu()


def _run(
    kernel, a: torch.Tensor, b: torch.Tensor, bias_1d: torch.Tensor
) -> torch.Tensor:
    """显式 ``kernel[None, LAUNCH_BLOCK_DIM]`` 启动——mix kernel 的方括号数是
    执行组数（1c+2v/组），不是 AIV 线程数（launch 语义 vs 索引语义，见 kernel.py
    docstring）；不依赖 eager 默认（auto 哨兵用满 stream 预算）。
    """
    bias_2d = bias_1d.reshape(1, N)
    out = torch.zeros([M, N], device=a.device, dtype=torch.float32)
    kernel[None, LAUNCH_BLOCK_DIM](a, b, bias_2d, out)
    torch.npu.synchronize()
    return out


def test_multicore_matches_reference(device, inputs, golden):
    a, b, bias = inputs
    out = _run(gemm_bias_silu_multicore, a, b, bias)
    torch.testing.assert_close(out.cpu(), golden, rtol=2e-2, atol=2e-2)


def test_preload0_and_preload2_agree_bit_for_bit(device, inputs):
    """1+1 拓扑下 preload 是死参数（crosscore-sync.md），这条测的是跨核同步本身
    对不对——同步不足的表现是数值错误或不稳定，与 preload 数值无关。
    """
    a, b, bias = inputs
    p0 = _run(gemm_bias_silu_multicore, a, b, bias)
    p2 = _run(gemm_bias_silu_multicore_p2, a, b, bias)
    torch.testing.assert_close(p0, p2, rtol=0, atol=0)


def test_two_runs_agree_bit_for_bit(device, inputs):
    a, b, bias = inputs
    first = _run(gemm_bias_silu_multicore, a, b, bias)
    second = _run(gemm_bias_silu_multicore, a, b, bias)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize(
    "core_idx",
    [0, NUM_CORES // 2, NUM_CORES - 1],
    ids=["core0", "core_mid", "core_last"],
)
def test_core_boundary_bands_match_reference(device, inputs, golden, core_idx):
    """定位性测试：抽样核（首/中/末）的首带与末带各自对参考切片。

    全量比较已覆盖正确性；这一条的价值是失败时直接指出「哪个核、哪条带」——
    ``core_id * M_PER_CORE`` 的偏移算式若有 off-by-one 或除法漏做
    （api/support.md §十五 的坑），全量失败信息分不清是核切分错还是带内逻辑错。
    """
    a, b, bias = inputs
    out = _run(gemm_bias_silu_multicore, a, b, bias).cpu()
    core_base = core_idx * M_PER_CORE
    for band in (0, NUM_BANDS_PER_CORE - 1):
        row0 = core_base + band * BAND_ROWS
        row1 = row0 + BAND_ROWS
        torch.testing.assert_close(
            out[row0:row1], golden[row0:row1], rtol=2e-2, atol=2e-2
        )
