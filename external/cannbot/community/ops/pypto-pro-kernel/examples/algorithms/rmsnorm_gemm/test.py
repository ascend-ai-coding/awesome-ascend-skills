# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Board checks for the aligned eight-group RMSNorm → GEMM pipeline.

The independent CPU reference computes RMSNorm in float64, narrows once to FP16, then multiplies in FP32. The full
output uses rtol=atol=2e-2; selected group/band boundaries retain rtol=2e-2, atol=3e-2. preload=0/2 and repeated calls
use zero tolerance. These checks preserve the launch geometry and exercise cross-core synchronization; they do not
measure device time.
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
    rmsnorm_gemm_multicore,
    rmsnorm_gemm_multicore_p2,
)
from .reference import rmsnorm_gemm_reference


@pytest.fixture(scope="session")
def inputs(device: str):
    """session 级：随机输入只生成一次。"""
    torch.manual_seed(0)
    x = torch.randn([M, K], device=device, dtype=torch.float16)
    gamma = torch.randn([K], device=device, dtype=torch.float16)
    w = torch.randn([K, N], device=device, dtype=torch.float16)
    return x, gamma, w


@pytest.fixture(scope="session")
def golden(inputs):
    """session 级：CPU 参考只算一次，全部正确性测试共用。"""
    x, gamma, w = inputs
    return rmsnorm_gemm_reference(x.cpu(), gamma.cpu(), w.cpu())


def _run(kernel, x: torch.Tensor, gamma: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """显式 ``kernel[None, LAUNCH_BLOCK_DIM]`` 启动——mix kernel 的方括号数是
    执行组数（1c+2v/组），不是 AIV 线程数（launch 语义 vs 索引语义，见 kernel.py
    docstring）；不依赖 eager 默认（auto 哨兵用满 stream 预算）。

    ``gamma`` 按 ``bias_fusion``/``gemm_bias_silu*`` 的既有惯例 reshape 成
    ``[1, K]``——kernel 参数声明的是这个形状，见 kernel.py docstring。
    """
    gamma_2d = gamma.reshape(1, K)
    out = torch.zeros([M, N], device=x.device, dtype=torch.float32)
    kernel[None, LAUNCH_BLOCK_DIM](x, gamma_2d, w, out)
    torch.npu.synchronize()
    return out


def test_matches_reference(device, inputs, golden):
    x, gamma, w = inputs
    out = _run(rmsnorm_gemm_multicore, x, gamma, w)
    torch.testing.assert_close(out.cpu(), golden, rtol=2e-2, atol=2e-2)


def test_preload0_and_preload2_agree_bit_for_bit(device, inputs):
    """1+1 拓扑下 preload 是死参数（crosscore-sync.md 槽数公式），这条测的是跨核
    同步本身对不对——同步不足的表现是数值错误或不稳定，与 preload 数值无关。
    """
    x, gamma, w = inputs
    p0 = _run(rmsnorm_gemm_multicore, x, gamma, w)
    p2 = _run(rmsnorm_gemm_multicore_p2, x, gamma, w)
    torch.testing.assert_close(p0, p2, rtol=0, atol=0)


def test_two_runs_agree_bit_for_bit(device, inputs):
    x, gamma, w = inputs
    first = _run(rmsnorm_gemm_multicore, x, gamma, w)
    second = _run(rmsnorm_gemm_multicore, x, gamma, w)
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
    （api/support.md §十五 的坑），全量失败信息分不清是核切分错、带内
    RMSNorm 错，还是 vector->cube 的半区插入错。
    """
    x, gamma, w = inputs
    out = _run(rmsnorm_gemm_multicore, x, gamma, w).cpu()
    core_base = core_idx * M_PER_CORE
    for band in (0, NUM_BANDS_PER_CORE - 1):
        row0 = core_base + band * BAND_ROWS
        row1 = row0 + BAND_ROWS
        torch.testing.assert_close(
            out[row0:row1], golden[row0:row1], rtol=2e-2, atol=3e-2
        )
