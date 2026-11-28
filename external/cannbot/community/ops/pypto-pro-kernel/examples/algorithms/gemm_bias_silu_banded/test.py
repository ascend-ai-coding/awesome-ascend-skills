# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""上板对照：带化流水版 ``silu(A @ B + bias)``。

**解决什么.** 三件事，分开验证：(1) 分带 + ``pl.gemm`` + bias 融合 + VF silu 拼起来数值
对不对；(2) Layer B 的验证纪律——``preload=0`` 与 ``preload=2`` 结果必须逐位一致（测的是
跨核同步对不对，不是调度差异，见 ``kernel.py`` 模块 docstring "同步设计"一节）；
(3) 局部 buffer 深度杠杆（L0A/L0B depth=1 的对照变体）数值上同样正确，这样后续的性能
对比才是在比较两个都正确的实现。

**容差依据.** ``rtol=atol=2e-2`` 与基线 ``examples/algorithms/gemm_bias_silu/test.py`` 相同
——FP16 输入、FP32 累加、K=256 的矩阵乘在本仓的既有板证取值，不是本次临时选的数，
验证失败时不放宽。

**preload 结果比较用 ``rtol=atol=0``**：比"各自都接近参考"强，两版偏同一方向的误差会被
前者放过，逐位比较不会。重复运行同一 kernel 两遍也逐位比较——跨核同步不足的典型表现是
不稳定，单跑一遍可能恰好对。
"""

from __future__ import annotations

import pytest
import torch

from .kernel import (
    LAUNCH_BLOCK_DIM,
    BAND_ROWS,
    K,
    M,
    N,
    NUM_BANDS,
    gemm_bias_silu_banded,
    gemm_bias_silu_banded_l0depth1,
    gemm_bias_silu_banded_p2,
)
from .reference import gemm_bias_silu_reference


def _inputs(device: str, seed: int = 0):
    torch.manual_seed(seed)
    a = torch.randn([M, K], device=device, dtype=torch.float16)
    b = torch.randn([K, N], device=device, dtype=torch.float16)
    bias = torch.randn([N], device=device, dtype=torch.float32)
    return a, b, bias


def _run(
    kernel, a: torch.Tensor, b: torch.Tensor, bias_1d: torch.Tensor
) -> torch.Tensor:
    """host 侧把 ``bias`` 从任务约定的 ``[256]`` reshape 成 kernel 要的 ``[1, 256]``（视图，不拷贝）。"""
    bias_2d = bias_1d.reshape(1, N)
    out = torch.zeros([M, N], device=a.device, dtype=torch.float32)
    kernel[None, LAUNCH_BLOCK_DIM](a, b, bias_2d, out)
    torch.npu.synchronize()
    return out


assert NUM_BANDS * BAND_ROWS == M, "带划分必须整除 M，否则下面的测试假设不成立"


@pytest.mark.parametrize(
    "kernel",
    [gemm_bias_silu_banded, gemm_bias_silu_banded_p2],
    ids=["preload=0", "preload=2"],
)
def test_matches_reference(device, kernel):
    a, b, bias = _inputs(device, seed=0)

    out = _run(kernel, a, b, bias)
    expected = gemm_bias_silu_reference(a, b, bias.reshape(1, N))

    torch.testing.assert_close(out.cpu(), expected, rtol=2e-2, atol=2e-2)


def test_l0_depth_variant_matches_reference(device):
    """depth 杠杆对照变体（L0A/L0B depth=1）必须同样正确，性能对比才有意义。"""
    a, b, bias = _inputs(device, seed=5)

    out = _run(gemm_bias_silu_banded_l0depth1, a, b, bias)
    expected = gemm_bias_silu_reference(a, b, bias.reshape(1, N))

    torch.testing.assert_close(out.cpu(), expected, rtol=2e-2, atol=2e-2)


def test_the_two_preloads_agree_bit_for_bit(device):
    """preload 只该改变时序，不该改变任何一个 bit——本拓扑下预期它连时序都不改变

    （kernel.py 模块 docstring："一核一 stage 时 preload 完全不起作用"），但这条测的是
    跨核同步本身对不对，与调度是否真的不同无关。
    """
    a, b, bias = _inputs(device, seed=1)

    p0 = _run(gemm_bias_silu_banded, a, b, bias)
    p2 = _run(gemm_bias_silu_banded_p2, a, b, bias)

    torch.testing.assert_close(p0, p2, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kernel",
    [gemm_bias_silu_banded, gemm_bias_silu_banded_p2],
    ids=["preload=0", "preload=2"],
)
def test_two_runs_agree_bit_for_bit(device, kernel):
    """跨核同步不足的表现是不稳定，preload 越深反向同步的压力越大。"""
    a, b, bias = _inputs(device, seed=2)

    first = _run(kernel, a, b, bias)
    second = _run(kernel, a, b, bias)

    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_every_band_was_written_by_its_own_stage(device):
    """四条带各自与参考比对应切片，失败时直接指出是哪条带（带号即 stage 调用序号）。"""
    a, b, bias = _inputs(device, seed=3)

    out = _run(gemm_bias_silu_banded, a, b, bias).cpu()
    golden = gemm_bias_silu_reference(a, b, bias.reshape(1, N)).cpu()

    wrong = [
        band
        for band in range(NUM_BANDS)
        if not torch.allclose(
            out[slice(band * BAND_ROWS, (band + 1) * BAND_ROWS)],
            golden[slice(band * BAND_ROWS, (band + 1) * BAND_ROWS)],
            rtol=2e-2,
            atol=2e-2,
        )
    ]
    assert not wrong, f"这些带算错了：{wrong}"
