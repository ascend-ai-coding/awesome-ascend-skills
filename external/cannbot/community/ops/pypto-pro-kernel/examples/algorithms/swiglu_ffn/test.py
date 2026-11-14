# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Checks for the fixed FP16 SwiGLU chain and its scaled comparator.

Numerical output must be finite and satisfy abs(out-golden) <= 2e-2*(abs(Z16) @ abs(W2)) + 2e-2. The reference narrows Z
to FP16 before the final GEMM. Preload=0/2 and repeated calls use zero tolerance; selected group/band slices use the
same scaled bound as the complete output.

Comparator checks reject NaN and both infinities. The CPU precision/reduction-order diagnostic demonstrates sensitivity
for its fixed seed; it is separate from device correctness and cannot attribute a device error to arithmetic order or
rule out a stable mapping defect. Device selection comes from examples/conftest.py.
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
    swiglu_ffn_multicore,
    swiglu_ffn_multicore_p0,
)
from .reference import cancellation_scale, swiglu_ffn_reference


@pytest.fixture(scope="session")
def inputs(device: str):
    """session 级：随机输入只生成一次。"""
    torch.manual_seed(0)
    x = torch.randn([M, K], device=device, dtype=torch.float16)
    w1 = torch.randn([K, N], device=device, dtype=torch.float16)
    w3 = torch.randn([K, N], device=device, dtype=torch.float16)
    w2 = torch.randn([K, N], device=device, dtype=torch.float16)
    return x, w1, w3, w2


@pytest.fixture(scope="session")
def golden_and_scale(inputs):
    """session 级：CPU FP32 参考与判据尺度只算一次，全部正确性测试共用。"""
    x, w1, w3, w2 = inputs
    args = (x.cpu(), w1.cpu(), w3.cpu(), w2.cpu())
    return swiglu_ffn_reference(*args), cancellation_scale(*args)


def _run(kernel, x, w1, w3, w2) -> torch.Tensor:
    """显式 ``kernel[None, LAUNCH_BLOCK_DIM]`` 启动——mix kernel 的方括号数是执行组数
    （1c+2v/组），不是 AIV 线程数；不依赖 eager 默认（auto 哨兵用满 stream 预算）。
    """
    out = torch.zeros([M, N], device=x.device, dtype=torch.float32)
    kernel[None, LAUNCH_BLOCK_DIM](x, w1, w3, w2, out)
    torch.npu.synchronize()
    return out


def _assert_close_scaled(
    out: torch.Tensor, golden: torch.Tensor, scale: torch.Tensor
) -> None:
    """Require finite output and the existing term-sum error bound."""
    if not torch.isfinite(out).all():
        raise AssertionError("non-finite output in scaled comparison")
    err = (out - golden).abs()
    bound = 2e-2 * scale + 2e-2
    bad = err > bound
    if bad.any():
        i = (err - bound).argmax()
        r, c = divmod(int(i), out.shape[1])
        raise AssertionError(
            f"{int(bad.sum())} elems exceed scaled bound; worst at [{r},{c}]: "
            f"err={err[r, c]:.4g} bound={bound[r, c]:.4g} golden={golden[r, c]:.4g}"
        )


def test_matches_reference(device, inputs, golden_and_scale):
    x, w1, w3, w2 = inputs
    golden, scale = golden_and_scale
    out = _run(swiglu_ffn_multicore, x, w1, w3, w2)
    _assert_close_scaled(out.cpu(), golden, scale)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_scaled_comparator_rejects_nonfinite_output(value):
    out = torch.zeros((2, 2), dtype=torch.float32)
    out[1, 0] = value
    with pytest.raises(AssertionError, match="non-finite output"):
        _assert_close_scaled(out, torch.zeros_like(out), torch.ones_like(out))


def test_preload0_and_preload2_agree_bit_for_bit(device, inputs):
    """cube 侧两个 stage：preload 预期真的改变调度，这条测的是跨核同步本身对不对——
    同步不足的表现是数值错误或不稳定，调度差异不应该改变结果。``swiglu_ffn_multicore``
    （出货形态）是 preload=2；``swiglu_ffn_multicore_p0`` 是 preload=0 对照档。
    """
    x, w1, w3, w2 = inputs
    p0 = _run(swiglu_ffn_multicore_p0, x, w1, w3, w2)
    p2 = _run(swiglu_ffn_multicore, x, w1, w3, w2)
    torch.testing.assert_close(p0, p2, rtol=0, atol=0)


def test_two_runs_agree_bit_for_bit(device, inputs):
    x, w1, w3, w2 = inputs
    first = _run(swiglu_ffn_multicore, x, w1, w3, w2)
    second = _run(swiglu_ffn_multicore, x, w1, w3, w2)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize(
    "core_idx",
    [0, NUM_CORES // 2, NUM_CORES - 1],
    ids=["core0", "core_mid", "core_last"],
)
def test_core_boundary_bands_match_reference(
    device, inputs, golden_and_scale, core_idx
):
    """定位性测试：抽样核（首/中/末）的首带与末带各自对参考切片。

    全量比较已覆盖正确性；这一条的价值是失败时直接指出"哪个核、哪条带"——
    ``core_id * M_PER_CORE`` 的偏移算式若有 off-by-one 或除法漏做，全量失败信息
    分不清是核切分错、带内 splitk 错，还是 fused_band->第三个 gemm 的半区插入错。
    判据与全量对照同一套（scale 版，见模块 docstring 容差依据）。
    """
    x, w1, w3, w2 = inputs
    golden, scale = golden_and_scale
    out = _run(swiglu_ffn_multicore, x, w1, w3, w2).cpu()
    core_base = core_idx * M_PER_CORE
    for band in (0, NUM_BANDS_PER_CORE - 1):
        row0 = core_base + band * BAND_ROWS
        row1 = row0 + BAND_ROWS
        _assert_close_scaled(out[row0:row1], golden[row0:row1], scale[row0:row1])


def test_cpu_precision_and_reduction_order_change_the_reference():
    """Compare CPU arithmetic choices on a fixed seed, including FP16 intermediate rounding.

    The two diagnostic gaps must exceed 2e-2 for this dataset. This assertion
    neither changes device acceptance nor identifies the cause of a mismatch.
    """
    torch.manual_seed(0)
    m, k, n = M, K, N

    def _chained(xf, w1f, w3f, w2f, chunk):
        if chunk is None:
            y1 = torch.matmul(xf, w1f)
            y3 = torch.matmul(xf, w3f)
        else:
            y1 = torch.zeros(m, n, dtype=xf.dtype)
            y3 = torch.zeros(m, n, dtype=xf.dtype)
            for k0 in range(0, k, chunk):
                y1 += torch.matmul(
                    xf[:, slice(k0, k0 + chunk)], w1f[slice(k0, k0 + chunk), :]
                )
                y3 += torch.matmul(
                    xf[:, slice(k0, k0 + chunk)], w3f[slice(k0, k0 + chunk), :]
                )
        silu = y1 / (1.0 + torch.exp(-y1))
        z16 = (silu * y3).to(torch.float16).to(xf.dtype)
        if chunk is None:
            return torch.matmul(z16, w2f)
        acc = torch.zeros(m, n, dtype=xf.dtype)
        for k0 in range(0, k, chunk):
            acc += torch.matmul(
                z16[:, slice(k0, k0 + chunk)], w2f[slice(k0, k0 + chunk), :]
            )
        return acc

    x = torch.randn([m, k], dtype=torch.float16)
    w1 = torch.randn([k, n], dtype=torch.float16)
    w3 = torch.randn([k, n], dtype=torch.float16)
    w2 = torch.randn([k, n], dtype=torch.float16)

    o_f32 = _chained(x.float(), w1.float(), w3.float(), w2.float(), chunk=None)
    o_f64 = _chained(x.double(), w1.double(), w3.double(), w2.double(), chunk=None)
    o_f32_sk64 = _chained(x.float(), w1.float(), w3.float(), w2.float(), chunk=64)

    precision_gap = (o_f32.double() - o_f64).abs().max().item()
    order_gap = (o_f32.double() - o_f32_sk64.double()).abs().max().item()

    assert precision_gap > 2e-2, (
        f"seed-0 CPU precision diagnostic: max gap {precision_gap:.4f} does not exceed 2e-2"
    )
    assert order_gap > 2e-2, (
        f"seed-0 CPU reduction-order diagnostic: max gap {order_gap:.4f} does not exceed 2e-2"
    )
