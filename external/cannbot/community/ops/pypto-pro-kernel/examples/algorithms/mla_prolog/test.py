# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：混合。Board checks for the MLA prolog kernel.

The check mirrors the CANN-Bench `relative_error` checker on BF16 outputs:
overall MERE/MARE (< 2^-7 / 10*2^-7), then the small-value, cancellation and
normal-region fallbacks, each compared against the same-precision FP32 chain
(the evaluator's "native" reference). Cases cover all supported hidden sizes,
projection dispatch boundaries, cancellation, seeds and epsilon options.
"""

import pytest
import torch

from .kernel import HCKV, HCQ, NH, DR, D, NKR
from .reference import mla_prolog_reference, native_reference

THRESHOLD = 2**-7
MARE_THRESHOLD = 10 * THRESHOLD


def _rand_bf16(shape, generator, device, low=-1.0, high=1.0):
    return (
        (torch.rand(shape, generator=generator) * (high - low) + low)
        .to(torch.bfloat16)
        .to(device)
    )


def _make_inputs(he, b, s, eps_cq, eps_ckv, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    token_x = _rand_bf16((b, s, he), g, device)
    w_dq = _rand_bf16((he, HCQ), g, device)
    w_uq_qr = _rand_bf16((HCQ, NH * (D + DR)), g, device)
    w_uk = _rand_bf16((NH, D, HCKV), g, device)
    w_dkv_kr = _rand_bf16((he, NKR), g, device)
    gamma_cq = _rand_bf16((HCQ,), g, device)
    gamma_ckv = _rand_bf16((HCKV,), g, device)
    theta = (torch.rand((b, s, DR // 2), generator=g) * 2 - 1) * torch.pi
    pair = torch.cat([theta, theta], dim=-1)
    rope_sin = torch.sin(pair).to(torch.bfloat16).to(device)
    rope_cos = torch.cos(pair).to(torch.bfloat16).to(device)
    return (
        token_x,
        w_dq,
        w_uq_qr,
        w_uk,
        w_dkv_kr,
        gamma_cq,
        gamma_ckv,
        rope_sin,
        rope_cos,
        NH,
        eps_cq,
        eps_ckv,
    )


def _evaluator_check(name, actual, golden, native):
    """Reproduce the CANN-Bench relative_error verdict for one output."""
    target = actual.dtype
    got = actual.double()
    want = golden.to(target).double()
    native_t = native.to(target).double()
    diff = (got - want).abs()
    rel = diff / (want.abs() + 1e-7)
    if rel.mean() < THRESHOLD and rel.max() < MARE_THRESHOLD:
        return
    small = want.abs() < 2**-8
    npu_small = int((small & (diff > 2**-16)).sum())
    cpu_small = int((small & ((native_t - want).abs() > 2**-16)).sum())
    assert npu_small / max(cpu_small, 1) <= 2, (
        name,
        "small-value",
        npu_small,
        cpu_small,
    )
    cancel = (got.abs() < 2**-3) & (want.abs() >= 2**-8) & (want.abs() < 2**-3)
    npu_cancel = int((cancel & (rel > MARE_THRESHOLD)).sum())
    cpu_cancel = int(
        (
            cancel & (((native_t - want).abs() / (want.abs() + 1e-7)) > MARE_THRESHOLD)
        ).sum()
    )
    assert npu_cancel / max(cpu_cancel, 1) <= 2, (
        name,
        "cancel",
        npu_cancel,
        cpu_cancel,
    )
    normal = ~small & ~cancel
    npu_normal = int((normal & (rel > MARE_THRESHOLD)).sum())
    cpu_normal = int(
        (
            normal & (((native_t - want).abs() / (want.abs() + 1e-7)) > MARE_THRESHOLD)
        ).sum()
    )
    assert npu_normal == 0 or (cpu_normal > 0 and npu_normal / cpu_normal <= 2), (
        name,
        "normal",
        npu_normal,
        cpu_normal,
    )


def test_checker_rejects_normal_error_with_exact_native():
    golden = torch.tensor([1.0], dtype=torch.float64)
    native = golden.float()
    actual = torch.tensor([1.25], dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="normal"):
        _evaluator_check("query", actual, golden, native)


@pytest.mark.parametrize(
    "he,b,s,eps,seed",
    [
        (7168, 1, 1, 1e-5, 0),
        (7168, 4, 4, 1e-5, 0),
        (6144, 32, 1, 1e-5, 0),
        (7168, 2, 8, 1e-6, 0),
        (6144, 1, 1, 1e-8, 0),
        (7680, 1, 1, 1e-5, 7),
        (6144, 1, 4, 1e-6, 11),
        (7168, 1, 3, 1e-8, 13),
        (7680, 1, 33, 1e-5, 17),
        (6144, 1, 63, 1e-6, 19),
        (7168, 1, 65, 1e-5, 23),
    ],
)
def test_mla_prolog(device, he, b, s, eps, seed):
    from .kernel import mla_prolog

    inputs = _make_inputs(he, b, s, eps, eps, device, seed=seed)
    cpu_inputs = [t.cpu() if isinstance(t, torch.Tensor) else t for t in inputs]
    golden = mla_prolog_reference(*cpu_inputs)
    native = native_reference(*cpu_inputs)
    names = ("query", "query_rope", "c_kv", "k_rope")
    previous = None
    for _ in range(3):
        outputs = mla_prolog(*inputs)
        torch.npu.synchronize()
        observed = [tensor.cpu() for tensor in outputs]
        for index, (name, got, want, nat) in enumerate(
            zip(names, observed, golden, native)
        ):
            assert got.shape == want.shape, (name, got.shape, want.shape)
            assert got.dtype == torch.bfloat16, (name, got.dtype)
            assert torch.isfinite(got).all(), f"{name}: non-finite output"
            _evaluator_check(name, got, want, nat)
            if previous is not None:
                assert torch.equal(got, previous[index]), (
                    f"{name}: repeated output changed"
                )
        previous = observed
    for original, tensor in zip(cpu_inputs, inputs):
        if isinstance(tensor, torch.Tensor):
            assert torch.equal(original, tensor.cpu()), "input mutated"
