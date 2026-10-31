# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CANN checker controls, guarded outputs, and serial/pipelined equivalence.

Set CANN_BENCH_ROOT to the evaluator checkout; the original golden and checker
own precision, including their same-dtype cancellation/small-value controls.
"""

import os

import pytest
import torch

from .kernel import make_kernel
from .reference import load_evaluator


@pytest.fixture(scope="module")
def evaluation():
    root = os.environ.get("CANN_BENCH_ROOT")
    if not root:
        pytest.skip("CANN_BENCH_ROOT is required for the original evaluator contract")
    pytest.importorskip("kernel_eval.utils.compare")
    return load_evaluator(root)


def _inputs(dense, masked=False):
    generator = torch.Generator().manual_seed(313)
    if dense:
        b, sq, sk, hq, hkv, dk, dv, top = 2, 64, 512, 8, 2, 192, 128, 128
        layout = "BSND"
        qshape, kshape = (b, sq, hq, dk), (b, sk, hkv, dk)
        ishape = (b, sq, hkv, top)
    else:
        b, sq, sk, hq, hkv, dk, dv, top = 2, 3, 384, 64, 1, 576, 512, 192
        if masked:
            b, sq, sk, top = 1, 128, 128, 64
        layout = "BNSD"
        qshape, kshape = (b, hq, sq, dk), (b, hkv, sk, dk)
        ishape = (b, hkv, sq, top)
    query = (torch.rand(qshape, generator=generator) * 2 - 1).half()
    key = (torch.rand(kshape, generator=generator) * 2 - 1).half()
    indices = (
        torch.stack(
            [torch.randperm(sk, generator=generator)[:top] for _ in range(b * sq * hkv)]
        )
        .reshape(ishape)
        .int()
    )
    if masked:
        query.zero_()
        key.fill_(0.25)
        indices.copy_(torch.arange(64, 128, dtype=torch.int32).expand(ishape))
    return query, key, indices, dv, layout


@pytest.mark.parametrize(
    "dense,workers,masked", [(True, 3, False), (False, 3, False), (False, 7, True)]
)
def test_pipeline_contract(dense, workers, masked, device, evaluation):
    golden, compare = evaluation
    q, k, ix, dv, layout = _inputs(dense, masked)
    sa, ha = (1, 2) if dense else (2, 1)
    b, sq, hq, dk = q.shape[0], q.shape[sa], q.shape[ha], q.shape[-1]
    sk, hkv, selected = k.shape[sa], k.shape[ha], ix.shape[-1]
    shape = q.shape[:-1] + (dv,)
    scale = dk**-0.5
    dq, dk_cache, dix = [tensor.to(device) for tensor in (q, k, ix)]
    for iteration in range(2):
        current = ix if masked else (ix + iteration * 17) % sk
        dix.copy_(current)
        value = k[..., :dv].contiguous()
        expected = golden(
            q.double(), k.double(), value.double(), current, scale, layout, True
        )
        native = golden(q, k, value, current, scale, layout, True)
        outputs = []
        for preload in (0, 1):
            kernel = make_kernel(
                b,
                sq,
                sk,
                hq,
                hkv,
                dk,
                dv,
                selected,
                layout,
                True,
                dense,
                preload,
                128 if dense else 64,
                256 if dense else 64,
            )
            backing = torch.full(
                (expected.numel() + 64,), 7.0, dtype=torch.float16, device=device
            )
            output = backing[32:-32].view(shape)
            output.fill_(float("nan"))
            kernel[None, workers](dq, dk_cache, dix, output, scale)
            torch.npu.synchronize()
            actual = output.cpu()
            checked = compare(actual, expected, dtype="float16", native_output=native)
            assert checked.passed, checked
            assert torch.isfinite(actual).all()
            assert bool((backing[:32] == 7).all()) and bool((backing[-32:] == 7).all())
            assert (
                output.shape == shape
                and output.dtype == torch.float16
                and output.is_contiguous()
            )
            outputs.append(actual)
        assert torch.equal(outputs[0], outputs[1])
        if masked:
            assert torch.equal(
                outputs[1][:, :, :64], torch.zeros_like(outputs[1][:, :, :64])
            )
            assert torch.equal(
                outputs[1][:, :, 64:], torch.full_like(outputs[1][:, :, 64:], 0.25)
            )
        assert (
            torch.equal(dq.cpu(), q)
            and torch.equal(dk_cache.cpu(), k)
            and torch.equal(dix.cpu(), current)
        )
    corrupted = outputs[1] + 1
    assert not compare(
        corrupted, expected, dtype="float16", native_output=native
    ).passed


@pytest.mark.parametrize("dense", [True, False])
def test_exact_owner_and_selected_rows(dense, device):
    # Zero scores give an exact power-of-two mean. Distinct coordinates expose
    # wrong batch/head/query ownership and incomplete private AIV fills.
    b, sq, sk, hkv, heads = 2, 64 if dense else 3, 1536, 2, 4 if dense else 64
    dk, dv, selected = (192, 128, 128) if dense else (576, 512, 128)
    q = torch.zeros((b, sq, hkv * heads, dk), dtype=torch.float16)
    batch = torch.arange(b)[:, None, None, None]
    row = torch.arange(sk)[None, :, None, None]
    head = torch.arange(hkv)[None, None, :, None]
    feature = torch.arange(dk)[None, None, None, :]
    k = (((batch * 17 + head * 11 + row * (feature + 1)) % 31 - 15) / 16).half()
    owner = torch.arange(b * sq * hkv).view(b, sq, hkv, 1)
    ix = ((owner * 37 + torch.arange(selected) * 13) % sk).int()
    chosen = k[..., :dv][
        torch.arange(b)[:, None, None, None],
        ix.long(),
        torch.arange(hkv)[None, None, :, None],
    ]
    expected = chosen.double().mean(dim=3).repeat_interleave(heads, dim=2).half()
    layout = "BSND" if dense else "BNSD"
    if not dense:
        q, k, ix, expected = [
            x.transpose(1, 2).contiguous() for x in (q, k, ix, expected)
        ]
    dq, dkey, dix = [x.to(device) for x in (q, k, ix)]
    for preload in (0, 1):
        kernel = make_kernel(
            b,
            sq,
            sk,
            hkv * heads,
            hkv,
            dk,
            dv,
            selected,
            layout,
            False,
            dense,
            preload,
            128 if dense else 64,
            256 if dense else 64,
        )
        out = torch.full(
            expected.shape, float("nan"), dtype=torch.float16, device=device
        )
        kernel[None, 3 if dense else 7](dq, dkey, dix, out, dk**-0.5)
        torch.npu.synchronize()
        assert torch.equal(out.cpu(), expected)
