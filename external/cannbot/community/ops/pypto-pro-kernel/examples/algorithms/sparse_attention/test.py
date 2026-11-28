# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""SFA: independent mathematics, two public workloads and guarded stage reuse.

Set CANN_BENCH_ROOT to the evaluator checkout; the original golden and checker
own precision, including their same-dtype cancellation/small-value controls.
The bundled FP64 reference gathers original coordinates without adopting native
blocking or FP16 P rounding. Dense and gathered paths keep separate physical
receive views, shared-stage event pairs, and same-core statistics rings.

The workloads are defined by representative_inputs. On Ascend950PR with 28
Cube/56 AIV, PyPTO bde9013b, PTO-ISA f7bb1b24, CANN 9.3.0 and torch/torch_npu
2.10.0, the shared timer compares 15 adjacent pairs in each AB/BA order, one
native task per call, using resident inputs and preallocated output:

* gqa keeps M128/Bk256, preload=1, buffers=2. Native exp_sub and separate
  sub/exp produce identical loaded libraries; no speedup is attributed to it.
* wide compares M64/Bk64, preload=1, buffers=2 with M64/Bk128, preload=0,
  buffers=1. Native median pair ratios are 0.939/0.940 (about 48 to 45 us).
  The smaller ring permits K128/N128 PV operands within the shared L0 budget.

Both use ROWPLUSONE; aligned and exact padded representations are numerical
controls. The raw-kernel synchronized-wall comparison is order dependent, so
the native improvement is not an end-to-end claim or an official benchmark
score. Revalidate current source and retune other shapes/toolchains.
"""

import os

import pytest
import torch

from .reference import load_evaluator, reference, representative_inputs


@pytest.fixture(scope="module")
def evaluation():
    root = os.environ.get("CANN_BENCH_ROOT")
    if not root:
        pytest.skip("CANN_BENCH_ROOT is required for the original evaluator contract")
    pytest.importorskip("kernel_eval.utils.compare")
    return load_evaluator(root)


def _inputs(dense, masked=False):
    generator = torch.Generator(device="cpu").manual_seed(313)
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
    query = (torch.rand(qshape, generator=generator, device="cpu") * 2 - 1).half()
    key = (torch.rand(kshape, generator=generator, device="cpu") * 2 - 1).half()
    indices = (
        torch.stack(
            [
                torch.randperm(sk, generator=generator, device="cpu")[:top]
                for _ in range(b * sq * hkv)
            ]
        )
        .reshape(ishape)
        .int()
    )
    if masked:
        query.zero_()
        key.fill_(0.25)
        indices.copy_(
            torch.arange(64, 128, dtype=torch.int32, device="cpu").expand(ishape)
        )
    return query, key, indices, dv, layout


@pytest.mark.parametrize(
    "dense,workers,masked", [(True, 3, False), (False, 3, False), (False, 7, True)]
)
def test_pipeline_contract(dense, workers, masked, device, evaluation):
    from .kernel import make_kernel

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
    from .kernel import make_kernel

    # Zero scores give an exact power-of-two mean. Distinct coordinates expose
    # wrong batch/head/query ownership and incomplete private AIV fills.
    b, sq, sk, hkv, heads = 2, 64 if dense else 3, 1536, 2, 4 if dense else 64
    dk, dv, selected = (192, 128, 128) if dense else (576, 512, 128)
    q = torch.zeros((b, sq, hkv * heads, dk), dtype=torch.float16, device="cpu")
    batch = torch.arange(b, device="cpu")[:, None, None, None]
    row = torch.arange(sk, device="cpu")[None, :, None, None]
    head = torch.arange(hkv, device="cpu")[None, None, :, None]
    feature = torch.arange(dk, device="cpu")[None, None, None, :]
    k = (((batch * 17 + head * 11 + row * (feature + 1)) % 31 - 15) / 16).half()
    owner = torch.arange(b * sq * hkv, device="cpu").view(b, sq, hkv, 1)
    ix = ((owner * 37 + torch.arange(selected, device="cpu") * 13) % sk).int()
    chosen = k[..., :dv][
        torch.arange(b, device="cpu")[:, None, None, None],
        ix.long(),
        torch.arange(hkv, device="cpu")[None, None, :, None],
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


@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
def test_independent_reference(layout):
    # All scores vanish; distinct heads and two causal rows expose coordinates.
    q = torch.zeros((1, 4, 4, 2), dtype=torch.float16, device="cpu")
    k = torch.arange(1 * 4 * 2 * 2, device="cpu").view(1, 4, 2, 2).half()
    ix = (
        torch.tensor([3, 0], dtype=torch.int32, device="cpu").expand(1, 4, 2, 2).clone()
    )
    expected = torch.empty((1, 4, 4, 1), dtype=torch.float64, device="cpu")
    for row in range(4):
        for head in range(4):
            keys = [0] if row < 3 else [3, 0]
            expected[0, row, head] = k[0, keys, head // 2, :1].double().mean()
    if layout == "BNSD":
        q, k, ix, expected = [
            x.transpose(1, 2).contiguous() for x in (q, k, ix, expected)
        ]
    actual = reference(q, k, k[..., :1], ix, 1.0, layout, True)
    assert torch.equal(actual, expected)
    empty_indices = ix.clone()
    empty_indices[..., 0] = 2
    empty_indices[..., 1] = 3
    masked = reference(q, k, k[..., :1], empty_indices, 1.0, layout, True)
    sequence_axis = 1 if layout == "BSND" else 2
    assert torch.equal(
        masked.narrow(sequence_axis, 0, 2),
        torch.zeros_like(masked.narrow(sequence_axis, 0, 2)),
    )
    duplicate = ix.clone().fill_(0)
    with pytest.raises(ValueError, match="distinct"):
        reference(q, k, k[..., :1], duplicate, 1.0, layout, True)


@pytest.mark.parametrize("case", ["gqa", "wide"])
def test_representative_case(case, device, evaluation):
    from .kernel import make_kernel

    golden, compare = evaluation
    q, k, v, ix, scale, layout, causal = representative_inputs(case)
    expected = reference(q, k, v, ix, scale, layout, causal)
    original = golden(q.double(), k.double(), v.double(), ix, scale, layout, causal)
    torch.testing.assert_close(expected, original, rtol=1e-12, atol=1e-12)
    native = golden(q, k, v, ix, scale, layout, causal)
    dense = layout == "BSND"
    sa, ha = (1, 2) if dense else (2, 1)
    dq, dk, dix = [x.to(device) for x in (q, k, ix)]
    baseline = (128, 256, 1, 2, "padded") if dense else (64, 64, 1, 2, "padded")
    candidate = (
        (128, 256, 1, 2, "row_plus_one") if dense else (64, 128, 0, 1, "row_plus_one")
    )
    for rows, block, preload, buffers, probability_layout in (
        baseline,
        candidate,
    ):
        kernel = make_kernel(
            q.shape[0],
            q.shape[sa],
            k.shape[sa],
            q.shape[ha],
            k.shape[ha],
            q.shape[-1],
            v.shape[-1],
            ix.shape[-1],
            layout,
            causal,
            dense,
            preload,
            rows,
            block,
            buffers,
            probability_layout,
        )
        backing = torch.full(
            (expected.numel() + 64,), 7.0, dtype=q.dtype, device=device
        )
        output = backing[32:-32].view(expected.shape)
        output.fill_(float("nan"))
        kernel(dq, dk, dix, output, scale)
        torch.npu.synchronize()
        actual = output.cpu()
        result = compare(actual, expected, dtype="float16", native_output=native)
        assert result.passed, result
        assert torch.isfinite(actual).all()
        assert bool((backing[:32] == 7).all()) and bool((backing[-32:] == 7).all())
    assert all(
        torch.equal(got.cpu(), original)
        for got, original in ((dq, q), (dk, k), (dix, ix))
    )
    assert not compare(
        actual + 1, expected, dtype="float16", native_output=native
    ).passed


@pytest.mark.parametrize("dense", [True, False])
def test_probability_layout(dense, device, evaluation):
    """ROWPLUSONE and both exact-pitch controls feed the same Cube consumer."""
    from .kernel import make_kernel

    golden, compare = evaluation
    q, k, ix, dv, layout = _inputs(dense)
    v = k[..., :dv].contiguous()
    scale = q.shape[-1] ** -0.5
    expected = reference(q, k, v, ix, scale, layout, True)
    native = golden(q, k, v, ix, scale, layout, True)
    sa, ha = (1, 2) if dense else (2, 1)
    inputs = [x.to(device) for x in (q, k, ix)]
    control = None
    for representation in ("aligned", "padded", "row_plus_one"):
        kernel = make_kernel(
            q.shape[0],
            q.shape[sa],
            k.shape[sa],
            q.shape[ha],
            k.shape[ha],
            q.shape[-1],
            dv,
            ix.shape[-1],
            layout,
            True,
            dense,
            0,
            128 if dense else 64,
            256 if dense else 64,
            1,
            representation,
        )
        output = torch.full(expected.shape, float("nan"), dtype=q.dtype, device=device)
        kernel[None, 3](*inputs, output, scale)
        torch.npu.synchronize()
        actual = output.cpu()
        checked = compare(actual, expected, dtype="float16", native_output=native)
        assert checked.passed, checked
        if control is None:
            control = actual
        else:
            assert torch.equal(actual, control)


@pytest.mark.parametrize("case", ["gqa", "wide"])
def test_public_entry(case, device, evaluation):
    """Final wrapper, same-shape causal switching, and an unfrozen runtime scale."""
    from .kernel import sparse_attention

    golden, compare = evaluation
    q, k, v, ix, scale, layout, causal = representative_inputs(case)
    inputs = [x.to(device) for x in (q, k, v, ix)]
    calls = (
        [(causal, scale), (causal, 32.0)]
        if case == "gqa"
        else [(True, scale), (False, scale), (True, scale)]
    )
    references = {}
    outputs = []
    for current_causal, current_scale in calls:
        key = current_causal, current_scale
        if key not in references:
            expected = reference(q, k, v, ix, current_scale, layout, current_causal)
            native = golden(q, k, v, ix, current_scale, layout, current_causal)
            references[key] = expected, native
        expected, native = references[key]
        actual = sparse_attention(*inputs, current_scale, layout, current_causal)
        torch.npu.synchronize()
        assert actual.dtype == q.dtype and actual.shape == expected.shape
        assert actual.device == inputs[0].device and actual.is_contiguous()
        result = actual.cpu()
        checked = compare(result, expected, dtype="float16", native_output=native)
        assert checked.passed, checked
        assert torch.isfinite(result).all()
        outputs.append(result)
    if case == "wide":
        assert not torch.equal(outputs[0], outputs[1])
        assert torch.equal(outputs[0], outputs[2])
    assert all(
        torch.equal(got.cpu(), original) for got, original in zip(inputs, (q, k, v, ix))
    )
