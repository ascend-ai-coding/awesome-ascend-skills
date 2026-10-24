# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Contract boundaries beyond the upstream 15-case evaluator.

Run with pytest on Ascend 950. Set CANNBOT_DSL_ROOT to the pinned upstream
checkout to include its original 15-case evaluator and unchanged tolerances.
That adapter additionally requires CANNBot DSL 0.7.0; the native kernel does not.
"""

import itertools
import os
from pathlib import Path
import tempfile

import pytest
import torch

from .kernel import quant_lightning_indexer
from .reference import check_topk, scores, visible_keys
from ._kernels.streaming_selection import make_streaming_select


@pytest.fixture(scope="module")
def device():
    pytest.importorskip("torch_npu")
    result = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', '0'))}"
    torch.npu.set_device(result)
    return result


def make_case(
    device,
    layout="PA_BBND",
    page=128,
    qlens=(3, 4),
    klens=(513, 777),
    topk=512,
    mask=3,
    ratio=2,
    capacity=-1,
    padded=False,
    zero=False,
    usedq=None,
    usedk=None,
    negative=False,
    seed=17,
):
    gen = torch.Generator().manual_seed(seed)
    rows = sum(qlens)
    q = torch.randint(0, 256, (rows, 32, 64), dtype=torch.uint8, generator=gen)
    qs = torch.randint(124, 129, (rows, 32, 2, 2), dtype=torch.uint8, generator=gen)
    w = torch.rand((rows, 32), generator=gen) * 2 - 1
    if zero:
        w.zero_()
    if negative:
        w = -w.abs()
    keys = [
        torch.randint(0, 256, (n, 1, 64), dtype=torch.uint8, generator=gen)
        for n in klens
    ]
    scales = [
        torch.randint(124, 129, (n, 1, 2, 2), dtype=torch.uint8, generator=gen)
        for n in klens
    ]
    uq = qlens if usedq is None else usedq
    uk = klens if usedk is None else usedk
    residual = (
        [ratio - 1] * len(qlens) if mask == 3 and ratio != 1 else [0] * len(qlens)
    )
    refs, allowed = [], []
    qstarts = [0, *itertools.accumulate(qlens)]
    for b, n in enumerate(qlens):
        start = qstarts[b]
        ref = scores(
            q[slice(start, start + n)],
            keys[b],
            w[slice(start, start + n)],
            qs[slice(start, start + n)],
            scales[b],
        )
        for row in range(n):
            refs.append(ref[row])
            visible = visible_keys(uk[b], uq[b], row, mask, ratio, residual[b])
            allowed.append(torch.arange(klens[b]) < visible)
    table = None
    if layout == "PA_BBND":
        pages = (max(klens) + page - 1) // page
        count = pages * len(klens)
        table = torch.randperm(count, generator=gen).int().view(len(klens), pages)
        kb = torch.full((count, page + int(padded), 1, 64), 0x5A, dtype=torch.uint8)
        sb = torch.full(
            (count, page + 3 * int(padded), 1, 2, 2), 0x7F, dtype=torch.uint8
        )
        for b, n in enumerate(klens):
            for i in range(n):
                physical = int(table[b, i // page])
                kb[physical, i % page] = keys[b][i]
                sb[physical, i % page] = scales[b][i]
        k = kb.to(device)[:, :page]
        ks = sb.to(device)[:, :page]
    else:
        flatk, flats = torch.cat(keys), torch.cat(scales)
        count = flatk.shape[0]
        kb = torch.full((count * (81 if padded else 64),), 0x5A, dtype=torch.uint8)
        sb = torch.full((count * (7 if padded else 4),), 0x7F, dtype=torch.uint8)
        kcpu = kb.as_strided((count, 1, 64), (81 if padded else 64, 64, 1))
        scpu = sb.as_strided((count, 1, 2, 2), (7 if padded else 4, 4, 2, 1))
        kcpu.copy_(flatk)
        scpu.copy_(flats)
        k = kb.to(device).as_strided(kcpu.shape, kcpu.stride())
        ks = sb.to(device).as_strided(scpu.shape, scpu.stride())

    def meta(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    kwargs = dict(
        cu_seqlens_q=meta(qstarts),
        seqused_q=meta(uq),
        seqused_k=meta(uk),
        topk=topk,
        quant_mode=1,
        max_seqlen_q=max(qlens),
        layout_k=layout,
        mask_mode=mask,
        cmp_ratio=ratio,
        candidate_topk_blocks=capacity,
        candidate_block_size=8 if capacity > 0 else -1,
        output_idx_offset=meta([i * 10000 for i in range(rows)]).view(-1, 1),
    )
    if table is not None:
        kwargs["block_table"] = table.to(device)
    else:
        kwargs["cu_seqlens_k"] = meta([0, *itertools.accumulate(klens)])
    if mask == 3 and ratio != 1:
        kwargs["cmp_residual_k"] = meta(residual)
    return [q.to(device), k, w.to(device), qs.to(device), ks], kwargs, refs, allowed


def verify(case, return_value):
    args, kwargs, refs, allowed = case
    outputs = quant_lightning_indexer(*args, **kwargs, return_value=return_value)
    torch.npu.synchronize()
    indices, values, candidates, lengths = [x.cpu() for x in outputs]
    rows, topk = len(refs), kwargs["topk"]
    assert indices.shape == (rows, 1, topk) and indices.dtype == torch.int32
    assert values.dtype == torch.bfloat16
    assert values.shape == indices.shape if return_value else values.numel() == 0
    assert all(x.is_contiguous() for x in outputs)
    for r, (ref, valid) in enumerate(zip(refs, allowed)):
        check_topk(
            indices[r, 0], values[r, 0] if return_value else None, ref, valid, r * 10000
        )
    capacity = kwargs["candidate_topk_blocks"]
    if capacity < 0:
        assert candidates.numel() == lengths.numel() == 0
        return
    assert candidates.shape == (rows, 1, capacity)
    assert lengths.shape == (rows, 1) and lengths.dtype == torch.int32
    for r, (ref, valid) in enumerate(zip(refs, allowed)):
        padded = torch.full((((ref.numel() + 7) // 8) * 8,), -float("inf"))
        padded[: ref.numel()] = ref.masked_fill(~valid, -float("inf"))
        block_scores = padded.view(-1, 8).amax(-1)
        valid_blocks = torch.isfinite(block_scores)
        visible = int(valid.sum())
        if visible % 8:
            block_scores[visible // 8] = float("inf")
        count = min(capacity, int(valid_blocks.sum()))
        assert lengths[r, 0] == count
        check_topk(candidates[r, 0, :count], None, block_scores, valid_blocks)


@pytest.mark.parametrize(
    "options",
    [
        dict(page=16, qlens=(1, 2), klens=(15, 17), topk=1, mask=0, ratio=1),
        dict(page=32, qlens=(2, 1), klens=(31, 65), topk=33, ratio=128),
        dict(page=48, qlens=(5, 7), klens=(47, 97), topk=65, ratio=1),
        dict(page=64, qlens=(1, 2), klens=(19, 73), topk=2048, capacity=16),
        dict(qlens=(7,), klens=(257,), topk=257, mask=0, ratio=1, zero=True),
        dict(qlens=(6,), klens=(8192,), usedq=(4,), ratio=128),
        dict(qlens=(7, 13), klens=(8192, 8448), padded=True),
        dict(layout="TND", qlens=(1, 6), klens=(1, 257), topk=128, padded=True),
        dict(layout="TND", qlens=(3, 7), klens=(64, 129), topk=8192),
        dict(layout="TND", qlens=(7,), klens=(1024,), topk=1024, negative=True),
        dict(page=1024, qlens=(1,), klens=(1023,), topk=8192, zero=True),
        dict(page=128, qlens=(3,), klens=(513,), capacity=32768),
        dict(qlens=(1,), klens=(131328,), topk=512, mask=0, ratio=1),
    ],
)
def test_boundaries(device, options):
    case = make_case(device, **options)
    verify(case, True)
    verify(case, False)


def test_large_capacity_selection(device):
    gen = torch.Generator().manual_seed(83)
    data = (torch.randn((2, 70003), generator=gen) * 3).bfloat16()
    data[0, 0] = float("inf")
    data[1].fill_(-float("inf"))
    topk = 16385
    n = 2 * topk
    guard = 64
    ib = torch.full((n + guard * 2,), -123456, dtype=torch.int32, device=device)
    vb = torch.full((n + guard * 2,), 97, dtype=torch.bfloat16, device=device)
    indices = ib[slice(guard, guard + n)].view(2, topk)
    values = vb[slice(guard, guard + n)].view(2, topk)
    dummy = torch.empty((1, 1), dtype=torch.int32, device=device)
    run = make_streaming_select()
    for workers in (1, 4):
        run[None, workers](data.to(device), dummy, indices, values)
        torch.npu.synchronize()
        for r in range(2):
            check_topk(
                indices[r].cpu(),
                values[r].cpu(),
                data[r].float(),
                ~torch.isneginf(data[r]),
            )
        assert (ib[:guard] == -123456).all() and (ib[-guard:] == -123456).all()
        assert (vb[:guard] == 97).all() and (vb[-guard:] == 97).all()


@pytest.mark.parametrize("layout", ["PA_BBND", "TND"])
def test_optional_metadata(device, layout):
    case = make_case(device, layout=layout, qlens=(3,), klens=(256,), mask=0, ratio=1)
    args, kwargs, _, _ = case
    if layout == "PA_BBND":
        table = kwargs.pop("block_table")
        args[1] = args[1][table[0].long()].contiguous()
        args[4] = args[4][table[0].long()].contiguous()
    kwargs.pop("cu_seqlens_q")
    kwargs.pop("cu_seqlens_k", None)
    kwargs.pop("seqused_q")
    kwargs.pop("seqused_k")
    kwargs["max_seqlen_q"] = -1
    verify(case, True)


@pytest.mark.parametrize("used", [0, 19])
def test_fast_unused_pages(device, used):
    case = make_case(device, qlens=(7,), klens=(32768,), usedk=(used,))
    table = case[1]["block_table"]
    table[:, slice(1 if used else 0, None)].fill_(-1)
    verify(case, True)
    verify(case, False)


def test_dynamic_batch_and_sequence_reuse(device):
    from ._kernels.generic import make_score
    from ._kernels.paged import make_paged

    score = make_score(
        128,
        128,
        128,
        True,
        3,
        2,
        True,
        128,
        0,
        64,
        4,
        True,
        True,
        True,
        False,
        True,
        1024,
        512,
        True,
        0,
    )
    for qlens, klens in (((3, 4), (513, 777)), ((1, 2, 3), (600, 700, 1000))):
        verify(make_case(device, qlens=qlens, klens=klens), True)
    fast = make_paged(32768, 128, 128, 128, True, 3, 2, True, 256, 7, True, True)
    for qlens, klens in (((2,), (32768,)), ((1, 7), (32000, 32768))):
        verify(make_case(device, qlens=qlens, klens=klens), True)
    # Internal diagnostic only: this installation exposes the compiled-signature
    # map. Fail explicitly if the capability changes instead of guessing reuse.
    for run in (score, fast):
        cache = getattr(run, "_compiled_by_signature", None)
        assert isinstance(cache, dict), "Installed JIT lacks the signature-cache probe"
        assert len(cache) == 1, "Dynamic shapes created an additional compiled variant"


@pytest.mark.parametrize("capacity,klens", [(2048, (513, 777)), (16, (2051, 4091))])
def test_fused_candidates(device, capacity, klens):
    case = make_case(device, page=64, capacity=capacity, klens=klens)
    verify(case, True)
    verify(case, False)


def test_checker_rejects_corruption():
    with pytest.raises(AssertionError):
        check_topk(
            torch.tensor([0]),
            None,
            torch.tensor([1.0, 8.0]),
            torch.tensor([True, True]),
        )
    with pytest.raises(AssertionError):
        check_topk(
            torch.tensor([-1]),
            torch.tensor([0.0]),
            torch.tensor([1.0]),
            torch.tensor([False]),
        )


_UPSTREAM_CASES = [
    ("test_mxfp4_decode", ()),
    ("test_topk_reference_rejects_non_topk", ()),
    ("test_topk_reference_rejects_zero_padding", ()),
    ("test_quant_lightning_indexer", ("PA_BBND", 128, 0, False)),
    ("test_quant_lightning_indexer", ("PA_BBND", 64, 3, True)),
    ("test_quant_lightning_indexer", ("TND", 128, 3, False)),
    ("test_candidate16_ld_tail", ()),
    *(
        ("test_invalid_values_are_negative_infinity", (layout,))
        for layout in ("PA_BBND", "TND")
    ),
    *(
        ("test_all_rows_empty", (layout, empty))
        for layout in ("PA_BBND", "TND")
        for empty in ("query", "key")
    ),
    *(("test_typical_case", (12, 6, length)) for length in (65536, 131072)),
]


@pytest.fixture(scope="module")
def upstream_evaluator(device):
    root = os.environ.get("CANNBOT_DSL_ROOT")
    if not root:
        pytest.skip("CANNBOT_DSL_ROOT is required for the original evaluator")
    from ._tests.upstream import load_upstream

    workers = min(32, torch.npu.get_device_properties(device).cube_core_num)
    with tempfile.TemporaryDirectory(prefix="qli-evaluator-") as temporary:
        modules = load_upstream(Path(root).resolve(), Path(temporary), workers)
        tests = modules["tests"]
        tests._load_operators = lambda: (
            quant_lightning_indexer,
            modules["metadata"].quant_lightning_indexer_metadata,
        )
        yield tests


@pytest.mark.parametrize("case,arguments", _UPSTREAM_CASES)
def test_original_evaluator(upstream_evaluator, case, arguments):
    getattr(upstream_evaluator, case)(*arguments)
