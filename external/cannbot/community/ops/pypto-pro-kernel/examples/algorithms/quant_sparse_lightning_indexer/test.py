# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""The 12 upstream cases plus contract boundaries; CPU scheduling is test setup."""

import ast
import importlib.util
import os
import sys
from pathlib import Path
from itertools import accumulate

import numpy as np
import pytest
import torch

_HERE = Path(__file__).resolve().parent
if __package__:
    from .reference import _scores_reference, _visible_tokens
    from .reference import _check_topk
else:
    from reference import _scores_reference, _visible_tokens
    from reference import _check_topk


def _implementation():
    return (
        importlib.import_module(".kernel", __package__)
        if __package__
        else importlib.import_module("kernel")
    )


pytestmark = pytest.mark.soc("950")

HEADS = 32
BLOCK_SIZE = 8
CANDIDATE_CAPACITY = 2048
TOPK = 512
BLOCKS_PER_TILE = 64
METADATA_ELEMENTS = 1024
METADATA_FIELDS = 8
LD_RECORD_START = 36
LD_RECORD_LIMIT = 108


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def _schedule_cpu(
    cand_len,
    cu_q,
    used_q,
    used_k,
    residual,
    *,
    mask_mode,
    cmp_ratio,
    num_workers,
    merge_cap,
):
    cu = np.asarray(cu_q, dtype=np.int64)
    batch = len(used_q)
    rows = int(cu[batch])
    meta = np.zeros(METADATA_ELEMENTS, dtype=np.int32)
    if rows == 0:
        return (meta, 0)
    row_batch = np.repeat(np.arange(batch), np.diff(cu[: batch + 1]))
    row_group = np.arange(rows) - cu[row_batch]
    used = np.asarray(used_q, dtype=np.int64)[row_batch]
    actual = np.asarray(used_k, dtype=np.int64)[row_batch]
    res = np.asarray(residual, dtype=np.int64)[row_batch]
    if mask_mode == 0:
        visible = actual
    else:
        visible = np.minimum(
            actual,
            np.maximum(
                0, (actual * cmp_ratio + res - used + row_group + 1) // cmp_ratio
            ),
        )
    visible = np.where(row_group >= used, 0, visible)
    blocks = np.asarray(cand_len, dtype=np.int64)[:rows]
    tiles = np.where((visible > 0) & (blocks > 0), -(-blocks // BLOCKS_PER_TILE), 0)
    total = int(tiles.sum())
    if total == 0:
        meta[0] = 1
        meta[1] = 0
        meta[2] = 0
        meta[3] = 0
        meta[4] = batch - 1
        meta[5] = int(rows - 1 - cu[batch - 1])
        meta[6] = 1
        return (meta, 0)
    csum = np.concatenate(([0], np.cumsum(tiles)))
    atom = max(1, _ceil_div(int(tiles.max()), merge_cap))
    atoms = -(-tiles // atom)
    within = np.arange(int(atoms.sum())) - np.repeat(np.cumsum(atoms) - atoms, atoms)
    eligible = np.repeat(csum[:-1], atoms) + within * atom
    pick = np.arange(num_workers) * len(eligible) // num_workers
    bounds = np.append(eligible[pick], total)
    busy = bounds[:-1] < bounds[1:]
    cores = np.flatnonzero(busy)
    lo = bounds[:-1][busy]
    hi = bounds[1:][busy] - 1
    row_lo = np.searchsorted(csum, lo, side="right") - 1
    row_hi = np.searchsorted(csum, hi, side="right") - 1
    records = meta.reshape(-1, METADATA_FIELDS)
    records[cores, 0] = 1
    records[cores, 1] = row_batch[row_lo]
    records[cores, 2] = row_group[row_lo]
    records[cores, 3] = lo - csum[row_lo]
    records[cores, 4] = row_batch[row_hi]
    records[cores, 5] = row_group[row_hi]
    records[cores, 6] = hi - csum[row_hi] + 1
    t0 = lo - csum[row_lo]
    row_index = np.arange(rows)
    last_task_row = np.where(tiles > 0, row_index, -1)
    run_start = np.empty(rows, dtype=np.int64)
    run_start[0] = 0
    run_start[1:] = np.maximum.accumulate(last_task_row)[:-1] + 1
    span_lo = np.where(t0 == 0, run_start[row_lo], row_lo)
    records[cores, 1] = row_batch[span_lo]
    records[cores, 2] = row_group[span_lo]
    records[cores, 3] = lo - csum[span_lo]
    last_core = int(cores[-1])
    records[last_core, 4] = row_batch[rows - 1]
    records[last_core, 5] = row_group[rows - 1]
    live = np.flatnonzero(tiles)
    c_lo = np.searchsorted(bounds, csum[:-1][live], side="right") - 1
    c_hi = np.searchsorted(bounds, csum[1:][live] - 1, side="right") - 1
    split = c_hi > c_lo
    core_base = {}
    slot = 0
    shared = int(split.sum())
    if shared > LD_RECORD_LIMIT - LD_RECORD_START:
        raise ValueError(f"{shared} merged rows exceed the metadata ABI")
    for n, i in enumerate(np.flatnonzero(split)):
        held = [c for c in range(int(c_lo[i]), int(c_hi[i]) + 1) if busy[c]]
        record = LD_RECORD_START + n
        records[record, 0] = 1
        records[record, 1] = row_batch[live[i]]
        records[record, 2] = row_group[live[i]]
        records[record, 3] = slot
        records[record, 4] = len(held)
        records[record, 5] = 0
        records[record, 6] = 1
        for core in held:
            core_base.setdefault(core, slot)
            slot += 1
    for core, base in core_base.items():
        records[core, 7] = base
    return (meta, slot)


def _make_metadata(
    lengths,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    mask_mode=0,
    cmp_ratio=1,
    **kwargs,
):
    from pypto_pro.runtime.platform import get_platform_info

    workers = min(32, get_platform_info().cube_core_num)
    rows = lengths.numel()
    cu = cu_seqlens_q.cpu().tolist() if cu_seqlens_q is not None else [0, rows]
    qlen = seqused_q.cpu().tolist() if seqused_q is not None else np.diff(cu).tolist()
    if seqused_k is not None:
        klen = seqused_k.cpu().tolist()
    elif cu_seqlens_k is not None:
        klen = np.diff(cu_seqlens_k.cpu().tolist()).tolist()
    else:
        klen = [kwargs["max_seqlen_k"]] * len(qlen)
    residual = (
        cmp_residual_k.cpu().tolist() if cmp_residual_k is not None else [0] * len(qlen)
    )
    blob, _ = _schedule_cpu(
        lengths.cpu().reshape(-1).tolist(),
        cu,
        qlen,
        klen,
        residual,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        num_workers=workers,
        merge_cap=workers,
    )
    return torch.from_numpy(blob).to(lengths.device)


def _load_operators():
    quant_sparse_lightning_indexer = _implementation().quant_sparse_lightning_indexer

    return quant_sparse_lightning_indexer, _make_metadata


def _prefix(lengths):
    return [0, *accumulate(lengths)]


def _npu_device():
    pytest.importorskip("torch_npu")
    device = torch.device(
        "npu",
        int(os.environ.get("NPU_DEVICE_ID", os.environ.get("TILE_FWK_DEVICE_ID", "0"))),
    )
    torch.npu.set_device(device)
    return device


def _make_inputs(
    layout,
    page_size,
    mask,
    *,
    sparse,
    candidates=False,
    candidate_capacity=CANDIDATE_CAPACITY,
    key_lengths=(513, 777),
    topk=TOPK,
    query_lengths=None,
    signed_weights=False,
    uniform_scale=False,
    compression_ratio=None,
):
    generator = torch.Generator().manual_seed(20260929)
    typical = query_lengths is not None
    query_lengths = list(query_lengths) if typical else [3, 4]
    used_queries = query_lengths if typical else [3, 3]
    assert len(query_lengths) == len(key_lengths)
    total_queries = sum(query_lengths)
    ratio = (
        compression_ratio if compression_ratio is not None else (2 if mask == 3 else 1)
    )
    residual = [0] * len(query_lengths) if typical else [1, 0]

    def random_bytes(shape, low=0, high=256):
        return torch.randint(low, high, shape, generator=generator, dtype=torch.uint8)

    q = random_bytes((total_queries, HEADS, 64))
    qs = (
        torch.full((total_queries, HEADS, 2, 2), 127, dtype=torch.uint8)
        if uniform_scale
        else random_bytes((total_queries, HEADS, 2, 2), 124, 128)
    )
    weights = torch.rand((total_queries, HEADS), generator=generator)
    weights = weights * 2 - 1 if signed_weights else weights + 0.1
    logical_keys = [random_bytes((length, 1, 64)) for length in key_lengths]
    logical_scales = [
        torch.full((length, 1, 2, 2), 127, dtype=torch.uint8)
        if uniform_scale
        else random_bytes((length, 1, 2, 2), 124, 128)
        for length in key_lengths
    ]
    table = None
    if layout == "TND":
        k, ks = (torch.cat(logical_keys), torch.cat(logical_scales))
    else:
        max_pages = (max(key_lengths) + page_size - 1) // page_size
        pages = len(key_lengths) * max_pages
        table = (
            torch.randperm(pages, generator=generator)
            .int()
            .reshape(len(key_lengths), max_pages)
        )
        k = torch.zeros((pages, page_size, 1, 64), dtype=torch.uint8)
        ks = torch.full((pages, page_size, 1, 2, 2), 127, dtype=torch.uint8)
        for batch, length in enumerate(key_lengths):
            tokens = torch.arange(length)
            physical_rows = (
                table[batch, tokens // page_size].long() * page_size
                + tokens % page_size
            )
            k.view(-1, 1, 64).index_copy_(0, physical_rows, logical_keys[batch])
            ks.view(-1, 1, 2, 2).index_copy_(0, physical_rows, logical_scales[batch])
        if sparse:
            k = torch.cat(
                (
                    k.reshape(pages, page_size // 8, 512),
                    ks.reshape(pages, page_size // 8, 32),
                ),
                -1,
            )
            ks = None
    candidate_ids = torch.full(
        (total_queries, 1, CANDIDATE_CAPACITY), -1, dtype=torch.int32
    )
    candidate_lengths = torch.zeros((total_queries, 1), dtype=torch.int32)
    references, permitted = ([], [])
    for batch, (start, length) in enumerate(zip(_prefix(query_lengths), query_lengths)):
        reference = _scores_reference(
            q[slice(start, start + length)],
            logical_keys[batch],
            weights[slice(start, start + length)],
            qs[slice(start, start + length)],
            logical_scales[batch],
        )
        for row in range(length):
            visible = _visible_tokens(
                key_lengths[batch],
                used_queries[batch],
                row,
                mask,
                ratio,
                residual[batch],
            )
            allowed = torch.arange(key_lengths[batch]) < visible
            count = (key_lengths[batch] + 7) // 8
            chosen = torch.randperm(count, generator=generator)[
                : min(CANDIDATE_CAPACITY, max(1, count - 5))
            ].int()
            if sparse and (not typical) and (start + row == 1):
                chosen = chosen[:0]
            candidate_ids[start + row, 0, : chosen.numel()] = chosen
            candidate_lengths[start + row] = chosen.numel()
            if sparse:
                allowed &= torch.isin(torch.arange(key_lengths[batch]) // 8, chosen)
            references.append(reference[row])
            permitted.append(allowed)
    device = _npu_device()

    def to_device(tensor):
        return tensor.to(device) if tensor is not None else None

    sequence = dict(
        cu_seqlens_q=torch.tensor(
            _prefix(query_lengths), dtype=torch.int32, device=device
        ),
        seqused_q=torch.tensor(used_queries, dtype=torch.int32, device=device),
        seqused_k=torch.tensor(key_lengths, dtype=torch.int32, device=device),
    )
    if layout == "TND":
        sequence["cu_seqlens_k"] = torch.tensor(
            _prefix(key_lengths), dtype=torch.int32, device=device
        )
    if mask == 3:
        sequence["cmp_residual_k"] = torch.tensor(
            residual, dtype=torch.int32, device=device
        )
    attributes = dict(
        topk=topk,
        max_seqlen_q=max(query_lengths),
        layout_q="TND",
        layout_k=layout,
        mask_mode=mask,
        cmp_ratio=ratio,
    )
    metadata_attributes = dict(
        attributes,
        max_seqlen_k=max(key_lengths),
        num_heads_q=HEADS,
        num_heads_k=1,
        head_dim=128,
    )
    if not sparse:
        attributes.update(
            candidate_topk_blocks=candidate_capacity if candidates else -1,
            candidate_block_size=8 if candidates else -1,
        )
        metadata_attributes.update(
            candidate_topk_blocks=attributes["candidate_topk_blocks"],
            candidate_block_size=attributes["candidate_block_size"],
        )
    else:
        attributes["candidate_block_size"] = 8
        metadata_attributes.update(candidate_block_size=8, quant_mode=1)
    return dict(
        q=to_device(q),
        k=to_device(k),
        w=to_device(weights),
        descale_q=to_device(qs),
        descale_k=to_device(ks),
        block_table=to_device(table),
        sequence=sequence,
        attributes=attributes,
        metadata_attributes=metadata_attributes,
        candidate_ids=to_device(candidate_ids),
        candidate_lengths=to_device(candidate_lengths),
        reference=references,
        allowed=permitted,
        offsets=to_device(
            torch.arange(total_queries, dtype=torch.int32).reshape(-1, 1) * 10000
        ),
    )


def _check_kernel(
    layout,
    page_size,
    mask,
    *,
    sparse,
    candidates=False,
    candidate_capacity=CANDIDATE_CAPACITY,
    key_lengths=(513, 777),
    topk=TOPK,
    empty=None,
    query_lengths=None,
    signed_weights=False,
    uniform_scale=False,
):
    data = _make_inputs(
        layout,
        page_size,
        mask,
        sparse=sparse,
        candidates=candidates,
        candidate_capacity=candidate_capacity,
        key_lengths=key_lengths,
        topk=topk,
        query_lengths=query_lengths,
        signed_weights=signed_weights,
        uniform_scale=uniform_scale,
    )
    if empty is not None:
        if empty == "query":
            data["sequence"]["seqused_q"].zero_()
        elif empty == "key":
            data["sequence"]["seqused_k"].zero_()
        elif empty == "candidate" and sparse:
            data["candidate_lengths"].zero_()
        else:
            raise ValueError(f"Unsupported empty input: {empty}")
        for allowed in data["allowed"]:
            allowed.zero_()
    kernel, make_metadata = _load_operators()
    metadata_args = (data["candidate_lengths"],) if sparse else ()
    metadata = make_metadata(
        *metadata_args, **data["sequence"], **data["metadata_attributes"]
    )
    args = [data[key] for key in ("q", "k", "w", "descale_q")]
    if sparse:
        args.extend((data["candidate_ids"], data["candidate_lengths"]))
    else:
        args.append(data["descale_k"])
    kwargs = dict(
        data["sequence"],
        **data["attributes"],
        quant_mode=1,
        block_table=data["block_table"],
        output_idx_offset=data["offsets"],
        metadata=metadata,
    )
    if sparse:
        kwargs["descale_k"] = data["descale_k"]
    for return_value in (True, False):
        outputs = kernel(*args, **kwargs, return_value=return_value)
        torch.npu.synchronize()
        indices, values = (tensor.cpu() for tensor in outputs[:2])
        assert indices.dtype == torch.int32
        assert indices.shape == (len(data["reference"]), 1, topk)
        assert values.dtype == torch.bfloat16
        assert values.shape == indices.shape if return_value else values.numel() == 0
        for row, (scores, allowed) in enumerate(
            zip(data["reference"], data["allowed"])
        ):
            _check_topk(
                indices[row, 0],
                values[row, 0] if return_value else None,
                scores,
                allowed,
                row * 10000,
            )
        if not sparse:
            candidate_ids, candidate_lengths = (tensor.cpu() for tensor in outputs[2:])
            if not candidates:
                assert candidate_ids.numel() == candidate_lengths.numel() == 0
                continue
            assert candidate_ids.shape == (indices.shape[0], 1, candidate_capacity)
            for row, (scores, allowed) in enumerate(
                zip(data["reference"], data["allowed"])
            ):
                padded = torch.full(((scores.numel() + 7) // 8 * 8,), -float("inf"))
                padded[: scores.numel()] = scores.masked_fill(~allowed, -float("inf"))
                block_scores = padded.reshape(-1, 8).amax(-1)
                valid_blocks = torch.isfinite(block_scores)
                visible = int(allowed.sum())
                if visible % BLOCK_SIZE:
                    block_scores[visible // BLOCK_SIZE] = float("inf")
                count = min(candidate_capacity, int(valid_blocks.sum()))
                assert candidate_lengths[row, 0] == count
                _check_topk(
                    candidate_ids[row, 0, :count], None, block_scores, valid_blocks
                )


def test_sparse_topk_ignores_unselected_blocks():
    _check_topk(
        torch.tensor([0, 2]),
        torch.tensor([2.0, 1.0]),
        torch.tensor([2.0, 9.0, 1.0]),
        torch.tensor([True, False, True]),
    )


@pytest.mark.parametrize(
    "layout,page_size,mask",
    [
        pytest.param("PA_BBND", 128, 0, id="pa128_zero_candidate_tail"),
        pytest.param("PA_BBND", 64, 3, id="pa64_causal_zero_candidate_tail"),
        pytest.param("TND", 128, 3, id="tnd_causal_zero_candidate_tail"),
    ],
)
def test_quant_sparse_lightning_indexer(layout, page_size, mask):
    _check_kernel(layout, page_size, mask, sparse=True)


def test_topk16_ld_tail():
    _check_kernel("PA_BBND", 128, 0, sparse=True, topk=16)


@pytest.mark.parametrize("layout", ["PA_BBND", "TND"])
def test_invalid_values_are_negative_infinity(layout):
    _check_kernel(layout, 128, 0, sparse=True, key_lengths=(19, 73))


@pytest.mark.parametrize("layout", ["PA_BBND", "TND"])
@pytest.mark.parametrize("empty", ["query", "candidate"])
def test_all_rows_empty(layout, empty):
    _check_kernel(layout, 128, 0, sparse=True, key_lengths=(19, 73), empty=empty)


@pytest.mark.parametrize(
    "batch_size,query_length,key_length",
    [pytest.param(12, 6, 131072, id="b12_s1_6_n32_s2_128k_c2048")],
)
def test_typical_case(batch_size, query_length, key_length):
    _check_kernel(
        "PA_BBND",
        128,
        3,
        sparse=True,
        query_lengths=(query_length,) * batch_size,
        key_lengths=(key_length,) * batch_size,
        signed_weights=True,
        uniform_scale=True,
    )


def test_comparator_rejects_corruption():
    scores = torch.tensor([3.0, 2.0, 1.0])
    allowed = torch.ones(3, dtype=torch.bool)
    with pytest.raises(AssertionError):
        _check_topk(torch.tensor([2]), torch.tensor([1.0]), scores, allowed)
    with pytest.raises(AssertionError):
        _check_topk(torch.tensor([0]), torch.tensor([30.0]), scores, allowed)
    with pytest.raises(AssertionError):
        _check_topk(torch.tensor([0, 0]), None, scores, allowed)
    with pytest.raises(AssertionError):
        _check_topk(
            torch.tensor([0, 1, 2, -1]),
            torch.tensor([3.0, 2.0, 1.0, 0.0]),
            scores,
            allowed,
        )


@pytest.mark.parametrize("topk", [1, 15, 17, 513, 4097, 8192])
def test_topk_boundaries(topk):
    _check_kernel(
        "PA_BBND",
        128,
        0,
        sparse=True,
        topk=topk,
        key_lengths=(16384,),
        query_lengths=(1,),
        signed_weights=True,
    )


@pytest.mark.parametrize("page_size", [16, 48, 1024])
def test_page_boundaries(page_size):
    _check_kernel(
        "PA_BBND",
        page_size,
        3,
        sparse=True,
        key_lengths=(257, 515),
        signed_weights=True,
    )


def _check_data(data):
    kernel, make_metadata = _load_operators()
    metadata = make_metadata(
        data["candidate_lengths"], **data["sequence"], **data["metadata_attributes"]
    )
    args = [
        data[name]
        for name in ("q", "k", "w", "descale_q", "candidate_ids", "candidate_lengths")
    ]
    kwargs = dict(
        data["sequence"],
        **data["attributes"],
        quant_mode=1,
        block_table=data["block_table"],
        output_idx_offset=data["offsets"],
        metadata=metadata,
        descale_k=data["descale_k"],
    )
    for rv in (False, True):
        indices, values = kernel(*args, **kwargs, return_value=rv)
        torch.npu.synchronize()
        assert indices.shape == (len(data["reference"]), 1, data["attributes"]["topk"])
        assert indices.dtype == torch.int32 and values.dtype == torch.bfloat16
        assert values.shape == indices.shape if rv else values.numel() == 0
        indices, values = indices.cpu(), values.cpu()
        offsets = (
            data["offsets"].cpu().reshape(-1)
            if data["offsets"] is not None
            else torch.zeros(len(data["reference"]), dtype=torch.int32)
        )
        for row, (scores, allowed) in enumerate(
            zip(data["reference"], data["allowed"])
        ):
            _check_topk(
                indices[row, 0],
                values[row, 0] if rv else None,
                scores,
                allowed,
                int(offsets[row]),
            )


@pytest.mark.parametrize("layout", ["PA_BBND", "TND"])
@pytest.mark.parametrize("topk", [1, 17, 511, 512, 513, 4097])
def test_strides_scale_dtype_and_guards(layout, topk, monkeypatch):
    data = _make_inputs(layout, 128, 0, sparse=True, key_lengths=(19, 73), topk=topk)
    for name in ("k", "descale_k"):
        value = data[name]
        if value is None:
            continue
        storage = torch.empty(
            (value.shape[0] * 3, *value.shape[1:]),
            dtype=value.dtype,
            device=value.device,
        )
        padded = storage[::3]
        padded.copy_(value)
        data[name] = padded
    data["descale_q"] = data["descale_q"].view(torch.float8_e8m0fnu)
    if layout == "TND":
        data["descale_k"] = data["descale_k"].view(torch.float8_e8m0fnu)
    else:
        data["candidate_lengths"] = data["candidate_lengths"].reshape(-1)
    original_empty = torch.empty
    guards = []
    output_size = len(data["reference"]) * data["attributes"]["topk"]

    def guarded_empty(*args, **kwargs):
        value = original_empty(*args, **kwargs)
        if value.device.type != "npu":
            return value
        value.view(torch.uint8).fill_(0xA5)
        if value.numel() == output_size and value.dtype in (
            torch.int32,
            torch.bfloat16,
        ):
            backing = original_empty(
                (output_size + 64,), dtype=value.dtype, device=value.device
            )
            backing.fill_(-37)
            guards.append(backing)
            return backing[32:-32].reshape(value.shape)
        return value

    monkeypatch.setattr(torch, "empty", guarded_empty)
    _check_data(data)
    for backing in guards:
        assert (backing[:32].cpu() == -37).all()
        assert (backing[-32:].cpu() == -37).all()
    assert guards


@pytest.mark.parametrize("factor", [0.0, -1.0, 2.0**-20, 2.0**20])
def test_weight_range(factor):
    data = _make_inputs("PA_BBND", 128, 0, sparse=True, key_lengths=(513, 777))
    data["w"] = data["w"] * factor
    data["reference"] = [scores * factor for scores in data["reference"]]
    _check_data(data)


@pytest.mark.parametrize("name", ["k", "descale_k"])
def test_independent_tnd_strides(name):
    data = _make_inputs("TND", 128, 3, sparse=True, key_lengths=(19, 73))
    value = data[name]
    storage = torch.empty(
        (value.shape[0] * 2, *value.shape[1:]), dtype=value.dtype, device=value.device
    )
    data[name] = storage[::2]
    data[name].copy_(value)
    _check_data(data)


@pytest.mark.parametrize("page_size", [48, 128])
@pytest.mark.parametrize("single_dma", [False, True])
def test_pa_address_and_dma_dispatch(page_size, single_dma, monkeypatch):
    impl = _implementation()

    monkeypatch.setattr(impl, "MAX_ADDRESS32_BYTES", 0)
    monkeypatch.setattr(impl, "_batch_consistency_enabled", lambda: single_dma)
    _check_kernel(
        "PA_BBND",
        page_size,
        3,
        sparse=True,
        key_lengths=(257, 515),
        signed_weights=True,
    )


@pytest.mark.parametrize("with_table", [False, True])
def test_pa_page_table_cache_boundary(with_table, monkeypatch):
    impl = _implementation()

    launches = _record_keys(impl, monkeypatch)
    for pages, mode in ((2047, 1), (2048, 1), (2049, 0)):
        data = _make_inputs(
            "PA_BBND",
            16,
            3,
            sparse=True,
            key_lengths=(pages * 16,),
            query_lengths=(1,),
            signed_weights=True,
        )
        if not with_table:
            data["k"] = data["k"].index_select(0, data["block_table"][0].long())
            data["block_table"] = None
        _check_data(data)
        assert all(key["PAGatherMode"] == mode for key in launches[-2:])
    assert launches[:2] == launches[2:4]


def test_candidate_tile_boundaries_and_ignored_suffix():
    data = _make_inputs(
        "PA_BBND",
        128,
        0,
        sparse=True,
        key_lengths=(16401,),
        query_lengths=(7,),
        signed_weights=True,
    )
    lengths = [0, 1, 63, 64, 65, 2047, 2048]
    ids = data["candidate_ids"].cpu()
    for row, length in enumerate(lengths):
        prefix = ids[row, 0, :length].clone()
        ids[row, 0, length:] = 0x7FFFFFFF
        data["allowed"][row] &= torch.isin(torch.arange(16401) // 8, prefix)
    data["candidate_ids"] = ids.to(data["q"].device)
    data["candidate_lengths"] = torch.tensor(
        lengths, dtype=torch.int32, device=data["q"].device
    ).reshape(-1, 1)
    _check_data(data)


@pytest.mark.parametrize("layout", ["PA_BBND", "TND"])
def test_pipeline_fill_drain_and_row_reuse(layout, monkeypatch):
    from dataclasses import replace
    import pypto_pro.runtime.platform as platform

    impl = _implementation()

    # One MIX worker retains whole rows so short panels exercise startup,
    # drain, slot wraparound and the following row's prefetched payload.
    single_worker = replace(platform.get_platform_info(), cube_core_num=1)
    monkeypatch.setattr(platform, "get_platform_info", lambda: single_worker)
    monkeypatch.setattr(impl, "get_platform_info", lambda: single_worker)
    lengths = [0, 1, 64, 65, 128, 129, 192, 193, 256, 257, 0]
    data = _make_inputs(
        layout,
        128,
        3,
        sparse=True,
        key_lengths=(4101,),
        query_lengths=(len(lengths),),
        signed_weights=True,
    )
    ids = data["candidate_ids"].cpu()
    for row, length in enumerate(lengths):
        prefix = ids[row, 0, :length].clone()
        ids[row, 0, length:] = 0x7FFFFFFF
        data["allowed"][row] &= torch.isin(torch.arange(4101) // 8, prefix)
    data["candidate_ids"] = ids.to(data["q"].device)
    data["candidate_lengths"] = torch.tensor(
        lengths, dtype=torch.int32, device=data["q"].device
    ).reshape(-1, 1)
    for _ in range(2):
        _check_data(data)


def test_dynamic_shape_reuse(monkeypatch):
    impl = _implementation()

    cache = getattr(impl.qsli_kernel, "_compiled_by_signature", None)
    launches = _record_keys(impl, monkeypatch)
    first = _make_inputs(
        "PA_BBND", 128, 0, sparse=True, key_lengths=(513, 777), query_lengths=(2, 3)
    )
    _check_data(first)
    signatures = len(cache) if isinstance(cache, dict) else None
    second = _make_inputs(
        "PA_BBND", 128, 0, sparse=True, key_lengths=(1025, 1537), query_lengths=(3, 5)
    )
    _check_data(second)
    assert launches[:2] == launches[2:]
    for queries, keys in (((7,), (1027,)), ((1, 2, 4), (121, 457, 979))):
        _check_data(
            _make_inputs(
                "PA_BBND", 128, 0, sparse=True, query_lengths=queries, key_lengths=keys
            )
        )
        assert launches[-2:] == launches[:2]
    if signatures is not None:
        assert len(cache) == signatures, (
            "Runtime axes introduced a new compiled specialization"
        )


def _record_keys(impl, monkeypatch):
    launches = []
    original = type(impl.qsli_kernel).__getitem__

    def record(kernel, config):
        if kernel is impl.qsli_kernel:
            launches.append(dict(config[2]))
        return original(kernel, config)

    monkeypatch.setattr(type(impl.qsli_kernel), "__getitem__", record)
    return launches


@pytest.mark.parametrize("layout", ["PA_BBND", "TND"])
def test_runtime_attributes_reuse(layout, monkeypatch):
    impl = _implementation()

    launches = _record_keys(impl, monkeypatch)
    cache = getattr(impl.qsli_kernel, "_compiled_by_signature", None)
    signatures = None
    for index, (page, ratio, topk) in enumerate(
        ((16, 2, 257), (48, 3, 383), (256, 5, 511), (1024, 128, 512))
    ):
        data = _make_inputs(
            layout,
            page,
            3,
            sparse=True,
            key_lengths=(263 + index, 533 + index),
            compression_ratio=ratio,
            topk=topk,
        )
        _check_data(data)
        assert launches[-2:] == launches[:2], (
            "Runtime attributes changed the tiling key"
        )
        if isinstance(cache, dict):
            if signatures is None:
                signatures = len(cache)
            else:
                assert len(cache) == signatures, (
                    "Runtime attributes recompiled the kernel"
                )


@pytest.mark.parametrize("topk", [31, 33, 127, 129, 255, 257, 1023, 1025, 2047, 2049])
def test_topk_capacity_tails(topk):
    _check_kernel(
        "PA_BBND",
        48,
        3,
        sparse=True,
        topk=topk,
        key_lengths=(4099,),
        query_lengths=(1,),
        signed_weights=True,
    )


def test_no_sugar_dependency(monkeypatch):
    import pypto_pro.language as pl

    impl = _implementation()

    source = Path(impl.__file__).read_text()
    forbidden = {"gemm", "gemm_mx", "mem_space"}
    assert not any(
        isinstance(node, ast.Attribute) and node.attr in forbidden
        for node in ast.walk(ast.parse(source))
    )
    for name in forbidden:
        monkeypatch.delattr(pl, name, raising=False)
    spec = importlib.util.spec_from_file_location("_qsli_without_sugar", impl.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, impl.__name__, module)
    for layout in ("PA_BBND", "TND"):
        _check_data(
            _make_inputs(layout, 48, 3, sparse=True, topk=23, key_lengths=(19, 73))
        )


@pytest.mark.parametrize("layout", ["PA_BBND", "TND"])
def test_optional_sequence_inputs(layout):
    lengths = (512,) if layout == "PA_BBND" else (513, 777)
    queries = (5,) if layout == "PA_BBND" else (2, 3)
    data = _make_inputs(
        layout, 128, 0, sparse=True, key_lengths=lengths, query_lengths=queries
    )
    del data["sequence"]["seqused_q"]
    del data["sequence"]["seqused_k"]
    data["offsets"] = None
    if layout == "PA_BBND":
        data["k"] = data["k"].index_select(0, data["block_table"][0].long())
        data["block_table"] = None
        del data["sequence"]["cu_seqlens_q"]
    _check_data(data)


def test_repeated_runtime_lengths():
    data = _make_inputs("PA_BBND", 128, 0, sparse=True, key_lengths=(513, 777))
    counts = data["candidate_lengths"].clone()
    permitted = [allowed.clone() for allowed in data["allowed"]]
    for _ in range(3):
        _check_data(data)
        data["candidate_lengths"].zero_()
        for allowed in data["allowed"]:
            allowed.zero_()
        _check_data(data)
        data["candidate_lengths"].copy_(counts)
        data["allowed"] = [allowed.clone() for allowed in permitted]


if __name__ == "__main__":
    torch.set_num_threads(8)
    raise SystemExit(pytest.main([str(Path(__file__).resolve()), "-q", *sys.argv[1:]]))
