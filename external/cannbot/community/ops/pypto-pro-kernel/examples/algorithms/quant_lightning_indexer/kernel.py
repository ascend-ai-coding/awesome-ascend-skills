# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：混合（原生 Cube/VF；Host 仅验证、分配和分发）。

The public MXFP4 QLI interface matches CANNBot DSL's four Tensor outputs.
Packed uint8 Q has shape [T,32,64]; grouped E8M0 scales cover 32 logical
features. PA_BBND and TND K layouts, causal compression, variable lengths,
candidate blocks, output offsets and optional scores are supported. K and
its scale may have independent padded row strides. Metadata is optional;
native kernels schedule from device-resident sequence arrays without Host
readback. The native implementation does not depend on CANNBot DSL.

QK and weights round to BF16; the sequential head FMA rounds after every head.
Indices are unordered with arbitrary ties, invalid items are -1/-inf, and the
library reference additionally requires exact scores and an exact Top-K boundary.

_kernels/paged.py handles PA128, TopK=512 without candidates up to 128K keys,
subject to merge-list capacity. _kernels/generic.py fuses scoring and selection
up to 16384 keys and candidate reduction up to capacity 8192. Larger inputs use
bounded selection or streaming-selection blocks. Shared native VF blocks keep
their explicit addresses, aliases and events. Shape changes reuse dynamic ABI
fields; only resource/layout/rounding choices select compilation variants.
"""

from math import gcd
import torch
from ._kernels.generic import make_score
from ._kernels.selection import make_select
from ._kernels.candidates import make_blocks
from ._kernels.paged import make_paged
from ._kernels.streaming_selection import make_streaming_select


def _select(scores, topk, offsets, workers=56):
    rows, width = scores.shape
    dummy = torch.empty((1, 1), dtype=torch.int32, device=scores.device)
    if topk > 8192:
        indices = torch.empty((rows, topk), dtype=torch.int32, device=scores.device)
        values = torch.empty((rows, topk), dtype=torch.bfloat16, device=scores.device)
        make_streaming_select(offsets is not None)[None, workers](
            scores, offsets if offsets is not None else dummy, indices, values
        )
        return indices.view(rows, 1, topk), values.view(rows, 1, topk)
    mapping = dummy
    mapped = False
    while True:
        parts = (width + 16383) // 16384
        indices = torch.empty(
            (rows, parts, topk), dtype=torch.int32, device=scores.device
        )
        values = torch.empty(
            (rows, parts, topk), dtype=torch.bfloat16, device=scores.device
        )
        final = parts == 1
        make_select(
            min(16384, (width + 255) // 256 * 256),
            topk,
            mapped=mapped,
            add_offset=final and offsets is not None,
        )[None, workers](
            scores, mapping, offsets if offsets is not None else dummy, indices, values
        )
        if final:
            return indices, values
        mapping = indices.view(rows, -1)
        scores = values.view(rows, -1)
        width = scores.shape[1]
        mapped = True


def quant_lightning_indexer(
    q,
    k,
    w,
    descale_q,
    descale_k,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_table=None,
    output_idx_offset=None,
    metadata=None,
    *,
    topk,
    quant_mode,
    max_seqlen_q=-1,
    mask_mode=0,
    cmp_ratio=1,
    layout_q="TND",
    layout_k="TND",
    return_value=False,
    candidate_topk_blocks=-1,
    candidate_block_size=-1,
):
    if q.ndim != 3 or tuple(q.shape[1:]) != (32, 64) or q.dtype != torch.uint8:
        raise ValueError("q must be packed uint8 [T,32,64]")
    if quant_mode != 1 or layout_q != "TND" or layout_k not in ("TND", "PA_BBND"):
        raise ValueError("expected quant_mode=1, Q=TND and K=TND/PA_BBND")
    if not 1 <= topk <= 8192 or mask_mode not in (0, 3) or not 1 <= cmp_ratio <= 128:
        raise ValueError("invalid topk, mask_mode or cmp_ratio")
    if candidate_topk_blocks == -1:
        if candidate_block_size != -1:
            raise ValueError("disabled candidates require block_size=-1")
    elif candidate_topk_blocks <= 0 or candidate_block_size != 8:
        raise ValueError("candidates require positive capacity and block_size=8")
    rows = q.shape[0]
    if q.device.type != "npu" or k.shape[0] <= 0:
        raise ValueError("NPU tensors and nonempty key storage are required")
    batches = cu_seqlens_q.numel() - 1 if cu_seqlens_q is not None else 1
    if rows <= 0 or batches <= 0:
        raise ValueError("positive query and batch dimensions are required")
    if tuple(w.shape) != (rows, 32) or w.dtype != torch.float32:
        raise ValueError("w must be float32 [T,32]")
    if tuple(descale_q.shape) != (rows, 32, 2, 2) or descale_q.dtype != torch.uint8:
        raise ValueError("descale_q must be uint8 [T,32,2,2]")
    for name, t in [("q", q), ("w", w), ("descale_q", descale_q)]:
        if not t.is_contiguous() or t.device != q.device:
            raise ValueError(f"{name} must be contiguous on the query device")
    for name, t in [("k", k), ("descale_k", descale_k)]:
        if t.dtype != torch.uint8 or t.device != q.device:
            raise ValueError(f"{name} must use uint8 on the query device")
    for name, t, shape in [
        ("cu_seqlens_q", cu_seqlens_q, (batches + 1,)),
        ("cu_seqlens_k", cu_seqlens_k, (batches + 1,)),
        ("seqused_q", seqused_q, (batches,)),
        ("seqused_k", seqused_k, (batches,)),
        ("cmp_residual_k", cmp_residual_k, (batches,)),
        ("output_idx_offset", output_idx_offset, (rows, 1)),
    ]:
        if t is not None and (
            tuple(t.shape) != shape
            or t.dtype != torch.int32
            or t.device != q.device
            or not t.is_contiguous()
        ):
            raise ValueError(f"invalid {name}")
    needs_residual = mask_mode == 3 and cmp_ratio != 1
    if needs_residual != (cmp_residual_k is not None):
        raise ValueError(
            "compression residual is required exactly for compressed causal mode"
        )
    if metadata is not None and (
        metadata.shape != (1024,)
        or metadata.dtype != torch.int32
        or metadata.device != q.device
        or not metadata.is_contiguous()
    ):
        raise ValueError("metadata must be contiguous int32[1024] on the query device")
    paged = layout_k == "PA_BBND"
    if paged:
        if (
            k.ndim != 4
            or tuple(k.shape[2:]) != (1, 64)
            or tuple(descale_k.shape) != (k.shape[0], k.shape[1], 1, 2, 2)
        ):
            raise ValueError("invalid PA key or scale shape")
        page = k.shape[1]
        if not 0 < page <= 1024 or page % 16:
            raise ValueError("page size must be a multiple of 16, at most 1024")
        if k.stride()[1:] != (64, 64, 1) or descale_k.stride()[1:] != (4, 4, 2, 1):
            raise ValueError("only the page axis may have storage padding")
        if (
            k.stride(0) < page * 64
            or k.stride(0) % 64
            or descale_k.stride(0) < page * 4
            or descale_k.stride(0) % 4
        ):
            raise ValueError(
                "page strides must preserve complete, non-overlapping rows"
            )
        if block_table is not None and (
            block_table.ndim != 2
            or block_table.shape[0] != batches
            or block_table.dtype != torch.int32
            or block_table.device != q.device
            or not block_table.is_contiguous()
        ):
            raise ValueError("invalid block_table")
        if block_table is None and batches != 1:
            raise ValueError("multiple PA batches require block_table")
        kp = k.stride(0) // 64
        ksp = descale_k.stride(0) // 4
        krows = (k.shape[0] - 1) * kp + page
        srows = (k.shape[0] - 1) * ksp + page
        kv = k.as_strided((krows, 64), (64, 1))
        sv = descale_k.as_strided((srows, 2, 2), (4, 2, 1))
        table_width = block_table.shape[1] if block_table is not None else k.shape[0]
        max_k = table_width * page
        k_stride, s_stride = 64, 4
        tile_n = gcd(page, 128)
    else:
        if (
            k.ndim != 3
            or tuple(k.shape[1:]) != (1, 64)
            or tuple(descale_k.shape) != (k.shape[0], 1, 2, 2)
        ):
            raise ValueError("invalid TND key or scale shape")
        if block_table is not None or (batches > 1 and cu_seqlens_k is None):
            raise ValueError("TND uses K prefixes, without a page table")
        if k.stride()[1:] != (64, 1) or descale_k.stride()[1:] != (4, 2, 1):
            raise ValueError("only the token axis may have storage padding")
        k_stride, s_stride = k.stride(0), descale_k.stride(0)
        if k_stride < 64 or s_stride < 4:
            raise ValueError("token rows must not overlap")
        krows = srows = k.shape[0]
        kv = k.as_strided((krows, 64), (k_stride, 1))
        sv = descale_k.as_strided((srows, 2, 2), (s_stride, 2, 1))
        max_k = krows
        page = 128
        kp = ksp = 0
        table_width = 1
        tile_n = 128
    max_q = rows if max_seqlen_q <= 0 else max_seqlen_q
    dummy = torch.empty((1,), dtype=torch.int32, device=q.device)
    table = block_table if block_table is not None else dummy.view(1, 1)
    seq = [
        v if v is not None else dummy
        for v in (cu_seqlens_q, seqused_q, seqused_k, cmp_residual_k, cu_seqlens_k)
    ]
    properties = torch.npu.get_device_properties(q.device.index)
    cube_workers = properties.cube_core_num
    vector_workers = properties.vector_core_num
    split_fast = 7
    shard_tiles = (max_k + 256 * split_fast - 1) // (256 * split_fast)
    shard_width = shard_tiles * 256
    parts = ((max_k + 255) // 256 + shard_tiles - 1) // shard_tiles
    if (
        paged
        and page == 128
        and max_k >= 8192
        and max_k <= 131072
        and max_k % 256 == 0
        and topk == 512
        and candidate_topk_blocks == -1
        and cu_seqlens_q is not None
        and seqused_q is not None
        and seqused_k is not None
        and parts * 512 <= shard_width
    ):
        partial_idx = torch.empty(
            (rows, parts * 512), dtype=torch.int32, device=q.device
        )
        partial_val = torch.empty(
            (rows, parts * 512), dtype=torch.bfloat16, device=q.device
        )
        indices = torch.empty((rows, 512), dtype=torch.int32, device=q.device)
        values = torch.empty((rows, 512), dtype=torch.bfloat16, device=q.device)
        make_paged(
            max_k,
            page,
            kp,
            ksp,
            True,
            mask_mode,
            cmp_ratio,
            needs_residual,
            256,
            split_fast,
            output_idx_offset is not None,
            return_value,
        )[None, cube_workers](
            q.view(-1, 64),
            kv,
            w,
            descale_q.view(-1, 2, 2),
            sv,
            *seq[:4],
            table,
            partial_idx,
            partial_val,
            indices,
            values,
            output_idx_offset if output_idx_offset is not None else dummy.view(1, 1),
            max_q,
        )
        empty = torch.empty((0,), dtype=torch.int32, device=q.device)
        return (
            indices.view(rows, 1, 512),
            values.view(rows, 1, 512) if return_value else values.reshape(-1)[:0],
            empty,
            empty,
        )
    scores = torch.empty((rows, max_k), dtype=torch.bfloat16, device=q.device)
    select_width = (max_k + 255) // 256 * 256 if max_k <= 16384 else 0
    indices = torch.empty((rows, 1, topk), dtype=torch.int32, device=q.device)
    values = torch.empty((rows, 1, topk), dtype=torch.bfloat16, device=q.device)
    count = candidate_topk_blocks
    fused_candidates = count if select_width and 0 < count <= 8192 else 0
    if count > 0:
        candidates = torch.empty((rows, 1, count), dtype=torch.int32, device=q.device)
        lengths = torch.empty((rows, 1), dtype=torch.int32, device=q.device)
    else:
        candidates = torch.empty((0,), dtype=torch.int32, device=q.device)
        lengths = torch.empty((0,), dtype=torch.int32, device=q.device)
    make_score(
        page,
        kp,
        ksp,
        paged,
        mask_mode,
        cmp_ratio,
        needs_residual,
        tile_n,
        0,
        k_stride,
        s_stride,
        cu_seqlens_q is not None,
        seqused_q is not None,
        seqused_k is not None,
        cu_seqlens_k is not None,
        block_table is not None,
        select_width,
        topk,
        output_idx_offset is not None,
        fused_candidates,
    )[None, cube_workers](
        q.view(-1, 64),
        kv,
        w,
        descale_q.view(-1, 2, 2),
        sv,
        *seq[:4],
        table,
        seq[4],
        scores,
        max_q,
        output_idx_offset if output_idx_offset is not None else dummy.view(1, 1),
        indices,
        values,
        candidates if count > 0 else dummy.view(1, 1, 1),
        lengths if count > 0 else dummy.view(1, 1),
    )
    if not select_width:
        indices, values = _select(scores, topk, output_idx_offset, vector_workers)
    if count > 0 and not fused_candidates:
        maxima = torch.empty(
            (rows, (max_k + 7) // 8), dtype=torch.bfloat16, device=q.device
        )
        make_blocks(
            count,
            mask_mode,
            cmp_ratio,
            cu_seqlens_q is not None,
            cu_seqlens_k is not None,
            seqused_q is not None,
            seqused_k is not None,
            needs_residual,
        )[None, vector_workers](
            scores, seq[0], seq[4], seq[1], seq[2], seq[3], maxima, lengths
        )
        candidates, _ = _select(maxima, count, None, vector_workers)
    return (
        indices,
        values if return_value else values.reshape(-1)[:0],
        candidates,
        lengths,
    )
