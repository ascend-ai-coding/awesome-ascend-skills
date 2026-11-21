# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent selected-row mathematics and the original CANN precision checker.

CANN_BENCH_ROOT selects the external evaluator checkout. Its src/ directory
must be on PYTHONPATH so that the original kernel_eval checker is importable.
The test calls the golden with both FP64 and native FP16 inputs, preserving
the evaluator's cancellation and small-value rules.
"""

from pathlib import Path
import runpy

import torch


def reference(query, key, value, indices, scale, layout="BSND", causal=False):
    """FP64 softmax over selected original coordinates; no native block order."""
    if layout not in ("BSND", "BNSD"):
        raise ValueError("layout must be BSND or BNSD")
    q, k, v = [x.detach().cpu().double() for x in (query, key, value)]
    selected = indices.detach().cpu().long()
    if layout == "BSND":
        q, k, v, selected = [x.transpose(1, 2) for x in (q, k, v, selected)]
    batch, query_heads, query_rows, _ = q.shape
    kv_heads, key_rows = k.shape[1:3]
    if query_heads % kv_heads or (causal and query_rows > key_rows):
        raise ValueError("invalid GQA ratio or causal query extent")
    if bool(((selected < 0) | (selected >= key_rows)).any()):
        raise ValueError("selected indices must be in the original key range")
    ordered = selected.sort(dim=-1).values
    if bool((ordered[..., 1:] == ordered[..., :-1]).any()):
        raise ValueError("selected indices must be distinct within each query/KV head")
    group = query_heads // kv_heads
    result = torch.empty(
        (batch, query_heads, query_rows, v.shape[-1]), dtype=torch.float64, device="cpu"
    )
    # Small CPU chunks bound golden memory; all selected keys take part at once.
    for b in range(batch):
        for head in range(kv_heads):
            for first in range(0, query_rows, 16):
                end = min(query_rows, first + 16)
                rows = selected[b, head, first:end]
                queries = q[b, head * group: (head + 1) * group, first:end].transpose(
                    0, 1
                )
                keys, values = k[b, head, rows], v[b, head, rows]
                scores = torch.einsum("qhd,qkd->qhk", queries, keys) * scale
                valid = torch.ones_like(rows, dtype=torch.bool)
                if causal:
                    limit = (
                        torch.arange(first, end, device="cpu") + key_rows - query_rows
                    )
                    valid = rows <= limit[:, None]
                scores.masked_fill_(~valid[:, None, :], -torch.inf)
                nonempty = valid.any(dim=-1)[:, None, None]
                maximum = scores.max(dim=-1, keepdim=True).values
                maximum = torch.where(nonempty, maximum, torch.zeros_like(maximum))
                weights = (scores - maximum).exp()
                denominator = weights.sum(dim=-1, keepdim=True)
                weights = weights / torch.where(
                    nonempty, denominator, torch.ones_like(denominator)
                )
                output = torch.einsum("qhk,qkd->qhd", weights, values)
                result[b, head * group: (head + 1) * group, first:end] = (
                    output.transpose(0, 1)
                )
    return result.transpose(1, 2).contiguous() if layout == "BSND" else result


def representative_inputs(case):
    """Two public workloads: small GQA groups and wide prefix-V head groups."""
    if case == "gqa":
        b, sq, sk, hq, hkv, dk, dv, selected = 2, 256, 4096, 32, 8, 192, 128, 512
        layout, causal = "BSND", False
    elif case == "wide":
        b, sq, sk, hq, hkv, dk, dv, selected = 16, 4, 8192, 64, 1, 576, 512, 512
        layout, causal = "BNSD", True
    else:
        raise ValueError("case must be gqa or wide")
    generator = torch.Generator().manual_seed(947)
    query = (
        2 * torch.rand((b, sq, hq, dk), generator=generator, device="cpu") - 1
    ).half()
    key = (
        2 * torch.rand((b, sk, hkv, dk), generator=generator, device="cpu") - 1
    ).half()
    indices = (
        torch.stack(
            [
                torch.randperm(sk, generator=generator, device="cpu")[:selected]
                for _ in range(b * sq * hkv)
            ]
        )
        .view(b, sq, hkv, selected)
        .int()
    )
    if layout == "BNSD":
        query, key, indices = [x.transpose(1, 2).contiguous() for x in (query, key, indices)]
    value = key[..., :dv].contiguous()
    return query, key, value, indices, dk**-0.5, layout, causal


def load_evaluator(root):
    from kernel_eval.utils.compare import compare_tensors

    source = Path(root) / "tasks/level4/sparse_flash_attention/golden.py"
    golden = runpy.run_path(str(source))["sparse_flash_attention"]
    return golden, compare_tensors
