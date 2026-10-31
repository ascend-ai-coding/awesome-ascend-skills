# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent FP64 math reference for the MLA prolog kernel.

Inputs are the BF16 values actually handed to the device; every product, row
statistic and rotation is computed in float64 on CPU. The torch golden shipped
with the CANN-Bench task narrows outputs to BF16; ``mla_prolog_reference`` keeps
FP64 so the test can measure the candidate's own rounding.
"""

import torch


def _rms_norm(x, gamma, epsilon):
    xf = x.double()
    rms = torch.sqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + epsilon)
    return gamma.double() * xf / rms


def _rope(x, cos, sin):
    xf = x.double()
    x1, x2 = xf.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return xf * cos.double() + rotated * sin.double()


def native_reference(*args, **kwargs):
    """FP32 chain from the BF16 inputs: the evaluator's same-precision reference.

    The candidate's own BF16 activation rounding is measured against this, so the
    test can reproduce the evaluator's small-value/cancellation counters.
    """
    kwargs["compute_dtype"] = torch.float32
    return mla_prolog_reference(*args, **kwargs)


def mla_prolog_reference(
    token_x,
    w_dq,
    w_uq_qr,
    w_uk,
    w_dkv_kr,
    rmsnorm_gamma_cq,
    rmsnorm_gamma_ckv,
    rope_sin,
    rope_cos,
    n_heads,
    rmsnorm_epsilon_cq=1e-5,
    rmsnorm_epsilon_ckv=1e-5,
    compute_dtype=torch.float64,
):
    """Return the four outputs in ``compute_dtype`` from the BF16 inputs."""
    b, s, _ = token_x.shape
    n = int(n_heads)
    hckv = w_uk.shape[2]
    d = w_uk.shape[1]
    dr = rope_sin.shape[-1]

    x = token_x.to(compute_dtype)
    if compute_dtype == torch.float64:
        c_q = _rms_norm(
            torch.matmul(x, w_dq.double()), rmsnorm_gamma_cq, rmsnorm_epsilon_cq
        )
        qr = torch.matmul(c_q, w_uq_qr.double()).reshape(b, s, n, d + dr)
        q_c = qr[..., :d]
        query = torch.einsum("bsnd,ndh->bsnh", q_c, w_uk.double())
        cos = rope_cos.double().unsqueeze(2).expand(-1, -1, n, -1)
        sin = rope_sin.double().unsqueeze(2).expand(-1, -1, n, -1)
        query_rope = _rope(qr[..., d:], cos, sin)
        dkv = torch.matmul(x, w_dkv_kr.double())
        c_kv = _rms_norm(dkv[..., :hckv], rmsnorm_gamma_ckv, rmsnorm_epsilon_ckv)
        k_rope = _rope(dkv[..., hckv:], rope_cos.double(), rope_sin.double())
        return query, query_rope, c_kv, k_rope

    # FP32 chain, mirroring the task golden's native path.
    def w(t):
        return t.to(compute_dtype)

    c_q_raw = torch.matmul(x, w(w_dq))
    rms = torch.sqrt(
        torch.mean(c_q_raw * c_q_raw, dim=-1, keepdim=True) + rmsnorm_epsilon_cq
    )
    c_q = w(rmsnorm_gamma_cq) * c_q_raw / rms
    qr = torch.matmul(c_q, w(w_uq_qr)).reshape(b, s, n, d + dr)
    query = torch.einsum("bsnd,ndh->bsnh", qr[..., :d], w(w_uk))
    cos = w(rope_cos).unsqueeze(2).expand(-1, -1, n, -1)
    sin = w(rope_sin).unsqueeze(2).expand(-1, -1, n, -1)
    x1, x2 = qr[..., d:].chunk(2, dim=-1)
    query_rope = qr[..., d:] * cos + torch.cat([-x2, x1], dim=-1) * sin
    dkv = torch.matmul(x, w(w_dkv_kr))
    ckv = dkv[..., :hckv]
    rms2 = torch.sqrt(torch.mean(ckv * ckv, dim=-1, keepdim=True) + rmsnorm_epsilon_ckv)
    c_kv = w(rmsnorm_gamma_ckv) * ckv / rms2
    k1, k2 = dkv[..., hckv:].chunk(2, dim=-1)
    k_rope = dkv[..., hckv:] * w(rope_cos) + torch.cat([-k2, k1], dim=-1) * w(rope_sin)
    return query, query_rope, c_kv, k_rope


def relative_error_metrics(actual, reference):
    """Evaluator-style MERE/MARE: both sides narrowed to the candidate dtype."""
    target = actual.dtype
    got = actual.double()
    want = reference.to(target).double()
    diff = (got - want).abs()
    rel = diff / (want.abs() + 1e-7)
    finite = torch.isfinite(rel)
    return float(rel[finite].mean()), float(rel[finite].max()), float(diff.max())
