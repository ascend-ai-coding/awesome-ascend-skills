# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU reference for the MXFP4/BF16 contract; no PyPTO imports."""

import torch


def decode(packed, scales):
    magnitude = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    mantissa = magnitude[codes & 7] * torch.where(codes < 8, 1.0, -1.0)
    exponent = scales.reshape(*packed.shape[:-1], 4).int() - 127
    return mantissa * torch.exp2(exponent.float()).repeat_interleave(32, -1)


def scores(q, k, w, qs, ks):
    qk = (decode(q, qs) @ decode(k, ks).squeeze(1).T).bfloat16().float()
    weight = w.bfloat16().float()
    result = torch.zeros((q.shape[0], k.shape[0]), dtype=torch.float32)
    for head in range(32):
        result = (
            (result + qk[:, head].clamp_min(0) * weight[:, head, None])
            .bfloat16()
            .float()
        )
    return result


def visible_keys(lk, lq, row, mask, ratio, residual):
    if row >= lq:
        return 0
    return (
        lk
        if mask == 0
        else min(lk, max(0, (ratio * lk + residual - lq + row + 1) // ratio))
    )


def check_topk(indices, values, expected, allowed, offset=0):
    present = indices != -1
    if values is not None:
        assert torch.isneginf(values[~present]).all()
    selected = indices[present].long() - offset
    assert selected.numel() == min(indices.numel(), int(allowed.sum()))
    assert selected.unique().numel() == selected.numel()
    assert ((selected >= 0) & (selected < expected.numel())).all()
    assert allowed[selected].all()
    if not selected.numel():
        return
    wanted = expected[selected]
    if values is not None:
        torch.testing.assert_close(values[present].float(), wanted, rtol=0, atol=0)
    boundary = expected[allowed].topk(selected.numel()).values[-1]
    if torch.isposinf(boundary):
        assert torch.isposinf(wanted).all()
    else:
        assert wanted.min() >= boundary
