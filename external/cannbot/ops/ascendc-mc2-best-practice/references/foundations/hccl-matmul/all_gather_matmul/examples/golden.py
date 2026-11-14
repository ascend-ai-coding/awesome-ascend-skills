# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU (torch) reference for XAllGatherMatmul: AllGather(x1) along dim0 then Matmul (+bias).

Ported from ops-transformer/mc2/x_all_gather_matmul/tests/assets/golden.py with the
transpose bug fixed (original used bare .transpose(); here .transpose(0, 1)).
Computes in fp32. Multi-rank faithful: takes a list of DISTINCT per-rank x1 shards.
"""
import torch


def x_all_gather_matmul_golden(x1_shards, x2, bias=None, is_trans_b=False, is_gather_out=True):
    x2 = x2.to(torch.float32)
    if is_trans_b:
        x2 = x2.transpose(0, 1)                       # fix: original used bare .transpose()
    gather_out = torch.cat(
        [s.to(torch.float32) for s in x1_shards], dim=0)   # [M_total, K]
    output = torch.matmul(gather_out, x2)                  # [M_total, N]
    if bias is not None:
        output = output + bias.to(torch.float32)
    return (output, gather_out) if is_gather_out else (output, None)
