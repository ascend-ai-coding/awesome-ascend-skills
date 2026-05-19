#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import os
import sys

import numpy as np
import torch


def write_artifacts(base_dir, A, B, out):
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    A.view(torch.uint16).numpy().tofile(os.path.join(input_dir, "input_a.bin"))
    B.view(torch.uint16).numpy().tofile(os.path.join(input_dir, "input_b.bin"))
    out.view(torch.uint16).numpy().tofile(os.path.join(output_dir, "cpu_output.bin"))


def gen_golden_data_simple(m, k, n, trans_b=True):
    M, K, N = m, k, n

    A = torch.from_numpy(np.random.uniform(-1.0, 1.0, (M, K)).astype(np.float32)).to(torch.bfloat16)
    if trans_b:
        # Host file stores B as (N, K); device interprets it as ColumnMajor (K, N).
        B_phys = torch.from_numpy(np.random.uniform(-1.0, 1.0, (N, K)).astype(np.float32)).to(torch.bfloat16)
        B_logical = B_phys.t().contiguous()
    else:
        # Host file stores B as (K, N) RowMajor.
        B_phys = torch.from_numpy(np.random.uniform(-1.0, 1.0, (K, N)).astype(np.float32)).to(torch.bfloat16)
        B_logical = B_phys

    out = (A.float() @ B_logical.float()).to(torch.bfloat16)

    current_dir = os.getcwd()
    write_artifacts(current_dir, A, B_phys, out)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.normcase(os.path.abspath(script_dir)) != os.path.normcase(os.path.abspath(current_dir)):
        write_artifacts(script_dir, A, B_phys, out)


if __name__ == "__main__":
    if len(sys.argv) not in (4, 6):
        print("Usage: python3 gen_data_a16w16.py m k n [transA transB]")
        sys.exit(1)

    m = int(sys.argv[1])
    k = int(sys.argv[2])
    n = int(sys.argv[3])
    if len(sys.argv) == 6:
        trans_b = sys.argv[5].lower() == "true"
    else:
        trans_b = True

    gen_golden_data_simple(m, k, n, trans_b=trans_b)
