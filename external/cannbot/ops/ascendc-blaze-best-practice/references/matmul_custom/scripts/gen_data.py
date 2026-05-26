#!/usr/bin/python3
# coding=utf-8
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# [MODIFY] 修改 dtype 时同步改 .to() 和 .view()
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


def gen_golden_data_simple(m, k, n, trans_a=False, trans_b=False):
    M, K, N = m, k, n
    A_logical = torch.from_numpy(np.random.uniform(-1.0, 1.0, (M, K)).astype(np.float32)).to(torch.bfloat16)
    A_phys = A_logical.t().contiguous() if trans_a else A_logical

    if trans_b:
        B_phys = torch.from_numpy(np.random.uniform(-1.0, 1.0, (N, K)).astype(np.float32)).to(torch.bfloat16)
        B_logical = B_phys.t().contiguous()
    else:
        B_phys = torch.from_numpy(np.random.uniform(-1.0, 1.0, (K, N)).astype(np.float32)).to(torch.bfloat16)
        B_logical = B_phys

    out = (A_logical.float() @ B_logical.float()).to(torch.bfloat16)
    current_dir = os.getcwd()
    write_artifacts(current_dir, A_phys, B_phys, out)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.normcase(os.path.abspath(script_dir)) != os.path.normcase(os.path.abspath(current_dir)):
        write_artifacts(script_dir, A_phys, B_phys, out)


if __name__ == "__main__":
    if len(sys.argv) not in (4, 6):
        print("Usage: python3 gen_data.py m k n [transA transB]")
        sys.exit(1)
    m, k, n = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    if len(sys.argv) == 6:
        trans_a = sys.argv[4].lower() == "true"
        trans_b = sys.argv[5].lower() == "true"
    else:
        trans_a = False
        trans_b = False
    gen_golden_data_simple(m, k, n, trans_a=trans_a, trans_b=trans_b)
