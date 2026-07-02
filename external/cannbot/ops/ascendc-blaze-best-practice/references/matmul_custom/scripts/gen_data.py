#!/usr/bin/python3
# coding=utf-8
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# [MODIFY N3] 修改 dtype 时同步改 .to() 和 .view()
# [MODIFY] 支持 A/B 的 ND/NZ 输入格式
# ----------------------------------------------------------------------------------------------------------

import os
import sys
import numpy as np
import torch


def to_nz_format(data, c0=16):
    """(dim0, dim1) ND → (dim1/C0, dim0/16, 16, C0) NZ 物理排列"""
    dim0, dim1 = data.shape
    dim0_pad = ((dim0 + 15) // 16) * 16
    dim1_pad = ((dim1 + c0 - 1) // c0) * c0
    padded = torch.zeros((dim0_pad, dim1_pad), dtype=data.dtype)
    padded[:dim0, :dim1] = data
    b_4d = padded.reshape(dim0_pad // 16, 16, dim1_pad // c0, c0)
    return b_4d.permute(2, 0, 1, 3).contiguous()


def write_artifacts(base_dir, A, B, out):
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    A.view(torch.uint16).numpy().tofile(os.path.join(input_dir, "input_a.bin"))
    B.view(torch.uint16).numpy().tofile(os.path.join(input_dir, "input_b.bin"))
    out.view(torch.uint16).numpy().tofile(os.path.join(output_dir, "cpu_output.bin"))


def gen_golden_data(m, k, n, trans_a=False, trans_b=False, a_layout="nd", b_layout="nd"):
    M, K, N = m, k, n
    C0 = 16  # [MODIFY N2] fp16/bf16: C0=16; fp8/int8: C0=32

    A_logical = torch.from_numpy(
        np.random.uniform(-1.0, 1.0, (M, K)).astype(np.float32)
    ).to(torch.bfloat16)

    if trans_a:
        A_raw = A_logical.t().contiguous()
    else:
        A_raw = A_logical

    if trans_b:
        B_raw = torch.from_numpy(
            np.random.uniform(-1.0, 1.0, (N, K)).astype(np.float32)
        ).to(torch.bfloat16)
        B_logical = B_raw.t().contiguous()
    else:
        B_raw = torch.from_numpy(
            np.random.uniform(-1.0, 1.0, (K, N)).astype(np.float32)
        ).to(torch.bfloat16)
        B_logical = B_raw

    out = (A_logical.float() @ B_logical.float()).to(torch.bfloat16)

    # NZ 转换统一应用于原始物理数据，不区分 transA/transB
    A_phys = to_nz_format(A_raw, C0) if a_layout == "nz" else A_raw
    B_phys = to_nz_format(B_raw, C0) if b_layout == "nz" else B_raw

    current_dir = os.getcwd()
    write_artifacts(current_dir, A_phys, B_phys, out)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.normcase(os.path.abspath(script_dir)) != os.path.normcase(
        os.path.abspath(current_dir)
    ):
        write_artifacts(script_dir, A_phys, B_phys, out)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 gen_data.py m k n [transA transB] [a_layout b_layout]")
        print("  a_layout: nd (default) or nz")
        print("  b_layout: nd (default) or nz")
        sys.exit(1)
    m, k, n = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    trans_a = False
    trans_b = False
    a_layout = "nd"
    b_layout = "nd"
    if len(sys.argv) >= 6:
        trans_a = sys.argv[4].lower() == "true"
        trans_b = sys.argv[5].lower() == "true"
    if len(sys.argv) >= 7:
        a_layout = sys.argv[6].lower()
    if len(sys.argv) >= 8:
        b_layout = sys.argv[7].lower()
    gen_golden_data(m, k, n, trans_a=trans_a, trans_b=trans_b,
                    a_layout=a_layout, b_layout=b_layout)
