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


def to_nz_format(data, c0):
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


def gen_golden_data(m, k, n, dtype=torch.bfloat16, trans_a=False, trans_b=False, a_layout="nd", b_layout="nd"):
    M, K, N = m, k, n
    C0 = 32 // torch.tensor([], dtype=dtype).element_size()  # 动态计算 C0

    # Step 1: 固定生成 A[M,K], B[K,N]
    if dtype.is_floating_point:
        A = torch.randn(M, K, dtype=dtype)
        B = torch.randn(K, N, dtype=dtype)
    else:
        A = torch.randint(-128, 128, (M, K), dtype=dtype)
        B = torch.randint(-128, 128, (K, N), dtype=dtype)

    # Step 2: golden = A @ B（固定，不感知转置）
    out = (A.float() @ B.float()).to(torch.bfloat16)

    # Step 3: 按 kernel 需求转换 A（先 transpose，再 to_nz_format）
    if trans_a:
        A_for_kernel = A.t().contiguous()
    else:
        A_for_kernel = A
    if a_layout == "nz":
        A_phys = to_nz_format(A_for_kernel, C0)
    else:
        A_phys = A_for_kernel

    # Step 3: 按 kernel 需求转换 B（先 transpose，再 to_nz_format）
    if trans_b:
        B_for_kernel = B.t().contiguous()
    else:
        B_for_kernel = B
    if b_layout == "nz":
        B_phys = to_nz_format(B_for_kernel, C0)
    else:
        B_phys = B_for_kernel

    current_dir = os.getcwd()
    write_artifacts(current_dir, A_phys, B_phys, out)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.normcase(os.path.abspath(script_dir)) != os.path.normcase(
        os.path.abspath(current_dir)
    ):
        write_artifacts(script_dir, A_phys, B_phys, out)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 gen_data.py m k n [dtype] [transA transB] [a_layout b_layout]")
        print("  dtype: bf16 (default), fp16, fp32, int8, int32")
        print("  a_layout: nd (default) or nz")
        print("  b_layout: nd (default) or nz")
        sys.exit(1)
    
    m, k, n = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    
    # dtype 映射
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "int8": torch.int8,
        "int32": torch.int32,
    }
    dtype = torch.bfloat16  # 默认值
    dtype_idx = 4  # dtype 参数的位置
    
    trans_a = False
    trans_b = False
    a_layout = "nd"
    b_layout = "nd"
    
    # 解析可选参数
    if len(sys.argv) >= 5:
        dtype_str = sys.argv[4].lower()
        if dtype_str in dtype_map:
            dtype = dtype_map[dtype_str]
            dtype_idx = 5
        else:
            # 如果没有 dtype 参数，则从第 5 个参数开始解析
            dtype_idx = 4
    
    if len(sys.argv) >= dtype_idx + 2:
        trans_a = sys.argv[dtype_idx].lower() == "true"
        trans_b = sys.argv[dtype_idx + 1].lower() == "true"
    
    if len(sys.argv) >= dtype_idx + 3:
        a_layout = sys.argv[dtype_idx + 2].lower()
    
    if len(sys.argv) >= dtype_idx + 4:
        b_layout = sys.argv[dtype_idx + 3].lower()
    
    gen_golden_data(m, k, n, dtype=dtype, trans_a=trans_a, trans_b=trans_b,
                    a_layout=a_layout, b_layout=b_layout)
