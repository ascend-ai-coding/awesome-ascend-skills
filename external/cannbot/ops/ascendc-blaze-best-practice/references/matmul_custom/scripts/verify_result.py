# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import sys

import numpy as np
import torch

ERROR_TOL = 1e-3
# The sample dumps bfloat16 tensors as raw 16-bit payloads, so verification
# reads them as uint16 first and then reinterprets the bits back to bfloat16.
DATA_TYPE = np.uint16

# If m*n is larger than this, avoid dumping full tensors; print summary stats instead.
FULL_TENSOR_PRINT_MAX_ELEMENTS = 1024
# Corner slice size for large-tensor summary (top-left block).
CORNER_ROWS = 4
CORNER_COLS = 4


def _print_large_tensor_summary(golden_tensor: torch.Tensor, npu_output_tensor: torch.Tensor, m: int, n: int) -> None:
    g = golden_tensor.float()
    p = npu_output_tensor.float()
    diff = p - g
    abs_err = diff.abs()
    denom = g.abs().clamp_min(1e-8)
    rel_err = abs_err / denom

    numel = m * n
    over_tol = (abs_err > ERROR_TOL).sum().item()

    print(f"\n[verify] shape=({m}, {n}), elements={numel} - summary (large matrix, full tensors omitted)")
    print(
        f"  abs_err: max={abs_err.max().item():.6e}, mean={abs_err.mean().item():.6e}, "
        f"rmse={(diff.pow(2).mean().sqrt()).item():.6e}"
    )
    print(f"  rel_err: max={rel_err.max().item():.6e}")
    print(f"  count(|abs_err| > {ERROR_TOL:g}): {over_tol} / {numel}")

    cr = min(CORNER_ROWS, m)
    cc = min(CORNER_COLS, n)
    if cr > 0 and cc > 0:
        print(f"  cpu golden (top-left {cr}x{cc}):\n{golden_tensor[:cr, :cc]}")
        print(f"  npu output (top-left {cr}x{cc}):\n{npu_output_tensor[:cr, :cc]}")


def verify_result(m, n):
    # The sample launcher and golden generator both write into the local
    # `output/` directory that sits next to the installed executable.
    output = np.fromfile("./output/output.bin", dtype=DATA_TYPE)
    golden = np.fromfile("./output/cpu_output.bin", dtype=DATA_TYPE)

    if output.size != golden.size:
        raise ValueError("npu output size != cpu output size")

    npu_output_tensor = torch.from_numpy(output).view(torch.bfloat16).reshape(m, n)
    golden_tensor = torch.from_numpy(golden).view(torch.bfloat16).reshape(m, n)

    numel = m * n
    if numel <= FULL_TENSOR_PRINT_MAX_ELEMENTS:
        print("\ncpu golden:\n", golden_tensor)
        print("npu output:\n", npu_output_tensor)
    else:
        _print_large_tensor_summary(golden_tensor, npu_output_tensor, m, n)

    return torch.allclose(
        golden_tensor, npu_output_tensor, rtol=ERROR_TOL, atol=ERROR_TOL, equal_nan=True
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 verify_result.py m n")
        sys.exit(1)

    m = int(sys.argv[1])
    n = int(sys.argv[2])
    try:
        res = verify_result(m, n)
        if not res:
            raise ValueError("[ERROR] NPU results differ from CPU.\n")
        print("[PASS] NPU results are consistent with CPU.\n")

    except Exception as e:
        print(e)
        sys.exit(1)
