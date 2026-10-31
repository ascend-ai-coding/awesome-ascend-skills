# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compare NPU output .bin vs fp32 golden .bin (np.isclose + error_ratio)."""
import logging
import os

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(message)s')

CUR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CUR, "gen_data_single_server")

VIEW_DTYPE = {
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
    torch.float32: torch.int32,
}


def read_tensor(path, dtype):
    """Read a .bin written by write_to_bin (raw dtype bytes) into a float32 numpy array."""
    with open(path, "rb") as f:
        buf = bytearray(f.read())
    return torch.frombuffer(buf, dtype=VIEW_DTYPE[dtype]).view(dtype).float().numpy()


def verify(out_arr, gold_arr, rtol=5e-3, atol=5e-3, err_ratio_thresh=1e-2):
    out_arr = out_arr.astype(np.float32).reshape(-1)
    gold_arr = gold_arr.astype(np.float32).reshape(-1)
    is_close = np.isclose(out_arr, gold_arr, rtol=rtol, atol=atol, equal_nan=True)
    not_close = int((~is_close).sum())
    err_ratio = not_close / gold_arr.size
    return (err_ratio < err_ratio_thresh), err_ratio, not_close


def main():
    with open(os.path.join(DATA_PATH, "meta.txt")) as f:
        rank_size, m, k, n, has_bias, is_gather_out, dt = (int(x) for x in f.read().split())
    out_dtype = torch.float16 if dt == 0 else torch.bfloat16
    # bf16 has lower precision than fp16 (~2^-8 vs ~2^-10); loosen tolerance accordingly.
    tol = 1e-2 if out_dtype == torch.bfloat16 else 5e-3

    ok = True
    gold_out = read_tensor(os.path.join(DATA_PATH, "golden_output.bin"), torch.float32)
    act_out = read_tensor(os.path.join(CUR, "output_0.bin"), out_dtype)
    passed, er, nc = verify(act_out, gold_out, rtol=tol, atol=tol)
    logging.info(f"[output] err_ratio={er:.6f} not_close={nc} (tol={tol}) -> {'PASS' if passed else 'FAIL'}")
    ok &= passed

    if is_gather_out:
        gold_g = read_tensor(os.path.join(DATA_PATH, "golden_gather_out.bin"), torch.float32)
        act_g = read_tensor(os.path.join(CUR, "gather_out_0.bin"), out_dtype)
        passed, er, nc = verify(act_g, gold_g, rtol=tol, atol=tol)
        logging.info(f"[gather_out] err_ratio={er:.6f} not_close={nc} (tol={tol}) -> {'PASS' if passed else 'FAIL'}")
        ok &= passed

    logging.info("RESULT: CHECK PASSED" if ok else "RESULT: CHECK FAILED")
    if not ok:
        raise RuntimeError("Verification failed")


if __name__ == "__main__":
    main()
