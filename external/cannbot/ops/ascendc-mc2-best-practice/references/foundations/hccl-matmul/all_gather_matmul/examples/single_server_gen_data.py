# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Generate per-rank inputs + fp32 golden for XAllGatherMatmul, dump to .bin + meta.txt."""
import logging
import os
import configparser

import numpy as np
import torch

from golden import x_all_gather_matmul_golden

logging.basicConfig(level=logging.INFO, format='%(message)s')

CUR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CUR, "gen_data_single_server")
INI = os.path.join(CUR, "single_server_config.ini")

DTYPE_MAP = {0: torch.float16, 1: torch.bfloat16}
# raw-bytes view dtype (fp16/bf16 -> int16; fp32 -> int32)
VIEW_DTYPE = {
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
    torch.float32: torch.int32,
    torch.int32: torch.int32,
    torch.int64: torch.int64,
}


def write_to_bin(tensor, prefix):
    if tensor is None:
        return
    t = tensor.contiguous().cpu()
    t.view(VIEW_DTYPE[t.dtype]).numpy().tofile(os.path.join(DATA_PATH, f"{prefix}.bin"))


def gen_random(shape, dtype):
    return torch.randn(shape, dtype=dtype)


def main():
    cfg = configparser.ConfigParser()
    cfg.read(INI)
    rank_size = int(cfg["global"]["rankSize"])
    dtype = DTYPE_MAP[int(cfg["global"]["dataType"])]
    m = int(cfg["matmul"]["m"])
    k = int(cfg["matmul"]["k"])
    n = int(cfg["matmul"]["n"])
    has_bias = int(cfg["matmul"]["hasBias"])
    is_gather_out = int(cfg["matmul"]["isGatherOut"])
    seed = int(cfg["run"].get("seed", 42))

    os.makedirs(DATA_PATH, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # distinct per-rank x1 shards [m, k] (faithful multi-rank)
    x1_shards = [gen_random((m, k), dtype) for _ in range(rank_size)]
    x2 = gen_random((k, n), dtype)             # [K, N], is_trans_b=false
    # Constraint (ops-transformer README 约束说明): non-empty bias must be 1-D and the op does NOT
    # support non-zero bias values yet ("暂不支持bias输入为非0的场景"). Use a zero bias to exercise
    # the bias-present tiling/kernel path (tiling key 7) while satisfying the constraint; golden
    # adds zero bias (no-op) and matches the op's matmul-only result for zero bias.
    bias = torch.zeros((n,), dtype=dtype) if has_bias else None

    out, gout = x_all_gather_matmul_golden(
        x1_shards, x2, bias=bias, is_trans_b=False, is_gather_out=bool(is_gather_out))

    for r, s in enumerate(x1_shards):
        write_to_bin(s, f"x1_{r}")
    write_to_bin(x2, "x2")
    if has_bias:
        write_to_bin(bias, "bias")
    write_to_bin(out, "golden_output")         # fp32 -> int32 bytes
    if is_gather_out:
        write_to_bin(gout, "golden_gather_out")

    with open(os.path.join(DATA_PATH, "meta.txt"), "w") as f:
        f.write(f"{rank_size}\n{m}\n{k}\n{n}\n{has_bias}\n{is_gather_out}\n"
                f"{int(cfg['global']['dataType'])}\n")
    logging.info(f"gen_data done: rank_size={rank_size} m={m} k={k} n={n} dtype={dtype}")


if __name__ == "__main__":
    main()
