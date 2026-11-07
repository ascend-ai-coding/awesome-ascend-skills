# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Multi-process NPU runner"""
import os
import configparser
from dataclasses import dataclass

import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp

import xops  # registers torch.ops.custom.* onto torch_npu

CUR = os.path.dirname(os.path.abspath(__file__))
INI = os.path.join(CUR, "single_server_config.ini")
DATA_PATH = os.path.join(CUR, "gen_data_single_server")

DTYPE_MAP = {0: torch.float16, 1: torch.bfloat16}
VIEW_DTYPE = {torch.float16: torch.int16, torch.bfloat16: torch.int16}


@dataclass
class WorkerArgs:
    rank_size: int
    master_addr: str
    master_port: int
    m: int
    k: int
    n: int
    has_bias: int
    is_gather_out: int
    dt: int


def read_bin(path, shape, dtype):
    with open(path, "rb") as f:
        buf = bytearray(f.read())
    return torch.frombuffer(buf, dtype=VIEW_DTYPE[dtype]).view(shape).view(dtype)


def write_bin(tensor, path, dtype):
    tensor.cpu().contiguous().view(VIEW_DTYPE[dtype]).numpy().tofile(path)


def worker(local_rank, args: WorkerArgs):
    global_rank = local_rank
    device_id = local_rank % 8
    torch_npu.npu.set_device(device_id)
    dtype = DTYPE_MAP[args.dt]

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    dist.init_process_group(backend="hccl", rank=global_rank,
                            init_method=f"tcp://{args.master_addr}:{args.master_port}",
                            world_size=args.rank_size)
    pg = dist.new_group(backend="hccl", ranks=list(range(args.rank_size)))
    hccl_group = pg._get_backend(torch.device("npu")).get_hccl_comm_name(global_rank)
    torch_npu.npu.synchronize(device_id)

    x1 = read_bin(os.path.join(DATA_PATH, f"x1_{global_rank}.bin"), (args.m, args.k), dtype).npu()
    x2 = read_bin(os.path.join(DATA_PATH, "x2.bin"), (args.k, args.n), dtype).npu()
    bias = None
    if args.has_bias:
        bias = read_bin(os.path.join(DATA_PATH, "bias.bin"), (args.n,), dtype).npu()

    out = torch.empty((args.m * args.rank_size, args.n), dtype=dtype, device=f"npu:{device_id}")
    gather_out = None
    if args.is_gather_out:
        gather_out = torch.empty((args.m * args.rank_size, args.k), dtype=dtype, device=f"npu:{device_id}")

    out, gather_out = torch_npu.npu_x_all_gather_matmul(
        x1, x2, bias, group=hccl_group,
        gatherIndex=0, commTurn=0, streamMode=1,
        output=out, gatherOut=gather_out)
    torch_npu.npu.synchronize(device_id)
    write_bin(out, os.path.join(CUR, f"output_{global_rank}.bin"), dtype)
    if args.is_gather_out:
        write_bin(gather_out, os.path.join(CUR, f"gather_out_{global_rank}.bin"), dtype)

    dist.destroy_process_group()


def main():
    cfg = configparser.ConfigParser()
    cfg.read(INI)
    rank_size = int(cfg["global"]["rankSize"])
    master_addr = cfg["global"].get("masterAddr", "127.0.0.1")
    master_port = int(cfg["global"].get("masterPort", 29500))
    m = int(cfg["matmul"]["m"])
    k = int(cfg["matmul"]["k"])
    n = int(cfg["matmul"]["n"])
    has_bias = int(cfg["matmul"]["hasBias"])
    is_gather_out = int(cfg["matmul"]["isGatherOut"])
    dt = int(cfg["global"]["dataType"])

    args = WorkerArgs(
        rank_size=rank_size,
        master_addr=master_addr,
        master_port=master_port,
        m=m, k=k, n=n,
        has_bias=has_bias,
        is_gather_out=is_gather_out,
        dt=dt,
    )
    mp.spawn(worker, args=(args,), nprocs=rank_size, join=True)


if __name__ == "__main__":
    main()
