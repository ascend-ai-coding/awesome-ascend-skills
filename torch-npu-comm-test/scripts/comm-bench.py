#!/usr/bin/env python3
"""Benchmark collective communication operators via torch.distributed on Ascend NPU.

Launch with torchrun:
    torchrun --nproc_per_node=8 comm-bench.py \
        --op all_reduce --shape 4096,12288 --dtype fp16 --iters 100 --warmup 10
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "int32": torch.int32,
}

REDUCE_OP_MAP = {
    "sum": dist.ReduceOp.SUM,
    "prod": dist.ReduceOp.PRODUCT,
    "max": dist.ReduceOp.MAX,
    "min": dist.ReduceOp.MIN,
}

SUPPORTED_OPS = [
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "broadcast",
    "all_to_all",
    "reduce",
    "send_recv",
    "barrier",
]


def parse_shape(shape_str: str) -> List[int]:
    return [int(x.strip()) for x in shape_str.split(",")]


def parse_group_ranks(ranks_str: str) -> List[int]:
    return [int(x.strip()) for x in ranks_str.split(",")]


def compute_data_size_bytes(shape: List[int], dtype: torch.dtype) -> int:
    numel = 1
    for d in shape:
        numel *= d
    return numel * torch.tensor([], dtype=dtype).element_size()


def compute_bandwidth(
    op: str, data_size_bytes: int, time_s: float, world_size: int
) -> Tuple[float, float]:
    """Returns (algbw_gbps, busbw_gbps)."""
    if time_s <= 0:
        return 0.0, 0.0

    n = world_size
    gb = 1e9

    if op == "all_reduce":
        algbw = data_size_bytes / time_s / gb
        busbw = data_size_bytes * 2 * (n - 1) / n / time_s / gb
    elif op == "all_gather":
        algbw = data_size_bytes * n / time_s / gb
        busbw = data_size_bytes * (n - 1) / time_s / gb
    elif op == "reduce_scatter":
        algbw = data_size_bytes / time_s / gb
        busbw = data_size_bytes * (n - 1) / n / time_s / gb
    elif op == "broadcast":
        algbw = data_size_bytes / time_s / gb
        busbw = algbw
    elif op == "reduce":
        algbw = data_size_bytes / time_s / gb
        busbw = data_size_bytes * (n - 1) / n / time_s / gb
    elif op == "all_to_all":
        algbw = data_size_bytes / time_s / gb
        busbw = data_size_bytes * (n - 1) / n / time_s / gb
    elif op == "send_recv":
        algbw = data_size_bytes / time_s / gb
        busbw = algbw
    else:
        algbw = data_size_bytes / time_s / gb
        busbw = algbw

    return algbw, busbw


def percentile(sorted_data: List[float], pct: float) -> float:
    idx = (len(sorted_data) - 1) * pct / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(sorted_data) - 1)
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


def create_op_tensors(
    op: str,
    shape: List[int],
    dtype: torch.dtype,
    device: torch.device,
    world_size: int,
) -> Dict[str, torch.Tensor]:
    """Pre-allocate all tensors needed for the operation."""
    tensors = {}

    if op == "all_reduce":
        tensors["tensor"] = torch.randn(shape, dtype=dtype, device=device)

    elif op == "all_gather":
        tensors["input"] = torch.randn(shape, dtype=dtype, device=device)
        out_shape = [shape[0] * world_size] + shape[1:]
        tensors["output"] = torch.empty(out_shape, dtype=dtype, device=device)

    elif op == "reduce_scatter":
        inp_shape = [shape[0] * world_size] + shape[1:]
        tensors["input"] = torch.randn(inp_shape, dtype=dtype, device=device)
        tensors["output"] = torch.empty(shape, dtype=dtype, device=device)

    elif op in ("broadcast", "reduce"):
        tensors["tensor"] = torch.randn(shape, dtype=dtype, device=device)

    elif op == "all_to_all":
        tensors["input"] = torch.randn(shape, dtype=dtype, device=device)
        tensors["output"] = torch.empty(shape, dtype=dtype, device=device)

    elif op == "send_recv":
        tensors["tensor"] = torch.randn(shape, dtype=dtype, device=device)

    return tensors


def run_op(
    op: str,
    tensors: Dict[str, torch.Tensor],
    rank: int,
    world_size: int,
    group: Optional[dist.ProcessGroup],
    reduce_op: dist.ReduceOp,
    src_rank: int,
    async_op: bool,
) -> Optional[dist.Work]:
    """Execute one communication operation. Returns Work handle if async."""
    work = None

    if op == "all_reduce":
        work = dist.all_reduce(tensors["tensor"], op=reduce_op, group=group, async_op=async_op)

    elif op == "all_gather":
        work = dist.all_gather_into_tensor(
            tensors["output"], tensors["input"], group=group, async_op=async_op
        )

    elif op == "reduce_scatter":
        work = dist.reduce_scatter_tensor(
            tensors["output"], tensors["input"], op=reduce_op, group=group, async_op=async_op
        )

    elif op == "broadcast":
        work = dist.broadcast(tensors["tensor"], src=src_rank, group=group, async_op=async_op)

    elif op == "all_to_all":
        work = dist.all_to_all_single(
            tensors["output"], tensors["input"], group=group, async_op=async_op
        )

    elif op == "reduce":
        work = dist.reduce(
            tensors["tensor"], dst=src_rank, op=reduce_op, group=group, async_op=async_op
        )

    elif op == "send_recv":
        if world_size < 2:
            raise ValueError("send_recv requires at least 2 ranks")
        peer = (rank + 1) % world_size
        if rank % 2 == 0:
            dist.send(tensors["tensor"], dst=peer, group=group)
            dist.recv(tensors["tensor"], src=peer, group=group)
        else:
            recv_buf = torch.empty_like(tensors["tensor"])
            dist.recv(recv_buf, src=peer, group=group)
            dist.send(tensors["tensor"], dst=peer, group=group)

    elif op == "barrier":
        work = dist.barrier(group=group, async_op=async_op)

    return work


def validate_result(
    op: str,
    shape: List[int],
    dtype: torch.dtype,
    device: torch.device,
    world_size: int,
    rank: int,
    group: Optional[dist.ProcessGroup],
    reduce_op: dist.ReduceOp,
    src_rank: int,
) -> bool:
    """Run a separate correctness check with known values."""
    torch.npu.synchronize()

    if op == "all_reduce":
        t = torch.ones(shape, dtype=dtype, device=device) * (rank + 1)
        dist.all_reduce(t, op=reduce_op, group=group)
        torch.npu.synchronize()
        expected = sum(range(1, world_size + 1))
        return torch.allclose(
            t.float(), torch.full(shape, expected, dtype=torch.float32, device=device), atol=1.0
        )

    elif op == "all_gather":
        inp = torch.ones(shape, dtype=dtype, device=device) * (rank + 1)
        out_shape = [shape[0] * world_size] + shape[1:]
        out = torch.empty(out_shape, dtype=dtype, device=device)
        dist.all_gather_into_tensor(out, inp, group=group)
        torch.npu.synchronize()
        for i in range(world_size):
            chunk = out[i * shape[0] : (i + 1) * shape[0]]
            if not torch.allclose(chunk.float(), torch.full(shape, float(i + 1), device=device), atol=1.0):
                return False
        return True

    elif op == "broadcast":
        t = torch.ones(shape, dtype=dtype, device=device) * (42.0 if rank == src_rank else 0.0)
        dist.broadcast(t, src=src_rank, group=group)
        torch.npu.synchronize()
        return torch.allclose(t.float(), torch.full(shape, 42.0, dtype=torch.float32, device=device), atol=1.0)

    return True


def run_benchmark(args: argparse.Namespace) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")

    dist.init_process_group(backend="hccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    shape = parse_shape(args.shape)
    dtype = DTYPE_MAP[args.dtype]
    reduce_op = REDUCE_OP_MAP[args.reduce_op]

    group = None
    group_world_size = world_size
    if args.group_ranks:
        group_rank_list = parse_group_ranks(args.group_ranks)
        group = dist.new_group(ranks=group_rank_list)
        if rank not in group_rank_list:
            dist.destroy_process_group()
            return
        group_world_size = len(group_rank_list)

    data_size_bytes = compute_data_size_bytes(shape, dtype)
    data_size_mb = data_size_bytes / (1024 * 1024)

    if args.op == "barrier":
        data_size_bytes = 0
        data_size_mb = 0

    tensors = create_op_tensors(args.op, shape, dtype, device, group_world_size)

    torch.npu.synchronize()
    dist.barrier(group=group)

    for _ in range(args.warmup):
        work = run_op(
            args.op, tensors, rank, group_world_size, group,
            reduce_op, args.src_rank, args.async_op,
        )
        if args.async_op and work is not None:
            work.wait()
        torch.npu.synchronize()

    dist.barrier(group=group)

    latencies_us: List[float] = []
    for _ in range(args.iters):
        torch.npu.synchronize()
        start = time.perf_counter()

        work = run_op(
            args.op, tensors, rank, group_world_size, group,
            reduce_op, args.src_rank, args.async_op,
        )
        if args.async_op and work is not None:
            work.wait()

        torch.npu.synchronize()
        end = time.perf_counter()
        latencies_us.append((end - start) * 1e6)

    if args.check and args.op != "barrier":
        check_pass = validate_result(
            args.op, shape, dtype, device, group_world_size,
            rank, group, reduce_op, args.src_rank,
        )
    else:
        check_pass = None

    if rank == 0 or (args.group_ranks and rank == parse_group_ranks(args.group_ranks)[0]):
        sorted_latencies = sorted(latencies_us)
        avg_us = sum(latencies_us) / len(latencies_us)
        min_us = sorted_latencies[0]
        max_us = sorted_latencies[-1]
        p50_us = percentile(sorted_latencies, 50)
        p95_us = percentile(sorted_latencies, 95)
        p99_us = percentile(sorted_latencies, 99)

        avg_time_s = avg_us / 1e6
        algbw, busbw = compute_bandwidth(args.op, data_size_bytes, avg_time_s, group_world_size)

        if args.output == "json":
            result = {
                "op": args.op,
                "shape": shape,
                "dtype": args.dtype,
                "ranks": group_world_size,
                "data_size_mb": round(data_size_mb, 2),
                "avg_us": round(avg_us, 1),
                "min_us": round(min_us, 1),
                "max_us": round(max_us, 1),
                "p50_us": round(p50_us, 1),
                "p95_us": round(p95_us, 1),
                "p99_us": round(p99_us, 1),
                "algbw_gbps": round(algbw, 2),
                "busbw_gbps": round(busbw, 2),
                "iters": args.iters,
                "warmup": args.warmup,
            }
            if check_pass is not None:
                result["check"] = "PASS" if check_pass else "FAIL"
            print(json.dumps(result, indent=2))
        else:
            sep = "=" * 76
            print(f"\n{sep}")
            header = f"Op: {args.op} | Shape: {shape} | Dtype: {args.dtype} | Ranks: {group_world_size}"
            if check_pass is not None:
                header += f" | Check: {'PASS' if check_pass else 'FAIL'}"
            print(header)
            print(sep)
            fmt = "{:<14s} {:<10s} {:<10s} {:<10s} {:<10s} {:<10s} {:<10s} {:<12s} {:<12s}"
            print(fmt.format(
                "data_size(MB)", "avg(us)", "min(us)", "max(us)",
                "p50(us)", "p95(us)", "p99(us)", "algbw(GB/s)", "busbw(GB/s)",
            ))
            val_fmt = "{:<14.2f} {:<10.1f} {:<10.1f} {:<10.1f} {:<10.1f} {:<10.1f} {:<10.1f} {:<12.2f} {:<12.2f}"
            print(val_fmt.format(
                data_size_mb, avg_us, min_us, max_us,
                p50_us, p95_us, p99_us, algbw, busbw,
            ))
            print(sep)

    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark collective communication operators on Ascend NPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--op", type=str, default="all_reduce", choices=SUPPORTED_OPS,
        help="Communication operator to test",
    )
    parser.add_argument(
        "--shape", type=str, default="1024,1024",
        help="Tensor shape, comma-separated (e.g., '4096,12288')",
    )
    parser.add_argument(
        "--dtype", type=str, default="fp16", choices=list(DTYPE_MAP.keys()),
        help="Data type",
    )
    parser.add_argument("--iters", type=int, default=50, help="Number of measured iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument(
        "--reduce-op", type=str, default="sum", choices=list(REDUCE_OP_MAP.keys()),
        help="Reduce operation type (for all_reduce, reduce, reduce_scatter)",
    )
    parser.add_argument(
        "--src-rank", type=int, default=0,
        help="Source rank for broadcast / destination rank for reduce",
    )
    parser.add_argument(
        "--group-ranks", type=str, default=None,
        help="Comma-separated rank list for subgroup testing (e.g., '0,1,2,3')",
    )
    parser.add_argument(
        "--output", type=str, default="table", choices=["table", "json"],
        help="Output format",
    )
    parser.add_argument("--async-op", action="store_true", help="Use async operations")
    parser.add_argument("--check", action="store_true", help="Enable result correctness check")

    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
