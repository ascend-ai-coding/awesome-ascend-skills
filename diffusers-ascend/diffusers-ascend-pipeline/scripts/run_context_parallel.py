#!/usr/bin/env python3
"""Run Diffusers distributed inference with torch.distributed.

Launch with torchrun, for example:
  torchrun --nproc_per_node=2 scripts/run_context_parallel.py --model ./fake_flux_dev
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def get_dtype(dtype_str: str):
    import torch

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return mapping[dtype_str]


def get_default_attention_backend(device_type: str) -> str:
    if device_type == "npu":
        return "_native_npu"
    if device_type == "cuda":
        return "_native_cudnn"
    return "native"


def setup_distributed(backend: str, device_type: str):
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if device_type == "npu":
        torch.npu.set_device(local_rank)
        device = f"npu:{local_rank}"
    elif device_type == "cuda":
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    return rank, world_size, local_rank, device


def save_rank0_output(result, args, elapsed: float, world_size: int, rank_latencies):
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(result, "images") and result.images:
        result.images[0].save(str(output_path))
    else:
        raise RuntimeError("No image output from pipeline")

    meta = {
        "model": args.model,
        "prompt": args.prompt,
        "parallel_mode": args.parallel_mode,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "dtype": args.dtype,
        "backend": args.backend,
        "world_size": world_size,
        "elapsed_s": round(elapsed, 3),
        "rank_latencies_s": rank_latencies,
    }
    with open(output_path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Context parallel inference for Diffusers pipelines"
    )
    parser.add_argument("--model", type=str, required=True, help="Local model path")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument(
        "--parallel-mode",
        type=str,
        default="context",
        choices=["context", "data"],
        help="context=Context Parallel API, data=multi-process data parallel",
    )
    parser.add_argument(
        "--device-type",
        type=str,
        default="npu",
        choices=["npu", "cuda", "cpu"],
        help="Distributed device type",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="hccl",
        help="torch.distributed backend (hccl/nccl/gloo)",
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default=None,
        help="Diffusers attention backend (default: _native_npu on NPU, _native_cudnn on CUDA)",
    )
    parser.add_argument(
        "--ring-degree",
        type=int,
        default=None,
        help="Ring Attention degree for context parallel",
    )
    parser.add_argument(
        "--ulysses-degree",
        type=int,
        default=None,
        help="Ulysses Attention degree for context parallel",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="cp_output.png")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: model path does not exist: {args.model}", file=sys.stderr)
        sys.exit(1)

    import torch
    import torch.distributed as dist
    from diffusers import ContextParallelConfig, DiffusionPipeline

    rank = -1
    try:
        rank, world_size, _, device = setup_distributed(args.backend, args.device_type)
        dtype = get_dtype(args.dtype)

        if rank == 0:
            print(f"Loading pipeline from {args.model} on {world_size} ranks")
        pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype).to(
            device
        )

        if args.parallel_mode == "context":
            if not hasattr(pipe, "transformer"):
                raise RuntimeError(
                    "Pipeline has no transformer; context parallel not supported"
                )
            if not hasattr(pipe.transformer, "enable_parallelism"):
                raise RuntimeError(
                    "Transformer has no enable_parallelism; check diffusers version"
                )

            attention_backend = args.attention_backend or get_default_attention_backend(
                args.device_type
            )
            if hasattr(pipe.transformer, "set_attention_backend"):
                if rank == 0:
                    print(f"Setting attention backend: {attention_backend}")
                pipe.transformer.set_attention_backend(attention_backend)

            ring_degree = args.ring_degree
            ulysses_degree = args.ulysses_degree
            if ring_degree is None and ulysses_degree is None:
                ulysses_degree = world_size

            cp_config = ContextParallelConfig(
                ring_degree=ring_degree,
                ulysses_degree=ulysses_degree,
            )
            pipe.transformer.enable_parallelism(config=cp_config)
        elif rank == 0:
            print("Running in data parallel mode (one process per device)")

        generator = torch.Generator(device).manual_seed(args.seed + rank)
        infer_kwargs = {
            "prompt": args.prompt,
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "generator": generator,
        }
        if args.height is not None:
            infer_kwargs["height"] = args.height
        if args.width is not None:
            infer_kwargs["width"] = args.width

        if device.startswith("npu"):
            torch.npu.synchronize()
        t0 = time.time()
        result = pipe(**infer_kwargs)
        if device.startswith("npu"):
            torch.npu.synchronize()
        elapsed = time.time() - t0

        rank_elapsed = round(elapsed, 3)
        rank_latencies = None
        if rank == 0:
            rank_latencies = [None] * world_size
        dist.gather_object(rank_elapsed, rank_latencies, dst=0)

        if rank == 0:
            save_rank0_output(result, args, elapsed, world_size, rank_latencies)
            print(
                f"Distributed inference succeeded (mode={args.parallel_mode}), "
                f"elapsed={elapsed:.2f}s"
            )

        dist.barrier()
        dist.destroy_process_group()
    except Exception as e:
        print(f"[rank={rank}] Error: {e}", file=sys.stderr)
        if dist.is_initialized():
            dist.destroy_process_group()
        sys.exit(1)


if __name__ == "__main__":
    main()
