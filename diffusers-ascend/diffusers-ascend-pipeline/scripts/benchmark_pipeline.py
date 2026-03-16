#!/usr/bin/env python3
"""
Benchmark Diffusers Pipeline inference performance on Ascend NPU.

Measures: latency (avg/min/max/P50/P95), throughput, memory usage.
Outputs results in JSON format for reporting.
"""

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path


def run_benchmark(args):
    """Run benchmark with warmup and multiple iterations."""
    import torch

    if args.device.startswith("npu"):
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            print("Error: torch_npu not installed.", file=sys.stderr)
            sys.exit(1)
        if not torch.npu.is_available():
            print("Error: NPU not available.", file=sys.stderr)
            sys.exit(1)

    import diffusers

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    if args.pipeline_class:
        pipe_cls = getattr(diffusers, args.pipeline_class, None)
        if pipe_cls is None:
            print(
                f"Error: Pipeline class '{args.pipeline_class}' not found",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        pipe_cls = diffusers.DiffusionPipeline

    # Load pipeline
    print(f"Loading pipeline from: {args.model}")
    t_load_start = time.time()
    pipe = pipe_cls.from_pretrained(args.model, torch_dtype=torch_dtype)
    t_load = time.time() - t_load_start
    print(f"Pipeline loaded in {t_load:.2f}s ({type(pipe).__name__})")

    # Move to device
    if args.cpu_offload:
        pipe.enable_sequential_cpu_offload(device=args.device)
        gen_device = "cpu"
    else:
        pipe = pipe.to(args.device)
        gen_device = args.device

    if hasattr(pipe, "enable_attention_slicing") and args.attention_slicing:
        pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing") and args.vae_slicing:
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling") and args.vae_tiling:
        pipe.enable_vae_tiling()

    # Reset memory stats
    if args.device.startswith("npu"):
        torch.npu.reset_peak_memory_stats()

    # Build inference kwargs
    infer_kwargs = {
        "prompt": args.prompt,
        "num_inference_steps": args.steps,
    }
    if args.guidance_scale is not None:
        infer_kwargs["guidance_scale"] = args.guidance_scale
    if args.height:
        infer_kwargs["height"] = args.height
    if args.width:
        infer_kwargs["width"] = args.width

    total_runs = args.warmup_runs + args.num_runs
    latencies = []

    print(f"\n{'=' * 50}")
    print(f"Benchmark Configuration:")
    print(f"  Device:     {args.device}")
    print(f"  Dtype:      {args.dtype}")
    print(f"  Steps:      {args.steps}")
    print(f"  Warmup:     {args.warmup_runs}")
    print(f"  Runs:       {args.num_runs}")
    prompt_display = args.prompt[:60] + "..." if len(args.prompt) > 60 else args.prompt
    print(f"  Prompt:     {prompt_display}")
    print(f"{'=' * 50}\n")

    for i in range(total_runs):
        is_warmup = i < args.warmup_runs
        label = (
            f"Warmup {i + 1}/{args.warmup_runs}"
            if is_warmup
            else f"Run {i - args.warmup_runs + 1}/{args.num_runs}"
        )

        generator = torch.Generator(gen_device).manual_seed(args.seed)
        infer_kwargs["generator"] = generator

        if args.device.startswith("npu"):
            torch.npu.synchronize()

        t_start = time.time()
        _ = pipe(**infer_kwargs)
        if args.device.startswith("npu"):
            torch.npu.synchronize()
        t_elapsed = time.time() - t_start

        latencies.append(t_elapsed)
        tag = "(warmup)" if is_warmup else ""
        print(f"  [{label}] {t_elapsed:.3f}s {tag}")

    # Compute statistics
    warmup_latencies = latencies[: args.warmup_runs]
    measured_latencies = latencies[args.warmup_runs :]

    if measured_latencies:
        avg_latency = statistics.mean(measured_latencies)
        min_latency = min(measured_latencies)
        max_latency = max(measured_latencies)
        sorted_lat = sorted(measured_latencies)
        p50_idx = max(0, int(len(sorted_lat) * 0.5) - 1)
        p95_idx = min(len(sorted_lat) - 1, max(0, int(len(sorted_lat) * 0.95) - 1))
        p50_latency = sorted_lat[p50_idx]
        p95_latency = sorted_lat[p95_idx]
        throughput = 1.0 / avg_latency if avg_latency > 0 else 0
        std_latency = (
            statistics.stdev(measured_latencies)
            if len(measured_latencies) >= 2
            else 0.0
        )
    else:
        avg_latency = min_latency = max_latency = 0
        p50_latency = p95_latency = std_latency = 0
        throughput = 0

    first_run_latency = latencies[0] if latencies else 0

    npu_peak_memory_gb = 0
    npu_current_memory_gb = 0
    if args.device.startswith("npu"):
        npu_peak_memory_gb = torch.npu.max_memory_allocated() / 1024**3
        npu_current_memory_gb = torch.npu.memory_allocated() / 1024**3

    # Print results
    print(f"\n{'=' * 50}")
    print(f"Benchmark Results:")
    print(f"{'=' * 50}")
    print(f"  Model loading:       {t_load:.3f}s")
    print(f"  First run (warmup):  {first_run_latency:.3f}s")
    print(f"  Average latency:     {avg_latency:.3f}s")
    print(f"  Min latency:         {min_latency:.3f}s")
    print(f"  Max latency:         {max_latency:.3f}s")
    print(f"  Std deviation:       {std_latency:.3f}s")
    print(f"  P50 latency:         {p50_latency:.3f}s")
    print(f"  P95 latency:         {p95_latency:.3f}s")
    print(f"  Throughput:          {throughput:.3f} images/s")
    if args.device.startswith("npu"):
        print(f"  NPU peak memory:    {npu_peak_memory_gb:.2f} GB")
        print(f"  NPU current memory: {npu_current_memory_gb:.2f} GB")
    print(f"{'=' * 50}")

    # Build report
    report = {
        "model": str(args.model),
        "pipeline_class": type(pipe).__name__,
        "device": args.device,
        "dtype": args.dtype,
        "prompt": args.prompt,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "warmup_runs": args.warmup_runs,
        "num_runs": args.num_runs,
        "results": {
            "model_load_time_s": round(t_load, 3),
            "first_run_latency_s": round(first_run_latency, 3),
            "avg_latency_s": round(avg_latency, 3),
            "min_latency_s": round(min_latency, 3),
            "max_latency_s": round(max_latency, 3),
            "std_latency_s": round(std_latency, 3),
            "p50_latency_s": round(p50_latency, 3),
            "p95_latency_s": round(p95_latency, 3),
            "throughput_images_per_s": round(throughput, 3),
            "npu_peak_memory_gb": round(npu_peak_memory_gb, 2),
            "npu_current_memory_gb": round(npu_current_memory_gb, 2),
        },
        "all_latencies_s": [round(lat, 3) for lat in latencies],
    }

    # Environment info
    env_info = {
        "torch_version": torch.__version__,
        "diffusers_version": diffusers.__version__,
    }
    try:
        import torch_npu as tnpu

        env_info["torch_npu_version"] = tnpu.__version__
    except Exception:
        pass
    try:
        import transformers

        env_info["transformers_version"] = transformers.__version__
    except Exception:
        pass
    if args.device.startswith("npu"):
        env_info["npu_device_name"] = torch.npu.get_device_name(0)
        env_info["npu_device_count"] = torch.npu.device_count()
    report["environment"] = env_info

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to: {output_path}")

    # Cleanup
    del pipe
    gc.collect()
    if args.device.startswith("npu"):
        torch.npu.empty_cache()

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Diffusers Pipeline on Ascend NPU"
    )

    parser.add_argument(
        "--model", type=str, required=True, help="Path to model weights directory"
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="Text prompt for generation"
    )

    parser.add_argument("--device", type=str, default="npu:0")
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

    parser.add_argument(
        "--warmup-runs", type=int, default=1, help="Number of warmup runs (default: 1)"
    )
    parser.add_argument(
        "--num-runs", type=int, default=5, help="Number of measured runs (default: 5)"
    )

    parser.add_argument("--pipeline-class", type=str, default=None)

    parser.add_argument("--attention-slicing", action="store_true")
    parser.add_argument("--vae-slicing", action="store_true")
    parser.add_argument("--vae-tiling", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")

    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save JSON benchmark report",
    )

    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Error: model path does not exist: {args.model}", file=sys.stderr)
        sys.exit(1)

    run_benchmark(args)


if __name__ == "__main__":
    main()
