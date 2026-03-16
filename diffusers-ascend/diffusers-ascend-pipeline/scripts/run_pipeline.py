#!/usr/bin/env python3
"""
Generic Diffusers Pipeline inference script for Ascend NPU.

Supports any Diffusers model with automatic pipeline type detection.
Works with both real and dummy (fake) weights.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path


def get_torch_dtype(dtype_str: str):
    """Convert string dtype to torch dtype."""
    import torch

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str not in dtype_map:
        print(f"Error: unsupported dtype '{dtype_str}'", file=sys.stderr)
        sys.exit(1)
    return dtype_map[dtype_str]


def load_pipeline(args):
    """Load the Diffusers pipeline."""
    import torch
    import diffusers

    torch_dtype = get_torch_dtype(args.dtype)

    if args.pipeline_class:
        pipe_cls = getattr(diffusers, args.pipeline_class, None)
        if pipe_cls is None:
            print(
                f"Error: Pipeline class '{args.pipeline_class}' not found in diffusers",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        pipe_cls = diffusers.DiffusionPipeline

    print(f"Loading pipeline from: {args.model}")
    print(f"Pipeline class: {pipe_cls.__name__}")
    print(f"Dtype: {args.dtype}")

    t0 = time.time()
    pipe = pipe_cls.from_pretrained(args.model, torch_dtype=torch_dtype)
    load_time = time.time() - t0
    print(f"Pipeline loaded in {load_time:.2f}s")
    print(f"Detected type: {type(pipe).__name__}")

    return pipe, load_time


def apply_optimizations(pipe, args):
    """Apply memory optimizations and move to target device."""
    device = args.device

    if args.cpu_offload:
        print("Enabling sequential CPU offload...")
        pipe.enable_sequential_cpu_offload(device=device)
    else:
        print(f"Moving pipeline to {device}...")
        pipe = pipe.to(device)

    if args.attention_slicing and hasattr(pipe, "enable_attention_slicing"):
        print("Enabling attention slicing...")
        pipe.enable_attention_slicing()

    if args.vae_slicing and hasattr(pipe, "enable_vae_slicing"):
        print("Enabling VAE slicing...")
        pipe.enable_vae_slicing()

    if args.vae_tiling and hasattr(pipe, "enable_vae_tiling"):
        print("Enabling VAE tiling...")
        pipe.enable_vae_tiling()

    return pipe


def load_lora(pipe, args):
    """Load LoRA weights if specified."""
    if not args.lora:
        return pipe

    print(f"Loading LoRA from: {args.lora}")
    try:
        pipe.load_lora_weights(args.lora, adapter_name="user_lora")
        pipe.set_adapters(["user_lora"], adapter_weights=[args.lora_scale])
        print(f"LoRA loaded (scale={args.lora_scale})")
    except Exception as e:
        print(f"Warning: failed to load LoRA '{args.lora}': {e}", file=sys.stderr)
    return pipe


def run_inference(pipe, args):
    """Execute the pipeline inference."""
    import torch

    device = args.device

    if args.cpu_offload:
        gen_device = "cpu"
    else:
        gen_device = device
    generator = torch.Generator(gen_device).manual_seed(args.seed)

    infer_kwargs = {
        "prompt": args.prompt,
        "num_inference_steps": args.steps,
        "generator": generator,
    }

    if args.guidance_scale is not None:
        infer_kwargs["guidance_scale"] = args.guidance_scale
    if args.height:
        infer_kwargs["height"] = args.height
    if args.width:
        infer_kwargs["width"] = args.width

    print(f"\n--- Inference ---")
    print(f"Prompt: {args.prompt}")
    print(f"Steps: {args.steps}")
    print(f"Seed: {args.seed}")
    if args.guidance_scale is not None:
        print(f"Guidance scale: {args.guidance_scale}")

    if device.startswith("npu"):
        torch.npu.synchronize()

    t0 = time.time()
    result = pipe(**infer_kwargs)
    if device.startswith("npu"):
        torch.npu.synchronize()
    infer_time = time.time() - t0

    return result, infer_time


def save_output(result, args, infer_time: float, load_time: float):
    """Save the inference output and optional metadata."""
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(result, "images") and result.images is not None:
        image = result.images[0]
        image.save(str(output_path))
        print(f"\nImage saved to: {output_path}")
        print(f"Image size: {image.size}")
    elif hasattr(result, "frames") and result.frames is not None:
        try:
            from diffusers.utils import export_to_video

            video_path = output_path.with_suffix(".mp4")
            export_to_video(result.frames[0], str(video_path), fps=8)
            print(f"\nVideo saved to: {video_path}")
        except Exception as e:
            print(f"Warning: could not save video: {e}")
            if len(result.frames[0]) > 0:
                result.frames[0][0].save(str(output_path))
                print(f"First frame saved to: {output_path}")
    else:
        print("Warning: no images or frames in result")

    if args.benchmark:
        import torch

        print(f"\n--- Timing ---")
        print(f"Model loading: {load_time:.2f}s")
        print(f"Inference:     {infer_time:.2f}s")
        print(f"Total:         {load_time + infer_time:.2f}s")

        if args.device.startswith("npu"):
            peak = torch.npu.max_memory_allocated() / 1024**3
            current = torch.npu.memory_allocated() / 1024**3
            print(f"\n--- NPU Memory ---")
            print(f"Peak:    {peak:.2f} GB")
            print(f"Current: {current:.2f} GB")

        meta_path = output_path.with_suffix(".json")
        metadata = {
            "model": str(args.model),
            "prompt": args.prompt,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "device": args.device,
            "dtype": args.dtype,
            "load_time_s": round(load_time, 3),
            "inference_time_s": round(infer_time, 3),
        }
        if args.device.startswith("npu"):
            metadata["npu_peak_memory_gb"] = round(
                torch.npu.max_memory_allocated() / 1024**3, 3
            )
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"Metadata saved to: {meta_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Diffusers Pipeline inference on Ascend NPU"
    )

    parser.add_argument(
        "--model", type=str, required=True, help="Path to model weights directory"
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="Text prompt for generation"
    )

    parser.add_argument(
        "--device", type=str, default="npu:0", help="Target device (default: npu:0)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype (default: bfloat16)",
    )

    parser.add_argument(
        "--steps", type=int, default=20, help="Number of inference steps (default: 20)"
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.5,
        help="Guidance scale (default: 3.5)",
    )
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Output file path (default: output.png)",
    )

    parser.add_argument(
        "--pipeline-class",
        type=str,
        default=None,
        help="Specific pipeline class (e.g., FluxPipeline)",
    )

    parser.add_argument("--lora", type=str, default=None, help="Path to LoRA weights")
    parser.add_argument(
        "--lora-scale", type=float, default=1.0, help="LoRA scale factor (default: 1.0)"
    )

    parser.add_argument(
        "--attention-slicing", action="store_true", help="Enable attention slicing"
    )
    parser.add_argument("--vae-slicing", action="store_true", help="Enable VAE slicing")
    parser.add_argument("--vae-tiling", action="store_true", help="Enable VAE tiling")
    parser.add_argument(
        "--cpu-offload", action="store_true", help="Enable sequential CPU offload"
    )

    parser.add_argument(
        "--benchmark", action="store_true", help="Print detailed timing and memory info"
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: model path does not exist: {args.model}", file=sys.stderr)
        sys.exit(1)
    if not (model_path / "model_index.json").exists():
        print(f"Error: model_index.json not found in {args.model}", file=sys.stderr)
        sys.exit(1)

    if args.device.startswith("npu"):
        try:
            import torch
            import torch_npu  # noqa: F401

            if not torch.npu.is_available():
                print("Error: NPU not available.", file=sys.stderr)
                sys.exit(1)
        except ImportError:
            print("Error: torch_npu not installed.", file=sys.stderr)
            sys.exit(1)

    pipe, load_time = load_pipeline(args)
    pipe = apply_optimizations(pipe, args)
    pipe = load_lora(pipe, args)
    result, infer_time = run_inference(pipe, args)
    save_output(result, args, infer_time, load_time)

    del pipe
    gc.collect()
    if args.device.startswith("npu"):
        import torch

        torch.npu.empty_cache()

    print("\nDone!")


if __name__ == "__main__":
    main()
