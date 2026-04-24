#!/usr/bin/env python3
"""
DiT (Diffusion Transformer) deployment script for xllm framework on NPU.
Used for image/video generation tasks.
"""

import argparse
import sys
import os

# Add xllm to path if needed
try:
    from xllm import DiT, RequestParams
except ImportError:
    print("Error: xllm not installed or DiT not available.")
    print("Note: DiT support may be in development.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy DiT on NPU with xllm")

    # Model params
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--devices", type=str, default="npu:0", help="Devices")

    # Performance params
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--max_cache_size", type=int, default=0)
    parser.add_argument("--max_memory_utilization", type=float, default=0.95)

    # Inference params
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--prompt", type=str, default="A beautiful sunset over mountains")
    parser.add_argument("--output", type=str, default="output.png", help="Output image path")

    return parser.parse_args()


def create_dit(args):
    """Create DiT instance with given args"""
    print(f"Loading DiT from {args.model}...")
    print(f"Devices: {args.devices}")

    dit = DiT(
        model=args.model,
        devices=args.devices,
        block_size=args.block_size,
        max_cache_size=args.max_cache_size,
        max_memory_utilization=args.max_memory_utilization,
    )

    print("DiT loaded successfully!")
    return dit


def run_inference(dit, args):
    """Run inference for image generation"""
    request_params = RequestParams()
    request_params.num_inference_steps = args.num_inference_steps
    request_params.guidance_scale = args.guidance_scale
    request_params.height = args.height
    request_params.width = args.width

    prompts = [args.prompt]
    print(f"\nPrompt: {args.prompt}")
    print(f"Generating image ({args.height}x{args.width})...")
    print(f"Inference steps: {args.num_inference_steps}")

    outputs = dit.generate(prompts, request_params)

    # Save generated image
    if hasattr(outputs[0], 'image'):
        outputs[0].image.save(args.output)
        print(f"Image saved to: {args.output}")

    return outputs


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model path {args.model} does not exist")
        sys.exit(1)

    dit = create_dit(args)

    try:
        run_inference(dit, args)
    finally:
        print("\nShutting down DiT...")
        dit.finish()


if __name__ == "__main__":
    main()
