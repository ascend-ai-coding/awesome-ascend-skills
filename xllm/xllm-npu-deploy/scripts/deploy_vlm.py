#!/usr/bin/env python3
"""
VLM deployment script for xllm framework on NPU.
"""

import argparse
import sys
import os

# Add xllm to path if needed
try:
    from xllm import VLM, RequestParams, MMType, MMData, ArgumentParser as XllmArgumentParser
except ImportError:
    print("Error: xllm not installed. Please install xllm first.")
    sys.exit(1)

try:
    from PIL import Image
    from transformers import AutoImageProcessor
except ImportError:
    print("Error: transformers and pillow required for VLM")
    print("Install: pip install transformers pillow")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy VLM on NPU with xllm")

    # Model params
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--devices", type=str, default="npu:0", help="Devices, e.g., npu:0,npu:1")

    # Performance params
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--max_cache_size", type=int, default=0)
    parser.add_argument("--max_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_tokens_per_batch", type=int, default=20000)
    parser.add_argument("--max_seqs_per_batch", type=int, default=256)

    # VLM specific - usually need these disabled
    parser.add_argument("--disable_prefix_cache", action="store_true", default=True)
    parser.add_argument("--disable_chunked_prefill", action="store_true", default=True)
    parser.add_argument("--enable_shm", action="store_true", help="Enable shared memory (recommended for VLM)")

    # Distributed params
    parser.add_argument("--dp_size", type=int, default=1)
    parser.add_argument("--ep_size", type=int, default=1)
    parser.add_argument("--communication_backend", type=str, default="lccl")

    # Inference params
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--prompt", type=str, default="描述这张图片")

    return parser.parse_args()


def create_vlm(args):
    """Create VLM instance with given args"""
    print(f"Loading VLM from {args.model}...")
    print(f"Devices: {args.devices}")
    print(f"Note: VLM typically requires disable_prefix_cache=True and disable_chunked_prefill=True")

    vlm = VLM(
        model=args.model,
        devices=args.devices,
        block_size=args.block_size,
        max_cache_size=args.max_cache_size,
        max_memory_utilization=args.max_memory_utilization,
        max_tokens_per_batch=args.max_tokens_per_batch,
        max_seqs_per_batch=args.max_seqs_per_batch,
        disable_prefix_cache=args.disable_prefix_cache,
        disable_chunked_prefill=args.disable_chunked_prefill,
        enable_shm=args.enable_shm,
        dp_size=args.dp_size,
        ep_size=args.ep_size,
        communication_backend=args.communication_backend,
    )

    print("VLM loaded successfully!")
    return vlm


def get_chat_template(model_name: str) -> str:
    """Get appropriate chat template for VLM model"""
    model_lower = model_name.lower()

    if "qwen2.5" in model_lower or "qwen2_5" in model_lower:
        return (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            "{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    elif "qwen" in model_lower:
        return (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<|image_pad|>{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    elif "llava" in model_lower:
        return "USER: <image>\n{prompt}\nASSISTANT:"
    elif "internvl" in model_lower:
        return "<image>\n{prompt}"
    else:
        # Generic template
        return "<image>\n{prompt}"


def prepare_multimodal_data(vlm, args):
    """Prepare multimodal data from image"""
    try:
        processor = AutoImageProcessor.from_pretrained(args.model, trust_remote_code=True)
    except Exception as e:
        print(f"Warning: Could not load image processor: {e}")
        print("Attempting to continue without custom processor...")
        processor = None

    if args.image and os.path.exists(args.image):
        images = [Image.open(args.image).convert("RGB")]
    else:
        # Create a simple test image
        print("No image provided, creating test image...")
        images = [Image.new('RGB', (224, 224), color='red')]

    if processor:
        data = processor.preprocess(images, return_tensors="pt").data
    else:
        # Fallback preprocessing
        import torch
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        data = {"pixel_values": torch.stack([transform(img) for img in images])}

    # Handle different VLM formats
    mm_data = {}
    if "pixel_values" in data:
        mm_data["pixel_values"] = data["pixel_values"]
    if "image_grid_thw" in data:
        mm_data["image_grid_thw"] = data["image_grid_thw"]
    if "image_sizes" in data:
        mm_data["image_sizes"] = data["image_sizes"]

    multi_modal_datas = [MMData(MMType.IMAGE, mm_data)]
    return multi_modal_datas


def run_inference(vlm, args):
    """Run inference with image and prompt"""
    request_params = RequestParams()
    request_params.temperature = args.temperature
    request_params.max_tokens = args.max_tokens

    # Format prompt with vision tokens (Qwen2.5-VL format)
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        f"{args.prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    print(f"\nPrompt: {args.prompt}")
    print("Processing image and generating...")

    multi_modal_datas = prepare_multimodal_data(vlm, args)
    prompts = [prompt]

    outputs = vlm.generate(prompts, multi_modal_datas, request_params, True)

    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated: {generated_text}")

    return outputs


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model path {args.model} does not exist")
        sys.exit(1)

    vlm = create_vlm(args)

    try:
        run_inference(vlm, args)
    finally:
        print("\nShutting down VLM...")
        vlm.finish()


if __name__ == "__main__":
    main()
