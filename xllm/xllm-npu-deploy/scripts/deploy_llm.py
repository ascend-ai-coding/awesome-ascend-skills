#!/usr/bin/env python3
"""
LLM deployment script for xllm framework on NPU.
"""

import argparse
import sys
import os

# Add xllm to path if needed
try:
    from xllm import LLM, RequestParams, ArgumentParser as XllmArgumentParser
except ImportError:
    print("Error: xllm not installed. Please install xllm first.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy LLM on NPU with xllm")

    # Model params
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--devices", type=str, default="npu:0", help="Devices, e.g., npu:0,npu:1")

    # Performance params
    parser.add_argument("--block_size", type=int, default=128, help="KV cache block size")
    parser.add_argument("--max_cache_size", type=int, default=0, help="Max cache size (0=auto)")
    parser.add_argument("--max_memory_utilization", type=float, default=0.9, help="Max memory usage")
    parser.add_argument("--max_tokens_per_batch", type=int, default=20480, help="Max tokens per batch")
    parser.add_argument("--max_seqs_per_batch", type=int, default=256, help="Max sequences per batch")

    # Optimization params
    parser.add_argument("--enable_mla", action="store_true", help="Enable MLA attention")
    parser.add_argument("--disable_prefix_cache", action="store_true", help="Disable prefix cache")
    parser.add_argument("--disable_chunked_prefill", action="store_true", help="Disable chunked prefill")
    parser.add_argument("--enable_shm", action="store_true", help="Enable shared memory")

    # Distributed params
    parser.add_argument("--dp_size", type=int, default=1, help="Data parallel size")
    parser.add_argument("--ep_size", type=int, default=1, help="Expert parallel size")
    parser.add_argument("--communication_backend", type=str, default="lccl", help="Communication backend")
    parser.add_argument("--rank_tablefile", type=str, default="", help="HCCL rank table file")

    # Inference params
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max output tokens")
    parser.add_argument("--prompt", type=str, default="Hello, my name is", help="Test prompt")

    # Interactive mode
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")

    return parser.parse_args()


def create_llm(args):
    """Create LLM instance with given args"""
    print(f"Loading LLM from {args.model}...")
    print(f"Devices: {args.devices}")

    llm = LLM(
        model=args.model,
        devices=args.devices,
        block_size=args.block_size,
        max_cache_size=args.max_cache_size,
        max_memory_utilization=args.max_memory_utilization,
        max_tokens_per_batch=args.max_tokens_per_batch,
        max_seqs_per_batch=args.max_seqs_per_batch,
        enable_mla=args.enable_mla,
        disable_prefix_cache=args.disable_prefix_cache,
        disable_chunked_prefill=args.disable_chunked_prefill,
        enable_shm=args.enable_shm,
        dp_size=args.dp_size,
        ep_size=args.ep_size,
        communication_backend=args.communication_backend,
        rank_tablefile=args.rank_tablefile,
    )

    print("LLM loaded successfully!")
    return llm


def run_inference(llm, args):
    """Run inference with test prompt"""
    request_params = RequestParams()
    request_params.temperature = args.temperature
    request_params.top_p = args.top_p
    request_params.max_tokens = args.max_tokens

    prompts = [args.prompt]
    print(f"\nPrompt: {args.prompt}")
    print("Generating...")

    outputs = llm.generate(prompts, request_params, True)

    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated: {generated_text}")

    return outputs


def interactive_mode(llm, args):
    """Run interactive chat mode"""
    print("\n=== Interactive Mode (type 'quit' to exit) ===\n")

    while True:
        try:
            prompt = input("User: ").strip()
            if prompt.lower() in ["quit", "exit", "q"]:
                break
            if not prompt:
                continue

            request_params = RequestParams()
            request_params.temperature = args.temperature
            request_params.top_p = args.top_p
            request_params.max_tokens = args.max_tokens

            outputs = llm.generate([prompt], request_params, True)
            generated_text = outputs[0].outputs[0].text
            print(f"Assistant: {generated_text}\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\nExiting interactive mode...")


def main():
    args = parse_args()

    # Validate model path
    if not os.path.exists(args.model):
        print(f"Error: Model path {args.model} does not exist")
        sys.exit(1)

    # Create LLM
    llm = create_llm(args)

    try:
        if args.interactive:
            interactive_mode(llm, args)
        else:
            run_inference(llm, args)
    finally:
        print("\nShutting down LLM...")
        llm.finish()


if __name__ == "__main__":
    main()
