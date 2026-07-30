#!/usr/bin/env python3
"""
Auto deployment script for xllm framework.
Automatically detects model type and deploys accordingly.
"""

import argparse
import sys
import os
import subprocess

try:
    from model_detector import detect_model_type, get_deployment_recommendations
except ImportError as e:
    print(f"Error: Could not import model_detector module: {e}")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Auto-deploy xllm model on NPU")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--devices", type=str, default="npu:0", help="Devices")
    parser.add_argument("--mode", type=str, default="offline", choices=["offline", "server"],
                        help="Deployment mode")
    parser.add_argument("--port", type=int, default=8080, help="Server port (for server mode)")
    parser.add_argument("--prompt", type=str, help="Test prompt (for offline mode)")
    parser.add_argument("--image", type=str, help="Image path (for VLM)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model path {args.model} does not exist")
        sys.exit(1)

    # Detect model type
    print(f"Analyzing model: {args.model}")
    result = detect_model_type(args.model)
    model_type = result["type"]
    confidence = result["confidence"]

    print(f"Detected model type: {model_type.upper()} (confidence: {confidence})")

    if model_type == "unknown":
        print("Error: Could not detect model type. Please specify manually.")
        print("Use: python deploy_llm.py|deploy_vlm.py|deploy_dit.py directly")
        sys.exit(1)

    # Get recommendations
    recs = get_deployment_recommendations(model_type)
    print(f"\nRecommended deployment:")
    print(f"  Script: {recs['script']}")
    if recs['default_flags']:
        print(f"  Recommended flags: {' '.join(recs['default_flags'])}")

    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Build command based on mode and type
    if args.mode == "server":
        script = os.path.join(script_dir, "deploy_server.py")
        cmd = [sys.executable, script, "--model", args.model, "--devices", args.devices, "--port", str(args.port)]

        # Add VLM-specific flags
        if model_type == "vlm":
            cmd.extend(["--disable_prefix_cache", "--disable_chunked_prefill"])
    else:
        # Offline mode
        script_map = {
            "llm": "deploy_llm.py",
            "vlm": "deploy_vlm.py",
            "dit": "deploy_dit.py"
        }
        script = os.path.join(script_dir, script_map.get(model_type, "deploy_llm.py"))
        cmd = [sys.executable, script, "--model", args.model, "--devices", args.devices]

        # Add model-specific recommended flags
        if recs['default_flags']:
            cmd.extend(recs['default_flags'])

        if args.prompt:
            cmd.extend(["--prompt", args.prompt])
        if args.image and model_type == "vlm":
            cmd.extend(["--image", args.image])

    # Run deployment
    print(f"\nRunning: {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nDeployment interrupted.")


if __name__ == "__main__":
    main()
