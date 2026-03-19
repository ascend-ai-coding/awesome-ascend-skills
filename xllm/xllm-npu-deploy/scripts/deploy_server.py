#!/usr/bin/env python3
"""
Online service deployment script for xllm framework on NPU.
Starts an OpenAI-compatible HTTP API server.
"""

import argparse
import sys
import os
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy xllm online service on NPU")

    # Model params
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--devices", type=str, default="npu:0", help="Devices")

    # Server params
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--api_name", type=str, default="xllm-api", help="API service name")

    # Performance params
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--max_cache_size", type=int, default=0)
    parser.add_argument("--max_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_tokens_per_batch", type=int, default=20480)
    parser.add_argument("--max_seqs_per_batch", type=int, default=256)

    # Optimization params
    parser.add_argument("--enable_mla", action="store_true", help="Enable MLA")
    parser.add_argument("--disable_prefix_cache", action="store_true")
    parser.add_argument("--disable_chunked_prefill", action="store_true")

    # Distributed params
    parser.add_argument("--dp_size", type=int, default=1, help="Data parallel size")
    parser.add_argument("--ep_size", type=int, default=1, help="Expert parallel size")
    parser.add_argument("--communication_backend", type=str, default="lccl")
    parser.add_argument("--rank_tablefile", type=str, default="", help="HCCL rank table")

    # Disaggregated PD (optional)
    parser.add_argument("--enable_disagg_pd", action="store_true", help="Enable disaggregated PD")
    parser.add_argument("--instance_role", type=str, default="DEFAULT", choices=["DEFAULT", "PREFILL", "DECODE"])

    # Other
    parser.add_argument("--log_level", type=str, default="INFO", help="Log level")
    parser.add_argument("--daemon", "-d", action="store_true", help="Run as daemon")
    parser.add_argument("--log_file", type=str, default="", help="Log file path (default: xllm_server_<port>.log)")
    parser.add_argument("--xllm_binary", type=str, default="", help="Path to xllm binary")

    return parser.parse_args()


def get_xllm_binary(args):
    """Get xllm binary path from args, env var, or default"""
    if args.xllm_binary:
        return args.xllm_binary
    if os.environ.get("XLLM_BINARY"):
        return os.environ.get("XLLM_BINARY")
    return "xllm"


def check_xllm_installed(xllm_binary):
    """Check if xllm CLI is available"""
    # If it's a full path, check if file exists
    if os.path.sep in xllm_binary:
        if os.path.exists(xllm_binary):
            return True
        print(f"Error: xllm binary not found at {xllm_binary}")
        return False

    # Otherwise check in PATH
    try:
        result = subprocess.run(["which", xllm_binary], capture_output=True, text=True)
        if result.returncode != 0:
            print("Error: xllm command not found in PATH")
            print("Please ensure xllm is installed and 'xllm' is in your PATH")
            print("Or specify binary path with --xllm_binary or XLLM_BINARY env var")
            return False
        return True
    except Exception as e:
        print(f"Error checking xllm: {e}")
        return False


def build_command(args):
    """Build xllm server command"""
    xllm_binary = get_xllm_binary(args)
    cmd = [xllm_binary]

    # Model and devices
    cmd.extend(["--model", args.model])
    cmd.extend(["--devices", args.devices])

    # Server config
    cmd.extend(["--host", args.host])
    cmd.extend(["--port", str(args.port)])

    # Performance
    cmd.extend(["--block_size", str(args.block_size)])
    cmd.extend(["--max_cache_size", str(args.max_cache_size)])
    cmd.extend(["--max_memory_utilization", str(args.max_memory_utilization)])
    cmd.extend(["--max_tokens_per_batch", str(args.max_tokens_per_batch)])
    cmd.extend(["--max_seqs_per_batch", str(args.max_seqs_per_batch)])

    # Optimizations
    if args.enable_mla:
        cmd.append("--enable_mla")
    if args.disable_prefix_cache:
        cmd.append("--disable_prefix_cache")
    if args.disable_chunked_prefill:
        cmd.append("--disable_chunked_prefill")

    # Distributed
    cmd.extend(["--dp_size", str(args.dp_size)])
    cmd.extend(["--ep_size", str(args.ep_size)])
    cmd.extend(["--communication_backend", args.communication_backend])

    if args.rank_tablefile:
        cmd.extend(["--rank_tablefile", args.rank_tablefile])

    # Disaggregated PD
    if args.enable_disagg_pd:
        cmd.append("--enable_disagg_pd")
        cmd.extend(["--instance_role", args.instance_role])

    return cmd


def print_server_info(args):
    """Print server information"""
    print("=" * 60)
    print("XLLM Online Service")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Devices: {args.devices}")
    print(f"Server: http://{args.host}:{args.port}")
    print("")
    print("API Endpoints:")
    print(f"  - Chat:     http://{args.host}:{args.port}/v1/chat/completions")
    print(f"  - Complete: http://{args.host}:{args.port}/v1/completions")
    print(f"  - Embeddings: http://{args.host}:{args.port}/v1/embeddings")
    print(f"  - Models:   http://{args.host}:{args.port}/v1/models")
    print("")
    print("Example request:")
    print(f"""  curl -X POST http://{args.host}:{args.port}/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{{"model": "{os.path.basename(args.model)}", "messages": [{{"role": "user", "content": "Hello!"}}]}}'""")
    print("=" * 60)


def check_model_type(model_path):
    """Check and warn about model type"""
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return

    import json
    with open(config_path) as f:
        config = json.load(f)

    # Warn for VLM
    if "vision_config" in config or config.get("mm_hidden_size", 0) > 0:
        print("\n[WARNING] Detected VLM model!")
        print("VLM models should use disable_prefix_cache=True and disable_chunked_prefill=True")
        print("Add --disable_prefix_cache --disable_chunked_prefill to your command\n")


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model path {args.model} does not exist")
        sys.exit(1)

    # Check xllm is installed
    xllm_binary = get_xllm_binary(args)
    if not check_xllm_installed(xllm_binary):
        sys.exit(1)

    # Check model type and warn
    check_model_type(args.model)

    # Build and run command
    cmd = build_command(args)

    print_server_info(args)
    print(f"\nStarting server...")
    print(f"Command: {' '.join(cmd)}\n")

    # Determine log file path
    log_file = args.log_file
    if not log_file:
        log_file = f"xllm_server_{args.port}.log"

    if args.daemon:
        # Run in background with log file
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True
            )
        print(f"Server started as daemon (PID: {process.pid})")
        print(f"Logs will be written to {log_file}")
        print(f"\nTo stop the server:")
        print(f"  kill {process.pid}")
        print(f"\nTo view logs:")
        print(f"  tail -f {log_file}")
    else:
        # Run in foreground with optional log file
        if log_file:
            print(f"Logs will be written to {log_file}")
            with open(log_file, 'w') as f:
                try:
                    subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
                except KeyboardInterrupt:
                    print("\nShutting down server...")
        else:
            try:
                subprocess.run(cmd)
            except KeyboardInterrupt:
                print("\nShutting down server...")


if __name__ == "__main__":
    main()
