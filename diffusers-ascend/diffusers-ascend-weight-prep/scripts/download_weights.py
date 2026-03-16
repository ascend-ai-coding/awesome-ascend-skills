#!/usr/bin/env python3
"""
Model weight download script for HuggingFace and ModelScope.
Supports proxy configuration and dry-run mode for testing.
"""

import argparse
import os
import subprocess
import sys
from typing import Optional


def check_command_exists(command: str) -> bool:
    """Check if a command exists in PATH."""
    try:
        subprocess.run(
            ["which", command],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def download_huggingface(
    model_id: str,
    output_dir: Optional[str] = None,
    proxy: Optional[str] = None,
    dry_run: bool = False,
) -> int:
    """
    Download model from HuggingFace.

    Args:
        model_id: HuggingFace model ID (e.g., 'Wan-AI/Wan2.2-I2V-A14B-Diffusers')
        output_dir: Optional output directory
        proxy: Optional HF_ENDPOINT proxy URL (e.g., 'https://hf-mirror.com')
        dry_run: If True, only print the command without executing

    Returns:
        Exit code (0 for success)
    """
    if not check_command_exists("hf"):
        print(
            "Error: 'hf' command not found. Install with: pip install -U huggingface_hub",
            file=sys.stderr,
        )
        return 1

    cmd = ["hf", "download", model_id]

    if output_dir:
        cmd.extend(["--local-dir", output_dir])

    env = os.environ.copy()
    if proxy:
        env["HF_ENDPOINT"] = proxy
        print(f"Using HuggingFace proxy: {proxy}")

    print(f"Downloading from HuggingFace: {model_id}")
    print(f"Command: {' '.join(cmd)}")

    if dry_run:
        print("[DRY RUN] Command would be executed with environment:")
        if proxy:
            print(f"  HF_ENDPOINT={proxy}")
        return 0

    try:
        result = subprocess.run(cmd, env=env, check=True)
        print(f"Successfully downloaded: {model_id}")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error downloading from HuggingFace: {e}", file=sys.stderr)
        return e.returncode


def download_modelscope(
    model_id: str, output_dir: Optional[str] = None, dry_run: bool = False
) -> int:
    """
    Download model from ModelScope.

    Args:
        model_id: ModelScope model ID (e.g., 'Wan-AI/Wan2.2-T2V-A14B')
        output_dir: Optional output directory
        dry_run: If True, only print the command without executing

    Returns:
        Exit code (0 for success)
    """
    if not check_command_exists("modelscope"):
        print(
            "Error: 'modelscope' command not found. Install with: pip install modelscope",
            file=sys.stderr,
        )
        return 1

    cmd = ["modelscope", "download", "--model", model_id]

    if output_dir:
        cmd.extend(["--local_dir", output_dir])

    print(f"Downloading from ModelScope: {model_id}")
    print(f"Command: {' '.join(cmd)}")

    if dry_run:
        print("[DRY RUN] Command would be executed")
        return 0

    try:
        result = subprocess.run(cmd, check=True)
        print(f"Successfully downloaded: {model_id}")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error downloading from ModelScope: {e}", file=sys.stderr)
        return e.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Download model weights from HuggingFace or ModelScope"
    )
    parser.add_argument(
        "source",
        choices=["hf", "modelscope"],
        help="Download source: 'hf' for HuggingFace, 'modelscope' for ModelScope",
    )
    parser.add_argument(
        "model_id", help="Model ID (e.g., 'Wan-AI/Wan2.2-I2V-A14B-Diffusers')"
    )
    parser.add_argument(
        "-o", "--output-dir", help="Output directory for downloaded weights"
    )
    parser.add_argument(
        "--proxy", help="HuggingFace proxy URL (e.g., 'https://hf-mirror.com')"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print command without executing"
    )

    args = parser.parse_args()

    if args.source == "hf":
        return download_huggingface(
            args.model_id, args.output_dir, args.proxy, args.dry_run
        )
    else:
        return download_modelscope(args.model_id, args.output_dir, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
