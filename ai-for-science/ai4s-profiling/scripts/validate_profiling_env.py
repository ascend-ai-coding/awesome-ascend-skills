#!/usr/bin/env python3
"""Validate torch_npu profiler prerequisites for AI4S profiling."""

import argparse
import os
import sys


def emit(status: str, label: str, message: str) -> None:
    print(f"[{status}] {label}: {message}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate profiling prerequisites")
    parser.add_argument("--device", default="npu:0", help="Target device, default: npu:0")
    parser.add_argument(
        "--output-dir",
        default="./profiling_output",
        help="Expected profiling output directory",
    )
    args = parser.parse_args()

    failures = 0

    if not os.path.isdir(os.path.dirname(os.path.abspath(args.output_dir))):
        emit("FAIL", "output-dir", f"parent path does not exist for {args.output_dir}")
        failures += 1
    else:
        emit("PASS", "output-dir", f"parent path exists for {args.output_dir}")

    try:
        import torch
        import torch_npu

        emit("PASS", "torch", torch.__version__)
        emit("PASS", "torch_npu", torch_npu.__version__)

        profiler = getattr(torch_npu, "profiler", None)
        if profiler is None:
            emit("FAIL", "profiler", "torch_npu.profiler is unavailable")
            failures += 1
        else:
            required = ["profile", "schedule", "tensorboard_trace_handler"]
            missing = [name for name in required if not hasattr(profiler, name)]
            if missing:
                emit("FAIL", "profiler-api", f"missing attributes: {', '.join(missing)}")
                failures += 1
            else:
                emit("PASS", "profiler-api", "required profiler APIs found")

        if not torch.npu.is_available():
            emit("FAIL", "npu", "torch.npu.is_available() returned False")
            failures += 1
        else:
            tensor = torch.randn(4, 4, device=args.device)
            value = (tensor @ tensor).mean().item()
            emit("PASS", "npu-runtime", f"device {args.device} is usable (mean={value:.4f})")
    except Exception as exc:
        emit("FAIL", "runtime", str(exc))
        failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
