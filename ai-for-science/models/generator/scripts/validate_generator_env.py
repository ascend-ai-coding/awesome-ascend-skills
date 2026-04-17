#!/usr/bin/env python3
"""Validate the key runtime dependencies for GENERator on Ascend."""

import argparse
import importlib
import sys
from pathlib import Path


MODULES = [
    "torch",
    "torch_npu",
    "transformers",
    "datasets",
    "pandas",
    "pyarrow",
    "sklearn",
    "tqdm",
]


def check_module(name: str) -> bool:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print(f"[PASS] {name}: {version}")
        return True
    except Exception as exc:
        print(f"[FAIL] {name}: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate GENERator runtime environment")
    parser.add_argument("--model-path", default=None, help="Optional local model directory to check")
    args = parser.parse_args()

    failures = 0
    for name in MODULES:
        if not check_module(name):
            failures += 1

    if args.model_path:
        model_dir = Path(args.model_path)
        if model_dir.exists():
            print(f"[PASS] model-path: {model_dir}")
        else:
            print(f"[FAIL] model-path: {model_dir} does not exist")
            failures += 1

    try:
        import torch
        import torch_npu

        if torch.npu.is_available():
            tensor = torch.randn(2, 2, device="npu:0")
            print(f"[PASS] npu-runtime: {(tensor @ tensor).sum().item():.4f}")
        else:
            print("[FAIL] npu: torch.npu.is_available() returned False")
            failures += 1
    except Exception as exc:
        print(f"[FAIL] npu-runtime: {exc}")
        failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
