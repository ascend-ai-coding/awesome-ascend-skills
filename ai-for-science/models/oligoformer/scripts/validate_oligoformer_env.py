#!/usr/bin/env python3
"""Validate OligoFormer and RNA-FM prerequisites on Ascend."""

import argparse
import importlib
import sys
from pathlib import Path


MODULES = [
    "torch",
    "torch_npu",
    "Bio",
    "pandas",
    "prefetch_generator",
    "ptflops",
    "sklearn",
    "yacs",
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
    parser = argparse.ArgumentParser(description="Validate OligoFormer runtime environment")
    parser.add_argument(
        "--rna-fm-path",
        default="./RNA-FM",
        help="Expected RNA-FM checkout path, default: ./RNA-FM",
    )
    args = parser.parse_args()

    failures = 0

    for name in MODULES:
        if not check_module(name):
            failures += 1

    rna_fm_path = Path(args.rna_fm_path)
    if rna_fm_path.exists():
        print(f"[PASS] rna-fm-path: {rna_fm_path}")
    else:
        print(f"[FAIL] rna-fm-path: {rna_fm_path} does not exist")
        failures += 1

    try:
        import torch
        import torch_npu

        if torch.npu.is_available():
            tensor = torch.randn(2, 2, device="npu:0")
            print(f"[PASS] npu-runtime: {(tensor + 1).sum().item():.4f}")
        else:
            print("[FAIL] npu: torch.npu.is_available() returned False")
            failures += 1
    except Exception as exc:
        print(f"[FAIL] npu-runtime: {exc}")
        failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
