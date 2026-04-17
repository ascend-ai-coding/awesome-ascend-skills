#!/usr/bin/env python3
"""Validate the key runtime dependencies for DiffSBDD on Ascend."""

import importlib
import sys


REQUIRED_MODULES = [
    "torch",
    "torch_npu",
    "pytorch_lightning",
    "torch_scatter",
    "numpy",
    "pandas",
    "tqdm",
    "rdkit",
]
OPTIONAL_ALTERNATIVES = [("openbabel", "pybel")]


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
    failures = 0

    for name in REQUIRED_MODULES:
        if not check_module(name):
            failures += 1

    for first, second in OPTIONAL_ALTERNATIVES:
        if check_module(first):
            continue
        if check_module(second):
            continue
        print(f"[FAIL] openbabel-stack: neither {first} nor {second} is importable")
        failures += 1

    try:
        import numpy as np

        if int(np.__version__.split(".")[0]) >= 2:
            print(f"[WARN] numpy: {np.__version__} detected; this skill prefers 1.26.x")
    except Exception:
        pass

    try:
        import torch
        import torch_npu

        if not torch.npu.is_available():
            print("[FAIL] npu: torch.npu.is_available() returned False")
            failures += 1
        else:
            tensor = torch.randn(3, 3, device="npu:0")
            print(f"[PASS] npu-runtime: {(tensor + tensor).sum().item():.4f}")
    except Exception as exc:
        print(f"[FAIL] npu-runtime: {exc}")
        failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
