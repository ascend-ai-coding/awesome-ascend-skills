#!/usr/bin/env python3
"""Check BoltzGen cache assets and optional NPU runtime availability."""

import argparse
import sys
from pathlib import Path


REQUIRED = [
    "boltzgen1_diverse.ckpt",
    "boltzgen1_adherence.ckpt",
    "boltzgen1_ifold.ckpt",
    "boltz2_conf_final.ckpt",
    "mols.zip",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate BoltzGen cached assets")
    parser.add_argument("--cache-dir", default="~/.cache", help="Cache directory, default: ~/.cache")
    parser.add_argument("--check-runtime", action="store_true", help="Also validate torch_npu runtime")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser()
    failures = 0

    print(f"Checking cache directory: {cache_dir}")
    if not cache_dir.exists():
        print(f"[FAIL] cache-dir: {cache_dir} does not exist")
        return 1

    for name in REQUIRED:
        path = cache_dir / name
        if path.exists() and path.stat().st_size > 0:
            print(f"[PASS] asset: {name} ({path.stat().st_size} bytes)")
        else:
            print(f"[FAIL] asset: {name} missing or empty")
            failures += 1

    if args.check_runtime:
        try:
            import torch
            import torch_npu

            if torch.npu.is_available():
                tensor = torch.randn(2, 2, device="npu:0")
                print(f"[PASS] runtime: torch={torch.__version__}, torch_npu={torch_npu.__version__}, mean={tensor.mean().item():.4f}")
            else:
                print("[FAIL] runtime: torch.npu.is_available() returned False")
                failures += 1
        except Exception as exc:
            print(f"[FAIL] runtime: {exc}")
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
