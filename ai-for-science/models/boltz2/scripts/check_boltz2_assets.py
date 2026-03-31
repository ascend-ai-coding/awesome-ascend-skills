#!/usr/bin/env python3
"""Check Boltz2 runtime assets and optional NPU runtime availability."""

import argparse
import os
import sys
from pathlib import Path


REQUIRED = [
    "boltz2_conf.ckpt",
    "boltz2_aff.ckpt",
    "ccd.pkl",
    "mols.tar",
]
OPTIONAL = ["boltz1_conf.ckpt"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Boltz2 assets")
    parser.add_argument("--boltz-home", default="~/.boltz", help="Asset directory, default: ~/.boltz")
    parser.add_argument("--check-runtime", action="store_true", help="Also validate torch_npu runtime")
    args = parser.parse_args()

    asset_dir = Path(args.boltz_home).expanduser()
    failures = 0

    print(f"Checking asset directory: {asset_dir}")
    if not asset_dir.exists():
        print(f"[FAIL] asset-dir: {asset_dir} does not exist")
        return 1

    for name in REQUIRED:
        path = asset_dir / name
        if path.exists() and path.stat().st_size > 0:
            print(f"[PASS] required: {name} ({path.stat().st_size} bytes)")
        else:
            print(f"[FAIL] required: {name} missing or empty")
            failures += 1

    for name in OPTIONAL:
        path = asset_dir / name
        if path.exists() and path.stat().st_size > 0:
            print(f"[PASS] optional: {name} ({path.stat().st_size} bytes)")
        else:
            print(f"[WARN] optional: {name} not found")

    if args.check_runtime:
        try:
            import torch
            import torch_npu

            if torch.npu.is_available():
                tensor = torch.randn(2, 2, device="npu:0")
                print(f"[PASS] runtime: torch={torch.__version__}, torch_npu={torch_npu.__version__}, sum={(tensor + tensor).sum().item():.4f}")
            else:
                print("[FAIL] runtime: torch.npu.is_available() returned False")
                failures += 1
        except Exception as exc:
            print(f"[FAIL] runtime: {exc}")
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
