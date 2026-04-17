#!/usr/bin/env python3
"""Basic Ascend NPU environment validation for AI4S migration."""

import argparse
import importlib
import os
import sys


def ok(label: str, message: str) -> None:
    print(f"[PASS] {label}: {message}")


def fail(label: str, message: str) -> None:
    print(f"[FAIL] {label}: {message}")


def warn(label: str, message: str) -> None:
    print(f"[WARN] {label}: {message}")


def check_module(name: str):
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        ok(name, f"imported ({version})")
        return module
    except Exception as exc:
        fail(name, str(exc))
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Basic AI4S NPU environment checks")
    parser.add_argument("--device", default="npu:0", help="Target NPU device, default: npu:0")
    args = parser.parse_args()

    failures = 0

    required_env = [
        "ASCEND_HOME_PATH",
        "ASCEND_TOOLKIT_HOME",
        "ASCEND_OPP_PATH",
        "ASCEND_AICPU_PATH",
    ]
    missing = [name for name in required_env if not os.environ.get(name)]
    if missing:
        warn("cann-env", f"missing vars: {', '.join(missing)}")
    else:
        ok("cann-env", "CANN environment variables look set")

    torch = check_module("torch")
    torch_npu = check_module("torch_npu")

    try:
        import numpy as np

        if int(np.__version__.split(".")[0]) >= 2:
            warn("numpy", f"{np.__version__} detected; some CANN stacks prefer < 2.0")
        else:
            ok("numpy", np.__version__)
    except Exception as exc:
        failures += 1
        fail("numpy", str(exc))

    if torch is None or torch_npu is None:
        return 1

    try:
        if not torch.npu.is_available():
            failures += 1
            fail("npu", "torch.npu.is_available() returned False")
        else:
            count = torch.npu.device_count()
            ok("npu", f"{count} device(s) visible")
            target_index = int(args.device.split(":")[-1]) if ":" in args.device else 0
            if target_index >= count:
                failures += 1
                fail("device", f"requested {args.device}, but only {count} device(s) are visible")
            else:
                tensor = torch.randn(2, 2, device=args.device)
                result = (tensor + tensor).sum().item()
                ok("tensor-op", f"basic tensor op succeeded on {args.device} (sum={result:.4f})")
    except Exception as exc:
        failures += 1
        fail("npu-runtime", str(exc))

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
