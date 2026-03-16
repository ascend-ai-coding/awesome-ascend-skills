#!/usr/bin/env python3
"""
Pre-flight validation for Diffusers Pipeline inference on Ascend NPU.

Checks: Python packages, NPU availability, memory, model weight integrity.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple


def check_torch() -> Tuple[bool, str]:
    """Check PyTorch installation."""
    try:
        import torch

        version = torch.__version__
        parts = version.split(".")[:2]
        major, minor = int(parts[0]), int(parts[1])
        if major > 2 or (major == 2 and minor >= 1):
            return True, f"torch {version}"
        return False, f"torch {version} (requires >= 2.1)"
    except ImportError:
        return False, "torch not installed"


def check_torch_npu() -> Tuple[bool, str]:
    """Check torch_npu installation."""
    try:
        import torch_npu

        return True, f"torch_npu {torch_npu.__version__}"
    except ImportError:
        return False, "torch_npu not installed"


def check_diffusers() -> Tuple[bool, str]:
    """Check Diffusers installation."""
    try:
        import diffusers

        version = diffusers.__version__
        parts = version.split(".")
        if int(parts[0]) > 0 or (int(parts[0]) == 0 and int(parts[1]) >= 28):
            return True, f"diffusers {version}"
        return False, f"diffusers {version} (requires >= 0.28)"
    except ImportError:
        return False, "diffusers not installed"


def check_transformers() -> Tuple[bool, str]:
    """Check Transformers installation."""
    try:
        import transformers

        return True, f"transformers {transformers.__version__}"
    except ImportError:
        return False, "transformers not installed"


def check_accelerate() -> Tuple[bool, str]:
    """Check Accelerate installation.

    Accelerate is optional for basic inference, but recommended for CPU offload
    and other orchestration features referenced in the skill.
    """
    try:
        import accelerate

        return True, f"accelerate {accelerate.__version__}"
    except ImportError:
        return True, "accelerate not installed (optional unless using CPU offload)"


def check_npu_available(device: str) -> Tuple[bool, str]:
    """Check if NPU device is available."""
    try:
        import torch
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            return False, "NPU unavailable (torch.npu.is_available() = False)"

        count = torch.npu.device_count()
        device_id = int(device.split(":")[-1]) if ":" in device else 0

        if device_id >= count:
            return False, f"Device {device} not found ({count} NPUs total)"

        name = torch.npu.get_device_name(device_id)
        return True, f"{count} NPU(s), {device} = {name}"
    except Exception as e:
        return False, f"NPU check failed: {e}"


def check_npu_memory(device: str, min_memory_gb: float = 16) -> Tuple[bool, str]:
    """Check NPU memory status."""
    try:
        import torch
        import torch_npu  # noqa: F401

        device_id = int(device.split(":")[-1]) if ":" in device else 0
        free, total = torch.npu.mem_get_info(device_id)
        free_gb = free / 1024**3
        total_gb = total / 1024**3
        used_gb = total_gb - free_gb

        status = (
            f"Total: {total_gb:.1f}GB  Used: {used_gb:.1f}GB  Free: {free_gb:.1f}GB"
        )
        if free_gb < min_memory_gb:
            return False, f"{status} (free memory < {min_memory_gb:.0f}GB)"
        return True, status
    except Exception as e:
        return False, f"Memory check failed: {e}"


def check_model_weights(model_path: str) -> Tuple[bool, str]:
    """Check model weight directory integrity.

    Expects `model_index.json` with component keys (e.g. transformer, vae,
    tokenizer) mapped to `[library, class_name]`, and each component directory
    to contain at least one valid config file.
    """
    if not model_path:
        return True, "No model path specified (skipped)"

    model_dir = Path(model_path)
    if not model_dir.exists():
        return False, f"Path does not exist: {model_path}"

    model_index = model_dir / "model_index.json"
    if not model_index.exists():
        return False, "Missing model_index.json"

    try:
        with open(model_index, "r") as f:
            index = json.load(f)
    except json.JSONDecodeError:
        return False, "Invalid model_index.json"

    missing = []
    components = []
    for key, value in index.items():
        if key.startswith("_"):
            continue
        if not isinstance(value, list) or len(value) != 2:
            continue

        comp_dir = model_dir / key
        components.append(key)
        if not comp_dir.exists():
            missing.append(key)
            continue

        has_config = (comp_dir / "config.json").exists()
        has_sched_config = (comp_dir / "scheduler_config.json").exists()
        has_tokenizer_config = (comp_dir / "tokenizer_config.json").exists()
        if not (has_config or has_sched_config or has_tokenizer_config):
            missing.append(f"{key}/config.json")

    if missing:
        return False, f"Missing components: {', '.join(missing)}"

    return True, f"OK ({len(components)} components: {', '.join(components)})"


def estimate_memory(model_path: str) -> Tuple[bool, str]:
    """Estimate model memory requirements vs available NPU memory."""
    if not model_path:
        return True, "No model path specified (skipped)"

    model_dir = Path(model_path)
    if not (model_dir / "model_index.json").exists():
        return True, "Cannot estimate (no model_index.json)"

    total_size = 0
    for ext in ("*.safetensors", "*.bin"):
        for f in model_dir.rglob(ext):
            total_size += f.stat().st_size

    if total_size == 0:
        return True, "No weight files found (metadata-only directory)"

    size_gb = total_size / 1024**3
    estimated_gb = size_gb * 1.8

    try:
        import torch
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            free, _ = torch.npu.mem_get_info(0)
            free_gb = free / 1024**3
            if estimated_gb > free_gb:
                return False, (
                    f"Model {size_gb:.1f}GB, estimated {estimated_gb:.1f}GB needed, "
                    f"{free_gb:.1f}GB available (enable memory optimization)"
                )
            return True, (
                f"Model {size_gb:.1f}GB, estimated {estimated_gb:.1f}GB needed, "
                f"{free_gb:.1f}GB available"
            )
    except Exception:
        pass

    return True, f"Model {size_gb:.1f}GB, estimated {estimated_gb:.1f}GB needed"


def main():
    parser = argparse.ArgumentParser(
        description="Diffusers Pipeline pre-flight validation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model weights path (checks weight integrity if provided)",
    )
    parser.add_argument(
        "--device", type=str, default="npu:0", help="Target device (default: npu:0)"
    )
    parser.add_argument(
        "--min-memory",
        type=float,
        default=16,
        help="Minimum free NPU memory in GB (default: 16)",
    )
    args = parser.parse_args()

    checks = [
        ("PyTorch", lambda: check_torch()),
        ("torch_npu", lambda: check_torch_npu()),
        ("Diffusers", lambda: check_diffusers()),
        ("Transformers", lambda: check_transformers()),
        ("Accelerate", lambda: check_accelerate()),
        ("NPU availability", lambda: check_npu_available(args.device)),
        ("NPU memory", lambda: check_npu_memory(args.device, args.min_memory)),
        ("Model weights", lambda: check_model_weights(args.model)),
        ("Memory estimate", lambda: estimate_memory(args.model)),
    ]

    print("=" * 60)
    print("Diffusers Pipeline Pre-flight Validation")
    print(f"Target device: {args.device}")
    if args.model:
        print(f"Model path:    {args.model}")
    print("=" * 60)

    passed = 0
    failed = 0

    for name, check_fn in checks:
        ok, msg = check_fn()
        if ok:
            status = "\u2705"
            passed += 1
        else:
            status = "\u274c"
            failed += 1
        print(f"  {status} {name}: {msg}")

    print("=" * 60)
    print(f"Result: {passed} passed, {failed} failed")

    if failed > 0:
        print("\n\u274c Validation FAILED. Fix issues above before running inference.")
        sys.exit(1)
    else:
        print("\n\u2705 Validation PASSED. Ready for inference.")
        sys.exit(0)


if __name__ == "__main__":
    main()
