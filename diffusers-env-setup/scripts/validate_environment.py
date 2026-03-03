#!/usr/bin/env python3
"""Environment validation for Diffusers on Ascend NPU."""

import sys
import os
from importlib.metadata import version, PackageNotFoundError


# Color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def check_cann_installation():
    """Check CANN installation and detect version."""
    try:
        # Check for CANN 8.5+ path: /usr/local/Ascend/cann
        if os.path.exists("/usr/local/Ascend/cann"):
            cann_path = "/usr/local/Ascend/cann"
            # Try to read version from version.txt if exists
            version_file = os.path.join(cann_path, "version.txt")
            if os.path.exists(version_file):
                with open(version_file, "r") as f:
                    version_line = f.readline().strip()
                    version_parts = version_line.split()
                    if len(version_parts) > 2:
                        return version_parts[2], "CANN 8.5+ detected"
            return "8.5+", "CANN 8.5+ detected"

        # Check for older path: /usr/local/Ascend/ascend-toolkit
        elif os.path.exists("/usr/local/Ascend/ascend-toolkit"):
            cann_path = "/usr/local/Ascend/ascend-toolkit"
            version_file = os.path.join(cann_path, "version.txt")
            if os.path.exists(version_file):
                with open(version_file, "r") as f:
                    version_line = f.readline().strip()
                    version_parts = version_line.split()
                    if len(version_parts) > 2:
                        return version_parts[2], "CANN before 8.5 detected"
            return "unknown", "CANN before 8.5 detected"

        return None, "CANN not found"
    except Exception as e:
        return None, f"Error checking CANN: {str(e)}"


def check_cann_env_vars():
    """Check CANN environment variables."""
    required_vars = ["ASCEND_HOME_PATH", "ASCEND_OPP_PATH", "ASCEND_AICPU_PATH"]

    missing_vars = []
    for var in required_vars:
        value = os.environ.get(var)
        if value:
            print(
                f"  {var}: {value[:50]}..." if len(value) > 50 else f"  {var}: {value}"
            )
        else:
            print(f"  {var}: ${var}")
            missing_vars.append(var)

    if missing_vars:
        return False, f"Missing environment variables: {', '.join(missing_vars)}"

    return True, "All environment variables set"


def check_pytorch():
    """Check PyTorch installation."""
    try:
        import torch

        return True, str(version("torch"))
    except ImportError:
        return False, "PyTorch not installed"
    except Exception as e:
        return False, f"Error checking PyTorch: {str(e)}"


def check_torch_npu():
    """Check torch_npu installation."""
    try:
        import torch_npu

        # Try to import torch first (required for torch_npu)
        import torch

        return True, str(version("torch_npu"))
    except ImportError:
        return False, "torch_npu not installed"
    except Exception as e:
        return False, f"Error checking torch_npu: {str(e)}"


def check_npu_visibility():
    """Check NPU device visibility."""
    try:
        import torch
        import torch_npu

        devices = torch.npu.device_count()
        if devices > 0:
            # Try to access the first device
            device = torch.npu.current_device()
            device_name = torch.npu.get_device_name(device)
            return True, f"{devices} devices ({device_name})"
        else:
            return False, "No NPU devices found"
    except ImportError:
        return False, "torch_npu not available"
    except Exception as e:
        return False, f"Error checking NPU visibility: {str(e)}"


def check_diffusers():
    """Check Diffusers installation."""
    try:
        import diffusers

        return True, str(version("diffusers"))
    except ImportError:
        return False, "diffusers not installed"
    except Exception as e:
        return False, f"Error checking diffusers: {str(e)}"


def check_numpy_version():
    """Check numpy version < 2.0."""
    try:
        import numpy

        numpy_version = version("numpy")
        major_version = int(numpy_version.split(".")[0])

        if major_version < 2:
            return True, f"{numpy_version} (< 2.0)"
        else:
            return False, f"{numpy_version} (>= 2.0)"
    except ImportError:
        return False, "numpy not installed"
    except Exception as e:
        return False, f"Error checking numpy: {str(e)}"


def main():
    """Run all checks and report results."""
    print("=" * 50)
    print("Diffusers Environment Validation")
    print("=" * 50)
    print()

    checks = [
        ("CANN Installation", check_cann_installation),
        ("CANN Environment", check_cann_env_vars),
        ("PyTorch", check_pytorch),
        ("torch_npu", check_torch_npu),
        ("NPU Visibility", check_npu_visibility),
        ("Diffusers", check_diffusers),
        ("numpy", check_numpy_version),
    ]

    passed = 0
    failed = 0
    warnings = 0

    for check_name, check_func in checks:
        print(f"[{GREEN}✓{RESET}] {check_name}", end=": ")

        result, message = check_func()

        if result is True:
            print(f"{GREEN}{message}{RESET}")
            passed += 1
        elif (
            result is False
            and "version" in message.lower()
            and "not installed" not in message.lower()
        ):
            print(f"{YELLOW}{message}{RESET}")
            warnings += 1
        else:
            print(f"{RED}{message}{RESET}")
            failed += 1

        print()

    print("=" * 50)
    print(f"Result: {passed}/{len(checks)} checks passed")

    if warnings > 0:
        print(f"         {warnings} warnings")

    if failed > 0:
        print(f"         {failed} failures")

    print("=" * 50)

    # Return appropriate exit code
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
