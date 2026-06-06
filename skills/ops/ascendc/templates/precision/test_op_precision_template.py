#!/usr/bin/env python3
"""
Comprehensive precision evaluation for {{OP_NAME}} NPU operator.

Template placeholders (replace before use):
  {{OP_NAME}}             -> operator name, e.g. acosh
  {{NPU_CALL}}            -> NPU invocation expr using `x`, e.g. torch.ops.npu.acosh(x)
  {{CPU_REF}}             -> CPU reference expr using `x` and `dtype`, e.g. torch.acosh(x.cpu().float()).to(dtype)
  {{SUPPORTED_DTYPES}}    -> dtype list, e.g. [torch.float16, torch.float32]
  {{INPUT_LOW}}           -> domain lower bound for random input, e.g. 1.0
  {{INPUT_HIGH}}          -> domain upper bound for random input, e.g. 11.0
  {{TEST_SHAPES}}         -> list of (category, description, shape) tuples
  {{BOUNDARY_VALUES}}     -> list of (description, scalar_value) tuples for boundary tests
"""

import torch
import torch_npu
import pytest

try:
    import ascend_kernel
except ImportError:
    import os, glob
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir
    for _ in range(5):
        if os.path.exists(os.path.join(project_root, "build.sh")):
            break
        project_root = os.path.dirname(project_root)
    lib_pattern = os.path.join(project_root, "python/ascend_kernel/ascend_kernel/lib/*.so")
    lib_files = glob.glob(lib_pattern)
    if lib_files:
        torch.ops.load_library(lib_files[0])
    else:
        raise ImportError("Could not find ascend_kernel library")


SUPPORTED_DTYPES = {{SUPPORTED_DTYPES}}

THRESHOLD = {
    torch.float32:  2**-13,   # ≈ 1.22e-4
    torch.float16:  2**-10,   # ≈ 9.77e-4
    torch.bfloat16: 2**-7,    # ≈ 7.81e-3
}

# fmt: off
TEST_SHAPES = {{TEST_SHAPES}}
# Example:
# [
#     ("1D",         "128 elements",      (128,)),
#     ("1D",         "4096 elements",     (4096,)),
#     ("2D",         "32x512",            (32, 512)),
#     ("2D",         "64x768",            (64, 768)),
#     ("3D",         "8x16x64",           (8, 16, 64)),
#     ("Production", "ViT 8x197x768",     (8, 197, 768)),
# ]

BOUNDARY_VALUES = {{BOUNDARY_VALUES}}
# Example:
# [
#     ("domain lower bound x=1.0",  1.0),
#     ("near boundary x=1.001",     1.001),
#     ("moderate value x=10.0",     10.0),
#     ("large value x=1000.0",      1000.0),
# ]
# fmt: on

BOUNDARY_SHAPE = (1024,)


def is_npu_available():
    try:
        return torch.npu.is_available()
    except Exception:
        return False


@pytest.fixture(scope="module")
def device():
    if not is_npu_available():
        pytest.skip("NPU not available")
    return torch.device("npu:0")


def _make_random(shape, dtype, device):
    x = torch.rand(shape, dtype=torch.float32, device=device) * ({{INPUT_HIGH}} - {{INPUT_LOW}}) + {{INPUT_LOW}}
    return x.to(dtype)


def _make_constant(shape, value, dtype, device):
    return torch.full(shape, value, dtype=dtype, device=device)


# ============================================================
# Regular shape tests: TEST_SHAPES × SUPPORTED_DTYPES
# ============================================================

class TestRegularShapes:

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    @pytest.mark.parametrize(
        "cat,desc,shape", TEST_SHAPES, ids=[f"{c}-{d}" for c, d, _ in TEST_SHAPES]
    )
    def test_shape(self, device, cat, desc, shape, dtype):
        thresh = THRESHOLD.get(dtype, 2**-10)
        x = _make_random(shape, dtype, device)
        npu_result = {{NPU_CALL}}
        cpu_ref = {{CPU_REF}}
        npu_f = npu_result.cpu().float()
        ref_f = cpu_ref.float()
        rel_err = (npu_f - ref_f).abs() / (ref_f.abs() + 1e-7)
        mere = rel_err.mean().item()
        mare = rel_err.max().item()
        assert mere < thresh and mare < 10 * thresh, \
            f"[{cat}] {desc} dtype={dtype} MERE={mere:.2e}(thr={thresh:.2e}) MARE={mare:.2e}(thr={10*thresh:.2e})"


# ============================================================
# Boundary value tests: BOUNDARY_VALUES × SUPPORTED_DTYPES
# ============================================================

class TestBoundaryValues:

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    @pytest.mark.parametrize(
        "desc,value", BOUNDARY_VALUES, ids=[d for d, _ in BOUNDARY_VALUES]
    )
    def test_boundary(self, device, desc, value, dtype):
        thresh = THRESHOLD.get(dtype, 2**-10)
        x = _make_constant(BOUNDARY_SHAPE, value, dtype, device)
        npu_result = {{NPU_CALL}}
        cpu_ref = {{CPU_REF}}
        npu_f = npu_result.cpu().float()
        ref_f = cpu_ref.float()
        rel_err = (npu_f - ref_f).abs() / (ref_f.abs() + 1e-7)
        mere = rel_err.mean().item()
        mare = rel_err.max().item()
        assert mere < thresh and mare < 10 * thresh, \
            f"[Boundary] {desc} dtype={dtype} MERE={mere:.2e}(thr={thresh:.2e}) MARE={mare:.2e}(thr={10*thresh:.2e})"
