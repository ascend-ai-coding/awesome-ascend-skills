#!/usr/bin/env python3
"""
Precision evaluation for {{OP_NAME}} operator on ascend950.
Uses aclnn interface for operator invocation (via torch.ops.npu, EXEC_NPU_CMD standard).

Template placeholders (replace before use):
  {{OP_NAME}}             -> operator name, e.g. acosh
  {{NPU_CALL}}            -> NPU invocation expr using `x`, e.g. torch.ops.npu.acosh(x)
  {{CPU_REF}}             -> CPU reference expr using `x` and `dtype`, e.g. torch.acosh(x.cpu().float()).to(dtype)
  {{SUPPORTED_DTYPES}}    -> dtype list, e.g. [torch.float16, torch.float32]
  {{INPUT_LOW}}           -> domain lower bound for random input, e.g. 1.0
  {{INPUT_HIGH}}          -> domain upper bound for random input, e.g. 11.0
  {{TEST_SHAPES}}         -> list of (category, description, shape) tuples
  {{BOUNDARY_VALUES}}     -> list of (description, scalar_value) tuples for boundary tests
  {{QUANT_DTYPE}}         -> whether operator involves quantized types, e.g. False
  {{EXTRA_TEST_CLASSES}}  -> operator-specific test classes (can be empty)

For multi-input operators, add additional input placeholders:
  {{INPUT2_NAME}}         -> second input name, e.g. gamma
  {{INPUT2_LOW}}          -> second input domain lower bound
  {{INPUT2_HIGH}}         -> second input domain upper bound
  {{NPU_CALL}}            -> e.g. torch.ops.npu.rms_norm(x, gamma, epsilon=1e-6)
  {{CPU_REF}}             -> e.g. torch_rms_norm(x.cpu().float(), gamma.cpu().float(), 1e-6)[0].to(dtype)
"""

import torch
import torch_npu
import pytest
import numpy as np

SUPPORTED_DTYPES = {{SUPPORTED_DTYPES}}

THRESHOLD = {
    torch.float32:       2**-13,
    torch.float16:       2**-10,
    torch.bfloat16:      2**-7,
    torch.float8_e4m3fn: 2**-3,
    torch.float8_e5m2:   2**-2,
}

QUANT_DTYPE = {{QUANT_DTYPE}}

TEST_SHAPES = {{TEST_SHAPES}}

BOUNDARY_VALUES = {{BOUNDARY_VALUES}}

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
    x = torch.rand(shape, dtype=torch.float32) * ({{INPUT_HIGH}} - {{INPUT_LOW}}) + {{INPUT_LOW}}
    return x.to(dtype=dtype, device=device)


def _make_constant(shape, value, dtype, device):
    x = torch.full(shape, float(value), dtype=torch.float32)
    return x.to(dtype=dtype, device=device)


def _compute_metrics(npu_out, cpu_ref, dtype):
    if npu_out.dtype in (torch.int8, torch.uint8, torch.int16, torch.int32):
        diff = (npu_out.cpu().to(torch.int32) - cpu_ref.to(torch.int32)).abs()
        max_abs = diff.max().item()
        mean_abs = diff.float().mean().item()
        return max_abs, mean_abs, 0.0, 0.0, 0.0
    npu_f = npu_out.cpu().float()
    ref_f = cpu_ref.float()
    abs_err = (npu_f - ref_f).abs()
    max_abs = abs_err.max().item()
    mean_abs = abs_err.mean().item()
    rel_err = abs_err / (ref_f.abs() + 1e-7)
    mare = rel_err.max().item()
    mere = rel_err.mean().item()
    cos = torch.nn.functional.cosine_similarity(
        npu_f.flatten().unsqueeze(0), ref_f.flatten().unsqueeze(0)
    ).item()
    return max_abs, mean_abs, mare, mere, cos


class TestRegularShapes:

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    @pytest.mark.parametrize(
        "cat,desc,shape", TEST_SHAPES, ids=[f"{c}-{d}" for c, d, _ in TEST_SHAPES]
    )
    def test_shape(self, device, cat, desc, shape, dtype):
        thresh = THRESHOLD.get(dtype, 2**-10)
        x = _make_random(shape, dtype, device)
        assert not torch.isnan(x).any(), f"Input x contains NaN!"
        npu_result = {{NPU_CALL}}
        cpu_ref = {{CPU_REF}}
        max_abs, mean_abs, mare, mere, cos = _compute_metrics(npu_result, cpu_ref, dtype)
        if QUANT_DTYPE and npu_result.dtype in (torch.int8, torch.uint8):
            assert max_abs <= 1, \
                f"[{cat}] {desc} dtype={dtype} MaxAbsErr={max_abs:.2e} (quantized int: expect MaxAbsErr <= 1)"
        else:
            assert mere < thresh and mare < 10 * thresh, \
                f"[{cat}] {desc} dtype={dtype} MERE={mere:.2e}(thr={thresh:.2e}) MARE={mare:.2e}(thr={10*thresh:.2e}) CosSim={cos:.10f}"


class TestBoundaryValues:

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    @pytest.mark.parametrize(
        "desc,value", BOUNDARY_VALUES, ids=[d for d, _ in BOUNDARY_VALUES]
    )
    def test_boundary(self, device, desc, value, dtype):
        thresh = THRESHOLD.get(dtype, 2**-10)
        x = _make_constant(BOUNDARY_SHAPE, value, dtype, device)
        assert not torch.isnan(x).any(), f"Input x contains NaN!"
        npu_result = {{NPU_CALL}}
        cpu_ref = {{CPU_REF}}
        max_abs, mean_abs, mare, mere, cos = _compute_metrics(npu_result, cpu_ref, dtype)
        if QUANT_DTYPE and npu_result.dtype in (torch.int8, torch.uint8):
            assert max_abs <= 1, \
                f"[Boundary] {desc} dtype={dtype} MaxAbsErr={max_abs:.2e} (quantized int: expect MaxAbsErr <= 1)"
        else:
            assert mere < thresh and mare < 10 * thresh, \
                f"[Boundary] {desc} dtype={dtype} MERE={mere:.2e}(thr={thresh:.2e}) MARE={mare:.2e}(thr={10*thresh:.2e}) CosSim={cos:.10f}"


class TestDeterminism:
    """Verify that multiple runs produce identical results (critical for L3 SIMT)."""

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_determinism(self, device, dtype):
        x = _make_random((1024,), dtype, device)
        results = []
        for _ in range(3):
            r = {{NPU_CALL}}
            results.append(r.clone())
        for i in range(1, len(results)):
            assert torch.equal(results[0], results[i]), \
                f"Determinism check failed: run 0 != run {i} for dtype={dtype}"


class TestSpecialValues:
    """Test zero handling for operators with no domain restrictions."""

    @pytest.mark.parametrize("dtype", [d for d in SUPPORTED_DTYPES if d not in (torch.int8, torch.float8_e4m3fn, torch.float8_e5m2)])
    def test_zero(self, device, dtype):
        x = torch.zeros(1024, dtype=dtype, device=device)
        npu_result = {{NPU_CALL}}
        cpu_ref = {{CPU_REF}}
        max_abs, mean_abs, mare, mere, cos = _compute_metrics(npu_result, cpu_ref, dtype)
        thresh = THRESHOLD.get(dtype, 2**-10)
        assert mere < thresh, f"Zero input: MERE={mere:.2e} dtype={dtype}"


{{EXTRA_TEST_CLASSES}}
