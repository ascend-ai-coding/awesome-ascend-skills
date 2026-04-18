#!/usr/bin/env python3
"""
Run precision evaluation for {{OP_NAME}} and generate report.

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
import json
import os
import sys

try:
    import ascend_kernel
except ImportError:
    import glob
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

device = torch.device("npu:0")

SUPPORTED_DTYPES = {{SUPPORTED_DTYPES}}

THRESHOLD = {
    torch.float32:  2**-13,   # ≈ 1.22e-4
    torch.float16:  2**-10,   # ≈ 9.77e-4
    torch.bfloat16: 2**-7,    # ≈ 7.81e-3
}

DTYPE_NAMES = {
    torch.float32:  "float32",
    torch.float16:  "float16",
    torch.bfloat16: "bfloat16",
}

# fmt: off
TEST_SHAPES = {{TEST_SHAPES}}
BOUNDARY_VALUES = {{BOUNDARY_VALUES}}
# fmt: on

BOUNDARY_SHAPE = (1024,)


def make_random(shape, dtype):
    x = torch.rand(shape, dtype=torch.float32, device=device) * ({{INPUT_HIGH}} - {{INPUT_LOW}}) + {{INPUT_LOW}}
    return x.to(dtype)


def make_constant(shape, value, dtype):
    return torch.full(shape, value, dtype=dtype, device=device)


def compute_metrics(npu_out, cpu_ref):
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


def run_one(case_id, cat, desc, x, dtype):
    thresh = THRESHOLD.get(dtype, 2**-10)
    dtype_name = DTYPE_NAMES.get(dtype, str(dtype))

    try:
        npu_result = {{NPU_CALL}}
        cpu_ref = {{CPU_REF}}
        max_abs, mean_abs, mare, mere, cos = compute_metrics(npu_result, cpu_ref)
        passed = (mere < thresh) and (mare < 10 * thresh)
    except Exception as e:
        max_abs = mean_abs = mare = mere = -1.0
        cos = -1.0
        passed = False
        print(f"  [CASE {case_id:02d}] ERROR: {e}", file=sys.stderr)

    status = "PASS" if passed else "FAIL"
    shape_list = list(x.shape)
    numel = x.numel()
    print(f"  [{status}] Case {case_id:02d}: {cat}/{desc} | shape={shape_list} "
          f"dtype={dtype_name} "
          f"| MERE={mere:.2e} MARE={mare:.2e} thr={thresh:.2e} cos={cos:.10f}")

    return {
        "case_id": case_id,
        "category": cat,
        "description": desc,
        "shape": str(shape_list),
        "dtype": dtype_name,
        "numel": numel,
        "max_abs_err": max_abs,
        "mean_abs_err": mean_abs,
        "MARE": mare,
        "MERE": mere,
        "cosine_sim": cos,
        "threshold": thresh,
        "passed": passed,
    }, passed


def main():
    results = []
    pass_count = 0
    fail_count = 0
    case_id = 0

    for cat, desc, shape in TEST_SHAPES:
        for dtype in SUPPORTED_DTYPES:
            case_id += 1
            x = make_random(shape, dtype)
            r, ok = run_one(case_id, cat, desc, x, dtype)
            results.append(r)
            if ok:
                pass_count += 1
            else:
                fail_count += 1

    for desc, value in BOUNDARY_VALUES:
        for dtype in SUPPORTED_DTYPES:
            case_id += 1
            x = make_constant(BOUNDARY_SHAPE, value, dtype)
            r, ok = run_one(case_id, "Boundary", desc, x, dtype)
            results.append(r)
            if ok:
                pass_count += 1
            else:
                fail_count += 1

    report_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(report_dir, "{{OP_NAME}}_precision_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total = pass_count + fail_count
    print(f"\n{'='*80}")
    print(f"  {{OP_NAME}} Precision Evaluation Summary")
    print(f"  Total: {total} | Passed: {pass_count} | Failed: {fail_count}")
    print(f"  Report: {report_path}")
    print(f"{'='*80}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
