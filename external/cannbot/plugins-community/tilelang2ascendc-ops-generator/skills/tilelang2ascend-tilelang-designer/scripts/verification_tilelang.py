#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import logging
import math
import os
import sys
import traceback
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


SCRIPT_DIR = Path(__file__).resolve().parent
WORKDIR = SCRIPT_DIR.parent

# Import shared utility functions from the canonical AscendC verification module.
# Canonical source: tilelang2ascend-translator/scripts/verification_ascendc.py
_ASCENDC_SCRIPTS = (
    Path(__file__).resolve().parents[1] / "tilelang2ascend-translator" / "scripts"
)
if str(_ASCENDC_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_ASCENDC_SCRIPTS))

from verification_ascendc import (
    _load_module,
    _find_model_class,
    _clone_value,
    _move_to_device,
    _normalize_output,
    _contains_int8_tensor,
    _summarize_value,
    _get_device,
    _compare_structured,
    _resolve_task_dir,
    _print_report as _ascendc_print_report,
    _find_first_mismatch,
    _int_diff_summary,
    _execute_models,
)


def _tensor_diff_summary(lhs: torch.Tensor, rhs: torch.Tensor, atol: float = 0.0, rtol: float = 0.0):
    if lhs.shape != rhs.shape:
        return f"shape mismatch: ref={tuple(lhs.shape)}, cand={tuple(rhs.shape)}"

    lhs_fp = torch.nan_to_num(lhs.to(torch.float32))
    rhs_fp = torch.nan_to_num(rhs.to(torch.float32))
    diff = (lhs_fp - rhs_fp).abs()
    allowed = atol + rtol * rhs_fp.abs()
    mismatch_mask = diff > allowed
    mismatch_count = mismatch_mask.sum().item() if diff.numel() else 0
    total = lhs.numel()
    mismatch_ratio = (mismatch_count / total) if total else 0.0
    max_abs = diff.max().item() if diff.numel() else 0.0
    mean_abs = diff[mismatch_mask].mean().item() if mismatch_count else 0.0

    if torch.is_floating_point(lhs) or torch.is_floating_point(rhs):
        return (
            f"dtype(ref={lhs.dtype}, cand={rhs.dtype}), "
            f"unequal_elements={mismatch_count}, mismatch_ratio={mismatch_ratio:.6%}, "
            f"max_abs_diff={max_abs:.6g}, mean_abs_diff={mean_abs:.6g}"
        )

    return _int_diff_summary(lhs, rhs, lhs.numel())


def _compare_values(lhs, rhs, atol: float, rtol: float, path: str = "output"):
    def _compare_leaf(a, b, p):
        if isinstance(a, torch.Tensor):
            if a.shape != b.shape:
                return False, f"{p}: shape mismatch: ref={tuple(a.shape)}, cand={tuple(b.shape)}"
            a_fp = torch.nan_to_num(a.to(torch.float32))
            b_fp = torch.nan_to_num(b.to(torch.float32))
            if torch.allclose(a_fp, b_fp, atol=atol, rtol=rtol):
                return True, f"{p}: matched"
            return False, f"{p}: {_tensor_diff_summary(a, b, atol=atol, rtol=rtol)}"
        if a == b:
            return True, f"{p}: matched"
        return False, f"{p}: value mismatch: ref={a}, cand={b}"

    return _compare_structured(lhs, rhs, _compare_leaf, path)


def _get_input_groups(module):
    # NOTE: Intentionally different from verification_ascendc.py's version,
    # which has a fallback to get_inputs(). TileLang tasks strictly require
    # get_input_groups(), so no fallback is provided here.
    if not hasattr(module, "get_input_groups"):
        raise AttributeError(f"get_input_groups() not found in {module.__file__}")

    input_groups = module.get_input_groups()
    if not isinstance(input_groups, list) or not input_groups:
        raise ValueError("get_input_groups() must return a non-empty list")
    return input_groups


def _make_report(op):
    return {
        "op": op,
        "ok": False,
        "device": str(_get_device()),
        "task_dir": "",
        "reference": "",
        "candidate": "",
        "atol": 1e-2,
        "rtol": 1e-2,
        "inputs": [],
        "comparisons": [],
        "comparison": "",
        "error": "",
    }


def _run_comparisons(ref_model, cand_model, input_groups, device, report):
    all_ok = True
    comparisons = []
    ref_outputs, cand_outputs, input_summaries = _execute_models(
        ref_model, cand_model, input_groups, device)
    for index, (ref_out, cand_out) in enumerate(zip(ref_outputs, cand_outputs)):
        atol = report["atol"]
        rtol = report["rtol"]
        if _contains_int8_tensor(ref_out) and _contains_int8_tensor(cand_out):
            atol = 1.5
            rtol = 0.0

        ok, comparison = _compare_values(ref_out, cand_out, atol=atol, rtol=rtol,
                                         path=f"output[{index}]")
        comparisons.append(f"case[{index}]: {comparison}")
        all_ok = all_ok and ok
        report["atol"] = max(report["atol"], atol)
        report["rtol"] = min(report["rtol"], rtol) if math.isclose(rtol, 0.0) else report["rtol"]

    return all_ok, comparisons, input_summaries


def _run_verification(op: str):
    report = _make_report(op)

    task_dir = _resolve_task_dir(op, workdir=WORKDIR)
    ref_path = task_dir / "model.py"
    cand_path = task_dir / "model_new_tilelang.py"
    report["task_dir"] = str(task_dir)
    report["reference"] = str(ref_path)
    report["candidate"] = str(cand_path)

    if not ref_path.is_file():
        report["error"] = f"missing reference model: {ref_path}"
        return report
    if not cand_path.is_file():
        report["error"] = f"missing candidate model: {cand_path}"
        return report

    sys.path.insert(0, str(WORKDIR))
    try:
        ref_module = _load_module(ref_path, f"{op}_ref_model")
        cand_module = _load_module(cand_path, f"{op}_tilelang_model")

        ref_cls = _find_model_class(ref_module, "Model")
        cand_cls = _find_model_class(cand_module, "ModelNew")

        torch.manual_seed(0)
        init_inputs = getattr(ref_module, "get_init_inputs", lambda: [])()
        input_groups = _get_input_groups(ref_module)
        device = _get_device()

        ref_model = ref_cls(*_clone_value(init_inputs)).to(device).eval()
        cand_model = cand_cls(*_clone_value(init_inputs)).to(device).eval()

        all_ok, comparisons, input_summaries = _run_comparisons(
            ref_model, cand_model, input_groups, device, report)

        report["inputs"] = input_summaries
        report["comparisons"] = comparisons
        report["comparison"] = "\n".join(comparisons)
        report["ok"] = all_ok
        return report
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        if os.environ.get("VERIFICATION_TILELANG_DEBUG") == "1":
            raise
        report["traceback"] = traceback.format_exc()
        return report
    finally:
        if str(WORKDIR) in sys.path:
            sys.path.remove(str(WORKDIR))


def verify(op: str) -> bool:
    return _run_verification(op)["ok"]


def _print_report(report):
    extra = []
    if report.get("atol") == 1.5:
        extra.append(f"Tolerance : atol={report['atol']}")
    else:
        extra.append(f"Tolerance : atol={report['atol']}, rtol={report['rtol']}")
    _ascendc_print_report(report, title="TileLang Verification Report",
                          extra_header_lines=extra,
                          debug_env_var="VERIFICATION_TILELANG_DEBUG")


def main():
    if len(sys.argv) != 2:
        logger.error(
            "Usage: python .claude/skills/tilelang2ascend-tilelang-designer/"
            "scripts/verification_tilelang.py <op>"
        )
        logger.error("Result: fail")
        raise SystemExit(1)

    report = _run_verification(sys.argv[1])
    _print_report(report)
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
