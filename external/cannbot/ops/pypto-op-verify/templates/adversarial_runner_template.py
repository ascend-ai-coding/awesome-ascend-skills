# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Template: custom/<op>/eval/adversarial_runner.py
#
# Copy this file VERBATIM to custom/<op>/eval/adversarial_runner.py and set
# OP_NAME below. The op-specific input construction lives entirely in
# test_inputs.py (make_inputs / PRIMARY_INPUT_ORDER) — this runner is generic.
#
# Verification runs on the NPU only (the kernel is executed via its impl
# wrapper, which pins the NPU device at import). DO NOT rename the CLI flags,
# the report keys, or the helper functions below — the debug tooling and the
# verification gates bind to these names.
#
# Comparison semantics (DIRECT MODE — no cross-phase imports). For
# --up-to-module k the runner builds the cumulative suffix (1, 12, 123, ...),
# takes the impl's cumulative wrapper for that suffix as the candidate, and
# compares it against the matching cumulative golden — the only golden it
# imports. See _phase_suffix / _resolve_impl_wrapper / _load_truth_golden.

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import traceback
from pathlib import Path

import torch

# Bare (message-only) stdout logging so the printed summary stays readable.
_LOGGER = logging.getLogger("adversarial_runner")

# ── Agent fills this in (the operator name, e.g. "relu") ─────────────────────
OP_NAME = "<op>"

# Keys that must never reach evaluation_report.json (information barrier).
_FORBIDDEN_REPORT_KEYS = {
    "golden_tensors", "golden_values", "golden_source", "raw_inputs",
    "input_tensors", "truth", "truth_tuple",
}


# ── Naming-contract helpers — DO NOT RENAME ──────────────────────────────────

def _phase_suffix(up_to_module: int) -> str:
    """Cumulative module suffix for up_to_module=k: 1, 12, 123, ..."""
    return "".join(str(i) for i in range(1, up_to_module + 1))


def _import_from_path(mod_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {mod_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _load_truth_golden(op_dir: str, op_name: str, suffix: str):
    """Import <op>_module<suffix>_golden from <op_dir>/modules/. The ONLY golden
    this runner imports — never any other module-suffix golden."""
    path = Path(op_dir) / "modules" / f"{op_name}_module{suffix}_golden.py"
    mod = _import_from_path(f"{op_name}_module{suffix}_golden", path)
    return getattr(mod, f"{op_name}_module{suffix}_golden")


def _resolve_impl_wrapper(impl_module, op_name: str, suffix: str):
    """Return impl_module.<op>_module<suffix>_wrapper; fail fast if absent."""
    fn_name = f"{op_name}_module{suffix}_wrapper"
    if not hasattr(impl_module, fn_name):
        raise AttributeError(
            f"naming-contract violation: {fn_name} not found in impl module"
        )
    return getattr(impl_module, fn_name)


def _compare(candidate_tuple, truth_tuple, atol, rtol) -> dict:
    """Leaf-wise comparison. Returns {all_close, per_tensor:[{name, max_abs_diff,
    max_rel_diff, all_close}]}. DO NOT substitute another implementation."""
    per_tensor = []
    all_ok = True
    for i, (c, t) in enumerate(zip(candidate_tuple, truth_tuple)):
        c = c.detach().float() if hasattr(c, "detach") else torch.as_tensor(c).float()
        t = t.detach().float() if hasattr(t, "detach") else torch.as_tensor(t).float()
        abs_diff = (c - t).abs()
        max_abs = float(abs_diff.max()) if abs_diff.numel() else 0.0
        denom = t.abs().clamp_min(1e-12)
        max_rel = float((abs_diff / denom).max()) if abs_diff.numel() else 0.0
        close = bool(torch.allclose(c, t, atol=atol, rtol=rtol))
        all_ok = all_ok and close
        per_tensor.append({"name": f"out{i}", "max_abs_diff": max_abs,
                           "max_rel_diff": max_rel, "all_close": close})
    return {"all_close": all_ok, "per_tensor": per_tensor}


def _sanitize(report: dict) -> dict:
    """Strip any forbidden key (raw golden/input tensors or source) recursively.
    Runs automatically before the report is written."""
    if isinstance(report, dict):
        return {k: _sanitize(v) for k, v in report.items()
                if k not in _FORBIDDEN_REPORT_KEYS}
    if isinstance(report, list):
        return [_sanitize(v) for v in report]
    return report


# ── Suite / case selection ───────────────────────────────────────────────────

def _select_cases(suite: dict, case_id: str | None, levels: list[str] | None) -> list[dict]:
    cases = suite.get("test_cases", suite.get("cases", []))
    if case_id:
        cases = [c for c in cases if c.get("id") == case_id]
    if levels:
        want = {lv.upper() for lv in levels}
        selected = []
        for c in cases:
            tags = {f"L{c.get('level')}".upper(), str(c.get("level")).upper()}
            if tags & want:
                selected.append(c)
        cases = selected
    return cases


def _evaluate_case(truth_fn, candidate_fn, ordered, case):
    """Run truth + candidate for one case and compare, capturing any dispatch or
    runtime crash as a 'runtime' failure. Returns (all_close, per_tensor, category)."""
    atol = case.get("atol", 1e-3)
    rtol = case.get("rtol", 1e-3)
    category = None
    try:
        truth = truth_fn(*ordered)
        truth = truth if isinstance(truth, tuple) else (truth,)
        candidate = candidate_fn(*ordered)  # runs on the NPU (impl pins device)
        candidate = candidate if isinstance(candidate, tuple) else (candidate,)
        cmp = _compare(candidate, truth, atol, rtol)
        all_close = cmp["all_close"]
        per_tensor = cmp["per_tensor"]
    except Exception as exc:  # noqa: BLE001 — record any dispatch/runtime crash
        all_close = False
        per_tensor = [{"name": "error", "error": f"{exc}\n{traceback.format_exc()}"}]
        category = "runtime"
    return all_close, per_tensor, category


def _run(args) -> int:
    op_dir = str(Path(args.suite).resolve().parent.parent) if args.suite else os.getcwd()
    suite = json.loads(Path(args.suite).read_text(encoding="utf-8"))
    import test_inputs  # op-specific make_inputs / PRIMARY_INPUT_ORDER

    suffix = _phase_suffix(args.up_to_module)
    truth_fn = _load_truth_golden(op_dir, OP_NAME, suffix)
    impl_module = _import_from_path(Path(args.impl).stem, Path(args.impl))
    candidate_fn = _resolve_impl_wrapper(impl_module, OP_NAME, suffix)

    report: dict = {
        "op_name": OP_NAME, "impl_file": args.impl, "up_to_module": args.up_to_module,
        "total_modules": suite.get("total_modules"), "status": "PASS",
        "checks": [], "cases_total": 0, "cases_passed": 0, "first_failure": None,
    }
    for case in _select_cases(suite, args.case, args.levels):
        inputs = test_inputs.make_inputs(case)
        ordered = [inputs[n] for n in test_inputs.PRIMARY_INPUT_ORDER if n in inputs]
        all_close, per_tensor, category = _evaluate_case(
            truth_fn, candidate_fn, ordered, case)
        report["checks"].append({
            "id": case.get("id"), "level": case.get("level"),
            "precision_checked": case.get("precision", True),
            "all_close": all_close, "per_tensor": per_tensor,
        })
        report["cases_total"] += 1
        report["cases_passed"] += 1 if all_close else 0
        if not all_close and report["first_failure"] is None:
            report["status"] = "FAIL"
            report["first_failure"] = {
                "case_id": case.get("id"),
                "failing_module_boundary": args.up_to_module,
                "failure_category": category or "precision",
                "summary": f"case {case.get('id')} failed on NPU "
                           f"(category={category or 'precision'})",
            }
    out_path = args.report or str(Path(op_dir) / "eval" / "evaluation_report.json")
    Path(out_path).write_text(json.dumps(_sanitize(report), indent=2), encoding="utf-8")
    _LOGGER.info("status=%s cases=%s/%s (NPU)", report["status"],
                 report["cases_passed"], report["cases_total"])
    return 0 if report["status"] == "PASS" else 1


def _self_test(args) -> int:
    """Structural check: composed modular golden reproduces the user golden.
    No impl needed. Imports <op>_golden and <op>_module<suffixN>_golden."""
    op_dir = str(Path(args.suite).resolve().parent.parent) if args.suite else os.getcwd()
    suite = json.loads(Path(args.suite).read_text(encoding="utf-8")) if args.suite else {}
    import test_inputs
    user_golden = _import_from_path(f"{OP_NAME}_golden",
                                    Path(op_dir) / f"{OP_NAME}_golden.py")
    n = suite.get("total_modules") or args.up_to_module
    suffix = _phase_suffix(n)
    composed = _load_truth_golden(op_dir, OP_NAME, suffix)
    case = (suite.get("test_cases") or suite.get("cases") or [{}])[0]
    inputs = test_inputs.make_inputs(case)
    ordered = [inputs[k] for k in test_inputs.PRIMARY_INPUT_ORDER if k in inputs]
    a = getattr(user_golden, f"{OP_NAME}_golden")(*ordered)
    b = composed(*ordered)
    a = a if isinstance(a, tuple) else (a,)
    b = b if isinstance(b, tuple) else (b,)
    res = _compare(b, a, atol=case.get("atol", 1e-3), rtol=case.get("rtol", 1e-3))
    _LOGGER.info("self-test %s: %s", "PASS" if res["all_close"] else "FAIL", res["per_tensor"])
    return 0 if res["all_close"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl")
    ap.add_argument("--up-to-module", type=int, default=1)
    ap.add_argument("--suite", default="./adversarial_suite.json")
    ap.add_argument("--case")
    ap.add_argument("--levels", type=lambda s: s.split(","))
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--report")
    ap.add_argument("--inspect")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    return _self_test(args) if args.self_test else _run(args)


if __name__ == "__main__":
    sys.exit(main())
