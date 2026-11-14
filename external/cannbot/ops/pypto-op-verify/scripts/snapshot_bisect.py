#!/usr/bin/env python3
# coding: utf-8
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# snapshot_bisect.py
#
# Per-iteration bisection runner for the snapshot automation pipeline.
#
# Given a snapshot manifest and the generated artifacts under custom/<op>/_debug/,
# it:
#   (a) loads test_inputs.make_inputs(case) to build concrete inputs,
#   (b) runs the generated snapshot kernel under each requested PyPTO mode
#       (sim/npu) and captures the inspection_<name> buffers,
#   (c) runs the generated torch golden's module_<k>_inspect() for reference,
#   (d) compares per-iteration (axis=2 is the NT axis for inside_nt_loop
#       probes) and prints a drift-onset report: the first iteration where
#       each intermediate diverges beyond atol+rtol*|truth|.
#
# Output is human-readable plus a JSON report under custom/<op>/_debug/snapshot_report.json.
#
# This is a DEBUG TOOL — no information-barrier sanitization is applied;
# @pypto-op-debugger sees full per-iteration metrics (but NOT raw golden tensor values,
# since the comparison reduces to scalar max diffs per iteration).
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import logging
import operator
import os
import sys
import time
import traceback
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch_npu  # Register torch.npu device type before any NPU usage.  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent))
load_manifest = import_module("snapshot_manifest_schema").load_manifest

TORCH_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}

# -----------------------------------------------------------------------------
# Module loading helpers
# -----------------------------------------------------------------------------


def _load_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _set_run_mode(impl_module, mode: str) -> None:
    """Reconfigure the JIT wrapper in impl_module to run under the given mode."""
    pypto = import_module("pypto")
    mode_map = {"sim": pypto.RunMode.SIM, "npu": pypto.RunMode.NPU}
    try:
        rm = mode_map[mode]
    except KeyError as e:
        raise KeyError(
            f"{mode!r} is not a valid mode; available: {sorted(mode_map)}"
        ) from e
    # PyPTO currently has no public API for changing a wrapper's run mode after
    # decoration. Isolate the compatibility access through vars() so the private
    # attribute does not leak into the rest of this module.
    for name in dir(impl_module):
        obj = getattr(impl_module, name)
        runtime_options = (
            vars(obj).get("_runtime_options") if hasattr(obj, "__dict__") else None
        )
        if isinstance(runtime_options, dict):
            runtime_options["run_mode"] = rm.value
            return
    raise RuntimeError(
        f"could not locate JIT wrapper in {impl_module.__name__} to set run_mode={mode}"
    )


# -----------------------------------------------------------------------------
# Inspection buffer allocation
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class InspectionLayout:
    batch_size: int
    head_count: int
    iteration_count: int
    shape_env: Dict[str, int]
    device: str


_SHAPE_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Cap on ** exponents so a manifest token cannot trigger a huge integer blowup.
_MAX_POW_EXPONENT = 64


def _eval_shape_expr(node, shape_env):
    """Whitelisted evaluator for shape expressions (Constant/Name/BinOp/UnaryOp)."""
    if isinstance(node, ast.Expression):
        return _eval_shape_expr(node.body, shape_env)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise ValueError(f"shape constant must be an int, got {node.value!r}")
        return node.value
    if isinstance(node, ast.Name):
        if node.id in shape_env:
            return int(shape_env[node.id])
        raise ValueError(f"unknown shape variable {node.id!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _SHAPE_BINOPS:
        left = _eval_shape_expr(node.left, shape_env)
        right = _eval_shape_expr(node.right, shape_env)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW_EXPONENT:
            raise ValueError(
                f"exponent {right} exceeds the allowed limit {_MAX_POW_EXPONENT}"
            )
        return _SHAPE_BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        operand = _eval_shape_expr(node.operand, shape_env)
        return -operand if isinstance(node.op, ast.USub) else operand
    raise ValueError(f"unsupported shape expression: {ast.dump(node)}")


def _resolve_shape_token(token, shape_env):
    if isinstance(token, int):
        return token
    if isinstance(token, str) and token in shape_env:
        return int(shape_env[token])
    try:
        tree = ast.parse(str(token), mode="eval")
        return int(_eval_shape_expr(tree, shape_env))
    except Exception as exc:
        keys = list(shape_env.keys())
        raise ValueError(
            f"cannot resolve shape token {token!r} with env keys {keys}"
        ) from exc


def _alloc_inspection_buffers(intermediates, layout):
    """
    Allocate zero buffers for every inspection_<name> the manifest requires.
    Shape is [B, H, NT, *shape] for inside_nt_loop probes, [B, H, *shape] else.
    """
    buffers: Dict[str, torch.Tensor] = {}
    for entry in intermediates:
        concrete = [
            _resolve_shape_token(shape_token, layout.shape_env)
            for shape_token in entry["shape"]
        ]
        if entry["probe_point"] == "inside_nt_loop":
            shape = [
                layout.batch_size,
                layout.head_count,
                layout.iteration_count,
                *concrete,
            ]
        else:
            shape = [layout.batch_size, layout.head_count, *concrete]
        buffers[f"inspection_{entry['name']}"] = torch.zeros(
            shape, dtype=TORCH_DTYPES[entry["dtype"]], device=layout.device
        )
    return buffers


# -----------------------------------------------------------------------------
# Per-mode execution
# -----------------------------------------------------------------------------


def _run_mode(
    impl_module,
    mode: str,
    inputs: Dict[str, Any],
    inspection_buffers: Dict[str, torch.Tensor],
) -> Tuple[str, Dict[str, torch.Tensor], str]:
    """
    Invoke the snapshot kernel under one mode. Returns (status, buffers, log).
    Buffers are moved to CPU on return so the caller can free device memory
    between modes.
    """
    try:
        _set_run_mode(impl_module, mode)
    except Exception as e:
        return ("SKIPPED_MODE_SETUP", {}, f"{e}")

    # Check mode availability at runtime.
    if mode == "npu":
        if not os.environ.get("ASCEND_HOME_PATH"):
            return ("SKIPPED_NO_NPU_ENV", {}, "ASCEND_HOME_PATH not set")
        try:
            if not torch.npu.is_available():
                return (
                    "SKIPPED_NO_NPU_DEVICE",
                    {},
                    "torch.npu.is_available() == False",
                )
        except Exception as e:
            return ("SKIPPED_NO_NPU_DEVICE", {}, f"{e}")
    t0 = time.time()
    try:
        impl_module.host_wrapper(**inputs, **inspection_buffers)
    except Exception as e:
        return ("ERROR", {}, f"{e}\n{traceback.format_exc()}")
    runtime = time.time() - t0
    cpu_buffers = {k: v.detach().cpu() for k, v in inspection_buffers.items()}
    return ("OK", cpu_buffers, f"runtime={runtime:.3f}s")


# -----------------------------------------------------------------------------
# Per-iteration compare
# -----------------------------------------------------------------------------


def _comparison_metrics(impl, gold, atol, rtol):
    if impl.numel() == 0:
        return 0.0, 0.0, True
    finite_pair = torch.isfinite(impl) & torch.isfinite(gold)
    same_inf = (
        torch.isinf(impl)
        & torch.isinf(gold)
        & (torch.signbit(impl) == torch.signbit(gold))
    )
    invalid_mask = ~(finite_pair | same_inf)
    safe_impl = torch.where(finite_pair, impl, 0.0)
    safe_gold = torch.where(finite_pair, gold, 0.0)
    finite_diff = (safe_impl - safe_gold).abs()
    diff = torch.where(invalid_mask, float("inf"), finite_diff)
    finite_rel = finite_diff / safe_gold.abs().clamp_min(1e-30)
    rel = torch.where(invalid_mask, float("inf"), finite_rel)
    tolerance_mask = same_inf | (
        finite_pair & (finite_diff <= atol + rtol * safe_gold.abs())
    )
    return (
        float(diff.max().item()),
        float(rel.max().item()),
        bool(tolerance_mask.all().item()),
    )


def _compare_per_iter(
    impl_tensor: torch.Tensor,
    gold_tensor: torch.Tensor,
    atol: float,
    rtol: float,
    has_iter_axis: bool,
) -> List[Dict[str, Any]]:
    """
    Return a list of per-iteration records:
      [{iter, max_abs_diff, max_rel_diff, all_close}, ...]
    If has_iter_axis=False (before/after probe), returns a single-element list.
    """
    impl = impl_tensor.float()
    gold = gold_tensor.float()
    if impl.shape != gold.shape:
        return [
            {
                "iter": -1,
                "max_abs_diff": float("inf"),
                "max_rel_diff": float("inf"),
                "all_close": False,
                "error": f"shape mismatch {impl.shape} vs {gold.shape}",
            }
        ]

    if not has_iter_axis:
        max_diff, max_rel, tol_ok = _comparison_metrics(impl, gold, atol, rtol)
        return [
            {
                "iter": -1,
                "max_abs_diff": max_diff,
                "max_rel_diff": max_rel,
                "all_close": tol_ok,
            }
        ]

    # axis=2 is the NT axis; iterate over it.
    iteration_count = impl.shape[2]
    out: List[Dict[str, Any]] = []
    for iteration in range(iteration_count):
        max_diff, max_rel, tol_ok = _comparison_metrics(
            impl[:, :, iteration], gold[:, :, iteration], atol, rtol
        )
        out.append(
            {
                "iter": iteration,
                "max_abs_diff": max_diff,
                "max_rel_diff": max_rel,
                "all_close": tol_ok,
            }
        )
    return out


def _drift_onset(per_iter: List[Dict[str, Any]]) -> Optional[int]:
    """First iter index where all_close=False, or None if all pass."""
    for rec in per_iter:
        if not rec["all_close"]:
            return int(rec["iter"])
    return None


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SnapshotPaths:
    op_dir: Path
    debug_dir: Path
    eval_dir: Path
    impl: Path
    golden: Path
    suite: Path
    test_inputs: Path


@dataclass(frozen=True)
class InspectionDimensions:
    batch_size: int
    head_count: int
    iteration_count: int
    shape_env: Dict[str, int]


@dataclass(frozen=True)
class ComparisonContext:
    modes: List[str]
    statuses: Dict[str, str]
    buffers: Dict[str, Dict[str, torch.Tensor]]
    golden: Dict[str, torch.Tensor]
    atol: float
    rtol: float
    verbose: bool


@dataclass(frozen=True)
class ModeRunRequest:
    manifest: Dict[str, Any]
    modes: List[str]
    impl_module: Any
    inputs: Dict[str, Any]
    dimensions: InspectionDimensions
    device: str
    verbose: bool


def _resolve_snapshot_paths(manifest_path, op, module):
    manifest_file = Path(manifest_path).resolve()
    debug_dir = manifest_file.parent
    op_dir = debug_dir.parent
    if debug_dir.name != "_debug" or op_dir.name != op:
        raise ValueError("snapshot manifest must live at custom/<op>/_debug/")
    if op_dir.parent.name != "custom":
        raise ValueError("snapshot operator directory must be under custom/")
    eval_dir = op_dir / "eval"
    paths = SnapshotPaths(
        op_dir=op_dir,
        debug_dir=debug_dir,
        eval_dir=eval_dir,
        impl=debug_dir / f"{op}_{module}_snapshot.py",
        golden=debug_dir / f"{op}_golden_modular_snapshot.py",
        suite=eval_dir / "adversarial_suite.json",
        test_inputs=eval_dir / "test_inputs.py",
    )
    for path in (paths.impl, paths.golden, paths.suite, paths.test_inputs):
        if not path.exists():
            raise FileNotFoundError(f"required file missing: {path}")
    return paths


def _load_case(suite_path, case_id):
    with suite_path.open("r", encoding="utf-8") as suite_file:
        suite = json.load(suite_file)
    cases = (
        suite
        if isinstance(suite, list)
        else suite.get("test_cases", suite.get("cases", []))
    )
    case = next((item for item in cases if item.get("id") == case_id), None)
    if case is None:
        raise ValueError(f"case {case_id!r} not found in {suite_path}")
    return case


def _load_snapshot_modules(paths, op, module):
    sys.path.insert(0, str(paths.eval_dir))
    sys.path.insert(0, str(paths.op_dir))
    return (
        _load_from_path(f"{op}_test_inputs", paths.test_inputs),
        _load_from_path(f"{op}_{module}_snapshot_impl", paths.impl),
        _load_from_path(f"{op}_golden_snapshot", paths.golden),
    )


def _inspection_dimensions(case):
    shape_env = case.get("shape", {})
    batch_size = int(shape_env.get("B", 1))
    head_count = int(shape_env.get("H", 1))
    token_count = int(shape_env.get("T", shape_env.get("S", 0)))
    block_size = int(shape_env.get("BT", 1))
    if block_size <= 0:
        raise ValueError(f"BT must be positive, got {block_size}")
    if "NT" in shape_env:
        iteration_count = max(1, int(shape_env["NT"]))
    elif token_count:
        iteration_count = max(1, (token_count + block_size - 1) // block_size)
    else:
        iteration_count = 1
    env = {
        **shape_env,
        "B": batch_size,
        "H": head_count,
        "T": token_count,
        "NT": iteration_count,
        "BT": block_size,
    }
    return InspectionDimensions(batch_size, head_count, iteration_count, env)


def _run_modes(request):
    mode_buffers: Dict[str, Dict[str, torch.Tensor]] = {}
    mode_logs: Dict[str, str] = {}
    mode_statuses: Dict[str, str] = {}
    device_str = (
        f"npu:{request.device}" if os.environ.get("ASCEND_HOME_PATH") else "cpu"
    )
    for mode in request.modes:
        layout = InspectionLayout(
            request.dimensions.batch_size,
            request.dimensions.head_count,
            request.dimensions.iteration_count,
            request.dimensions.shape_env,
            device_str,
        )
        buffers = _alloc_inspection_buffers(request.manifest["intermediates"], layout)
        status, cpu_buffers, log = _run_mode(
            request.impl_module, mode, request.inputs, buffers
        )
        mode_statuses[mode] = status
        mode_logs[mode] = log
        mode_buffers[mode] = cpu_buffers
        if request.verbose:
            logging.info("[%-5s] status=%s  %s", mode, status, log[:180])
    return mode_statuses, mode_logs, mode_buffers


def _run_golden_inspection(golden_module, golden_path, module, inputs):
    inspect_fn_name = f"module_{int(module[1:])}_inspect"
    if not hasattr(golden_module, inspect_fn_name):
        raise AttributeError(
            f"{golden_path.name} does not define {inspect_fn_name}. "
            "Either the generator failed, or @pypto-op-debugger removed the stub — "
            "restore it and fill in per-iteration capture per the manifest."
        )
    return getattr(golden_module, inspect_fn_name)(**inputs)


def _mode_comparison(mode, entry, gold_tensor, context):
    status = context.statuses[mode]
    if status != "OK":
        return {"status": status, "per_iter": [], "drift_onset": None}
    impl_tensor = context.buffers[mode].get(f"inspection_{entry['name']}")
    if impl_tensor is None:
        return {"status": "NO_BUFFER", "per_iter": [], "drift_onset": None}
    per_iter = _compare_per_iter(
        impl_tensor,
        gold_tensor,
        context.atol,
        context.rtol,
        entry["probe_point"] == "inside_nt_loop",
    )
    return {
        "status": "OK",
        "per_iter": per_iter,
        "drift_onset": _drift_onset(per_iter),
        "max_abs_diff": max(record["max_abs_diff"] for record in per_iter),
        "max_rel_diff": max(record["max_rel_diff"] for record in per_iter),
    }


def _intermediate_comparison(entry, context):
    name = entry["name"]
    gold_tensor = context.golden.get(name)
    if gold_tensor is None:
        if context.verbose:
            logging.info("  [%s] golden stub not filled — skipping compare", name)
        return {
            "status": "GOLDEN_STUB_UNFILLED",
            "note": "module_<k>_inspect returned None for this intermediate. "
            "@pypto-op-debugger must fill in the per-iteration capture before bisecting.",
        }
    return {
        "per_mode": {
            mode: _mode_comparison(mode, entry, gold_tensor, context)
            for mode in context.modes
        }
    }


def _compare_intermediates(intermediates, context):
    return {
        entry["name"]: _intermediate_comparison(entry, context)
        for entry in intermediates
    }


def _summary_cell(mode_report):
    if mode_report.get("status") != "OK":
        return f"[{mode_report.get('status', '?')[:14]}]"
    onset = mode_report.get("drift_onset")
    return "PASS all" if onset is None else f"FAIL@iter={onset}"


def _log_drift_summary(report):
    modes = report["modes"]
    logging.info("\n%s", "=" * 72)
    logging.info(
        "Drift-onset summary — %s/%s, case %s",
        report["op"],
        report["module"],
        report["case"],
    )
    logging.info("=" * 72)
    header = "{:<28s} " + " ".join("{:>16s}".format(mode) for mode in modes)
    logging.info(header.format("intermediate", *modes))
    for name, entry_report in report["intermediates"].items():
        if "per_mode" not in entry_report:
            logging.info("%-28s [%s]", name, entry_report.get("status"))
            continue
        per_mode = entry_report["per_mode"]
        cells = [_summary_cell(per_mode.get(mode, {})) for mode in modes]
        row = "{:<28s} ".format(name) + " ".join(
            "{:>16s}".format(cell) for cell in cells
        )
        logging.info(row)
    logging.info("=" * 72)
    logging.info("Drift-onset interpretation:")
    logging.info(
        "  * If intermediate X drifts first at iter K under `npu` but passes under `sim`,"
    )
    logging.info(
        "    the bug is NPU-specific in the expression that produces X (tile/pipe/memory)."
    )
    logging.info(
        "  * If all intermediates drift under both modes at the same iter, the bug is in"
    )
    logging.info(
        "    the shared expression upstream — narrow by eyeballing which expression they all depend on."
    )


def run(
    manifest_path: str | Path,
    *,
    modes_override: Optional[List[str]] = None,
    device: str = "9",
    verbose: bool = True,
) -> Dict[str, Any]:
    manifest = load_manifest(manifest_path)
    op, module, case_id = manifest["op"], manifest["module"], manifest["case"]
    modes = modes_override or manifest["modes"]
    paths = _resolve_snapshot_paths(manifest_path, op, module)
    test_inputs_mod, impl_mod, golden_mod = _load_snapshot_modules(paths, op, module)
    case = _load_case(paths.suite, case_id)
    inputs = test_inputs_mod.make_inputs(case)
    dimensions = _inspection_dimensions(case)

    per_mode_status, per_mode_log, per_mode_buffers = _run_modes(
        ModeRunRequest(manifest, modes, impl_mod, inputs, dimensions, device, verbose)
    )

    gold_intermediates = _run_golden_inspection(
        golden_mod, paths.golden, module, inputs
    )

    comparison_context = ComparisonContext(
        modes,
        per_mode_status,
        per_mode_buffers,
        gold_intermediates,
        manifest["atol"],
        manifest["rtol"],
        verbose,
    )
    report: Dict[str, Any] = {
        "op": op,
        "module": module,
        "case": case_id,
        "modes": modes,
        "per_mode_status": per_mode_status,
        "per_mode_log": per_mode_log,
        "intermediates": _compare_intermediates(
            manifest["intermediates"], comparison_context
        ),
    }

    if verbose:
        _log_drift_summary(report)

    # 7. Write JSON report.
    report_path = paths.debug_dir / "snapshot_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if verbose:
        logging.info("\nreport written: %s", report_path)
    return report


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Per-iteration snapshot bisection runner.")
    ap.add_argument(
        "--manifest",
        required=True,
        help="path to custom/<op>/_debug/snapshot_manifest.yaml",
    )
    ap.add_argument(
        "--modes",
        default=None,
        help="comma-separated subset of {sim,npu} (default: from manifest)",
    )
    ap.add_argument("--device", default="9", help="NPU device id (default: 9)")
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="suppress stdout prints; still writes JSON report",
    )
    args = ap.parse_args()
    modes = args.modes.split(",") if args.modes else None
    try:
        run(
            args.manifest,
            modes_override=modes,
            device=args.device,
            verbose=not args.quiet,
        )
    except Exception as e:
        logging.error("[snapshot_bisect FAILED] %s", e)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
