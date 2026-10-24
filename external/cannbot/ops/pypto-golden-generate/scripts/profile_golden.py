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

"""Profile a PyTorch NPU golden and write GOLDEN_PERF_REPORT.md.

E2E method: noise-injection profiling loop + kernel_details.csv extraction.
Each iteration injects ``torch.randn(480MB).npu()`` + ``torch.max(a)`` before
the golden call, creating inter-iteration boundaries.  Performance is
extracted from ``kernel_details.csv`` by grouping kernels by ``Type``,
filtering noise (all ``TensorMove`` + first ``ReduceMax`` by Start Time),
and computing ``mean(Duration(us))`` per op type.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import importlib.util
import inspect
import json
import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_SHAPE = (8, 1024, 4096)
DEFAULT_DTYPE = "float32"
DEFAULT_ITERS = 1
NOISE_SIZE = int(192 * 1024 * 1024 * 2.5)

_REQUIRED_KINDS = {
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.KEYWORD_ONLY,
}


# ---------------------------------------------------------------------------
# Tensor spec & CLI parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass
class CaseResult:
    case_name: str
    specs: list[TensorSpec]
    per_op: dict[str, dict[str, Any]]
    e2e_a: float
    e2e_b: float
    prof_dir: Path


@dataclass
class ReportConfig:
    golden_path: Path
    output_dir: Path
    device: Any
    iters: int
    cases: list[CaseResult]


@dataclass
class ProfileConfig:
    golden_path: Path
    fn_name: str | None
    specs: list[TensorSpec]
    scalar_args: dict[str, Any]
    output_dir: Path | None
    iters: int
    device_id: int | None
    factory_name: str | None = None


@dataclass(frozen=True)
class ProfileRuntime:
    torch: Any
    torch_npu: Any
    fn: Callable
    config: ProfileConfig
    base_prof_dir: Path
    device: Any


def _parse_shape(text: str) -> tuple[int, ...]:
    v = text.strip()
    if v.startswith(("(", "[")) and v.endswith((")", "]")):
        v = v[1:-1].strip()
    if v in {"", "()", "scalar"}:
        return ()
    parts = [p for p in re.split(r"[xX,]", v) if p]
    shape = tuple(int(p) for p in parts)
    if any(d <= 0 for d in shape):
        raise ValueError(f"invalid shape: {text!r}")
    return shape


def _parse_tensor_spec(text: str) -> TensorSpec:
    parts = text.split(":")
    if len(parts) == 1:
        return TensorSpec("", _parse_shape(parts[0]), DEFAULT_DTYPE)
    if len(parts) == 2:
        return TensorSpec(parts[0].strip(), _parse_shape(parts[1]), DEFAULT_DTYPE)
    if len(parts) == 3:
        return TensorSpec(
            parts[0].strip(), _parse_shape(parts[1]), parts[2].strip() or DEFAULT_DTYPE
        )
    raise ValueError("input spec must be SHAPE, NAME:SHAPE, or NAME:SHAPE:DTYPE")


def _parse_arg(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise ValueError(f"--arg must use NAME=VALUE format: {text!r}")
    name, raw = text.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"empty argument name: {text!r}")
    try:
        return name, json.loads(raw)
    except json.JSONDecodeError:
        return name, raw


# ---------------------------------------------------------------------------
# Module loading & call building
# ---------------------------------------------------------------------------


def _load_module(path: Path):
    name = f"_pypto_golden_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        try:
            sys.path.remove(str(path.parent))
        except ValueError:
            pass
    return mod


def _find_fn(mod, path: Path, fn_name: str | None) -> Callable:
    if fn_name:
        fn = getattr(mod, fn_name, None)
        if callable(fn):
            return fn
        raise RuntimeError(f"{fn_name!r} not found in {path}")
    stem_fn = getattr(mod, path.stem, None)
    if callable(stem_fn):
        return stem_fn
    candidates = [
        (name, fn)
        for name, fn in vars(mod).items()
        if callable(fn) and name.endswith("_golden") and not name.startswith("_")
    ]
    if len(candidates) == 1:
        return candidates[0][1]
    names = [name for name, _ in candidates]
    raise RuntimeError(
        f"cannot infer golden function; pass --function. "
        f"Candidates: {', '.join(names) or '(none)'}"
    )


def _resolve_dtype(torch, dtype_name: str):
    attr = dtype_name.strip().replace("torch.", "")
    dtype = getattr(torch, attr, None)
    if dtype is None:
        raise ValueError(f"unsupported dtype: {dtype_name!r}")
    return dtype


def _make_tensor(torch, spec: TensorSpec, device):
    dtype = _resolve_dtype(torch, spec.dtype)
    size = spec.shape or ()
    if getattr(dtype, "is_floating_point", False) or getattr(
        dtype, "is_complex", False
    ):
        return torch.randn(size, dtype=dtype, device=device)
    if dtype is torch.bool:
        return torch.randint(0, 2, size, device=device).to(dtype)
    low = 0 if "uint" in str(dtype) else -3
    return torch.randint(low, 4, size, dtype=dtype, device=device)


def _build_call(
    torch, fn: Callable, specs: list[TensorSpec], scalar_args: dict[str, Any], device
) -> Callable:
    sig = inspect.signature(fn)
    values: dict[str, Any] = dict(scalar_args)

    required = [
        p
        for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty and p.kind in _REQUIRED_KINDS
    ]

    named = [s for s in specs if s.name]
    unnamed = iter(s for s in specs if not s.name)

    for s in named:
        values[s.name] = _make_tensor(torch, s, device)
    for p in required:
        if p.name in values:
            continue
        try:
            s = next(unnamed)
        except StopIteration:
            continue
        values[p.name] = _make_tensor(
            torch, TensorSpec(p.name, s.shape, s.dtype), device
        )

    missing = [p.name for p in required if p.name not in values]
    if missing:
        raise RuntimeError(
            f"missing required arguments: {', '.join(missing)}. "
            f"Use --input NAME:SHAPE[:DTYPE] or --arg NAME=VALUE."
        )

    if not specs and not scalar_args:
        for p in required:
            if p.name not in values:
                values[p.name] = _make_tensor(
                    torch, TensorSpec(p.name, DEFAULT_SHAPE, DEFAULT_DTYPE), device
                )

    pos, kw = [], {}
    for p in sig.parameters.values():
        if p.name not in values:
            continue
        if p.kind is inspect.Parameter.POSITIONAL_ONLY:
            pos.append(values[p.name])
        elif p.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            kw[p.name] = values[p.name]

    return lambda: fn(*pos, **kw)


def _is_multi_case_list(result) -> bool:
    if not isinstance(result, list) or not result:
        return False
    first = result[0]
    return isinstance(first, tuple) and len(first) == 3 and isinstance(first[0], str)


def _normalize_factory_result(result, fn: Callable) -> list[tuple[str, list, dict]]:
    """Normalize factory function result to a list of (case_name, args, kwargs).

    Detects two formats:
      - Single case: (args_list, kwargs_dict) or args_list
      - Multi case:  [(case_name, args_list, kwargs_dict), ...]

    Returns:
        List of (case_name, args_list, kwargs_dict) tuples.
    """
    if _is_multi_case_list(result):
        return [(name, args, kw) for name, args, kw in result]

    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        args_list, kwargs = result
    else:
        args_list, kwargs = result, {}

    fn_name = getattr(fn, "__name__", "golden")
    return [(fn_name, args_list, kwargs)]


def _build_call_from_args(
    torch, fn: Callable, args_list: list, kwargs: dict
) -> tuple[Callable, list[TensorSpec]]:
    """Build a zero-arg callable from one set of args/kwargs.

    Returns:
        (callable, specs) — the zero-arg callable and inferred TensorSpec list.
    """
    sig = inspect.signature(fn)
    param_names = list(sig.parameters.keys())

    specs = []
    for i, arg in enumerate(args_list):
        name = param_names[i] if i < len(param_names) else f"arg{i}"
        if isinstance(arg, torch.Tensor):
            shape = tuple(arg.shape)
            dtype = str(arg.dtype).replace("torch.", "")
            specs.append(TensorSpec(name, shape, dtype))

    def call():
        fresh_args = []
        for arg in args_list:
            if isinstance(arg, torch.Tensor):
                fresh_args.append(arg.clone())
            else:
                fresh_args.append(arg)
        fresh_kwargs = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        return fn(*fresh_args, **fresh_kwargs)

    return call, specs


def _build_call_from_factory(
    torch, fn: Callable, factory_fn: Callable, device
) -> tuple[Callable, list[TensorSpec]]:
    """Build callable from a factory function (single-case backward compat).

    Returns:
        (callable, specs) — the zero-arg callable and inferred TensorSpec list.
    """
    result = factory_fn(device)
    cases = _normalize_factory_result(result, fn)
    _, args_list, kwargs = cases[0]
    return _build_call_from_args(torch, fn, args_list, kwargs)


# ---------------------------------------------------------------------------
# kernel_details.csv reading & E2E extraction
# ---------------------------------------------------------------------------


def _find_ascend_output(prof_dir: Path) -> Path | None:
    candidates = []
    for root, dirs, _ in os.walk(prof_dir):
        if "ASCEND_PROFILER_OUTPUT" in dirs:
            candidates.append(Path(root) / "ASCEND_PROFILER_OUTPUT")
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _read_kernel_details(path: Path) -> list[tuple[str, float, float]]:
    """Read kernel_details.csv and return list of (type, duration, start_time)."""
    kernels = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                op_type = row["Type"].strip()
                dur = float(row["Duration(us)"])
                start_time = float(row["Start Time(us)"])
                kernels.append((op_type, dur, start_time))
            except (KeyError, ValueError):
                continue
    return kernels


def _first_reducemax_index(kernels: list[tuple[str, float, float]]) -> int | None:
    candidates = [
        (start_time, index)
        for index, (op_type, _, start_time) in enumerate(kernels)
        if op_type == "ReduceMax"
    ]
    return min(candidates)[1] if candidates else None


def _is_noise_kernel(op_type: str, index: int, first_reducemax: int | None) -> bool:
    if op_type == "TensorMove":
        return True
    return op_type == "ReduceMax" and index == first_reducemax


def _split_noise_kernels(
    kernels: list[tuple[str, float, float]],
) -> tuple[dict[str, list[float]], float]:
    first_reducemax = _first_reducemax_index(kernels)
    noise_total = 0.0
    op_durs: dict[str, list[float]] = {}
    for index, (op_type, duration, _) in enumerate(kernels):
        if _is_noise_kernel(op_type, index, first_reducemax):
            noise_total += duration
            continue
        op_durs.setdefault(op_type, []).append(duration)
    return op_durs, noise_total


def _summarize_durations(
    op_durs: dict[str, list[float]],
) -> tuple[dict[str, dict[str, Any]], float]:
    per_op = {}
    for op_type, durations in op_durs.items():
        if durations:
            total = sum(durations)
            per_op[op_type] = {
                "mean": total / len(durations),
                "count": len(durations),
                "total": total,
            }
    return per_op, sum(stats["total"] for stats in per_op.values())


def _extract_e2e(
    kernels: list[tuple[str, float, float]],
) -> tuple[dict[str, dict[str, Any]], float, float]:
    """Extract per-op stats and E2E via two independent paths.

    Noise filtering:
    - TensorMove: all instances filtered (from noise injection's .npu() and golden's .to(device))
    - ReduceMax: only the first one (by Start Time) filtered (noise marker)

    Path A (subtraction): E2E = total_all - noise_total
    Path B (summation):   E2E = sum of non-noise kernel durations

    Returns:
        (per_op, e2e_a, e2e_b)
        - per_op: {op_type: {"mean": float, "count": int, "total": float}}
        - e2e_a: total - noise (subtraction path)
        - e2e_b: direct sum of non-noise kernels (summation path)
    """
    op_durs, noise_total = _split_noise_kernels(kernels)
    per_op, e2e_b = _summarize_durations(op_durs)
    e2e_a = sum(duration for _, duration, _ in kernels) - noise_total
    return per_op, e2e_a, e2e_b


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(v: float) -> str:
    if v < 1.0:
        return f"{v:.3f}us"
    if v < 10.0:
        return f"{v:.2f}us"
    return f"{v:.1f}us"


def _shape_str(specs: list[TensorSpec]) -> str:
    if len(specs) == 1:
        return str(specs[0].shape)
    return "{" + ", ".join(f"{s.name or '?'}: {s.shape}" for s in specs) + "}"


def _dtype_str(specs: list[TensorSpec]) -> str:
    if len(specs) == 1:
        return f"torch.{specs[0].dtype.replace('torch.', '')}"
    return (
        "{"
        + ", ".join(
            f"{s.name or '?'}: torch.{s.dtype.replace('torch.', '')}" for s in specs
        )
        + "}"
    )


def _report_header(cfg: ReportConfig, op_name: str) -> list[str]:
    return [
        f"# {op_name} Golden NPU Performance Report",
        "",
        f"- **Device**: {cfg.device}",
        f"- **Timestamp**: {datetime.datetime.now(tz=datetime.timezone.utc).isoformat()}",
        f"- **Iterations**: {cfg.iters}",
        "",
    ]


def _summary_lines(cases: list[CaseResult]) -> list[str]:
    lines = [
        "## Performance Summary",
        "",
        "| case | Input Shape | dtype | E2E (us) |",
        "|------|------------|-------|----------|",
    ]
    lines.extend(
        f"| {case.case_name} | {_shape_str(case.specs)} | "
        f"{_dtype_str(case.specs)} | {_fmt(case.e2e_a)} |"
        for case in cases
    )
    return lines + [""]


def _case_metadata(case: CaseResult, multi: bool) -> list[str]:
    lines = []
    if multi:
        lines.extend([f"## Case: {case.case_name}", ""])
    lines.extend(
        [
            f"- **Input Shape**: {_shape_str(case.specs)}",
            f"- **dtype**: {_dtype_str(case.specs)}",
            f"- **Profiling Data**: `prof/{case.prof_dir.name}/`",
            "",
        ]
    )
    return lines


def _case_performance(case: CaseResult) -> list[str]:
    lines = [
        "### E2E Performance",
        "",
        f"**Total kernel duration**: {_fmt(case.e2e_a)} (total - noise)",
        f"- Cross-check (Σ per-op total): {_fmt(case.e2e_b)}",
        "",
        "### Op Performance",
        "",
        "| op | count | mean_duration | total |",
        "|----|-------|--------------|-------|",
    ]
    for op_type in sorted(case.per_op, key=lambda key: -case.per_op[key]["total"]):
        stats = case.per_op[op_type]
        lines.append(
            f"| {op_type} | {stats['count']} | {_fmt(stats['mean'])} | "
            f"{_fmt(stats['total'])} |"
        )
    if not case.per_op:
        lines.append("| (no data) | — | — | — |")
    return lines + [""]


def _report_notes() -> list[str]:
    return [
        "## Notes",
        "",
        "- Data source: `ASCEND_PROFILER_OUTPUT/kernel_details.csv`",
        "- Noise filtered: TensorMove (all) + first ReduceMax (by Start Time)",
        "- Each iteration: noise injection (480MB randn + max) → golden call → sync",
    ]


def _write_report(cfg: ReportConfig) -> Path:
    path = cfg.output_dir / "GOLDEN_PERF_REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    op_name = cfg.golden_path.stem.replace("_golden", "")
    multi = len(cfg.cases) > 1
    lines = _report_header(cfg, op_name)
    if multi:
        lines.extend(_summary_lines(cfg.cases))
    for case in cfg.cases:
        lines.extend(_case_metadata(case, multi))
        lines.extend(_case_performance(case))
    lines.extend(_report_notes())
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Profiling entry
# ---------------------------------------------------------------------------


def _create_profiler(torch_npu, prof_dir: Path):
    exp_cfg_cls = getattr(torch_npu.profiler, "_ExperimentalConfig")
    exp_cfg = exp_cfg_cls(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    )
    return torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.NPU,
            torch_npu.profiler.ProfilerActivity.CPU,
        ],
        with_stack=False,
        record_shapes=False,
        profile_memory=True,
        experimental_config=exp_cfg,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            str(prof_dir), analyse_flag=True
        ),
    )


def _run_profile_iterations(torch, torch_npu, call, profiler, iters: int) -> None:
    with profiler:
        for iteration in range(iters):
            noise = torch.randn(NOISE_SIZE).to(torch.float32).npu()
            _ = torch.max(noise)
            try:
                call()
            except Exception as exc:
                raise RuntimeError(
                    f"Golden function crashed on iteration {iteration}. "
                    "Random tensor values may violate semantic constraints "
                    "(e.g. block table indices, sequence lengths). "
                    "Fix: (1) add _make_inputs(device) to the golden file "
                    "and use --factory _make_inputs, or "
                    "(2) use --input NAME:SHAPE[:DTYPE] and --arg NAME=VALUE "
                    f"to provide valid inputs. Original error: {exc}"
                ) from exc
            torch_npu.npu.synchronize()
            profiler.step()


def _collect_profile_stats(prof_dir: Path) -> tuple[dict, float, float]:
    ascend = _find_ascend_output(prof_dir)
    details_path = ascend / "kernel_details.csv" if ascend else None
    if not (details_path and details_path.exists()):
        logger.warning("kernel_details.csv not found — report will have no data")
        return {}, 0.0, 0.0

    kernels = _read_kernel_details(details_path)
    per_op, e2e_a, e2e_b = _extract_e2e(kernels)
    logger.info(
        "%s kernels, %s op types after noise filter, E2E_A=%s, E2E_B=%s",
        len(kernels),
        len(per_op),
        _fmt(e2e_a),
        _fmt(e2e_b),
    )
    if abs(e2e_a - e2e_b) > 0.01:
        logger.warning("E2E mismatch: path_A=%s, path_B=%s", _fmt(e2e_a), _fmt(e2e_b))
    return per_op, e2e_a, e2e_b


def _profile_one_case(
    torch, torch_npu, call, prof_dir: Path, iters: int
) -> tuple[dict, float, float]:
    """Profile a single case and return (per_op, e2e_a, e2e_b)."""
    profiler = _create_profiler(torch_npu, prof_dir)
    _run_profile_iterations(torch, torch_npu, call, profiler, iters)
    logger.info("Profiling output: %s", prof_dir)
    return _collect_profile_stats(prof_dir)


def _profile_runtime():
    try:
        torch = import_module("torch")
        torch_npu = import_module("torch_npu")
    except ImportError as exc:
        raise RuntimeError(
            "torch_npu not installed. Run: pip install torch_npu"
        ) from exc
    return torch, torch_npu


def _profile_case(
    runtime: ProfileRuntime,
    case_name: str,
    args_list: list,
    kwargs: dict,
    prof_dir: Path,
) -> CaseResult:
    call, specs = _build_call_from_args(
        runtime.torch,
        runtime.fn,
        args_list,
        kwargs,
    )
    logger.info(
        "Case %s: %s tensor args, shapes=%s",
        case_name,
        len(specs),
        [spec.shape for spec in specs],
    )
    prof_dir.mkdir(parents=True, exist_ok=True)
    per_op, e2e_a, e2e_b = _profile_one_case(
        runtime.torch,
        runtime.torch_npu,
        call,
        prof_dir,
        runtime.config.iters,
    )
    return CaseResult(case_name, specs, per_op, e2e_a, e2e_b, prof_dir)


def _factory_cases(runtime: ProfileRuntime, mod, golden_path: Path) -> list[CaseResult]:
    factory_fn = getattr(mod, runtime.config.factory_name, None)
    if factory_fn is None or not callable(factory_fn):
        raise RuntimeError(
            f"Factory function {runtime.config.factory_name!r} not found in {golden_path}. "
            "Add a _make_inputs(device) function that returns "
            "(args_list, kwargs_dict) or "
            "[(case_name, args_list, kwargs_dict), ...]."
        )
    normalized = _normalize_factory_result(factory_fn(runtime.device), runtime.fn)
    multi = len(normalized) > 1
    return [
        _profile_case(
            runtime,
            case_name,
            args_list,
            kwargs,
            runtime.base_prof_dir / case_name if multi else runtime.base_prof_dir,
        )
        for case_name, args_list, kwargs in normalized
    ]


def _default_report_specs(fn: Callable) -> list[TensorSpec]:
    required = [
        TensorSpec(param.name, DEFAULT_SHAPE, DEFAULT_DTYPE)
        for param in inspect.signature(fn).parameters.values()
        if param.default is inspect.Parameter.empty and param.kind in _REQUIRED_KINDS
    ]
    return required[:1] or [TensorSpec("x", DEFAULT_SHAPE, DEFAULT_DTYPE)]


def _direct_case(runtime: ProfileRuntime) -> CaseResult:
    cfg = runtime.config
    call = _build_call(
        runtime.torch,
        runtime.fn,
        cfg.specs,
        cfg.scalar_args,
        runtime.device,
    )
    per_op, e2e_a, e2e_b = _profile_one_case(
        runtime.torch,
        runtime.torch_npu,
        call,
        runtime.base_prof_dir,
        cfg.iters,
    )
    case_name = cfg.fn_name or getattr(runtime.fn, "__name__", "golden")
    return CaseResult(
        case_name,
        cfg.specs or _default_report_specs(runtime.fn),
        per_op,
        e2e_a,
        e2e_b,
        runtime.base_prof_dir,
    )


def profile_golden(cfg: ProfileConfig) -> int:
    torch, torch_npu = _profile_runtime()

    if not (torch.npu.is_available() and torch.npu.device_count() > 0):
        logger.info("No NPU hardware — profiling skipped.")
        return 0

    golden_path = cfg.golden_path.resolve()
    out = (cfg.output_dir or golden_path.parent).resolve()
    base_prof_dir = out / "prof" / golden_path.stem
    base_prof_dir.mkdir(parents=True, exist_ok=True)

    mod = _load_module(golden_path)
    fn = _find_fn(mod, golden_path, cfg.fn_name)
    device = (
        torch.device(f"npu:{cfg.device_id}")
        if cfg.device_id is not None
        else torch.device("npu")
    )
    runtime = ProfileRuntime(torch, torch_npu, fn, cfg, base_prof_dir, device)
    if cfg.factory_name:
        cases = _factory_cases(runtime, mod, golden_path)
    else:
        cases = [_direct_case(runtime)]

    report = _write_report(
        ReportConfig(
            golden_path=golden_path,
            output_dir=out,
            device=device,
            iters=cfg.iters,
            cases=cases,
        )
    )
    logger.info("Report: %s", report)
    return 0


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _ok(condition: bool, message: str = "") -> None:
    if not condition:
        raise AssertionError(message)


def _test_kernel_extraction(base: Path) -> None:
    details = base / "kernel_details.csv"
    details.write_text(
        "Device_id,Name,Type,Start Time(us),Duration(us)\n"
        "0,aclnnMul,Mul,1000.0,5.0\n"
        "0,aclnnReduceMax,ReduceMax,1050.0,100.0\n"
        "0,aclnnMul,Mul,2000.0,4.0\n"
        "0,aclnnReduceMax,ReduceMax,2050.0,95.0\n"
        "0,aclnnAdd,Add,3000.0,3.0\n"
        "0,aclnnAdd,Add,3100.0,3.5\n"
    )
    kernels = _read_kernel_details(details)
    _ok(len(kernels) == 6, f"expected 6 kernels, got {len(kernels)}")
    per_op, e2e_a, e2e_b = _extract_e2e(kernels)
    _ok(per_op["ReduceMax"]["count"] == 1, "golden ReduceMax was not preserved")
    _ok(abs(per_op["ReduceMax"]["mean"] - 95.0) < 0.01)
    _ok(per_op["Mul"]["count"] == 2 and abs(per_op["Mul"]["mean"] - 4.5) < 0.01)
    _ok(per_op["Add"]["count"] == 2 and abs(per_op["Add"]["mean"] - 3.25) < 0.01)
    _ok(abs(e2e_a - 110.5) < 0.01 and abs(e2e_b - 110.5) < 0.01)


def _test_tensor_move_filtering(base: Path) -> None:
    details = base / "kernel_details2.csv"
    details.write_text(
        "Device_id,Name,Type,Start Time(us),Duration(us)\n"
        "0,aclnnTensorMove,TensorMove,1000.0,30.0\n"
        "0,aclnnReduceMax,ReduceMax,1050.0,1400.0\n"
        "0,aclnnTensorMove,TensorMove,1100.0,20.0\n"
        "0,aclnnMul,Mul,2000.0,5.0\n"
        "0,aclnnAdd,Add,3000.0,3.0\n"
    )
    per_op, e2e_a, e2e_b = _extract_e2e(_read_kernel_details(details))
    _ok("TensorMove" not in per_op and "ReduceMax" not in per_op)
    _ok("Mul" in per_op and "Add" in per_op, "golden ops should be preserved")
    _ok(abs(e2e_a - 8.0) < 0.01 and abs(e2e_b - 8.0) < 0.01)


def _test_parsers_and_tensors() -> None:
    torch = import_module("torch")
    spec = _parse_tensor_spec("x:2x3:float32")
    _ok(spec == TensorSpec("x", (2, 3), "float32"), f"spec: {spec}")
    _ok(_parse_shape("(2, 3)") == (2, 3))
    _ok(_parse_arg("eps=1e-5") == ("eps", 1e-5))
    int_tensor = _make_tensor(
        torch, TensorSpec("i", (2, 3), "int64"), torch.device("cpu")
    )
    bool_tensor = _make_tensor(
        torch, TensorSpec("b", (2, 3), "bool"), torch.device("cpu")
    )
    _ok(int_tensor.dtype == torch.int64)
    _ok(bool_tensor.dtype == torch.bool)


def _single_report_case(base: Path) -> CaseResult:
    return CaseResult(
        case_name="sample",
        specs=[TensorSpec("x", (8, 1024, 4096), "float32")],
        per_op={
            "Mul": {"mean": 4.5, "count": 2, "total": 9.0},
            "Add": {"mean": 3.25, "count": 2, "total": 6.5},
        },
        e2e_a=15.5,
        e2e_b=15.5,
        prof_dir=base / "prof" / "sample",
    )


def _test_single_report(base: Path) -> None:
    report = _write_report(
        ReportConfig(
            base / "sample_golden.py",
            base,
            "npu",
            1,
            [_single_report_case(base)],
        )
    )
    content = report.read_text()
    for expected in (
        "Op Performance",
        "E2E Performance",
        "Mul",
        "Add",
        "mean_duration",
        "count",
        "total",
        "15.5",
        "Total kernel duration",
        "Cross-check",
    ):
        _ok(expected in content, f"{expected!r} missing from report")


def _test_multi_report(base: Path) -> None:
    cases = [
        CaseResult(
            "perf_p0_small",
            [TensorSpec("x", (8, 1024), "bfloat16")],
            {"Mul": {"mean": 2.0, "count": 1, "total": 2.0}},
            2.0,
            2.0,
            base / "prof" / "case_a",
        ),
        CaseResult(
            "perf_p0_large",
            [TensorSpec("x", (16, 2048), "bfloat16")],
            {"Mul": {"mean": 8.0, "count": 1, "total": 8.0}},
            8.0,
            8.0,
            base / "prof" / "case_b",
        ),
    ]
    content = _write_report(
        ReportConfig(
            base / "multi_golden.py",
            base,
            "npu",
            1,
            cases,
        )
    ).read_text()
    for expected in (
        "Performance Summary",
        "perf_p0_small",
        "perf_p0_large",
        "Case: perf_p0_small",
        "Case: perf_p0_large",
    ):
        _ok(expected in content, f"{expected!r} missing from multi-case report")


def _test_factory_normalization() -> None:
    def dummy_fn(value):
        return value

    single = _normalize_factory_result(([1], {}), dummy_fn)
    multi = _normalize_factory_result(
        [("a", [1], {}), ("b", [2], {"e": 1})],
        dummy_fn,
    )
    _ok(len(single) == 1 and single[0][0] == "dummy_fn")
    _ok(len(multi) == 2 and multi[0][0] == "a" and multi[1][0] == "b")


def _self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _test_kernel_extraction(base)
        _test_tensor_move_filtering(base)
        _test_parsers_and_tensors()
        _test_single_report(base)
        _test_multi_report(base)
        _test_factory_normalization()
    logger.info("self-test ok")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """The CLI: one golden file, its function, and how to feed it."""
    p = argparse.ArgumentParser(
        description="Profile a golden script on NPU and write GOLDEN_PERF_REPORT.md."
    )
    p.add_argument("golden", nargs="?", help="Path to *_golden.py")
    p.add_argument("--function", help="Golden function name")
    p.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="NAME:SHAPE[:DTYPE]",
        help="Tensor input spec (repeatable)",
    )
    p.add_argument(
        "--arg",
        action="append",
        default=[],
        metavar="NAME=JSON",
        help="Non-tensor argument (repeatable)",
    )
    p.add_argument(
        "--factory",
        metavar="FUNC_NAME",
        help="Name of a factory function in the golden module that "
        "returns (args_list, kwargs_dict) for single case, or "
        "[(case_name, args_list, kwargs_dict), ...] for multi-case. "
        "Use for operators with semantic constraints on tensor "
        "values (block tables, state caches, index tensors, etc.) "
        "where random values would crash. "
        "When set, --input and --arg are ignored.",
    )
    p.add_argument(
        "--output-dir", type=Path, help="Output directory (default: golden file dir)"
    )
    p.add_argument("--device", type=int, default=None, help="NPU device ID")
    p.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    p = _build_parser()
    args = p.parse_args(argv)

    if args.self_test:
        _self_test()
        return 0
    if not args.golden:
        p.error("golden path is required")
    if args.iters <= 0:
        p.error("--iters must be > 0")

    try:
        specs = [_parse_tensor_spec(t) for t in args.input]
        scalars = dict(_parse_arg(a) for a in args.arg)
        return profile_golden(
            ProfileConfig(
                golden_path=Path(args.golden),
                fn_name=args.function,
                specs=specs,
                scalar_args=scalars,
                output_dir=args.output_dir,
                iters=args.iters,
                device_id=args.device,
                factory_name=args.factory,
            )
        )
    except Exception as exc:
        logger.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
