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
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import environment_context as envctx
import runtime_guard as rtguard

from collect_validate import (
    DEFAULT_CSV_SUITE_NAMES, STANDARD_CSV_NAMES, bin_contains_all, bin_contains_any, csv_artifact_map,
    detect_simt_evidence_from_bin, extract_log_diagnostics, fallback_allowed, has_standard_csv_suite,
    infer_semantic_failure_reason, inspect_source_payload, metric_expression_contains,
    metric_expression_key, normalize_metric_token, rel, split_metric_expression, trace_event_count,
    validate_block_entry, validate_profiler_options, _TRACE_EVENT_PH, _block_log_text,
)
from collect_discovery import (
    KNOWN_METRICS, TEST_LIKE_EXECUTABLE,
    detect_kernel_scale_instrumentation, detect_relative_resource_hints, discover_executable_candidates,
    discover_operators, find_latest_file, find_latest_result_dir, find_result_dirs,
    is_candidate_executable, parse_cmake_cache, parse_supported_metrics, parse_supported_options,
    resolve_app_cwd, resolve_executable, resolve_executable_details, run_capture, run_shell_capture,
    test_like_executable_warning, _candidate_score, _debug_scope_root, _row_value_case_insensitive,
    BLOCK_DIRS, DEFAULT_BLOCK_TIMEOUTS, derive_timeout_profile, parse_block_timeouts, parse_key_value,
    ensure_msprof_output_permissions, fingerprint, fingerprint_key, is_single_exact_kernel_filter,
    now_iso, previous_run_allows_reuse, read_json, sanitize_output_tree, shell_join, write_json,
    BLOCKS, CONTRACT_SCHEMA, FEATURE_TO_BLOCK, FEATURE_TO_BLOCKS, PRESETS, RAW_COUNTER_METRICS,
    command_description, feature_catalog, normalize_feature_name, print_feature_explanation,
    print_feature_list, visualization_contract,
)
import collect_discovery as _collect_discovery

cli_logger = logging.getLogger(__name__ + ".cli")
cli_logger.propagate = False
if not cli_logger.handlers:
    _cli_handler = logging.StreamHandler(sys.stdout)
    _cli_handler.setFormatter(logging.Formatter("%(message)s"))
    cli_logger.addHandler(_cli_handler)


def detect_debug_symbols_details(operator_root, app):
    """Wrapper so this module's ``run_capture`` stays the effective patch point."""
    return _collect_discovery.detect_debug_symbols_details(operator_root, app, capture=run_capture)

SKILL_ROOT = Path(__file__).resolve().parents[1]
VERSION = next(
    (line.strip() for line in (SKILL_ROOT / "VERSION").read_text(encoding="utf-8").splitlines()
     if line.strip() and not line.lstrip().startswith("#")),
    "unknown",
)


def select_metric_candidates(block_id: str, supported: Iterable[str]) -> List[str]:
    lookup: Dict[str, str] = {}
    for actual in supported:
        lookup.setdefault(normalize_metric_token(actual), actual)
    out: List[str] = []
    seen: Set[Tuple[str, ...]] = set()
    for candidate_expression in BLOCKS[block_id]["metric_candidates"]:
        candidate_tokens = split_metric_expression(candidate_expression)
        actual_tokens: List[str] = []
        for candidate in candidate_tokens:
            actual = lookup.get(normalize_metric_token(candidate))
            if actual is None:
                actual_tokens = []
                break
            actual_tokens.append(actual)
        if not actual_tokens:
            continue
        expression = ",".join(actual_tokens)
        key = metric_expression_key(expression)
        if key not in seen:
            seen.add(key)
            out.append(expression)
    return out


def select_metric(block_id: str, supported: Iterable[str]) -> Optional[str]:
    candidates = select_metric_candidates(block_id, supported)
    return candidates[0] if candidates else None


def _check_required_options(options: Set[str]) -> None:
    missing = [x for x in ["--aic-metrics", "--output"] if x not in options]
    if missing:
        raise rtguard.UsageError(
            f"ERROR: installed msprof op help does not expose required option(s): {', '.join(missing)}"
        )


def _normalize_metric_expression(metric: str) -> str:
    metric_tokens = split_metric_expression(metric)
    if not metric_tokens:
        raise rtguard.UsageError("ERROR: --aic-metrics expression cannot be empty.")
    return ",".join(metric_tokens)


def _require_option(options: Set[str], cli_name: str) -> None:
    if cli_name not in options:
        raise rtguard.UsageError(
            f"ERROR: {cli_name} was requested but is not exposed by the installed msprof op CLI."
        )


_PROFILER_OPTION_ORDER = [
    ("launch_skip_before_match", "--launch-skip-before-match"),
    ("replay_mode", "--replay-mode"),
    ("warm_up", "--warm-up"),
    ("kill", "--kill"),
    ("mstx", "--mstx"),
    ("mstx_include", "--mstx-include"),
    ("dump", "--dump"),
    ("core_id", "--core-id"),
]


def _append_profiler_options(
    cmd: List[str],
    metric: str,
    options: Set[str],
    profiler_options: Optional[Mapping[str, Any]],
) -> None:
    extra = dict(profiler_options or {})
    if not metric_expression_contains(metric, "TimelineDetail"):
        extra["dump"] = None
        extra["core_id"] = None
    validate_profiler_options(metric, extra)
    for key, cli_name in _PROFILER_OPTION_ORDER:
        value = extra.get(key)
        if value is None:
            continue
        _require_option(options, cli_name)
        cmd.append(f"{cli_name}={value}")


class MsprofCommandSpec(NamedTuple):
    """Bundled inputs for ``build_msprof_command`` (G.FNM.03)."""

    msprof: str
    metric: str
    run_dir: Path
    app: Path
    app_args: Sequence[str]
    kernel_name: Optional[str]
    launch_count: int
    instr_timeline_pipe: Optional[str] = None
    supported_options: Optional[Iterable[str]] = None
    profiler_options: Optional[Mapping[str, Any]] = None


def build_msprof_command(spec: MsprofCommandSpec) -> List[str]:
    options = set(spec.supported_options or [
        "--aic-metrics", "--output", "--launch-count", "--kernel-name", "--instr-timeline-pipe"
    ])
    _check_required_options(options)
    metric = _normalize_metric_expression(spec.metric)
    cmd = [spec.msprof, "op", f"--aic-metrics={metric}", f"--output={spec.run_dir}"]
    if "--launch-count" in options:
        cmd.append(f"--launch-count={spec.launch_count}")
    if spec.kernel_name:
        _require_option(options, "--kernel-name")
        cmd.append(f"--kernel-name={spec.kernel_name}")
    if metric_expression_contains(metric, "instrTimeLine") and spec.instr_timeline_pipe:
        if "--instr-timeline-pipe" in options:
            cmd.append(f"--instr-timeline-pipe={spec.instr_timeline_pipe}")
    _append_profiler_options(cmd, metric, options, spec.profiler_options)
    cmd.append(str(spec.app))
    cmd.extend(spec.app_args)
    return cmd


def build_artifacts(result_dir: Optional[Path], output_root: Path) -> Dict[str, Any]:
    visualize = find_latest_file(result_dir, "visualize_data.bin")
    trace = find_latest_file(result_dir, "trace.json")
    return {
        "visualize_data": rel(visualize, output_root),
        "trace": rel(trace, output_root),
        "csv": csv_artifact_map(result_dir, output_root),
    }


def classify_status(return_code: int, result_dir: Optional[Path]) -> str:
    # A timeout or non-zero command is never considered successful merely because
    # it left a partial OPPROF directory behind.
    if return_code != 0:
        return "failed"
    return "ok" if result_dir is not None else "empty"


def write_command_record(
    internal_dir: Path,
    block_id: str,
    metric: str,
    cmd: Sequence[str],
    description: Dict[str, Any],
) -> None:
    write_json(internal_dir / "commands" / f"{block_id}.json", {
        "block_id": block_id,
        "metric": metric,
        "argv": [str(x) for x in cmd],
        "shell": shell_join(cmd),
        "description": description,
    })


class _BlockSpec(NamedTuple):
    """Bundled identity of one collector block execution (G.FNM.03)."""

    block_id: str
    metric: str
    relative_dir: str
    requested_by: Sequence[str]


class BlockEntrySpec(NamedTuple):
    """Bundled inputs for ``build_block_entry`` (G.FNM.03)."""

    block_id: str
    metric: Optional[str]
    status: str
    relative_dir: str
    result_dir: Optional[Path]
    output_root: Path
    elapsed: float
    return_code: Optional[int]
    requested_by: Sequence[str]
    reason: Optional[str] = None


def _block_fingerprint_key(
    metric: str,
    cmd: Sequence[str],
    app_fp: Dict[str, Any],
    msprof_fp: Optional[Dict[str, Any]],
    cwd: Optional[Path],
) -> str:
    return fingerprint_key({
        "metric": metric,
        "command": list(cmd),
        "app": app_fp,
        "msprof": msprof_fp,
        "collector": VERSION,
        "cwd": str(cwd) if cwd else None,
    })


def _reuse_match(old: Mapping[str, Any], key: str, reuse_existing: bool, existing: Optional[Path]) -> bool:
    state_matches = (
        reuse_existing
        and old.get("fingerprint") == key
        and old.get("reusable") is True
    )
    previous_status_ok = (old.get("entry") or {}).get("status") in {"ok", "partial"}
    return bool(state_matches and previous_status_ok and existing)


def _reused_block_entry(spec: _BlockSpec, existing: Path, output_root: Path, cwd: Optional[Path]) -> Dict[str, Any]:
    entry = build_block_entry(BlockEntrySpec(
        spec.block_id, spec.metric, "reused", spec.relative_dir, existing, output_root,
        0.0, None, spec.requested_by,
    ))
    entry["cwd"] = str(cwd) if cwd else None
    entry["reused_from_state"] = True
    return entry


def _planned_block_entry(spec: _BlockSpec, output_root: Path, timeout: int, cwd: Optional[Path]) -> Dict[str, Any]:
    entry = build_block_entry(BlockEntrySpec(
        spec.block_id, spec.metric, "planned", spec.relative_dir, None, output_root,
        0.0, None, spec.requested_by,
    ))
    entry["cwd"] = str(cwd) if cwd else None
    entry["timeout_seconds"] = timeout
    return entry


def _execution_fields(
    result: Mapping[str, Any],
    logs: Path,
    stem: str,
    output_root: Path,
    timeout: int,
) -> Dict[str, Any]:
    stdout_log = logs / f"{stem}.stdout.log"
    stderr_log = logs / f"{stem}.stderr.log"
    return {
        "timed_out": bool(result.get("timed_out")),
        "termination": result.get("termination") or {},
        "heartbeat_count": int(result.get("heartbeat_count") or 0),
        "pid": result.get("pid"),
        "cwd": result.get("cwd"),
        "timeout_seconds": timeout,
        "logs": {
            "stdout": rel(stdout_log, output_root),
            "stderr": rel(stderr_log, output_root),
        },
        "diagnostic_warnings": extract_log_diagnostics(stderr_log),
    }


def _write_block_state(
    state_path: Path,
    key: str,
    started_at: str,
    reusable: bool,
    entry: Dict[str, Any],
) -> None:
    write_json(state_path, {
        "fingerprint": key,
        "started_at": started_at,
        "finished_at": now_iso(),
        "reusable": reusable,
        "entry": entry,
    })


def _prepare_fresh_run_dir(run_dir: Path) -> None:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ensure_msprof_output_permissions(run_dir)


class _CompletedBlockSpec(NamedTuple):
    """Bundled inputs for ``_completed_block_entry`` (G.FNM.03)."""

    spec: _BlockSpec
    output_root: Path
    result: Mapping[str, Any]
    logs: Path
    stem: str
    timeout: int


def _completed_block_entry(run: _CompletedBlockSpec) -> Tuple[Dict[str, Any], Optional[Path]]:
    spec = run.spec
    output_root = run.output_root
    result = run.result
    logs = run.logs
    stem = run.stem
    timeout = run.timeout
    result_dir = find_latest_result_dir(output_root / spec.relative_dir)
    status = classify_status(int(result["return_code"]), result_dir)
    entry_spec = BlockEntrySpec(
        spec.block_id, spec.metric, status, spec.relative_dir, result_dir, output_root,
        float(result["elapsed_seconds"]), int(result["return_code"]), spec.requested_by,
    )
    entry = build_block_entry(entry_spec)
    entry.update(_execution_fields(result, logs, stem, output_root, timeout))
    return entry, result_dir


class BlockRunSpec(NamedTuple):
    """Bundled inputs for ``execute_block`` (G.FNM.03)."""

    block_id: str
    metric: str
    cmd: Sequence[str]
    output_root: Path
    internal_dir: Path
    timeout: int
    reuse_existing: bool
    app_fp: Dict[str, Any]
    dry_run: bool
    requested_by: Sequence[str]
    log_stem: Optional[str] = None
    cwd: Optional[Path] = None
    heartbeat_seconds: int = 30
    msprof_fp: Optional[Dict[str, Any]] = None


def execute_block(run: BlockRunSpec) -> Dict[str, Any]:
    spec = _BlockSpec(run.block_id, run.metric, BLOCK_DIRS[run.block_id], run.requested_by)
    run_dir = run.output_root / spec.relative_dir
    state_path = run.internal_dir / "state" / f"{run.block_id}.json"
    key = _block_fingerprint_key(spec.metric, run.cmd, run.app_fp, run.msprof_fp, run.cwd)

    old = read_json(state_path, {})
    existing = find_latest_result_dir(run_dir)
    if _reuse_match(old, key, run.reuse_existing, existing):
        return _reused_block_entry(spec, existing, run.output_root, run.cwd)
    if run.dry_run:
        return _planned_block_entry(spec, run.output_root, run.timeout, run.cwd)

    _prepare_fresh_run_dir(run_dir)

    started_at = now_iso()
    logs = run.internal_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stem = run.log_stem or run.block_id
    result = rtguard.run_managed_process(rtguard.ManagedProcessSpec(
        run.cmd,
        timeout=run.timeout,
        cwd=run.cwd,
        heartbeat_seconds=run.heartbeat_seconds,
        stdout_log=logs / f"{stem}.stdout.log",
        stderr_log=logs / f"{stem}.stderr.log",
        progress_log=logs / f"{stem}.progress.log",
        heartbeat_label=f"{run.block_id}/{run.metric}",
    ))

    entry, result_dir = _completed_block_entry(_CompletedBlockSpec(
        spec, run.output_root, result, logs, stem, run.timeout,
    ))
    reusable = (
        entry["status"] in {"ok", "partial"}
        and entry.get("return_code") == 0
        and not entry.get("timed_out")
        and result_dir is not None
    )
    _write_block_state(state_path, key, started_at, reusable, entry)
    return entry


def _raw_suite_planned(spec: _BlockSpec, timeout: int, cwd: Optional[Path], cmd: Sequence[str]) -> Dict[str, Any]:
    return {
        "block_id": "raw_data",
        "display_name": BLOCKS["raw_data"]["display_name"],
        "metric": spec.metric,
        "status": "planned",
        "reason": "Default was unavailable; all exposed standard counters are planned in one command.",
        "requested_by": list(spec.requested_by),
        "relative_dir": spec.relative_dir,
        "result_dir": None,
        "artifacts": {"visualize_data": None, "trace": None, "csv": {}},
        "visual_features": list(BLOCKS["raw_data"]["visual_features"]),
        "elapsed_seconds": 0.0,
        "return_code": None,
        "attempts": [{"metric": spec.metric, "status": "planned", "command": shell_join(cmd)}],
        "timed_out": False,
        "cwd": str(cwd) if cwd else None,
        "timeout_seconds": timeout,
        "collection_passes": 1,
    }


def _raw_suite_status(result: Mapping[str, Any], csv_map: Mapping[str, str]) -> str:
    if result.get("timed_out") or result["return_code"] != 0:
        return "failed"
    required = {f"{name}.csv" for name in RAW_COUNTER_METRICS}
    present = set(csv_map)
    if required.issubset(present):
        return "ok"
    return "partial" if present else "empty"


class _RawSuiteCtx(NamedTuple):
    """Bundled context for the raw counter suite result builder (G.FNM.03)."""

    spec: _BlockSpec
    metrics: Sequence[str]
    output_root: Path
    timeout: int
    cwd: Optional[Path]
    result: Mapping[str, Any]


def _raw_suite_result(ctx: _RawSuiteCtx) -> Dict[str, Any]:
    spec = ctx.spec
    result = ctx.result
    result_dir = find_latest_result_dir(ctx.output_root / spec.relative_dir)
    csv_map = csv_artifact_map(result_dir, ctx.output_root) if result["return_code"] == 0 else {}
    status = _raw_suite_status(result, csv_map)
    return {
        "block_id": "raw_data",
        "display_name": BLOCKS["raw_data"]["display_name"],
        "metric": spec.metric,
        "status": status,
        "reason": "Default was unavailable; collected all exposed standard counters in one comma-separated command.",
        "requested_by": list(spec.requested_by),
        "relative_dir": spec.relative_dir,
        "result_dir": rel(result_dir, ctx.output_root),
        "artifacts": build_artifacts(result_dir, ctx.output_root),
        "visual_features": list(BLOCKS["raw_data"]["visual_features"]),
        "elapsed_seconds": round(float(result["elapsed_seconds"]), 3),
        "return_code": int(result["return_code"]),
        "attempts": [{
            "metric": spec.metric,
            "metrics": list(ctx.metrics),
            "status": status,
            "return_code": int(result["return_code"]),
            "elapsed_seconds": float(result["elapsed_seconds"]),
            "result_dir": rel(result_dir, ctx.output_root),
            "csv": csv_map,
            "timed_out": bool(result.get("timed_out")),
            "termination": result.get("termination") or {},
        }],
        "timed_out": bool(result.get("timed_out")),
        "cwd": str(ctx.cwd) if ctx.cwd else None,
        "timeout_seconds": ctx.timeout,
        "collection_passes": 1,
    }


def _write_raw_suite_record(
    internal_dir: Path,
    stem: str,
    spec: _BlockSpec,
    cmd: Sequence[str],
    metrics: Sequence[str],
) -> None:
    write_command_record(
        internal_dir, stem, spec.metric, cmd,
        {
            "purpose": "One-pass fallback raw counter bundle because Default is not exposed.",
            "metrics": list(metrics),
            "bundle_policy": "documented comma-separated standard counters",
        },
    )


class RawCounterSuiteSpec(NamedTuple):
    """Bundled inputs for ``execute_raw_counter_suite`` (G.FNM.03)."""

    metrics: Sequence[str]
    msprof: str
    output_root: Path
    internal_dir: Path
    app: Path
    app_args: Sequence[str]
    kernel_name: Optional[str]
    launch_count: int
    supported_options: Sequence[str]
    timeout: int
    dry_run: bool
    requested_by: Sequence[str]
    cwd: Optional[Path] = None
    heartbeat_seconds: int = 30
    profiler_options: Optional[Mapping[str, Any]] = None


def execute_raw_counter_suite(run: RawCounterSuiteSpec) -> Dict[str, Any]:
    """Collect all exposed standard counters in one documented comma bundle.

    msprof explicitly supports multiple standard counter metrics separated by commas.
    Splitting these counters into seven application replays is both redundant and less
    representative when the application has run-to-run variance.
    """
    spec = _BlockSpec("raw_data", ",".join(run.metrics), BLOCK_DIRS["raw_data"], run.requested_by)
    run_dir = run.output_root / spec.relative_dir
    _prepare_fresh_run_dir(run_dir)

    cmd = build_msprof_command(MsprofCommandSpec(
        run.msprof, spec.metric, run_dir, run.app, run.app_args, run.kernel_name,
        run.launch_count, supported_options=run.supported_options,
        profiler_options=run.profiler_options,
    ))
    stem = "raw_data_standard_counter_bundle"
    _write_raw_suite_record(run.internal_dir, stem, spec, cmd, run.metrics)

    if run.dry_run:
        return _raw_suite_planned(spec, run.timeout, run.cwd, cmd)

    logs = run.internal_dir / "logs"
    result = rtguard.run_managed_process(rtguard.ManagedProcessSpec(
        cmd,
        timeout=run.timeout,
        cwd=run.cwd,
        heartbeat_seconds=run.heartbeat_seconds,
        stdout_log=logs / f"{stem}.stdout.log",
        stderr_log=logs / f"{stem}.stderr.log",
        progress_log=logs / f"{stem}.progress.log",
        heartbeat_label="raw_data/standard-counter-bundle",
    ))
    ctx = _RawSuiteCtx(spec=spec, metrics=run.metrics, output_root=run.output_root, timeout=run.timeout,
        cwd=run.cwd, result=result)
    return _raw_suite_result(ctx)


def build_block_entry(entry: BlockEntrySpec) -> Dict[str, Any]:
    spec = BLOCKS[entry.block_id]
    return {
        "block_id": entry.block_id,
        "display_name": spec["display_name"],
        "metric": entry.metric,
        "status": entry.status,
        "reason": entry.reason,
        "requested_by": list(entry.requested_by),
        "relative_dir": entry.relative_dir,
        "result_dir": rel(entry.result_dir, entry.output_root),
        "artifacts": build_artifacts(entry.result_dir, entry.output_root),
        "visual_features": list(spec["visual_features"]),
        "elapsed_seconds": entry.elapsed,
        "return_code": entry.return_code,
    }


def unavailable_block_entry(block_id: str, requested_by: Sequence[str], reason: str) -> Dict[str, Any]:
    return build_block_entry(BlockEntrySpec(
        block_id=block_id,
        metric=None,
        status="unavailable",
        relative_dir=BLOCK_DIRS[block_id],
        result_dir=None,
        output_root=Path.cwd(),  # no artifact paths are produced
        elapsed=0.0,
        return_code=None,
        requested_by=requested_by,
        reason=reason,
    ))


def resolve_requested_features(preset: str, explicit_features: Sequence[str]) -> List[str]:
    if explicit_features:
        raw = [normalize_feature_name(x) for x in explicit_features]
    else:
        raw = list(PRESETS[preset])
    unknown = sorted(set(raw) - set(FEATURE_TO_BLOCK))
    if unknown:
        raise rtguard.UsageError(f"ERROR: unknown feature(s): {', '.join(unknown)}. Use --list-features.")
    out: List[str] = []
    seen: Set[str] = set()
    for feature in raw:
        if feature not in seen:
            seen.add(feature)
            out.append(feature)
    return out


def resolve_blocks(features: Sequence[str], source_stall: str = "auto") -> Tuple[List[str], Dict[str, List[str]]]:
    requested_by: Dict[str, List[str]] = {}
    order: List[str] = []
    for feature in features:
        for block_id in FEATURE_TO_BLOCKS[feature]:
            requested_by.setdefault(block_id, []).append(feature)
            if block_id not in order:
                order.append(block_id)
    # A targeted Source report may consume PCSampling as an optional enhancer.
    # This is a feature dependency, not an assumption about one operator.
    if "source" in features and source_stall != "off":
        requested_by.setdefault("warp_stall", []).append("source:optional-stall-augmentation")
        if "warp_stall" not in order:
            order.append("warp_stall")
    # Stable execution order is the fixed directory order, not user argument order.
    ordered = [b for b in BLOCK_DIRS if b != "discovery" and b in order]
    return ordered, requested_by


def _resolve_explicit_memory_info(operator_root: Path, explicit_path: Path) -> Tuple[Optional[Path], Optional[str]]:
    """Return (candidate, error_reason) for an explicitly supplied sidecar."""
    explicit = explicit_path.expanduser()
    if not explicit.is_absolute():
        explicit = operator_root / explicit
    explicit = explicit.resolve()
    if not explicit.is_file() or explicit.is_symlink():
        return None, f"Explicit memory_info.json is not a usable regular file: {explicit}"
    return explicit, None


def _discover_memory_info_candidates(operator_root: Path, output_root: Path) -> List[Path]:
    output_resolved = output_root.resolve()
    candidates: List[Path] = []
    for candidate in operator_root.rglob("memory_info.json"):
        try:
            resolved = candidate.resolve()
            resolved.relative_to(output_resolved)
            continue
        except ValueError:
            logger.debug("memory_info candidate outside output tree: %s", candidate)
        if candidate.is_file() and not candidate.is_symlink():
            candidates.append(candidate)
    candidates.sort(key=lambda x: (x.stat().st_mtime_ns, str(x)), reverse=True)
    return candidates


def _memory_info_selection_reason(candidate_count: int) -> Optional[str]:
    if candidate_count == 1:
        return None
    return (f"Selected newest of {candidate_count} "
        "memory_info.json files; pass --memory-info to bind an exact sidecar.")


def collect_onchip_memory_artifact(
    operator_root: Path,
    output_root: Path,
    requested_by: Sequence[str],
    explicit_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Copy the exact compiler-generated memory_info.json into the manifest contract."""
    block_id = "onchip_memory"
    block_dir = output_root / BLOCK_DIRS[block_id]
    block_dir.mkdir(parents=True, exist_ok=True)
    if explicit_path is not None:
        explicit, error = _resolve_explicit_memory_info(operator_root, explicit_path)
        if error is not None:
            return unavailable_block_entry(block_id, requested_by, error)
        candidates = [explicit]
        selection_reason: Optional[str] = "Used the explicitly supplied memory_info.json."
    else:
        candidates = _discover_memory_info_candidates(operator_root, output_root)
        if not candidates:
            return unavailable_block_entry(block_id, requested_by, ("No memory_info.json was found in the "
                "selected operator project/build tree. Generate the compiler sidecar before requesting On-Chip "
                "Memory."))
        selection_reason = _memory_info_selection_reason(len(candidates))
    source = candidates[0]
    target = block_dir / "memory_info.json"
    shutil.copy2(source, target)
    return {
        "block_id": block_id,
        "display_name": BLOCKS[block_id]["display_name"],
        "metric": None,
        "status": "ok",
        "reason": selection_reason,
        "requested_by": list(requested_by),
        "relative_dir": BLOCK_DIRS[block_id],
        "result_dir": BLOCK_DIRS[block_id],
        "artifacts": {"visualize_data": None, "trace": None, "csv": {}, "memory_info": rel(target, output_root)},
        "visual_features": list(BLOCKS[block_id]["visual_features"]),
        "elapsed_seconds": 0.0,
        "return_code": 0,
        "coverage": {"memory_info": True, "candidate_count": len(candidates)},
    }


def write_block_reference(output_root: Path, block_id: str, provider_block: str) -> None:
    block_dir = output_root / BLOCK_DIRS[block_id]
    block_dir.mkdir(parents=True, exist_ok=True)
    write_json(block_dir / "block_reference.json", {
        "schema": CONTRACT_SCHEMA,
        "block_id": block_id,
        "provider_block": provider_block,
        "reason": ("The provider block already produced the required artifact set; no duplicate msprof replay "
            "was executed."),
    })


def alias_block_entry(
    block_id: str,
    provider: Dict[str, Any],
    requested_by: Sequence[str],
    output_root: Path,
) -> Dict[str, Any]:
    write_block_reference(output_root, block_id, provider["block_id"])
    spec = BLOCKS[block_id]
    return {
        "block_id": block_id,
        "display_name": spec["display_name"],
        "metric": provider.get("metric"),
        "status": "aliased",
        "reason": f"Reused artifacts from block '{provider['block_id']}'.",
        "requested_by": list(requested_by),
        "relative_dir": BLOCK_DIRS[block_id],
        "result_dir": provider.get("result_dir"),
        "provider_block": provider["block_id"],
        "artifacts": provider.get("artifacts", {"visualize_data": None, "trace": None, "csv": {}}),
        "visual_features": list(spec["visual_features"]),
        "elapsed_seconds": 0.0,
        "return_code": None,
    }


def semantic_alias_candidate(
    block_id: str,
    provider: Dict[str, Any],
    requested_by: Sequence[str],
    output_root: Path,
) -> Optional[Dict[str, Any]]:
    """Return a validated cross-metric alias when provider artifacts satisfy block semantics.

    This is intentionally stricter than matching metric names. For example, a
    Roofline replay already carries the Default CSV suite and may also contain
    the compute/memory structures needed by the Memory Detail renderer. When
    that semantic validation succeeds, a separate Default replay is redundant.
    """
    if provider.get("status") not in {"ok", "reused", "partial", "aliased"}:
        return None
    alias = alias_block_entry(block_id, provider, requested_by, output_root)
    alias = validate_block_entry(block_id, alias, output_root)
    if alias.get("status") not in {"ok", "reused", "partial", "aliased"}:
        return None
    alias["semantic_alias"] = True
    alias["collection_passes_saved"] = 1
    alias["reason"] = (
        f"Reused semantically complete artifacts from block '{provider['block_id']}' "
        f"({provider.get('metric')}); no redundant {block_id} replay was executed."
    )
    return alias


def _preflight_age(previous: Mapping[str, Any], preflight_path: Path) -> float:
    created_at = previous.get("created_at")
    try:
        return max(0.0, time.time() - float(created_at))
    except (TypeError, ValueError):
        return max(0.0, time.time() - preflight_path.stat().st_mtime)


def _cache_identity_match(
    previous: Mapping[str, Any],
    expected_command: str,
    app_fp: Dict[str, Any],
    cwd: Path,
) -> bool:
    return (
        previous.get("status") in {"ok", "reused"}
        and previous.get("command") == expected_command
        and previous.get("cwd") == str(cwd)
        and previous.get("app_fingerprint") == app_fp
    )


def _cache_payload_fresh(
    previous_result: Path,
    previous_csv: Path,
    age_seconds: float,
    cache_seconds: int,
) -> bool:
    return (
        age_seconds <= cache_seconds
        and previous_result.is_dir()
        and previous_csv.is_file()
        and previous_csv.stat().st_size > 0
    )


class _ReusePreflightSpec(NamedTuple):
    """Bundled inputs for ``_try_reuse_preflight`` (G.FNM.03)."""

    internal_dir: Path
    output_root: Path
    expected_command: str
    app_fp: Dict[str, Any]
    cwd: Path
    cache_seconds: int


def _try_reuse_preflight(spec: _ReusePreflightSpec) -> Optional[Dict[str, Any]]:
    internal_dir = spec.internal_dir
    output_root = spec.output_root
    expected_command = spec.expected_command
    app_fp = spec.app_fp
    cwd = spec.cwd
    cache_seconds = spec.cache_seconds
    preflight_path = internal_dir / "preflight.json"
    if cache_seconds <= 0 or not preflight_path.is_file():
        return None
    previous = read_json(preflight_path, {})
    # Age comes from the payload creation time, not the file mtime: rewrites
    # or unrelated touches of preflight.json must not extend the TTL forever.
    age_seconds = _preflight_age(previous, preflight_path)
    previous_result = output_root / str(previous.get("result_dir") or "")
    previous_csv = output_root / str(previous.get("op_basic_info") or "")
    identity_ok = _cache_identity_match(previous, expected_command, app_fp, cwd)
    payload_ok = _cache_payload_fresh(previous_result, previous_csv, age_seconds, cache_seconds)
    if not (identity_ok and payload_ok):
        return None
    original_elapsed = float(previous.get("original_elapsed_seconds") or previous.get("elapsed_seconds") or 0.0)
    reused = dict(previous)
    reused.update({
        "status": "reused",
        "reason": f"Reused successful preflight from {age_seconds:.1f}s ago (TTL {cache_seconds}s).",
        "elapsed_seconds": 0.0,
        "original_elapsed_seconds": original_elapsed,
        "cache_age_seconds": round(age_seconds, 3),
        "cache_seconds": cache_seconds,
    })
    # Do not rewrite preflight.json on reuse; the on-disk payload keeps
    # its original created_at so the cache actually expires.
    return reused


def _planned_preflight_result(internal_dir: Path, cmd: Sequence[str], cwd: Path) -> Dict[str, Any]:
    result = {
        "status": "planned",
        "return_code": None,
        "timed_out": False,
        "elapsed_seconds": 0.0,
        "command": shell_join(cmd),
        "cwd": str(cwd),
    }
    write_json(internal_dir / "preflight.json", result)
    return result


def _canary_failure_reason(
    proc: Mapping[str, Any],
    basic_csv: Optional[Path],
    combined_output: str,
    timeout: int,
) -> Optional[str]:
    if proc.get("timed_out"):
        return f"BasicInfo canary exceeded {timeout}s; collection was circuit-broken before formal feature runs."
    if proc["return_code"] != 0:
        return f"BasicInfo canary exited with return code {proc['return_code']}."
    if basic_csv is not None:
        return None
    reason = "BasicInfo canary returned zero but produced no non-empty OpBasicInfo.csv."
    if "writable by any other users or group users" in combined_output:
        reason += (
            " msprof rejected the output directory as group/other-writable; "
            "strip group/other write bits (chmod -R go-w) on the output tree and rerun."
        )
    return reason


class _CanaryCtx(NamedTuple):
    """Bundled context for the preflight canary execution (G.FNM.03)."""

    internal_dir: Path
    output_root: Path
    cmd: Sequence[str]
    timeout: int
    cwd: Path
    heartbeat_seconds: int
    app_fp: Dict[str, Any]
    cache_seconds: int


def _execute_canary(ctx: _CanaryCtx) -> Dict[str, Any]:
    logs = ctx.internal_dir / "logs"
    proc = rtguard.run_managed_process(rtguard.ManagedProcessSpec(
        ctx.cmd,
        timeout=ctx.timeout,
        cwd=ctx.cwd,
        heartbeat_seconds=ctx.heartbeat_seconds,
        stdout_log=logs / "preflight_canary.stdout.log",
        stderr_log=logs / "preflight_canary.stderr.log",
        progress_log=logs / "preflight_canary.progress.log",
        heartbeat_label="preflight/BasicInfo",
    ))
    preflight_root = ctx.internal_dir / "preflight_output"
    result_dir = find_latest_result_dir(preflight_root)
    basic_csv = find_latest_file(result_dir, "OpBasicInfo.csv")
    status = "ok" if proc["return_code"] == 0 and basic_csv is not None else "failed"
    combined_output = f"{proc.get('stdout', '')}\n{proc.get('stderr', '')}"
    reason = _canary_failure_reason(proc, basic_csv, combined_output, ctx.timeout)
    result = {
        "status": status,
        "reason": reason,
        "return_code": proc["return_code"],
        "timed_out": bool(proc.get("timed_out")),
        "elapsed_seconds": proc["elapsed_seconds"],
        "pid": proc.get("pid"),
        "termination": proc.get("termination") or {},
        "heartbeat_count": proc.get("heartbeat_count") or 0,
        "command": shell_join(ctx.cmd),
        "cwd": str(ctx.cwd),
        "created_at": time.time(),
        "app_fingerprint": ctx.app_fp,
        "finished_at": now_iso(),
        "cache_seconds": ctx.cache_seconds,
        "result_dir": rel(result_dir, ctx.output_root),
        "op_basic_info": rel(basic_csv, ctx.output_root),
    }
    if status == "failed":
        excerpt_lines = [line for line in combined_output.strip().splitlines() if line.strip()]
        result["log_excerpt"] = "\n".join(excerpt_lines[-20:])
    write_json(ctx.internal_dir / "preflight.json", result)
    return result


class PreflightCanarySpec(NamedTuple):
    """Bundled inputs for ``run_preflight_canary`` (G.FNM.03)."""

    msprof: str
    metric: str
    output_root: Path
    internal_dir: Path
    app: Path
    app_args: Sequence[str]
    kernel_name: Optional[str]
    supported_options: Sequence[str]
    timeout: int
    cwd: Path
    heartbeat_seconds: int
    dry_run: bool
    profiler_options: Optional[Mapping[str, Any]] = None
    reuse_existing: bool = False
    cache_seconds: int = 0


def run_preflight_canary(canary: PreflightCanarySpec) -> Dict[str, Any]:
    preflight_root = canary.internal_dir / "preflight_output"
    cmd = build_msprof_command(MsprofCommandSpec(
        canary.msprof, canary.metric, preflight_root, canary.app, canary.app_args, canary.kernel_name, 1,
        supported_options=canary.supported_options,
        profiler_options=canary.profiler_options,
    ))
    write_command_record(canary.internal_dir, "preflight_canary", canary.metric, cmd, {
        "purpose": "Short BasicInfo device canary before any expensive feature replay.",
        "output_directory": "_internal/preflight_output",
        "timeout_seconds": canary.timeout,
        "application_cwd": str(canary.cwd),
        "cache_seconds": canary.cache_seconds,
    })
    app_fp = fingerprint(canary.app)
    if canary.reuse_existing:
        reused = _try_reuse_preflight(_ReusePreflightSpec(canary.internal_dir, canary.output_root,
            shell_join(cmd), app_fp, canary.cwd, canary.cache_seconds))
        if reused is not None:
            return reused
    if preflight_root.exists():
        shutil.rmtree(preflight_root)
    preflight_root.mkdir(parents=True, exist_ok=True)
    ensure_msprof_output_permissions(preflight_root)
    if canary.dry_run:
        return _planned_preflight_result(canary.internal_dir, cmd, canary.cwd)
    ctx = _CanaryCtx(canary.internal_dir, canary.output_root, cmd, canary.timeout, canary.cwd,
        canary.heartbeat_seconds, app_fp, canary.cache_seconds)
    return _execute_canary(ctx)


def can_promote_preflight_to_discovery(
    preflight: Mapping[str, Any],
    *,
    kernel_name: Optional[str],
    launch_count: int,
    dry_run: bool,
) -> bool:
    """Decide whether the one-launch BasicInfo canary is sufficient discovery.

    Reuse is safe when formal collection targets only one launch, or when the
    kernel selector names one exact target. Multi-launch/wildcard discovery
    still performs its own wider BasicInfo pass.
    """
    return bool(
        not dry_run
        and preflight.get("status") in {"ok", "reused"}
        and (launch_count == 1 or is_single_exact_kernel_filter(kernel_name))
    )


def promote_preflight_to_discovery(
    preflight: Mapping[str, Any], output_root: Path, metric: str,
) -> Optional[Dict[str, Any]]:
    """Reuse the successful BasicInfo canary as discovery for one exact target kernel."""
    result_rel = preflight.get("result_dir")
    if preflight.get("status") not in {"ok", "reused"} or not result_rel:
        return None
    source = output_root / str(result_rel)
    if not source.is_dir():
        return None
    discovery_root = output_root / BLOCK_DIRS["discovery"]
    if discovery_root.exists():
        shutil.rmtree(discovery_root)
    discovery_root.mkdir(parents=True, exist_ok=True)
    destination = discovery_root / source.name
    shutil.copytree(source, destination)
    entry = build_block_entry(BlockEntrySpec(
        "discovery", metric, "reused", BLOCK_DIRS["discovery"], destination,
        output_root, 0.0, 0, ["discovery"],
        reason="Promoted the successful BasicInfo preflight result; no second BasicInfo replay was needed.",
    ))
    entry["provider"] = "preflight_canary"
    entry["collection_passes_saved"] = 1
    return entry


def append_skipped_after_circuit_breaker(
    block_entries: List[Dict[str, Any]],
    requested_blocks: Sequence[str],
    requested_by: Mapping[str, Sequence[str]],
    executed: Mapping[str, Dict[str, Any]],
    reason: str,
) -> None:
    for block_id in requested_blocks:
        if block_id in executed:
            continue
        block_entries.append(unavailable_block_entry(
            block_id,
            requested_by.get(block_id, []),
            f"Skipped by circuit breaker: {reason}",
        ))


class ManifestSpec(NamedTuple):
    """Bundled inputs for ``build_manifest`` (G.FNM.03)."""

    preset: str
    selection_mode: str
    requested_features: Sequence[str]
    supported: Sequence[str]
    supported_options: Sequence[str]
    op_type: str
    debug_symbols: bool
    simt_evidence: bool
    kernel_scale_instrumentation: bool
    build_provenance: Dict[str, Any]
    environment_context: Dict[str, Any]
    blocks: Sequence[Dict[str, Any]]
    run_status: str = "completed"
    run_reason: Optional[str] = None
    preflight: Optional[Dict[str, Any]] = None
    timeout_profile: Optional[Dict[str, int]] = None
    executable_context: Optional[Dict[str, Any]] = None


def build_manifest(manifest: ManifestSpec) -> Dict[str, Any]:
    block_map = {b["block_id"]: b for b in manifest.blocks}
    return {
        "schema": CONTRACT_SCHEMA,
        "collector_version": VERSION,
        "created_at": now_iso(),
        "path_rule": "All artifact paths are POSIX-style paths relative to this manifest.",
        "default_behavior": "No --feature means preset=complete; duplicate Default replays are automatically aliased.",
        "preset": manifest.preset,
        "selection_mode": manifest.selection_mode,
        "requested_features": list(manifest.requested_features),
        "supported_metrics": list(manifest.supported),
        "supported_options": list(manifest.supported_options),
        "detected_capabilities": {
            "operator_type": manifest.op_type or "unknown",
            "debug_symbols": manifest.debug_symbols,
            "simt_evidence": manifest.simt_evidence,
            "kernel_scale_instrumentation": manifest.kernel_scale_instrumentation,
        },
        "build_provenance": manifest.build_provenance,
        "environment_context": manifest.environment_context,
        "run_status": manifest.run_status,
        "run_reason": manifest.run_reason,
        "preflight": manifest.preflight or {},
        "timeout_profile": manifest.timeout_profile or {},
        "executable_context": manifest.executable_context or {},
        "block_directories": BLOCK_DIRS,
        "feature_to_block": FEATURE_TO_BLOCK,
        "visualization_contract": visualization_contract(),
        "blocks": block_map,
    }


def _add_target_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--operator-path", help="Operator sample/project root.")
    parser.add_argument("--output", help="Collection root; fixed relative feature directories are created here.")
    parser.add_argument("--app", help="Executable path; relative paths are resolved from --operator-path.")
    parser.add_argument("--app-cwd", help=("Application working directory; relative paths resolve from "
        "--operator-path. Defaults to the operator root so ./input resources are stable."))
    parser.add_argument("--kernel-name", help="Kernel filter. Omit for single-operator samples.")
    parser.add_argument("--op-type", help="Override discovered operator type.")
    parser.add_argument("--simt", choices=["auto", "on", "off"], default="auto", help=("SIMT evidence "
        "override for Warp Stall gating."))
    parser.add_argument("--source-stall", choices=["auto", "on", "off"], default="auto", help=("Collect "
        "PCSampling as an optional Source Explorer stall overlay; auto/on attempt it when Source is requested, "
        "off disables it."))
    parser.add_argument("--kernel-scale", choices=["auto", "on", "off"], default="auto", help=("KernelScale "
        "instrumentation override."))
    parser.add_argument("--app-arg", action="append", default=[], help="Repeatable application argument.")
    parser.add_argument("--msprof", default="msprof")


def _add_environment_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env-profile", help=("Reusable CANN/msOpProf environment profile. Relative paths "
        "resolve from --operator-path."))
    parser.add_argument("--env-profile-mode", choices=["auto",
        "readonly", "refresh", "off"], default="auto", help=("auto=reuse "
        "valid profile or refresh; readonly=require valid; refresh=re-probe and overwrite; off=ignore profile."))
    parser.add_argument("--env-source", action="append", default=[], help=("Repeatable shell source/init "
        "command used only when creating or refreshing the environment profile."))
    parser.add_argument("--env-var", action="append", default=[], metavar="KEY=VALUE", help=("Repeatable "
        "non-secret environment value to apply and save in the reusable profile."))
    parser.add_argument("--env-source-timeout", type=int, default=120, help=("Timeout for environment "
        "source/init commands."))


def _add_feature_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", choices=sorted(PRESETS), default="complete", help=("Default complete; "
        "ignored when --feature is supplied."))
    parser.add_argument("--feature", action="append", default=[], help=("Repeatable targeted visual feature, "
        "e.g. cache, timeline, raw-data."))
    parser.add_argument("--memory-info", help=("Exact memory_info.json for On-Chip Memory collection; "
        "relative paths resolve from --operator-path."))
    parser.add_argument("--list-features", action="store_true", help="List available user-facing features and exit.")
    parser.add_argument("--explain", metavar="FEATURE", help=("Explain command, data, visualization, and use "
        "cases for one feature and exit."))


def _add_profiler_option_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--launch-count", type=int, default=1)
    parser.add_argument("--launch-skip-before-match", type=int, help=("Skip this many launched operators "
        "before matching/collecting."))
    parser.add_argument("--replay-mode", choices=["kernel", "application", "range"], help=("Explicit msprof "
        "replay mode. Omit to use the installed CLI default (normally kernel)."))
    parser.add_argument("--warm-up", type=int, help="Profiler warm-up count; installed CLI default is normally 5.")
    parser.add_argument("--kill", choices=["on", "off"], help=("Stop the user application after "
        "--launch-count is reached. Omit for profiler default off."))
    parser.add_argument("--mstx", choices=["on", "off"], help="Enable or disable mstx range selection.")
    parser.add_argument("--mstx-include", help="Pipe-separated mstx message filter; requires --mstx=on.")
    parser.add_argument("--dump", choices=["on", "off"], help="TimelineDetail simulator dump control.")
    parser.add_argument("--core-id", help="TimelineDetail logical-core filter, e.g. 0|31.")
    parser.add_argument("--instr-timeline-pipe", help="Optional pipe filter for instrTimeLine, e.g. mte1|vector.")
    parser.add_argument("--debug-rebuild-command", help=("Optional shell command executed only when Source is "
        "requested and debug symbols are missing."))
    parser.add_argument("--debug-rebuild-timeout", type=int, default=1800)
    parser.add_argument("--build-config", action="append", default=[], metavar="KEY=VALUE", help=("Repeatable "
        "build provenance metadata, e.g. SCENARIO_NUM=0."))
    parser.add_argument("--validation-note", action="append", default=[], help=("Repeatable "
        "validation/provenance note stored in the manifest."))


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--timeout", type=int, default=1200, help=("Global hard timeout ceiling for one "
        "profiler command."))
    parser.add_argument("--preflight-mode", choices=["auto", "on", "off"], default="auto", help=("Run a short "
        "BasicInfo device canary before formal collection."))
    parser.add_argument("--preflight-timeout", type=int, default=60, help=("Canary timeout; a failure "
        "circuit-breaks the run."))
    parser.add_argument("--preflight-cache-seconds", type=int, default=300, help=("Reuse an identical "
        "successful canary from the same completed output for this many seconds; 0 disables."))
    parser.add_argument("--heartbeat-seconds", type=int, default=30, help=("Emit progress heartbeats while "
        "profiler commands are still running; 0 disables."))
    parser.add_argument("--circuit-breaker", action=argparse.BooleanOptionalAction, default=True, help=("Abort "
        "remaining blocks after any command timeout."))
    parser.add_argument("--adaptive-timeout", action=argparse.BooleanOptionalAction, default=True, help=("Use "
        "feature-class timeout floors scaled by observed canary cost."))
    parser.add_argument("--block-timeout", action="append", default=[], metavar="BLOCK=SECONDS", help=("Repeatable "
        "per-block timeout override."))
    parser.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--independent-default", action="store_true", help=("Force a separate "
        "MemoryDetail/Default replay even when Roofline is semantically reusable; use for cross-replay "
        "diagnostics, not the minimum-pass default."))
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--strict", action="store_true", help=("Exit non-zero when any explicitly requested "
        "block is unavailable, failed, or semantically empty."))
    parser.add_argument("--dry-run", action="store_true", help=("Write plan/manifest without executing "
        "profiling commands."))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect msOpProf feature blocks into a fixed visualization-ready directory contract."
    )
    _add_target_arguments(parser)
    _add_environment_arguments(parser)
    _add_feature_arguments(parser)
    _add_profiler_option_arguments(parser)
    _add_runtime_arguments(parser)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.timeout <= 0 or args.preflight_timeout <= 0:
        parser.error("--timeout and --preflight-timeout must be positive.")
    if args.heartbeat_seconds < 0:
        parser.error("--heartbeat-seconds cannot be negative.")
    if args.preflight_cache_seconds < 0:
        parser.error("--preflight-cache-seconds cannot be negative.")
    if not 1 <= args.launch_count <= 5000:
        parser.error("--launch-count must be in [1, 5000].")
    if args.launch_skip_before_match is not None and not 0 <= args.launch_skip_before_match <= 1000:
        parser.error("--launch-skip-before-match must be in [0, 1000].")
    if args.warm_up is not None and not 0 <= args.warm_up <= 500:
        parser.error("--warm-up must be in [0, 500].")


def _plan_requested(run: Any) -> None:
    args = run.args
    run.requested_features = resolve_requested_features(args.preset, args.feature)
    run.requested_blocks, run.requested_by = resolve_blocks(run.requested_features, source_stall=args.source_stall)
    run.user_build_config = parse_key_value(args.build_config, "--build-config")
    run.explicit_environment = envctx.parse_key_value(args.env_var, "--env-var")
    run.block_timeout_overrides = parse_block_timeouts(args.block_timeout)
    run.profiler_options = {
        "launch_skip_before_match": args.launch_skip_before_match,
        "replay_mode": args.replay_mode,
        "warm_up": args.warm_up,
        "kill": args.kill,
        "mstx": args.mstx,
        "mstx_include": args.mstx_include,
        "dump": args.dump,
        "core_id": args.core_id,
    }
    # Discovery must use the same operator-selection filters as formal
    # collection, while replay-specific or expensive feature options remain
    # isolated to their intended blocks.
    run.selection_profiler_options = {
        "launch_skip_before_match": args.launch_skip_before_match,
        "mstx": args.mstx,
        "mstx_include": args.mstx_include,
    }


def _prepare_output(run: Any) -> None:
    args = run.args
    run.operator_root = Path(args.operator_path).expanduser().resolve()
    run.output_root = Path(args.output).expanduser().resolve()
    if not run.operator_root.exists():
        raise rtguard.UsageError(f"ERROR: operator path does not exist: {run.operator_root}")
    if args.clean and run.output_root.exists():
        shutil.rmtree(run.output_root)
    run.output_root.mkdir(parents=True, exist_ok=True)
    run.internal_dir = run.output_root / "_internal"
    run.internal_dir.mkdir(parents=True, exist_ok=True)
    reuse_allowed, run.reuse_reason = previous_run_allows_reuse(run.output_root)
    run.effective_reuse = bool(args.reuse_existing and reuse_allowed)
    run.run_guard = rtguard.RunStateGuard(run.internal_dir / "run_state.json", {
        "collector_version": VERSION,
        "operator_path": str(run.operator_root),
        "output": str(run.output_root),
        "requested_features": run.requested_features,
        "reuse_requested": bool(args.reuse_existing),
        "reuse_enabled": run.effective_reuse,
        "reuse_reason": run.reuse_reason,
    })

    # Ensure every fixed block directory exists, even when the block is unavailable.
    for block_id, dirname in BLOCK_DIRS.items():
        if block_id != "discovery":
            (run.output_root / dirname).mkdir(parents=True, exist_ok=True)

    # msprof refuses to write into group/other-writable directories and exits 0
    # with zero artifacts; sanitize before any profiler command runs.
    permission_fixes = sanitize_output_tree(run.output_root)
    write_json(run.internal_dir / "output_permission_fixes.json", {
        "schema": "msopprof-output-permissions/v1",
        "checked_at": now_iso(),
        "fix_count": len(permission_fixes),
        "fixes": permission_fixes,
    })


def _resolve_environment(run: Any) -> None:
    args = run.args
    env_profile_path = envctx.resolve_profile_path(args.env_profile, run.operator_root)
    phase_started = time.perf_counter()
    environment_result = envctx.prepare_environment(envctx.EnvironmentRequest(
        profile_path=env_profile_path,
        mode=args.env_profile_mode,
        source_commands=args.env_source,
        source_cwd=run.operator_root,
        requested_msprof=args.msprof,
        source_timeout=args.env_source_timeout,
        parse_supported_metrics=parse_supported_metrics,
        parse_supported_options=parse_supported_options,
        explicit_environment=run.explicit_environment,
    ))
    run.environment_context = envctx.public_context(environment_result)
    run.local_phase_seconds["environment_resolution"] = time.perf_counter() - phase_started
    write_json(run.internal_dir / "environment_profile_status.json", run.environment_context)

    run.msprof = str(environment_result["msprof"])
    help_text = str(environment_result.get("help_text") or "")
    # Toolchain identity participates in block reuse: a different msprof binary
    # or CLI surface must invalidate cached command state and force a re-run.
    run.msprof_fp = {"path": run.msprof, "help_sha256": envctx.text_sha256(help_text)}
    (run.internal_dir / "msprof_op_help.txt").write_text(help_text, encoding="utf-8")
    run.supported = list(environment_result.get("supported_metrics") or [])
    run.supported_options = list(environment_result.get("supported_options") or [])
    run.basic_info_supported = any(
        normalize_metric_token(x) == normalize_metric_token("BasicInfo") for x in run.supported
    )


def _executable_context_payload(selection: Mapping[str, Any], app: Path, app_cwd: Path) -> Dict[str, Any]:
    context = {k: v for k, v in selection.items() if k != "selected"}
    context["selected"] = str(app)
    context["application_cwd"] = str(app_cwd)
    context["relative_resource_hints"] = detect_relative_resource_hints(app, app_cwd)
    context["test_like_executable_warning"] = test_like_executable_warning(app)
    return context


def _resolve_executable(run: Any) -> None:
    args = run.args
    phase_started = time.perf_counter()
    run.executable_selection = resolve_executable_details(run.operator_root, args.app)
    run.app = Path(run.executable_selection["selected"])
    run.app_cwd = resolve_app_cwd(run.operator_root, args.app_cwd)
    run.executable_context = _executable_context_payload(run.executable_selection, run.app, run.app_cwd)
    write_json(run.internal_dir / "executable_selection.json", run.executable_context)
    run.local_phase_seconds["executable_resolution"] = time.perf_counter() - phase_started


def _check_capabilities(run: Any) -> None:
    args = run.args
    if not run.basic_info_supported and not args.kernel_name and not args.op_type:
        raise rtguard.UsageError(
            "ERROR: BasicInfo is not exposed and neither --kernel-name nor --op-type was supplied."
        )
    if "--aic-metrics" not in run.supported_options or "--output" not in run.supported_options:
        raise rtguard.UsageError(("ERROR: installed msprof op CLI does not expose the required --aic-metrics/--output "
            "interface."))


def _run_preflight(run: Any) -> Optional[int]:
    args = run.args
    run.preflight = {"status": "skipped", "reason": "Preflight disabled or BasicInfo unavailable."}
    if args.preflight_mode == "on" and not run.basic_info_supported:
        run.run_guard.finalize("aborted", "Preflight was required but BasicInfo is not exposed.")
        raise rtguard.UsageError("ERROR: --preflight-mode=on requires BasicInfo support.")
    if not run.basic_info_supported or args.preflight_mode == "off":
        return None
    basic_metric = next(x for x in run.supported if normalize_metric_token(x) == normalize_metric_token("BasicInfo"))
    run.preflight = run_preflight_canary(PreflightCanarySpec(
        msprof=run.msprof, metric=basic_metric, output_root=run.output_root, internal_dir=run.internal_dir,
        app=run.app, app_args=args.app_arg, kernel_name=args.kernel_name,
        supported_options=run.supported_options, timeout=args.preflight_timeout,
        cwd=run.app_cwd, heartbeat_seconds=args.heartbeat_seconds, dry_run=args.dry_run,
        profiler_options=run.selection_profiler_options, reuse_existing=run.effective_reuse,
        cache_seconds=args.preflight_cache_seconds,
    ))
    if run.preflight.get("status") != "failed":
        return None
    return _abort_after_preflight(run)


def _abort_after_preflight(run: Any) -> int:
    args = run.args
    reason = str(run.preflight.get("reason") or "Preflight failed.")
    rtguard.collect_device_diagnostics(
        run.internal_dir / "diagnostics" / "preflight", reason=reason, command_result=run.preflight,
    )
    skipped: List[Dict[str, Any]] = []
    for block_id in run.requested_blocks:
        skipped.append(unavailable_block_entry(
            block_id, run.requested_by[block_id], f"Skipped by preflight circuit breaker: {reason}"
        ))
    manifest = build_manifest(ManifestSpec(
        preset=args.preset, selection_mode="features" if args.feature else "preset",
        requested_features=run.requested_features, supported=run.supported, supported_options=run.supported_options,
        op_type=args.op_type or "unknown", debug_symbols=False, simt_evidence=False,
        kernel_scale_instrumentation=False, build_provenance=parse_cmake_cache(run.operator_root),
        environment_context=run.environment_context, blocks=skipped, run_status="aborted",
        run_reason=reason, preflight=run.preflight, timeout_profile={}, executable_context=run.executable_context,
    ))
    write_json(run.output_root / "collection_manifest.json", manifest)
    run.run_guard.finalize("aborted", reason, {"diagnostics": "diagnostics/preflight/device_diagnostics.json"})
    cli_logger.info(json.dumps({
        "schema": CONTRACT_SCHEMA, "output": str(run.output_root), "run_status": "aborted",
        "reason": reason, "preflight": run.preflight,
        "diagnostics": str(run.internal_dir / "diagnostics" / "preflight" / "device_diagnostics.json"),
    }, ensure_ascii=False, indent=2))
    return 3


def _derive_timeouts(run: Any) -> None:
    args = run.args
    run.timeout_profile = derive_timeout_profile(
        args.timeout, run.block_timeout_overrides,
        # A reused canary reports elapsed_seconds=0.0; derive adaptive timeouts
        # from the original observed cost so blocks do not fall back to floors.
        preflight_elapsed=run.preflight.get("original_elapsed_seconds") or run.preflight.get("elapsed_seconds"),
        adaptive=args.adaptive_timeout,
    )
    write_json(run.internal_dir / "timeout_profile.json", run.timeout_profile)


def _rederive_timeouts(run: Any, discovery_entry: Mapping[str, Any]) -> None:
    args = run.args
    run.timeout_profile = derive_timeout_profile(
        args.timeout, run.block_timeout_overrides,
        preflight_elapsed=max(
            float(run.preflight.get("original_elapsed_seconds") or run.preflight.get("elapsed_seconds") or 0),
            float(discovery_entry.get("elapsed_seconds") or 0),
        ),
        adaptive=args.adaptive_timeout,
    )
    write_json(run.internal_dir / "timeout_profile.json", run.timeout_profile)


def _run_debug_rebuild(run: Any) -> None:
    args = run.args
    run.debug_rebuild = run_shell_capture(args.debug_rebuild_command, run.operator_root, args.debug_rebuild_timeout)
    logs = run.internal_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "debug_rebuild.stdout.log").write_text(run.debug_rebuild.get("stdout", ""), encoding="utf-8")
    (logs / "debug_rebuild.stderr.log").write_text(run.debug_rebuild.get("stderr", ""), encoding="utf-8")
    run.executable_selection = resolve_executable_details(run.operator_root, args.app)
    run.app = Path(run.executable_selection["selected"])
    run.executable_context.update(_executable_context_payload(run.executable_selection, run.app, run.app_cwd))
    write_json(run.internal_dir / "executable_selection.json", run.executable_context)
    run.debug_details = detect_debug_symbols_details(run.operator_root, run.app)


def _detect_debug_symbols(run: Any) -> None:
    args = run.args
    # readelf-based detection can cost 160 sequential subprocesses; only pay it
    # when Source is actually requested, and never during a dry run.
    if args.dry_run:
        run.debug_details = {
            "detected": True,
            "reason": "Debug-symbol detection skipped during dry run; Source eligibility is assumed for planning.",
        }
    elif "source" not in run.requested_blocks:
        run.debug_details = {
            "detected": False,
            "reason": "Source block was not requested; debug-symbol detection skipped.",
        }
    else:
        run.debug_details = detect_debug_symbols_details(run.operator_root, run.app)
    run.debug_rebuild = None
    rebuild_requested = (
        "source" in run.requested_blocks
        and not run.debug_details.get("detected")
        and args.debug_rebuild_command
    )
    if rebuild_requested:
        _run_debug_rebuild(run)
    run.debug_symbols = bool(run.debug_details.get("detected"))


def _detect_kernel_scale(run: Any) -> None:
    args = run.args
    if args.dry_run:
        run.kernel_scale_details = {
            "detected": True,
            "source_files": [],
            "binary_evidence": False,
            "reason": ("Instrumentation detection skipped during dry run; KernelScale eligibility is assumed "
                "for planning."),
        }
    elif "kernel_scale" not in run.requested_blocks:
        run.kernel_scale_details = {
            "detected": False,
            "source_files": [],
            "binary_evidence": False,
            "reason": "KernelScale block was not requested; instrumentation detection skipped.",
        }
    else:
        run.kernel_scale_details = detect_kernel_scale_instrumentation(run.operator_root, run.app)


def _detect_capabilities(run: Any) -> None:
    args = run.args
    _detect_debug_symbols(run)
    run.app_fp = fingerprint(run.app)
    _detect_kernel_scale(run)
    provenance = parse_cmake_cache(run.operator_root)
    provenance["user_build_config"] = run.user_build_config
    provenance["validation_notes"] = list(args.validation_note)
    provenance["executable_selection"] = run.executable_context
    run.build_provenance = provenance
    run.block_entries = []


def _execute_discovery_block(run: Any, discovery_metric: str) -> Dict[str, Any]:
    args = run.args
    discovery_dir = run.output_root / BLOCK_DIRS["discovery"]
    discovery_cmd = build_msprof_command(MsprofCommandSpec(
        run.msprof, discovery_metric, discovery_dir, run.app, args.app_arg,
        args.kernel_name, 10, supported_options=run.supported_options,
        profiler_options=run.selection_profiler_options,
    ))
    write_command_record(run.internal_dir, "discovery", discovery_metric, discovery_cmd, {
        "purpose": "Discover operator names and reported operator type before feature collection.",
        "output_directory": BLOCK_DIRS["discovery"],
        "application_cwd": str(run.app_cwd),
        "timeout_seconds": run.timeout_profile["discovery"],
    })
    return execute_block(BlockRunSpec(
        "discovery", discovery_metric, discovery_cmd, run.output_root, run.internal_dir,
        run.timeout_profile["discovery"], run.effective_reuse, run.app_fp, args.dry_run, ["discovery"],
        cwd=run.app_cwd, heartbeat_seconds=args.heartbeat_seconds, msprof_fp=run.msprof_fp,
    ))


def _abort_after_discovery_timeout(run: Any, discovery_entry: Mapping[str, Any]) -> int:
    args = run.args
    reason = f"Discovery timed out after {discovery_entry.get('timeout_seconds')}s."
    rtguard.collect_device_diagnostics(
        run.internal_dir / "diagnostics" / "discovery", reason=reason, command_result=discovery_entry,
    )
    append_skipped_after_circuit_breaker(run.block_entries, run.requested_blocks,
        run.requested_by, {"discovery": discovery_entry}, reason)
    manifest = build_manifest(ManifestSpec(
        preset=args.preset, selection_mode="features" if args.feature else "preset",
        requested_features=run.requested_features, supported=run.supported, supported_options=run.supported_options,
        op_type=args.op_type or "unknown", debug_symbols=run.debug_symbols, simt_evidence=False,
        kernel_scale_instrumentation=bool(run.kernel_scale_details.get("detected")),
        build_provenance=run.build_provenance, environment_context=run.environment_context, blocks=run.block_entries,
        run_status="aborted", run_reason=reason, preflight=run.preflight, timeout_profile=run.timeout_profile,
        executable_context=run.executable_context,
    ))
    write_json(run.output_root / "collection_manifest.json", manifest)
    run.run_guard.finalize("aborted", reason, {"diagnostics": "diagnostics/discovery/device_diagnostics.json"})
    cli_logger.info(json.dumps({"run_status": "aborted", "reason": reason, "manifest": str(run.output_root /
        "collection_manifest.json")}, ensure_ascii=False, indent=2))
    return 3


def _run_discovery(run: Any) -> Optional[int]:
    args = run.args
    # Discovery is infrastructure, not a visual feature block.
    run.discovered_ops = []
    if not run.basic_info_supported or (args.kernel_name and args.op_type):
        return None
    discovery_metric = ""
    for candidate in run.supported:
        if normalize_metric_token(candidate) == normalize_metric_token("BasicInfo"):
            discovery_metric = candidate
            break
    discovery_entry: Optional[Dict[str, Any]] = None
    if can_promote_preflight_to_discovery(
        run.preflight, kernel_name=args.kernel_name, launch_count=args.launch_count, dry_run=args.dry_run
    ):
        discovery_entry = promote_preflight_to_discovery(run.preflight, run.output_root, discovery_metric)
    if discovery_entry is None:
        discovery_entry = _execute_discovery_block(run, discovery_metric)
    run.block_entries.append(discovery_entry)
    if discovery_entry.get("timed_out") and args.circuit_breaker:
        return _abort_after_discovery_timeout(run, discovery_entry)
    result_dir = run.output_root / discovery_entry["result_dir"] if discovery_entry.get("result_dir") else None
    run.discovered_ops = discover_operators(result_dir)
    _rederive_timeouts(run, discovery_entry)
    return None


def _resolve_operator_identity(run: Any) -> None:
    args = run.args
    run.op_type = args.op_type or "unknown"
    run.kernel_name = args.kernel_name
    if not run.discovered_ops:
        return
    if len(run.discovered_ops) == 1:
        run.op_type = args.op_type or run.discovered_ops[0].get("type") or "unknown"
    elif not run.kernel_name:
        names = ", ".join(x.get("name", "?") for x in run.discovered_ops)
        raise rtguard.UsageError(f"ERROR: multiple operators discovered ({names}). Re-run with --kernel-name.")


def _environment_record(run: Any) -> Dict[str, Any]:
    args = run.args
    return {
        "collector_version": VERSION,
        "environment_context": run.environment_context,
        "app": str(run.app),
        "app_cwd": str(run.app_cwd),
        "executable_context": run.executable_context,
        "app_fingerprint": run.app_fp,
        "msprof": run.msprof,
        "supported_metrics": run.supported,
        "supported_options": run.supported_options,
        "debug_symbols": run.debug_symbols,
        "debug_symbol_details": run.debug_details,
        "debug_rebuild": run.debug_rebuild,
        "kernel_scale_instrumentation": run.kernel_scale_details,
        "build_provenance": run.build_provenance,
        "kernel_name": run.kernel_name,
        "operator_type": run.op_type,
        "requested_features": run.requested_features,
        "requested_blocks": run.requested_blocks,
        "profiler_options": {k: v for k, v in run.profiler_options.items() if v is not None},
        "selection_profiler_options": {k: v for k, v in run.selection_profiler_options.items() if v is not None},
        "preflight": run.preflight,
        "timeout_profile": run.timeout_profile,
        "reuse_enabled": run.effective_reuse,
        "reuse_reason": run.reuse_reason,
        "created_at": now_iso(),
    }


def _build_plan_rows(run: Any) -> List[Dict[str, Any]]:
    # Write the user-visible plan before execution.
    plan_rows: List[Dict[str, Any]] = []
    for block_id in run.requested_blocks:
        metric = select_metric(block_id, run.supported)
        plan_rows.append({
            "block_id": block_id,
            "requested_by": run.requested_by[block_id],
            "metric": metric,
            "relative_dir": BLOCK_DIRS[block_id],
            "timeout_seconds": run.timeout_profile.get(block_id),
            "application_cwd": str(run.app_cwd),
            "profiler_options": {k: v for k, v in run.profiler_options.items() if v is not None},
            "description": command_description(block_id, metric or "UNAVAILABLE"),
        })
    return plan_rows


def _write_context_files(run: Any) -> None:
    write_json(run.output_root / "feature_catalog.json", feature_catalog())
    write_json(run.internal_dir / "environment.json", _environment_record(run))
    write_json(run.internal_dir / "collection_plan.json", _build_plan_rows(run))


def _semantic_memory_detail_alias(
    run: Any,
    block_id: str,
    candidate_metrics: Sequence[str],
    req: Sequence[str],
    executed: Mapping[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    # When the installed CLI has no native MemoryDetail token, do not launch
    # a redundant Default replay if a richer completed replay already
    # satisfies the same compute/memory/cache/advice contract. Roofline is
    # checked first because it is documented to include Default counters.
    native_memory_detail = any(
        normalize_metric_token(metric) == normalize_metric_token("MemoryDetail")
        for metric in candidate_metrics
    )
    if native_memory_detail:
        return None
    for provider_id in ["roofline", "raw_data", "details"]:
        provider = executed.get(provider_id)
        if not provider:
            continue
        candidate_alias = semantic_alias_candidate(block_id, provider, req, run.output_root)
        if candidate_alias is not None:
            return candidate_alias
    return None


def _standard_csv_provider(executed: Mapping[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for provider_id in ["memory_detail", "roofline", "details"]:
        candidate = executed.get(provider_id)
        reusable_status = candidate and candidate.get("status") in {"ok", "reused", "partial", "aliased"}
        if reusable_status and has_standard_csv_suite(candidate):
            return candidate
    return None


def _available_raw_metrics(supported: Sequence[str]) -> List[str]:
    # Some CANN versions expose the seven raw counters but not Default.
    raw_metrics: List[str] = []
    for canonical in RAW_COUNTER_METRICS:
        actual = next(
            (x for x in supported if normalize_metric_token(x) == normalize_metric_token(canonical)),
            None,
        )
        if actual:
            raw_metrics.append(actual)
    return raw_metrics


def _raw_data_shortcut(
    run: Any,
    block_id: str,
    candidate_metrics: Sequence[str],
    req: Sequence[str],
    executed: Mapping[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    args = run.args
    # Raw Data first reuses any richer replay that already emitted the standard CSV suite.
    provider = _standard_csv_provider(executed)
    if provider is not None:
        entry = alias_block_entry(block_id, provider, req, run.output_root)
        return validate_block_entry(block_id, entry, run.output_root)
    if candidate_metrics:
        return None
    raw_metrics = _available_raw_metrics(run.supported)
    if not raw_metrics:
        return None
    entry = execute_raw_counter_suite(RawCounterSuiteSpec(
        raw_metrics, run.msprof, run.output_root, run.internal_dir, run.app, args.app_arg,
        run.kernel_name, args.launch_count, run.supported_options, run.timeout_profile["raw_data"],
        args.dry_run, req, cwd=run.app_cwd, heartbeat_seconds=args.heartbeat_seconds,
        profiler_options=run.profiler_options,
    ))
    if entry.get("timed_out") and args.circuit_breaker:
        run.circuit_breaker_reason = (f"Raw Data metric suite timed out after "
            f"{entry.get('timeout_seconds')}s.")
        diag_dir = run.internal_dir / "diagnostics" / "raw_data"
        rtguard.collect_device_diagnostics(diag_dir,
            reason=run.circuit_breaker_reason, command_result=entry)
        run.circuit_breaker_diagnostics = rel(diag_dir / "device_diagnostics.json", run.output_root)
    return entry


def _resolve_shortcut_entry(
    run: Any,
    block_id: str,
    candidate_metrics: Sequence[str],
    req: Sequence[str],
    executed: Mapping[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    args = run.args
    if block_id == "onchip_memory":
        return collect_onchip_memory_artifact(run.operator_root, run.output_root,
            req, Path(args.memory_info) if args.memory_info else None)
    if block_id == "memory_detail" and not args.independent_default:
        entry = _semantic_memory_detail_alias(run, block_id, candidate_metrics, req, executed)
        if entry is not None:
            return entry
    if block_id == "raw_data":
        return _raw_data_shortcut(run, block_id, candidate_metrics, req, executed)
    return None


def _simt_eligible(args: argparse.Namespace, block_id: str, requested_features: Sequence[str],
    simt_evidence: bool) -> bool:
    explicit_target = bool(args.feature) and (
        any(block_id in FEATURE_TO_BLOCKS[f] for f in requested_features)
        or (block_id == "warp_stall" and "source" in requested_features and args.source_stall != "off")
    )
    if args.simt == "on":
        return True
    if args.simt == "off":
        return False
    return bool(simt_evidence or explicit_target)


def _kernel_scale_eligible(args: argparse.Namespace, kernel_scale_details: Mapping[str, Any]) -> bool:
    if args.kernel_scale == "on":
        return True
    if args.kernel_scale == "off":
        return False
    return bool(kernel_scale_details.get("detected"))


def _check_eligibility(
    run: Any,
    block_id: str,
    spec: Mapping[str, Any],
    candidate_metrics: Sequence[str],
    req: Sequence[str],
) -> Optional[Dict[str, Any]]:
    args = run.args
    if not candidate_metrics:
        return unavailable_block_entry(
            block_id, req,
            f"Installed msprof op CLI exposes none of: {', '.join(spec['metric_candidates'])}.",
        )
    if spec.get("requires_debug_symbols") and not run.debug_symbols:
        return unavailable_block_entry(
            block_id, req,
            ("Usable .debug_line information was not detected in the selected executable build tree. "
                "Rebuild the exact selected executable/kernel with -g, or pass --debug-rebuild-command so "
                "the collector can rebuild conditionally before Source collection."),
        )
    if spec.get("requires_simt") and not _simt_eligible(args, block_id, run.requested_features, run.simt_evidence):
        return unavailable_block_entry(
            block_id, req,
            ("No SIMT evidence detected. Run Details first, use --simt on, or request warp-stall "
                "explicitly for best-effort collection."),
        )
    if spec.get("requires_kernel_scale_instrumentation") and not _kernel_scale_eligible(
        args, run.kernel_scale_details
    ):
        return unavailable_block_entry(
            block_id, req,
            ("KernelScale requires MetricsProfStart/MetricsProfStop instrumentation. Add the markers "
                "or use --kernel-scale on for a best-effort run."),
        )
    return None


def _find_reusable_provider(
    executed: Mapping[str, Dict[str, Any]],
    candidate_metrics: Sequence[str],
) -> Optional[Dict[str, Any]]:
    # Reuse a previous identical metric replay when no block-specific command option differs.
    for provider in executed.values():
        if provider.get("status") not in {"ok", "reused", "partial", "aliased"}:
            continue
        provider_metric = provider.get("metric")
        if (provider_metric and normalize_metric_token(provider_metric)
                == normalize_metric_token(candidate_metrics[0])):
            return provider
    return None


def _find_metric_provider(executed: Mapping[str, Dict[str, Any]], metric: str) -> Optional[Dict[str, Any]]:
    for provider in executed.values():
        provider_metric = provider.get("metric")
        if (provider.get("status") in {"ok", "reused", "partial", "aliased"}
                and provider_metric
                and normalize_metric_token(provider_metric) == normalize_metric_token(metric)):
            return provider
    return None


class _SingleAttemptSpec(NamedTuple):
    """Bundled inputs for ``_run_single_attempt`` (G.FNM.03)."""

    run: Any
    block_id: str
    metric: str
    attempt_index: int
    candidate_metrics: Sequence[str]
    req: Sequence[str]


def _run_single_attempt(spec: _SingleAttemptSpec) -> Dict[str, Any]:
    run = spec.run
    block_id = spec.block_id
    metric = spec.metric
    attempt_index = spec.attempt_index
    candidate_metrics = spec.candidate_metrics
    req = spec.req
    args = run.args
    run_dir = run.output_root / BLOCK_DIRS[block_id]
    cmd = build_msprof_command(MsprofCommandSpec(
        run.msprof, metric, run_dir, run.app, args.app_arg, run.kernel_name, args.launch_count,
        instr_timeline_pipe=args.instr_timeline_pipe,
        supported_options=run.supported_options,
        profiler_options=run.profiler_options,
    ))
    command_id = block_id
    if len(candidate_metrics) > 1:
        metric_token = normalize_metric_token(metric)
        command_id = f"{block_id}_attempt_{attempt_index:02d}_{metric_token}"
    write_command_record(run.internal_dir, command_id, metric, cmd, command_description(block_id, metric))
    entry = execute_block(BlockRunSpec(
        block_id, metric, cmd, run.output_root, run.internal_dir,
        run.timeout_profile[block_id], run.effective_reuse, run.app_fp, args.dry_run, req,
        log_stem=command_id, cwd=run.app_cwd, heartbeat_seconds=args.heartbeat_seconds,
        msprof_fp=run.msprof_fp,
    ))
    if not args.dry_run:
        entry = validate_block_entry(block_id, entry, run.output_root)
    return entry


def _run_metric_attempts(
    run: Any,
    block_id: str,
    candidate_metrics: Sequence[str],
    req: Sequence[str],
    executed: Mapping[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    args = run.args
    attempts: List[Dict[str, Any]] = []
    best_entry: Optional[Dict[str, Any]] = None
    for attempt_index, metric in enumerate(candidate_metrics, start=1):
        metric_provider = _find_metric_provider(executed, metric)
        if metric_provider is not None:
            entry = alias_block_entry(block_id, metric_provider, req, run.output_root)
            entry = validate_block_entry(block_id, entry, run.output_root)
            attempts.append({"metric": metric, "status": entry.get("status"),
                "provider_block": metric_provider.get("block_id"), "aliased": True})
            if entry.get("status") in {"aliased", "ok", "reused", "partial"}:
                best_entry = entry
                break
        entry = _run_single_attempt(_SingleAttemptSpec(run, block_id, metric, attempt_index,
            candidate_metrics, req))
        attempts.append({
            "metric": metric, "status": entry.get("status"),
            "return_code": entry.get("return_code"), "coverage": entry.get("coverage"),
            "timed_out": entry.get("timed_out", False),
        })
        if entry.get("status") in {"ok", "reused"}:
            best_entry = entry
            break
        best_entry = entry
        if entry.get("timed_out") and args.circuit_breaker:
            run.circuit_breaker_reason = f"{block_id}/{metric} timed out after {entry.get('timeout_seconds')}s."
            diag_dir = run.internal_dir / "diagnostics" / block_id
            rtguard.collect_device_diagnostics(diag_dir, reason=run.circuit_breaker_reason, command_result=entry)
            run.circuit_breaker_diagnostics = rel(diag_dir / "device_diagnostics.json", run.output_root)
            break
        has_fallback = attempt_index < len(candidate_metrics)
        if has_fallback and fallback_allowed(entry):
            continue
        if has_fallback:
            entry["fallback_suppressed"] = {
                "next_metric": candidate_metrics[attempt_index],
                "reason": ("Fallback is allowed only when the preferred command exits 0 and semantic "
                    "validation returns empty; timeout/non-zero/partial results are not retried."),
            }
        break
    return attempts, best_entry


def _execute_block_with_fallbacks(
    run: Any,
    block_id: str,
    candidate_metrics: Sequence[str],
    req: Sequence[str],
    executed: Mapping[str, Dict[str, Any]],
) -> Dict[str, Any]:
    reusable_provider = _find_reusable_provider(executed, candidate_metrics)
    if reusable_provider is not None and block_id not in {"raw_data"}:
        entry = alias_block_entry(block_id, reusable_provider, req, run.output_root)
        return validate_block_entry(block_id, entry, run.output_root)
    attempts, best_entry = _run_metric_attempts(run, block_id, candidate_metrics, req, executed)
    entry = best_entry or unavailable_block_entry(block_id, req, "No metric attempt produced an entry.")
    entry["attempts"] = attempts
    entry["preferred_metric"] = candidate_metrics[0]
    entry["fallback_used"] = bool(
        entry.get("metric")
        and normalize_metric_token(entry["metric"]) != normalize_metric_token(candidate_metrics[0])
    )
    if entry["fallback_used"]:
        entry["reason"] = (f"Preferred metric {candidate_metrics[0]} did not provide a usable payload; "
            f"used {entry.get('metric')} fallback.")
    return entry


def _update_simt_evidence(run: Any, block_id: str, entry: Mapping[str, Any]) -> None:
    # BasicInfo may report Vector while the profiling payload contains SIMT instructions.
    if block_id != "details" or entry.get("status") not in {"ok", "reused", "partial", "aliased"}:
        return
    path = (entry.get("artifacts") or {}).get("visualize_data")
    bin_path = run.output_root / path if path else None
    if detect_simt_evidence_from_bin(bin_path):
        run.simt_evidence = True


def _execute_blocks(run: Any) -> None:
    executed: Dict[str, Dict[str, Any]] = {x["block_id"]: x for x in run.block_entries}
    run.simt_evidence = "simt" in run.op_type.lower() if run.op_type else False

    for block_id in run.requested_blocks:
        spec = BLOCKS[block_id]
        candidate_metrics = select_metric_candidates(block_id, run.supported)
        req = run.requested_by[block_id]
        entry = _resolve_shortcut_entry(run, block_id, candidate_metrics, req, executed)
        if entry is None:
            entry = _check_eligibility(run, block_id, spec, candidate_metrics, req)
        if entry is None:
            entry = _execute_block_with_fallbacks(run, block_id, candidate_metrics, req, executed)
        run.block_entries.append(entry)
        executed[block_id] = entry
        if run.circuit_breaker_reason:
            break
        _update_simt_evidence(run, block_id, entry)

    if run.circuit_breaker_reason:
        append_skipped_after_circuit_breaker(
            run.block_entries, run.requested_blocks, run.requested_by, executed, run.circuit_breaker_reason,
        )


def _block_timing_row(block: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "phase": str(block.get("block_id")),
        "elapsed_seconds": round(float(block.get("elapsed_seconds") or 0.0), 6),
        "kind": "profiler" if block.get("metric") else "metadata",
        "status": block.get("status"),
        "metric": block.get("metric"),
        "provider_block": block.get("provider_block"),
        "collection_passes_saved": int(block.get("collection_passes_saved") or 0),
    }


def _timing_rows(run: Any) -> List[Dict[str, Any]]:
    timing_rows: List[Dict[str, Any]] = []
    for name, seconds in run.local_phase_seconds.items():
        timing_rows.append({"phase": name, "elapsed_seconds": round(seconds, 6), "kind": "local"})
    if run.debug_rebuild is not None:
        timing_rows.append({
            "phase": "debug_rebuild",
            "elapsed_seconds": round(float(run.debug_rebuild.get("elapsed_seconds") or 0.0), 6),
            "kind": "build",
            "status": "ok" if int(run.debug_rebuild.get("return_code") or 0) == 0 else "failed",
        })
    timing_rows.append({
        "phase": "preflight",
        "elapsed_seconds": round(float(run.preflight.get("elapsed_seconds") or 0.0), 6),
        "kind": "profiler",
        "status": run.preflight.get("status"),
    })
    for block in run.block_entries:
        timing_rows.append(_block_timing_row(block))
    return timing_rows


def _build_timing_summary(run: Any) -> Dict[str, Any]:
    timing_rows = _timing_rows(run)
    collector_wall_seconds = time.perf_counter() - run.collector_wall_started
    profiler_seconds = sum(
        float(row.get("elapsed_seconds") or 0.0)
        for row in timing_rows if row.get("kind") == "profiler"
    )
    saved_replays = sum(int(row.get("collection_passes_saved") or 0) for row in timing_rows)
    timing_summary = {
        "schema": "msopprof-timing/v1",
        "collector_version": VERSION,
        "wall_seconds": round(collector_wall_seconds, 6),
        "profiler_seconds": round(profiler_seconds, 6),
        "saved_replay_count": saved_replays,
        "preflight_discovery_fused": any(
            block.get("block_id") == "discovery" and block.get("provider") == "preflight_canary"
            for block in run.block_entries
        ),
        "phases": timing_rows,
        "scope_note": ("Measures pipeline execution only; agent reading, manual repository search, and "
            "external shell preparation are outside this timer."),
    }
    write_json(run.internal_dir / "timing_summary.json", timing_summary)
    return timing_summary


def _write_final_manifest(run: Any) -> str:
    args = run.args
    run_status = "aborted" if run.circuit_breaker_reason else "completed"
    manifest = build_manifest(ManifestSpec(
        preset=args.preset,
        selection_mode="features" if args.feature else "preset",
        requested_features=run.requested_features,
        supported=run.supported,
        supported_options=run.supported_options,
        op_type=run.op_type,
        debug_symbols=run.debug_symbols,
        simt_evidence=run.simt_evidence,
        kernel_scale_instrumentation=bool(run.kernel_scale_details.get("detected")),
        build_provenance=run.build_provenance,
        environment_context=run.environment_context,
        blocks=run.block_entries,
        run_status=run_status,
        run_reason=run.circuit_breaker_reason,
        preflight=run.preflight,
        timeout_profile=run.timeout_profile,
        executable_context=run.executable_context,
    ))
    write_json(run.output_root / "collection_manifest.json", manifest)
    return run_status


def _run_summary(run: Any, timing_summary: Mapping[str, Any], run_status: str) -> Dict[str, Any]:
    args = run.args
    blocks: List[Dict[str, Any]] = []
    for entry in run.block_entries:
        blocks.append({
            "block_id": entry["block_id"], "metric": entry.get("metric"),
            "status": entry["status"], "reason": entry.get("reason"), "coverage": entry.get("coverage"),
        })
    return {
        "schema": CONTRACT_SCHEMA,
        "output": str(run.output_root),
        "manifest": str(run.output_root / "collection_manifest.json"),
        "preset": args.preset,
        "selection_mode": "features" if args.feature else "preset",
        "requested_features": run.requested_features,
        "simt_evidence": run.simt_evidence,
        "environment_context": run.environment_context,
        "run_status": run_status,
        "run_reason": run.circuit_breaker_reason,
        "preflight": run.preflight,
        "timeout_profile": run.timeout_profile,
        "diagnostics": run.circuit_breaker_diagnostics,
        "timing": timing_summary,
        "blocks": blocks,
    }


def _finalize(run: Any) -> int:
    run_status = _write_final_manifest(run)
    timing_summary = _build_timing_summary(run)
    failures = [
        b for b in run.block_entries
        if b["block_id"] != "discovery" and b["status"] in {"failed", "unavailable", "empty"}
    ]
    cli_logger.info(json.dumps(_run_summary(run, timing_summary, run_status), ensure_ascii=False, indent=2))

    if run.circuit_breaker_reason:
        run.run_guard.finalize("aborted", run.circuit_breaker_reason, {
            "diagnostics": run.circuit_breaker_diagnostics,
        })
        return 3
    run.run_guard.finalize("completed", extra={
        "manifest": "collection_manifest.json",
        "block_count": len(run.block_entries),
    })
    if run.args.strict and failures:
        return 2
    return 0


def _run(args: argparse.Namespace) -> int:
    run = SimpleNamespace(
        args=args,
        collector_wall_started=time.perf_counter(),
        local_phase_seconds={},
        circuit_breaker_reason=None,
        circuit_breaker_diagnostics=None,
    )
    _plan_requested(run)
    _prepare_output(run)
    _resolve_environment(run)
    _resolve_executable(run)
    _check_capabilities(run)
    preflight_rc = _run_preflight(run)
    if preflight_rc is not None:
        return preflight_rc
    _derive_timeouts(run)
    _detect_capabilities(run)
    discovery_rc = _run_discovery(run)
    if discovery_rc is not None:
        return discovery_rc
    _resolve_operator_identity(run)
    _write_context_files(run)
    _execute_blocks(run)
    return _finalize(run)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    if args.list_features:
        print_feature_list()
        return 0
    try:
        if args.explain:
            print_feature_explanation(args.explain)
            return 0
        if not args.operator_path or not args.output:
            parser.error("--operator-path and --output are required unless using --list-features or --explain.")
        return _run(args)
    except rtguard.UsageError as exc:
        rtguard.log_usage_error(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
