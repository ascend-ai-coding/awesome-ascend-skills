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

import contextlib
import hashlib
import json
import math
import os
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TextIO, cast

from bench_runner import normalize_bench_mode, run_local_probe, run_remote_probe
from perf_artifacts import (
    MetricSource,
    parse_perf_pair_for_comparison,
)

_PER_CASE_IMPROVEMENT_THRESHOLD = 1.01
_PER_CASE_REGRESSION_THRESHOLD = 0.99
_GEOMEAN_GAIN_THRESHOLD = 1.10
_GEOMEAN_REGRESSION_THRESHOLD = 0.95
_SINGLE_CASE_GAIN_BLOCK_THRESHOLD = 0.92
_SINGLE_CASE_REGRESSION_THRESHOLD = 0.85

_PROBE_SCHEMA_VERSION = 1
_PROBE_WARMUP_CAP = 1
_PROBE_REPEATS_CAP = 3
_PROBE_CACHE_DIR_NAME = ".triton-agent"
_BASELINE_PERF_NAME = "baseline_probe_perf.txt"
_BASELINE_SIDECAR_NAME = "baseline_probe_perf.meta.json"
_BASELINE_LOCK_NAME = "baseline_probe_perf.lock"
_CANDIDATE_PERF_PREFIX = "candidate_probe_perf."
_BASELINE_SNAPSHOT_PREFIX = "baseline_probe_snapshot."
_LOCK_RETRY_SECONDS = 1.0
_LOCK_RETRY_ATTEMPTS = 1800
_LOCK_STALE_SECONDS = 1800


@dataclass(frozen=True)
class ProbeComparison:
    classification: str
    geomean_speedup: float
    avg_improvement_pct: float
    improved_cases: int
    regressed_cases: int
    min_case_speedup: float
    metric_source_resolved: str
    case_speedups: dict[str, float]


@dataclass(frozen=True)
class ProbeBenchResult:
    return_code: int
    default_lines: list[str]
    verbose_lines: list[str]
    warnings: list[str]
    remote_workspace: str | None


def per_case_direction(baseline_value: float, compare_value: float) -> str:
    speedup = baseline_value / compare_value
    if speedup > _PER_CASE_IMPROVEMENT_THRESHOLD:
        return "improved"
    if speedup < _PER_CASE_REGRESSION_THRESHOLD:
        return "regressed"
    return "unchanged"


def classify_probe_result(
    *,
    geomean_speedup: float,
    improved_cases: int,
    regressed_cases: int,
    min_case_speedup: float,
) -> str:
    if (
        geomean_speedup >= _GEOMEAN_GAIN_THRESHOLD
        and improved_cases > regressed_cases
        and min_case_speedup >= _SINGLE_CASE_GAIN_BLOCK_THRESHOLD
    ):
        return "likely_gain"
    if (
        geomean_speedup <= _GEOMEAN_REGRESSION_THRESHOLD
        or min_case_speedup <= _SINGLE_CASE_REGRESSION_THRESHOLD
    ):
        return "likely_regression"
    return "inconclusive"


def compute_probe_comparison(
    baseline_values: dict[str, float],
    candidate_values: dict[str, float],
    *,
    comparison_modes: dict[str, str] | None = None,
) -> ProbeComparison:
    if set(baseline_values) != set(candidate_values):
        raise ValueError("probe comparison requires matching case ids between baseline and candidate")
    comparable_ids = sorted(baseline_values)
    if not comparable_ids:
        raise ValueError("probe comparison requires at least one comparable case id")
    case_speedups: dict[str, float] = {}
    improved = 0
    regressed = 0
    improvement_pcts: list[float] = []
    for latency_id in comparable_ids:
        baseline_value = baseline_values[latency_id]
        candidate_value = candidate_values[latency_id]
        speedup = baseline_value / candidate_value
        case_speedups[latency_id] = speedup
        improvement_pcts.append((speedup - 1.0) * 100.0)
        direction = per_case_direction(baseline_value, candidate_value)
        if direction == "improved":
            improved += 1
        elif direction == "regressed":
            regressed += 1
    geomean = math.exp(
        sum(math.log(speedup) for speedup in case_speedups.values())
        / len(case_speedups)
    )
    avg_improvement_pct = sum(improvement_pcts) / len(improvement_pcts)
    min_speedup = min(case_speedups.values())
    classification = classify_probe_result(
        geomean_speedup=geomean,
        improved_cases=improved,
        regressed_cases=regressed,
        min_case_speedup=min_speedup,
    )
    return ProbeComparison(
        classification=classification,
        geomean_speedup=geomean,
        avg_improvement_pct=avg_improvement_pct,
        improved_cases=improved,
        regressed_cases=regressed,
        min_case_speedup=min_speedup,
        metric_source_resolved=_resolve_metric_source(comparison_modes),
        case_speedups=case_speedups,
    )


def _resolve_metric_source(comparison_modes: dict[str, str] | None) -> str:
    if not comparison_modes:
        return "auto"
    distinct = set(comparison_modes.values())
    if len(distinct) == 1:
        return next(iter(distinct))
    return "mixed"


_CACHE_MATCH_KEYS: tuple[str, ...] = (
    "schema_version",
    "measurement_profile",
    "probe_contract",
    "baseline_operator_fingerprint",
    "bench_file_fingerprint",
    "bench_mode",
    "remote",
    "remote_workdir",
    "npu_devices",
)


def cache_is_valid(
    stored_sidecar: dict[str, object] | None,
    expected: dict[str, object],
    perf_file_exists: bool,
) -> tuple[bool, str]:
    if stored_sidecar is None:
        return False, "missing sidecar"
    if not perf_file_exists:
        return False, "missing perf file"
    for key in _CACHE_MATCH_KEYS:
        if stored_sidecar.get(key) != expected.get(key):
            return False, f"mismatched {key}"
    if stored_sidecar.get("bench_cases_fingerprint") != expected.get("bench_cases_fingerprint"):
        return False, "mismatched bench_cases_fingerprint"
    return True, "hit"


def _compute_fingerprint(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def _resolve_bench_cases_file(bench_file: Path) -> Path | None:
    candidate = bench_file.with_suffix(".json")
    return candidate if candidate.exists() else None


def build_probe_sidecar(request: _ProbeRequest, bench_cases_file: Path | None) -> dict[str, object]:
    metadata: dict[str, object] = {
        "schema_version": _PROBE_SCHEMA_VERSION,
        "measurement_profile": "probe",
        "probe_contract": {
            "name": "fast-probe",
            "warmup_cap": _PROBE_WARMUP_CAP,
            "repeats_cap": _PROBE_REPEATS_CAP,
        },
        "baseline_operator_file": str(request.baseline_operator_file),
        "baseline_operator_fingerprint": _compute_fingerprint(request.baseline_operator_file),
        "bench_file": str(request.bench_file),
        "bench_file_fingerprint": _compute_fingerprint(request.bench_file),
        "bench_mode": request.bench_mode,
        "remote": request.remote,
        "remote_workdir": request.remote_workdir,
        "npu_devices": request.npu_devices,
        "bench_cases_file": str(bench_cases_file) if bench_cases_file is not None else None,
        "bench_cases_fingerprint": _compute_fingerprint(bench_cases_file) if bench_cases_file is not None else None,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return metadata


def _read_sidecar(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(loaded, dict):
        return None
    return cast(dict[str, object], loaded)


def _write_sidecar(path: Path, metadata: dict[str, object]) -> None:
    _atomic_write_text(path, json.dumps(metadata, indent=2, sort_keys=True))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + f".tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    try:
        tmp.write_bytes(src.read_bytes())
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


@contextlib.contextmanager
def _baseline_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    acquired = False
    for _ in range(_LOCK_RETRY_ATTEMPTS):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode=0o600)
            os.close(fd)
            acquired = True
            break
        except FileExistsError:
            if _lock_is_stale(lock_path):
                lock_path.unlink(missing_ok=True)
                continue
            time.sleep(_LOCK_RETRY_SECONDS)
    if not acquired:
        raise RuntimeError(f"could not acquire baseline probe cache lock: {lock_path}")
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _lock_is_stale(lock_path: Path) -> bool:
    try:
        age = time.time() - lock_path.stat().st_mtime
    except OSError:
        return False
    return age >= _LOCK_STALE_SECONDS


def _probe_cache_dir() -> Path:
    return Path.cwd() / _PROBE_CACHE_DIR_NAME


def _candidate_perf_path(cache_dir: Path) -> Path:
    token = uuid.uuid4().hex[:8]
    return cache_dir / f"{_CANDIDATE_PERF_PREFIX}{os.getpid()}.{token}.txt"


def _baseline_snapshot_path(cache_dir: Path) -> Path:
    token = uuid.uuid4().hex[:8]
    return cache_dir / f"{_BASELINE_SNAPSHOT_PREFIX}{os.getpid()}.{token}.txt"


def _format_speedup(value: float) -> str:
    return f"{value:.2f}x"


def _format_percent_value(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}%"


def render_probe_output(comparison: ProbeComparison, **values: object) -> tuple[list[str], list[str], list[str]]:
    cache_hit = bool(values["cache_hit"])
    baseline_perf_path = cast(Path, values["baseline_perf_path"])
    candidate_perf_path = cast(Path, values["candidate_perf_path"])
    mismatch_reason = cast(str | None, values.get("mismatch_reason"))
    verbose = bool(values["verbose"])
    default_lines: list[str] = [
        f"Probe classification: {comparison.classification}",
        f"Metric source: {comparison.metric_source_resolved}",
        f"Advisory geomean speedup: {_format_speedup(comparison.geomean_speedup)}",
        f"Advisory avg improvement: {_format_percent_value(comparison.avg_improvement_pct)}",
        f"Advisory improved cases: {comparison.improved_cases}",
        f"Advisory regressed cases: {comparison.regressed_cases}",
        _summary_line(comparison.classification),
    ]
    warnings: list[str] = []
    if comparison.metric_source_resolved == "mixed":
        warnings.append("metric source resolved to mixed across cases; treat classification with caution")
    verbose_lines: list[str] = []
    if verbose:
        verbose_lines.append(f"Baseline cache: {'hit' if cache_hit else 'miss'}")
        if mismatch_reason:
            verbose_lines.append(f"Cache miss reason: {mismatch_reason}")
        verbose_lines.append(f"Baseline probe perf: {baseline_perf_path}")
        verbose_lines.append(f"Candidate probe perf: {candidate_perf_path}")
    return default_lines, verbose_lines, warnings


def _summary_line(classification: str) -> str:
    if classification == "likely_gain":
        return (
            "Summary: Fast probe indicates a likely gain over the baseline. Use canonical "
            "run-bench before recording any official speedup."
        )
    if classification == "likely_regression":
        return (
            "Summary: Fast probe indicates a likely regression versus the baseline. Use "
            "canonical run-bench before recording any official conclusion."
        )
    return (
        "Summary: Fast probe signal is inconclusive. Use canonical run-bench before "
        "recording any official conclusion."
    )


def _compare_probe_artifacts(baseline_perf_path: Path, candidate_perf_path: Path, **values: object) -> ProbeBenchResult:
    baseline_read_path = cast(Path, values["baseline_read_path"])
    metric_source = str(values["metric_source"])
    cache_hit = bool(values["cache_hit"])
    mismatch_reason = cast(str | None, values.get("mismatch_reason"))
    verbose = bool(values["verbose"])
    extra_warnings = cast(list[str], values["extra_warnings"])
    comparison, failure = _load_probe_comparison(
        baseline_read_path,
        candidate_perf_path,
        metric_source,
    )
    if failure is not None:
        return _probe_comparison_failure(failure, extra_warnings)
    if comparison is None:
        return _probe_comparison_failure("probe comparison produced no result", extra_warnings)
    default_lines, verbose_lines, warnings = render_probe_output(
        comparison,
        cache_hit=cache_hit,
        baseline_perf_path=baseline_perf_path,
        candidate_perf_path=candidate_perf_path,
        mismatch_reason=mismatch_reason,
        verbose=verbose,
    )
    return ProbeBenchResult(
        return_code=0,
        default_lines=default_lines,
        verbose_lines=verbose_lines,
        warnings=warnings + list(extra_warnings),
        remote_workspace=None,
    )


def _load_probe_comparison(
    baseline_read_path: Path,
    candidate_perf_path: Path,
    metric_source: str,
) -> tuple[ProbeComparison | None, str | None]:
    try:
        baseline_values, candidate_values, comparison_modes = parse_perf_pair_for_comparison(
            baseline_read_path,
            candidate_perf_path,
            metric_source=cast(MetricSource, metric_source),
        )
        return compute_probe_comparison(
            baseline_values,
            candidate_values,
            comparison_modes={
                latency_id: ("kernel" if mode == "latency" else mode)
                for latency_id, mode in comparison_modes.items()
            },
        ), None
    except ValueError as exc:
        return None, str(exc)


def _probe_comparison_failure(error: str, warnings: list[str]) -> ProbeBenchResult:
    return ProbeBenchResult(
        return_code=1,
        default_lines=[f"FAIL: {error}"],
        verbose_lines=[],
        warnings=list(warnings),
        remote_workspace=None,
    )


@dataclass(frozen=True)
class _LocalRun:
    payload: dict[str, object]
    perf_path: Path | None


@dataclass(frozen=True)
class _RemoteRun:
    payload: dict[str, object]
    perf_path: Path | None
    remote_workspace: str


@dataclass(frozen=True)
class _ProbeRequest:
    bench_file: Path
    operator_file: Path
    baseline_operator_file: Path
    bench_mode: str
    metric_source: str
    npu_devices: str | None
    verbose: bool
    remote: str | None
    remote_workdir: str | None
    keep_remote_workdir: bool
    stderr: TextIO | None


@dataclass(frozen=True)
class _ProbePaths:
    baseline_perf: Path
    baseline_sidecar: Path
    baseline_lock: Path
    candidate_perf: Path
    baseline_snapshot: Path


def _run_one_probe(request: _ProbeRequest, *, warmup_cap: int, repeats_cap: int) -> _LocalRun | _RemoteRun:
    if request.remote is not None:
        payload, perf_path, remote_workspace = run_remote_probe(
            request.bench_file,
            request.operator_file,
            request.bench_mode,
            request.remote,
            request.remote_workdir,
            warmup_cap=warmup_cap,
            repeats_cap=repeats_cap,
            npu_devices=request.npu_devices,
            keep_remote_workdir=request.keep_remote_workdir,
            verbose=request.verbose,
            stderr=request.stderr,
        )
        return _RemoteRun(payload=dict(payload), perf_path=perf_path, remote_workspace=remote_workspace)
    payload, perf_path = run_local_probe(
        request.bench_file,
        request.operator_file,
        request.bench_mode,
        warmup_cap=warmup_cap,
        repeats_cap=repeats_cap,
        npu_devices=request.npu_devices,
        verbose=request.verbose,
        output=None,
    )
    return _LocalRun(payload=dict(payload), perf_path=perf_path)


def _remote_verbose(payload: dict[str, object], verbose: bool) -> list[str]:
    lines: list[str] = []
    if not verbose:
        return lines
    stderr_text = str(payload.get("stderr") or "")
    if stderr_text:
        lines.append(stderr_text.rstrip())
    return lines


def _build_probe_request(values: dict[str, object]) -> _ProbeRequest:
    bench_file = cast(Path, values["bench_file"])
    operator_file = cast(Path, values["operator_file"])
    baseline_operator_file = cast(Path, values["baseline_operator_file"])
    bench_mode = str(values["bench_mode"])
    metric_source = str(values["metric_source"])
    npu_devices = cast(str | None, values.get("npu_devices"))
    verbose = bool(values.get("verbose", False))
    remote = cast(str | None, values.get("remote"))
    remote_workdir = cast(str | None, values.get("remote_workdir"))
    keep_remote_workdir = bool(values.get("keep_remote_workdir", False))
    stderr = cast(TextIO | None, values.get("stderr"))
    return _ProbeRequest(
        bench_file=bench_file.resolve(),
        operator_file=operator_file.resolve(),
        baseline_operator_file=baseline_operator_file.resolve(),
        bench_mode=normalize_bench_mode(bench_mode),
        metric_source=metric_source,
        npu_devices=npu_devices,
        verbose=verbose,
        remote=remote,
        remote_workdir=remote_workdir,
        keep_remote_workdir=keep_remote_workdir,
        stderr=stderr,
    )


def _probe_paths() -> _ProbePaths:
    cache_dir = _probe_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return _ProbePaths(
        baseline_perf=cache_dir / _BASELINE_PERF_NAME,
        baseline_sidecar=cache_dir / _BASELINE_SIDECAR_NAME,
        baseline_lock=cache_dir / _BASELINE_LOCK_NAME,
        candidate_perf=_candidate_perf_path(cache_dir),
        baseline_snapshot=_baseline_snapshot_path(cache_dir),
    )


def _probe_warnings(request: _ProbeRequest) -> list[str]:
    if request.bench_mode == "torch-npu-profiler":
        return []
    return [
        "probe warmup/repeats caps apply only to torch-npu-profiler mode; "
        "this run used canonical benchmark execution"
    ]


def _prepare_probe_baseline(
    request: _ProbeRequest,
    paths: _ProbePaths,
    warnings: list[str],
) -> tuple[bool, str | None] | ProbeBenchResult:
    bench_cases_file = _resolve_bench_cases_file(request.bench_file)
    expected_sidecar = build_probe_sidecar(request, bench_cases_file)
    with _baseline_lock(paths.baseline_lock):
        valid, reason = cache_is_valid(
            _read_sidecar(paths.baseline_sidecar),
            expected_sidecar,
            paths.baseline_perf.exists(),
        )
        if not valid:
            baseline_request = replace(request, operator_file=request.baseline_operator_file)
            baseline_result = _run_one_probe(
                baseline_request,
                warmup_cap=_PROBE_WARMUP_CAP,
                repeats_cap=_PROBE_REPEATS_CAP,
            )
            failure = _probe_run_failure(
                baseline_result,
                "baseline",
                warnings,
                request.verbose,
            )
            if failure is not None:
                return failure
            baseline_perf = cast(Path, baseline_result.perf_path)
            _atomic_copy(baseline_perf, paths.baseline_perf)
            _write_sidecar(paths.baseline_sidecar, expected_sidecar)
        _atomic_copy(paths.baseline_perf, paths.baseline_snapshot)
    return valid, None if valid else reason


def _probe_run_failure(
    run: _LocalRun | _RemoteRun,
    role: str,
    warnings: list[str],
    verbose: bool,
) -> ProbeBenchResult | None:
    remote_workspace = run.remote_workspace if isinstance(run, _RemoteRun) else None
    if run.payload.get("return_code") != 0:
        return ProbeBenchResult(
            return_code=1,
            default_lines=[f"FAIL: {role} probe execution failed"],
            verbose_lines=_remote_verbose(run.payload, verbose),
            warnings=list(warnings),
            remote_workspace=remote_workspace,
        )
    if run.perf_path is None:
        return ProbeBenchResult(
            return_code=1,
            default_lines=[f"FAIL: {role} probe produced no perf artifact"],
            verbose_lines=_remote_verbose(run.payload, verbose),
            warnings=list(warnings),
            remote_workspace=remote_workspace,
        )
    return None


def _run_probe_candidate(
    request: _ProbeRequest,
    paths: _ProbePaths,
    warnings: list[str],
    cache_hit: bool,
    mismatch_reason: str | None,
) -> ProbeBenchResult:
    candidate_result = _run_one_probe(
        request,
        warmup_cap=_PROBE_WARMUP_CAP,
        repeats_cap=_PROBE_REPEATS_CAP,
    )
    failure = _probe_run_failure(candidate_result, "candidate", warnings, request.verbose)
    if failure is not None:
        return failure
    candidate_perf = cast(Path, candidate_result.perf_path)
    _atomic_copy(candidate_perf, paths.candidate_perf)
    result = _compare_probe_artifacts(
        paths.baseline_perf,
        paths.candidate_perf,
        baseline_read_path=paths.baseline_snapshot,
        metric_source=request.metric_source,
        cache_hit=cache_hit,
        mismatch_reason=mismatch_reason,
        verbose=request.verbose,
        extra_warnings=warnings,
    )
    remote_workspace = (
        candidate_result.remote_workspace
        if isinstance(candidate_result, _RemoteRun)
        else None
    )
    return replace(result, remote_workspace=remote_workspace)


def _execute_probe(values: dict[str, object]) -> ProbeBenchResult:
    request = _build_probe_request(values)
    paths = _probe_paths()
    warnings = _probe_warnings(request)
    baseline = _prepare_probe_baseline(request, paths, warnings)
    if isinstance(baseline, ProbeBenchResult):
        return baseline
    cache_hit, mismatch_reason = baseline
    try:
        return _run_probe_candidate(request, paths, warnings, cache_hit, mismatch_reason)
    finally:
        paths.baseline_snapshot.unlink(missing_ok=True)


def run_local_probe_bench(*args: object, **kwargs: object) -> ProbeBenchResult:
    if len(args) >= 4:
        bench_file, operator_file, baseline_operator_file, bench_mode = args[:4]
    else:
        values = tuple(
            kwargs[k]
            for k in ("bench_file", "operator_file", "baseline_operator_file", "bench_mode")
        )
        bench_file, operator_file, baseline_operator_file, bench_mode = values
    metric_source = str(kwargs.get("metric_source", "auto"))
    npu_devices = cast(str | None, kwargs.get("npu_devices"))
    verbose = bool(kwargs.get("verbose", False))
    return _execute_probe(locals())


def run_remote_probe_bench(*args: object, **kwargs: object) -> ProbeBenchResult:
    names = ("bench_file", "operator_file", "baseline_operator_file", "bench_mode", "remote", "remote_workdir")
    values = list(args) + [kwargs[name] for name in names[len(args):]]
    bench_file, operator_file, baseline_operator_file, bench_mode, remote, remote_workdir = values
    metric_source = str(kwargs.get("metric_source", "auto"))
    npu_devices = cast(str | None, kwargs.get("npu_devices"))
    keep_remote_workdir = bool(kwargs.get("keep_remote_workdir", False))
    verbose = bool(kwargs.get("verbose", False))
    stderr = cast(TextIO | None, kwargs.get("stderr"))
    return _execute_probe(locals())
