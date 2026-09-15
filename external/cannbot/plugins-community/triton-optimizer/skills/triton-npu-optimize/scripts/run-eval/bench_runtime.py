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
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
import importlib
import importlib.util
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, cast

from bench_contract import KernelResolution, resolve_bench_kernel_resolution
from perf_artifacts import (
    PerfCaseRecord,
    PerfMetrics,
    perf_output_path,
    render_perf_case_records_jsonl,
    write_perf_lines,
)
from profile_csv_parser import (
    ParsedProfileCsvRows,
    find_latest_op_statistic_csv,
    find_optional_profile_csv,
    parse_kernel_details_csv,
    parse_op_statistic_csv,
    resolve_perf_metrics,
)
from result_payload import ResultPayload, make_result
from runtime_bootstrap import bootstrap_torch_npu


# ---------------------------------------------------------------------------
# Constants & data model
# ---------------------------------------------------------------------------

LoadedBenchCases = tuple[list["BenchCase"], KernelResolution]
RuntimeBenchResult = tuple[ResultPayload, Path]
ProfileCaseOutcome = tuple[PerfMetrics | None, str | None]
ResolvedProfileOutputRoot = tuple[str | None, str]
PreservedRunDir = tuple[Path, tempfile.TemporaryDirectory[str] | None]

WARMUP_DEFAULT = 5
REPEATS_DEFAULT = 50
_MISSING_KERNEL_MATCH_ERROR = "no resolved kernels matched profiler kernel view"
_LOCAL_BENCH_OUTPUT_DIR_ENV = "TRITON_AGENT_BENCH_OUTPUT_DIR"


@dataclass(frozen=True)
class BenchCase:
    case_id: str
    fn: Callable[[], object]
    warmup: int
    repeats: int
    case_data: Mapping[str, object]


# ---------------------------------------------------------------------------
# Module loading & case construction
# ---------------------------------------------------------------------------

def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(f"{module_name}_{time.time_ns()}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _require_callable(module: object, name: str, bench_file: Path) -> Callable[..., Any]:
    candidate = getattr(module, name, None)
    if not callable(candidate):
        raise ValueError(f"Benchmark module missing required hook '{name}': {bench_file}")
    return candidate


def _normalize_cases(
    raw_cases: object,
    operator_api: object,
    build_case_fn: Callable[[object, Mapping[str, object]], object],
) -> list[BenchCase]:
    if isinstance(raw_cases, (str, bytes)) or isinstance(raw_cases, Mapping) or not isinstance(raw_cases, Iterable):
        raise ValueError("Benchmark hook 'build_bench_cases' must return an iterable of cases")
    cases: list[BenchCase] = []
    seen_case_ids: set[str] = set()
    for raw_case in cast(Iterable[object], raw_cases):
        if not isinstance(raw_case, Mapping):
            raise ValueError("Benchmark cases must be mappings")
        case_map = cast(Mapping[str, object], raw_case)
        case_id = case_map.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("Benchmark case is missing required string field 'id'")
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate benchmark case id: {case_id}")
        warmup = _normalize_non_negative_int(case_map.get("warmup", WARMUP_DEFAULT), "warmup", case_id)
        repeats = _normalize_positive_int(case_map.get("repeats", REPEATS_DEFAULT), "repeats", case_id)
        seen_case_ids.add(case_id)
        cases.append(
            BenchCase(
                case_id=case_id,
                fn=_resolve_case_fn(operator_api, build_case_fn, case_map, case_id),
                warmup=warmup,
                repeats=repeats,
                case_data=case_map,
            )
        )
    if not cases:
        raise ValueError("Benchmark hook 'build_bench_cases' returned no cases")
    return cases


def _resolve_case_fn(
    operator_api: object,
    build_case_fn: Callable[[object, Mapping[str, object]], object],
    case_map: Mapping[str, object],
    case_id: str,
) -> Callable[[], object]:
    candidate = build_case_fn(operator_api, case_map)
    if not callable(candidate):
        raise ValueError(f"Benchmark hook 'build_bench_case_fn' for case '{case_id}' must return a callable")
    return cast(Callable[[], object], candidate)


def _normalize_non_negative_int(value: object, field_name: str, case_id: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"Benchmark case '{case_id}' field '{field_name}' must be an integer")
    if value < 0:
        raise ValueError(f"Benchmark case '{case_id}' field '{field_name}' must be >= 0")
    return value


def _normalize_positive_int(value: object, field_name: str, case_id: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"Benchmark case '{case_id}' field '{field_name}' must be an integer")
    if value <= 0:
        raise ValueError(f"Benchmark case '{case_id}' field '{field_name}' must be > 0")
    return value


def load_bench_cases(
    bench_file: Path,
    operator_file: Path,
) -> LoadedBenchCases:
    bench_path = bench_file.resolve()
    operator_path = operator_file.resolve()
    bootstrap_torch_npu()
    bench_module = _load_module(bench_path, f"bench_runtime_bench_{bench_path.stem}")
    operator_module = _load_module(operator_path, f"bench_runtime_operator_{operator_path.stem}")
    build_operator_api = _require_callable(bench_module, "build_operator_api", bench_path)
    build_cases = _require_callable(bench_module, "build_bench_cases", bench_path)
    build_case_fn = _require_callable(bench_module, "build_bench_case_fn", bench_path)
    operator_api = build_operator_api(operator_module)
    raw_cases = build_cases()
    return _normalize_cases(raw_cases, operator_api, build_case_fn), resolve_bench_kernel_resolution(
        bench_path,
        operator_path,
    )


def select_bench_case(cases: list[BenchCase], case_id: str | None) -> BenchCase:
    if case_id is None:
        if len(cases) == 1:
            return cases[0]
        available = ", ".join(case.case_id for case in cases)
        raise ValueError(
            "Benchmark profiling requires --case-id when multiple cases exist. "
            f"Available case ids: {available}"
        )
    for case in cases:
        if case.case_id == case_id:
            return case
    available = ", ".join(case.case_id for case in cases)
    raise ValueError(f"Unknown benchmark case id '{case_id}'. Available case ids: {available}")


# ---------------------------------------------------------------------------
# Bench execution (public API)
# ---------------------------------------------------------------------------

def execute_bench_case(
    bench_file: Path,
    operator_file: Path,
    case_id: str | None = None,
    *,
    iterations: int = 1,
) -> ResultPayload:
    import torch
    cases, _resolution = load_bench_cases(bench_file, operator_file)
    case = select_bench_case(cases, case_id)
    for _ in range(iterations):
        case.fn()
        _synchronize(torch)
    return make_result(return_code=0, stdout="", stderr="")


def profile_bench_case(*args: object, **kwargs: object) -> PerfCaseRecord:
    bench_file = cast(Path, args[0] if args else kwargs["bench_file"])
    operator_file = cast(Path, args[1] if len(args) > 1 else kwargs["operator_file"])
    case_id = cast(str | None, args[2] if len(args) > 2 else kwargs.get("case_id"))
    preserved_run_dir = cast(Path | None, kwargs.get("preserved_run_dir"))
    verbose = bool(kwargs.get("verbose", False))
    preloaded = cast(LoadedBenchCases | None, kwargs.get("preloaded"))
    cases, resolution = preloaded or load_bench_cases(bench_file, operator_file)
    case = select_bench_case(cases, case_id)
    return _run_bench_case(
        case,
        resolution,
        preserved_run_dir=preserved_run_dir,
        cleanup_workdir=bench_file.parent,
        verbose=verbose,
    )


def profile_bench_case_quick(
    bench_file: Path,
    operator_file: Path,
    case_id: str | None,
) -> ResultPayload:
    cases, resolution = load_bench_cases(bench_file, operator_file)
    case = select_bench_case(cases, case_id)
    profile_root = _profile_output_root(bench_file.parent, case.case_id)
    _metrics, error_message = _profile_case_with_profiler(case, resolution, profile_root)
    if error_message is not None:
        return make_result(return_code=1, stdout="", stderr=error_message)
    return make_result(return_code=0, stdout="", stderr="")


def profile_all_bench_cases(
    bench_file: Path,
    operator_file: Path,
    *,
    verbose: bool = False,
    output: str | None = None,
    preloaded: LoadedBenchCases | None = None,
) -> RuntimeBenchResult:
    prev = os.environ.get("TRITON_ALWAYS_COMPILE")
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    try:
        cases, resolution = preloaded or load_bench_cases(bench_file, operator_file)
        case_records: list[PerfCaseRecord] = []
        preserved_run_dir = _create_local_preserved_profile_run_dir(prefix="triton-agent-bench-")

        for case in cases:
            record = profile_bench_case(
                bench_file,
                operator_file,
                case.case_id,
                preserved_run_dir=preserved_run_dir,
                verbose=verbose,
                preloaded=(cases, resolution),
            )
            case_records.append(record)

        perf_path = _resolve_output_path(operator_file, output=output)
        write_perf_lines(
            perf_path,
            render_perf_case_records_jsonl(
                case_records,
                missing_kernel_match_error=_MISSING_KERNEL_MATCH_ERROR,
            ),
        )
        return _records_result(case_records), perf_path
    finally:
        if prev is None:
            del os.environ["TRITON_ALWAYS_COMPILE"]
        else:
            os.environ["TRITON_ALWAYS_COMPILE"] = prev


# ---------------------------------------------------------------------------
# Perf-counter timing (no profiler)
# ---------------------------------------------------------------------------


def _time_case_iterations(
    *,
    fn: Callable[[], object],
    warmup: int,
    repeats: int,
) -> PerfMetrics:
    import torch
    for _ in range(warmup):
        fn()
        _synchronize(torch)
    iteration_times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        _synchronize(torch)
        iteration_times.append(time.perf_counter() - t0)
    avg_us = statistics.mean(iteration_times) * 1_000_000
    return {"kernel_avg_time_us": avg_us, "ops": []}


def _time_bench_case(
    case: BenchCase,
    resolution: KernelResolution,
    *,
    bench_mode: str,
) -> PerfCaseRecord:
    t0 = time.monotonic()
    try:
        metrics = _time_case_iterations(
            fn=case.fn,
            warmup=case.warmup,
            repeats=case.repeats,
        )
        elapsed = time.monotonic() - t0
        return PerfCaseRecord(
            case_label=case.case_id,
            kernel_names=resolution.kernel_names,
            kernel_source=resolution.kernel_source,
            metrics=metrics,
            case_wall_clock_seconds=elapsed,
            bench_mode=bench_mode,
        )
    except Exception as exc:
        elapsed = time.monotonic() - t0
        return PerfCaseRecord(
            case_label=case.case_id,
            kernel_names=resolution.kernel_names,
            kernel_source=resolution.kernel_source,
            error_message=f"{type(exc).__name__}: {exc}",
            case_wall_clock_seconds=elapsed,
            bench_mode=bench_mode,
        )


def time_all_bench_cases(
    bench_file: Path,
    operator_file: Path,
    *,
    bench_mode: str = "perf-counter",
    output: str | None = None,
) -> RuntimeBenchResult:
    prev = os.environ.get("TRITON_ALWAYS_COMPILE")
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    try:
        cases, resolution = load_bench_cases(bench_file, operator_file)
        case_records: list[PerfCaseRecord] = []

        for case in cases:
            record = _time_bench_case(
                case,
                resolution,
                bench_mode=bench_mode,
            )
            case_records.append(record)

        perf_path = _resolve_output_path(operator_file, output=output)
        write_perf_lines(
            perf_path,
            render_perf_case_records_jsonl(
                case_records,
            ),
        )
        return _records_result(case_records), perf_path
    finally:
        if prev is None:
            del os.environ["TRITON_ALWAYS_COMPILE"]
        else:
            os.environ["TRITON_ALWAYS_COMPILE"] = prev


def _records_result(case_records: list[PerfCaseRecord]) -> ResultPayload:
    failures = [record for record in case_records if record.error_message is not None]
    return make_result(
        return_code=1 if failures else 0,
        stdout="",
        stderr="\n".join(f"{record.case_label}: {record.error_message}" for record in failures),
    )


# ---------------------------------------------------------------------------
# Profiling internals
# ---------------------------------------------------------------------------


def _profile_case_with_profiler(
    case: BenchCase,
    resolution: KernelResolution,
    profile_root: Path,
    *,
    verbose: bool = False,
) -> ProfileCaseOutcome:
    try:
        import torch
        torch_npu = cast(Any, importlib.import_module("torch_npu"))
    except ImportError as exc:
        return None, f"Missing profiler dependency: {exc}"
    try:
        _reset_profile_root(profile_root)
        _run_torch_npu_profile(case, torch, torch_npu, profile_root, verbose)
        metrics = _read_profiler_metrics(profile_root, case.repeats, resolution.kernel_names, verbose=verbose)
        return metrics, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _reset_profile_root(profile_root: Path) -> None:
    try:
        profile_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        shutil.rmtree(profile_root, ignore_errors=True)
        profile_root.mkdir(parents=True, exist_ok=False)


def _run_torch_npu_profile(
    case: BenchCase,
    torch_module: Any,
    torch_npu: Any,
    profile_root: Path,
    verbose: bool,
) -> None:
    profiler_api = torch_npu.profiler
    suppress_context = _suppress_output_streams() if not verbose else nullcontext()
    with suppress_context:
        case.fn()
        _synchronize(torch_module)
        with profiler_api.profile(**_torch_npu_profile_options(profiler_api, case, profile_root)) as profiler:
            _profile_case_steps(case, torch_module, profiler)


def _torch_npu_profile_options(profiler_api: Any, case: BenchCase, profile_root: Path) -> dict[str, Any]:
    return {
        "activities": [profiler_api.ProfilerActivity.NPU, profiler_api.ProfilerActivity.CPU],
        "schedule": profiler_api.schedule(
            wait=0, warmup=case.warmup, active=case.repeats, repeat=1, skip_first=1
        ),
        "on_trace_ready": profiler_api.tensorboard_trace_handler(str(profile_root)),
        "record_shapes": False,
        "profile_memory": False,
        "with_stack": False,
        "with_flops": False,
        "with_modules": False,
    }


def _profile_case_steps(case: BenchCase, torch_module: Any, profiler: Any) -> None:
    for _ in range(1 + case.warmup + case.repeats):
        case.fn()
        _synchronize(torch_module)
        profiler.step()


def _run_bench_case(case: BenchCase, resolution: KernelResolution, **values: object) -> PerfCaseRecord:
    preserved_run_dir = cast(Path | None, values.get("preserved_run_dir"))
    cleanup_workdir = cast(Path, values["cleanup_workdir"])
    verbose = bool(values.get("verbose", False))
    bench_mode = str(values.get("bench_mode", "torch-npu-profiler"))
    profile_root, temp_dir = _create_local_bench_profile_dir(case.case_id, preserved_run_dir)
    try:
        t0 = time.monotonic()
        metrics, error_message = _profile_case_with_profiler(
            case,
            resolution,
            profile_root,
            verbose=verbose,
        )
        elapsed = time.monotonic() - t0
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
        _cleanup_local_bench_extra_info(cleanup_workdir)
    if error_message is not None:
        return PerfCaseRecord(
            case_label=case.case_id,
            kernel_names=resolution.kernel_names,
            kernel_source=resolution.kernel_source,
            error_message=error_message,
            case_wall_clock_seconds=elapsed,
            bench_mode=bench_mode,
        )
    return PerfCaseRecord(
        case_label=case.case_id,
        kernel_names=resolution.kernel_names,
        kernel_source=resolution.kernel_source,
        metrics=metrics,
        case_wall_clock_seconds=elapsed,
        bench_mode=bench_mode,
    )


def _read_profiler_metrics(
    profile_root: Path,
    active_count: int,
    kernel_names: list[str],
    *,
    verbose: bool = False,
) -> PerfMetrics:
    kernel_details_rows = _read_kernel_details_rows(profile_root, active_count, verbose)
    if kernel_details_rows is not None and kernel_details_rows.total_time_us > 0:
        return _resolve_profile_metrics(kernel_details_rows, kernel_names, verbose)
    op_statistic_metrics = _read_op_statistic_metrics(
        profile_root,
        active_count,
        kernel_names,
        verbose,
    )
    if op_statistic_metrics is not None:
        return op_statistic_metrics
    if kernel_details_rows is not None:
        if verbose:
            print("[metrics] kernel_details.csv total_time_us=0, no other CSV found", file=sys.stderr)
        return _resolve_profile_metrics(kernel_details_rows, kernel_names, verbose)
    raise FileNotFoundError(
        f"No kernel_details.csv or op_statistic.csv found under {profile_root}"
    )


def _read_kernel_details_rows(
    profile_root: Path,
    active_count: int,
    verbose: bool,
) -> ParsedProfileCsvRows | None:
    kernel_details_path = find_optional_profile_csv(profile_root, "kernel_details.csv")
    if kernel_details_path is None:
        return None
    if verbose:
        print(f"[metrics] found kernel_details.csv at {kernel_details_path}", file=sys.stderr)
    rows = parse_kernel_details_csv(kernel_details_path, active_count=active_count, verbose=verbose)
    if rows.total_time_us <= 0 and verbose:
        print(
            f"[metrics] kernel_details.csv total_time_us={rows.total_time_us}, "
            "falling back to op_statistic.csv",
            file=sys.stderr,
        )
    return rows


def _read_op_statistic_metrics(
    profile_root: Path,
    active_count: int,
    kernel_names: list[str],
    verbose: bool,
) -> PerfMetrics | None:
    op_statistic_path = find_latest_op_statistic_csv(profile_root)
    if op_statistic_path is None:
        return None
    if verbose:
        print(f"[metrics] found op_statistic.csv at {op_statistic_path}", file=sys.stderr)
    rows = parse_op_statistic_csv(op_statistic_path, active_count=active_count, verbose=verbose)
    return _resolve_profile_metrics(rows, kernel_names, verbose)


def _resolve_profile_metrics(
    rows: ParsedProfileCsvRows,
    kernel_names: list[str],
    verbose: bool,
) -> PerfMetrics:
    return resolve_perf_metrics(
        rows.ops,
        kernel_names,
        total_op_avg_time_us=rows.total_op_avg_time_us,
        verbose=verbose,
    )


@contextmanager
def _suppress_output_streams() -> Iterator[None]:
    with open(os.devnull, "w", encoding="utf-8") as quiet_output:
        stdout_fd = os.dup(1)
        stderr_fd = os.dup(2)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            with redirect_stdout(quiet_output), redirect_stderr(quiet_output):
                os.dup2(quiet_output.fileno(), 1)
                os.dup2(quiet_output.fileno(), 2)
                yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)


def _synchronize(torch_module: Any) -> None:
    if hasattr(torch_module, "npu"):
        torch_module.npu.synchronize()


# ---------------------------------------------------------------------------
# Directory & path management
# ---------------------------------------------------------------------------

def _resolve_output_path(operator_file: Path, *, output: str | None = None) -> Path:
    if output is not None:
        return Path(output).expanduser().resolve()
    return perf_output_path(operator_file)


def _profile_output_root(parent: Path, case_id: str) -> Path:
    return parent / f"PROF_{_sanitize_case_id(case_id)}_{int(time.time() * 1000)}"


def _resolve_local_bench_profile_output_root() -> ResolvedProfileOutputRoot:
    configured_root = os.environ.get(_LOCAL_BENCH_OUTPUT_DIR_ENV)
    if configured_root:
        return str(Path(configured_root).expanduser().resolve()), _LOCAL_BENCH_OUTPUT_DIR_ENV
    return None, _LOCAL_BENCH_OUTPUT_DIR_ENV


def _create_local_preserved_profile_run_dir(prefix: str) -> Path | None:
    configured_root, configured_env = _resolve_local_bench_profile_output_root()
    if not configured_root:
        return None
    root = Path(configured_root).expanduser()
    if root.exists() and not root.is_dir():
        raise ValueError(f"{configured_env} must point to a directory: {root}")
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
        _set_directory_owner_only(root)
    run_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=str(root)))
    _set_directory_owner_only(run_dir)
    return run_dir


def create_local_preserved_profile_run_dir(prefix: str) -> Path | None:
    return _create_local_preserved_profile_run_dir(prefix)


def _create_local_bench_profile_dir(
    case_id: str,
    preserved_run_dir: Path | None,
) -> PreservedRunDir:
    if preserved_run_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix=f"triton-agent-bench-{_sanitize_case_id(case_id)}-")
        return Path(temp_dir.name), temp_dir
    profile_root = preserved_run_dir.resolve() / f"case-{_sanitize_case_id(case_id)}"
    profile_root.mkdir(parents=True, exist_ok=False)
    _set_directory_owner_only(profile_root)
    return profile_root, None


def _set_directory_owner_only(path: Path) -> None:
    path.chmod(0o700)


def _sanitize_case_id(case_id: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in case_id)
    return sanitized or "case"


def _cleanup_local_bench_extra_info(workdir: Path) -> None:
    extra_info_dir = workdir / "extra-info"
    if not extra_info_dir.is_dir():
        return
    shutil.rmtree(extra_info_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def runtime_support_paths() -> list[Path]:
    script_dir = Path(__file__).resolve().parent
    return [
        script_dir / "result_payload.py",
        script_dir / "bench_runtime.py",
        script_dir / "bench_contract.py",
        script_dir / "metadata_utils.py",
        script_dir / "perf_artifacts.py",
        script_dir / "profile_csv_parser.py",
        script_dir / "runtime_bootstrap.py",
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Shared benchmark runtime helper.",
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_cases = subparsers.add_parser("list-cases")
    _add_common_case_arguments(list_cases)

    run_one = subparsers.add_parser("run-one")
    _add_common_case_arguments(run_one)
    run_one.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of times to invoke the case (default 1).",
    )

    profile_one = subparsers.add_parser("profile-one")
    _add_common_case_arguments(profile_one)
    profile_one.add_argument(
        "--emit-record",
        action="store_true",
        help="Print the selected case record as compact JSON.",
    )
    profile_one.add_argument(
        "--verbose",
        action="store_true",
        help="Keep profiler helper diagnostics visible.",
    )

    run_all = subparsers.add_parser("run-all")
    _add_common_case_arguments(run_all, include_case_id=False)
    run_all.add_argument("--output")
    run_all.add_argument("--verbose", action="store_true")
    return parser


def _add_common_case_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_case_id: bool = True,
) -> None:
    parser.add_argument("--bench-file", required=True)
    parser.add_argument("--operator-file", required=True)
    if include_case_id:
        parser.add_argument("--case-id")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bench_file = _resolve_existing_path(args.bench_file, "Bench file")
    operator_file = _resolve_existing_path(args.operator_file, "Operator file")

    try:
        if args.command == "list-cases":
            cases, _resolution = load_bench_cases(bench_file, operator_file)
            for case in cases:
                print(case.case_id)
            return 0

        if args.command == "run-one":
            if args.iterations < 1:
                raise ValueError("--iterations must be >= 1")
            result = execute_bench_case(
                bench_file,
                operator_file,
                args.case_id,
                iterations=args.iterations,
            )
            return _emit_result(result)

        if args.command == "profile-one":
            if bool(args.emit_record):
                record = profile_bench_case(
                    bench_file,
                    operator_file,
                    args.case_id,
                    verbose=bool(args.verbose),
                )
                print(_render_case_record_json(record))
                return 1 if record.error_message is not None else 0
            result = profile_bench_case_quick(
                bench_file,
                operator_file,
                args.case_id,
            )
            return _emit_result(result)

        if args.command == "run-all":
            result, perf_path = profile_all_bench_cases(
                bench_file,
                operator_file,
                verbose=bool(args.verbose),
                output=args.output,
            )
            print(f"Perf file: {perf_path}")
            return _emit_result(result)
    except (FileNotFoundError, RuntimeError, ValueError, ImportError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    raise AssertionError(f"Unhandled bench runtime command: {args.command}")


def _resolve_existing_path(raw_path: str, label: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} path does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} path is not a file: {path}")
    return path


def _emit_result(result: ResultPayload) -> int:
    stdout_text = str(result["stdout"]).strip()
    stderr_text = str(result["stderr"]).strip()
    if stdout_text:
        print(stdout_text)
    if stderr_text:
        print(stderr_text, file=sys.stderr)
    return int(result["return_code"])


def _render_case_record_json(record: PerfCaseRecord) -> str:
    payload = {
        "case_label": record.case_label,
        "kernel_names": record.kernel_names,
        "kernel_source": record.kernel_source,
        "metrics": record.metrics,
        "error_message": record.error_message,
        "case_wall_clock_seconds": record.case_wall_clock_seconds,
    }
    return json.dumps(payload, separators=(",", ":"))


if __name__ == "__main__":
    raise SystemExit(main())
