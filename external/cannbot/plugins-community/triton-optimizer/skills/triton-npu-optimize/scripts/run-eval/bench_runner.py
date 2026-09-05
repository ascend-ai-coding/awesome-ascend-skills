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
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TextIO, TypeVar, cast

from bench_contract import parse_bench_metadata, resolve_bench_kernel_names
from npu_affinity import NpuDevicePool, affinity_env_for_device, parse_npu_devices
from debug_device import maybe_print_visible_devices
from perf_artifacts import (
    PerfCaseRecord,
    PerfMetrics,
    perf_output_path,
    render_perf_case_records_jsonl,
    write_perf_lines,
)
from result_payload import ResultPayload, make_result
from runtime_loader import load_script_module
from run_runtime import (
    RemoteSpec,
    cleanup_remote_workspace,
    copy_file_from_remote,
    copy_file_to_remote,
    eval_stall_timeout_seconds,
    create_remote_workspace,
    emit_verbose,
    local_python_executable,
    result_succeeded,
    run_buffered_process,
    run_remote_command_buffered,
    run_remote_command_streaming,
    run_streaming_process,
)


NpuDevices = tuple[str, ...]
ProbeCaps = tuple[int, int]
BenchRunResult = tuple[ResultPayload, Path | None]
BenchRunResultWithPerfPath = tuple[ResultPayload, Path]
RemoteBenchRunResult = tuple[ResultPayload, Path | None, str]
RemoteBenchRunResultWithPerfPath = tuple[ResultPayload, Path, str]
ResolvedProfileOutputRoot = tuple[str | None, str]
CaseWorkspaceRoots = tuple[Path, Path]
CaseWorkspace = tuple[Path, Callable[[], None]]


def _unpack_call(
    args: tuple[object, ...],
    kwargs: dict[str, object],
    names: tuple[str, ...],
    defaults: tuple[object, ...] = (),
) -> list[object]:
    values: list[object] = []
    for index, name in enumerate(names):
        if index < len(args):
            values.append(args[index])
        elif name in kwargs:
            values.append(kwargs[name])
        elif defaults and index >= len(names) - len(defaults):
            values.append(defaults[index - (len(names) - len(defaults))])
        else:
            raise TypeError(f"missing required argument: {name}")
    return values

_LOCAL_BENCH_OUTPUT_DIR_ENV = "TRITON_AGENT_BENCH_OUTPUT_DIR"
_BENCH_COPY_FILES_ENV = "TRITON_AGENT_BENCH_COPY_FILES"
_bench_runtime_module_cache = None
_bench_runtime_module_lock = threading.Lock()
_T = TypeVar("_T")
_PRESERVED_RUN_DIR_NONE_SENTINEL = "__NONE__"


@dataclass(frozen=True)
class _BenchRequest:
    bench_file: Path
    operator_file: Path
    bench_mode: str
    devices: NpuDevices | None
    invocation_root: Path
    verbose: bool
    output: str | None


@dataclass(frozen=True)
class _RemoteBenchRequest:
    request: _BenchRequest
    remote: str
    remote_workdir: str | None
    keep_remote_workdir: bool
    stderr: TextIO | None
    probe_caps: ProbeCaps | None


@dataclass(frozen=True)
class _RemoteScriptRun:
    spec: RemoteSpec
    remote_workspace: str
    script: str
    bench_file: Path
    operator_file: Path
    perf_path: Path
    verbose: bool
    stderr: TextIO | None
    extra_env: dict[str, str] | None


@dataclass(frozen=True)
class _RemoteParallelCaseContext:
    spec: RemoteSpec
    remote_workspace: str
    bench_file: Path
    operator_file: Path
    pool: NpuDevicePool
    source_root: Path
    json_search_root: Path
    verbose: bool
    stderr: TextIO | None


@dataclass(frozen=True)
class _LocalParallelRequest:
    bench_file: Path
    operator_file: Path
    devices: NpuDevices
    source_root: Path
    json_search_root: Path
    verbose: bool
    output: str | None


@dataclass(frozen=True)
class _RemoteParallelRequest:
    context: _RemoteParallelCaseContext
    output: str | None
    max_workers: int


@dataclass(frozen=True)
class _LocalCaseRequest:
    workspace_root: Path
    bench_file: Path
    operator_file: Path
    case_id: str
    device: str
    source_root: Path
    verbose: bool
    command_script: str
    parse_result: Callable[[ResultPayload], PerfCaseRecord]
    preserved_run_dir: Path | None


@dataclass(frozen=True)
class _RemoteCaseRequest:
    spec: RemoteSpec
    case_workspace: str
    bench_file: Path
    operator_file: Path
    case_id: str
    device: str
    source_root: Path
    verbose: bool
    stderr: TextIO | None


def _collect_env_copy_files(search_dir: Path) -> list[Path]:
    patterns_str = os.environ.get(_BENCH_COPY_FILES_ENV, "")
    if not patterns_str.strip():
        return []
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in _copy_file_patterns(patterns_str):
        _collect_pattern_matches(search_dir, pattern, seen, paths)
    return paths


def _copy_file_patterns(raw_patterns: str) -> list[str]:
    return [pattern.strip() for pattern in raw_patterns.split(",") if pattern.strip()]


def _collect_pattern_matches(
    search_dir: Path,
    pattern: str,
    seen: set[Path],
    paths: list[Path],
) -> None:
    for matched in sorted(search_dir.glob(pattern)):
        if not matched.is_file():
            continue
        resolved = matched.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        paths.append(matched)


def normalize_bench_mode(bench_mode: str) -> str:
    return "torch-npu-profiler" if bench_mode == "standalone" else bench_mode


def run_local_bench(*args: object, **kwargs: object) -> BenchRunResult:
    bench_file = cast(Path, args[0] if args else kwargs["bench_file"])
    operator_file = cast(Path, args[1] if len(args) > 1 else kwargs["operator_file"])
    bench_mode = str(args[2] if len(args) > 2 else kwargs["bench_mode"])
    npu_devices = cast(str | None, kwargs.get("npu_devices", args[3] if len(args) > 3 else None))
    verbose = bool(kwargs.get("verbose", False))
    output = cast(str | None, kwargs.get("output"))
    request = _BenchRequest(
        bench_file=bench_file,
        operator_file=operator_file,
        bench_mode=normalize_bench_mode(bench_mode),
        devices=parse_npu_devices(npu_devices),
        invocation_root=Path.cwd().resolve(),
        verbose=verbose,
        output=output,
    )
    maybe_print_visible_devices()
    with _local_bench_workdir(bench_file.parent):
        return _run_local_bench_request(request)


def _run_local_bench_request(request: _BenchRequest) -> BenchRunResult:
    if request.devices is None:
        return _run_local_bench_single_device(request)
    source_root, json_search_root = _resolve_case_workspace_roots(
        request.bench_file, request.operator_file, invocation_root=request.invocation_root
    )
    return _run_local_bench_parallel(request, source_root, json_search_root)


def _run_local_bench_single_device(request: _BenchRequest) -> BenchRunResult:
    if request.bench_mode == "perf-counter":
        return _run_local_bench_perf_counter(
            request.bench_file, request.operator_file, verbose=request.verbose, output=request.output
        )
    return _run_local_bench_torch_npu_profiler(
        request.bench_file, request.operator_file, verbose=request.verbose, output=request.output
    )


def _run_local_bench_parallel(
    request: _BenchRequest,
    source_root: Path,
    json_search_root: Path,
) -> BenchRunResult:
    if request.devices is None:
        raise ValueError("parallel benchmark requires NPU devices")
    common = {
        "source_root": source_root,
        "json_search_root": json_search_root,
        "verbose": request.verbose,
        "output": request.output,
    }
    if request.bench_mode == "perf-counter":
        return _run_local_bench_perf_counter_parallel(
            request.bench_file, request.operator_file, request.devices, **common
        )
    return _run_local_bench_torch_npu_profiler_parallel(
        request.bench_file, request.operator_file, request.devices, **common
    )


def run_remote_bench(*args: object, **kwargs: object) -> RemoteBenchRunResult:
    bench_file = cast(Path, args[0] if args else kwargs["bench_file"])
    operator_file = cast(Path, args[1] if len(args) > 1 else kwargs["operator_file"])
    bench_mode = str(args[2] if len(args) > 2 else kwargs["bench_mode"])
    remote = str(args[3] if len(args) > 3 else kwargs["remote"])
    remote_workdir = cast(str | None, args[4] if len(args) > 4 else kwargs.get("remote_workdir"))
    npu_devices = cast(str | None, kwargs.get("npu_devices"))
    keep_remote_workdir = bool(kwargs.get("keep_remote_workdir", False))
    verbose = bool(kwargs.get("verbose", False))
    stderr = cast(TextIO | None, kwargs.get("stderr"))
    output = cast(str | None, kwargs.get("output"))
    probe_caps = cast(ProbeCaps | None, kwargs.get("probe_caps"))
    request = _RemoteBenchRequest(
        request=_BenchRequest(
            bench_file=bench_file,
            operator_file=operator_file,
            bench_mode=normalize_bench_mode(bench_mode),
            devices=parse_npu_devices(npu_devices),
            invocation_root=Path.cwd().resolve(),
            verbose=verbose,
            output=output,
        ),
        remote=remote,
        remote_workdir=remote_workdir,
        keep_remote_workdir=keep_remote_workdir,
        stderr=stderr,
        probe_caps=probe_caps,
    )
    maybe_print_visible_devices()
    spec, remote_workspace = create_remote_workspace(
        request.remote, request.remote_workdir, verbose=request.request.verbose, stderr=request.stderr
    )
    try:
        _stage_remote_bench_inputs(spec, remote_workspace, request)
        return _run_remote_bench_request(spec, remote_workspace, request)
    finally:
        if not request.keep_remote_workdir:
            cleanup_remote_workspace(spec, remote_workspace, verbose=request.request.verbose, stderr=request.stderr)


def _stage_remote_bench_inputs(
    spec: RemoteSpec,
    remote_workspace: str,
    request: _RemoteBenchRequest,
) -> None:
    paths = [request.request.bench_file, request.request.operator_file]
    bench_cases = request.request.bench_file.with_suffix(".json")
    if bench_cases.exists():
        paths.append(bench_cases)
    for path in paths:
        copy_file_to_remote(
            spec, path, f"{remote_workspace}/{path.name}",
            verbose=request.request.verbose, stderr=request.stderr,
        )


def _run_remote_bench_request(
    spec: RemoteSpec,
    remote_workspace: str,
    remote_request: _RemoteBenchRequest,
) -> RemoteBenchRunResult:
    request = remote_request.request
    if request.devices is None:
        return _run_remote_bench_single_device(spec, remote_workspace, remote_request)
    if request.bench_mode == "torch-npu-profiler" and remote_request.probe_caps is not None:
        return _run_remote_bench_torch_npu_profiler(
            spec, remote_workspace, request.bench_file, request.operator_file,
            verbose=request.verbose, stderr=remote_request.stderr, output=request.output,
            probe_caps=remote_request.probe_caps, devices=request.devices,
        )
    source_root, json_search_root = _resolve_case_workspace_roots(
        request.bench_file, request.operator_file, invocation_root=request.invocation_root
    )
    return _run_remote_bench_parallel(spec, remote_workspace, remote_request, source_root, json_search_root)


def _run_remote_bench_single_device(
    spec: RemoteSpec,
    remote_workspace: str,
    remote_request: _RemoteBenchRequest,
) -> RemoteBenchRunResult:
    request = remote_request.request
    common = {
        "verbose": request.verbose,
        "stderr": remote_request.stderr,
        "output": request.output,
    }
    if request.bench_mode == "perf-counter":
        return _run_remote_bench_perf_counter(
            spec, remote_workspace, request.bench_file, request.operator_file, **common
        )
    return _run_remote_bench_torch_npu_profiler(
        spec, remote_workspace, request.bench_file, request.operator_file,
        probe_caps=remote_request.probe_caps, devices=None, **common,
    )


def _run_remote_bench_parallel(
    spec: RemoteSpec,
    remote_workspace: str,
    remote_request: _RemoteBenchRequest,
    source_root: Path,
    json_search_root: Path,
) -> RemoteBenchRunResult:
    request = remote_request.request
    if request.devices is None:
        raise ValueError("parallel benchmark requires NPU devices")
    common = {
        "source_root": source_root, "json_search_root": json_search_root,
        "verbose": request.verbose, "stderr": remote_request.stderr, "output": request.output,
    }
    if request.bench_mode == "perf-counter":
        return _run_remote_bench_perf_counter_parallel(
            spec, remote_workspace, request.bench_file, request.operator_file, request.devices, **common
        )
    return _run_remote_bench_torch_npu_profiler_parallel(
        spec, remote_workspace, request.bench_file, request.operator_file, request.devices, **common
    )


def run_local_probe(*args: object, **kwargs: object) -> BenchRunResult:
    bench_file, operator_file, bench_mode, warmup_cap, repeats_cap, npu_devices, verbose, output = _unpack_call(
        args,
        kwargs,
        (
            "bench_file", "operator_file", "bench_mode", "warmup_cap",
            "repeats_cap", "npu_devices", "verbose", "output",
        ),
        defaults=(None, None, None),
    )
    bench_file, operator_file, bench_mode = cast(Path, bench_file), cast(Path, operator_file), str(bench_mode)
    warmup_cap, repeats_cap = int(warmup_cap), int(repeats_cap)
    npu_devices, verbose, output = cast(str | None, npu_devices), bool(verbose), cast(str | None, output)
    bench_mode = normalize_bench_mode(bench_mode)
    if bench_mode != "torch-npu-profiler":
        return run_local_bench(
            bench_file,
            operator_file,
            bench_mode,
            npu_devices=npu_devices,
            verbose=verbose,
            output=output,
        )
    runtime = _load_bench_runtime_module()
    cases, resolution = runtime.load_bench_cases(bench_file, operator_file)
    clamped = [
        replace(
            case,
            warmup=min(case.warmup, warmup_cap),
            repeats=min(case.repeats, repeats_cap),
        )
        for case in cases
    ]
    return runtime.profile_all_bench_cases(
        bench_file,
        operator_file,
        preloaded=(clamped, resolution),
        verbose=verbose,
        output=output,
    )


def run_remote_probe(*args: object, **kwargs: object) -> RemoteBenchRunResult:
    values = _unpack_call(
        args,
        kwargs,
        (
            "bench_file", "operator_file", "bench_mode", "remote",
            "remote_workdir", "warmup_cap", "repeats_cap", "npu_devices",
            "keep_remote_workdir", "verbose", "stderr", "output",
        ),
        (None, None, None, None, None, None, None),
    )
    (
        bench_file, operator_file, bench_mode, remote, remote_workdir,
        warmup_cap, repeats_cap, npu_devices, keep_remote_workdir, verbose,
        stderr, output,
    ) = values
    bench_file, operator_file = cast(Path, bench_file), cast(Path, operator_file)
    bench_mode, remote = str(bench_mode), str(remote)
    remote_workdir = cast(str | None, remote_workdir)
    return run_remote_bench(
        bench_file,
        operator_file,
        bench_mode,
        remote,
        remote_workdir,
        npu_devices=npu_devices,
        keep_remote_workdir=keep_remote_workdir,
        verbose=verbose,
        stderr=stderr,
        output=output,
        probe_caps=(warmup_cap, repeats_cap),
    )


def _run_local_bench_torch_npu_profiler(
    bench_file: Path,
    operator_file: Path,
    *,
    verbose: bool = False,
    output: str | None = None,
) -> BenchRunResult:
    runtime = _load_bench_runtime_module()
    return runtime.profile_all_bench_cases(
        bench_file,
        operator_file,
        verbose=verbose,
        output=output,
    )


def _run_local_bench_perf_counter(
    bench_file: Path,
    operator_file: Path,
    *,
    verbose: bool = False,
    output: str | None = None,
) -> BenchRunResult:
    runtime = _load_bench_runtime_module()
    return runtime.time_all_bench_cases(
        bench_file,
        operator_file,
        bench_mode="perf-counter",
        output=output,
    )


def _run_local_bench_perf_counter_parallel(*args: object, **kwargs: object) -> BenchRunResult:
    request = _local_parallel_request_from_call(args, kwargs)
    case_records = _run_local_parallel_cases(request, _run_local_perf_counter_case_in_subprocess)
    perf_path = _write_perf_counter_perf(request.operator_file, case_records, output=request.output)
    return _build_perf_counter_result(case_records), perf_path


def _local_parallel_request_from_call(
    args: tuple[object, ...], kwargs: dict[str, object],
) -> _LocalParallelRequest:
    values = _unpack_call(
        args,
        kwargs,
        ("bench_file", "operator_file", "devices", "source_root", "json_search_root", "verbose", "output"),
        (None, None),
    )
    return _LocalParallelRequest(
        cast(Path, values[0]),
        cast(Path, values[1]),
        cast(NpuDevices, values[2]),
        cast(Path, values[3]),
        cast(Path, values[4]),
        bool(values[5]),
        cast(str | None, values[6]),
    )


def _run_local_parallel_cases(
    request: _LocalParallelRequest,
    run_case: Callable[..., PerfCaseRecord],
    preserved_run_dir: Path | None = None,
) -> list[PerfCaseRecord]:
    runtime = _load_bench_runtime_module()
    cases, _resolution = runtime.load_bench_cases(request.bench_file, request.operator_file)
    case_ids = [case.case_id for case in cases]
    pool = NpuDevicePool(request.devices)

    def _worker(case_id: str) -> PerfCaseRecord:
        return _run_local_parallel_case_worker(
            request.bench_file,
            request.operator_file,
            case_id,
            pool=pool,
            source_root=request.source_root,
            json_search_root=request.json_search_root,
            verbose=request.verbose,
            run_case=run_case,
            preserved_run_dir=preserved_run_dir,
        )

    case_records = _run_parallel_case_workers(case_ids, min(len(case_ids), len(request.devices)), _worker)
    _sort_case_records(case_records, case_ids)
    return case_records


def _write_perf_counter_perf(
    operator_file: Path,
    case_records: list[PerfCaseRecord],
    *,
    output: str | None = None,
) -> Path:
    perf_path = _resolve_perf_output_path(operator_file, output=output)
    write_perf_lines(perf_path, render_perf_case_records_jsonl(case_records))
    return perf_path


def _build_perf_counter_result(case_records: list[PerfCaseRecord]) -> ResultPayload:
    had_failures = any(record.error_message is not None for record in case_records)
    stderr = "\n".join(
        str(record.error_message) for record in case_records if record.error_message is not None
    )
    return make_result(return_code=1 if had_failures else 0, stdout="", stderr=stderr)


def _run_local_parallel_case_worker(*args: object, **kwargs: object) -> PerfCaseRecord:
    bench_file, operator_file, case_id = cast(Path, args[0]), cast(Path, args[1]), str(args[2])
    pool = cast(NpuDevicePool, kwargs["pool"])
    source_root = cast(Path, kwargs["source_root"])
    json_search_root = cast(Path, kwargs["json_search_root"])
    verbose = bool(kwargs["verbose"])
    run_case = cast(Callable[..., PerfCaseRecord], kwargs["run_case"])
    preserved_run_dir = cast(Path | None, kwargs.get("preserved_run_dir"))
    case_workspace, cleanup = _create_local_torch_npu_profiler_case_workspace(
        bench_file,
        operator_file,
        case_id,
        source_root=source_root,
        json_search_root=json_search_root,
        verbose=verbose,
    )
    try:
        with pool.acquire() as device:
            return run_case(
                case_workspace,
                bench_file,
                operator_file,
                case_id,
                device,
                preserved_run_dir=preserved_run_dir,
                source_root=source_root,
                verbose=verbose,
            )
    finally:
        cleanup()


def _run_remote_bench_perf_counter(*args: object, **kwargs: object) -> RemoteBenchRunResult:
    spec, remote_workspace, bench_file, operator_file, verbose, stderr, output = _unpack_call(
        args, kwargs, ("spec", "remote_workspace", "bench_file", "operator_file", "verbose", "stderr", "output"),
        (False, None, None),
    )
    spec, remote_workspace = cast(RemoteSpec, spec), str(remote_workspace)
    bench_file, operator_file = cast(Path, bench_file), cast(Path, operator_file)
    verbose, stderr, output = bool(verbose), cast(TextIO | None, stderr), cast(str | None, output)
    _stage_remote_bench_runtime_support_files(
        spec,
        remote_workspace,
        verbose=verbose,
        stderr=stderr,
    )
    perf_path = _resolve_perf_output_path(operator_file, output=output)
    return _run_remote_script_and_copy_perf(
        _RemoteScriptRun(
            spec, remote_workspace, _build_remote_perf_counter_run_all_script(verbose=verbose),
            bench_file, operator_file, perf_path, verbose, stderr, {"TRITON_ALWAYS_COMPILE": "1"},
        )
    )


def _run_remote_bench_perf_counter_parallel(*args: object, **kwargs: object) -> RemoteBenchRunResult:
    request = _remote_parallel_request_from_call(args, kwargs, (False, None, None))
    case_records = _run_remote_parallel_cases(request, _run_remote_perf_counter_parallel_case)
    perf_path = _write_perf_counter_perf(request.context.operator_file, case_records, output=request.output)
    return _build_perf_counter_result(case_records), perf_path, request.context.remote_workspace


def _remote_parallel_request_from_call(
    args: tuple[object, ...],
    kwargs: dict[str, object],
    defaults: tuple[object, ...] = (),
) -> _RemoteParallelRequest:
    values = _unpack_call(
        args,
        kwargs,
        (
            "spec", "remote_workspace", "bench_file", "operator_file", "devices",
            "source_root", "json_search_root", "verbose", "stderr", "output",
        ),
        defaults,
    )
    context = _RemoteParallelCaseContext(
        cast(RemoteSpec, values[0]),
        str(values[1]),
        cast(Path, values[2]),
        cast(Path, values[3]),
        NpuDevicePool(cast(NpuDevices, values[4])),
        cast(Path, values[5]),
        cast(Path, values[6]),
        bool(values[7]),
        cast(TextIO | None, values[8]),
    )
    return _RemoteParallelRequest(
        context,
        cast(str | None, values[9]),
        len(cast(NpuDevices, values[4])),
    )


def _run_remote_parallel_cases(
    request: _RemoteParallelRequest,
    run_case: Callable[[_RemoteParallelCaseContext, str], PerfCaseRecord],
) -> list[PerfCaseRecord]:
    context = request.context
    runtime = _load_bench_runtime_module()
    cases, _resolution = runtime.load_bench_cases(context.bench_file, context.operator_file)
    case_ids = [case.case_id for case in cases]
    case_records = _run_parallel_case_workers(
        case_ids,
        min(len(case_ids), request.max_workers),
        lambda case_id: run_case(context, case_id),
    )
    _sort_case_records(case_records, case_ids)
    return case_records


def _stage_remote_parallel_case(
    context: _RemoteParallelCaseContext,
    case_id: str,
) -> tuple[str, str]:
    case_workspace = f"{context.remote_workspace}/case-{case_id}"
    run_remote_command_buffered(
        context.spec, context.remote_workspace, ["mkdir", "-p", case_workspace],
        verbose=context.verbose, stderr=context.stderr,
    )
    workspace_root = _stage_remote_torch_npu_profiler_case_workspace(
        context.spec, context.bench_file, context.operator_file, case_workspace,
        source_root=context.source_root, json_search_root=context.json_search_root,
        verbose=context.verbose, stderr=context.stderr,
    )
    return case_workspace, workspace_root


def _run_remote_perf_counter_parallel_case(
    context: _RemoteParallelCaseContext,
    case_id: str,
) -> PerfCaseRecord:
    _case_workspace, workspace_root = _stage_remote_parallel_case(context, case_id)
    with context.pool.acquire() as device:
        return _run_remote_perf_counter_case(
            context.spec, workspace_root, context.bench_file, context.operator_file,
            case_id, device, source_root=context.source_root,
            verbose=context.verbose, stderr=context.stderr,
        )


@contextlib.contextmanager
def _local_bench_workdir(workdir: Path):
    original_cwd = Path.cwd()
    os.chdir(workdir)
    try:
        yield
    finally:
        os.chdir(original_cwd)


def _cleanup_local_bench_extra_info(workdir: Path) -> None:
    extra_info_dir = workdir / "extra-info"
    if not extra_info_dir.is_dir():
        return
    shutil.rmtree(extra_info_dir)


@contextlib.contextmanager
def stream_target_for_verbosity(verbose: bool) -> Iterator[TextIO]:
    if verbose:
        yield sys.stdout
        return
    with open(os.devnull, "w", encoding="utf-8") as quiet_stdout:
        yield quiet_stdout


def _stage_remote_bench_runtime_support_files(
    spec: RemoteSpec,
    remote_workspace: str,
    *,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> None:
    for support_path in _bench_runtime_support_paths():
        copy_file_to_remote(
            spec,
            support_path,
            f"{remote_workspace}/{support_path.name}",
            verbose=verbose,
            stderr=stderr,
        )


def _run_remote_bench_torch_npu_profiler(*args: object, **kwargs: object) -> RemoteBenchRunResult:
    (
        spec, remote_workspace, bench_file, operator_file, verbose, stderr,
        output, probe_caps, devices,
    ) = _unpack_call(
        args,
        kwargs,
        (
            "spec", "remote_workspace", "bench_file", "operator_file", "verbose",
            "stderr", "output", "probe_caps", "devices",
        ),
        (False, None, None, None, None),
    )
    spec, remote_workspace = cast(RemoteSpec, spec), str(remote_workspace)
    bench_file, operator_file = cast(Path, bench_file), cast(Path, operator_file)
    verbose, stderr, output = bool(verbose), cast(TextIO | None, stderr), cast(str | None, output)
    probe_caps, devices = cast(ProbeCaps | None, probe_caps), cast(NpuDevices | None, devices)
    _stage_remote_bench_runtime_support_files(
        spec,
        remote_workspace,
        verbose=verbose,
        stderr=stderr,
    )
    perf_path = _resolve_perf_output_path(operator_file, output=output)
    if probe_caps is not None:
        script = _build_remote_torch_npu_profiler_probe_run_all_script(
            verbose=verbose, warmup_cap=probe_caps[0], repeats_cap=probe_caps[1]
        )
    else:
        script = _build_remote_torch_npu_profiler_run_all_script(verbose=verbose)
    extra_env: dict[str, str] = {"TRITON_ALWAYS_COMPILE": "1"}
    if devices is not None:
        extra_env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(devices)
    return _run_remote_script_and_copy_perf(
        _RemoteScriptRun(
            spec, remote_workspace, script, bench_file, operator_file, perf_path,
            verbose, stderr, extra_env,
        )
    )


def _run_remote_script_and_copy_perf(request: _RemoteScriptRun) -> RemoteBenchRunResult:
    with stream_target_for_verbosity(request.verbose) as stream_target:
        result = run_remote_command_streaming(
            request.spec,
            request.remote_workspace,
            ["python3", "-c", request.script, request.bench_file.name,
             request.operator_file.name, request.perf_path.name],
            stdout=stream_target,
            verbose=request.verbose,
            stderr=request.stderr,
            stall_timeout_seconds=eval_stall_timeout_seconds(),
            extra_env=request.extra_env,
        )
    try:
        copy_file_from_remote(
            request.spec,
            f"{request.remote_workspace}/{request.perf_path.name}",
            request.perf_path,
            verbose=request.verbose,
            stderr=request.stderr,
        )
    except RuntimeError:
        if result_succeeded(result):
            raise
        return result, None, request.remote_workspace
    return result, request.perf_path, request.remote_workspace


def _run_local_bench_torch_npu_profiler_parallel(*args: object, **kwargs: object) -> BenchRunResultWithPerfPath:
    request = _local_parallel_request_from_call(args, kwargs)
    runtime = _load_bench_runtime_module()
    preserved_run_dir: Path | None = None
    create_preserved_run_dir = getattr(runtime, "create_local_preserved_profile_run_dir", None)
    if callable(create_preserved_run_dir):
        preserved_run_dir = cast(
            Path | None,
            create_preserved_run_dir(prefix="triton-agent-torch-npu-profiler-bench-"),
        )
    case_records = _run_local_parallel_cases(
        request,
        _run_local_torch_npu_profiler_case_in_subprocess,
        preserved_run_dir,
    )
    perf_path = _write_torch_npu_profiler_perf(request.operator_file, case_records, output=request.output)
    return _build_torch_npu_profiler_result(case_records), perf_path


def _run_remote_bench_torch_npu_profiler_parallel(*args: object, **kwargs: object) -> RemoteBenchRunResultWithPerfPath:
    request = _remote_parallel_request_from_call(args, kwargs)
    case_records = _run_remote_parallel_cases(request, _run_remote_torch_npu_profiler_parallel_case)
    perf_path = _write_torch_npu_profiler_perf(request.context.operator_file, case_records, output=request.output)
    return _build_torch_npu_profiler_result(case_records), perf_path, request.context.remote_workspace


def _run_remote_torch_npu_profiler_parallel_case(
    context: _RemoteParallelCaseContext,
    case_id: str,
) -> PerfCaseRecord:
    case_workspace, workspace_root = _stage_remote_parallel_case(context, case_id)
    try:
        with context.pool.acquire() as device:
            return _run_remote_torch_npu_profiler_case(
                context.spec, workspace_root, context.bench_file, context.operator_file,
                case_id, device, source_root=context.source_root,
                verbose=context.verbose, stderr=context.stderr,
            )
    finally:
        run_remote_command_buffered(
            context.spec, context.remote_workspace, ["rm", "-rf", case_workspace],
            verbose=context.verbose, stderr=context.stderr,
        )


def _bench_runtime_script_path() -> Path:
    return Path(__file__).resolve().with_name("bench_runtime.py")


def _bench_runtime_support_paths() -> list[Path]:
    runtime = _load_bench_runtime_module()
    return cast(list[Path], runtime.runtime_support_paths())


def _bench_flat_input_paths(bench_file: Path) -> list[Path]:
    return _bench_runtime_support_paths() + _collect_env_copy_files(bench_file.parent)


def _load_bench_runtime_module():
    global _bench_runtime_module_cache
    cached_module = _bench_runtime_module_cache
    if cached_module is not None:
        return cached_module

    with _bench_runtime_module_lock:
        cached_module = _bench_runtime_module_cache
        if cached_module is not None:
            return cached_module

        script_path = _bench_runtime_script_path()
        module = load_script_module(
            script_path,
            f"triton_agent_bench_runtime_{script_path.stem}",
            remove_after_load=False,
        )
        _bench_runtime_module_cache = module
        return module


def _run_parallel_case_workers(
    case_keys: Sequence[str],
    max_workers: int,
    worker: Callable[[str], _T],
) -> list[_T]:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, case_key) for case_key in case_keys]
        return [future.result() for future in futures]


def _sort_case_records(case_records: list[PerfCaseRecord], ordered_case_labels: Sequence[str]) -> None:
    case_order = {label: index for index, label in enumerate(ordered_case_labels)}
    case_records.sort(key=lambda record: case_order[record.case_label])


def _resolve_local_bench_profile_output_root() -> ResolvedProfileOutputRoot:
    configured_root = os.environ.get(_LOCAL_BENCH_OUTPUT_DIR_ENV)
    if configured_root:
        return str(Path(configured_root).expanduser().resolve()), _LOCAL_BENCH_OUTPUT_DIR_ENV
    return None, _LOCAL_BENCH_OUTPUT_DIR_ENV


def _bench_case_input_paths(
    bench_file: Path,
    operator_file: Path,
    *,
    json_search_root: Path | None = None,
) -> list[Path]:
    input_paths: list[Path] = [bench_file]
    json_roots = [bench_file.parent.resolve(), operator_file.parent.resolve()]
    if json_search_root is not None:
        resolved_json_root = json_search_root.resolve()
        if resolved_json_root not in json_roots:
            json_roots.insert(0, resolved_json_root)
    for json_root in json_roots:
        input_paths.extend(
            sorted(path for path in json_root.glob("*.json") if path.is_file())
        )
    input_paths.append(operator_file)
    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for input_path in input_paths:
        resolved_path = input_path.resolve()
        if resolved_path in seen:
            continue
        seen.add(resolved_path)
        unique_paths.append(input_path)
    return unique_paths


def _path_is_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _resolve_case_workspace_roots(
    bench_file: Path,
    operator_file: Path,
    *,
    invocation_root: Path | None,
) -> CaseWorkspaceRoots:
    if invocation_root is not None:
        resolved_invocation_root = invocation_root.resolve()
        workspace_dirs = [bench_file.parent.resolve(), operator_file.parent.resolve()]
        if all(_path_is_within_root(path, resolved_invocation_root) for path in workspace_dirs):
            return resolved_invocation_root, resolved_invocation_root
    source_root = Path(
        os.path.commonpath(
            [
                str(bench_file.parent.resolve()),
                str(operator_file.parent.resolve()),
            ]
        )
    )
    return source_root, bench_file.parent.resolve()


def _case_workspace_root_name(source_root: Path) -> str:
    return source_root.name or "workspace"


def _case_workspace_root_relative_path(path: Path, *, source_root: Path) -> Path:
    try:
        return path.resolve().relative_to(source_root.resolve())
    except ValueError:
        return Path(path.name)


def _case_workspace_command_path(path: Path, *, source_root: Path) -> str:
    return _case_workspace_root_relative_path(path, source_root=source_root).as_posix()


def _emit_case_workspace_verbose(message: str, *, stderr: TextIO | None = None) -> None:
    emit_verbose(stderr or sys.stderr, "files", message)


def _create_local_case_workspace(
    *,
    prefix: str,
    input_paths: Sequence[Path],
    flat_input_paths: Sequence[Path] = (),
    source_root: Path,
    verbose: bool = False,
) -> CaseWorkspace:
    temp_dir = tempfile.TemporaryDirectory(prefix=prefix)
    workspace = Path(temp_dir.name)
    workspace_root = workspace / _case_workspace_root_name(source_root)
    workspace_root.mkdir(parents=True, exist_ok=True)
    if verbose:
        _emit_case_workspace_verbose(f"created local case workspace: {workspace_root}")
    for input_path in input_paths:
        relative_path = _case_workspace_root_relative_path(input_path, source_root=source_root)
        target_path = workspace_root / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(input_path, target_path)
        if verbose:
            _emit_case_workspace_verbose(f"copied local case file: {input_path} -> {target_path}")
    for input_path in flat_input_paths:
        target_path = workspace_root / input_path.name
        shutil.copyfile(input_path, target_path)
        if verbose:
            _emit_case_workspace_verbose(f"copied local case support file: {input_path} -> {target_path}")
    return workspace_root, temp_dir.cleanup


def _stage_remote_case_workspace(*args: object, **kwargs: object) -> str:
    spec, case_workspace, input_paths, source_root, flat_input_paths, verbose, stderr = _unpack_call(
        args,
        kwargs,
        (
            "spec", "case_workspace", "input_paths", "source_root",
            "flat_input_paths", "verbose", "stderr",
        ),
        ((), False, None),
    )
    spec, case_workspace = cast(RemoteSpec, spec), str(case_workspace)
    input_paths, source_root = cast(Sequence[Path], input_paths), cast(Path, source_root)
    flat_input_paths = cast(Sequence[Path], flat_input_paths)
    verbose = bool(verbose)
    stderr = cast(TextIO | None, stderr)
    workspace_root = f"{case_workspace}/{_case_workspace_root_name(source_root)}"
    _create_remote_case_workspace_root(
        spec, case_workspace, workspace_root, verbose, stderr
    )
    _copy_remote_case_tree(
        spec, case_workspace, workspace_root, input_paths, source_root, verbose, stderr
    )
    _copy_remote_case_flat_files(
        spec, workspace_root, flat_input_paths, verbose, stderr
    )
    return workspace_root


def _create_remote_case_workspace_root(
    spec: RemoteSpec,
    case_workspace: str,
    workspace_root: str,
    verbose: bool,
    stderr: TextIO | None,
) -> None:
    run_remote_command_buffered(
        spec, case_workspace, ["mkdir", "-p", workspace_root], verbose=verbose, stderr=stderr
    )
    if verbose:
        _emit_case_workspace_verbose(f"created remote case workspace: {workspace_root}", stderr=stderr)


def _copy_remote_case_tree(*args: object, **kwargs: object) -> None:
    spec, case_workspace, workspace_root, input_paths, source_root, verbose, stderr = _unpack_call(
        args, kwargs, ("spec", "case_workspace", "workspace_root", "input_paths", "source_root", "verbose", "stderr")
    )
    spec, case_workspace, workspace_root = cast(RemoteSpec, spec), str(case_workspace), str(workspace_root)
    input_paths, source_root = cast(Sequence[Path], input_paths), cast(Path, source_root)
    verbose, stderr = bool(verbose), cast(TextIO | None, stderr)
    created_dirs = {workspace_root}
    for input_path in input_paths:
        relative_path = _case_workspace_root_relative_path(input_path, source_root=source_root)
        target_dir = (
            workspace_root
            if relative_path.parent == Path(".")
            else f"{workspace_root}/{relative_path.parent.as_posix()}"
        )
        if target_dir not in created_dirs:
            run_remote_command_buffered(
                spec,
                case_workspace,
                ["mkdir", "-p", target_dir],
                verbose=verbose,
                stderr=stderr,
            )
            created_dirs.add(target_dir)
        copy_file_to_remote(
            spec,
            input_path,
            f"{workspace_root}/{relative_path.as_posix()}",
            verbose=verbose,
            stderr=stderr,
        )
        if verbose:
            _emit_case_workspace_verbose(
                f"copied remote case file: {input_path} -> {workspace_root}/{relative_path.as_posix()}",
                stderr=stderr,
            )


def _copy_remote_case_flat_files(
    spec: RemoteSpec,
    workspace_root: str,
    input_paths: Sequence[Path],
    verbose: bool,
    stderr: TextIO | None,
) -> None:
    for input_path in input_paths:
        target_path = f"{workspace_root}/{input_path.name}"
        copy_file_to_remote(
            spec,
            input_path,
            target_path,
            verbose=verbose,
            stderr=stderr,
        )
        if verbose:
            _emit_case_workspace_verbose(
                f"copied remote case support file: {input_path} -> {target_path}",
                stderr=stderr,
            )


def _create_local_torch_npu_profiler_case_workspace(*args: object, **kwargs: object) -> CaseWorkspace:
    bench_file, operator_file, case_id, source_root, json_search_root, verbose = _unpack_call(
        args, kwargs, ("bench_file", "operator_file", "case_id", "source_root", "json_search_root", "verbose"), (False,)
    )
    bench_file, operator_file, case_id = cast(Path, bench_file), cast(Path, operator_file), str(case_id)
    source_root, json_search_root, verbose = cast(Path, source_root), cast(Path, json_search_root), bool(verbose)
    return _create_local_case_workspace(
        prefix=f"triton-agent-torch-npu-profiler-case-{case_id}-",
        input_paths=_bench_case_input_paths(
            bench_file,
            operator_file,
            json_search_root=json_search_root,
        ),
        flat_input_paths=_bench_flat_input_paths(bench_file),
        source_root=source_root,
        verbose=verbose,
    )


def _build_remote_torch_npu_profiler_run_all_script(*, verbose: bool = False) -> str:
    return (
        "import pathlib, shutil, sys; "
        "import bench_runtime as runtime; "
        "bench_file = pathlib.Path(sys.argv[1]); "
        "operator_file = pathlib.Path(sys.argv[2]); "
        "target_path = pathlib.Path(sys.argv[3]); "
        f"result, perf_path = runtime.profile_all_bench_cases(bench_file, operator_file, verbose={verbose}); "
        "target_path.parent.mkdir(parents=True, exist_ok=True); "
        "shutil.copyfile(perf_path, target_path) if perf_path != target_path else None; "
        "raise SystemExit(int(result['return_code']))"
    )


def _build_remote_torch_npu_profiler_probe_run_all_script(
    *, verbose: bool = False, warmup_cap: int, repeats_cap: int
) -> str:
    return (
        "import dataclasses, pathlib, shutil, sys; "
        "import bench_runtime as runtime; "
        "bench_file = pathlib.Path(sys.argv[1]); "
        "operator_file = pathlib.Path(sys.argv[2]); "
        "target_path = pathlib.Path(sys.argv[3]); "
        "cases, resolution = runtime.load_bench_cases(bench_file, operator_file); "
        "clamped = [dataclasses.replace("
        f"c, warmup=min(c.warmup, {warmup_cap}), repeats=min(c.repeats, {repeats_cap})"
        ") for c in cases]; "
        "result, perf_path = runtime.profile_all_bench_cases("
        "bench_file, operator_file, preloaded=(clamped, resolution), "
        f"verbose={verbose}); "
        "target_path.parent.mkdir(parents=True, exist_ok=True); "
        "shutil.copyfile(perf_path, target_path) if perf_path != target_path else None; "
        "raise SystemExit(int(result['return_code']))"
    )


def _build_remote_perf_counter_run_all_script(*, verbose: bool = False) -> str:
    del verbose
    return (
        "import pathlib, shutil, sys; "
        "import bench_runtime as runtime; "
        "bench_file = pathlib.Path(sys.argv[1]); "
        "operator_file = pathlib.Path(sys.argv[2]); "
        "target_path = pathlib.Path(sys.argv[3]); "
        "result, perf_path = runtime.time_all_bench_cases(bench_file, operator_file, bench_mode='perf-counter'); "
        "target_path.parent.mkdir(parents=True, exist_ok=True); "
        "shutil.copyfile(perf_path, target_path) if perf_path != target_path else None; "
        "raise SystemExit(int(result['return_code']))"
    )


def _build_perf_counter_run_one_case_script(*, verbose: bool = False) -> str:
    del verbose
    return (
        "import json, pathlib, sys; "
        "import bench_runtime as runtime; "
        "bench_file = pathlib.Path(sys.argv[1]); "
        "operator_file = pathlib.Path(sys.argv[2]); "
        "case_id = sys.argv[3]; "
        "cases, resolution = runtime.load_bench_cases(bench_file, operator_file); "
        "case = runtime.select_bench_case(cases, case_id); "
        "record = runtime._time_bench_case(case, resolution, bench_mode='perf-counter'); "
        "payload = {"
        "'case_label': record.case_label, "
        "'kernel_names': record.kernel_names, "
        "'kernel_source': record.kernel_source, "
        "'metrics': record.metrics, "
        "'error_message': record.error_message, "
        "'case_wall_clock_seconds': record.case_wall_clock_seconds, "
        "'bench_mode': record.bench_mode"
        "}; "
        "print(json.dumps(payload, separators=(',', ':')))"
    )


def _build_torch_npu_profiler_run_one_case_script(*, verbose: bool = False) -> str:
    return (
        "import json, pathlib, sys; "
        "import bench_runtime as runtime; "
        "bench_file = pathlib.Path(sys.argv[1]); "
        "operator_file = pathlib.Path(sys.argv[2]); "
        "case_id = sys.argv[3]; "
        "preserved_run_dir_arg = sys.argv[4]; "
        "preserved_run_dir = (None if preserved_run_dir_arg == "
        f"{_PRESERVED_RUN_DIR_NONE_SENTINEL!r} else pathlib.Path(preserved_run_dir_arg)); "
        "record = runtime.profile_bench_case("
        "bench_file, operator_file, case_id, preserved_run_dir=preserved_run_dir, "
        f"verbose={verbose}"
        "); "
        "payload = {"
        "'case_label': record.case_label, "
        "'kernel_names': record.kernel_names, "
        "'kernel_source': record.kernel_source, "
        "'metrics': record.metrics, "
        "'error_message': record.error_message, "
        "'case_wall_clock_seconds': record.case_wall_clock_seconds, "
        "'bench_mode': getattr(record, 'bench_mode', None)"
        "}; "
        "print(json.dumps(payload, separators=(',', ':')))"
    )


def _run_local_torch_npu_profiler_case_in_subprocess(*args: object, **kwargs: object) -> PerfCaseRecord:
    workspace_root, bench_file, operator_file, case_id, device, preserved_run_dir, source_root, verbose = _unpack_call(
        args,
        kwargs,
        (
            "workspace_root", "bench_file", "operator_file", "case_id", "device",
            "preserved_run_dir", "source_root", "verbose",
        ),
        (False,),
    )
    workspace_root = cast(Path, workspace_root)
    bench_file = cast(Path, bench_file)
    operator_file = cast(Path, operator_file)
    case_id = str(case_id)
    device = str(device)
    preserved_run_dir = cast(Path | None, preserved_run_dir)
    source_root = cast(Path, source_root)
    verbose = bool(verbose)
    return _run_local_case_in_subprocess(
        workspace_root, bench_file, operator_file, case_id, device,
        source_root=source_root, verbose=verbose,
        command_script=_build_torch_npu_profiler_run_one_case_script(verbose=verbose),
        parse_result=lambda result: _parse_torch_npu_profiler_case_result_payload(
            result, case_id=case_id, fallback_kernel_source="metadata"
        ),
        preserved_run_dir=preserved_run_dir,
    )


def _run_local_case_in_subprocess(*args: object, **kwargs: object) -> PerfCaseRecord:
    request = _local_case_request_from_call(args, kwargs)
    command, extra_env = _local_case_command_and_env(request)
    result = _run_local_case_process(request, command, extra_env)
    return request.parse_result(result)


def _local_case_request_from_call(
    args: tuple[object, ...], kwargs: dict[str, object],
) -> _LocalCaseRequest:
    values = _unpack_call(
        args,
        kwargs,
        (
            "workspace_root", "bench_file", "operator_file", "case_id", "device",
            "source_root", "verbose", "command_script", "parse_result", "preserved_run_dir",
        ),
        (None,),
    )
    return _LocalCaseRequest(
        cast(Path, values[0]),
        cast(Path, values[1]),
        cast(Path, values[2]),
        str(values[3]),
        str(values[4]),
        cast(Path, values[5]),
        bool(values[6]),
        str(values[7]),
        cast(Callable[[ResultPayload], PerfCaseRecord], values[8]),
        cast(Path | None, values[9]),
    )


def _local_case_command_and_env(request: _LocalCaseRequest) -> tuple[list[str], dict[str, str]]:
    extra_env = affinity_env_for_device(request.device)
    configured_profile_root, _configured_env = _resolve_local_bench_profile_output_root()
    if configured_profile_root:
        extra_env[_LOCAL_BENCH_OUTPUT_DIR_ENV] = str(Path(configured_profile_root).expanduser().resolve())
    extra_env["TRITON_ALWAYS_COMPILE"] = "1"
    command = [
        local_python_executable(),
        "-c",
        request.command_script,
        _case_workspace_command_path(request.bench_file, source_root=request.source_root),
        _case_workspace_command_path(request.operator_file, source_root=request.source_root),
        request.case_id,
    ]
    include_preserved_run_dir = (
        request.preserved_run_dir is not None
        or request.command_script.startswith("import json, pathlib, sys")
    )
    if include_preserved_run_dir:
        command.append(
            _PRESERVED_RUN_DIR_NONE_SENTINEL
            if request.preserved_run_dir is None
            else request.preserved_run_dir.resolve().as_posix()
        )
    return command, extra_env


def _run_local_case_process(
    request: _LocalCaseRequest,
    command: list[str],
    extra_env: dict[str, str],
) -> ResultPayload:
    if request.verbose:
        with stream_target_for_verbosity(True) as stream_target:
            return run_streaming_process(
                command,
                str(request.workspace_root),
                stall_timeout_seconds=eval_stall_timeout_seconds(),
                stdout=stream_target,
                extra_env=extra_env,
            )
    return run_buffered_process(
        command,
        str(request.workspace_root),
        stall_timeout_seconds=eval_stall_timeout_seconds(),
        extra_env=extra_env,
    )


def _run_local_perf_counter_case_in_subprocess(*args: object, **kwargs: object) -> PerfCaseRecord:
    workspace_root, bench_file, operator_file, case_id, device, source_root, verbose = _unpack_call(
        args,
        kwargs,
        ("workspace_root", "bench_file", "operator_file", "case_id", "device", "source_root", "verbose"),
        (False,),
    )
    workspace_root = cast(Path, workspace_root)
    bench_file = cast(Path, bench_file)
    operator_file = cast(Path, operator_file)
    case_id = str(case_id)
    device = str(device)
    source_root, verbose = cast(Path, source_root), bool(verbose)
    return _run_local_case_in_subprocess(
        workspace_root, bench_file, operator_file, case_id, device,
        source_root=source_root, verbose=verbose,
        command_script=_build_perf_counter_run_one_case_script(),
        parse_result=lambda result: _parse_worker_case_result_payload(
            result, case_id=case_id, fallback_kernel_source="metadata", bench_mode="perf-counter"
        ),
    )


def _stage_remote_torch_npu_profiler_case_workspace(*args: object, **kwargs: object) -> str:
    spec, bench_file, operator_file, case_workspace, source_root, json_search_root, verbose, stderr = _unpack_call(
        args,
        kwargs,
        (
            "spec", "bench_file", "operator_file", "case_workspace",
            "source_root", "json_search_root", "verbose", "stderr",
        ),
        (False, None),
    )
    spec = cast(RemoteSpec, spec)
    bench_file = cast(Path, bench_file)
    operator_file = cast(Path, operator_file)
    case_workspace = str(case_workspace)
    source_root, json_search_root = cast(Path, source_root), cast(Path, json_search_root)
    verbose, stderr = bool(verbose), cast(TextIO | None, stderr)
    return _stage_remote_case_workspace(
        spec,
        case_workspace,
        _bench_case_input_paths(
            bench_file,
            operator_file,
            json_search_root=json_search_root,
        ),
        source_root=source_root,
        flat_input_paths=_bench_flat_input_paths(bench_file),
        verbose=verbose,
        stderr=stderr,
    )


def _run_remote_torch_npu_profiler_case(*args: object, **kwargs: object) -> PerfCaseRecord:
    request = _remote_case_request_from_call(args, kwargs)
    return _run_remote_case_common(
        request,
        command_script=_build_torch_npu_profiler_run_one_case_script(verbose=request.verbose),
        parse_result=_parse_torch_npu_profiler_case_result_payload,
        extra_args=[_PRESERVED_RUN_DIR_NONE_SENTINEL],
    )


def _run_remote_perf_counter_case(*args: object, **kwargs: object) -> PerfCaseRecord:
    request = _remote_case_request_from_call(args, kwargs)
    return _run_remote_case_common(
        request,
        command_script=_build_perf_counter_run_one_case_script(),
        parse_result=_parse_worker_case_result_payload,
        bench_mode="perf-counter",
    )


def _remote_case_request_from_call(
    args: tuple[object, ...], kwargs: dict[str, object],
) -> _RemoteCaseRequest:
    values = _unpack_call(
        args,
        kwargs,
        (
            "spec", "case_workspace", "bench_file", "operator_file", "case_id",
            "device", "source_root", "verbose", "stderr",
        ),
        (False, None),
    )
    return _RemoteCaseRequest(
        cast(RemoteSpec, values[0]),
        str(values[1]),
        cast(Path, values[2]),
        cast(Path, values[3]),
        str(values[4]),
        str(values[5]),
        cast(Path, values[6]),
        bool(values[7]),
        cast(TextIO | None, values[8]),
    )


def _run_remote_case_common(
    request: _RemoteCaseRequest,
    *,
    command_script: str,
    parse_result: Callable[..., PerfCaseRecord],
    bench_mode: str | None = None,
    extra_args: Sequence[str] = (),
) -> PerfCaseRecord:
    extra_env = affinity_env_for_device(request.device)
    extra_env["TRITON_ALWAYS_COMPILE"] = "1"
    command = [
        "python3",
        "-c",
        command_script,
        _case_workspace_command_path(request.bench_file, source_root=request.source_root),
        _case_workspace_command_path(request.operator_file, source_root=request.source_root),
        request.case_id,
        *extra_args,
    ]
    result = run_remote_command_streaming(
        request.spec,
        request.case_workspace,
        command,
        verbose=request.verbose,
        stderr=request.stderr,
        extra_env=extra_env,
        stall_timeout_seconds=eval_stall_timeout_seconds(),
    )
    parse_kwargs = {
        "case_id": request.case_id,
        "fallback_kernel_source": "metadata",
    }
    if bench_mode is not None:
        parse_kwargs["bench_mode"] = bench_mode
    return parse_result(result, **parse_kwargs)


def _parse_torch_npu_profiler_case_result_payload(
    result: ResultPayload,
    *,
    case_id: str,
    fallback_kernel_source: str,
) -> PerfCaseRecord:
    return _parse_worker_case_result_payload(
        result,
        case_id=case_id,
        fallback_kernel_source=fallback_kernel_source,
        bench_mode="torch-npu-profiler",
    )


def _parse_worker_case_result_payload(
    result: ResultPayload,
    *,
    case_id: str,
    fallback_kernel_source: str,
    bench_mode: str,
) -> PerfCaseRecord:
    if not result_succeeded(result):
        return _worker_payload_error_record(
            case_id, fallback_kernel_source, bench_mode, _format_worker_command_failure(result, bench_mode)
        )
    stdout_text = str(result["stdout"]).strip()
    if not stdout_text:
        return _worker_payload_error_record(
            case_id, fallback_kernel_source, bench_mode, f"{bench_mode} worker produced no JSON payload"
        )
    try:
        parsed = json.loads(stdout_text.splitlines()[-1].strip())
    except (IndexError, json.JSONDecodeError) as exc:
        return _worker_payload_error_record(
            case_id, fallback_kernel_source, bench_mode, f"failed to parse {bench_mode} worker payload: {exc}"
        )
    if not isinstance(parsed, dict):
        return _worker_payload_error_record(
            case_id, fallback_kernel_source, bench_mode, f"{bench_mode} worker payload must be a JSON object"
        )
    return _worker_payload_record(cast(dict[str, object], parsed), fallback_kernel_source)


def _worker_payload_error_record(
    case_id: str,
    fallback_kernel_source: str,
    bench_mode: str,
    error_message: str,
) -> PerfCaseRecord:
    return PerfCaseRecord(
        case_label=case_id,
        kernel_names=[],
        kernel_source=fallback_kernel_source,
        error_message=error_message,
        case_wall_clock_seconds=None,
        bench_mode=bench_mode,
    )


def _worker_payload_record(parsed: dict[str, object], fallback_kernel_source: str) -> PerfCaseRecord:
    metrics_payload = parsed.get("metrics")
    parsed_bench_mode = parsed.get("bench_mode")
    case_label = str(parsed.get("case_label", ""))
    kernel_names = [str(name) for name in parsed.get("kernel_names", [])]
    kernel_source = str(parsed.get("kernel_source", fallback_kernel_source))
    error_message_raw = parsed.get("error_message")
    case_wall_clock_raw = parsed.get("case_wall_clock_seconds")
    return PerfCaseRecord(
        case_label=case_label,
        kernel_names=kernel_names,
        kernel_source=kernel_source,
        metrics=None if metrics_payload is None else cast(PerfMetrics, metrics_payload),
        error_message=None if error_message_raw is None else str(error_message_raw),
        case_wall_clock_seconds=None
        if case_wall_clock_raw is None
        else float(case_wall_clock_raw),
        bench_mode=None if parsed_bench_mode is None else str(parsed_bench_mode),
    )


def _format_worker_command_failure(result: ResultPayload, bench_mode: str) -> str:
    details = str(result["stderr"]).strip() or str(result["stdout"]).strip()
    prefix = f"{bench_mode} command failed with return code {int(result['return_code'])}"
    return f"{prefix}: {details}" if details else prefix


def _format_torch_npu_profiler_command_failure(result: ResultPayload) -> str:
    return _format_worker_command_failure(result, "torch-npu-profiler")


def _write_torch_npu_profiler_perf(
    operator_file: Path,
    case_records: list[PerfCaseRecord],
    output: str | None = None,
) -> Path:
    return write_perf_lines(
        _resolve_perf_output_path(operator_file, output=output),
        render_perf_case_records_jsonl(
            case_records,
            missing_kernel_match_error="no resolved kernels matched profiler kernel view",
        ),
    )


def _resolve_perf_output_path(operator_file: Path, *, output: str | None = None) -> Path:
    if output is not None:
        return Path(output).expanduser().resolve()
    return perf_output_path(operator_file)


def _build_torch_npu_profiler_result(case_records: list[PerfCaseRecord]) -> ResultPayload:
    errors = [
        f"{record.case_label}: {record.error_message}"
        for record in case_records
        if record.error_message is not None
    ]
    return make_result(
        return_code=1 if errors else 0,
        stdout="",
        stderr="\n".join(errors),
    )
