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

import os
from pathlib import Path
from typing import TextIO, cast

from bench_runner import stream_target_for_verbosity
from runtime_loader import load_script_module
from run_runtime import (
    RemoteSpec,
    ResultPayload,
    env_int,
    cleanup_remote_workspace,
    copy_directory_from_remote,
    copy_file_to_remote,
    create_remote_workspace,
    result_succeeded,
    run_remote_command_buffered,
    run_remote_command_streaming,
)


class _RemoteProfileRequest:
    def __init__(self, **values: object) -> None:
        self.remote = cast(str, values["remote"])
        self.remote_workdir = cast(str | None, values.get("remote_workdir"))
        self.case_id = cast(str | None, values.get("case_id"))
        self.keep_remote_workdir = bool(values.get("keep_remote_workdir", False))
        self.verbose = bool(values.get("verbose", False))
        self.stderr = cast(TextIO | None, values.get("stderr"))


def _profile_timeout() -> int:
    return env_int("TRITON_AGENT_PROFILE_TIMEOUT_SECONDS", 900)


def run_local_profile_bench(
    bench_file: Path,
    operator_file: Path,
    case_id: str | None = None,
    kernel_name: str | None = None,
) -> tuple[ResultPayload, Path | None]:
    del kernel_name
    if case_id is None:
        raise ValueError("torch-npu-profiler benchmark profiling requires --case-id <id>.")
    result = _run_local_profile_torch_npu_profiler(
        bench_file, operator_file, case_id,
    )
    if not result_succeeded(result):
        return result, None
    profile_dir = _resolve_local_profile_dir(bench_file.parent)
    return result, profile_dir


def run_remote_profile_bench(*args: object, **kwargs: object) -> tuple[ResultPayload, Path | None, str]:
    bench_file = cast(Path, args[0] if args else kwargs["bench_file"])
    operator_file = cast(Path, args[1] if len(args) > 1 else kwargs["operator_file"])
    remote = str(args[2] if len(args) > 2 else kwargs["remote"])
    remote_workdir = cast(str | None, args[3] if len(args) > 3 else kwargs.get("remote_workdir"))
    case_id = cast(str | None, args[4] if len(args) > 4 else kwargs.get("case_id"))
    keep_remote_workdir = bool(args[5] if len(args) > 5 else kwargs.get("keep_remote_workdir", False))
    verbose = bool(args[6] if len(args) > 6 else kwargs.get("verbose", False))
    stderr = cast(TextIO | None, args[7] if len(args) > 7 else kwargs.get("stderr"))
    request = _RemoteProfileRequest(
        remote=remote,
        remote_workdir=remote_workdir,
        case_id=case_id,
        keep_remote_workdir=keep_remote_workdir,
        verbose=verbose,
        stderr=stderr,
    )
    spec, remote_workspace = create_remote_workspace(
        request.remote, request.remote_workdir, verbose=request.verbose, stderr=request.stderr
    )
    try:
        return _collect_remote_profile(spec, remote_workspace, bench_file, operator_file, request)
    finally:
        if not request.keep_remote_workdir:
            cleanup_remote_workspace(spec, remote_workspace, verbose=request.verbose, stderr=request.stderr)


def _collect_remote_profile(
    spec: RemoteSpec,
    remote_workspace: str,
    bench_file: Path,
    operator_file: Path,
    request: _RemoteProfileRequest,
) -> tuple[ResultPayload, Path | None, str]:
    _copy_profile_inputs(spec, remote_workspace, bench_file, operator_file, request)
    if request.case_id is None:
        raise ValueError("torch-npu-profiler benchmark profiling requires --case-id <id>.")
    result = _run_remote_profile_torch_npu_profiler(
        spec, remote_workspace, bench_file, operator_file, request.case_id,
        verbose=request.verbose, stderr=request.stderr,
    )
    if not result_succeeded(result):
        return result, None, remote_workspace
    profile_dir = _copy_remote_profile_output(spec, remote_workspace, operator_file, request)
    return result, profile_dir, remote_workspace


def _copy_profile_inputs(
    spec: RemoteSpec,
    remote_workspace: str,
    bench_file: Path,
    operator_file: Path,
    request: _RemoteProfileRequest,
) -> None:
    for path in (bench_file, operator_file):
        copy_file_to_remote(
            spec, path, f"{remote_workspace}/{path.name}", verbose=request.verbose, stderr=request.stderr
        )


def _copy_remote_profile_output(
    spec: RemoteSpec,
    remote_workspace: str,
    operator_file: Path,
    request: _RemoteProfileRequest,
) -> Path:
    profile_name = _resolve_remote_profile_name(
        spec, remote_workspace, verbose=request.verbose, stderr=request.stderr
    )
    local_profile_dir = operator_file.parent / profile_name
    if local_profile_dir.exists():
        raise FileExistsError(f"Local profile directory already exists: {local_profile_dir}")
    copy_directory_from_remote(
        spec, f"{remote_workspace}/{profile_name}", local_profile_dir,
        verbose=request.verbose, stderr=request.stderr,
    )
    _validate_profile_dir(local_profile_dir)
    return local_profile_dir


def _run_local_profile_torch_npu_profiler(
    bench_file: Path,
    operator_file: Path,
    case_id: str,
) -> ResultPayload:
    prev = os.environ.get("TRITON_ALWAYS_COMPILE")
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    try:
        return profile_local_torch_npu_profiler_case(bench_file, operator_file, case_id)
    finally:
        if prev is None:
            del os.environ["TRITON_ALWAYS_COMPILE"]
        else:
            os.environ["TRITON_ALWAYS_COMPILE"] = prev


def _run_remote_profile_torch_npu_profiler(*args: object, **kwargs: object) -> ResultPayload:
    spec, remote_workspace, bench_file, operator_file, case_id = args[:5]
    verbose = bool(kwargs.get("verbose", False))
    stderr = cast(TextIO | None, kwargs.get("stderr"))
    spec, remote_workspace = cast(RemoteSpec, spec), str(remote_workspace)
    bench_file, operator_file, case_id = cast(Path, bench_file), cast(Path, operator_file), str(case_id)
    for support_path in _bench_runtime_support_paths():
        copy_file_to_remote(
            spec,
            support_path,
            f"{remote_workspace}/{support_path.name}",
            verbose=verbose,
            stderr=stderr,
        )
    extra_env = {"TRITON_ALWAYS_COMPILE": "1"}
    with stream_target_for_verbosity(verbose) as stream_target:
        return run_remote_command_streaming(
            spec,
            remote_workspace,
            [
                "python3",
                "-c",
                _build_remote_torch_npu_profiler_profile_script(),
                bench_file.name,
                operator_file.name,
                case_id,
            ],
            stdout=stream_target,
            stall_timeout_seconds=_profile_timeout(),
            verbose=verbose,
            stderr=stderr,
            extra_env=extra_env,
        )


def _resolve_local_profile_dir(search_root: Path) -> Path:
    candidates = [
        candidate
        for candidate in search_root.iterdir()
        if candidate.is_dir() and candidate.name.startswith("PROF_")
    ]
    if not candidates:
        raise FileNotFoundError(f"No PROF_* directory found under {search_root}")
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    _validate_profile_dir(latest)
    return latest


def _resolve_remote_profile_name(
    spec: RemoteSpec,
    remote_workspace: str,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> str:
    result = run_remote_command_buffered(
        spec,
        remote_workspace,
        (
            "python3 -c "
            + repr(
                "import pathlib; "
                "candidates = [p for p in pathlib.Path('.').iterdir() if p.is_dir() and p.name.startswith('PROF_')]; "
                "candidates.sort(key=lambda p: p.stat().st_mtime); "
                "print(candidates[-1].name if candidates else '')"
            )
        ),
        verbose=verbose,
        stderr=stderr,
    )
    if not result_succeeded(result):
        message = str(result["stderr"]) or str(result["stdout"])
        raise RuntimeError(message or "Failed to resolve remote profiler output.")
    profile_name = str(result["stdout"]).strip().splitlines()[-1].strip() if str(result["stdout"]).strip() else ""
    if not profile_name:
        raise FileNotFoundError(f"No PROF_* directory found in remote workspace {remote_workspace}")
    return profile_name


def _validate_profile_dir(profile_dir: Path) -> None:
    output_dir = profile_dir / "mindstudio_profiler_output"
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Profiler output is incomplete: missing {output_dir}")
    if not list(output_dir.glob("op_statistic_*.csv")):
        raise FileNotFoundError(f"Profiler output is incomplete: no op_statistic_*.csv under {output_dir}")


def profile_local_torch_npu_profiler_case(
    bench_file: Path,
    operator_file: Path,
    case_id: str,
) -> ResultPayload:
    runtime = _load_bench_runtime_module()
    return runtime.profile_bench_case_quick(bench_file, operator_file, case_id)


def _bench_runtime_script_path() -> Path:
    return Path(__file__).resolve().with_name("bench_runtime.py")


def _bench_runtime_support_paths() -> list[Path]:
    runtime = _load_bench_runtime_module()
    return list(runtime.runtime_support_paths())


def _load_bench_runtime_module():
    script_path = _bench_runtime_script_path()
    return load_script_module(
        script_path,
        f"triton_agent_bench_runtime_{script_path.stem}",
        remove_after_load=True,
    )


def _build_remote_torch_npu_profiler_profile_script() -> str:
    return (
        "import pathlib, sys; "
        "import bench_runtime as runtime; "
        "bench_file = pathlib.Path(sys.argv[1]); "
        "operator_file = pathlib.Path(sys.argv[2]); "
        "case_id = sys.argv[3]; "
        "result = runtime.profile_bench_case_quick(bench_file, operator_file, case_id); "
        "raise SystemExit(int(result['return_code']))"
    )
