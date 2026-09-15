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
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
import importlib
import importlib.util
import inspect
import json
import os
import sys
import tempfile
import textwrap
import traceback
from io import StringIO
from pathlib import Path
from typing import Any, TextIO, cast

from debug_device import maybe_print_visible_devices
from metadata_utils import parse_header_metadata
from runtime_bootstrap import bootstrap_torch_npu
from run_runtime import (
    ResultPayload,
    RemoteSpec,
    cleanup_remote_workspace,
    copy_file_from_remote,
    copy_file_to_remote,
    create_remote_workspace,
    eval_stall_timeout_seconds,
    local_python_executable,
    make_result,
    result_succeeded,
    run_buffered_process,
    run_remote_command_streaming,
    run_streaming_process,
)


SCRIPT_DIR = Path(__file__).resolve().parent
_LOCAL_TEST_WORKER_COMMAND = "local-test-worker"
_WARNING_PREFIX = "[WARNING]"


@dataclass(frozen=True)
class DifferentialTestCase:
    case_id: str
    inputs: tuple[object, ...] | list[object]
    fn: Callable[[], object]


@dataclass(frozen=True)
class _RemoteTestRequest:
    test_file: Path
    operator_file: Path
    test_mode: str
    remote: str
    remote_workdir: str | None
    keep_remote_workdir: bool
    verbose: bool
    stderr: TextIO | None


@contextmanager
def _always_compile_enabled() -> Iterator[None]:
    previous = os.environ.get("TRITON_ALWAYS_COMPILE")
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    try:
        yield
    finally:
        if previous is None:
            del os.environ["TRITON_ALWAYS_COMPILE"]
        else:
            os.environ["TRITON_ALWAYS_COMPILE"] = previous


def _test_result_from_exception(stdout: StringIO, stderr: StringIO) -> ResultPayload:
    return make_result(
        return_code=1,
        stdout=stdout.getvalue(),
        stderr=stderr.getvalue() + traceback.format_exc(),
    )


def _differential_archive_path(operator_file: Path) -> Path:
    return operator_file.parent / f"{operator_file.stem}_result.pt"


def _write_local_test_worker_payload(
    result_file: Path,
    result: ResultPayload,
    archived_result: Path | None,
) -> None:
    result_file.write_text(
        json.dumps(
            {
                "result": {
                    "return_code": int(result["return_code"]),
                    "stdout": str(result["stdout"]),
                    "stderr": str(result["stderr"]),
                    "stalled": bool(result["stalled"]),
                    "session_id": result["session_id"],
                },
                "archived_result": None if archived_result is None else str(archived_result.resolve()),
            }
        ),
        encoding="utf-8",
    )


def _read_local_test_worker_payload(result_file: Path) -> tuple[ResultPayload, Path | None]:
    payload = json.loads(result_file.read_text(encoding="utf-8"))
    result_payload = payload["result"]
    result = make_result(
        return_code=int(result_payload["return_code"]),
        stdout=str(result_payload["stdout"]),
        stderr=str(result_payload["stderr"]),
        stalled=bool(result_payload["stalled"]),
        session_id=cast(str | None, result_payload["session_id"]),
    )
    archived_raw = payload.get("archived_result")
    archived_result = None if archived_raw is None else Path(str(archived_raw)).expanduser().resolve()
    return result, archived_result


def _merge_failed_worker_result(result: ResultPayload) -> tuple[ResultPayload, None]:
    return result, None


def _local_test_worker_command(
    test_file: Path,
    operator_file: Path,
    test_mode: str,
    result_file: Path,
    *,
    verbose: bool,
) -> list[str]:
    command = [
        local_python_executable(),
        str(Path(__file__).resolve()),
        _LOCAL_TEST_WORKER_COMMAND,
        "--test-file",
        str(test_file.resolve()),
        "--operator-file",
        str(operator_file.resolve()),
        "--test-mode",
        test_mode,
        "--result-file",
        str(result_file),
    ]
    if verbose:
        command.append("--verbose")
    return command


def _build_local_test_worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=Path(__file__).name)
    parser.add_argument("command", choices=[_LOCAL_TEST_WORKER_COMMAND])
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--operator-file", required=True)
    parser.add_argument("--test-mode", choices=["standalone", "differential"], required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--verbose", action="store_true")
    return parser


def _run_local_test_worker(
    test_file: Path,
    operator_file: Path,
    test_mode: str,
    result_file: Path,
    *,
    verbose: bool,
) -> int:
    if test_mode == "standalone":
        result = _run_import_only_standalone_test(test_file, operator_file, verbose=verbose)
        archived_result = None
    elif test_mode == "differential":
        archive_path = _differential_archive_path(operator_file)
        result = _run_declarative_differential_test(test_file, operator_file, archive_path, verbose=verbose)
        archived_result = archive_path if result_succeeded(result) and archive_path.exists() else None
    else:
        raise ValueError(f"Unsupported test mode: {test_mode}")
    _write_local_test_worker_payload(result_file, result, archived_result)
    return 0


def _run_local_test_worker_main(argv: list[str] | None = None) -> int:
    parser = _build_local_test_worker_parser()
    args = parser.parse_args(argv)
    return _run_local_test_worker(
        Path(args.test_file).expanduser().resolve(),
        Path(args.operator_file).expanduser().resolve(),
        cast(str, args.test_mode),
        Path(args.result_file).expanduser().resolve(),
        verbose=bool(args.verbose),
    )


def run_local_test(
    test_file: Path,
    operator_file: Path,
    test_mode: str,
    *,
    verbose: bool = False,
) -> tuple[ResultPayload, Path | None]:
    maybe_print_visible_devices()
    with tempfile.TemporaryDirectory() as tmp:
        result_file = Path(tmp) / "local-test-result.json"
        command = _local_test_worker_command(
            test_file,
            operator_file,
            test_mode,
            result_file,
            verbose=verbose,
        )
        if verbose:
            runner_result = run_streaming_process(
                command,
                str(test_file.resolve().parent),
                stall_timeout_seconds=eval_stall_timeout_seconds(),
            )
        else:
            runner_result = run_buffered_process(
                command,
                str(test_file.resolve().parent),
                stall_timeout_seconds=eval_stall_timeout_seconds(),
            )
        if result_succeeded(runner_result):
            if not result_file.exists():
                return _merge_failed_worker_result(
                    make_result(
                        return_code=1,
                        stdout=str(runner_result["stdout"]),
                        stderr=(
                            str(runner_result["stderr"])
                            + f"Local test worker did not write result payload: {result_file}"
                        ),
                        stalled=bool(runner_result["stalled"]),
                        session_id=runner_result["session_id"],
                    )
                )
            result, archived_result = _read_local_test_worker_payload(result_file)
            return _filter_result_payload(result, verbose=verbose), archived_result
        return _merge_failed_worker_result(runner_result)


def parse_test_metadata(test_file: Path) -> dict[str, str]:
    return parse_header_metadata(test_file)


def load_differential_test_cases(
    test_file: Path,
    operator_file: Path,
) -> list[DifferentialTestCase]:
    test_path = test_file.resolve()
    operator_path = operator_file.resolve()
    bootstrap_torch_npu()
    with _temporary_sys_path_entries(test_path.parent, operator_path.parent, SCRIPT_DIR):
        test_module = _load_module(test_path, f"differential_test_{test_path.stem}")
        build_operator_api = _require_callable(test_module, "build_operator_api", test_path)
        build_cases = _require_callable(test_module, "build_differential_test_cases", test_path)
        operator_module = _load_module(operator_path, f"differential_operator_{operator_path.stem}")
        operator_api = build_operator_api(operator_module)
        raw_cases = build_cases(operator_api)
    return _normalize_differential_cases(raw_cases)


def _normalize_differential_cases(raw_cases: object) -> list[DifferentialTestCase]:
    return [
        DifferentialTestCase(
            case_id=cast(str, record["id"]),
            inputs=cast(tuple[object, ...] | list[object], record["inputs"]),
            fn=cast(Callable[[], object], record["fn"]),
        )
        for record in _normalize_differential_case_records(raw_cases)
    ]


def _normalize_differential_case_records(raw_cases: object) -> list[dict[str, object]]:
    if isinstance(raw_cases, (str, bytes)) or isinstance(raw_cases, Mapping) or not isinstance(raw_cases, Iterable):
        raise ValueError("Differential test hook 'build_differential_test_cases' must return an iterable of cases")
    records: list[dict[str, object]] = []
    seen_case_ids: set[str] = set()
    for raw_case in cast(Iterable[object], raw_cases):
        if not isinstance(raw_case, Mapping):
            raise ValueError("Differential test cases must be mappings")
        case_map = cast(Mapping[str, object], raw_case)
        case_id = case_map.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("Differential test case is missing required string field 'id'")
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate differential test case id: {case_id}")
        raw_inputs = case_map.get("inputs")
        if not isinstance(raw_inputs, (list, tuple)):
            raise ValueError(
                f"Differential test case '{case_id}' is missing required list/tuple field 'inputs'"
            )
        case_fn = case_map.get("fn")
        if not callable(case_fn):
            raise ValueError(f"Differential test case '{case_id}' is missing required callable field 'fn'")
        seen_case_ids.add(case_id)
        if isinstance(raw_inputs, tuple):
            normalized_inputs: tuple[object, ...] | list[object] = tuple(cast(tuple[object, ...], raw_inputs))
        else:
            normalized_inputs = list(cast(list[object], raw_inputs))
        records.append({"id": case_id, "inputs": normalized_inputs, "fn": case_fn})
    if not records:
        raise ValueError("Differential test hook 'build_differential_test_cases' returned no cases")
    return records


def _run_import_only_standalone_test(
    test_file: Path,
    operator_file: Path,
    *,
    verbose: bool = False,
) -> ResultPayload:
    real_stderr = sys.stderr
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    with _always_compile_enabled():
        try:
            _run_standalone_test_body(
                test_file,
                operator_file,
                stdout_buffer,
                stderr_buffer,
                verbose=verbose,
                real_stderr=real_stderr,
            )
        except Exception:
            if verbose:
                print("[run-test] FAILED", file=real_stderr)
            return _test_result_from_exception(stdout_buffer, stderr_buffer)
    if verbose:
        print("[run-test] PASSED", file=real_stderr)
    return make_result(return_code=0, stdout=stdout_buffer.getvalue(), stderr=stderr_buffer.getvalue())


def _run_standalone_test_body(
    test_file: Path,
    operator_file: Path,
    stdout_buffer: StringIO,
    stderr_buffer: StringIO,
    **values: object,
) -> None:
    verbose = bool(values["verbose"])
    real_stderr = cast(TextIO, values["real_stderr"])
    test_path, operator_path = test_file.resolve(), operator_file.resolve()
    metadata = parse_test_metadata(test_path)
    if verbose:
        _print_standalone_test_start(test_path, operator_path, metadata, real_stderr)
    bootstrap_torch_npu()
    with _temporary_sys_path_entries(test_path.parent, operator_path.parent, SCRIPT_DIR):
        test_module = _load_module(test_path, f"standalone_test_{test_path.stem}")
        main_fn = _require_callable(test_module, "main", test_path, kind="Standalone test module")
        operator_module = _load_module(operator_path, f"standalone_operator_{operator_path.stem}")
        operator_api = _resolve_operator_api(operator_module, metadata, operator_path)
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            main_fn(operator_api)
            _maybe_synchronize_torch()


def _print_standalone_test_start(
    test_path: Path, operator_path: Path, metadata: dict[str, str], stderr: TextIO,
) -> None:
    print("[run-test] mode: standalone", file=stderr)
    print(f"[run-test] test file: {test_path}", file=stderr)
    print(f"[run-test] operator file: {operator_path}", file=stderr)
    print(f"[run-test] operator api: {metadata.get('api-name', '?')} ({metadata.get('api-kind', '?')})", file=stderr)
    print("[run-test] running...", file=stderr)


def _run_declarative_differential_test(
    test_file: Path,
    operator_file: Path,
    archive_path: Path,
    *,
    verbose: bool = False,
) -> ResultPayload:
    real_stderr = sys.stderr
    try:
        bootstrap_torch_npu()
        torch = importlib.import_module("torch")
    except ImportError as exc:
        return make_result(
            return_code=1,
            stdout="",
            stderr=f"Missing differential test dependency: {exc}",
        )
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    with _always_compile_enabled():
        try:
            compute, records = _run_differential_cases(
                test_file,
                operator_file,
                torch,
                verbose=verbose,
                real_stderr=real_stderr,
                stdout_buffer=stdout_buffer,
                stderr_buffer=stderr_buffer,
            )
            torch.save({"compute": compute, "cases": records}, archive_path)
        except Exception:
            if verbose:
                print("[run-test] FAILED", file=real_stderr)
            return _test_result_from_exception(stdout_buffer, stderr_buffer)
    if verbose:
        print(f"[run-test] archive saved: {archive_path}", file=real_stderr)
    return make_result(return_code=0, stdout=stdout_buffer.getvalue(), stderr=stderr_buffer.getvalue())


def _run_differential_cases(
    test_file: Path,
    operator_file: Path,
    torch_module: Any,
    **values: object,
) -> tuple[bool, list[dict[str, object]]]:
    verbose = bool(values["verbose"])
    real_stderr = cast(TextIO, values["real_stderr"])
    stdout_buffer = cast(StringIO, values["stdout_buffer"])
    stderr_buffer = cast(StringIO, values["stderr_buffer"])
    metadata = parse_test_metadata(test_file)
    compute = _compute_flag_from_metadata(metadata)
    cases = load_differential_test_cases(test_file, operator_file)
    if verbose:
        print("[run-test] mode: differential", file=real_stderr)
        print(f"[run-test] test file: {test_file}", file=real_stderr)
        print(f"[run-test] operator file: {operator_file}", file=real_stderr)
        print(f"[run-test] total cases: {len(cases)}", file=real_stderr)
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        return compute, _execute_differential_cases(cases, torch_module, verbose, real_stderr)


def _execute_differential_cases(
    cases: list[DifferentialTestCase], torch_module: Any, verbose: bool, stderr: TextIO,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        if verbose:
            print(f"[run-test] [{index}/{len(cases)}] {case.case_id} ...", file=stderr, end="", flush=True)
        records.append({"id": case.case_id, "inputs": case.inputs, "result": case.fn()})
        _synchronize(torch_module)
        if verbose:
            print(" ok", file=stderr)
    return records


def _filter_result_payload(result: ResultPayload, *, verbose: bool) -> ResultPayload:
    if verbose:
        return result
    filtered_stdout = _filter_known_warning_lines(str(result["stdout"]))
    filtered_stderr = _filter_known_warning_lines(str(result["stderr"]))
    if filtered_stdout == result["stdout"] and filtered_stderr == result["stderr"]:
        return result
    return make_result(
        return_code=int(result["return_code"]),
        stdout=filtered_stdout,
        stderr=filtered_stderr,
        stalled=bool(result["stalled"]),
        session_id=result["session_id"],
    )


def _filter_known_warning_lines(text: str) -> str:
    filtered_lines = [
        line
        for line in text.splitlines(keepends=True)
        if not line.rstrip("\r\n").startswith(_WARNING_PREFIX)
    ]
    return "".join(filtered_lines)


def _load_module(module_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"{module_name}_{module_path.stem}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _require_callable(
    module: object,
    name: str,
    source_path: Path,
    *,
    kind: str = "Differential test module",
) -> Callable[..., Any]:
    candidate = getattr(module, name, None)
    if not callable(candidate):
        raise ValueError(f"{kind} missing required hook '{name}': {source_path}")
    return cast(Callable[..., Any], candidate)


def _resolve_operator_api(operator_module: object, metadata: Mapping[str, str], operator_path: Path) -> object:
    api_name = metadata.get("api-name")
    api_kind = metadata.get("api-kind")
    if not api_name:
        raise ValueError(f"Test metadata is missing required 'api-name' entry: {operator_path}")
    if api_kind not in {"triton-wrapper", "torch-function", "torch-module"}:
        raise ValueError(f"Test metadata is missing required 'api-kind' entry: {operator_path}")
    candidate = getattr(operator_module, api_name, None)
    if candidate is None:
        raise ValueError(f"Runtime operator file is missing required API '{api_name}': {operator_path}")
    if api_kind == "torch-module":
        if not callable(candidate):
            raise ValueError(f"Runtime operator API '{api_name}' is not callable: {operator_path}")
        try:
            return candidate()
        except TypeError as exc:
            raise RuntimeError(
                "torch-module entrypoints must support no-argument construction; "
                "constructor arguments are not supported in generated harnesses"
            ) from exc
    return candidate


def _compute_flag_from_metadata(metadata: Mapping[str, str]) -> bool:
    return _parse_compute_kind(metadata.get("compute-kind"))


def _maybe_synchronize_torch() -> None:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return
    _synchronize(torch)


def _synchronize(torch_module: Any) -> None:
    if hasattr(torch_module, "npu"):
        torch_module.npu.synchronize()


def run_remote_test(*args: object, **kwargs: object) -> tuple[ResultPayload, Path | None, str]:
    names = ("test_file", "operator_file", "test_mode", "remote", "remote_workdir")
    values = list(args) + [kwargs[name] for name in names[len(args):]]
    test_file, operator_file, test_mode, remote, remote_workdir = values
    keep_remote_workdir = bool(args[5] if len(args) > 5 else kwargs.get("keep_remote_workdir", False))
    verbose = bool(args[6] if len(args) > 6 else kwargs.get("verbose", False))
    stderr = cast(TextIO | None, args[7] if len(args) > 7 else kwargs.get("stderr"))
    test_file, operator_file = cast(Path, test_file), cast(Path, operator_file)
    test_mode, remote = str(test_mode), str(remote)
    remote_workdir = cast(str | None, remote_workdir)
    maybe_print_visible_devices()
    request = _RemoteTestRequest(
        test_file=test_file,
        operator_file=operator_file,
        test_mode=test_mode,
        remote=remote,
        remote_workdir=remote_workdir,
        keep_remote_workdir=keep_remote_workdir,
        verbose=verbose,
        stderr=stderr,
    )
    spec, remote_workspace = create_remote_workspace(
        request.remote, request.remote_workdir, verbose=request.verbose, stderr=request.stderr
    )
    try:
        return _execute_remote_test(spec, remote_workspace, request)
    finally:
        if not request.keep_remote_workdir:
            cleanup_remote_workspace(spec, remote_workspace, verbose=request.verbose, stderr=request.stderr)


def _execute_remote_test(
    spec: RemoteSpec,
    remote_workspace: str,
    request: _RemoteTestRequest,
) -> tuple[ResultPayload, Path | None, str]:
    _stage_remote_test_inputs(spec, remote_workspace, request)
    if request.test_mode == "standalone":
        result = _run_remote_test_command(
            spec, remote_workspace,
            _build_remote_standalone_command(request.test_file.name, request.operator_file.name), request,
        )
        return _filter_result_payload(result, verbose=request.verbose), None, remote_workspace
    if request.test_mode == "differential":
        result = _run_remote_test_command(
            spec, remote_workspace,
            _build_remote_differential_command(request.test_file.name, request.operator_file.name), request,
        )
        archive = _collect_remote_test_archive(spec, remote_workspace, request, result)
        return _filter_result_payload(result, verbose=request.verbose), archive, remote_workspace
    raise ValueError(f"Unsupported test mode: {request.test_mode}")


def _stage_remote_test_inputs(spec: RemoteSpec, remote_workspace: str, request: _RemoteTestRequest) -> None:
    paths = (request.test_file, request.operator_file, SCRIPT_DIR / "npu_compare.py")
    for path in paths:
        copy_file_to_remote(
            spec, path, f"{remote_workspace}/{path.name}", verbose=request.verbose, stderr=request.stderr
        )


def _run_remote_test_command(
    spec: RemoteSpec,
    remote_workspace: str,
    command: list[str],
    request: _RemoteTestRequest,
) -> ResultPayload:
    return run_remote_command_streaming(
        spec, remote_workspace, command, stall_timeout_seconds=eval_stall_timeout_seconds(),
        verbose=request.verbose, stderr=request.stderr, extra_env={"TRITON_ALWAYS_COMPILE": "1"},
    )


def _collect_remote_test_archive(
    spec: RemoteSpec,
    remote_workspace: str,
    request: _RemoteTestRequest,
    result: ResultPayload,
) -> Path | None:
    if not result_succeeded(result):
        return None
    return _copy_remote_differential_archive(
        spec, remote_workspace, _differential_archive_path(request.operator_file),
        verbose=request.verbose, stderr=request.stderr,
    )


def _copy_remote_differential_archive(
    spec: RemoteSpec,
    remote_workspace: str,
    archive_path: Path,
    *,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> Path:
    copy_file_from_remote(
        spec,
        f"{remote_workspace}/{archive_path.name}",
        archive_path,
        verbose=verbose,
        stderr=stderr,
    )
    return archive_path


def _remote_script_command(remote_script: str) -> list[str]:
    return ["python3", "-c", remote_script]


def _remote_module_loader_source() -> str:
    return '''
def _load_module(module_path, module_name):
    spec = importlib.util.spec_from_file_location(f"{module_name}_{Path(module_path).stem}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
'''


def _remote_metadata_parser_source() -> str:
    return '''
def _parse_metadata(test_file):
    metadata = {}
    for line in Path(test_file).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            if metadata:
                break
            continue
        if not stripped.startswith("#"):
            break
        body = stripped[1:].strip()
        if ":" not in body:
            continue
        key, value = body.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata
'''


def _remote_standalone_body(test_name: str, operator_name: str) -> str:
    return f'''
test_file = Path({test_name!r})
operator_file = Path({operator_name!r})
sys.path.insert(0, str(Path(".").resolve()))
stdout_buffer = StringIO()
stderr_buffer = StringIO()
try:
    metadata = _parse_metadata(test_file)
    test_module = _load_module(test_file, f"standalone_test_{{test_file.stem}}")
    main_fn = _require_callable(test_module, "main", test_file)
    operator_module = _load_module(operator_file, f"standalone_operator_{{operator_file.stem}}")
    operator_api = _resolve_operator_api(operator_module, metadata, operator_file)
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        main_fn(operator_api)
        _maybe_synchronize()
    sys.stdout.write(stdout_buffer.getvalue())
    sys.stderr.write(stderr_buffer.getvalue())
except Exception:
    sys.stdout.write(stdout_buffer.getvalue())
    sys.stderr.write(stderr_buffer.getvalue())
    traceback.print_exc()
    raise
'''


def _remote_differential_body(test_name: str, operator_name: str) -> str:
    return f'''
test_file = Path({test_name!r})
operator_file = Path({operator_name!r})
archive_file = operator_file.parent / f"{{operator_file.stem}}_result.pt"
sys.path.insert(0, str(Path(".").resolve()))
try:
    torch = importlib.import_module("torch")
    metadata = _parse_metadata(test_file)
    compute = _compute_flag(metadata)
    test_module = _load_module(test_file, f"differential_test_{{test_file.stem}}")
    build_operator_api = _require_callable(test_module, "build_operator_api", test_file)
    build_cases = _require_callable(test_module, "build_differential_test_cases", test_file)
    operator_module = _load_module(operator_file, f"differential_operator_{{operator_file.stem}}")
    raw_cases = build_cases(build_operator_api(operator_module))
    case_records = _normalize_differential_case_records(raw_cases)
    records = []
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        for case_record in case_records:
            case_fn = case_record["fn"]
            records.append({{"id": case_record["id"], "inputs": case_record["inputs"], "result": case_fn()}})
            _synchronize(torch)
    torch.save({{"compute": compute, "cases": records}}, archive_file)
except Exception:
    traceback.print_exc()
    raise
'''


def _build_remote_standalone_command(test_name: str, operator_name: str) -> list[str]:
    remote_script = f"""
import importlib.util
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

{_remote_module_loader_source()}

{_remote_metadata_parser_source()}

def _require_callable(module, name, source_path):
    candidate = getattr(module, name, None)
    if not callable(candidate):
        raise ValueError(f"Standalone test module missing required hook '{{name}}': {{source_path}}")
    return candidate

def _resolve_operator_api(operator_module, metadata, operator_path):
    api_name = metadata.get("api-name")
    api_kind = metadata.get("api-kind")
    if not api_name:
        raise ValueError(f"Test metadata is missing required 'api-name' entry: {{operator_path}}")
    if api_kind not in {{"triton-wrapper", "torch-function", "torch-module"}}:
        raise ValueError(f"Test metadata is missing required 'api-kind' entry: {{operator_path}}")
    candidate = getattr(operator_module, api_name, None)
    if candidate is None:
        raise ValueError(f"Runtime operator file is missing required API '{{api_name}}': {{operator_path}}")
    if api_kind == "torch-module":
        return candidate()
    return candidate

def _maybe_synchronize():
    try:
        import torch
    except ImportError:
        return
    if hasattr(torch, "npu"):
        torch.npu.synchronize()
""" + _remote_standalone_body(test_name, operator_name)
    return _remote_script_command(remote_script)


def _build_remote_differential_command(test_name: str, operator_name: str) -> list[str]:
    remote_script = f"""
import importlib
import importlib.util
import sys
import traceback
from collections.abc import Iterable, Mapping
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import cast

{_remote_module_loader_source()}

{_remote_metadata_parser_source()}

{_remote_function_source(_parse_compute_kind)}

{_remote_function_source(_normalize_differential_case_records)}

def _compute_flag(metadata):
    return _parse_compute_kind(metadata.get("compute-kind"))

def _require_callable(module, name, test_path):
    candidate = getattr(module, name, None)
    if not callable(candidate):
        raise ValueError(f"Differential test module missing required hook '{{name}}': {{test_path}}")
    return candidate

def _synchronize(torch_module):
    if hasattr(torch_module, "npu"):
        torch_module.npu.synchronize()
""" + _remote_differential_body(test_name, operator_name)
    return _remote_script_command(remote_script)


def _remote_function_source(function: Callable[..., Any]) -> str:
    return textwrap.dedent(inspect.getsource(function)).strip()


@contextmanager
def _temporary_sys_path_entries(*paths: Path) -> Iterator[None]:
    added: list[str] = []
    try:
        for path in paths:
            text = str(path)
            if text not in sys.path:
                sys.path.insert(0, text)
                added.append(text)
        yield
    finally:
        for path in reversed(added):
            if path in sys.path:
                sys.path.remove(path)


def _parse_compute_kind(raw_value: object) -> bool:
    if raw_value is None:
        return True
    if not isinstance(raw_value, str):
        raise ValueError("Test metadata 'compute-kind' must be 'compute' or 'non-compute'")
    normalized = raw_value.strip().lower()
    if normalized == "compute":
        return True
    if normalized == "non-compute":
        return False
    raise ValueError("Test metadata 'compute-kind' must be 'compute' or 'non-compute'")


if __name__ == "__main__":
    try:
        raise SystemExit(_run_local_test_worker_main())
    except Exception as exc:
        traceback.print_exc()
        raise SystemExit(1) from exc
