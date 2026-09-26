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

import errno
import locale
import os
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Optional, TextIO, TypedDict, cast

from result_payload import ResultPayload, make_result

_IS_WINDOWS = sys.platform == "win32"
_BLOCKS_PARALLEL_ENV = "TRITON_ALL_BLOCKS_PARALLEL"
_BLOCKS_PARALLEL_UNSAFE_VALUE = "1"
_BLOCKS_PARALLEL_SAFE_VALUE = "0"

if not _IS_WINDOWS:
    import pty
    import select
else:
    pty = None
    select = None


class RemoteSpec(TypedDict):
    user_host: str
    port: int | None


@dataclass(frozen=True)
class _RemoteCommandRequest:
    spec: RemoteSpec
    remote_workspace: str
    remote_command: str | Sequence[str]
    stdout: TextIO | None = None
    verbose: bool = False
    stderr: TextIO | None = None
    extra_env: dict[str, str] | None = None
    stall_timeout_seconds: int | None = None


@dataclass
class _WindowsStreamState:
    target: TextIO | None
    output_chunks: list[str] = field(default_factory=list)
    last_output_at: float = field(default_factory=time.monotonic)
    stalled: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


def local_python_executable() -> str:
    configured = os.environ.get("TRITON_AGENT_PYTHON", "").strip()
    if configured:
        return configured
    if getattr(sys, "frozen", False):
        return "python"
    return sys.executable


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {raw!r}")
    return value


def _ssh_timeout() -> int:
    return env_int("TRITON_AGENT_SSH_TIMEOUT_SECONDS", 120)


def _scp_timeout() -> int:
    return env_int("TRITON_AGENT_SCP_TIMEOUT_SECONDS", 300)


def eval_stall_timeout_seconds() -> int:
    return env_int("TRITON_AGENT_EVAL_TIMEOUT_SECONDS", 300)


def emit_verbose(stderr: TextIO, category: str, message: str) -> None:
    print(f"[{category}] {message}", file=stderr)


def _drain_pipe(pipe: Any, output: list[str]) -> None:
    if pipe is None:
        return
    try:
        try:
            iterator = iter(pipe)
        except TypeError:
            read = getattr(pipe, "read", None)
            chunk = read() if callable(read) else None
            if isinstance(chunk, (bytes, str)) and chunk:
                output.append(_coerce_output_text(chunk))
            return
        for line in iterator:
            output.append(_coerce_output_text(line))
    except ValueError:
        return


def _close_pipe(pipe: Any) -> None:
    close = getattr(pipe, "close", None)
    if callable(close):
        close()


def _buffered_process_result(*args: object, **kwargs: object) -> ResultPayload:
    process, stdout_pipe, stdout_lines, stderr_lines, stderr_thread, stall_timeout_seconds = args
    process = cast(subprocess.Popen[bytes], process)
    stdout_lines = cast(list[str], stdout_lines)
    stderr_lines = cast(list[str], stderr_lines)
    stderr_thread = cast(threading.Thread, stderr_thread)
    stall_timeout_seconds = int(stall_timeout_seconds)
    last_output_at = time.monotonic()
    while True:
        line = stdout_pipe.readline() if stdout_pipe is not None else ""
        if line:
            stdout_lines.append(_coerce_output_text(line))
            last_output_at = time.monotonic()
            continue
        if process.poll() is not None:
            stderr_thread.join(timeout=5)
            return make_result(
                return_code=_resolved_returncode(process.returncode),
                stdout="".join(stdout_lines),
                stderr="".join(stderr_lines),
            )
        if stall_timeout_seconds > 0 and time.monotonic() - last_output_at > stall_timeout_seconds:
            process.terminate()
            stderr_thread.join(timeout=5)
            return make_result(
                return_code=1,
                stdout="".join(stdout_lines),
                stderr="".join(stderr_lines),
                stalled=True,
            )


def run_buffered_process(
    command: list[str],
    workdir: str,
    stall_timeout_seconds: int,
    extra_env: dict[str, str] | None = None,
) -> ResultPayload:
    process = subprocess.Popen(
        command,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        env=_merged_env(extra_env),
    )
    stdout_pipe = process.stdout
    stderr_pipe = process.stderr
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    stderr_thread = threading.Thread(target=_drain_pipe, args=(stderr_pipe, stderr_lines), daemon=True)
    stderr_thread.start()
    try:
        return _buffered_process_result(
            process,
            stdout_pipe,
            stdout_lines,
            stderr_lines,
            stderr_thread,
            stall_timeout_seconds,
        )
    finally:
        _close_pipe(stdout_pipe)
        _close_pipe(stderr_pipe)


def run_streaming_process(
    command: list[str],
    workdir: str,
    stall_timeout_seconds: int,
    stdout: Optional[TextIO] = None,
    extra_env: dict[str, str] | None = None,
) -> ResultPayload:
    if _IS_WINDOWS:
        return _run_streaming_windows(command, workdir, stall_timeout_seconds, stdout, extra_env)
    return _run_streaming_pty(command, workdir, stall_timeout_seconds, stdout, extra_env)


def _run_streaming_windows(
    command: list[str],
    workdir: str,
    stall_timeout_seconds: int,
    stdout: Optional[TextIO] = None,
    extra_env: dict[str, str] | None = None,
) -> ResultPayload:
    process = subprocess.Popen(
        command,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        env=_merged_env(extra_env),
    )
    state = _WindowsStreamState(target=stdout)
    reader_thread = _start_windows_output_reader(process, state)
    _wait_for_windows_stream(process, reader_thread, state, stall_timeout_seconds)
    reader_thread.join()
    rc = process.wait() if not state.stalled else 1
    return make_result(
        return_code=rc,
        stdout="".join(state.output_chunks),
        stderr="",
        stalled=state.stalled,
    )


def _start_windows_output_reader(
    process: subprocess.Popen[bytes],
    state: _WindowsStreamState,
) -> threading.Thread:
    def reader() -> None:
        if process.stdout is None:
            raise RuntimeError("process.stdout is None")
        while chunk := process.stdout.read(4096):
            text = _coerce_output_text(chunk)
            with state.lock:
                state.output_chunks.append(text)
                print(text, file=state.target or sys.stdout, end="", flush=True)
                state.last_output_at = time.monotonic()

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    return thread


def _wait_for_windows_stream(
    process: subprocess.Popen[bytes],
    reader_thread: threading.Thread,
    state: _WindowsStreamState,
    stall_timeout_seconds: int,
) -> None:
    while reader_thread.is_alive() or process.poll() is None:
        reader_thread.join(timeout=0.1)
        with state.lock:
            elapsed = time.monotonic() - state.last_output_at
        if stall_timeout_seconds > 0 and elapsed > stall_timeout_seconds:
            process.terminate()
            state.stalled = True
            return


def _run_streaming_pty(
    command: list[str],
    workdir: str,
    stall_timeout_seconds: int,
    stdout: Optional[TextIO] = None,
    extra_env: dict[str, str] | None = None,
) -> ResultPayload:
    pty_module = pty
    select_module = select
    if pty_module is None or select_module is None:
        raise RuntimeError("PTY streaming is unavailable on this platform")
    master_fd, slave_fd = cast(Any, pty_module).openpty()
    output_chunks: list[str] = []
    process = subprocess.Popen(
        command,
        cwd=workdir,
        stdin=subprocess.DEVNULL,
        stdout=slave_fd,
        stderr=slave_fd,
        text=False,
        close_fds=True,
        env=_merged_env(extra_env),
    )
    os.close(slave_fd)
    start = time.monotonic()

    try:
        while True:
            chunk = _read_pty_chunk(process, master_fd, select_module)
            if chunk is None:
                break
            if chunk:
                text = _coerce_output_text(chunk)
                output_chunks.append(text)
                print(text, file=stdout or sys.stdout, end="")
                start = time.monotonic()
                continue
            if _pty_process_stalled(process, start, stall_timeout_seconds):
                return make_result(return_code=1, stdout="".join(output_chunks), stderr="", stalled=True)
        return make_result(return_code=process.wait(), stdout="".join(output_chunks), stderr="")
    finally:
        os.close(master_fd)


def _read_pty_chunk(
    process: subprocess.Popen[bytes],
    master_fd: int,
    select_module: Any,
) -> bytes | None:
    ready, _, _ = cast(Any, select_module).select([master_fd], [], [], 0.1)
    if not ready:
        return None if process.poll() is not None else b""
    try:
        chunk = os.read(master_fd, 4096)
    except OSError as error:
        if error.errno != errno.EIO:
            raise
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        return None
    if not chunk and process.poll() is not None:
        return None
    return chunk


def _pty_process_stalled(
    process: subprocess.Popen[bytes],
    last_output_at: float,
    stall_timeout_seconds: int,
) -> bool:
    if stall_timeout_seconds <= 0 or time.monotonic() - last_output_at <= stall_timeout_seconds:
        return False
    process.terminate()
    return True


def parse_remote_spec(raw: str) -> RemoteSpec:
    if "@" not in raw:
        raise ValueError(f"Remote target must be in user@host[:port] form: {raw}")
    if ":" not in raw:
        return {"user_host": raw, "port": None}

    user_host, possible_port = raw.rsplit(":", 1)
    if not possible_port.isdigit():
        raise ValueError(f"Remote target port must be numeric: {raw}")
    return {"user_host": user_host, "port": int(possible_port)}


def create_remote_workspace(
    remote: str,
    remote_workdir: str | None,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> tuple[RemoteSpec, str]:
    spec = parse_remote_spec(remote)
    if remote_workdir:
        root = shlex.quote(remote_workdir)
        pattern = shlex.quote(str(PurePosixPath(remote_workdir) / "triton-agent-XXXXXX"))
        remote_command = f"mkdir -p {root} && mktemp -d {pattern}"
    else:
        remote_command = "mktemp -d"
    command = _ssh_command(spec, remote_command)
    _maybe_emit_remote_command(command, verbose, stderr)
    result = run_buffered_process(command, ".", stall_timeout_seconds=_ssh_timeout())
    if not result_succeeded(result):
        raise RuntimeError(result["stderr"] or result["stdout"] or "Failed to create remote workspace.")
    workspace = result["stdout"].strip().splitlines()[-1].strip()
    if not workspace:
        raise RuntimeError("Remote workspace command did not return a path.")
    return spec, workspace


def cleanup_remote_workspace(
    spec: RemoteSpec,
    remote_workspace: str,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> None:
    command = _ssh_command(spec, f"rm -rf {shlex.quote(remote_workspace)}")
    _maybe_emit_remote_command(command, verbose, stderr)
    run_buffered_process(command, ".", stall_timeout_seconds=_ssh_timeout())


def copy_file_to_remote(
    spec: RemoteSpec,
    local_path: Path,
    remote_path: str,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> None:
    local_workdir, local_arg = _scp_local_operand(local_path)
    command = _scp_to_remote_command(spec, local_arg, remote_path)
    _maybe_emit_remote_command(command, verbose, stderr)
    result = run_buffered_process(command, local_workdir, stall_timeout_seconds=_scp_timeout())
    if not result_succeeded(result):
        raise RuntimeError(result["stderr"] or result["stdout"] or f"Failed to copy {local_path} to remote.")


def copy_file_from_remote(
    spec: RemoteSpec,
    remote_path: str,
    local_path: Path,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> None:
    local_workdir, local_arg = _scp_local_operand(local_path)
    command = _scp_from_remote_command(spec, remote_path, local_arg)
    _maybe_emit_remote_command(command, verbose, stderr)
    result = run_buffered_process(command, local_workdir, stall_timeout_seconds=_scp_timeout())
    if not result_succeeded(result):
        raise RuntimeError(result["stderr"] or result["stdout"] or f"Failed to copy {remote_path} from remote.")


def copy_directory_from_remote(
    spec: RemoteSpec,
    remote_path: str,
    local_path: Path,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_workdir, local_arg = _scp_local_operand(local_path)
    command = _scp_from_remote_command(spec, remote_path, local_arg, recursive=True)
    _maybe_emit_remote_command(command, verbose, stderr)
    result = run_buffered_process(command, local_workdir, stall_timeout_seconds=_scp_timeout())
    if not result_succeeded(result):
        raise RuntimeError(
            result["stderr"] or result["stdout"] or f"Failed to copy directory {remote_path} from remote."
        )


def _build_remote_command(request: _RemoteCommandRequest) -> list[str]:
    env_prefix = _shell_env_prefix(request.extra_env)
    command_text = _normalize_remote_command(request.remote_command)
    return _ssh_command(
        request.spec,
        f"cd {shlex.quote(request.remote_workspace)} && {env_prefix + ' ' if env_prefix else ''}{command_text}",
    )


def run_remote_command_streaming(*args: object, **kwargs: object) -> ResultPayload:
    stdout = cast(TextIO | None, args[3] if len(args) > 3 else kwargs.get("stdout"))
    verbose = bool(args[4] if len(args) > 4 else kwargs.get("verbose", False))
    stderr = cast(TextIO | None, args[5] if len(args) > 5 else kwargs.get("stderr"))
    extra_env = cast(dict[str, str] | None, args[6] if len(args) > 6 else kwargs.get("extra_env"))
    stall_timeout = cast(int | None, args[7] if len(args) > 7 else kwargs.get("stall_timeout_seconds"))
    request = _RemoteCommandRequest(
        spec=cast(RemoteSpec, args[0] if args else kwargs["spec"]),
        remote_workspace=str(args[1] if len(args) > 1 else kwargs["remote_workspace"]),
        remote_command=cast(str | Sequence[str], args[2] if len(args) > 2 else kwargs["remote_command"]),
        stdout=stdout, verbose=verbose, stderr=stderr, extra_env=extra_env,
        stall_timeout_seconds=stall_timeout,
    )
    command = _build_remote_command(request)
    _maybe_emit_remote_command(command, request.verbose, request.stderr)
    timeout = (
        request.stall_timeout_seconds
        if request.stall_timeout_seconds is not None
        else eval_stall_timeout_seconds()
    )
    return run_streaming_process(command, ".", stall_timeout_seconds=timeout, stdout=request.stdout)


def run_remote_command_buffered(*args: object, **kwargs: object) -> ResultPayload:
    verbose = bool(args[3] if len(args) > 3 else kwargs.get("verbose", False))
    stderr = cast(TextIO | None, args[4] if len(args) > 4 else kwargs.get("stderr"))
    extra_env = cast(dict[str, str] | None, args[5] if len(args) > 5 else kwargs.get("extra_env"))
    stall_timeout = cast(int | None, args[6] if len(args) > 6 else kwargs.get("stall_timeout_seconds"))
    request = _RemoteCommandRequest(
        spec=cast(RemoteSpec, args[0] if args else kwargs["spec"]),
        remote_workspace=str(args[1] if len(args) > 1 else kwargs["remote_workspace"]),
        remote_command=cast(str | Sequence[str], args[2] if len(args) > 2 else kwargs["remote_command"]),
        verbose=verbose, stderr=stderr, extra_env=extra_env,
        stall_timeout_seconds=stall_timeout,
    )
    command = _build_remote_command(request)
    _maybe_emit_remote_command(command, request.verbose, request.stderr)
    timeout = (
        request.stall_timeout_seconds
        if request.stall_timeout_seconds is not None
        else eval_stall_timeout_seconds()
    )
    return run_buffered_process(command, ".", stall_timeout_seconds=timeout)


def _ssh_command(spec: RemoteSpec, remote_command: str) -> list[str]:
    command = ["ssh"]
    if spec["port"] is not None:
        command.extend(["-p", str(spec["port"])])
    command.extend([spec["user_host"], f"bash -lc {shlex.quote(remote_command)}"])
    return command


def _scp_to_remote_command(spec: RemoteSpec, local_path_arg: str, remote_path: str) -> list[str]:
    command = ["scp"]
    if spec["port"] is not None:
        command.extend(["-P", str(spec["port"])])
    command.extend([local_path_arg, f"{spec['user_host']}:{remote_path}"])
    return command


def _scp_from_remote_command(
    spec: RemoteSpec,
    remote_path: str,
    local_path_arg: str,
    recursive: bool = False,
) -> list[str]:
    command = ["scp"]
    if recursive:
        command.append("-r")
    if spec["port"] is not None:
        command.extend(["-P", str(spec["port"])])
    command.extend([f"{spec['user_host']}:{remote_path}", local_path_arg])
    return command


def _maybe_emit_remote_command(command: list[str], verbose: bool, stderr: TextIO | None) -> None:
    if not verbose or stderr is None:
        return
    emit_verbose(stderr, "remote", f"command: {shlex.join(command)}")


def result_succeeded(result: ResultPayload) -> bool:
    return result["return_code"] == 0 and not result["stalled"]


def _normalize_remote_command(remote_command: str | Sequence[str]) -> str:
    if isinstance(remote_command, str):
        return remote_command
    return shlex.join(str(part) for part in remote_command)


def _resolved_returncode(returncode: int | None) -> int:
    return returncode if returncode is not None else 1


def _normalized_execution_extra_env(extra_env: dict[str, str] | None) -> dict[str, str]:
    normalized = {} if extra_env is None else dict(extra_env)
    blocks_parallel = normalized.get(_BLOCKS_PARALLEL_ENV)
    if blocks_parallel == _BLOCKS_PARALLEL_UNSAFE_VALUE:
        normalized[_BLOCKS_PARALLEL_ENV] = _BLOCKS_PARALLEL_SAFE_VALUE
    elif blocks_parallel is None and os.environ.get(_BLOCKS_PARALLEL_ENV) == _BLOCKS_PARALLEL_SAFE_VALUE:
        normalized[_BLOCKS_PARALLEL_ENV] = _BLOCKS_PARALLEL_SAFE_VALUE
    return normalized


def _merged_env(extra_env: dict[str, str] | None) -> dict[str, str] | None:
    normalized = _normalized_execution_extra_env(extra_env)
    if extra_env is None and not normalized:
        return None
    merged = dict(os.environ)
    merged.update(normalized)
    return merged


def _shell_env_prefix(extra_env: dict[str, str] | None) -> str:
    normalized = _normalized_execution_extra_env(extra_env)
    if not normalized:
        return ""
    return " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(normalized.items()))


def _coerce_output_text(data: bytes | str) -> str:
    # Keep this decoder local to the skill helper instead of importing a shared
    # triton_agent utility: skill-side scripts must stay self-contained.
    if isinstance(data, str):
        return data
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        preferred = locale.getpreferredencoding(False) or "utf-8"
        return data.decode(preferred, errors="replace")


def _scp_local_operand(local_path: Path) -> tuple[str, str]:
    return str(local_path.parent), local_path.name
