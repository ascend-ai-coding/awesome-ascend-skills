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

import atexit
import json
import logging
import os
import shutil
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


class UsageError(Exception):
    """User-facing CLI usage error shared by the collector scripts.

    Raised instead of ``SystemExit`` inside library-style helpers; the CLI
    ``main`` entry points convert it into a logged error and return code 1.
    """


def log_usage_error(exc: UsageError) -> None:
    """Log a user-facing usage error once, without duplicating the ERROR prefix."""
    message = str(exc)
    if message.startswith("ERROR: "):
        message = message[len("ERROR: "):]
    logger.error("%s", message)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


class RunStateGuard:
    """Persist a top-level run state and mark unfinished runs as aborted.

    The guard protects reuse semantics: only a run finalized as ``completed`` is
    eligible to provide cached command state to a future invocation.
    """

    def __init__(self, path: Path, metadata: Dict[str, Any]):
        self.path = path
        self.metadata = dict(metadata)
        self.finalized = False
        self.started_at = now_iso()
        self.write("running")
        atexit.register(self._on_exit)

    def write(self, status: str, reason: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {
            **self.metadata,
            "status": status,
            "started_at": self.started_at,
            "updated_at": now_iso(),
        }
        if reason:
            payload["reason"] = reason
        if extra:
            payload.update(extra)
        _atomic_write_json(self.path, payload)

    def finalize(self, status: str, reason: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        self.write(status, reason, extra)
        self.finalized = True

    def _on_exit(self) -> None:
        if self.finalized:
            return
        try:
            self.write("aborted", "Collector exited before finalizing the run state.")
        except Exception:
            logger.debug("failed to persist aborted run state on exit", exc_info=True)


def _reader_thread(pipe: Any, buffer: bytearray, log_path: Optional[Path]) -> None:
    log_handle = None
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("ab")
        fd = pipe.fileno()
        while True:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            buffer.extend(chunk)
            if log_handle is not None:
                log_handle.write(chunk)
                log_handle.flush()
    finally:
        try:
            pipe.close()
        except Exception:
            logger.debug("reader thread failed to close pipe", exc_info=True)
        if log_handle is not None:
            log_handle.close()


def _fallback_signal(fallback: Any, result: Dict[str, Any], sent_key: str, error_key: str) -> None:
    """Record a direct-process fallback after a process-group signal failed."""
    try:
        fallback()
        result[sent_key] = True
    except Exception as inner:
        result[error_key] = repr(inner)


def _send_term(proc: subprocess.Popen[Any], result: Dict[str, Any]) -> bool:
    """SIGTERM the process group; return True when the process is already gone."""
    try:
        if os.name == "posix":
            pgid = os.getpgid(proc.pid)
            result["process_group"] = pgid
            os.killpg(pgid, signal.SIGTERM)
        else:
            proc.terminate()
        result["term_sent"] = True
    except ProcessLookupError:
        return True
    except Exception as exc:
        result["term_error"] = repr(exc)
        _fallback_signal(proc.terminate, result, "term_sent", "fallback_term_error")
    return False


def _send_kill(proc: subprocess.Popen[Any], result: Dict[str, Any]) -> None:
    """SIGKILL the process group after the grace period expired."""
    try:
        if os.name == "posix":
            pgid = os.getpgid(proc.pid)
            result["process_group"] = pgid
            os.killpg(pgid, signal.SIGKILL)
        else:
            proc.kill()
        result["kill_sent"] = True
    except ProcessLookupError:
        logger.debug("process group already exited before SIGKILL")
    except Exception as exc:
        result["kill_error"] = repr(exc)
        _fallback_signal(proc.kill, result, "kill_sent", "fallback_kill_error")


def _terminate_process_group(proc: subprocess.Popen[Any], grace_seconds: float = 3.0) -> Dict[str, Any]:
    result: Dict[str, Any] = {"term_sent": False, "kill_sent": False, "process_group": None}
    if proc.poll() is not None:
        return result
    if _send_term(proc, result):
        return result
    deadline = time.monotonic() + max(0.0, grace_seconds)
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if proc.poll() is None:
        _send_kill(proc, result)
    return result


def _emit_heartbeat(
    progress_log: Optional[Path],
    label: str,
    proc: subprocess.Popen[Any],
    elapsed: float,
    timeout: int,
) -> None:
    message = (f"[{now_iso()}] {label} still running: pid={proc.pid}, elapsed={elapsed:.1f}s, "
        f"timeout={timeout}s")
    if progress_log is not None:
        progress_log.parent.mkdir(parents=True, exist_ok=True)
        with progress_log.open("a", encoding="utf-8") as f:
            f.write(message + "\n")
    logger.info("%s", message)


class _MonitorSpec(NamedTuple):
    """Bundled inputs for ``_monitor_process`` (G.FNM.03)."""

    proc: subprocess.Popen[Any]
    timeout: int
    heartbeat_seconds: int
    progress_log: Optional[Path]
    label: str
    started: float


def _monitor_process(spec: _MonitorSpec) -> Tuple[bool, int, Dict[str, Any]]:
    """Poll the process until exit or timeout, emitting heartbeats meanwhile."""
    timed_out = False
    heartbeat_count = 0
    termination: Dict[str, Any] = {}
    next_heartbeat = time.monotonic() + spec.heartbeat_seconds if spec.heartbeat_seconds > 0 else float("inf")
    try:
        while spec.proc.poll() is None:
            now = time.monotonic()
            elapsed = now - spec.started
            if elapsed >= spec.timeout:
                timed_out = True
                termination = _terminate_process_group(spec.proc)
                break
            if now >= next_heartbeat:
                heartbeat_count += 1
                _emit_heartbeat(spec.progress_log, spec.label, spec.proc, elapsed, spec.timeout)
                next_heartbeat = now + spec.heartbeat_seconds
            time.sleep(0.1)
    except KeyboardInterrupt:
        termination = _terminate_process_group(spec.proc)
        raise
    return timed_out, heartbeat_count, termination


def _reap_process(proc: subprocess.Popen[Any], termination: Dict[str, Any]) -> None:
    """Wait for process exit, escalating to a forced group kill when stuck."""
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        more = _terminate_process_group(proc, grace_seconds=0.5)
        termination.update({k: v for k, v in more.items() if v})
        try:
            proc.wait(timeout=2)
        except Exception:
            logger.debug("process still not reaped after forced termination", exc_info=True)


def _append_timeout_notice(stderr: str, stderr_log: Optional[Path], timeout: int) -> str:
    stderr = stderr + ("\n" if stderr and not stderr.endswith("\n") else "") + (f"TIMEOUT after "
        f"{timeout}s; process group terminated.")
    if stderr_log is not None:
        with stderr_log.open("a", encoding="utf-8") as f:
            f.write(("\n" if stderr_log.stat().st_size else "") + (f"TIMEOUT after {timeout}s; process "
                f"group terminated.\n"))
    return stderr


def _spawn_process(argv: List[str], cwd: Optional[Path], env: Optional[Dict[str, str]]) -> subprocess.Popen[Any]:
    proc = subprocess.Popen(
        argv,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=(os.name == "posix"),
        env=env,
    )
    if proc.stdout is None or proc.stderr is None:
        raise RuntimeError("subprocess stdout/stderr pipes were not created")
    return proc


class _RunState(NamedTuple):
    """Bundled state of one managed process execution (G.FNM.03)."""

    proc: subprocess.Popen[Any]
    argv: List[str]
    started: float
    started_wall: str
    cwd: Optional[Path]
    timeout: int
    stdout_buffer: bytearray
    stderr_buffer: bytearray
    stderr_log: Optional[Path]


def _managed_result(
    run: _RunState,
    outcome: Tuple[bool, int, Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble the public result payload from the finished process state."""
    timed_out, heartbeat_count, termination = outcome
    elapsed = round(time.monotonic() - run.started, 3)
    stdout = bytes(run.stdout_buffer).decode("utf-8", errors="replace")
    stderr = bytes(run.stderr_buffer).decode("utf-8", errors="replace")
    return_code = 124 if timed_out else int(run.proc.returncode if run.proc.returncode is not None else 1)
    if timed_out:
        stderr = _append_timeout_notice(stderr, run.stderr_log, run.timeout)
    return {
        "argv": run.argv,
        "return_code": return_code,
        "stdout": stdout,
        "stderr": stderr,
        "elapsed_seconds": elapsed,
        "started_at": run.started_wall,
        "finished_at": now_iso(),
        "pid": run.proc.pid,
        "timed_out": timed_out,
        "termination": termination,
        "heartbeat_count": heartbeat_count,
        "cwd": str(run.cwd) if run.cwd else None,
    }


class ManagedProcessSpec(NamedTuple):
    """Bundled inputs for ``run_managed_process`` (G.FNM.03)."""

    command: Sequence[str]
    timeout: int
    cwd: Optional[Path] = None
    heartbeat_seconds: int = 0
    stdout_log: Optional[Path] = None
    stderr_log: Optional[Path] = None
    progress_log: Optional[Path] = None
    heartbeat_label: Optional[str] = None
    env: Optional[Dict[str, str]] = None


def run_managed_process(spec: ManagedProcessSpec) -> Dict[str, Any]:
    """Run one command with process-group cleanup, live log streaming, and heartbeat.

    On timeout the entire process group is terminated, not only the direct
    ``msprof`` process. This prevents orphaned application processes from
    remaining queued on a device lock.
    """

    argv = [str(x) for x in spec.command]
    started_wall = now_iso()
    started = time.monotonic()
    stdout_buffer = bytearray()
    stderr_buffer = bytearray()
    for stale_log in (spec.stdout_log, spec.stderr_log, spec.progress_log):
        if stale_log and stale_log.exists():
            stale_log.unlink()

    proc = _spawn_process(argv, spec.cwd, spec.env)
    out_thread = threading.Thread(target=_reader_thread, args=(proc.stdout, stdout_buffer, spec.stdout_log),
        daemon=True)
    err_thread = threading.Thread(target=_reader_thread, args=(proc.stderr, stderr_buffer, spec.stderr_log),
        daemon=True)
    out_thread.start()
    err_thread.start()

    label = spec.heartbeat_label or Path(argv[0]).name
    outcome: Tuple[bool, int, Dict[str, Any]] = (False, 0, {})
    monitor = _MonitorSpec(proc, spec.timeout, spec.heartbeat_seconds, spec.progress_log, label, started)
    try:
        outcome = _monitor_process(monitor)
    finally:
        _reap_process(proc, outcome[2])
        out_thread.join(timeout=3)
        err_thread.join(timeout=3)

    run = _RunState(
        proc=proc, argv=argv, started=started, started_wall=started_wall, cwd=spec.cwd,
        timeout=spec.timeout, stdout_buffer=stdout_buffer, stderr_buffer=stderr_buffer,
        stderr_log=spec.stderr_log,
    )
    return _managed_result(run, outcome)


def _read_text(path: Path, limit: int = 2_000_000) -> Tuple[str, Optional[str]]:
    try:
        data = path.read_bytes()[:limit]
        return data.decode("utf-8", errors="replace"), None
    except Exception as exc:
        logger.debug("failed to read %s", path, exc_info=True)
        return "", repr(exc)


def _parse_proc_status(status: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for line in status.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key in {"Name", "State", "PPid", "Uid", "Gid", "Threads", "VmRSS"}:
            fields[key] = value.strip()
    return fields


def _process_info(pid: int) -> Dict[str, Any]:
    root = Path("/proc") / str(pid)
    info: Dict[str, Any] = {"pid": pid}
    status, status_error = _read_text(root / "status", 200_000)
    if status_error:
        info["status_error"] = status_error
    else:
        info["status"] = _parse_proc_status(status)
    cmdline, cmdline_error = _read_text(root / "cmdline", 200_000)
    if cmdline_error:
        info["cmdline_error"] = cmdline_error
    else:
        info["cmdline"] = cmdline.replace("\x00", " ").strip()
    wchan, wchan_error = _read_text(root / "wchan", 4096)
    if wchan_error:
        info["wchan_error"] = wchan_error
    else:
        info["wchan"] = wchan.strip()
    try:
        info["cwd"] = os.readlink(root / "cwd")
    except Exception as exc:
        info["cwd_error"] = repr(exc)
    return info


def _lock_identity(path: Path) -> Dict[str, Any]:
    st = path.stat()
    return {
        "path": str(path),
        "inode": st.st_ino,
        "dev_major": os.major(st.st_dev),
        "dev_minor": os.minor(st.st_dev),
    }


def _parse_proc_lock_line(line: str) -> Optional[Dict[str, Any]]:
    raw = line.strip()
    waiting = "->" in raw
    clean = raw.replace("->", " ")
    parts = clean.split()
    if len(parts) < 8:
        return None
    # Typical: id: POSIX ADVISORY WRITE pid major:minor:inode start end
    try:
        pid = int(parts[4])
        dev_inode = parts[5]
        major_s, minor_s, inode_s = dev_inode.split(":", 2)
        return {
            "raw": raw,
            "waiting": waiting,
            "lock_type": parts[1],
            "mode": parts[3],
            "pid": pid,
            "dev_major": int(major_s, 16),
            "dev_minor": int(minor_s, 16),
            "inode": int(inode_s),
        }
    except Exception:
        return {"raw": raw, "waiting": waiting}


def _ascend_install_roots() -> List[Path]:
    """Candidate Ascend install roots: env-derived first, hardcoded fallback.

    ASCEND_HOME_PATH/ASCEND_TOOLKIT_HOME usually point at the toolkit tree
    (e.g. /usr/local/Ascend/ascend-toolkit/latest), so a few ancestors are
    included to reach the sibling driver tree.
    """
    roots: List[Path] = []
    for var in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        raw = os.environ.get(var)
        if not raw:
            continue
        base = Path(raw).expanduser()
        for candidate in [base, *base.parents[:3]]:
            if candidate.is_dir() and candidate not in roots:
                roots.append(candidate)
    if not roots:
        roots.append(Path("/usr/local/Ascend"))
    return roots


def _candidate_lock_identities() -> List[Dict[str, Any]]:
    patterns = [
        "driver/lib64/driver/sink_file_mutex_*.cfg",
        "driver/lib64/driver/sink_file_mutex_*",
        "driver/lib64/common/sink_file_mutex_*",
    ]
    candidates: List[Path] = []
    for root in _ascend_install_roots():
        for pattern in patterns:
            candidates.extend(root.glob(pattern))
    identities: List[Dict[str, Any]] = []
    for path in sorted({p.resolve() for p in candidates if p.is_file()}):
        try:
            identities.append(_lock_identity(path))
        except Exception as exc:
            logger.debug("failed to inspect lock candidate %s", path, exc_info=True)
            identities.append({"path": str(path), "error": repr(exc)})
    return identities


def _lock_matches(lock: Mapping[str, Any], identity: Mapping[str, Any]) -> bool:
    return (
        lock.get("inode") == identity.get("inode")
        and lock.get("dev_major") == identity.get("dev_major")
        and lock.get("dev_minor") == identity.get("dev_minor")
    )


def _match_identity_locks(
    identity: Dict[str, Any],
    parsed: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], set]:
    matched: List[Dict[str, Any]] = []
    pids = set()
    for lock in parsed:
        if not _lock_matches(lock, identity):
            continue
        matched.append({**lock, "path": identity["path"]})
        if isinstance(lock.get("pid"), int):
            pids.add(int(lock["pid"]))
    return matched, pids


def _match_proc_locks(
    identities: List[Dict[str, Any]],
    parsed: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], set]:
    matched: List[Dict[str, Any]] = []
    pids = set()
    for identity in identities:
        if "inode" not in identity:
            continue
        identity_matched, identity_pids = _match_identity_locks(identity, parsed)
        matched.extend(identity_matched)
        pids.update(identity_pids)
    return matched, pids


def inspect_ascend_file_locks() -> Dict[str, Any]:
    identities = _candidate_lock_identities()
    proc_locks, error = _read_text(Path("/proc/locks"), 8_000_000)
    parsed = [x for x in (_parse_proc_lock_line(line) for line in proc_locks.splitlines()) if x]
    matched, pids = _match_proc_locks(identities, parsed)
    return {
        "candidate_lock_files": identities,
        "matched_locks": matched,
        "processes": [_process_info(pid) for pid in sorted(pids)],
        "proc_locks_error": error,
        "proc_locks_line_count": len(proc_locks.splitlines()),
    }


def collect_device_diagnostics(
    diagnostics_dir: Path,
    *,
    reason: str,
    command_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    lock_text, lock_error = _read_text(Path("/proc/locks"), 8_000_000)
    (diagnostics_dir / "proc_locks.txt").write_text(lock_text, encoding="utf-8")
    npu_smi_path = shutil.which("npu-smi")
    npu_result: Dict[str, Any]
    if npu_smi_path:
        npu_result = run_managed_process(ManagedProcessSpec(
            [npu_smi_path, "info"],
            timeout=15,
            heartbeat_seconds=0,
            stdout_log=diagnostics_dir / "npu_smi.stdout.log",
            stderr_log=diagnostics_dir / "npu_smi.stderr.log",
        ))
    else:
        npu_result = {"return_code": None, "reason": "npu-smi not found in PATH"}
    ps_result = run_managed_process(ManagedProcessSpec(
        ["ps", "-eo", "pid,ppid,pgid,stat,etimes,wchan:32,cmd"],
        timeout=10,
        heartbeat_seconds=0,
        stdout_log=diagnostics_dir / "processes.stdout.log",
        stderr_log=diagnostics_dir / "processes.stderr.log",
    ))
    payload = {
        "created_at": now_iso(),
        "reason": reason,
        "command_result": command_result,
        "ascend_file_locks": inspect_ascend_file_locks(),
        "proc_locks_error": lock_error,
        "npu_smi": {k: v for k, v in npu_result.items() if k not in {"stdout", "stderr"}},
        "process_snapshot": {k: v for k, v in ps_result.items() if k not in {"stdout", "stderr"}},
        "files": {
            "proc_locks": "proc_locks.txt",
            "npu_smi_stdout": "npu_smi.stdout.log",
            "npu_smi_stderr": "npu_smi.stderr.log",
            "processes_stdout": "processes.stdout.log",
            "processes_stderr": "processes.stderr.log",
        },
    }
    _atomic_write_json(diagnostics_dir / "device_diagnostics.json", payload)
    return payload
