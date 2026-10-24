#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""run_kernel_test.py — standardized wrapper for device-side kernel tests.

Python rewrite of run_kernel_test.sh, structured so it can grow into a test
suite runner (precision compare, perf baseline, ... as future test types).

Wraps an arbitrary test command with a two-tier timeout plus a liveness
probe that distinguishes a genuinely slow compile (host compiler busy /
compile-cache artifacts growing) from a pathological JIT stall (no host
activity at all). The probe is host-side only; it never touches the NPU.

Usage:
    python run_kernel_test.py <log_prefix> [--json] -- <test command...>
    python run_kernel_test.py --self-test

Environment (all optional, same names as the old .sh):
    KERNEL_TEST_TIMEOUT     initial timeout in seconds        (default 300)
    KERNEL_TEST_MAX_EXTEND  one-time extension after COMPILE_SLOW (default 600)
    KERNEL_TEST_PROBE_SEC   liveness probe window in seconds  (default 30)

Conclusion line (always the LAST stdout line, machine-greppable):
    RESULT: PASS | FAIL | PASS_SLOW_COMPILE | JIT_STALL_SUSPECTED ...
With --json, a machine-readable JSON object (verdict, elapsed, probe
samples, timeout tiers) is printed right before the RESULT line.

Exit codes: test's own exit code on completion; 124 on stall/kill; 2 on
usage error.

Extension points for future test types (precision compare, perf baseline):
  - ``Verdict`` is the single enumeration of all conclusion values;
  - ``ProbeResult`` / ``TestResult`` are structured dataclasses, ready to
    be serialized or extended with per-test-type fields;
  - ``run_test()`` returns a ``TestResult`` instead of only printing, so a
    suite driver can collect results across many cases.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# Named logger bound to stdout (never the root logger — basicConfig would
# raise the root level and unleash third-party INFO noise into the output).
# The emitted text is the machine-readable protocol agents consume (RESULT
# stays the LAST stdout line), byte-identical to the previous print output.
_log = logging.getLogger("run_kernel_test")
_log.setLevel(logging.INFO)
_log.propagate = False
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
_log.addHandler(_handler)

# --------------------------------------------------------------------------
# Verdicts (single source of truth for conclusion values; extend here when
# adding new test types such as precision compare or perf baseline).
# --------------------------------------------------------------------------


class Verdict(str, enum.Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    PASS_SLOW_COMPILE = "PASS_SLOW_COMPILE"
    COMPILE_SLOW = "COMPILE_SLOW"
    JIT_STALL_SUSPECTED = "JIT_STALL_SUSPECTED"
    # Reserved for future test types:
    PRECISION_FAIL = "PRECISION_FAIL"
    EXECUTION_HANG_SUSPECTED = "EXECUTION_HANG_SUSPECTED"


STALL_EXIT_CODE = 124  # same convention as timeout(1) / the old .sh

# --------------------------------------------------------------------------
# Configuration & structured results
# --------------------------------------------------------------------------


@dataclasses.dataclass
class TimeoutConfig:
    """Two-tier timeout + probe window, in seconds."""

    initial: float = 300.0
    extend: float = 600.0
    probe: float = 30.0

    @classmethod
    def from_env(cls) -> "TimeoutConfig":
        def _get(name: str, default: float) -> float:
            raw = os.environ.get(name)
            if raw is None:
                return default
            try:
                return float(raw)
            except ValueError:
                _log.warning(
                    f"[run_kernel_test] WARNING: ${name}={raw!r} "
                    f"not a number; using default {default}"
                )
                return default

        return cls(
            initial=_get("KERNEL_TEST_TIMEOUT", 300.0),
            extend=_get("KERNEL_TEST_MAX_EXTEND", 600.0),
            probe=_get("KERNEL_TEST_PROBE_SEC", 30.0),
        )


@dataclasses.dataclass
class ProbeResult:
    """One liveness-probe sample taken after the initial timeout fired."""

    window_sec: float
    cpu_jiffies_delta: int
    cache_new_artifacts: bool
    cache_dirs: list
    finished_during_probe: bool = False

    @property
    def alive(self) -> bool:
        """Host side shows activity -> genuine slow compile, not a stall."""
        return self.cpu_jiffies_delta > 0 or self.cache_new_artifacts


@dataclasses.dataclass
class TestResult:
    verdict: Verdict
    exit_code: int
    elapsed_sec: float
    log_file: str
    command: list
    timeouts: TimeoutConfig
    probe: ProbeResult | None = None
    note: str = ""

    def result_line(self) -> str:
        parts = [f"RESULT: {self.verdict.value}"]
        if self.exit_code:
            parts.append(f"exit={self.exit_code}")
        else:
            parts.append("exit=0")
        parts.append(f"elapsed={self.elapsed_sec:.0f}s")
        parts.append(f"log={self.log_file}")
        if self.note:
            parts.append(f"({self.note})")
        return " ".join(parts)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "exit_code": self.exit_code,
            "elapsed_sec": round(self.elapsed_sec, 2),
            "log_file": self.log_file,
            "command": self.command,
            "timeouts": {
                "initial": self.timeouts.initial,
                "extend": self.timeouts.extend,
                "probe": self.timeouts.probe,
                "total_cap": self.timeouts.initial + self.timeouts.extend,
            },
            "probe": dataclasses.asdict(self.probe) if self.probe else None,
            "note": self.note,
        }


# --------------------------------------------------------------------------
# Process control (spawn as session leader, timed wait, group kill)
# --------------------------------------------------------------------------


class MonitoredProcess:
    """Child process running in its own session/group (setsid equivalent).

    pgid == pid, so a single killpg reaches the whole compiler process tree
    and no orphan compiler processes are left behind.
    """

    def __init__(self, argv: list, log_file: Path):
        self.argv = argv
        self.log_file = log_file
        self._log_fh = open(log_file, "w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(
            argv,
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # setsid: child is session/group leader
        )
        self.pid = self.proc.pid
        self.pgid = self.pid

    @property
    def running(self) -> bool:
        return self.proc.poll() is None

    @property
    def returncode(self) -> int | None:
        return self.proc.poll()

    def wait_up_to(self, seconds: float) -> bool:
        """Wait up to `seconds`. True if the child exited in time."""
        deadline = time.monotonic() + seconds
        while self.running:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.2)
        return True

    def kill_tree(self) -> None:
        """TERM the whole group, escalate to KILL after a short grace."""
        if not self.running:
            return
        try:
            os.killpg(self.pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + 2.0
        while self.running and time.monotonic() < deadline:
            time.sleep(0.1)
        if self.running:
            try:
                os.killpg(self.pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

    def close(self) -> None:
        self._log_fh.close()


# --------------------------------------------------------------------------
# Liveness probe primitives (host-side only, never touches the NPU)
# --------------------------------------------------------------------------


def _read_proc_stat(pid: str) -> tuple | None:
    """Parse /proc/<pid>/stat -> (state, pgid, utime+stime jiffies)."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # comm may contain spaces/parens; everything after the last ')' is fields
    # 3..: state ppid pgrp session ... utime(14) stime(15)
    fields_start = text.rfind(")") + 2
    rest = text[fields_start:].split()
    if len(rest) < 13:
        return None
    state = rest[0]
    try:
        pgid = int(rest[2])
        jiffies = int(rest[11]) + int(rest[12])
    except ValueError:
        return None
    return state, pgid, jiffies


def group_cpu_jiffies(pgid: int) -> int:
    """Sum of utime+stime jiffies across every process in the group.

    Each /proc entry is read individually: a single vanished process must
    not abort the whole scan (that produced bogus deltas in the bash
    version's awk glob).
    """
    total = 0
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        stat = _read_proc_stat(entry)
        if stat is None:
            continue
        _, proc_pgid, jiffies = stat
        if proc_pgid == pgid:
            total += jiffies
    return total


def cache_dirs() -> list:
    """Existing compile-cache dirs on the host; empty means CPU-only probing."""
    home = Path.home()
    candidates = [
        home / ".ascend",
        home / ".cache" / "ascend",
        home / ".cache" / "cann",
    ]
    extra = os.environ.get("ASCEND_CACHE_PATH")
    if extra:
        candidates.append(Path(extra))
    return [str(d) for d in candidates if d.is_dir()]


def _is_newer_than(path: str, since_epoch: float) -> bool:
    """mtime-based comparison; unreadable entries count as not new."""
    try:
        return os.path.getmtime(path) >= since_epoch
    except OSError:
        return False


def _dir_has_new_artifacts(since_epoch: float, directory: str) -> bool:
    """True if any file under one cache dir is newer than `since_epoch`."""
    for root, _subdirs, files in os.walk(directory):
        if any(_is_newer_than(os.path.join(root, name), since_epoch) for name in files):
            return True
    return False


def cache_has_new_artifacts(since_epoch: float, dirs: list) -> bool:
    """True if any file under the cache dirs is newer than `since_epoch`."""
    return any(_dir_has_new_artifacts(since_epoch, d) for d in dirs)


def run_probe(child: MonitoredProcess, window: float) -> ProbeResult:
    """Watch the child for `window` seconds and sample liveness signals."""
    probe_start = time.time()
    cpu_before = group_cpu_jiffies(child.pgid)
    dirs = cache_dirs()
    if not dirs:
        _log.info(
            "[run_kernel_test] no compile-cache dir found; CPU-activity criterion only"
        )
    else:
        _log.info(f"[run_kernel_test] watching compile-cache dirs: {','.join(dirs)}")

    # The process may legitimately finish during the probe window.
    deadline = time.monotonic() + window
    while child.running and time.monotonic() < deadline:
        time.sleep(0.5)

    finished = not child.running
    cpu_after = group_cpu_jiffies(child.pgid)
    result = ProbeResult(
        window_sec=window,
        cpu_jiffies_delta=cpu_after - cpu_before,
        cache_new_artifacts=cache_has_new_artifacts(probe_start, dirs),
        cache_dirs=dirs,
        finished_during_probe=finished,
    )
    _log.info(
        f"[run_kernel_test] probe: cpu_jiffies_delta={result.cpu_jiffies_delta} "
        f"cache_new_artifacts={int(result.cache_new_artifacts)}"
    )
    return result


# --------------------------------------------------------------------------
# Verdict decision (conclusion judgement, separated from probing/control)
# --------------------------------------------------------------------------


@dataclasses.dataclass
class FinishedRun:
    """A child process that exited on its own (before or during the probe)."""

    returncode: int
    elapsed: float
    log_file: str
    command: list


def decide_exit_verdict(
    run: FinishedRun, cfg: TimeoutConfig, probe: ProbeResult | None = None
) -> TestResult:
    """Verdict for a test that exited on its own."""
    if run.returncode == 0:
        verdict = Verdict.PASS if probe is None else Verdict.PASS_SLOW_COMPILE
        note = ""
        if probe is not None:
            note = (
                "finished during probe"
                if probe.finished_during_probe
                else "slow_compile risk: check unroll/codegen bloat"
            )
        return TestResult(
            verdict, 0, run.elapsed, run.log_file, run.command, cfg, probe, note
        )
    return TestResult(
        Verdict.FAIL, run.returncode, run.elapsed, run.log_file, run.command, cfg, probe
    )


# --------------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------------


def run_test(command: list, log_prefix: str, cfg: TimeoutConfig) -> TestResult:
    """Run `command` under the two-tier timeout + liveness probe.

    Returns a structured TestResult; callers (suite drivers) may collect it
    or just read the printed RESULT line.
    """
    log_file = Path(f"{log_prefix}.log")
    _log.info(f"[run_kernel_test] cmd: {' '.join(command)}")
    _log.info(
        f"[run_kernel_test] log: {log_file}  timeout={cfg.initial:.0f}s "
        f"probe={cfg.probe:.0f}s extend={cfg.extend:.0f}s"
    )

    start = time.time()
    child = MonitoredProcess(command, log_file)

    def finished_run() -> FinishedRun:
        return FinishedRun(
            child.returncode, time.time() - start, str(log_file), command
        )

    try:
        if child.wait_up_to(cfg.initial):
            return decide_exit_verdict(finished_run(), cfg)

        # ---- initial timeout hit: enter the liveness probe ----
        _log.info(
            f"[run_kernel_test] initial timeout {cfg.initial:.0f}s hit; "
            f"probing liveness for {cfg.probe:.0f}s ..."
        )
        probe = run_probe(child, cfg.probe)

        if probe.finished_during_probe:
            return decide_exit_verdict(finished_run(), cfg, probe)

        if probe.alive:
            # Host side is alive -> genuine slow compile; extend ONCE.
            _log.info("COMPILE_SLOW")
            _log.info(
                f"[run_kernel_test] liveness detected -> COMPILE_SLOW; "
                f"extending once by {cfg.extend:.0f}s (one-shot)"
            )
            if child.wait_up_to(cfg.extend):
                return decide_exit_verdict(finished_run(), cfg, probe)
            _log.info("[run_kernel_test] still stuck after one extension -> stall")
        else:
            _log.info("[run_kernel_test] no host-side activity -> stall")

        child.kill_tree()
        return TestResult(
            Verdict.JIT_STALL_SUSPECTED,
            STALL_EXIT_CODE,
            time.time() - start,
            str(log_file),
            command,
            cfg,
            probe,
            note=(
                f"host cpu_delta={probe.cpu_jiffies_delta} "
                f"cache_new={int(probe.cache_new_artifacts)}; "
                "report failure_category=jit_stall"
            ),
        )
    finally:
        if child.running:
            child.kill_tree()
        child.close()


def emit_result(result: TestResult, as_json: bool) -> None:
    """Print the optional JSON blob, then the RESULT line (always last)."""
    if as_json:
        _log.info(json.dumps(result.to_dict(), indent=2))
    _log.info(result.result_line())


# --------------------------------------------------------------------------
# Self-test (host-side only; fake commands, never touches a device)
# --------------------------------------------------------------------------


def _self_test_scenarios(fake_cache: Path) -> list:
    """Scenario tuples: name, argv, env additions, expected verdict, exit code."""
    py = sys.executable
    return [
        ("normal_pass", [py, "-c", "pass"], {}, Verdict.PASS, 0),
        ("precision_fail", [py, "-c", "import sys; sys.exit(1)"], {}, Verdict.FAIL, 1),
        (
            "stall_killed",
            [py, "-c", "import time; time.sleep(60)"],
            {},
            Verdict.JIT_STALL_SUSPECTED,
            STALL_EXIT_CODE,
        ),
        # Busy CPU through the probe window -> COMPILE_SLOW -> finishes
        # inside the one-shot extension -> PASS_SLOW_COMPILE.
        (
            "slow_compile_extend",
            [
                py,
                "-c",
                "import time\n"
                "t0 = time.time()\n"
                "while time.time() - t0 < 4.5:\n"
                "    pass",
            ],
            {},
            Verdict.PASS_SLOW_COMPILE,
            0,
        ),
        # No CPU burn, but a cache artifact appears during the probe window
        # -> cache criterion fallback counts as liveness -> COMPILE_SLOW ->
        # finishes inside the extension -> PASS_SLOW_COMPILE.
        (
            "cache_fallback",
            [
                py,
                "-c",
                "import pathlib, sys, time\n"
                "time.sleep(2.5)\n"
                "pathlib.Path(sys.argv[1], 'kernel.o').write_text('x')\n"
                "time.sleep(3.0)",
                str(fake_cache),
            ],
            {"ASCEND_CACHE_PATH": str(fake_cache)},
            Verdict.PASS_SLOW_COMPILE,
            0,
        ),
    ]


def _restore_env(saved: dict) -> None:
    """Restore env vars captured by ``{key: os.environ.get(key)}``."""
    for key, old in saved.items():
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


def _run_scenario(
    argv: list, env_add: dict, log_prefix: str, cfg: TimeoutConfig
) -> TestResult:
    """Run one scenario with the env additions temporarily applied."""
    saved = {key: os.environ.get(key) for key in env_add}
    os.environ.update(env_add)
    try:
        return run_test(argv, log_prefix, cfg)
    finally:
        _restore_env(saved)


def _check_scenario(scenario: tuple, work: Path, cfg: TimeoutConfig) -> bool:
    """Run one scenario and check verdict + exit code. Prints a SELFTEST line."""
    name, argv, env_add, want_verdict, want_exit = scenario
    result = _run_scenario(argv, env_add, str(work / name), cfg)
    emit_result(result, as_json=False)
    ok = result.verdict == want_verdict and result.exit_code == want_exit
    _log.error(
        f"SELFTEST {'PASS' if ok else 'FAIL'} {name}: "
        f"verdict={result.verdict.value} "
        f"(want {want_verdict.value}) exit={result.exit_code} "
        f"(want {want_exit})"
    )
    # Structured result must round-trip through JSON.
    json.dumps(result.to_dict())
    return ok


def _self_test() -> int:
    """Cover the 5 conclusion paths with tiny timeouts and fake commands.

    Scenarios: normal PASS / stall kill / slow-compile extension /
    cache-criterion fallback / precision (non-zero exit) FAIL.
    """
    work = Path.cwd() / "_debug" / "run_kernel_test_selftest"
    shutil.rmtree(work, ignore_errors=True)
    fake_home = work / "home"
    fake_cache = work / "ascend_cache"
    fake_home.mkdir(parents=True)
    fake_cache.mkdir(parents=True)

    # Isolate cache discovery from the real ~/.ascend etc. so the verdicts
    # are deterministic regardless of other activity on this host.
    saved_env = {k: os.environ.get(k) for k in ("HOME", "ASCEND_CACHE_PATH")}
    os.environ["HOME"] = str(fake_home)

    cfg = TimeoutConfig(initial=2.0, extend=4.0, probe=2.0)
    scenarios = _self_test_scenarios(fake_cache)
    bad = 0
    try:
        for scenario in scenarios:
            bad += 0 if _check_scenario(scenario, work, cfg) else 1
    finally:
        _restore_env(saved_env)
        shutil.rmtree(work, ignore_errors=True)
        try:
            work.parent.rmdir()  # remove _debug too, but only if empty
        except OSError:
            pass

    _log.error(
        f"SELFTEST {'PASS' if not bad else 'FAIL'}: "
        f"{len(scenarios) - bad}/{len(scenarios)} scenarios passed"
    )
    return 1 if bad else 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> int:
    # Split argv at '--' ourselves: argparse.REMAINDER would swallow our own
    # options (e.g. --json) once the positional is consumed.
    argv = sys.argv[1:]
    command: list = []
    if "--" in argv:
        sep = argv.index("--")
        cmd_start = sep + 1
        argv, command = argv[:sep], argv[cmd_start:]

    ap = argparse.ArgumentParser(
        description="Two-tier timeout + liveness-probe wrapper for device-side "
        "kernel tests (see module docstring)."
    )
    ap.add_argument(
        "log_prefix", nargs="?", help="prefix for the <prefix>.log output file"
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="print a machine-readable JSON result before the RESULT line",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="run the host-side self-test (fake commands, no device)",
    )
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()

    if not args.log_prefix or not command:
        ap.error("usage: run_kernel_test.py <log_prefix> [--json] -- <test command...>")

    cfg = TimeoutConfig.from_env()
    result = run_test(command, args.log_prefix, cfg)
    emit_result(result, as_json=args.json)
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
