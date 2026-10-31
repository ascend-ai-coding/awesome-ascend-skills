# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""RealEvaluator — wire the FROZEN PANKO E(x) as the BO objective on the NPU.

    objective(cfg) -> (s, p)
        s = test_command exit 0  (golden_compare AND layout_check, per evaluator.md)
            AND  lint exit 0     (the Stage-7 implementation gates, if lint_root given)
        p = scripts/measure_latency.measure(...) (frozen warm latency)  when s==1, else None

Same referee as the harness, so a config BO returns is drop-in for cmd_record / cmd_close and
its J is comparable to every other candidate. Correctness stays a hard gate: s==0 => (0, None) =>
the driver scores it with the infeasible penalty and it can never be best.

Trial independence
------------------
BO explores configs; it does NOT accumulate them. So each trial applies cfg on top of the BASE
impl captured at construction (the accumulated global best at the moment BO is invoked), evaluates,
then RESTORES the base — trials never contaminate each other. After the search, call finalize(best)
once to write the winning config so the harness snapshots it via cmd_record.

Testability
-----------
`run_cmd` (shell -> returncode) and `measure_fn` (op_file -> {"p",...}) are injectable. Production
leaves them None: run_cmd = subprocess, measure_fn = the frozen scripts/measure_latency.py. Tests
pass fakes so the whole (s, p) contract, correctness gating, and base-restore run without an NPU.
"""

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time

from . import apply


class UnsupportedCommand(Exception):
    """The correctness command needs a shell feature this runner does not implement."""


# Everything only a shell can do. `&&` is implemented here so it is not listed;
# a lone `&` is rejected per segment, because backgrounding a correctness command
# would make its exit code -- which is `s` -- meaningless.
_SHELL_ONLY = ("|", ";", "<", ">", "`", "$(", "${", "*", "?", "\n")
_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _chdir(cwd, argv):
    """The working directory after one `cd` segment."""
    if len(argv) != 2:
        raise UnsupportedCommand(f"`{' '.join(argv)}` is not a plain `cd <dir>`")
    target = os.path.expanduser(argv[1])
    if os.path.isabs(target) or cwd is None:
        return target
    return os.path.join(cwd, target)


def _split_env(argv):
    """(argv, env) for a segment that may carry `NAME=value` prefixes."""
    env = {}
    while argv and _ENV_ASSIGN.match(argv[0]):
        name, value = argv.pop(0).split("=", 1)
        env[name] = value
    if not argv:
        raise UnsupportedCommand("a command segment sets variables but runs nothing")
    return argv, env


def command_steps(cmd):
    """The correctness command as ([(argv, env)], cwd), parsed WITHOUT a shell.

    PANKO runs the operator's correctness command directly rather than through
    `bash -c`. The command reaches the harness as a string in a dispatch prompt,
    and handing that string to a shell makes every character in it executable.
    What a correctness command actually needs is the documented form --
    `cd custom/<op> && python3 test_<op>.py`, optionally with `NAME=value`
    prefixes -- and that is exactly what this parses.

    Anything that genuinely needs a shell (a pipe, a redirect, a subshell, a
    background job, a glob) is refused by name rather than silently mis-run: put
    it in a script and name the script instead.
    """
    for ch in _SHELL_ONLY:
        if ch in cmd:
            raise UnsupportedCommand(
                f"the correctness command uses `{ch}`, which needs a shell. PANKO "
                f"runs it directly; put the command in a script and call that."
            )
    cwd, steps = None, []
    for part in cmd.split("&&"):
        if "&" in part:
            raise UnsupportedCommand("`&` (background) makes the exit code meaningless")
        argv = shlex.split(part)
        if not argv:
            raise UnsupportedCommand("the correctness command has an empty segment")
        if argv[0] == "cd":
            cwd = _chdir(cwd, argv)
            continue
        steps.append(_split_env(argv))
    if not steps:
        raise UnsupportedCommand("the correctness command runs nothing")
    return steps, cwd


def _run_one(step, cwd, timeout, env=None):
    """One parsed step: (combined output, exit code).

    The program is resolved to an absolute path before it runs, so which binary
    executes does not depend on how PATH happens to be ordered.
    """
    argv, step_env = step
    exe = shutil.which(argv[0], path=(env or os.environ).get("PATH"))
    if not exe:
        return f"{argv[0]}: not found on PATH\n", 127
    run_env = dict(env or os.environ)
    run_env.update(step_env)
    try:
        r = subprocess.run(
            [os.path.abspath(exe), *argv[1:]],
            cwd=cwd,
            env=run_env,
            timeout=timeout,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.TimeoutExpired as e:
        out = e.output
        text = (
            out
            if isinstance(out, str)
            else (out.decode(errors="replace") if out else "")
        )
        return text, 124
    return (r.stdout or ""), r.returncode


def run_command(cmd, timeout, env=None):
    """Run a correctness command without a shell. Returns (combined output, exit code).

    `&&` semantics: the steps run in order and stop at the first failure.
    """
    steps, cwd = command_steps(cmd)
    text, code = "", 0
    for step in steps:
        out, code = _run_one(step, cwd, timeout, env)
        text += out
        if code:
            break
    return text, code


def default_run(cmd, timeout, env=None):
    """The correctness command as E(x) runs it: (exit code, combined output).

    A command this runner will not execute is reported as exit 126 with the
    reason as its output, rather than as an exception in the middle of a
    campaign: the caller's contract here is an exit code, and 126 is the shell's
    own "found it, could not run it".
    """
    try:
        out, code = run_command(cmd, timeout, env)
    except UnsupportedCommand as e:
        return 126, str(e)
    return code, out


# (default runner is a bound method RealEvaluator._default_run so it can capture output for
#  diagnostics — see below; a plain function can't stash the failing command's stderr.)


# Failures that say nothing about the candidate. A tile that overflows the buffer
# is evidence; a device that was busy is not, and the two arrive through the same
# non-zero exit code.
#
# The first recorded block spent five evaluations on `MAP_REG_ADDR_FAILED`,
# scored them as five bad tile shapes, and closed on its stagnation counter
# having measured nothing at all. Charging budget for them is the smaller half of
# the damage: they also poison the surrogate, because the sampler is told that a
# perfectly good region returned the infeasible penalty.
#
# They are split by KIND, because the two want opposite responses and because
# conflating them corrupted the record of a run: a campaign can end a fraction
# of the way into its budget with `device_unavailable`, and nothing anywhere says
# whether the device had actually been held or a path had simply been mistyped.
#
# device     : another process holds the device. Waiting is the correct response.
# invocation : the command never started. Waiting cannot help; it is a bug here.
DEVICE_FAULT_MARKERS = (
    "MAP_REG_ADDR_FAILED",  # another process holds the device's register map
    "DEVICE_MEM_ERROR",
    "aclrtSetDevice failed",
    "Device is busy",
    "no available device",
    "RESOURCE_EXHAUSTED",
    "ERROR_DEVICE_NOT_AVAILABLE",
    # The device allocator running dry, which is contention and not a verdict on
    # any tile. It read as capacity evidence because CAPACITY_MARKERS below
    # carries the generic "out of memory", and the two allocators are different
    # things: a tile that does not fit says `exceeds MEM_UB size [196608]`, while
    #     torch.OutOfMemoryError: NPU out of memory. Tried to allocate 194.00 MiB
    #     (NPU 2; 60.96 GiB total capacity; ... 2.44 MiB free ...)
    # says another process is holding nearly all of the device memory. Read as
    # capacity evidence it sets a learned ceiling far below the real one, and the
    # rest of the block is then rejected statically -- many trials, none
    # feasible, on the lever that mattered most. Naming it here makes it a fault,
    # so the
    # device is waited out and the candidate is never scored at all.
    "NPU out of memory",
    "torch.OutOfMemoryError",
    "CUDA out of memory",
)

# A block that passes its test command without the working directory the command
# expects loses every candidate to `can't open file '<test script>'` -- those
# evaluations are charged, and those good regions are reported to the sampler as
# infeasible. A command that cannot find its own script has said nothing
# whatsoever about the tile, and it will say nothing after a wait either.
INVOCATION_FAULT_MARKERS = (
    "can't open file",
    "cannot open file",
    "No such file or directory",
    "ModuleNotFoundError",
    # `python -m missing_module` does NOT print ModuleNotFoundError; runpy
    # prints "<interpreter>: No module named <name>". A recorded block passed
    # --lint-root into an environment without pypto_op_lint and six tile values
    # were scored as candidate failures on that message, six evaluations gone,
    # none of them having reached the device.
    "No module named",
    "command not found",
    "ImportError",
)

# Scanned in this order, device first, deliberately. Several invocation markers
# are generic enough to appear inside a genuine device error -- CANN reports a
# missing device node as `/dev/davinci5: No such file or directory` -- and
# reading that as a harness bug would send the next run chasing its own paths.
# The reverse mistake cannot happen: a plain ImportError carries no device
# marker at all.
ENV_FAULT_MARKERS = DEVICE_FAULT_MARKERS + INVOCATION_FAULT_MARKERS


def fault_kind(marker):
    """ "device" (wait and retry) or "invocation" (a bug in how we called it)."""
    return "invocation" if marker in INVOCATION_FAULT_MARKERS else "device"


# Failures that ARE monotone evidence about size: if this footprint did not fit,
# no larger one will. Everything else -- a golden mismatch, a lint failure, a
# timeout -- says nothing about size and must not set a ceiling.
#
# ERR_CONFIG_TILE is deliberately NOT here. It is the matmul tile-config error,
# and its dominant form is a DIVISIBILITY refusal -- "Invalid L1/L0 relation:
# nL0=192, nL1=128, require nL0 <= nL1 && nL1 % nL0 == 0". That is not monotone
# in footprint: nL0=192 fails while the strictly larger mL0=256,kL0=128,nL0=128
# is legal. Treating it as capacity evidence set a ceiling from a divisibility
# failure and then free-rejected every larger tile, legal ones included -- and
# the ceiling is expressed in `mL0*kL0` bytes, so the axis that actually failed
# is invisible to the quantity being bounded. This class is handled properly by
# the learned L1 relation in driver, which reads the reported extent off the
# message and gates on divisibility rather than on size.
CAPACITY_MARKERS = (
    "ALLOC_FAILED",
    "out of memory",
    "OUT_OF_MEMORY",
    "buffer overflow",
    "exceeds UB",
    "exceed the buffer",
    "UB size",
)


class EnvironmentFault(Exception):
    """The evaluation could not be performed. Not a verdict on the candidate.

    `kind` says which of the two it was and therefore what may be concluded from
    it; `retries` says how many backoffs were already spent before giving up, so
    "the device was busy for four minutes" is distinguishable from "the device
    answered instantly and said no".
    """

    def __init__(self, marker, detail="", kind=None, retries=0):
        super().__init__(marker)
        self.marker, self.detail = marker, detail
        self.kind = kind or fault_kind(marker)
        self.retries = retries


# What a measurement carries forward into the symptom index, beside (s, p).
_CARRIED_METRICS = (
    "util",
    "bubble",
    "aic_util",
    "aiv_util",
    "pred_stall",
    "aic_bubble",
    "aiv_bubble",
)


class RealEvaluator:
    def __init__(
        self,
        op,
        op_dir,
        op_file,
        device,
        test_command,
        lint_root=None,
        lint_stage=7,
        warmup=1,
        runs=3,
        agg="min",
        eval_timeout_s=300,
        run_cmd=None,
        measure_fn=None,
        fault_backoff_s=(30, 90),
        sleep_fn=None,
        evaluation_mode="subprocess",
    ):
        """test_command : shell string that exits 0 iff golden+layout pass (from the dispatch prompt).
        lint_root      : path to the pypto-op-lint hook directory under
                         cannbot-skills/plugins-official/pypto-op-orchestrator/hooks
                         (enables the Stage-7 lint gate). None skips it.
        warmup/runs/agg: frozen measure_latency params (o4 defaults 1/3/min).
        fault_backoff_s: waits before re-running the command after a DEVICE fault.
                         Contention on a shared device is usually brief, and the
                         alternative to waiting is what a recorded run did --
                         three faults in a row aborted the block and ended the
                         campaign with 74 of its 85 evaluations unspent. Two
                         waits bound a trial at about two extra minutes, which
                         is cheaper than one lost run by a wide margin.
        sleep_fn       : injected in tests so the schedule is asserted, not waited on.
        """
        self.op = op
        self.op_dir = op_dir
        self.op_file = op_file
        self.device = str(device)
        self.test_command = test_command
        self.lint_root = lint_root
        self.lint_stage = lint_stage
        self.warmup = warmup
        self.runs = runs
        self.agg = agg
        self.eval_timeout_s = eval_timeout_s
        self._last_output = ""  # combined stdout+stderr of the last default-runner cmd
        # Diagnostics of the LAST successful measurement (util / bubble). The
        # objective returns only (s, p) because latency is what the study
        # optimises, and for a long time these were simply dropped -- which is
        # how nine block winners reached the state carrying `util: 0.0` on
        # programs the block had in fact measured. Keeping the dict costs
        # nothing and lets the winner be recorded with the reading it already
        # produced, without a second trip to the device.
        self.last_metrics = {}
        # The last failure's final lines, carried out so the block can persist
        # them. Without this a run reports "23 of 24 candidates failed" and
        # nothing about why -- and the capacity markers cannot be tuned
        # against output nobody kept.
        self.last_detail = ""
        self.fault_backoff_s = tuple(fault_backoff_s or ())
        self.last_retries = 0  # backoffs spent on the last evaluation
        self._sleep = sleep_fn or time.sleep
        self._run_cmd = run_cmd or self._default_run
        self._measure_fn = measure_fn
        if evaluation_mode not in ("subprocess", "resident", "cached", "prepared"):
            raise ValueError("unknown evaluation mode")
        if evaluation_mode != "subprocess" and (
            run_cmd is not None or measure_fn is not None
        ):
            raise ValueError("injected runners require subprocess mode")
        self.evaluation_mode = evaluation_mode
        self._session_result = {}
        with open(op_file, "r", encoding="utf-8") as f:
            self.base_src = f.read()
        # Discover the per-site tunable tile calls once; pass self.sites to driver.run_bo.
        try:
            self.sites = apply.tunable_sites(self.base_src)
        except SyntaxError:
            if evaluation_mode not in ("cached", "prepared"):
                raise
            self.sites = []

    # ------------------------------------------------------------------ objective
    def __call__(self, site_configs):
        """Evaluate one PER-SITE config. Returns (s, p): (1, latency) if correct, else (0, None)."""
        # Cleared FIRST, so a caller reading `last_detail` after this call can
        # never be handed the PREVIOUS trial's text. The `n == 0` guard below
        # returns without running anything, and a driver that files the refusal
        # under whatever `last_detail` still held attributed one trial's compiler
        # error to another trial's parameters.
        self.last_detail = ""
        new_src, n = apply.apply(self.base_src, site_configs)
        if n == 0:
            return 0, None  # nothing to apply -> guard (misconfigured site ids)
        self._write(new_src)
        try:
            # `_run_test` distinguishes "this candidate is wrong" from "the device
            # could not be used", and waits out the second. Only the first is a
            # verdict; the second must not be charged, counted, or learned from.
            if self._run_test() != 0:
                # The third element says whether this failure is evidence about
                # SIZE. Only a capacity refusal is monotone; a golden mismatch at
                # 4096 elements tells the search nothing about 8192.
                self.last_detail = self._tail(4)
                return 0, None, self._failure_kind()
            if not self._lint_ok():
                # The lint is a subprocess like the others and fails for the
                # same non-verdict reasons. It was the one gate that never
                # asked, so `No module named pypto_op_lint` was recorded six
                # times as "this tile is wrong".
                marker = self._env_fault()
                if marker:
                    self.last_detail = f"lint: {marker}: {self._tail(4)}"
                    raise EnvironmentFault(marker, self._tail(), fault_kind(marker))
                self.last_detail = "lint: " + self._tail(4)
                return 0, None, "other"  # Stage-7 lint S0/S1 FAIL
            m = self._measure() or {}
            # aic_util/aiv_util/pred_stall ride along on the same measurement:
            # `measure` splits the per-core rows it already parsed by the AIC/AIV
            # prefix, so a block winner carries the per-pipe reading forward to
            # the symptom index exactly as it carries util and bubble.
            self.last_metrics = {}
            for k in _CARRIED_METRICS:
                if m.get(k) is not None:
                    self.last_metrics[k] = m.get(k)
            p = m.get("p")
            if p is None:
                # The measurement runs on the device too, so it can fail for the
                # same reasons the correctness command can. Scoring a busy
                # device as "this tile produced no latency" is the mistake this
                # whole class of fix exists to stop.
                marker = self._env_fault()
                if marker:
                    self.last_detail = f"{marker}: {self._tail(4)}"
                    raise EnvironmentFault(marker, self._tail(), fault_kind(marker))
                self.last_detail = "no latency from the profiler"
                return 0, None, "other"
            self.last_detail = ""
            return 1, float(p), "ok"
        finally:
            self._write(self.base_src)  # restore base for the next trial

    def evaluate_current(self):
        """Evaluate the op_file AS-IS (no apply, no restore): (s, m) where m = {p, util, bubble}.

        Used by the shared INIT for the generated baseline and, after finalize(), for the post-BO
        baseline that seeds PANKO (its util/bubble drive the structural symptom re-boost).
        On failure returns (0, {reason, detail}) so a broken baseline is diagnosable (e.g. the test
        can't import its golden) instead of an opaque preopt_incorrect.

        Same retry as a trial: a preopt measured while another process holds the
        device is not a baseline, and every later speedup is quoted against it.
        """
        try:
            rc = self._run_test()
        except EnvironmentFault as e:
            return 0, {
                "reason": f"environment fault: {e.marker}",
                "detail": e.detail,
                "fault_kind": e.kind,
            }
        if rc != 0:
            return 0, {
                "reason": "test_command failed (golden/layout)",
                "detail": self._tail(),
            }
        if not self._lint_ok():
            return 0, {"reason": "lint gate failed", "detail": self._tail()}
        m = self._measure() or {}
        if m.get("p") is None:
            return 0, {
                "reason": "perf measure returned no latency",
                "detail": self._tail(),
            }
        return 1, m

    def finalize(self, site_configs):
        """Write the winning per-site config to op_file (call once after run_bo, before cmd_record)."""
        new_src, _ = apply.apply(self.base_src, site_configs)
        self._write(new_src)
        return new_src

    def lint_result(self):
        """The original lint gate and its captured diagnostics for a worker."""
        return self._lint_ok(), self._last_output

    # ------------------------------------------------------------------ internals
    def _default_run(self, cmd, timeout, env=None):
        """Run the correctness command; return its exit code (124 on timeout).

        No shell: see `command_steps`. The exit code is the result here, so
        `check=False` is the contract rather than an oversight -- a non-zero code
        means s=0, not an error. The combined output is captured so a failing
        correctness command (e.g. a golden ImportError) can be surfaced, not just
        s=0.
        """
        code, self._last_output = default_run(cmd, timeout, env)
        return code

    def _tail(self, n=12):
        return "\n".join(self._last_output.strip().splitlines()[-n:])

    def _failure_kind(self):
        """Why the last command failed: "capacity" or "other".

        A device fault wins, whatever else the output says. `_run_test` already
        raises on the markers it knows, so this only matters for wording it does
        not -- but the failure it guards against is expensive and silent: one
        contention OOM read as capacity teaches a ceiling that free-rejects every
        larger tile for the rest of the run.
        """
        out = self._last_output or ""
        if any(m in out for m in DEVICE_FAULT_MARKERS):
            return "other"
        return "capacity" if any(m in out for m in CAPACITY_MARKERS) else "other"

    def _env_fault(self):
        """The marker naming an environment fault in the last command's output."""
        out = self._last_output or ""
        for m in ENV_FAULT_MARKERS:
            if m in out:
                return m
        return None

    def _test_once(self):
        """Dispatch one correctness attempt through the selected evaluator."""
        if self.evaluation_mode == "subprocess":
            return self._run_cmd(self.test_command, self.eval_timeout_s)
        import evaluation_session

        self._session_result = evaluation_session.request(self)
        self._last_output = self._session_result.get("test_output", "")
        return self._session_result["test_rc"]

    def _run_test(self):
        """Run the correctness command, waiting out a busy device but not a bug.

        Returns the exit code, or raises EnvironmentFault when the command could
        not be performed at all. Three outcomes, three responses:

            exit 0 or a plain failure  -> a verdict on the candidate; return it
            a DEVICE marker            -> wait and run it again
            an INVOCATION marker       -> raise at once; a wait cannot fix a path

        The retry is here rather than in the driver because it is the device that
        is being retried, not the search: a candidate re-run after a wait is an
        ordinary trial with an ordinary result, and the driver never learns it
        happened beyond `last_retries`.
        """
        self.last_retries = 0
        for wait in self.fault_backoff_s + (None,):
            rc = self._test_once()
            if rc == 0:
                return rc
            marker = self._env_fault()
            if marker is None:
                return rc
            kind = fault_kind(marker)
            if kind != "device" or wait is None:
                self.last_detail = f"{marker}: {self._tail(4)}"
                raise EnvironmentFault(marker, self._tail(), kind, self.last_retries)
            self.last_retries += 1
            self._sleep(wait)
        # The backoff tuple ends with `None`, whose iteration either returns or
        # raises, so control does not reach here. Spelled out so that every
        # branch of this function has the same kind of exit.
        raise EnvironmentFault(
            "backoff_exhausted", self._tail(), "device", self.last_retries
        )

    def _write(self, src):
        with open(self.op_file, "w", encoding="utf-8") as f:
            f.write(src)

    def _lint_ok(self):
        if self.evaluation_mode != "subprocess":
            self._last_output = self._session_result.get("lint_output", "")
            return self._session_result.get("lint_rc", 1) == 0
        if not self.lint_root:
            return True
        env = dict(os.environ)
        env["PYTHONPATH"] = self.lint_root + os.pathsep + env.get("PYTHONPATH", "")
        cmd = (
            f"{sys.executable} -m pypto_op_lint --lint-impl "
            f'--op-dir "{self.op_dir}" --stage {self.lint_stage}'
        )
        return self._run_cmd(cmd, self.eval_timeout_s, env=env) == 0

    def _measure(self):
        """Run the frozen measure_latency.py AS A SUBPROCESS and read its JSON.

        It used to be imported and called in-process, which is what made a block
        able to measure exactly one tile. `_build_run_once` calls
        `torch.npu.set_device`, and the process then holds the device for the
        rest of its life -- so the NEXT trial's test command, which is a
        subprocess, could not map the registers:

            HostLauncherErr::MAP_REG_ADDR_FAILED. Map reg addr fail,
            maybe others are using current device. (ret=13, DEVICE_ID=4)

        The "others" was this process. Measured directly: eleven of twelve
        kernels succeed on the first evaluation and fail on the second and
        third, while the same test command run twice from a shell passes both
        times. Four recorded blocks show the signature exactly -- one
        measurement, then three consecutive environment faults, then abort --
        and were written up at the time as a concurrent campaign holding the
        device.

        A subprocess also makes this evaluator's E(x) the same shape as the
        baseline arm's, which invoked measure_latency.py from the command line
        for every evaluation. The two arms were paying different process costs
        for the same nominal measurement; now they do not.
        """
        if self.evaluation_mode != "subprocess":
            self._last_output = self._session_result.get("measure_output", "")
            return self._session_result.get("metrics", {})
        if self._measure_fn is not None:
            return self._measure_fn(self.op_file)
        ml_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "measure_latency.py"
        )
        # An argv, not a command line: nothing here needs a shell, and building
        # one meant quoting `op_file` by hand and hoping. `sys.executable` is
        # already an absolute path, so which interpreter runs does not depend on
        # PATH either.
        argv = [
            sys.executable,
            ml_path,
            "--op-dir",
            str(self.op_dir),
            "--op",
            str(self.op),
            "--op-file",
            str(self.op_file),
            "--device",
            str(self.device),
            "--warmup",
            str(self.warmup),
            "--runs",
            str(self.runs),
            "--agg",
            str(self.agg),
        ]
        try:
            r = subprocess.run(
                argv,
                timeout=self.eval_timeout_s,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            out = r.stdout or ""
        except subprocess.TimeoutExpired as e:
            out = e.output if isinstance(e.output, str) else ""
        # Kept so `_env_fault` can see a device fault raised during measurement,
        # not only during the correctness command.
        self._last_output = out
        for line in reversed(out.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except ValueError:
                    continue
        return {}
