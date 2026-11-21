#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""measure_latency.py — the frozen warm-latency measurement for PANKO E(x).

Used IDENTICALLY for the preopt baseline (INIT) and every candidate during the search, so the
`p` values are directly comparable. It fixes two bugs seen in ad-hoc measurement:

  1. Cold vs warm. The kernel's first call is cold (JIT compile / cache fill) and reads ~20-25%
     slower than the warm steady state. We run `warmup` calls and DISCARD them, then measure
     `runs` warm calls.
  2. Capture. The on-device profiler prints its "AICORE Prof Summary" at the native (fd) level,
     so Python's `redirect_stdout` (sys.stdout only) misses it. We redirect at the file-descriptor
     level (os.dup2) around the calls — keeping ONE warm process — then parse the captured text.

`p` = aggregate (median by default; min optional) of the warm "AICore End-to-End Time" values.
Only the on-device AICore time is used — never wall-clock (which includes host/profiling overhead).

The measurement MECHANICS below are the frozen part. The op-specific `run_once()` (build inputs +
one wrapper call + synchronize) is supplied by the caller / CLI.
"""

import argparse
import glob
import importlib.util
import json
import os
import re
import statistics
import sys
import tempfile
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jsonio  # noqa: E402

# One home for the two runtime questions, rather than a copy here: ask for an
# optional module, and treat a driver's refusal as the answer. `chip_profile`
# owns them because it is where the refusals are load-bearing.
from chip_profile import optional as _optional, quietly as _quietly  # noqa: E402

leaked_capture = []  # capture files a cleanup could not remove


def _read_text(path, errors="replace"):
    """The file's text, with the handle closed before the caller sees it."""
    with open(path, encoding="utf-8", errors=errors) as f:
        return f.read()


AICORE_RE = re.compile(r"AICore End-to-End Time:\s*([\d.]+)")
# Per-core scheduling record written by the profiler to
# output/output_<ts>/bubble_analysis.log. Bubble is NOT a string in the
# profiler summary: it is derived from these four fields, exactly as
# cannbot-skills/ops/pypto-op-perf-tune/perf-analyzer/scripts/analyze_perf.py
# derives it. Grepping stdout for "Bubble:" therefore always yielded None,
# which silently disabled the bubble half of the Stage-7 symptom index.
CORE_RE = re.compile(
    r"\[(AIC_\d+|AIV_\d+)\] Execute task num:(\d+)\s+"
    r"Core Total Work Time: ([\d.]+)\s+Total Wait Time: ([\d.]+)\s+"
    r"Wait Schedule Time: ([\d.]+)\s+Wait Predecessor Time: ([\d.]+)"
)
UTIL_RE = re.compile(r"AICore Utilization:\s*([\d.]+)")
BUBBLE_RE = re.compile(r"[Bb]ubble(?:\s*[Rr]ate)?:\s*([\d.]+)")


def _core_rows(path):
    """The per-core profiler records, or []. `CORE_RE` captures six fields."""
    try:
        return CORE_RE.findall(_read_text(path))
    except OSError:
        return []


def _bubble_from_log(path):
    """Mean bubble rate and mean core utilisation over all AIC/AIV cores.

    Definitions follow analyze_perf.py exactly:
        aicore_time = Core Total Work Time - Total Wait Time   (net busy)
        core_util   = aicore_time / (aicore_time + total_wait)
        bubble_rate = wait_schedule / (aicore_time + wait_schedule)
    Verified against the summary metric: both routes agree to a few points.
    """
    rows = _core_rows(path)
    if not rows:
        return None, None
    utils, bubbles = [], []
    for _name, _n, work, wait, sched, _pred in rows:
        work, wait, sched = float(work), float(wait), float(sched)
        busy = work - wait
        utils.append(busy / (busy + wait) * 100 if busy + wait else 0.0)
        bubbles.append(sched / (busy + sched) * 100 if busy + sched else 0.0)
    return (round(sum(bubbles) / len(bubbles), 4), round(sum(utils) / len(utils), 4))


def _pipes_from_log(path):
    """Per-pipe utilisation and dependency stall: {"AIC": {...}, "AIV": {...}}.

    Two of the six fields `CORE_RE` already captures were being discarded, and
    each of them is a bottleneck the aggregate cannot express.

    THE CORE NAME. `util` above is a mean over AIC and AIV together, so with 20
    cube cores against 40 vector cores the vector pipe carries two thirds of the
    weight. A kernel whose vector pipe is saturated while its cube pipe is
    nearly idle therefore reads comfortably above the 50% floor that would have
    raised a symptom, and a whole campaign can run with nothing pointing at the
    idle pipe. Splitting by the `AIC_`/`AIV_` prefix recovers it, and agrees
    with the swim-graph's own Cube and Vector Utilisation on the same kernel.

    WAIT PREDECESSOR. `bubble_rate` counts only `wait_schedule`. On a
    pipe-serialized core nearly all of the stall is `wait_predecessor` instead
    -- the whole cost of a serial cube->vector->cube recurrence -- so it is
    excluded from the metric by construction.

    Neither number costs a measurement: both are in the rows already parsed.
    """
    rows = _core_rows(path)
    if not rows:
        return {}
    acc = {}
    for name, n, work, wait, sched, pred in rows:
        work, wait, sched, pred = float(work), float(wait), float(sched), float(pred)
        busy = work - wait
        d = acc.setdefault(
            name.split("_")[0], {"u": [], "b": [], "s": [], "cores": 0, "tasks": 0}
        )
        d["u"].append(busy / (busy + wait) * 100 if busy + wait else 0.0)
        # `bubble_rate`'s own formula, per pipe rather than averaged over both.
        d["b"].append(sched / (busy + sched) * 100 if busy + sched else 0.0)
        # The same shape again, against the other wait.
        d["s"].append(pred / (busy + pred) * 100 if busy + pred else 0.0)
        d["cores"] += 1
        d["tasks"] += int(n)
    return {
        p: {
            "util": round(sum(d["u"]) / len(d["u"]), 2),
            "bubble": round(sum(d["b"]) / len(d["b"]), 2),
            "pred_stall": round(sum(d["s"]) / len(d["s"]), 2),
            "cores": d["cores"],
            "tasks": d["tasks"],
        }
        for p, d in sorted(acc.items())
    }


def _newest(roots, since, name):
    """Newest output/output_*/<name> created after `since`."""
    best, best_mt = None, since
    for root in roots:
        if not root:
            continue
        for d in glob.glob(os.path.join(root, "output", "output_*")):
            p = os.path.join(d, name)
            try:
                mt = os.path.getmtime(p)
            except OSError:
                continue
            if mt >= best_mt:
                best, best_mt = p, mt
    return best


def _find_bubble_log(roots, since):
    return _newest(roots, since, "bubble_analysis.log")


def _core_tasks(core):
    """(start, end) for every task on this core that carries both."""
    out = []
    for t in core.get("tasks") or []:
        s, e = t.get("execStart"), t.get("execEnd")
        if s is not None and e is not None:
            out.append((s, e))
    return out


def _core_totals(cores):
    """({coreType: {busy, cores, tasks}}, wall span) over the AIC/AIV cores."""
    acc, lo, hi = {}, None, None
    for c in cores:
        kind = c.get("coreType") if isinstance(c, dict) else None
        if kind not in ("AIC", "AIV"):
            continue
        d = acc.setdefault(kind, {"busy": 0, "cores": 0, "tasks": 0})
        d["cores"] += 1
        for s, e in _core_tasks(c):
            d["busy"] += e - s
            d["tasks"] += 1
            lo = s if lo is None or s < lo else lo
            hi = e if hi is None or e > hi else hi
    span = (hi - lo) if (lo is not None and hi is not None) else 0
    return acc, span


def _pipes_from_prof(path):
    """Per-pipe utilisation from the RAW profile, when the log is absent.

    `bubble_analysis.log` is written by `analysis_wait_cycles`, and the installed
    draw_swim_lane.py carries a "PANKO latency-only fast path" that exits
    before calling it -- so most evaluations of a run can produce no log at all
    and the per-pipe split is simply missing. That patch lives in
    site-packages, outside version control, alongside four earlier backups of the
    same file; adding a fifth would reproduce exactly the fragility that silently
    disabled the bubble half of the symptom index in the first place.

    `tilefwk_L1_prof_data.json` is the INPUT that fast path still parses, so it
    is upstream of the file people keep patching and is written on every run:
    one entry per core, `coreType` in {AIC, AIV}, and `tasks` carrying
    `execStart` / `execEnd`.

    What this recovers is utilisation only, and by print_aicore_summary's own
    definition -- busy task time over (core count x wall span) -- rather than the
    log's per-core busy/(busy+wait). The two nearly agree because a core's
    assigned window is nearly the whole span (a measured flash core: 23527 of
    23740). The WAIT BREAKDOWN is not recoverable: bubble and pred_stall come
    from analysis_wait_cycles, which is the thing the fast path skips. So
    `cube_starved`, which needs only utilisation, becomes available on every
    evaluation; `pipe_serialized`, which needs the predecessor wait, still needs
    the log. Absent is left absent rather than zeroed.
    """
    try:
        with open(path, encoding="utf-8") as f:
            cores = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(cores, list):
        return {}
    acc, span = _core_totals(cores)
    if not span:
        return {}
    return {
        p: {
            "util": round(d["busy"] / (d["cores"] * span) * 100, 2),
            "cores": d["cores"],
            "tasks": d["tasks"],
        }
        for p, d in sorted(acc.items())
        if d["cores"]
    }


def _agg(vals, how):
    if not vals:
        return None
    return round(min(vals) if how == "min" else statistics.median(vals), 4)


def _capture(run_once, total):
    """Run `run_once` `total` times with fds 1 and 2 redirected to a temp file.

    Native (fd-level) capture, because the profiler writes from C rather than
    through `sys.stdout`. Returns everything it printed.
    """
    sys.stdout.flush()
    sys.stderr.flush()  # don't let pre-existing buffered output leak into capture
    saved_out, saved_err = os.dup(1), os.dup(2)
    tf = tempfile.NamedTemporaryFile(mode="w+", suffix=".prof", delete=False)
    try:
        os.dup2(tf.fileno(), 1)
        os.dup2(tf.fileno(), 2)
        for _ in range(total):
            run_once()
            sys.stdout.flush()
            sys.stderr.flush()
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        tf.flush()
        tf.seek(0)
        text = tf.read()
        tf.close()
        try:
            os.unlink(tf.name)
        except OSError:
            # A capture file we could not remove is litter in a temp directory,
            # not a failed measurement. The numbers are already read.
            leaked_capture.append(tf.name)
    return text


def _pipe_readings(op_dir, t0):
    """Per-pipe counters, and which file they came from."""
    log = _find_bubble_log([op_dir, os.getcwd()], t0)
    log_bubble, log_util = _bubble_from_log(log) if log else (None, None)
    pipes = _pipes_from_log(log) if log else {}
    source = "bubble_analysis.log" if pipes else None
    if not pipes:  # the swimlane fast path skipped the log
        prof = _newest([op_dir, os.getcwd()], t0, "tilefwk_L1_prof_data.json")
        pipes = _pipes_from_prof(prof) if prof else {}
        source = "tilefwk_L1_prof_data.json" if pipes else None
    return {
        "log": log,
        "log_bubble": log_bubble,
        "log_util": log_util,
        "pipes": pipes,
        "source": source,
    }


def measure(run_once, warmup=1, runs=3, agg="min", op_dir=None):
    """Run `run_once` warmup+runs times capturing native (fd-level) profiler output; discard the
    first `warmup`, aggregate the warm `runs`. `run_once` must perform exactly one profiled kernel
    call (+ device sync) and let the profiler print its summary. Returns {p, util, bubble, samples}.
    """
    total = max(1, warmup) + max(1, runs)
    t0 = time.time() - 1.0  # tolerate coarse mtime granularity
    text = _capture(run_once, total)
    lat = [float(x) for x in AICORE_RE.findall(text)]
    util = [float(x) for x in UTIL_RE.findall(text)]
    bub = [float(x) for x in BUBBLE_RE.findall(text)]
    warm_lat = lat[warmup:] if len(lat) > warmup else lat[-runs:]
    warm_util = util[warmup:] if len(util) > warmup else util[-runs:]
    warm_bub = bub[warmup:] if len(bub) > warmup else bub
    # Prefer the profiler's own per-core record; fall back to the (usually
    # empty) stdout regex so behaviour never regresses if the log is absent.
    read = _pipe_readings(op_dir, t0)
    log_bubble = read["log_bubble"]
    aic, aiv = read["pipes"].get("AIC") or {}, read["pipes"].get("AIV") or {}
    return {
        "p": _agg(warm_lat, agg),
        "util": _agg(warm_util, "median"),
        "bubble": log_bubble if log_bubble is not None else _agg(warm_bub, "median"),
        "bubble_source": "bubble_analysis.log" if log_bubble is not None else "stdout",
        "bubble_log": read["log"],
        "util_from_log": read["log_util"],
        # Per-pipe. Absent (None) when no log was produced -- the installed
        # draw_swim_lane.py carries a latency-only fast path that exits
        # before writing one, so this is missing far more often than `util`
        # is, and "not measured" must stay distinguishable from "zero".
        "aic_util": aic.get("util"),
        "aiv_util": aiv.get("util"),
        # `bubble` above is a mean over both pipes. Per pipe it separates a
        # scheduler that cannot feed the cores from one that can: widening
        # a vec tile can drop AIV utilisation and raise AIV bubble while
        # latency ROSE, which is the signature of fewer, larger tasks
        # starving the vector cores rather than of more work.
        "aic_bubble": aic.get("bubble"),
        "aiv_bubble": aiv.get("bubble"),
        "pred_stall": max(
            [d["pred_stall"] for d in read["pipes"].values() if "pred_stall" in d],
            default=None,
        ),
        "pipes": read["pipes"],
        "pipe_source": read["source"],
        "samples": warm_lat,
        "raw_count": len(lat),
    }


# --------------------------------------------------------------------------- CLI, by convention
def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_run_once(op_dir, op, op_file, device, inputs_file=None):
    """Convention: `custom/<op>/eval/test_inputs.py` exposes `build_inputs()` (or `INPUTS`) giving
    the args for the wrapper `op`. `op_file` accepts a path or an already imported module.
    Adapt here if an op differs; the MECHANICS
    in `measure()` stay frozen regardless.
    """
    import torch  # noqa

    # Importing torch_npu is what registers the NPU backend; on a box without it
    # there is nothing to register and the caller finds out at the first device
    # call, with a message about the device rather than about an import.
    _optional("torch_npu")
    os.environ["TILE_FWK_DEVICE_ID"] = str(device)
    _quietly(torch.npu.set_device, int(device))
    impl = (
        op_file
        if isinstance(op_file, types.ModuleType)
        else _load_module(op_file, "op_impl")
    )
    # `<op>_wrapper` is the more common convention in this repo, but `--op` names
    # the operator, not the callable. Without the fallback every trial on such a
    # kernel raises AttributeError inside the measurement subprocess, which the
    # evaluator cannot distinguish from a device fault -- so a whole campaign
    # scores as environment faults and no candidate is ever measured.
    wrapper = getattr(impl, op, None) or getattr(impl, f"{op}_wrapper")
    # `--inputs` names a DIFFERENT build_inputs, for measuring a kernel on a shape
    # other than the one the campaign optimised against. It defaults to the
    # campaign's own file, so E(x) is unchanged and every recorded number stays
    # comparable; this exists only for after-the-fact re-measurement.
    #
    # An operator's eval/test_inputs.py can measure at a smoke-test shape far
    # smaller than the one SPEC.md declares. Every loop-granularity result is
    # then drawn on a shape where the tile makes the loop run exactly once,
    # which it does not do at the declared size -- so the campaign optimises a
    # loop that is not there.
    ti = (
        inputs_file
        if isinstance(inputs_file, types.ModuleType)
        else _load_module(
            inputs_file or os.path.join(op_dir, "eval", "test_inputs.py"), "test_inputs"
        )
    )
    builder = getattr(ti, "build_inputs", None)
    inputs = builder(device) if callable(builder) else getattr(ti, "INPUTS")
    args = inputs if isinstance(inputs, (list, tuple)) else [inputs]

    def run_once():
        _ = wrapper(*args)
        # Synchronising is what makes the measured interval the kernel's; a
        # runtime that has no synchronize has already run it synchronously.
        _quietly(torch.npu.synchronize)

    return run_once


build_run_once = _build_run_once
capture_output = _capture


def main():
    ap = argparse.ArgumentParser(
        description="Frozen warm-latency measure for PANKO E(x)."
    )
    ap.add_argument("--op-dir", required=True)
    ap.add_argument("--op", required=True)
    ap.add_argument("--op-file", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--agg", choices=["median", "min"], default="min")
    ap.add_argument(
        "--inputs",
        default=None,
        help="alternative build_inputs module; defaults to "
        "<op-dir>/eval/test_inputs.py. Re-measurement only -- "
        "passing it means the number is NOT comparable to the "
        "campaign's recorded latencies.",
    )
    a = ap.parse_args()
    run_once = _build_run_once(a.op_dir, a.op, a.op_file, a.device, a.inputs)
    jsonio.emit(measure(run_once, a.warmup, a.runs, a.agg, op_dir=a.op_dir))


if __name__ == "__main__":
    main()
