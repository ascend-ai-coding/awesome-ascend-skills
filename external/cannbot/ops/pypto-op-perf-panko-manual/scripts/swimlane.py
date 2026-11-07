#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Features from a swim-lane trace, for the deterministic symptom index.

Why this exists
---------------
Until now the device half of that index read two scalars, `util` and `bubble`.
They answer one question -- *are the cores idle, and is the scheduler why* --
and they answer it well: the profiler's own `bubble_analysis.log` and the
median lane in `merged_swimlane.json` agree to the decimal on the same kernel,
so neither number is suspect.

What they cannot say is what the busy time was SPENT ON. A kernel can sit at
high utilisation with a low bubble rate -- healthy by both -- and still be
several times slower than the hand-written Ascend C kernel. A kernel that moves
twice the bytes it needs to is busy while it does it. Occupancy metrics score
that as success.

So the features here are deliberately about the SHAPE of the work rather than
its presence: how finely it is chopped, whether the two pipes are balanced,
whether anything was merged, and which tensors cross from Cube to Vector (the
hand-off that a fused kernel keeps on-chip and this one does not).

Nothing here enters `p` or `J`. These are diagnostics, on the same footing as
util and bubble, and every one of them may be absent: a trace that cannot be
read yields None and the symptoms simply do not fire.

Trace format
------------
`merged_swimlane.json` is Chrome-trace JSON. `ph:"M" name:"thread_name"` events
name the lanes (`AIC_n` = Cube, `AIV_n` = Vector, plus a `Fake Core_0` the
profiler uses for its own bookkeeping, which is not a core and is excluded).
`ph:"X"` events are tasks, with `ts`, `dur` and an `args` dict carrying
`hashOrder-hint` (subgraph merge info) and `ioperand-hint` / `ooperand-hint`
(tensor identities, as `rawmagic` integers).

`rawmagic` is coarse -- one recorded trace has 2 880 tasks over 37 distinct
magics -- so it identifies a BUFFER, not an individual value. Counting
producer/consumer task pairs therefore inflates wildly (1.18 M "edges" on that
trace). Cross-pipe traffic is counted per DISTINCT TENSOR instead, which is
small, interpretable, and matches what an action would act on.
"""

import collections
import json
import math
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jsonio  # noqa: E402

# `'rawmagic': 13` inside the operand hints. The hints are Python reprs embedded
# in a JSON string, not JSON, so they are read with a regex rather than parsed.
_MAGIC_RE = re.compile(r"'rawmagic':\s*(\d+)")
_SUBGRAPH_RE = re.compile(r"subGraphCount:\s*(\d+)")

# `<kind>Info hashOrder: <order>, subGraphCount: N` -- the profiler reports one
# of these per merge pass, and `tune-swimlane/references/merge-optimization.md`
# maps each to the pass_option that controls it.
_MERGE_RE = re.compile(r"(\w+)Info hashOrder:\s*(\S+?),\s*subGraphCount:\s*(\d+)")
_MERGE_KNOB = {
    "cubeMerge": "cube_nbuffer_setting",
    "l1Reuse": "cube_l1_reuse_setting",
    "vecMerge": "vec_nbuffer_setting",
}

# A task shorter than this is dominated by the cost of dispatching it. 2 us is
# the figure the CANNBot write-up uses for the same judgement ("60 % of task
# instances < 2 us is pure scheduling overhead"), and it is the only externally
# published number available here -- treat it as a convention, not a law.
SHORT_TASK_US = 2.0


def _names_a_thread(e):
    """Is this event the metadata record that gives a thread its name?"""
    return (
        e.get("ph") == "M"
        and e.get("name") == "thread_name"
        and isinstance(e.get("args"), dict)
    )


def _lane_names(events):
    return {
        (e["pid"], e["tid"]): e["args"].get("name")
        for e in events
        if _names_a_thread(e)
    }


def _pipe(lane):
    """'AIC_3' -> 'aic'. Anything that is not a core -> None."""
    if not lane:
        return None
    if lane.startswith("AIC"):
        return "aic"
    if lane.startswith("AIV"):
        return "aiv"
    return None  # 'Fake Core_0' is profiler bookkeeping, not a core


def _pipe_features(lanes, span):
    """Per-pipe shape of the work. `lanes` maps lane name -> [(ts, dur), ...]."""
    durs = []
    for v in lanes.values():
        for _, d in v:
            durs.append(d)
    if not durs:
        return {}
    busy = [sum(d for _, d in v) for v in lanes.values()]
    counts = [len(v) for v in lanes.values()]
    gaps = []
    for v in lanes.values():
        ordered = sorted(v)
        gaps += [
            b_ts - (a_ts + a_dur)
            for (a_ts, a_dur), (b_ts, _) in zip(ordered, ordered[1:])
            if b_ts > a_ts + a_dur
        ]
    f = {
        "n_lanes": len(lanes),
        "n_tasks": len(durs),
        "tasks_per_lane_med": statistics.median(counts),
        "med_dur_us": round(statistics.median(durs), 4),
        "min_dur_us": round(min(durs), 4),
        "frac_short": round(sum(1 for d in durs if d < SHORT_TASK_US) / len(durs), 4),
        # Busy fraction of the WALL SPAN. Verified against the profiler's own
        # per-core record, which agrees to the decimal.
        "busy_frac": round(statistics.median(busy) / span, 4) if span else None,
        "n_gaps": len(gaps),
        "gap_total_us": round(sum(gaps), 2),
    }
    if gaps:
        f["gap_med_us"] = round(statistics.median(gaps), 4)
        f["gap_max_us"] = round(max(gaps), 2)
    return f


def _cross_pipe_tensors(tasks, lane_of):
    """Distinct tensors written on one pipe and read on the other.

    This is the Cube->Vector hand-off. A fused kernel keeps that intermediate
    on-chip; an unfused one round-trips it through global memory, which is
    exactly what action S-14 exists to remove. Counted per tensor, not per task
    pair -- see the module docstring on why the pair count is meaningless.
    """
    produced_on = collections.defaultdict(set)
    consumed_on = collections.defaultdict(set)
    for e in tasks:
        p = _pipe(lane_of.get((e["pid"], e["tid"])))
        if not p:
            continue
        args = e.get("args") or {}
        for m in _MAGIC_RE.findall(args.get("ooperand-hint") or ""):
            produced_on[m].add(p)
        for m in _MAGIC_RE.findall(args.get("ioperand-hint") or ""):
            consumed_on[m].add(p)
    crossing = set()
    for m, produced in produced_on.items():
        consumed = consumed_on.get(m)
        touches_both = (produced | (consumed or set())) >= {"aic", "aiv"}
        if touches_both and consumed and produced != consumed:
            crossing.add(m)
    return {"n_tensors": len(produced_on), "n_cross_pipe_tensors": len(crossing)}


def _tasks_by_pipe(tasks, lane_of):
    """The measured tasks grouped pipe -> lane -> [(ts, dur)]."""
    by_pipe = {"aic": {}, "aiv": {}}
    for e in tasks:
        lane = lane_of.get((e.get("pid"), e.get("tid")))
        # `_pipe` answers "aic", "aiv" or None, and the lookup is by `.get` so a
        # lane on neither pipe -- or a task the trace named oddly -- is skipped
        # rather than raising in the middle of a measurement.
        lanes = by_pipe.get(_pipe(lane))
        if lanes is not None:
            lanes.setdefault(lane, []).append((e.get("ts"), e.get("dur")))
    return by_pipe


def _wall_span(by_pipe):
    """Wall-clock span over every on-core task, or None when there are none."""
    on_cores = []
    for pipe in by_pipe.values():
        for v in pipe.values():
            on_cores.extend(v)
    if not on_cores:
        return None
    return max(ts + d for ts, d in on_cores) - min(ts for ts, _ in on_cores)


def _merged_frac(tasks):
    """How many tasks carry a subGraphCount above 1.

    subGraphCount == 1 everywhere means no merging was applied at all. None when
    no task carries the hint.
    """
    merged = total = 0
    for e in tasks:
        hint = (e.get("args") or {}).get("hashOrder-hint") or ""
        m = _SUBGRAPH_RE.search(hint)
        if m:
            total += 1
            if int(m.group(1)) > 1:
                merged += 1
    return round(merged / total, 4) if total else None


def features(trace_path):
    """Read `merged_swimlane.json` and return a flat feature dict, or None.

    None on anything unreadable, on purpose: a symptom that cannot be computed
    must stay quiet rather than fire on a fabricated number. The same rule the
    source-derived occupancy symptoms already follow.
    """
    try:
        with open(trace_path, encoding="utf-8", errors="replace") as fh:
            doc = json.load(fh)
        events = doc["traceEvents"]
    except (OSError, ValueError, KeyError, TypeError):
        return None

    lane_of = _lane_names(events)
    tasks = [e for e in events if e.get("ph") == "X" and e.get("dur") is not None]
    if not tasks:
        return None

    by_pipe = _tasks_by_pipe(tasks, lane_of)
    span = _wall_span(by_pipe)
    if span is None:
        return None

    out = {"span_us": round(span, 2)}
    for name, lanes in by_pipe.items():
        for k, v in _pipe_features(lanes, span).items():
            out[f"{name}_{k}"] = v

    # Both pipes present: how lopsided is the split? A kernel whose Cube sits
    # idle while Vector saturates is a different problem from one where both
    # idle, and no single scalar separates them.
    ba, bv = out.get("aic_busy_frac"), out.get("aiv_busy_frac")
    if ba is not None and bv is not None:
        out["pipe_imbalance"] = round(abs(ba - bv), 4)

    merged = _merged_frac(tasks)
    if merged is not None:
        out["subgraph_merged_frac"] = merged

    out.update(_cross_pipe_tensors(tasks, lane_of))
    usage = pipe_usage(find_pipe_usage(trace_path))
    if usage:
        out.update(usage)
        out["pipe_usage_source"] = "pipe_usage.csv"
    return out


def subgraph_counts(trace_path):
    """Largest homogeneous subgraph group each merge pass has to work with.

    A merge granularity of N groups N subgraphs into one, so N subgraphs is the
    ceiling: `{-1: 16}` on a pass whose groups hold 8 cannot merge sixteen of
    anything. The bound is per PASS, and the passes do not share it -- a trace
    routinely reports very different group sizes for `cubeMergeInfo` and for
    `l1ReuseInfo` on the same kernel.

    It can also say "not applicable": when every `cubeMergeInfo` group holds
    exactly 1, one subgraph cannot be merged with anything. That collapses the
    domain to [1] and saves the device trial that would have found it out.

    Returns {} when the trace cannot be read, which leaves the documented
    ladder intact rather than inventing a bound.
    """
    try:
        with open(trace_path, encoding="utf-8", errors="replace") as fh:
            events = json.load(fh)["traceEvents"]
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    out = {}
    for e in events:
        hint = (e.get("args") or {}).get("hashOrder-hint") or ""
        for kind, _order, n in _MERGE_RE.findall(hint):
            knob = _MERGE_KNOB.get(kind)
            if knob:
                out[knob] = max(out.get(knob, 0), int(n))
    return out


# The pipes `pipe_usage.csv` reports, in the writer's own order. The file is
# produced by pypto's `draw_swim_lane.py` (`calculate_pipe_usage`) next to the
# trace, under `--gen_exe_topo_json`. PANKO reads it rather than deriving the
# same numbers a second time; absent, the keys are simply missing.
PIPES = ("MTE_IN", "MTE1", "MTE_OUT", "CUBE", "VECTOR_ALU")


def pipe_usage(path):
    """`{pipe_<name>_frac: fraction}` from pypto's `pipe_usage.csv`, or `{}`.

    Its "Total Pipe Usage" block carries one average usage per pipe over the
    cores that ran. That is a DIFFERENT quantity from `busy_frac` and a more
    useful one: `busy_frac` says the cores were occupied, this says by WHAT. An
    MTE_IN-bound kernel and a VECTOR_ALU-bound one are both busy at the same
    fraction and want opposite actions.

    Percentages become fractions, to read like `busy_frac`. A non-finite value
    is dropped rather than carried: the writer divides by a core count and by a
    span, and a symptom must stay quiet rather than fire on `inf`.
    """
    rows = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                _read_pipe_row(line, rows)
    except OSError:
        return {}
    return rows


def _read_pipe_row(line, rows):
    """One `Pipe, AverageTime, TotalExecuteTime, AverageUsage` row into `rows`.

    The per-core block below it carries twelve columns and the header carries a
    pipe name this does not know, so both are skipped by the same two tests.
    """
    parts = [c.strip() for c in line.split(",")]
    if len(parts) != 4 or parts[0] not in PIPES or not parts[3].endswith("%"):
        return
    try:
        pct = float(parts[3][:-1])
    except ValueError:
        return
    if math.isfinite(pct) and pct >= 0:
        rows[f"pipe_{parts[0].lower()}_frac"] = round(pct / 100.0, 4)


def find_pipe_usage(trace_path):
    """`pipe_usage.csv` beside the trace, or "" -- the writer puts both in the
    same output directory.
    """
    if not trace_path:
        return ""
    path = os.path.join(os.path.dirname(trace_path), "pipe_usage.csv")
    return path if os.path.isfile(path) else ""


def find_trace(*roots):
    """Newest `output/output_*/merged_swimlane.json` under any of `roots`.

    The campaign runs with the repo root as cwd, so traces land in
    `./output/output_*`, NOT under `<op_dir>/output` -- a run analysed by the
    op_dir alone finds nothing. Both are searched, newest wins.
    """
    best, best_mtime = None, -1.0
    for root in roots:
        if not root:
            continue
        base = os.path.join(root, "output")
        try:
            entries = os.listdir(base)
        except OSError:
            continue
        for name in entries:
            if not name.startswith("output_"):
                continue
            path = os.path.join(base, name, "merged_swimlane.json")
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if mtime > best_mtime:
                best, best_mtime = path, mtime
    return best


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trace", help="merged_swimlane.json; default: newest found")
    ap.add_argument("--op-dir", default=None)
    a = ap.parse_args()
    path = a.trace or find_trace(a.op_dir, os.getcwd())
    jsonio.emit({"trace": path, "features": features(path) if path else None})


if __name__ == "__main__":
    main()
