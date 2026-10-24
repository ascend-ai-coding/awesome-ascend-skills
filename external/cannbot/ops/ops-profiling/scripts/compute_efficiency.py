# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Turn a measured op duration into MBU / MFU / bound / theoretical ceiling.

This is the analysis-layer piece the ops-perf-optimize-workflow workflow lacks: it takes a
duration (already collected by collect_perf_application.py or msprof op) plus the
op's intrinsic cost (FLOP + bytes, from a registered cost model or CLI) and the
hardware peaks, and reports:

  MBU  = bytes / time / peak_bw          (bandwidth utilization)
  MFU  = flops / time / peak_flops       (compute utilization)
  ceiling_us = max(bytes/peak_bw, flops/peak_flops)   (roofline wall)
  bound      = Memory-Bound | Compute-Bound  (from the cost roofline;
               refined by measured pipe ratios when --pipe-util is given)
  verdict    = a one-line pruning hint. GOAL-FIRST: when --target-us (or
               --baseline-us + --target-speedup) is given, the user's target is the
               primary judge -- GOAL-MET (stop, even below the ceiling) /
               GOAL-REACHABLE (target above the roofline ceiling -> gap is
               implementation, tune tiling/overlap) / GOAL-EXCEEDS-ROOFLINE (target
               below the ceiling -> tuning can't reach it, must cut bytes/FLOPs).
               MFU/MBU only decide, once the target is unmet, where to push. Without a
               target it falls back to a goal-agnostic roofline hint.

Duration source (one of):
  --duration-us <float>
  --perf-csv <performance.csv> [--case-id <id>]     (reads median total_us)

Cost source -- operator-agnostic, no per-op registration:
  bytes (MBU numerator), one of:
    --bytes-from-profiling  MEASURED HBM traffic from --op-summary (main_mem bw * dur);
                            correct for in-place/cache/L2-resident ops. Needs the op_summary
                            collected with --aic-metrics=Memory.
    --io '<json>'          {"inputs":[{"shape":[..],"dtype":".."},...],"outputs":[...]}
                            shape-ESTIMATED bytes; over-counts in-place cache / L2 traffic.
    --case <jsonl> [--case-name <name>]   same, from a project case descriptor
    --bytes <B>            explicit
  flops (MFU numerator, OPTIONAL -- shapes alone cannot give it):
    --flops-from-profiling --op-summary <csv> [--op-name <name>] [--flop-field aic_cube_fops] [--flop-scale 1.0]
    --flops <F>            explicit
    (omit both -> report MBU only; the binding metric for memory-bound ops)

  (--op-summary is shared: --bytes-from-profiling and --flops-from-profiling both read it.)

Peaks (CLI > env PEAK_HBM_BW/PEAK_CUBE_FLOPS > default). Defaults are the
Ascend950PR-PCIE figures from testkit (1.6 TB/s, 378.47 TFLOPS); ALWAYS pass the
peaks for the SoC actually under test, and the report records which were used.
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cost_models  # noqa: E402

DEFAULT_PEAK_HBM_BW = float(os.environ.get("PEAK_HBM_BW", 1.6e12))  # bytes/s
DEFAULT_PEAK_CUBE_FLOPS = float(os.environ.get("PEAK_CUBE_FLOPS", 378.47e12))  # FLOP/s
# Vector-unit peak has no safe default (varies by SoC); require it for vector ops.
DEFAULT_PEAK_VEC_FLOPS = float(os.environ["PEAK_VEC_FLOPS"]) if os.environ.get("PEAK_VEC_FLOPS") else None


def _read_perf_csv(path, case_id):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"empty perf csv: {path}")
    if case_id is not None:
        rows = [r for r in rows if r.get("case_id") == case_id]
        if not rows:
            raise ValueError(f"case_id '{case_id}' not found in {path}")
    if len(rows) > 1:
        raise ValueError(
            f"{path} has {len(rows)} rows; pass --case-id to select one (available: {[r.get('case_id') for r in rows]})"
        )
    return float(rows[0]["total_us"])


def _read_case_io(path, case_name=None):
    """Extract {'inputs':..., 'outputs':...} from a project case JSONL.

    Each line is one case with ``inputs`` (list of tensor/scalar descriptors) and,
    when present, ``outputs``. With many cases, ``case_name`` selects one by matching
    any of name/op_name/op/case_id.
    """
    with open(path, "r", encoding="utf-8") as fh:
        cases = [json.loads(ln) for ln in fh if ln.strip()]
    if not cases:
        raise ValueError(f"no cases in {path}")
    if case_name is not None:
        keys = ("name", "op_name", "op", "case_id")
        sel = [c for c in cases if any(str(c.get(k)) == case_name for k in keys)]
        if not sel:
            raise ValueError(f"case '{case_name}' not found in {path}")
        case = sel[0]
    elif len(cases) > 1:
        raise ValueError(f"{path} has {len(cases)} cases; pass --case-name to select one")
    else:
        case = cases[0]
    return {"inputs": case.get("inputs", []), "outputs": case.get("outputs", [])}


def _pipe_bound(pipe_util_path):
    """Classify bound from measured PipeUtilization (testkit bound_type logic).

    Accepts an OPPROF dir or a PipeUtilization.csv path. Averages *_ratio columns
    across cores. Returns (bound_str, {pipe: ratio}) or (None, {}) if unavailable.
    """
    p = pipe_util_path
    if os.path.isdir(p):
        p = os.path.join(p, "PipeUtilization.csv")
    if not os.path.isfile(p):
        return None, {}
    with open(p, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None, {}

    def avg(*keys):
        vals = []
        for r in rows:
            for k in keys:
                if k in r and r[k] not in ("", None):
                    try:
                        vals.append(float(r[k]))
                        break
                    except ValueError:
                        pass
        return sum(vals) / len(vals) if vals else 0.0

    # op-mode PipeUtilization uses aic_cube_ratio; application op_summary uses aic_mac_ratio.
    pipes = {
        "Cube": avg("aic_cube_ratio", "aic_mac_ratio"),
        "Vector": avg("aiv_vec_ratio"),
        "MTE2": avg("aic_mte2_ratio"),
        "MTE3": avg("aiv_mte3_ratio", "aic_mte3_ratio"),
    }
    name = max(pipes, key=pipes.get)
    ratio = pipes[name]
    if ratio < 0.8:
        return "Latency Bound", pipes
    if name in ("Cube", "Vector"):
        return f"Compute Bound ({name})", pipes
    return f"Memory Bound ({name})", pipes


def analyze(
    duration_us,
    flops,
    hbm_bytes,
    peak_bw,
    peak_flops,
    pipe_util=None,
    label="",
    compute_unit="cube",
    mte2_us=None,
    target_us=None,
):
    # flops is optional: without a measured/explicit FLOP count we report MBU only
    # (peak_flops is then irrelevant and may be None).
    have_flops = flops is not None
    if duration_us <= 0 or peak_bw <= 0 or (have_flops and (peak_flops is None or peak_flops <= 0)):
        raise ValueError(
            f"duration_us / peak_bw must be > 0, and peak_flops > 0 when flops is given "
            f"(got duration_us={duration_us}, peak_bw={peak_bw}, peak_flops={peak_flops})"
        )
    sec = duration_us * 1e-6
    achieved_bw = hbm_bytes / sec
    mbu = achieved_bw / peak_bw
    t_mem = hbm_bytes / peak_bw

    if have_flops:
        achieved_flops = flops / sec
        mfu = achieved_flops / peak_flops
        t_compute = flops / peak_flops
        ceiling_us = max(t_mem, t_compute) * 1e6
        cost_bound = "Memory-Bound" if t_mem >= t_compute else "Compute-Bound"
    else:
        achieved_flops = mfu = t_compute = None
        # No FLOP -> the only roofline wall we can draw is the bandwidth one.
        ceiling_us = t_mem * 1e6
        cost_bound = "Memory-Bound (bytes-only; no FLOP measured)"

    measured_bound, pipes = (None, {})
    if pipe_util:
        measured_bound, pipes = _pipe_bound(pipe_util)

    # MTE2 (memory-load pipe) utilization vs peak. Detects "weight load already at
    # the bandwidth floor" even when wall-MBU is low because the pipeline is not
    # overlapped (the wall-MTE2 gap is bubble, not extra bytes). If --mte2-us is not
    # given, approximate its active time from the measured MTE2 pipe ratio.
    if mte2_us is None and pipes.get("MTE2"):
        mte2_us = pipes["MTE2"] * duration_us
    mte2_util = (hbm_bytes / (mte2_us * 1e-6) / peak_bw) if mte2_us else None

    headroom_us = duration_us - ceiling_us
    headroom_pct = headroom_us / duration_us if duration_us else 0.0

    # --- roofline lever hint (goal-agnostic): IF you keep optimizing, where is the
    #     headroom and which lever moves it. This is SUPPORT, not the primary verdict. ---
    is_mem = cost_bound.startswith("Memory-Bound")
    if is_mem and mbu >= 0.90:
        roofline_hint = (
            "memory-bound at >=90% MBU -- non-quant pipeline/tiling reshuffles "
            "cannot move the wall; the only lever is fewer bytes (quant / weight "
            "format / dtype)."
        )
    elif is_mem and mte2_util is not None and mte2_util >= 0.90:
        roofline_hint = (
            f"load pipe MTE2 already ~{mte2_util * 100:.0f}% of bandwidth peak while "
            f"wall-MBU is only {mbu * 100:.0f}% -- the {headroom_us:.1f}us gap is pipeline "
            f"non-overlap, not extra bytes. Closing it usually needs RE-READING weights "
            f"(net loss); treat as near-ceiling unless a no-re-read overlap is proven, "
            f"else cut bytes (quant / format / dtype)."
        )
    elif is_mem and not have_flops:
        roofline_hint = (
            f"MBU={mbu * 100:.0f}% (bytes-only: no FLOP measured, MFU/compute-bound "
            f"cannot be judged). For a memory-bound op MBU is the binding metric."
        )
    elif is_mem:
        roofline_hint = (
            f"memory-bound, MBU={mbu * 100:.0f}%; headroom toward the bandwidth "
            f"ceiling ({ceiling_us:.1f}us) -- overlap/prefetch tuning may help."
        )
    elif mfu >= 0.70:
        roofline_hint = (
            "compute-bound at >=70% MFU -- little headroom without reducing FLOPs (algorithm / lower precision)."
        )
    else:
        roofline_hint = (
            f"compute-bound, MFU={mfu * 100:.0f}%; below the compute ceiling -- "
            f"tiling / pipeline / occupancy tuning is the lever."
        )

    # --- goal-first verdict: the USER'S TARGET is the primary judge; MFU/MBU/roofline
    #     only decide, once the target is unmet, whether it's even physically reachable
    #     and where to push. No target -> fall back to the goal-agnostic roofline hint. ---
    goal_status = None
    if target_us is not None and target_us > 0:
        if duration_us <= target_us:
            goal_status = "GOAL-MET"
            verdict = (
                f"✅ GOAL-MET: {duration_us:.1f}us <= target {target_us:.1f}us. Stop -- "
                f"target reached regardless of MFU/MBU headroom. (roofline: {roofline_hint})"
            )
        elif ceiling_us <= target_us:
            goal_status = "GOAL-REACHABLE"
            verdict = (
                f"🟡 GOAL-REACHABLE: {duration_us:.1f}us > target {target_us:.1f}us "
                f"(gap {duration_us - target_us:.1f}us), but the roofline ceiling "
                f"{ceiling_us:.1f}us <= target -- physically reachable, the gap is "
                f"implementation. Lever: {roofline_hint}"
            )
        else:
            goal_status = "GOAL-EXCEEDS-ROOFLINE"
            verdict = (
                f"🔴 GOAL-EXCEEDS-ROOFLINE: target {target_us:.1f}us is below the roofline "
                f"ceiling {ceiling_us:.1f}us -- conventional tiling/pipeline tuning cannot "
                f"reach it. Must lower the ceiling: cut bytes (quant/format/dtype) or FLOPs "
                f"(algorithm/precision). Current roofline: {roofline_hint}"
            )
    else:
        verdict = roofline_hint

    return {
        "label": label,
        "compute_unit": compute_unit,
        "duration_us": round(duration_us, 4),
        "flops": flops,
        "hbm_bytes": hbm_bytes,
        "peak_hbm_bw_TBps": round(peak_bw / 1e12, 4),
        "peak_compute_flops_TFLOPS": round(peak_flops / 1e12, 4) if (have_flops and peak_flops) else None,
        "achieved_bw_TBps": round(achieved_bw / 1e12, 4),
        "achieved_flops_TFLOPS": round(achieved_flops / 1e12, 4) if have_flops else None,
        "MBU": round(mbu, 4),
        "MFU": round(mfu, 4) if have_flops else None,
        "mte2_util": round(mte2_util, 4) if mte2_util is not None else None,
        "cost_bound": cost_bound,
        "measured_bound": measured_bound,
        "pipe_ratios": {k: round(v, 4) for k, v in pipes.items()} if pipes else None,
        "ceiling_us": round(ceiling_us, 4),
        "headroom_us": round(headroom_us, 4),
        "headroom_pct": round(headroom_pct, 4),
        "target_us": round(target_us, 4) if target_us else None,
        "goal_status": goal_status,
        "verdict": verdict,
    }


def _human(r):
    have_flops = r["flops"] is not None
    peaks = f"  peaks             HBM {r['peak_hbm_bw_TBps']} TB/s"
    if have_flops:
        peaks += f" | {r['compute_unit']} {r['peak_compute_flops_TFLOPS']} TFLOPS"
    flop_str = f"{r['flops'] / 1e9:.3f} GFLOP" if have_flops else "n/a (no FLOP measured)"
    lines = [
        f"== efficiency: {r['label'] or '(op)'} ==",
        f"  duration          {r['duration_us']:.2f} us",
        peaks,
        f"  FLOP / bytes      {flop_str} / {r['hbm_bytes'] / 1e6:.1f} MB",
    ]
    if have_flops:
        lines.append(
            f"  MFU ({r['compute_unit']})       {r['MFU'] * 100:.1f}%  ({r['achieved_flops_TFLOPS']:.1f} TFLOPS)"
        )
    lines += [
        f"  MBU               {r['MBU'] * 100:.1f}%  ({r['achieved_bw_TBps']:.2f} TB/s)",
        f"  bound (cost)      {r['cost_bound']}",
    ]
    if r.get("mte2_util") is not None:
        lines.append(f"  MTE2 util         {r['mte2_util'] * 100:.1f}%  (载入管道对内存峰值)")
    if r["measured_bound"]:
        lines.append(f"  bound (measured)  {r['measured_bound']}  {r['pipe_ratios']}")
    lines.append(
        f"  ceiling           {r['ceiling_us']:.2f} us  "
        f"(headroom {r['headroom_us']:.2f} us / {r['headroom_pct'] * 100:.0f}%)"
    )
    if r.get("target_us") is not None:
        lines.append(f"  target            {r['target_us']:.2f} us  [{r['goal_status']}]")
    lines.append(f"  VERDICT           {r['verdict']}")
    return "\n".join(lines)


def parse_args():
    ap = argparse.ArgumentParser(description="MBU/MFU/bound/ceiling from a measured op duration")
    d = ap.add_mutually_exclusive_group(required=True)
    d.add_argument("--duration-us", type=float)
    d.add_argument("--perf-csv", help="performance.csv from collect_perf_application.py")
    ap.add_argument("--case-id", help="select a row when --perf-csv has many")

    # bytes source (MBU numerator) -- operator-agnostic, no per-op code.
    ap.add_argument(
        "--bytes-from-profiling",
        action="store_true",
        help="read MEASURED HBM bytes from --op-summary (main_mem bw * duration); "
        "correct for in-place/cache/L2 ops. Needs --aic-metrics=Memory profiling.",
    )
    ap.add_argument("--io", help='JSON I/O: {"inputs":[{"shape":[..],"dtype":".."}],"outputs":[...]} (shape-estimated)')
    ap.add_argument("--case", help="project case JSONL to derive I/O bytes from (shape-estimated)")
    ap.add_argument("--case-name", help="select a case when --case has many")
    ap.add_argument("--bytes", type=float, dest="bytes_", help="explicit byte count")
    ap.add_argument(
        "--mem-bytes-scale",
        type=float,
        default=1.0,
        help="manual multiplier on profiling-measured bytes (parity with --flop-scale)",
    )

    # shared profiling source: --op-summary feeds both --bytes-from-profiling and
    # --flops-from-profiling; --op-name selects the row in it.
    ap.add_argument("--op-summary", help="profiler op_summary CSV (source for measured bytes and/or FLOP)")
    ap.add_argument("--op-name", help="Op Name to select in --op-summary (else the sole compute row)")

    # flops source (MFU numerator) -- OPTIONAL; cannot be inferred from shapes.
    ap.add_argument(
        "--flops-from-profiling", action="store_true", help="read measured FLOP from --op-summary (--flop-field)"
    )
    ap.add_argument(
        "--flop-field", default="aic_cube_fops", help="op_summary column holding the FLOP count (default aic_cube_fops)"
    )
    ap.add_argument(
        "--flop-scale",
        type=float,
        default=1.0,
        help="multiplier on the FLOP field (validate MAC-vs-FLOP semantics on your SoC)",
    )
    ap.add_argument("--flops", type=float, help="explicit FLOP count")

    ap.add_argument("--peak-hbm-bw", type=float, help="TB/s (CLI > env > default 1.6)")
    ap.add_argument("--peak-cube-flops", type=float, help="cube TFLOPS (CLI > env > default 378.47)")
    ap.add_argument(
        "--peak-vec-flops", type=float, help="vector-unit TFLOPS; required when the op runs on the vector unit"
    )
    ap.add_argument(
        "--compute-unit",
        choices=["cube", "vector"],
        default="cube",
        help="which unit does the FLOPs (default cube); selects the compute peak",
    )
    # user's optimization goal -- the PRIMARY judge; roofline becomes support (see analyze()).
    ap.add_argument("--target-us", type=float, help="target latency (us); makes the verdict goal-first")
    ap.add_argument(
        "--baseline-us", type=float, help="baseline latency (us); with --target-speedup derives --target-us"
    )
    ap.add_argument("--target-speedup", type=float, help="target speedup x over --baseline-us (target = baseline / x)")

    ap.add_argument("--pipe-util", help="OPPROF dir or PipeUtilization.csv to refine bound")
    ap.add_argument(
        "--mte2-us",
        type=float,
        help="MTE2 (load pipe) active time in us; detects weight-load-at-bandwidth-floor "
        "even when wall-MBU is low. If omitted, derived from --pipe-util MTE2 ratio.",
    )
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    return ap.parse_args()


def main():
    a = parse_args()
    duration_us = a.duration_us if a.duration_us is not None else _read_perf_csv(a.perf_csv, a.case_id)
    unit = a.compute_unit

    # FLOP (optional): explicit > profiling (opt-in) > none.
    if a.flops is not None:
        flops = a.flops
    elif a.flops_from_profiling:
        if not a.op_summary:
            raise SystemExit("--flops-from-profiling needs --op-summary <csv>")
        flops = cost_models.flops_from_profiling(a.op_summary, a.op_name, a.flop_field, a.flop_scale)
    else:
        flops = None

    # bytes: explicit > profiling-measured > shape-estimated (--io/--case). All op-agnostic.
    if a.bytes_ is not None:
        hbm_bytes, label = a.bytes_, "explicit"
    elif a.bytes_from_profiling:
        if not a.op_summary:
            raise SystemExit("--bytes-from-profiling needs --op-summary <csv>")
        hbm_bytes = cost_models.bytes_from_profiling(a.op_summary, a.op_name, scale=a.mem_bytes_scale)
        label = f"{a.op_name or 'op'} (bytes measured)"
    elif a.io or a.case:
        io = json.loads(a.io) if a.io else _read_case_io(a.case, a.case_name)
        c = cost_models.compute_cost(io, flops=flops, unit=unit)
        hbm_bytes, label = c.total_bytes, c.label
    else:
        raise SystemExit("provide op bytes via --bytes-from-profiling, --io '<json>', --case <jsonl>, or --bytes")

    peak_bw = (a.peak_hbm_bw * 1e12) if a.peak_hbm_bw is not None else DEFAULT_PEAK_HBM_BW
    if flops is None:
        peak_flops = None  # MBU-only; no compute peak needed
    elif unit == "vector":
        peak_flops = (a.peak_vec_flops * 1e12) if a.peak_vec_flops is not None else DEFAULT_PEAK_VEC_FLOPS
        if not peak_flops:
            raise SystemExit(
                "vector-unit op with FLOP given: pass --peak-vec-flops <TFLOPS> (or env PEAK_VEC_FLOPS). "
                "MFU against the cube peak would be meaningless; MBU is usually the binding metric for pure-vector ops."
            )
    else:
        peak_flops = (a.peak_cube_flops * 1e12) if a.peak_cube_flops is not None else DEFAULT_PEAK_CUBE_FLOPS

    if duration_us <= 0:
        raise SystemExit(f"duration must be > 0 (got {duration_us}); check --duration-us or the perf csv total_us")
    if peak_bw <= 0:
        raise SystemExit(f"peak_hbm_bw must be > 0 (got {peak_bw})")
    if flops is not None and (peak_flops is None or peak_flops <= 0):
        raise SystemExit(f"peak compute flops must be > 0 when flops is given (got {peak_flops})")

    target_us = a.target_us
    if target_us is None and a.baseline_us and a.target_speedup:
        target_us = a.baseline_us / a.target_speedup

    r = analyze(
        duration_us,
        flops,
        hbm_bytes,
        peak_bw,
        peak_flops,
        pipe_util=a.pipe_util,
        label=label,
        compute_unit=unit,
        mte2_us=a.mte2_us,
        target_us=target_us,
    )
    print(json.dumps(r, indent=2) if a.json else _human(r))


if __name__ == "__main__":
    main()
