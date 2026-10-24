#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Analyze cannsim trace_core*.json to extract pipeline timing and determine bound type.

Usage:
    python3 pipeline_bound_analyze.py <trace_file> [-o output.md] [--stdout]

Parses Chrome Trace JSON from cannsim, computes per-pipeline-unit busy time
(with overlap merging), bubble time, and determines the bottleneck (bound) type.

Target pipelines: MTE2, MTE1, CUBE, SCALAR, SCALARLDST, FIXP, MTE3, PUSHQ
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

TARGET_UNITS = ["MTE2", "MTE1", "CUBE", "SCALAR", "SCALARLDST", "FIXP", "MTE3", "PUSHQ"]

# SCALAR / SCALARLDST: 不计算、不输出气泡时间（报告气泡列为 —）。
NO_BUBBLE_UNITS = frozenset({"SCALAR", "SCALARLDST"})

BOUND_THRESHOLD = 0.80
# 「占比最大」分支：该单元 busy / case 须严格大于此比例，才与 MTE2/CUBE BOUND 的「最大占比」条件同时成立
MAX_SHARE_THRESHOLD = 0.70


def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merge overlapping or touching intervals; return sorted disjoint intervals."""
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged: List[Tuple[float, float]] = [sorted_iv[0]]
    for s, e in sorted_iv[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def compute_busy_time(intervals: List[Tuple[float, float]]) -> float:
    """Total busy time after merging overlapping intervals."""
    return sum(e - s for s, e in merge_intervals(intervals))


def compute_bubble_time(intervals: List[Tuple[float, float]]) -> float:
    """Total gap (bubble) time between merged intervals."""
    merged = merge_intervals(intervals)
    if len(merged) < 2:
        return 0.0
    bubble = 0.0
    for i in range(1, len(merged)):
        gap = merged[i][0] - merged[i - 1][1]
        if gap > 0:
            bubble += gap
    return bubble


def parse_trace(trace_path: str) -> Dict:
    """Parse trace JSON and return per-unit intervals and global time range.

    Returns:
        {
            "intervals": { unit_name: [(start, end), ...], ... },
            "time_range": (global_min, global_max),
        }
    """
    with open(trace_path, "r", encoding="utf-8") as f:
        events = json.load(f)

    if isinstance(events, dict):
        events = events.get("traceEvents", [])

    pid_to_unit: Dict[int, str] = {}
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "process_name":
            raw_name = e.get("args", {}).get("name", "")
            parts = raw_name.split("_", 1)
            unit_name = parts[1] if len(parts) > 1 else raw_name
            pid_to_unit[e["pid"]] = unit_name

    intervals_by_unit: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    all_times: List[float] = []

    for e in events:
        if e.get("ph") == "X" and "pid" in e:
            pid = e["pid"]
            unit = pid_to_unit.get(pid)
            if unit is None:
                continue
            ts = e.get("ts", 0)
            dur = e.get("dur", 0)
            start = ts
            end = ts + dur
            intervals_by_unit[unit].append((start, end))
            all_times.append(start)
            all_times.append(end)

    time_range = (min(all_times), max(all_times)) if all_times else (0, 0)

    mte2_ivs = intervals_by_unit.get("MTE2", [])
    first_mte2_start = min(s for s, _ in mte2_ivs) if mte2_ivs else None

    return {
        "intervals": dict(intervals_by_unit),
        "time_range": time_range,
        "first_mte2_start": first_mte2_start,
    }


def format_cycles(cycles: float) -> str:
    """Human-readable cycle count."""
    if cycles <= 0:
        return "0"
    if cycles >= 1_000_000:
        return f"{cycles / 1_000_000:.2f}M"
    if cycles >= 10_000:
        return f"{cycles / 1_000:.1f}k"
    if cycles >= 1.0:
        return f"{cycles:.0f}"
    return f"{cycles:.2f}"


def primary_max_share_unit(unit_busy: Dict[str, float], case_time: float) -> str:
    """First unit in TARGET_UNITS order that achieves max busy / case_time."""
    if case_time <= 0:
        return ""
    best_u = ""
    best_pct = -1.0
    for u in TARGET_UNITS:
        p = unit_busy.get(u, 0.0) / case_time
        if p > best_pct:
            best_pct = p
            best_u = u
    return best_u


def determine_bound(unit_busy: Dict[str, float], case_time: float) -> Tuple[str, List[str]]:
    """Determine bound type based on the rules.

    Priority (high → low): MTE2 → CUBE → VEC → FIXP → MTE3 → SCALAR (lowest) → 无 bound.

    MTE2/CUBE：>80% 或与「8 条中占比最大」同时成立时需该单元占比 >70%。
    若均不匹配则为「无 bound」（不再使用 MIXED）。

    Returns (bound_type, [reasons]).
    """
    if case_time <= 0:
        return "UNKNOWN", []

    get = lambda name: unit_busy.get(name, 0.0)
    pct = lambda name: get(name) / case_time

    primary = primary_max_share_unit(unit_busy, case_time)

    mte2_high = pct("MTE2") > BOUND_THRESHOLD
    mte2_max_share = primary == "MTE2" and pct("MTE2") > MAX_SHARE_THRESHOLD
    if mte2_high or mte2_max_share:
        reasons = []
        if mte2_high:
            reasons.append(f"MTE2 占比 {pct('MTE2') * 100:.1f}% > 80%")
        if mte2_max_share:
            reasons.append(f"MTE2 指令执行时间占比在 8 条流水线中最大，且占比 {pct('MTE2') * 100:.1f}% > 70%")
        return "MTE2 BOUND", reasons

    cube_high = pct("CUBE") > BOUND_THRESHOLD
    cube_max_share = primary == "CUBE" and pct("CUBE") > MAX_SHARE_THRESHOLD
    if cube_high or cube_max_share:
        reasons = []
        if cube_high:
            reasons.append(f"CUBE 占比 {pct('CUBE') * 100:.1f}% > 80%")
        if cube_max_share:
            reasons.append(f"CUBE 指令执行时间占比在 8 条流水线中最大，且占比 {pct('CUBE') * 100:.1f}% > 70%")
        return "CUBE BOUND", reasons

    if pct("PUSHQ") > BOUND_THRESHOLD:
        return "VEC BOUND", [f"PUSHQ 占比 {pct('PUSHQ') * 100:.1f}% > 80%"]

    if pct("FIXP") > BOUND_THRESHOLD:
        return "FIXP BOUND", [f"FIXP 占比 {pct('FIXP') * 100:.1f}% > 80%"]

    if pct("MTE3") > BOUND_THRESHOLD:
        return "MTE3 BOUND", [f"MTE3 占比 {pct('MTE3') * 100:.1f}% > 80%"]

    scalar_over = pct("SCALAR") > BOUND_THRESHOLD
    scalarldst_over = pct("SCALARLDST") > BOUND_THRESHOLD

    if scalar_over or scalarldst_over:
        reasons = []
        if scalar_over:
            reasons.append(f"SCALAR 占比 {pct('SCALAR') * 100:.1f}% > 80%")
        if scalarldst_over:
            reasons.append(f"SCALARLDST 占比 {pct('SCALARLDST') * 100:.1f}% > 80%")
        return "SCALAR BOUND", reasons

    return "无 bound", ["以上 bound 判定条件均不满足"]


def generate_report(trace_path: str) -> str:
    """Generate the full Markdown analysis report."""
    data = parse_trace(trace_path)
    intervals = data["intervals"]
    time_range = data["time_range"]
    first_mte2_start = data.get("first_mte2_start")
    case_time = time_range[1] - time_range[0]
    case_start = time_range[0]

    unit_busy: Dict[str, float] = {}
    unit_bubble: Dict[str, float] = {}

    for unit in TARGET_UNITS:
        ivs = intervals.get(unit, [])
        unit_busy[unit] = compute_busy_time(ivs)
        if unit in NO_BUBBLE_UNITS:
            unit_bubble[unit] = 0.0  # 不统计气泡，表中显示 —
        else:
            unit_bubble[unit] = compute_bubble_time(ivs)

    bound_type, bound_reasons = determine_bound(unit_busy, case_time)

    lines = []
    lines.append("# 流水图 Bound 分析报告")
    lines.append("")
    lines.append(f"**Trace 文件**: `{trace_path}`")
    lines.append("")

    # --- Section 1: per-pipeline summary ---
    lines.append("## 各流水线指令总时间与气泡时间")
    lines.append("")
    lines.append("SCALAR、SCALARLDST 不统计气泡时间（表中为 —）。")
    lines.append("")
    lines.append("| 流水线 | 指令执行总时间 (cycles) | 占比 | 气泡时间 (cycles) |")
    lines.append("|--------|------------------------|------|-------------------|")

    for unit in TARGET_UNITS:
        busy = unit_busy[unit]
        bubble = unit_bubble[unit]
        pct_val = (busy / case_time * 100) if case_time > 0 else 0
        if unit in NO_BUBBLE_UNITS:
            bubble_cell = "—"
        else:
            bubble_cell = format_cycles(bubble)
        lines.append(f"| {unit} | {format_cycles(busy)} | {pct_val:.1f}% | {bubble_cell} |")

    lines.append("")
    lines.append(
        f"**Case 执行总时间**: {format_cycles(case_time)} cycles "
        f"(trace 上全局时间范围: {format_cycles(case_start)} ~ {format_cycles(time_range[1])})"
    )
    if first_mte2_start is not None:
        head_cycles = first_mte2_start - case_start
        head_pct = (head_cycles / case_time * 100) if case_time > 0 else 0
        lines.append(
            f"**头开销时间**: {format_cycles(head_cycles)} cycles "
            f"（case 起点 {format_cycles(case_start)} 到首条 MTE2 指令起点 {format_cycles(first_mte2_start)}，"
            f"约占 case {head_pct:.1f}%）"
        )
    else:
        lines.append("**头开销时间**: 无 MTE2 事件，无法计算（首条 MTE2 起点不存在）")
    lines.append("")

    # --- Section 2: bound result ---
    lines.append("## Bound 分析结果")
    lines.append("")
    lines.append(f"**判定结果: {bound_type}**")
    lines.append("")
    if bound_reasons:
        lines.append("判定依据：")
        for r in bound_reasons:
            lines.append(f"- {r}")
        lines.append("")

    lines.append("### 判断规则参考（优先级从高到低）")
    lines.append("")
    lines.append("| 优先级 | Bound 类型 | 判断条件 |")
    lines.append("|--------|-----------|---------|")
    lines.append("| 1 | MTE2 BOUND | MTE2 > 80%，或（MTE2 占比最大且 MTE2 > 70%） |")
    lines.append("| 2 | CUBE BOUND | CUBE > 80%，或（CUBE 占比最大且 CUBE > 70%） |")
    lines.append("| 3 | VEC BOUND | PUSHQ > 80% |")
    lines.append("| 4 | FIXP BOUND | FIXP > 80% |")
    lines.append("| 5 | MTE3 BOUND | MTE3 > 80% |")
    lines.append("| 6（最低） | SCALAR BOUND | SCALAR > 80%，或 SCALARLDST > 80% |")
    lines.append("| — | 无 bound | 以上条件均不满足 |")
    lines.append("")

    return "\n".join(lines)


def default_report_path(trace_path: str) -> str:
    """`<trace_stem>_pipeline_bound_report.md` next to the trace file."""
    base = os.path.splitext(os.path.abspath(trace_path))[0]
    return base + "_pipeline_bound_report.md"


def main():
    parser = argparse.ArgumentParser(description="Analyze cannsim trace JSON to determine pipeline bound type.")
    parser.add_argument(
        "trace_file",
        help="Path to trace_core*.json file from cannsim output",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="Output Markdown report path (default: <trace_stem>_pipeline_bound_report.md beside trace)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print report to stdout instead of writing a Markdown file",
    )
    args = parser.parse_args()

    trace_path = args.trace_file
    if not trace_path.lower().endswith(".json"):
        print(f"Warning: '{trace_path}' has no .json suffix; expecting Chrome Trace JSON.", file=sys.stderr)

    report = generate_report(trace_path)

    if args.stdout:
        print(report)
        return

    out_path = args.output or default_report_path(trace_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
        if not report.endswith("\n"):
            f.write("\n")
    print(f"Report written to: {out_path}")


if __name__ == "__main__":
    main()
