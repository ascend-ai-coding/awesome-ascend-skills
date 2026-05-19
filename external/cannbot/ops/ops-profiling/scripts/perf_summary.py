# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Generate a statistical summary from msprof op CSV output and archive to operator directory.

Usage:
    python3 perf_summary.py <OPPROF_dir> <ops_dir>

Example:
    python3 perf_summary.py /path/to/OPPROF_xxx ops/Softmax

The script:
1. Creates docs/perf/round_NNN/ under <ops_dir> (auto-incrementing)
2. Copies all CSV files from OPPROF directory to the archive
3. Generates summary.txt with min/avg/max statistics for ALL non-zero metrics
4. Does NOT make any judgments, thresholds, or optimization recommendations
"""

import argparse
import csv
import glob
import os
import re
import shutil
import statistics
import sys
from typing import Any, Dict, List, Optional, Tuple


def safe_float(val: Any, default: float = 0.0) -> float:
    if val is None or str(val).strip() in ("", "N/A", "NA"):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def stat_line(name: str, values: List[float], fmt: str = ".1f", unit: str = "", multiply: float = 1.0) -> Optional[str]:
    """Generate a min/avg/max stat line. Returns None if all values are zero."""
    scaled = [v * multiply for v in values]
    if all(abs(v) < 0.001 for v in scaled):
        return None
    mn, avg, mx = min(scaled), statistics.mean(scaled), max(scaled)
    u = f"    ({unit})" if unit else ""
    return f"{'  ' + name:<24s} {mn:>10{fmt}} {avg:>10{fmt}} {mx:>10{fmt}}{u}"


def detect_core_prefix(rows: List[Dict[str, str]]) -> str:
    """Detect whether this is a vector (aiv) or cube (aic) workload."""
    if not rows:
        return "aiv"
    row = rows[0]
    aiv_time = safe_float(row.get("aiv_time(us)"))
    aic_time = safe_float(row.get("aic_time(us)"))
    return "aic" if aic_time > aiv_time else "aiv"


def find_next_round(perf_dir: str) -> str:
    """Find the next round_NNN directory number."""
    if not os.path.exists(perf_dir):
        return os.path.join(perf_dir, "round_001")
    existing = [d for d in os.listdir(perf_dir) if re.match(r"round_\d+", d)]
    if not existing:
        return os.path.join(perf_dir, "round_001")
    nums = [int(re.search(r"\d+", d).group()) for d in existing]
    return os.path.join(perf_dir, f"round_{max(nums) + 1:03d}")


def archive_csvs(opprof_dir: str, round_dir: str) -> List[str]:
    """Copy all CSV files from OPPROF directory to archive. Returns list of copied files."""
    os.makedirs(round_dir, exist_ok=True)
    copied = []
    for csv_file in sorted(glob.glob(os.path.join(opprof_dir, "*.csv"))):
        dest = os.path.join(round_dir, os.path.basename(csv_file))
        shutil.copy2(csv_file, dest)
        copied.append(os.path.basename(csv_file))
    return copied


def generate_summary(opprof_dir: str, round_dir: str, ops_dir: str) -> str:
    """Generate summary.txt content from CSV data."""
    lines = []

    # === OpBasicInfo ===
    basic_rows = read_csv(os.path.join(opprof_dir, "OpBasicInfo.csv"))
    if basic_rows:
        b = basic_rows[0]
        op_name = b.get("Op Name", "unknown")
        op_type = b.get("Op Type", "unknown")
        duration = safe_float(b.get("Task Duration(us)"))
        block_dim = int(safe_float(b.get("Block Dim", "1")))
        cur_freq = safe_float(b.get("Current Freq"))
        rated_freq = safe_float(b.get("Rated Freq"))
        lines.append("=== 上板性能统计摘要 ===")
        lines.append(f"Op: {op_name} | Type: {op_type} | Duration: {duration}us | BlockDim: {block_dim} | Freq: {cur_freq:.0f}/{rated_freq:.0f}")
    else:
        lines.append("=== 上板性能统计摘要 ===")
        lines.append("(OpBasicInfo.csv 未找到)")
        op_name = "unknown"
        duration = 0
        block_dim = 0

    # === PipeUtilization ===
    pipe_rows = read_csv(os.path.join(opprof_dir, "PipeUtilization.csv"))
    if pipe_rows:
        n = len(pipe_rows)
        prefix = detect_core_prefix(pipe_rows)
        lines.append("")
        lines.append(f"--- PipeUtilization ({n} cores, prefix={prefix}) ---")
        lines.append(f"  {'':24s} {'min':>10s} {'avg':>10s} {'max':>10s}")

        # Core time
        core_times = [safe_float(r.get(f"{prefix}_time(us)")) for r in pipe_rows]
        line = stat_line(f"{prefix}_time(us)", core_times, ".2f")
        if line:
            lines.append(line)

        # Pipeline ratios (stored as 0-1 in CSV, display as %)
        ratio_fields = [
            ("vec_ratio%", f"{prefix}_vec_ratio"),
            ("scalar_ratio%", f"{prefix}_scalar_ratio"),
            ("mte2_ratio%", f"{prefix}_mte2_ratio"),
            ("mte3_ratio%", f"{prefix}_mte3_ratio"),
            ("icache_miss%", f"{prefix}_icache_miss_rate"),
        ]
        # Cube-specific
        if prefix == "aic":
            ratio_fields.extend([
                ("cube_ratio%", "aic_cube_ratio"),
                ("fixpipe_ratio%", "aic_fixpipe_ratio"),
                ("mte1_ratio%", "aic_mte1_ratio"),
            ])

        for display_name, field in ratio_fields:
            vals = [safe_float(r.get(field)) for r in pipe_rows]
            line = stat_line(display_name, vals, ".2f", multiply=100)
            if line:
                lines.append(line)

        # Active bandwidth
        bw_fields = [
            ("mte2_active_bw", f"{prefix}_mte2_active_bw(GB/s)"),
            ("mte3_active_bw", f"{prefix}_mte3_active_bw(GB/s)"),
        ]
        if prefix == "aic":
            bw_fields.extend([
                ("mte1_active_bw", "aic_mte1_active_bw(GB/s)"),
                ("fixpipe_active_bw", "aic_fixpipe_active_bw(GB/s)"),
            ])
        for display_name, field in bw_fields:
            vals = [safe_float(r.get(field)) for r in pipe_rows]
            line = stat_line(display_name, vals, ".1f", unit="GB/s")
            if line:
                lines.append(line)

        # SCALAR subcategories
        scalar_sub = [
            ("single", f"{prefix}_scalar_single_time(us)"),
            ("dual", f"{prefix}_scalar_dual_time(us)"),
            ("wait", f"{prefix}_scalar_wait_time(us)"),
            ("mte2_stall", f"{prefix}_scalar_mte2_stall_time(us)"),
            ("mte3_stall", f"{prefix}_scalar_mte3_stall_time(us)"),
        ]
        if prefix == "aiv":
            scalar_sub.extend([
                ("vec_stall", "aiv_scalar_vector_stall_time(us)"),
                ("ub_stall", "aiv_scalar_stall_by_ub_time(us)"),
            ])
        elif prefix == "aic":
            scalar_sub.extend([
                ("cube_stall", "aic_scalar_cube_stall_time(us)"),
                ("mte1_stall", "aic_scalar_mte1_stall_time(us)"),
            ])
        scalar_sub.append(("wait_ib", f"{prefix}_scalar_wait_ib_time(us)"))

        parts = []
        for display_name, field in scalar_sub:
            vals = [safe_float(r.get(field)) for r in pipe_rows]
            avg = statistics.mean(vals)
            if avg > 0.001:
                parts.append(f"{display_name}: {avg:.2f}us")
        if parts:
            lines.append("")
            lines.append("--- SCALAR 子类耗时 (avg) ---")
            lines.append("  " + " | ".join(parts))

        # Head overhead
        if duration > 0 and core_times:
            max_core = max(core_times)
            overhead = max(0, duration - max_core)
            overhead_pct = overhead / duration * 100
            lines.append("")
            lines.append(f"--- 头开销 ---")
            lines.append(f"  Task Duration: {duration}us | 最长核: {max_core:.2f}us | 头开销: {overhead:.2f}us ({overhead_pct:.1f}%)")

    # === Memory ===
    mem_rows = read_csv(os.path.join(opprof_dir, "Memory.csv"))
    if mem_rows:
        n = len(mem_rows)
        lines.append("")
        lines.append("--- Memory ---")

        gm_ub = [safe_float(r.get("GM_to_UB_datas(KB)")) for r in mem_rows]
        ub_gm = [safe_float(r.get("UB_to_GM_datas(KB)")) for r in mem_rows]
        gm_l1 = [safe_float(r.get("GM_to_L1_datas(KB)")) for r in mem_rows]
        read_mm = [safe_float(r.get("read_main_memory_datas(KB)")) for r in mem_rows]
        write_mm = [safe_float(r.get("write_main_memory_datas(KB)")) for r in mem_rows]

        if sum(gm_ub) > 0:
            bw_usage = [safe_float(r.get("GM_to_UB_bw_usage_rate(%)")) for r in mem_rows]
            lines.append(f"  GM→UB: {sum(gm_ub):.1f}KB total ({statistics.mean(gm_ub):.1f}KB/core), BW usage: {statistics.mean(bw_usage):.2f}%")
        if sum(ub_gm) > 0:
            bw_usage = [safe_float(r.get("UB_to_GM_bw_usage_rate(%)")) for r in mem_rows]
            lines.append(f"  UB→GM: {sum(ub_gm):.1f}KB total ({statistics.mean(ub_gm):.1f}KB/core), BW usage: {statistics.mean(bw_usage):.2f}%")
        if sum(gm_l1) > 0:
            bw_usage = [safe_float(r.get("GM_to_L1_bw_usage_rate(%)")) for r in mem_rows]
            lines.append(f"  GM→L1: {sum(gm_l1):.1f}KB total ({statistics.mean(gm_l1):.1f}KB/core), BW usage: {statistics.mean(bw_usage):.2f}%")
        if sum(read_mm) > 0:
            lines.append(f"  主存读取: {sum(read_mm):.1f}KB total")
        if sum(write_mm) > 0:
            lines.append(f"  主存写入: {sum(write_mm):.1f}KB total")

        # Average transfer size per MTE2
        prefix = detect_core_prefix(pipe_rows) if pipe_rows else "aiv"
        mte2_instr = sum(safe_float(r.get(f"{prefix}_mte2_instructions")) + safe_float(r.get("aic_mte2_instructions" if prefix == "aiv" else "aiv_mte2_instructions")) for r in mem_rows)
        # Simpler: just sum both
        mte2_instr_total = sum(safe_float(r.get("aiv_mte2_instructions", 0)) + safe_float(r.get("aic_mte2_instructions", 0)) for r in mem_rows)
        if mte2_instr_total > 0 and sum(gm_ub) > 0:
            avg_transfer = sum(gm_ub) * 1024 / mte2_instr_total
            lines.append(f"  Avg MTE2 transfer: {avg_transfer / 1024:.2f}KB ({int(mte2_instr_total)} instructions total)")

        # Bandwidth rates
        gm_ub_bw = [safe_float(r.get("aiv_gm_to_ub_bw(GB/s)")) for r in mem_rows]
        ub_gm_bw = [safe_float(r.get("aiv_ub_to_gm_bw(GB/s)")) for r in mem_rows]
        if sum(gm_ub_bw) > 0:
            lines.append(f"  GM→UB avg BW: {statistics.mean(gm_ub_bw):.2f} GB/s")
        if sum(ub_gm_bw) > 0:
            lines.append(f"  UB→GM avg BW: {statistics.mean(ub_gm_bw):.2f} GB/s")

    # === MemoryUB ===
    ub_rows = read_csv(os.path.join(opprof_dir, "MemoryUB.csv"))
    if ub_rows:
        lines.append("")
        lines.append("--- MemoryUB ---")
        ub_fields = [
            ("UB read BW (vector)", "aiv_ub_read_bw_vector(GB/s)"),
            ("UB write BW (vector)", "aiv_ub_write_bw_vector(GB/s)"),
            ("UB read BW (scalar)", "aiv_ub_read_bw_scalar(GB/s)"),
            ("UB write BW (scalar)", "aiv_ub_write_bw_scalar(GB/s)"),
        ]
        for display, field in ub_fields:
            vals = [safe_float(r.get(field)) for r in ub_rows]
            avg = statistics.mean(vals)
            if avg > 0.001:
                lines.append(f"  {display}: avg={avg:.1f} GB/s")

    # === MemoryL0 ===
    l0_rows = read_csv(os.path.join(opprof_dir, "MemoryL0.csv"))
    if l0_rows:
        l0_fields = [
            ("L0A read BW", "aic_l0a_read_bw(GB/s)"),
            ("L0A write BW", "aic_l0a_write_bw(GB/s)"),
            ("L0B read BW", "aic_l0b_read_bw(GB/s)"),
            ("L0B write BW", "aic_l0b_write_bw(GB/s)"),
            ("L0C read BW (cube)", "aic_l0c_read_bw_cube(GB/s)"),
            ("L0C write BW (cube)", "aic_l0c_write_bw_cube(GB/s)"),
        ]
        parts = []
        for display, field in l0_fields:
            vals = [safe_float(r.get(field)) for r in l0_rows]
            avg = statistics.mean(vals)
            if avg > 0.001:
                parts.append(f"{display}: {avg:.1f} GB/s")
        if parts:
            lines.append("")
            lines.append("--- MemoryL0 ---")
            for p in parts:
                lines.append(f"  {p}")

    # === L2Cache ===
    l2_rows = read_csv(os.path.join(opprof_dir, "L2Cache.csv"))
    if l2_rows:
        lines.append("")
        lines.append("--- L2Cache ---")
        prefix = detect_core_prefix(pipe_rows) if pipe_rows else "aiv"
        for rate_name, field in [("total_hit", f"{prefix}_total_hit_rate(%)"),
                                  ("read_hit", f"{prefix}_read_hit_rate(%)"),
                                  ("write_hit", f"{prefix}_write_hit_rate(%)")]:
            vals = [safe_float(r.get(field)) for r in l2_rows]
            if any(v > 0 for v in vals):
                lines.append(f"  {rate_name}: avg={statistics.mean(vals):.1f}% (min={min(vals):.1f}%, max={max(vals):.1f}%)")

        # Raw hit/miss counts
        hit_field = f"{prefix}_write_cache_hit"
        miss_field = f"{prefix}_write_cache_miss_allocate"
        total_hit = sum(safe_float(r.get(hit_field)) for r in l2_rows)
        total_miss = sum(safe_float(r.get(miss_field)) for r in l2_rows)
        if total_hit + total_miss > 0:
            lines.append(f"  write cache: hit={int(total_hit)} miss={int(total_miss)}")

    # === ResourceConflict ===
    rc_rows = read_csv(os.path.join(opprof_dir, "ResourceConflictRatio.csv"))
    if rc_rows:
        lines.append("")
        lines.append("--- ResourceConflict ---")
        prefix = detect_core_prefix(pipe_rows) if pipe_rows else "aiv"

        cflt_fields = [
            ("vec_total_cflt", f"{prefix}_vec_total_cflt_ratio"),
            ("bankgroup_cflt", f"{prefix}_vec_bankgroup_cflt_ratio"),
            ("bank_cflt", f"{prefix}_vec_bank_cflt_ratio"),
            ("resc_cflt", f"{prefix}_vec_resc_cflt_ratio"),
            ("mte_cflt", f"{prefix}_vec_mte_cflt_ratio"),
        ]
        parts = []
        for display, field in cflt_fields:
            vals = [safe_float(r.get(field)) * 100 for r in rc_rows]
            parts.append(f"{display}: {statistics.mean(vals):.2f}%")
        lines.append("  " + " | ".join(parts))

        wait_fields = [
            ("vec_wait", f"{prefix}_vec_wait_ratio"),
            ("mte2_wait", f"{prefix}_mte2_wait_ratio"),
            ("mte3_wait", f"{prefix}_mte3_wait_ratio"),
        ]
        if prefix == "aic":
            wait_fields.insert(0, ("cube_wait", "aic_cube_wait_ratio"))

        parts = []
        for display, field in wait_fields:
            vals = [safe_float(r.get(field)) * 100 for r in rc_rows]
            parts.append(f"{display}: {statistics.mean(vals):.2f}%")
        lines.append("  " + " | ".join(parts))

    # === ArithmeticUtilization ===
    arith_rows = read_csv(os.path.join(opprof_dir, "ArithmeticUtilization.csv"))
    if arith_rows:
        lines.append("")
        lines.append("--- ArithmeticUtilization ---")

        vec_fields = [
            ("vec_fp32", "aiv_vec_fp32_ratio"),
            ("vec_fp16", "aiv_vec_fp16_ratio"),
            ("vec_int32", "aiv_vec_int32_ratio"),
            ("vec_int16", "aiv_vec_int16_ratio"),
            ("vec_misc", "aiv_vec_misc_ratio"),
        ]
        parts = []
        for display, field in vec_fields:
            vals = [safe_float(r.get(field)) * 100 for r in arith_rows]
            avg = statistics.mean(vals)
            if avg > 0.01:
                parts.append(f"{display}: {avg:.1f}%")
        if parts:
            lines.append("  " + " | ".join(parts))

        vec_fops = [safe_float(r.get("aiv_vec_fops")) for r in arith_rows]
        if statistics.mean(vec_fops) > 0:
            lines.append(f"  vec_fops: {statistics.mean(vec_fops):.0f}/core")

        # Cube
        cube_fields = [
            ("cube_fp16", "aic_cube_fp16_ratio"),
            ("cube_int8", "aic_cube_int8_ratio"),
        ]
        parts = []
        for display, field in cube_fields:
            vals = [safe_float(r.get(field)) * 100 for r in arith_rows]
            avg = statistics.mean(vals)
            if avg > 0.01:
                parts.append(f"{display}: {avg:.1f}%")
        cube_fops = [safe_float(r.get("aic_cube_fops")) for r in arith_rows]
        if statistics.mean(cube_fops) > 0:
            parts.append(f"cube_fops: {statistics.mean(cube_fops):.0f}/core")
        if parts:
            lines.append("  " + " | ".join(parts))

    # === Footer ===
    rel_round = os.path.relpath(round_dir, os.path.dirname(ops_dir)) if ops_dir else round_dir
    lines.append("")
    lines.append("--- 原始数据位置 ---")
    lines.append(f"  CSV 文件: {round_dir}/")
    lines.append("  如需逐核详情，请 Read 对应 CSV 文件。")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate msprof statistical summary and archive CSV data.")
    parser.add_argument("opprof_dir", help="Path to OPPROF_{timestamp}_XXX directory")
    parser.add_argument("ops_dir", help="Path to operator directory (e.g., ops/Softmax)")
    parser.add_argument("--round-name", help="Override round directory name (default: auto-increment)")
    args = parser.parse_args()

    opprof_dir = os.path.abspath(args.opprof_dir)
    ops_dir = os.path.abspath(args.ops_dir)

    if not os.path.isdir(opprof_dir):
        print(f"Error: '{opprof_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(ops_dir):
        print(f"Error: '{ops_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    # Check PipeUtilization.csv exists
    if not os.path.exists(os.path.join(opprof_dir, "PipeUtilization.csv")):
        print(f"Error: PipeUtilization.csv not found in {opprof_dir}", file=sys.stderr)
        sys.exit(1)

    # Determine round directory
    perf_dir = os.path.join(ops_dir, "docs", "perf")
    if args.round_name:
        round_dir = os.path.join(perf_dir, args.round_name)
    else:
        round_dir = find_next_round(perf_dir)

    # Archive CSVs
    copied = archive_csvs(opprof_dir, round_dir)
    print(f"Archived {len(copied)} CSV files to: {round_dir}")

    # Generate summary
    summary = generate_summary(opprof_dir, round_dir, ops_dir)
    summary_path = os.path.join(round_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"Summary written to: {summary_path}")
    print()
    print(summary)


if __name__ == "__main__":
    main()
