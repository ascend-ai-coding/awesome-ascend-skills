#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# ----------------------------------------------------------------------------------------------------------
# msprof 解析 & 归档脚本（msprof 方案）
#
# 当 CANN 环境缺少 `msopprof` 二进制、无法使用 `msprof op` 时，
# 用 msprof 分多次 `--aic-metrics=<group>` 采集（由 msprof_profile_run.sh 串起来），
# 本脚本读取每次采集的 `mindstudio_profiler_output/op_summary_*.csv`，按 Op Name 合并列，
# 产出与 scripts/perf_summary.py 对齐的 `summary.txt`，并把原始 CSV 复制到 round_NNN 归档目录。
#
# 用法:
#     python3 msprof_perf_summary.py <PROF_GROUP_dir> <ops_dir>
#
# 其中 <PROF_GROUP_dir> 是 msprof_profile_run.sh 生成的 PROF_GROUP_<timestamp> 目录，
# 其下包含多个 PROF_<Metric>/PROF_*/mindstudio_profiler_output/op_summary_*.csv
#
# 归档位置 (与 perf_summary.py 保持一致):
#     <ops_dir>/docs/perf/round_NNN/
#         op_summary_<Metric>.csv    (7 份，分别对应 7 个 aic-metrics)
#         task_time.csv              (从任一 PROF 目录复制过来，若存在)
#         op_statistic.csv           (同上)
#         summary.txt                (合并后的统计摘要)
# ----------------------------------------------------------------------------------------------------------

import argparse
import csv
import logging
import glob
import os
import re
import shutil
import sqlite3
import statistics
import sys
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)

METRICS = [
    "PipeUtilization",
    "ArithmeticUtilization",
    "Memory",
    "MemoryL0",
    "MemoryUB",
    "L2Cache",
    "ResourceConflictRatio",
]


def safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    s = str(val).strip().rstrip("\t ")
    if s in ("", "N/A", "NA", "-"):
        return default
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


def safe_int(val: Any, default: int = 0) -> int:
    return int(safe_float(val, default))


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def find_op_summary(prof_metric_dir: str) -> Optional[str]:
    """Find mindstudio_profiler_output/op_summary_*.csv under a PROF_* tree."""
    pattern = os.path.join(prof_metric_dir, "**", "mindstudio_profiler_output", "op_summary_*.csv")
    hits = sorted(glob.glob(pattern, recursive=True))
    return hits[-1] if hits else None


def pick_target_row(rows: List[Dict[str, str]], target_name: Optional[str]) -> Optional[Dict[str, str]]:
    """Pick the row of the target kernel. Preference:
    1. Exact match on Op Name == target_name.
    2. Skip ACL/fuzzy kernels (e.g. AscendInitializeOp, memcpy, etc.).
    3. Pick the row with the largest Task Duration(us).
    """
    if not rows:
        return None
    if target_name:
        for r in rows:
            if r.get("Op Name", "").strip() == target_name:
                return r
    ai_core_rows = [
        r for r in rows
        if "AI_CORE" in r.get("Task Type", "")
        or "AIV" in r.get("Task Type", "")
        or "MIX" in r.get("Task Type", "")
    ]
    candidates = ai_core_rows or rows
    return max(candidates, key=lambda r: safe_float(r.get("Task Duration(us)")))


def _merge_row_values(merged: Dict[str, Any], row: Dict[str, str]) -> None:
    for k, v in row.items():
        if k in (None, ""):
            continue
        if k not in merged:
            merged[k] = v
        else:
            old = merged[k]
            if (old in (None, "", "N/A", "NA")) and v not in (None, "", "N/A", "NA"):
                merged[k] = v
            elif safe_float(old) == 0 and safe_float(v) != 0:
                merged[k] = v


def merge_metric_rows(group_dir: str, target_name: Optional[str]) -> Dict[str, Any]:
    """Walk PROF_<Metric>/ subdirs under group_dir, read op_summary.csv, merge columns."""
    merged: Dict[str, Any] = {}
    merged["_metric_sources"] = {}  # metric -> csv path
    merged["_missing_metrics"] = []

    for metric in METRICS:
        prof_metric_dir = os.path.join(group_dir, f"PROF_{metric}")
        if not os.path.isdir(prof_metric_dir):
            merged["_missing_metrics"].append(metric)
            continue
        csv_path = find_op_summary(prof_metric_dir)
        if not csv_path:
            merged["_missing_metrics"].append(metric)
            continue
        rows = read_csv_rows(csv_path)
        row = pick_target_row(rows, target_name)
        if not row:
            merged["_missing_metrics"].append(metric)
            continue
        merged["_metric_sources"][metric] = csv_path
        _merge_row_values(merged, row)
    return merged


def load_per_core_cycles(group_dir: str) -> List[Tuple[int, int]]:
    """Read per-core cycle counts from the sample-based PROF_Sample run.

    Returns a list of (coreid, total_task_cyc) sorted by coreid.
    Returns [] if the sample-based PROF_Sample/.../device_0/sqlite/aicore.db
    is missing (e.g. user ran without sample pass).
    """
    candidates = glob.glob(os.path.join(group_dir, "PROF_Sample", "PROF_*", "device_0", "sqlite", "aicore.db"))
    if not candidates:
        return []
    db = sorted(candidates)[-1]
    try:
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        rows = list(cur.execute(
            "SELECT coreid, SUM(task_cyc) FROM AICoreOriginalData WHERE task_cyc>0 GROUP BY coreid ORDER BY coreid"
        ))
        conn.close()
        return [(int(cid), int(cyc)) for cid, cyc in rows if cid is not None]
    except sqlite3.Error:
        return []


def per_core_balance_section(merged: Dict[str, Any], group_dir: str) -> List[str]:
    """Build the `逐核负载均衡` section using per-core cycles + aicore_time anchor."""
    core_rows = load_per_core_cycles(group_dir)
    if not core_rows:
        return []
    aicore_time_us = safe_float(merged.get("aicore_time(us)"))
    if aicore_time_us <= 0:
        return []
    max_cyc = max(c for _, c in core_rows)
    if max_cyc <= 0:
        return []
    ns_per_cyc = aicore_time_us * 1000.0 / max_cyc
    freq_ghz = 1.0 / ns_per_cyc
    times = [(cid, cyc * ns_per_cyc / 1000.0) for cid, cyc in core_rows]  # us
    t_values = [t for _, t in times]
    t_min = min(t_values)
    t_max = max(t_values)
    t_avg = statistics.mean(t_values)
    spread_pct = (t_max - t_min) / t_max * 100.0 if t_max > 0 else 0.0

    if spread_pct < 10:
        verdict = "达标 (<10%)"
    elif spread_pct < 30:
        verdict = "警告 (10~30%)"
    else:
        verdict = "严重问题 (>30%)"

    lines = ["", "--- 逐核负载均衡 (sample-based aicore.db) ---"]
    lines.append(f"  有效核数: {len(times)}  | 主频推算: {freq_ghz:.3f} GHz ({ns_per_cyc:.4f} ns/cycle)")
    lines.append(f"  min={t_min:.3f}us  avg={t_avg:.3f}us  max={t_max:.3f}us")
    lines.append(f"  (max-min)/max = {spread_pct:.2f}%  ->  {verdict}")

    # Top-3 slow / fast
    sorted_desc = sorted(times, key=lambda x: -x[1])
    slow_top = sorted_desc[:3]
    fast_top = sorted_desc[-3:][::-1]
    lines.append("  Top-3 慢核: " + ", ".join(f"Core{cid}={t:.2f}us" for cid, t in slow_top))
    lines.append("  Top-3 快核: " + ", ".join(f"Core{cid}={t:.2f}us" for cid, t in fast_top))

    # Bimodal (cluster) split heuristic: sort by coreid, split into halves
    sorted_by_id = sorted(times, key=lambda x: x[0])
    if len(sorted_by_id) >= 4:
        mid = len(sorted_by_id) // 2
        g1 = [t for _, t in sorted_by_id[:mid]]
        g2 = [t for _, t in sorted_by_id[mid:]]
        g1_avg = statistics.mean(g1)
        g2_avg = statistics.mean(g2)
        gap = abs(g1_avg - g2_avg) / max(g1_avg, g2_avg) * 100.0
        if gap >= 2.0:
            lines.append(f"  [提示] 前半段 core 均值 {g1_avg:.2f}us vs 后半段 {g2_avg:.2f}us，差距 {gap:.2f}%")
            lines.append(f"         疑似两簇 (NUMA / L2 slice) 负载偏斜，建议尝试 block swat / 尾轮均衡策略。")
    return lines


def archive_per_core_csv(group_dir: str, round_dir: str) -> Optional[str]:
    """Dump per-core cycle/time table to round_dir/per_core_time.csv."""
    core_rows = load_per_core_cycles(group_dir)
    if not core_rows:
        return None
    # We do not have aicore_time here; caller writes after summary. Store cycles only;
    # time is derivable and already reported in summary.txt.
    out = os.path.join(round_dir, "per_core_cycles.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["coreid", "task_cycles"])
        for cid, cyc in core_rows:
            w.writerow([cid, cyc])
    return out


def archive_csvs(group_dir: str, round_dir: str) -> List[str]:
    os.makedirs(round_dir, exist_ok=True)
    copied = []
    for metric in METRICS:
        prof_metric_dir = os.path.join(group_dir, f"PROF_{metric}")
        if not os.path.isdir(prof_metric_dir):
            continue
        op_csv = find_op_summary(prof_metric_dir)
        if op_csv:
            dst = os.path.join(round_dir, f"op_summary_{metric}.csv")
            shutil.copy2(op_csv, dst)
            copied.append(os.path.basename(dst))
        # Extra helpful files
        mso_dir = os.path.dirname(op_csv) if op_csv else None
        if mso_dir:
            _copy_extra_csvs(mso_dir, metric, round_dir, copied)
    return copied


def _copy_extra_csvs(mso_dir: str, metric: str, round_dir: str, copied: List[str]) -> None:
    for extra in ("op_statistic_", "task_time_", "api_statistic_"):
        for f in sorted(glob.glob(os.path.join(mso_dir, f"{extra}*.csv"))):
            name = f"{extra.rstrip('_')}_{metric}.csv"
            dst = os.path.join(round_dir, name)
            if not os.path.exists(dst):
                shutil.copy2(f, dst)
                copied.append(os.path.basename(dst))
            break


def find_next_round(perf_dir: str) -> str:
    if not os.path.exists(perf_dir):
        return os.path.join(perf_dir, "round_001")
    existing = [d for d in os.listdir(perf_dir) if re.match(r"round_\d+", d)]
    if not existing:
        return os.path.join(perf_dir, "round_001")
    nums = [int(re.search(r"\d+", d).group()) for d in existing]
    return os.path.join(perf_dir, f"round_{max(nums) + 1:03d}")


def fmt_ratio(val: Any, width: int = 6) -> str:
    v = safe_float(val) * 100.0
    return f"{v:>{width}.2f}%"


def fmt_float(val: Any, width: int = 10, prec: int = 2) -> str:
    return f"{safe_float(val):>{width}.{prec}f}"


def _add_memory_section(lines: List[str], merged: Dict[str, Any]) -> None:
    _mem_keys = [
        "aic_main_mem_read_bw(GB/s)", "aic_main_mem_write_bw(GB/s)",
        "aiv_main_mem_read_bw(GB/s)", "aiv_main_mem_write_bw(GB/s)",
        "aic_l1_read_bw(GB/s)", "aic_l1_write_bw(GB/s)",
        "aiv_ub_read_bw(GB/s)", "aiv_ub_write_bw(GB/s)",
    ]
    has_mem = any(safe_float(merged.get(k)) > 0 for k in _mem_keys)
    if not has_mem:
        return
    lines.append("")
    lines.append("--- Memory 带宽 (aic-metrics=Memory) ---")
    mem_rows = [
        ("aic main_mem read", "aic_main_mem_read_bw(GB/s)"),
        ("aic main_mem write", "aic_main_mem_write_bw(GB/s)"),
        ("aiv main_mem read", "aiv_main_mem_read_bw(GB/s)"),
        ("aiv main_mem write", "aiv_main_mem_write_bw(GB/s)"),
        ("aic L1 read", "aic_l1_read_bw(GB/s)"),
        ("aic L1 write", "aic_l1_write_bw(GB/s)"),
        ("aiv UB read", "aiv_ub_read_bw(GB/s)"),
        ("aiv UB write", "aiv_ub_write_bw(GB/s)"),
    ]
    for label, key in mem_rows:
        v = safe_float(merged.get(key))
        if v > 0:
            lines.append(f"  {label}: {v:.2f} GB/s")


def _add_memory_l0_section(lines: List[str], merged: Dict[str, Any]) -> None:
    _l0_keys = [
        "aic_l0a_read_bw(GB/s)", "aic_l0a_write_bw(GB/s)",
        "aic_l0b_read_bw(GB/s)", "aic_l0b_write_bw(GB/s)",
        "aic_l0c_read_bw_cube(GB/s)", "aic_l0c_write_bw_cube(GB/s)",
    ]
    has_l0 = any(safe_float(merged.get(k)) > 0 for k in _l0_keys)
    if not has_l0:
        return
    lines.append("")
    lines.append("--- MemoryL0 ---")
    for label, key in [
        ("L0A read", "aic_l0a_read_bw(GB/s)"),
        ("L0A write", "aic_l0a_write_bw(GB/s)"),
        ("L0B read", "aic_l0b_read_bw(GB/s)"),
        ("L0B write", "aic_l0b_write_bw(GB/s)"),
        ("L0C read (cube)", "aic_l0c_read_bw_cube(GB/s)"),
        ("L0C write (cube)", "aic_l0c_write_bw_cube(GB/s)"),
    ]:
        v = safe_float(merged.get(key))
        if v > 0:
            lines.append(f"  {label}: {v:.2f} GB/s")


def _add_memory_ub_section(lines: List[str], merged: Dict[str, Any]) -> None:
    _ub_keys = [
        "aiv_ub_read_bw_vector(GB/s)", "aiv_ub_write_bw_vector(GB/s)",
        "aiv_ub_read_bw_scalar(GB/s)", "aiv_ub_write_bw_scalar(GB/s)",
        "aic_ub_read_bw_scalar(GB/s)", "aic_ub_write_bw_scalar(GB/s)",
        "aiv_fixp2ub_write_bw(GB/s)", "aic_fixp2ub_write_bw(GB/s)",
    ]
    has_ub = any(safe_float(merged.get(k)) > 0 for k in _ub_keys)
    if not has_ub:
        return
    lines.append("")
    lines.append("--- MemoryUB ---")
    for label, key in [
        ("UB read (vector)", "aiv_ub_read_bw_vector(GB/s)"),
        ("UB write (vector)", "aiv_ub_write_bw_vector(GB/s)"),
        ("UB read (scalar)", "aiv_ub_read_bw_scalar(GB/s)"),
        ("UB write (scalar)", "aiv_ub_write_bw_scalar(GB/s)"),
        ("aic UB read (scalar)", "aic_ub_read_bw_scalar(GB/s)"),
        ("aic UB write (scalar)", "aic_ub_write_bw_scalar(GB/s)"),
        ("aiv fixp2ub write", "aiv_fixp2ub_write_bw(GB/s)"),
        ("aic fixp2ub write", "aic_fixp2ub_write_bw(GB/s)"),
    ]:
        v = safe_float(merged.get(key))
        if v > 0:
            lines.append(f"  {label}: {v:.2f} GB/s")


def _add_l2cache_section(lines: List[str], merged: Dict[str, Any]) -> None:
    l2_fields_aic = [
        ("aic read hit", "aic_read_local_l2_hit"),
        ("aic read miss", "aic_read_local_l2_miss"),
        ("aic read victim", "aic_read_local_l2_victim"),
        ("aic write hit", "aic_write_local_l2_hit"),
        ("aic write miss", "aic_write_local_l2_miss"),
        ("aic write victim", "aic_write_local_l2_victim"),
    ]
    l2_fields_aiv = [
        ("aiv read hit", "aiv_read_local_l2_hit"),
        ("aiv read miss", "aiv_read_local_l2_miss"),
        ("aiv read victim", "aiv_read_local_l2_victim"),
        ("aiv write hit", "aiv_write_local_l2_hit"),
        ("aiv write miss", "aiv_write_local_l2_miss"),
        ("aiv write victim", "aiv_write_local_l2_victim"),
    ]
    l2_has = any(safe_float(merged.get(k)) > 0 for _, k in l2_fields_aic + l2_fields_aiv)
    if not l2_has:
        return
    lines.append("")
    lines.append("--- L2Cache ---")

    def _emit(group_label, fields):
        hit = safe_float(merged.get(fields[0][1]))
        miss = safe_float(merged.get(fields[1][1]))
        total = hit + miss
        if total > 0:
            rate = hit / total * 100.0
            lines.append(f"  {group_label} read: hit={int(hit)} miss={int(miss)} hit_rate={rate:.2f}%")
        whit = safe_float(merged.get(fields[3][1]))
        wmiss = safe_float(merged.get(fields[4][1]))
        wtotal = whit + wmiss
        if wtotal > 0:
            rate = whit / wtotal * 100.0
            lines.append(f"  {group_label} write: hit={int(whit)} miss={int(wmiss)} hit_rate={rate:.2f}%")

    _emit("aic", l2_fields_aic)
    _emit("aiv", l2_fields_aiv)


def _add_rc_section(lines: List[str], merged: Dict[str, Any]) -> None:
    rc_fields = [
        ("vec_bank_cflt", "aiv_vec_bank_cflt_ratio"),
        ("vec_resc_cflt", "aiv_vec_resc_cflt_ratio"),
    ]
    if not any(safe_float(merged.get(k)) > 0 for _, k in rc_fields):
        return
    lines.append("")
    lines.append("--- ResourceConflict ---")
    parts = [f"{label}={fmt_ratio(merged.get(key))}" for label, key in rc_fields]
    lines.append("  " + " | ".join(parts))


def _add_arith_section(lines: List[str], merged: Dict[str, Any]) -> None:
    arith_fields = [
        ("mac_fp16", "aic_mac_fp16_ratio"),
        ("mac_int8", "aic_mac_int8_ratio"),
    ]
    has_arith_fields = any(
        safe_float(merged.get(k)) > 0 for _, k in arith_fields
    )
    arith_has = has_arith_fields or safe_float(merged.get("aic_cube_fops")) > 0
    if not arith_has:
        return
    lines.append("")
    lines.append("--- ArithmeticUtilization ---")
    parts = []
    for label, key in arith_fields:
        v = safe_float(merged.get(key))
        if v > 0:
            parts.append(f"{label}={fmt_ratio(merged.get(key))}")
    fops = safe_float(merged.get("aic_cube_fops"))
    if fops > 0:
        parts.append(f"cube_fops={fops:.0f}")
    if parts:
        lines.append("  " + " | ".join(parts))


def _add_basic_info(lines: List[str], merged: Dict[str, Any]) -> Tuple[float, float, float]:
    op_name = merged.get("Op Name", "unknown")
    op_type = merged.get("OP Type", "unknown")
    task_type = merged.get("Task Type", "")
    duration = safe_float(merged.get("Task Duration(us)"))
    block_dim = safe_int(merged.get("Block Num", 0))
    mix_block = safe_int(merged.get("Mix Block Num", 0))
    aicore_time = safe_float(merged.get("aicore_time(us)"))
    aiv_time = safe_float(merged.get("aiv_time(us)"))

    lines.append("=== 上板性能统计摘要 (msprof) ===")
    lines.append(f"Op: {op_name}")
    lines.append(
        f"Type: {op_type} | TaskType: {task_type} | Duration: {duration}us"
        f" | BlockDim: {block_dim} (mix={mix_block})"
    )
    if merged.get("_missing_metrics"):
        lines.append(f"[WARN] 缺失指标: {', '.join(merged['_missing_metrics'])}")
    lines.append("")
    lines.append("[注] msprof 的 op_summary 是 per-op 聚合值（不含逐核 min/avg/max）；")
    lines.append("     如需逐核数据请改用 msprof op (需要 msopprof 二进制)。")
    return duration, aicore_time, aiv_time


def _add_pipe_ratios(lines: List[str], merged: Dict[str, Any], aicore_time: float, aiv_time: float) -> None:
    lines.append("")
    aic_cube_like = max(safe_float(merged.get("aic_mac_ratio")), safe_float(merged.get("aic_mte2_ratio")))
    aiv_vec_like = safe_float(merged.get("aiv_vec_ratio"))
    prefix = "aic" if aic_cube_like >= aiv_vec_like else "aiv"
    lines.append(f"--- Pipe ratios (主导核 = {prefix}) ---")
    lines.append(f"  aicore_time: {aicore_time:.3f}us | aiv_time: {aiv_time:.3f}us")

    aic_fields = [
        ("aic_mac_ratio", "mac"),
        ("aic_cube_ratio", "cube"),
        ("aic_mte1_ratio", "mte1"),
        ("aic_mte2_ratio", "mte2"),
        ("aic_mte3_ratio", "mte3"),
        ("aic_fixpipe_ratio", "fixpipe"),
        ("aic_scalar_ratio", "scalar"),
        ("aic_icache_miss_rate", "icache_miss"),
    ]
    parts = [f"{label}={fmt_ratio(merged.get(key))}" for key, label in aic_fields if safe_float(merged.get(key)) > 0]
    if parts:
        lines.append("  aic: " + " | ".join(parts))

    aiv_fields = [
        ("aiv_vec_ratio", "vec"),
        ("aiv_scalar_ratio", "scalar"),
        ("aiv_mte2_ratio", "mte2"),
        ("aiv_mte3_ratio", "mte3"),
        ("aiv_icache_miss_rate", "icache_miss"),
    ]
    parts = [f"{label}={fmt_ratio(merged.get(key))}" for key, label in aiv_fields if safe_float(merged.get(key)) > 0]
    if parts:
        lines.append("  aiv: " + " | ".join(parts))

    util = safe_float(merged.get("cube_utilization(%)"))
    if util > 0:
        lines.append(f"  cube_utilization: {util:.2f}%")


def _add_overhead(lines: List[str], duration: float, aicore_time: float, aiv_time: float) -> None:
    lines.append("")
    core_time_max = max(aicore_time, aiv_time)
    overhead = max(0.0, duration - core_time_max)
    overhead_pct = (overhead / duration * 100.0) if duration > 0 else 0.0
    lines.append("--- 头开销 ---")
    lines.append(
        f"  Task Duration: {duration}us | 核最长耗时: {core_time_max:.3f}us"
        f" | 头开销: {overhead:.3f}us ({overhead_pct:.1f}%)"
    )


def _add_footer(lines: List[str], merged: Dict[str, Any], round_dir: str, group_dir: str) -> None:
    lines.append("")
    lines.append("--- 原始数据位置 ---")
    lines.append(f"  归档 CSV : {round_dir}/")
    lines.append(f"  原始 PROF: {group_dir}/")
    lines.append("  按 aic-metrics 拆分的 op_summary_<Metric>.csv 均已复制到归档目录，")
    lines.append("  如需逐列查看可直接 Read。")
    if merged.get("_metric_sources"):
        lines.append("")
        lines.append("--- Metric 来源 ---")
        for m in METRICS:
            src = merged["_metric_sources"].get(m)
            if src:
                lines.append(f"  {m:<22s} <- {src}")
            else:
                lines.append(f"  {m:<22s} <MISSING>")


def generate_summary(merged: Dict[str, Any], round_dir: str, group_dir: str) -> str:
    lines: List[str] = []

    duration, aicore_time, aiv_time = _add_basic_info(lines, merged)
    _add_pipe_ratios(lines, merged, aicore_time, aiv_time)
    _add_overhead(lines, duration, aicore_time, aiv_time)
    _add_memory_section(lines, merged)
    _add_memory_l0_section(lines, merged)
    _add_memory_ub_section(lines, merged)
    _add_l2cache_section(lines, merged)
    _add_rc_section(lines, merged)
    _add_arith_section(lines, merged)
    lines.extend(per_core_balance_section(merged, group_dir))
    _add_footer(lines, merged, round_dir, group_dir)

    return "\n".join(lines)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Parse msprof PROF_GROUP and generate skill-compatible summary.")
    parser.add_argument("prof_group_dir", help="PROF_GROUP_<timestamp> directory produced by msprof_profile_run.sh")
    parser.add_argument("ops_dir", help="Operator directory (summary archived to <ops_dir>/docs/perf/round_NNN/)")
    parser.add_argument("--op-name", default=None,
                        help="Exact Op Name to pick in op_summary.csv; default = longest-duration AI_CORE op")
    parser.add_argument("--round-name", default=None, help="Override round directory name (default: auto-increment)")
    args = parser.parse_args()

    group_dir = os.path.abspath(args.prof_group_dir)
    ops_dir = os.path.abspath(args.ops_dir)

    if not os.path.isdir(group_dir):
        LOGGER.error("'%s' is not a directory.", group_dir)
        sys.exit(1)
    if not os.path.isdir(ops_dir):
        LOGGER.error("'%s' is not a directory.", ops_dir)
        sys.exit(1)

    merged = merge_metric_rows(group_dir, args.op_name)
    if not merged.get("Op Name"):
        LOGGER.error("no op_summary_*.csv rows discovered under %s", group_dir)
        sys.exit(1)

    perf_dir = os.path.join(ops_dir, "docs", "perf")
    round_dir = os.path.join(perf_dir, args.round_name) if args.round_name else find_next_round(perf_dir)

    copied = archive_csvs(group_dir, round_dir)
    pc_csv = archive_per_core_csv(group_dir, round_dir)
    if pc_csv:
        copied.append(os.path.basename(pc_csv))
    LOGGER.info("Archived %d CSV files to: %s", len(copied), round_dir)

    summary = generate_summary(merged, round_dir, group_dir)
    summary_path = os.path.join(round_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    LOGGER.info("Summary written to: %s", summary_path)
    LOGGER.info("\n%s", summary)


if __name__ == "__main__":
    main()
