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
# msprof 解析 & 归档 & 对比测试脚本（统一入口）
#
# 支持三种模式：
#   1. 标准模式 (默认): 解析 PROF_GROUP，生成 summary.txt 并归档 CSV
#      python3 msprof_perf_summary.py <PROF_GROUP_dir> <ops_dir>
#
#   2. 对比模式 (--compare): 对算子目录做 model.py vs model_new_ascendc.py 对比测试
#      python3 msprof_perf_summary.py --compare --output-dir <op_dir> [--warm-up=N] [--device=N]
#
#   3. 批量模式 (--batch): 扫描多个算子目录，汇总批量报告
#      python3 msprof_perf_summary.py --batch <base_dir> [--output-md <path>] [--output-json <path>]
#
# 归档位置 (与 perf_summary.py 保持一致):
#     <ops_dir>/docs/perf/round_NNN/
#         op_summary_<Metric>.csv    (7 份)
#         task_time.csv              (若存在)
#         op_statistic.csv           (若存在)
#         summary.txt                (合并后的统计摘要)
# ----------------------------------------------------------------------------------------------------------

import argparse
import csv
import importlib.util
import inspect
import json
import logging
import glob
import math
import os
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
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


# ============================================================================
# 通用工具函数
# ============================================================================

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


def find_next_round(perf_dir: str) -> str:
    if not os.path.exists(perf_dir):
        return os.path.join(perf_dir, "round_001")
    existing = [d for d in os.listdir(perf_dir) if re.match(r"round_\d+", d)]
    if not existing:
        return os.path.join(perf_dir, "round_001")
    nums = [int(re.search(r"\d+", d).group()) for d in existing]
    return os.path.join(perf_dir, f"round_{max(nums) + 1:03d}")


# ============================================================================
# 标准模式：解析 PROF_GROUP
# ============================================================================

def find_op_summary(prof_metric_dir: str) -> Optional[str]:
    pattern = os.path.join(prof_metric_dir, "**", "mindstudio_profiler_output", "op_summary_*.csv")
    hits = sorted(glob.glob(pattern, recursive=True))
    return hits[-1] if hits else None


def pick_target_row(rows: List[Dict[str, str]], target_name: Optional[str]) -> Optional[Dict[str, str]]:
    if not rows:
        return None
    if target_name:
        for r in rows:
            if r.get("Op Name", "").strip() == target_name:
                return r
    ai_core_rows = []
    for r in rows:
        if "AI_CORE" in r.get("Task Type", "") \
                or "AIV" in r.get("Task Type", "") \
                or "MIX" in r.get("Task Type", ""):
            ai_core_rows.append(r)
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
    merged: Dict[str, Any] = {}
    merged["_metric_sources"] = {}
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
    times = [(cid, cyc * ns_per_cyc / 1000.0) for cid, cyc in core_rows]
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

    sorted_desc = sorted(times, key=lambda x: -x[1])
    slow_top = sorted_desc[:3]
    fast_top = sorted_desc[-3:][::-1]
    lines.append("  Top-3 慢核: " + ", ".join(f"Core{cid}={t:.2f}us" for cid, t in slow_top))
    lines.append("  Top-3 快核: " + ", ".join(f"Core{cid}={t:.2f}us" for cid, t in fast_top))

    sorted_by_id = sorted(times, key=lambda x: x[0])
    if len(sorted_by_id) >= 4:
        mid = len(sorted_by_id) // 2
        g1 = [t for _, t in sorted_by_id[:mid]]
        g2 = [t for _, t in sorted_by_id[mid:]]
        g1_avg = statistics.mean(g1)
        g2_avg = statistics.mean(g2)
        gap = abs(g1_avg - g2_avg) / max(g1_avg, g2_avg) * 100.0
        if gap >= 2.0:
            lines.append(
                f"  [提示] 前半段 core 均值 {g1_avg:.2f}us vs 后半段 {g2_avg:.2f}us，"
                f"差距 {gap:.2f}%"
            )
            lines.append(
                f"         疑似两簇 (NUMA / L2 slice) 负载偏斜，"
                f"建议尝试 block swat / 尾轮均衡策略。"
            )
    return lines


def archive_per_core_csv(group_dir: str, round_dir: str) -> Optional[str]:
    core_rows = load_per_core_cycles(group_dir)
    if not core_rows:
        return None
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


# ============================================================================
# 对比模式：model.py vs model_new_ascendc.py
# ============================================================================

def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _find_cls(module, preferred: str):
    import torch.nn as nn
    c = getattr(module, preferred, None)
    if inspect.isclass(c) and issubclass(c, nn.Module):
        return c
    for _, v in vars(module).items():
        if inspect.isclass(v) and issubclass(v, nn.Module) and v is not nn.Module:
            return v
    raise AttributeError(f"no nn.Module subclass found in {module.__file__}")


def _move(v, d):
    import torch
    if isinstance(v, torch.Tensor):
        return v.to(d)
    if isinstance(v, (list, tuple)):
        return type(v)(_move(x, d) for x in v)
    return v


def _clone(v):
    """Deep clone tensor / list of tensors."""
    import torch
    if isinstance(v, torch.Tensor):
        return v.clone()
    if isinstance(v, (list, tuple)):
        return type(v)(_clone(x) for x in v)
    return v


def _read_jsonl_file(path: Path):
    """读取 JSONL 文件，返回解析后的 cases 列表。"""
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def _load_cases_jsonl(out_dir: Path):
    patterns = ["*_perf_cases.jsonl", "*.jsonl"]
    for pattern in patterns:
        jsonl_files = sorted(out_dir.glob(pattern))
        if jsonl_files:
            path = jsonl_files[0]
            cases = _read_jsonl_file(path)
            return cases, f"jsonl:{path.name}"
    return [], None


def _load_cases_json(out_dir: Path):
    """从输出目录中查找并解析 JSON 文件。

    找到第一个非备份、非排除的 JSON 文件，优先按标准 JSON 文档解析（单对象或数组），
    失败时回退到 JSONL 逐行解析。
    """
    json_files = sorted(out_dir.glob("*.json"))
    json_path = None
    for f in json_files:
        if not f.name.endswith(".bak") and f.name not in ("performance.json", "perf_report.json"):
            json_path = f
            break
    if not json_path:
        return [], None

    with open(json_path, "r", encoding="utf-8") as f:
        raw = f.read()
    # 优先按标准 JSON 文档解析（单对象或数组），失败时回退到 JSONL 逐行解析
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            cases = parsed
        elif isinstance(parsed, dict):
            cases = [parsed]
        else:
            raise ValueError("Unexpected JSON root type")
    except ValueError:
        cases = []
        for line in raw.splitlines():
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases, f"json:{json_path.name}"


def _extract_shape_dtype_from_jsonl(case):
    inputs = case.get("inputs", [])
    if not inputs:
        return "?", "?"
    for inp in inputs:
        if inp.get("type") == "tensor":
            return str(inp.get("shape", "?")), inp.get("dtype", "?")
    return str(inputs[0].get("shape", "?")), inputs[0].get("dtype", "?")


def _case_has_empty_tensor(case):
    """判断 case 中是否包含 0 元素张量（空 tensor）。

    shape=None 表示形状未指定（如可选 bias），不应视作空 tensor。
    只有形状明确包含 0 维度或为空列表的才是真正的空 tensor。
    """
    if not case:
        return False
    for inp in case.get("inputs", []):
        if inp.get("type") == "tensor":
            shape = inp.get("shape", [])
            if shape is None:
                continue
            if not shape or any(s == 0 for s in shape):
                return True
    return False


def _resolve_jsonl_scalar_value(inp):
    """Resolve a raw scalar value from a JSONL/KernelBench input descriptor.

    Returns the resolved value or None if no value can be determined.
    """
    val = inp.get("value")
    if val is not None:
        return val
    rv = inp.get("range_values")
    if isinstance(rv, (int, float, bool, complex)):
        return rv
    if isinstance(rv, list) and len(rv) > 0:
        return rv[0]
    if isinstance(rv, dict):
        mean = rv.get("mean")
        if isinstance(mean, list) and len(mean) > 0:
            return mean[0]
    return None


def _jsonl_scalar_default_by_dtype(dtype_str):
    """Return a default scalar value for the given dtype string."""
    if dtype_str == "bool":
        return True
    if dtype_str.startswith("int") or dtype_str.startswith("uint"):
        return 1
    return 1.0


def _cast_jsonl_scalar_by_dtype(val, dtype_str):
    """Cast a resolved scalar value to the Python type indicated by dtype_str."""
    if dtype_str == "str" or isinstance(val, str):
        return str(val)
    if dtype_str == "list" or isinstance(val, list):
        return list(val)
    if dtype_str == "bool":
        return bool(val)
    if dtype_str.startswith("complex"):
        return complex(val)
    if dtype_str.startswith("int") or dtype_str.startswith("uint"):
        return int(val)
    return float(val)


def _jsonl_scalar_value(inp):
    """从 JSONL/KernelBench 标量描述中提取一个 Python 标量值。"""
    val = _resolve_jsonl_scalar_value(inp)
    if val is None:
        val = _jsonl_scalar_default_by_dtype(inp.get("dtype", ""))
    return _cast_jsonl_scalar_by_dtype(val, inp.get("dtype", ""))


# JSONL/KernelBench dtype string → torch dtype expression mapping.
# Shared by _jsonl_tensor_code and _jsonl_tensor_dtype_repr.
_JSONL_DTYPE_MAP = {
    "fp16": "torch.float16", "float16": "torch.float16", "half": "torch.float16",
    "fp32": "torch.float32", "float32": "torch.float32", "float": "torch.float32",
    "fp64": "torch.float64", "float64": "torch.float64",
    "bf16": "torch.bfloat16", "bfloat16": "torch.bfloat16",
    "int8": "torch.int8", "int16": "torch.int16", "int32": "torch.int32",
    "int": "torch.int32", "int64": "torch.int64", "long": "torch.int64",
    "uint8": "torch.uint8", "uint16": "torch.uint16", "uint32": "torch.uint32", "uint64": "torch.uint64",
    "bool": "torch.bool",
    "complex64": "torch.complex64", "complex128": "torch.complex128",
}


def _jsonl_tensor_code(shape, dtype_str: str) -> str:
    """根据 KernelBench 的 dtype/shape 生成构造 tensor 的代码。"""
    dtype = _JSONL_DTYPE_MAP.get(dtype_str, "torch.float32")
    shape_expr = str(shape)

    if dtype_str == "bool":
        return f"inputs.append(torch.randint(0, 2, {shape_expr}, dtype={dtype}))"
    if dtype_str.startswith("int") or dtype_str.startswith("uint"):
        # 先以 int64 生成再转换到目标类型，避免 torch.randint 不支持 uint/低精度 int
        return f"inputs.append(torch.randint(-100, 100, {shape_expr}, dtype=torch.int64).to({dtype}))"
    if dtype_str.startswith("complex"):
        return f"inputs.append(torch.randn({shape_expr}, dtype={dtype}))"
    # float/half/bf16
    return f"inputs.append(torch.randn({shape_expr}, dtype={dtype}))"


def _jsonl_tensor_expr(shape, dtype_str: str) -> str:
    """返回构造 tensor 的表达式字符串（不带 ``inputs.append()`` 包装）。"""
    code = _jsonl_tensor_code(shape, dtype_str)
    # _jsonl_tensor_code returns "inputs.append(TORCH_EXPR)"
    prefix = "inputs.append("
    if not (code.startswith(prefix) and code.endswith(")")):
        raise ValueError(f"unexpected tensor code: {code}")
    return code[len(prefix):-1]  # strip "inputs.append(" and trailing ")"


def _jsonl_tensor_dtype_repr(dtype_str: str) -> str:
    """将 JSONL dtype 字符串映射为 torch dtype 表达式（用于 torch.tensor(..., dtype=...)）。"""
    return _JSONL_DTYPE_MAP.get(dtype_str, "torch.float32")


# attr dtype=str 且值为 torch dtype 名称时，映射到 torch.xxx 表达式（而非字符串字面量）
_TORCH_DTYPE_EXPR_MAP = {
    "float16": "torch.float16", "bfloat16": "torch.bfloat16",
    "float32": "torch.float32", "float64": "torch.float64",
    "int8": "torch.int8", "int16": "torch.int16",
    "int32": "torch.int32", "int64": "torch.int64",
    "uint8": "torch.uint8", "bool": "torch.bool",
}


# 名称提示为 offsets/cumsum 类结构性整型参数的集合（随机合成时保证单调不减）
_OFFSETS_LIKE_NAMES = frozenset({
    "offs", "offsets", "group_list", "group_offsets", "cumsum",
    "cu_seqlens", "cu_seqlens_q", "cu_seqlens_k", "seqlens", "seq_lens",
})


def _append_tensor_input(lines, inp, name):
    """Append a tensor-type input line to *lines*, keeping nesting depth ≤ 4."""
    dtype_str = inp.get("dtype", "float16")
    # 优先使用显式 value（如 group_list / cumsum 等结构参数）
    explicit_val = inp.get("value")
    if explicit_val is not None:
        dtype_repr = _jsonl_tensor_dtype_repr(dtype_str)
        lines.append(f"_inputs_dict[{repr(name)}] = torch.tensor({repr(explicit_val)}, dtype={dtype_repr})")
        return
    shape = inp.get("shape", [])
    # offsets/cumsum 类结构性整型参数：完全随机的值几乎必然违反
    # 单调性约束导致被测实现直接抛异常，退化为单调不减序列更稳妥
    if name.lower() in _OFFSETS_LIKE_NAMES and dtype_str.startswith(("int", "uint")):
        dtype_repr = _jsonl_tensor_dtype_repr(dtype_str)
        lines.append(f"_inputs_dict[{repr(name)}] = "
                     f"torch.randint(0, 100, {shape}, dtype=torch.int64).cumsum(0).to({dtype_repr})")
        return
    tensor_expr = _jsonl_tensor_expr(shape, dtype_str)
    lines.append(f"_inputs_dict[{repr(name)}] = {tensor_expr}")


def _append_attr_scalar_input(lines, inp, name):
    """Append an attr/scalar-type input line to *lines*, keeping nesting depth ≤ 4."""
    # 显式 "value": null 表示可选参数未启用（如 offsets=None），
    # 必须生成 None 而不是默认标量，否则会把 None 语义错传成 1.0/1
    if "value" in inp and inp.get("value") is None:
        lines.append(f"_inputs_dict[{repr(name)}] = None")
        return
    val = _jsonl_scalar_value(inp)
    # attr 的 dtype 为 str 且值是 torch dtype 名时，生成 torch.xxx 表达式而非字符串字面量
    if inp.get("dtype") == "str" and isinstance(val, str):
        dtype_expr = _TORCH_DTYPE_EXPR_MAP.get(val)
        if dtype_expr is not None:
            lines.append(f"_inputs_dict[{repr(name)}] = {dtype_expr}")
            return
    lines.append(f"_inputs_dict[{repr(name)}] = {repr(val)}")


def _serialize_jsonl_inputs(case):
    """Generate kwargs dict code from JSONL inputs, using 'name' fields as keys.

    Uses named keyword arguments (``model(**kwargs)``) so that parameter binding
    is independent of the order of inputs in the JSONL file.
    """
    lines = ["_inputs_dict = {}"]
    for inp in case.get("inputs", []):
        name = inp.get("name", "")
        typ = inp.get("type", "tensor")
        if typ == "tensor":
            _append_tensor_input(lines, inp, name)
        elif typ in ("attr", "scalar"):
            _append_attr_scalar_input(lines, inp, name)
        elif typ == "tensor_list":
            lines.append("_tensors = []")
            for tinfo in inp.get("value", []):
                shape = tinfo.get("shape", [])
                dtype_str = tinfo.get("dtype", "float16")
                tensor_expr = _jsonl_tensor_expr(shape, dtype_str)
                lines.append(f"_tensors.append({tensor_expr})")
            lines.append(f"_inputs_dict[{repr(name)}] = _tensors")
        else:
            val = _jsonl_scalar_value(inp)
            lines.append(f"_inputs_dict[{repr(name)}] = {repr(val)}")
    return "\n".join(lines)


@dataclass
class _WrapperConfig:
    """封装 _generate_wrapper_script 的参数。"""
    out_dir: Path
    case_idx: int
    impl: str
    seed: int
    device_id: int
    warmup: int
    jsonl_case: Optional[Dict[str, Any]] = None
    repeats: int = 1


_WRAPPER_SCRIPT_TEMPLATE = """\
#!/usr/bin/env python3
import importlib.util
import inspect
import logging
import os
import sys
import torch
from pathlib import Path

LOGGER = logging.getLogger(__name__)

out_dir = Path("{out_dir}")
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "{device_id}"
sys.path.insert(0, str(out_dir / "kernel" / "build"))
sys.path.insert(0, str(out_dir))

torch.manual_seed({seed})
try:
    torch.npu.manual_seed_all({seed})
except Exception:
    pass

def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

mod = _load(out_dir / "{model_file}", "prof_mod")
cls = getattr(mod, "{cls_name}")

{inputs_code}

device = torch.device("npu")
def _move(v):
    if isinstance(v, torch.Tensor):
        return v.to(device)
    if isinstance(v, (list, tuple)):
        return type(v)(_move(x) for x in v)
    return v

_use_kwargs = inputs is None
if _use_kwargs:
    for _k, _v in _inputs_dict.items():
        if isinstance(_v, torch.Tensor):
            _inputs_dict[_k] = _v.to(device)
        elif isinstance(_v, list):
            _inputs_dict[_k] = [_move(x) for x in _v]
    # build positional list from dict values (preserving JSONL order) for fallback
    _inputs_list = list(_inputs_dict.values())
else:
    inputs = _move(inputs)

# ---- model construction (honour get_init_inputs if present) ----
_init_args = []
_init_kwargs = {{}}
try:
    ref_for_init = _load(out_dir / "{model_file}", "ref_for_init")
    if hasattr(ref_for_init, "get_init_inputs"):
        _all_init = ref_for_init.get_init_inputs()
        _idx = {case_idx}
        if isinstance(_all_init, list) and _idx < len(_all_init):
            _entry = _all_init[_idx]
            if isinstance(_entry, dict):
                _init_kwargs = _entry
            elif isinstance(_entry, (list, tuple)):
                _init_args = list(_entry)
except Exception:
    pass

if _init_kwargs:
    model = cls(**_init_kwargs).to(device).eval()
elif _init_args:
    model = cls(*_init_args).to(device).eval()
else:
    model = cls().to(device).eval()

def _bind_ok(sig, args, kwargs):
    try:
        sig.bind(*args, **kwargs)
        return True
    except TypeError:
        return False


_forward_sig = None
try:
    _forward_sig = inspect.signature(model.forward)
except Exception:
    pass

# 0=kwargs, 1=spread, 2=tuple
# 先按 forward 签名预绑定选择调用方式：仅当确认参数确实绑定不上时才降级；
# 运行期调用抛出的 TypeError 视为模型内部真实错误，直接失败，不再静默降级
# （否则 forward 内部的业务 TypeError 会被误判成"参数形式不匹配"而测错调用方式）。
_use_positional_fallback = 0
if _use_kwargs and _forward_sig is not None:
    if _bind_ok(_forward_sig, (), _inputs_dict):
        _use_positional_fallback = 0
    elif _bind_ok(_forward_sig, tuple(_inputs_list), dict()):
        _use_positional_fallback = 1
    elif _bind_ok(_forward_sig, (tuple(_inputs_list),), dict()):
        _use_positional_fallback = 2


def _one_iter():
    with torch.no_grad():
        if _use_kwargs:
            if _use_positional_fallback == 0:
                _ = model(**_inputs_dict)
            elif _use_positional_fallback == 1:
                _ = model(*_inputs_list)
            else:
                _ = model(tuple(_inputs_list))
        else:
            _ = model(*inputs)
    torch.npu.synchronize()


# ---- in-process warmup: 与 torch_npu profiler 的 warmup 语义一致 ----
# 这些迭代同样落在 msprof 采集窗口内，解析时按 run 拆分后丢弃前 {warmup} 次。
# 任一迭代失败（含模型内部异常）时记录 case 编号后优雅退出，不打印原始 traceback。
try:
    for _ in range({warmup}):
        _one_iter()

    # ---- timed repeats ----
    for _ in range({repeats}):
        _one_iter()
except Exception as _e:
    LOGGER.info("[CASE %s] model call failed, skip case: %s", {case_idx}, _e)
    sys.exit(1)
"""


def _build_wrapper_script_content(cfg, model_file, cls_name, inputs_code):
    """Build the wrapper script string from components."""
    return _WRAPPER_SCRIPT_TEMPLATE.format(
        out_dir=cfg.out_dir,
        device_id=cfg.device_id,
        case_idx=cfg.case_idx,
        warmup=cfg.warmup,
        repeats=cfg.repeats,
        seed=cfg.seed,
        model_file=model_file,
        cls_name=cls_name,
        inputs_code=inputs_code,
    )


# 优先从 model.py 自带的输入生成器取真实输入（保证 offsets/cumsum 等结构性
# 参数合法，且与 evaluate 输入同源）；取不到时 inputs 保持 None，回退到
# JSONL 随机合成的 _inputs_dict kwargs 路径。
_REAL_INPUTS_CODE = """\
# Prefer real inputs from model.py's own generator (valid structural params,
# same source as evaluate); fall back to serialized kwargs when unavailable.
inputs = None
try:
    _ref_mod = _load(out_dir / "model.py", "ref_for_inputs")
    _gen = getattr(_ref_mod, "get_input_groups", getattr(_ref_mod, "get_inputs", None))
    if _gen is not None:
        _all_input_groups = _gen()
        if 0 <= {case_idx} < len(_all_input_groups):
            _entry = _all_input_groups[{case_idx}]
            inputs = list(_entry) if isinstance(_entry, (list, tuple)) else [_entry]
except Exception:
    inputs = None
"""


def _generate_wrapper_script(cfg: _WrapperConfig):
    if cfg.impl == "reference":
        model_file = "model.py"
        cls_name = "Model"
    else:
        model_file = "model_new_ascendc.py"
        cls_name = "ModelNew"

    if cfg.jsonl_case is not None:
        # 先准备 JSONL 随机合成的 kwargs 作为兜底，再尝试真实输入覆盖
        inputs_code = (_serialize_jsonl_inputs(cfg.jsonl_case) + "\n\n"
                       + _REAL_INPUTS_CODE.format(case_idx=cfg.case_idx))
    else:
        # 无 JSONL case 时只能依赖 model.py 的输入生成器
        inputs_code = (_REAL_INPUTS_CODE.format(case_idx=cfg.case_idx) + """
if inputs is None:
    raise RuntimeError("failed to obtain inputs from model.py get_input_groups()/get_inputs()")
""")
    return _build_wrapper_script_content(cfg, model_file, cls_name, inputs_code)


def _find_msprof_script():
    script_dir = Path(__file__).resolve().parent
    candidate = script_dir / "msprof_profile_run.sh"
    if candidate.exists():
        return str(candidate)
    return "msprof_profile_run.sh"


def _save_app_output(output_dir: str, stdout: str, stderr: str) -> str:
    """把被测 app 的输出落盘，便于采集失败时回溯真实原因。返回日志路径。"""
    log_path = os.path.join(output_dir, "app_output.log")
    try:
        with open(log_path, "w", encoding="utf-8", errors="replace") as f:
            f.write("=== stdout ===\n" + (stdout or "") + "\n=== stderr ===\n" + (stderr or ""))
    except OSError:
        pass
    return log_path


def _extract_app_crash(stdout: str, stderr: str):
    """从 msprof 输出中检测被测 app 的 Python 异常。

    msprof 在 app 崩溃时仍返回退出码 0（只打 WARNING），若不主动检测，
    上层只能看到 "no csv found" 这类误导性错误。返回最后一行异常信息或 None。
    """
    text = (stdout or "") + "\n" + (stderr or "")
    if "Traceback" not in text and "An exception has occurred in process App" not in text:
        return None
    exc_lines = re.findall(r"^(\w[\w.]*(?:Error|Exception|Interrupt)\b[^\n]*)", text, re.M)
    if exc_lines:
        return exc_lines[-1].strip()[:200]
    return "app raised an exception during profiling"


def _run_msprof_standard(wrapper_script: str, output_dir: str, device_id: int, warmup: int = 3):
    """调用 msprof_profile_run.sh 进行完整采集（7 组 aic-metrics + sample-based）。

    注意：不设置 timeout，与原来 kernel_perf.py 行为一致（默认无限等待）。
    8 轮采集（7 metrics + 1 sample）耗时较长，由调用方控制整体超时。
    """
    wrapper_path = os.path.join(output_dir, "_wrapper.py")
    os.makedirs(output_dir, exist_ok=True)
    with open(wrapper_path, "w", encoding="utf-8") as f:
        f.write(wrapper_script)

    cmd = [
        "bash", _find_msprof_script(),
        f"--warm-up={warmup}",
        f"--output={output_dir}",
        "--",
        sys.executable, wrapper_path
    ]
    env = os.environ.copy()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    try:
        os.remove(wrapper_path)
    except OSError:
        pass

    app_log = _save_app_output(output_dir, result.stdout, result.stderr)

    if result.returncode != 0:
        return None, f"msprof failed: {result.stderr[-500:]}\n(app log: {app_log})"

    app_crash = _extract_app_crash(result.stdout, result.stderr)
    if app_crash:
        return None, f"profiled app crashed: {app_crash} (app log: {app_log})"

    prof_dirs = sorted(Path(output_dir).glob("PROF_GROUP_*"))
    if not prof_dirs:
        return None, f"no PROF_GROUP directory found (app log: {app_log})"
    return str(prof_dirs[-1]), None


def _run_msprof_quick(wrapper_script: str, output_dir: str, device_id: int, warmup: int = 3):
    """快速模式：只采集 1 轮（不采集 7 个 aic-metrics，只获取 kernel 时间）。

    直接调用 msprof 命令（不通过 msprof_profile_run.sh，避免循环调用）。
    使用 msprof --ai-core=on --task-time=on --ascendcl=on，不设置 --aic-metrics。

    warmup 在 msprof 外部执行：先单独跑 warmup 次 wrapper 预热 NPU，
    msprof 只采集正式 timed run，确保 task_time.csv 不含预热数据。
    """
    wrapper_path = os.path.join(output_dir, "_wrapper.py")
    os.makedirs(output_dir, exist_ok=True)
    with open(wrapper_path, "w", encoding="utf-8") as f:
        f.write(wrapper_script)

    env = os.environ.copy()

    # Warmup: 在 msprof 外部执行，不采集
    warmup_crash = None
    for _ in range(warmup):
        w = subprocess.run([sys.executable, wrapper_path],
                           capture_output=True, text=True, env=env)
        if w.returncode != 0 and warmup_crash is None:
            warmup_crash = _extract_app_crash(w.stdout, w.stderr) or (w.stderr or "")[-200:]

    # Measurement: msprof 只采集正式 timed run
    cmd = [
        "msprof",
        f"--output={output_dir}",
        "--ai-core=on",
        "--task-time=on",
        "--ascendcl=on",
        sys.executable, wrapper_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    try:
        os.remove(wrapper_path)
    except OSError:
        pass

    app_log = _save_app_output(output_dir, result.stdout, result.stderr)

    if result.returncode != 0:
        return None, f"msprof failed: {result.stderr[-500:]}\n(app log: {app_log})"

    # msprof 在 app 崩溃时仍返回 0，必须主动检测，否则只能看到误导性的
    # "no task_time csv found"
    app_crash = _extract_app_crash(result.stdout, result.stderr) or warmup_crash
    if app_crash:
        return None, f"profiled app crashed: {app_crash} (app log: {app_log})"

    prof_dirs = sorted(Path(output_dir).glob("PROF_*"))
    if not prof_dirs:
        return None, f"no PROF directory found (app log: {app_log})"
    return str(prof_dirs[-1]), None


def _extract_compute_row(r: dict):
    """从 op_summary 行提取计算算子的 (duration, op_name)。

    非计算行、非法值或非正耗时返回 None。
    """
    task_type = r.get("Task Type", "")
    if "AI_CORE" not in task_type and "AIV" not in task_type and "MIX" not in task_type:
        return None
    try:
        duration = float(r.get("Task Duration(us)", 0) or 0)
    except ValueError:
        return None
    if duration <= 0:
        return None
    return duration, r.get("Op Name", "unknown")


def _parse_msprof_duration(prof_group_dir: str):
    csv_pattern = os.path.join(prof_group_dir, "PROF_*/PROF_*/mindstudio_profiler_output/op_summary_*.csv")
    csv_files = sorted(glob.glob(csv_pattern))
    if not csv_files:
        return None, None, "no op_summary csv found"

    try:
        with open(csv_files[0], "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        return None, None, f"read csv error: {e}"

    if not rows:
        return None, None, "empty csv"

    # 汇总所有实际计算算子的 Task Duration；参考实现可能包含多个 PyTorch native kernel，
    # AscendC 实现也可能拆分为多个 kernel，因此采用累加而非取最大。
    compute_rows = [cr for cr in (_extract_compute_row(r) for r in rows) if cr is not None]

    if compute_rows:
        total_duration = sum(d for d, _ in compute_rows)
        op_name = compute_rows[0][1] if len(compute_rows) == 1 else "multiple_kernels"
        return total_duration, op_name, None

    return None, None, "no compute rows found"


_META_KERNEL_TYPES = ("PROFILING_ENABLE", "PROFILING_DISABLE", "TASK_TIMEOUT_SET", "")


def _split_task_time_runs(rows: List[Dict[str, str]], n_runs: int):
    """把 task_time.csv 的 kernel 行按「重复的 kernel 序列」拆成 n_runs 次 run。

    采集窗口内包含 warmup + repeats 次迭代，每次迭代发射完全相同的 kernel 序列。
    按 task_start 排序后，总行数应能被 n_runs 整除，且每个分块的 kernel_name
    序列必须逐位相同 —— 否则说明迭代次数与预期不符，返回 None 由调用方回退。

    Returns: list[list[float]] 每个 run 的 kernel 耗时列表，或 None。
    """
    kernels = []
    for r in rows:
        if r.get("kernel_type", "") in _META_KERNEL_TYPES:
            continue
        try:
            dur = float(r.get("task_time(us)", "") or 0)
            start = float((r.get("task_start(us)", "") or "0").strip())
        except ValueError:
            continue
        if dur <= 0:
            continue
        kernels.append((start, r.get("kernel_name", "unknown"), dur))

    if not kernels or n_runs < 1 or len(kernels) % n_runs:
        return None
    kernels.sort(key=lambda x: x[0])

    n = len(kernels) // n_runs
    runs, seq = [], [k[1] for k in kernels[:n]]
    for i in range(n_runs):
        chunk = kernels[i * n:(i + 1) * n]
        if [k[1] for k in chunk] != seq:
            return None
        runs.append([k[2] for k in chunk])
    return runs


def _parse_msprof_duration_quick(prof_group_dir: str, warmup: int = 0, repeats: int = 1):
    """快速模式解析：从 PROF 目录中查找 task_time.csv 并提取时间。

    快速模式只跑 1 轮（不采集 7 个 aic-metrics），因此没有 op_summary_*.csv。

    采集窗口内包含 warmup + repeats 次 in-process 迭代（与 torch_npu profiler 的
    warmup/active 语义一致）。这里按 kernel 序列把它们拆开，**只对后 repeats 次
    （active）求平均**，丢弃 warmup 次 —— 避免首次迭代的冷启动污染测量值。
    若拆分失败（行数不匹配等），回退为按时间排序后的位置分片，同样丢弃 warmup 份。

    兼容两种目录结构：
      - 无 --aic-metrics: PROF_*/mindstudio_profiler_output/task_time_*.csv
      - 有 --aic-metrics: PROF_*/PROF_PipeUtilization/mindstudio_profiler_output/task_time_*.csv
    """
    task_time_patterns = [
        os.path.join(prof_group_dir, "mindstudio_profiler_output", "task_time_*.csv"),
        os.path.join(prof_group_dir, "PROF_*", "mindstudio_profiler_output", "task_time_*.csv"),
    ]
    for pattern in task_time_patterns:
        task_time_files = sorted(glob.glob(pattern))
        if not task_time_files:
            continue
        try:
            with open(task_time_files[0], "r", encoding="utf-8", errors="replace") as f:
                rows = list(csv.DictReader(f))

            # 优先：按 run 拆分，只取 active（后 repeats 次）的均值
            runs = _split_task_time_runs(rows, warmup + repeats)
            if runs is not None:
                active = runs[warmup:] if warmup < len(runs) else runs
                per_run = [sum(x) for x in active]
                return (sum(per_run) / len(per_run)), "multiple_kernels", None

            # 回退路径：run 拆分失败（kernel 名称序列不一致等），位置分片
            kernels = _collect_kernels(rows)
            result = _split_kernels_by_position(kernels, warmup, repeats)
            if result is None:
                continue
            kernel_names = list({k[1] for k in kernels})
            kernel_name = kernel_names[0] if len(kernel_names) == 1 else "multiple_kernels"
            return result, kernel_name, None
        except Exception as e:
            LOGGER.debug("Failed to parse %s: %s", task_time_files[0], e)

    return None, None, "no task_time or api_statistic csv found"


def _collect_kernels(rows):
    """从 task_time 行中收集有效 kernel：过滤元数据 kernel、空/非法/非正耗时。"""
    kernels = []
    for r in rows:
        kernel_type = r.get("kernel_type", "")
        task_time = r.get("task_time(us)", "")
        if kernel_type in _META_KERNEL_TYPES or not task_time:
            continue
        try:
            dur = float(task_time)
            start = float((r.get("task_start(us)", "") or "0").strip())
        except ValueError:
            continue
        if dur <= 0:
            continue
        kernels.append((start, r.get("kernel_name", "unknown"), dur))
    return kernels


def _split_kernels_by_position(kernels, warmup: int, repeats: int):
    """按时间排序后做位置分片，丢弃前 warmup 份，返回单次迭代估算耗时。

    行数可整除时按位置精确分片；不可整除时按比例估算。kernels 为空返回 None。
    """
    if not kernels:
        return None
    kernels.sort(key=lambda x: x[0])
    total = sum(d for _, _, d in kernels)
    n_total = warmup + repeats
    if n_total <= 0:
        n_total = 1
    n_kernels = len(kernels)

    if n_kernels % n_total == 0:
        # 行数可整除：按位置精确分片，丢弃前 warmup 份
        k_per = n_kernels // n_total
        active_kernels = kernels[warmup * k_per:]
        return sum(d for _, _, d in active_kernels) / max(1, repeats)

    # 行数不可整除：按比例估算，丢弃时间轴上前 warmup/n_total 的部分
    split_idx = max(0, int(n_kernels * warmup / n_total))
    active_kernels = kernels[split_idx:]
    if not active_kernels:
        return total / n_total
    # 按 active kernel 数占总 kernel 数的比例折算单次迭代耗时
    active_total = sum(d for _, _, d in active_kernels)
    return active_total / (len(active_kernels) / (n_kernels / n_total))


def pick_idle_npu(default=0):
    try:
        p = subprocess.run(["/usr/local/bin/npu-smi", "info"], capture_output=True, text=True, timeout=10)
        if p.returncode != 0:
            return default
    except Exception:
        return default

    devices = {}
    cur = None
    head_re = re.compile(r"^\|\s+(\d+)\s+\S+\s+\|\s+\w+\s+\|")
    bus_re = re.compile(
        r"^\|\s+\d+\s+\|\s+[0-9A-Fa-f:.]+\s+\|\s+(\d+)\s+(\d+)\s*/\s*(\d+)(?:\s+(\d+)\s*/\s*(\d+))?"
    )
    for line in p.stdout.splitlines():
        m2 = bus_re.match(line)
        if m2 and cur is not None:
            aicore = int(m2.group(1))
            mem_used = int(m2.group(2))
            mem_total = max(int(m2.group(3)), 1)
            hbm_used = int(m2.group(4)) if m2.group(4) else 0
            hbm_total = max(int(m2.group(5)), 1) if m2.group(5) else 1
            mem_ratio = max(mem_used / mem_total, hbm_used / hbm_total)
            devices[cur] = (aicore, mem_ratio)
            cur = None
            continue
        m1 = head_re.match(line)
        if m1:
            cur = int(m1.group(1))

    if not devices:
        return default
    best_id, _ = min(devices.items(), key=lambda kv: (kv[1][0], kv[1][1]))
    return best_id


def _add_compare_header(lines, report):
    """Add the header section to the compare markdown report."""
    lines.append("# 性能评估结果")
    lines.append("")
    lines.append(f"- **Operator**: {report['task']}")
    lines.append(f"- **Device**: npu:{report['device_id']} (source={report['device_select_source']})")
    lines.append(f"- **Warmup**: {report['warmup']}")
    lines.append(f"- **Repeats**: {report['repeats']}")
    lines.append(f"- **Seed**: {report['seed']}")
    lines.append(f"- **Timing method**: {report['timing_method']}")
    lines.append("")


def _add_per_case_table(lines, report):
    """Add the per-case comparison table."""
    if not report.get("per_case"):
        return
    lines.append("## 性能对比")
    lines.append("")
    lines.append("| Case | Shape | DType | 自定义算子(us) | 标杆(us) | 加速比 |")
    lines.append("| ---- | ----- | ----- | ------------- | -------- | -------------- |")
    for case in report["per_case"]:
        shape = case.get("shape", "?")
        dtype = case.get("dtype", "?")
        ref = case.get("ref_us")
        asc = case.get("asc_us")
        sp = case.get("speedup")
        ref_str = f"{ref:.2f}" if ref is not None else "N/A"
        asc_str = f"{asc:.2f}" if asc is not None else "N/A"
        sp_str = f"{sp:.3f}" if sp is not None else "N/A"
        lines.append(f"| {case['case']} | {shape} | {dtype} | {asc_str} | {ref_str} | {sp_str} |")
    lines.append("")


def _add_summary_section(lines, report):
    """Add the summary and dtype tables."""
    has_speedup = report.get("geomean_speedup") is not None
    has_timing = (report.get("mean_ref_us") is not None
                  or report.get("mean_asc_us") is not None)
    if not has_speedup and not has_timing:
        return
    lines.append("## 全量汇总")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("| ---- | -- |")
    lines.append(f"| 用例数 | {report['n_cases_total']} |")
    if has_speedup:
        lines.append(f"| 平均加速比（>1 表示自定义算子更快） | {report['mean_speedup']:.3f} |")
    else:
        lines.append("| 平均加速比 | N/A（--impl 单边采集或无双侧有效 case） |")
    if report.get("geomean_ref_us") is not None:
        lines.append(f"| 标杆几何平均耗时 (us) | {report['geomean_ref_us']:.2f} |")
    if report.get("geomean_asc_us") is not None:
        lines.append(f"| 自定义算子几何平均耗时 (us) | {report['geomean_asc_us']:.2f} |")
    if has_speedup:
        better = sum(1 for c in report.get('per_case', []) if c.get('speedup') and c['speedup'] > 1)
        worse = sum(1 for c in report.get('per_case', []) if c.get('speedup') and c['speedup'] < 1)
        lines.append(f"| 自定义算子更优（比值>1） | {better} |")
        lines.append(f"| 标杆更优（比值<1） | {worse} |")
    lines.append("")

    dtype_groups = {}
    for case in report.get("per_case", []):
        dtype = case.get("dtype", "?")
        sp = case.get("speedup")
        if sp is not None:
            dtype_groups.setdefault(dtype, []).append(sp)
    if dtype_groups:
        lines.append("### 按数据类型汇总")
        lines.append("")
        lines.append("| DType | 用例数 | 平均加速比 | 自定义算子更优 | 标杆更优 |")
        lines.append("| ----- | ------ | ------------------- | ------------- | -------- |")
        for dtype, sps in sorted(dtype_groups.items()):
            mean_sp = statistics.mean(sps)
            better = sum(1 for sp in sps if sp > 1)
            worse = sum(1 for sp in sps if sp < 1)
            lines.append(f"| {dtype} | {len(sps)} | {mean_sp:.3f} | {better} | {worse} |")
        lines.append("")


def _add_analysis_sections(lines, report):
    """Add the short analysis and deep bottleneck analysis sections."""
    lines.append("## 简短分析")
    lines.append("")
    if report.get("mean_speedup") is not None:
        if report["mean_speedup"] > 1:
            lines.append(f"- 平均加速比 {report['mean_speedup']:.3f} 大于 1，自定义算子整体有优势。")
        else:
            lines.append(f"- 平均加速比 {report['mean_speedup']:.3f} 小于 1，标杆路径整体更优。")
    lines.append("- 详细瓶颈分析见 msprof 归档目录（op_summary_*.csv + summary.txt）。")
    lines.append("")

    lines.append("## 深度瓶颈分析")
    lines.append("")
    lines.append(
        "如需进一步分析性能瓶颈（各流水线利用率、核间负载均衡、主 Bound 判定），"
        "可运行："
    )
    lines.append("```bash")
    lines.append(
        f"python3 ${{SKILL_PATH}}/scripts/msprof_perf_summary.py "
        f"{report.get('prof_group_dir', './PROF_GROUP_*')} {report['task']}"
    )
    lines.append("```")
    lines.append("")
    lines.append("或参考 `ops-profiling/references/optimization_quickref.md` 获取优化建议。")
    lines.append("")


def _report_compare_to_markdown(report: Dict[str, Any]) -> str:
    lines = []
    _add_compare_header(lines, report)
    _add_per_case_table(lines, report)
    _add_summary_section(lines, report)
    _add_analysis_sections(lines, report)
    return "\n".join(lines)


def _report_compare_to_text(report: Dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 100)
    lines.append(f"Kernel-level Performance (msprof): {report['task']}  "
                 f"(warmup={report['warmup']}, repeats={report['repeats']}, seed={report['seed']})")
    lines.append("=" * 100)
    lines.append(f"{'Case':<5} {'Shape':<35} {'dtype':<10} {'Ref(us)':>12} {'Asc(us)':>12} {'Speedup':>10}")
    lines.append("-" * 100)

    for case in report.get("per_case", []):
        shape = case.get("shape", "?")
        dtype = case.get("dtype", "?")
        ref_us = case.get("ref_us")
        asc_us = case.get("asc_us")
        sp = case.get("speedup")
        if ref_us is not None and asc_us is not None and sp is not None:
            lines.append(f"{case['case']:<5} {shape:<35} {dtype:<10} {ref_us:>12.2f} {asc_us:>12.2f} {sp:>9.3f}x")
        else:
            ref_str = f"{ref_us:.2f}" if ref_us is not None else "N/A"
            asc_str = f"{asc_us:.2f}" if asc_us is not None else "N/A"
            ref_err = case.get("ref_error", "")
            asc_err = case.get("asc_error", "")
            lines.append(f"{case['case']:<5} {shape:<35} {dtype:<10} "
                         f"{ref_str:>12} {asc_str:>12} "
                         f"{'N/A':>10}  (ref_err={ref_err}, asc_err={asc_err})")

    lines.append("-" * 100)
    if report.get("geomean_speedup") is not None:
        lines.append("--- Speedup ---")
        lines.append(f"  Geomean : {report['geomean_speedup']:.2f}x  ← 主指标")
        lines.append(f"  Mean    : {report['mean_speedup']:.2f}x")
        lines.append(f"  Median  : {report['median_speedup']:.2f}x")
        lines.append(f"  Min/Max : {report['min_speedup']:.2f}x / {report['max_speedup']:.2f}x")
        lines.append(f"  Valid   : {report['n_cases_valid']}/{report['n_cases_total']}")
    # 单边采集（--impl ref/asc）时只渲染已测到的一侧，未测的一侧不出现
    if report.get("mean_ref_us") is not None:
        lines.append("--- Task Duration (us) ---")
        lines.append(
            f"  Ref  mean/median/geomean/total : {report['mean_ref_us']:.2f} / "
            f"{report['median_ref_us']:.2f} / {report['geomean_ref_us']:.2f} / "
            f"{report['total_ref_us']:.2f}"
        )
    if report.get("mean_asc_us") is not None:
        lines.append(
            f"  Asc  mean/median/geomean/total : {report['mean_asc_us']:.2f} / "
            f"{report['median_asc_us']:.2f} / {report['geomean_asc_us']:.2f} / "
            f"{report['total_asc_us']:.2f}"
        )
    if report.get("total_speedup") is not None:
        lines.append(f"  Total speedup (Σref/Σasc) : {report['total_speedup']:.2f}x")
    lines.append("=" * 100)

    return "\n".join(lines)


def _select_device_id(args):
    """Select NPU device from CLI arg, env var, or auto-detect."""
    if args.device is not None:
        return args.device, "cli"
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        return int(os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")[0]), "env"
    return pick_idle_npu(default=0), "auto"


@dataclass
class _MeasureInput:
    """封装 _measure_one_impl 的测量参数。

    Args:
        out_dir: 算子输出目录（Path）
        case_idx: case 索引号
        impl: 实现类型，\"reference\" 或 \"ascendc\"
        args: argparse.Namespace（需含 retry, seed, warmup 属性）
        device_id: NPU 设备 ID
        jsonl_case: JSONL case 字典，可选
    """
    out_dir: Path
    case_idx: int
    impl: str
    args: argparse.Namespace
    device_id: int
    jsonl_case: Optional[Dict[str, Any]] = None


def _measure_one_impl(mi: _MeasureInput):
    """Measure one implementation (reference/ascendc) with retries.

    Returns (duration_us, error, prof_dir).
    """
    prof_dir = None
    impl_abbr = "ref" if mi.impl == "reference" else "asc"
    for _ in range(1 + mi.args.retry):
        wrapper = _generate_wrapper_script(_WrapperConfig(
            mi.out_dir, mi.case_idx, mi.impl, mi.args.seed, mi.device_id,
            mi.args.warmup, mi.jsonl_case))
        tmpdir = f"/tmp/msprof_{impl_abbr}_{mi.out_dir.name}_c{mi.case_idx}"
        prof_dir, err = _run_msprof_standard(wrapper, tmpdir, mi.device_id, mi.args.warmup)
        if prof_dir:
            duration, _op_name, parse_err = _parse_msprof_duration(prof_dir)
            if duration is not None:
                return duration, None, prof_dir
            err = f"{parse_err} (app log: {tmpdir}/app_output.log)"
        time.sleep(0.5)
    return None, err, prof_dir


def _measure_one_impl_quick(mi: _MeasureInput):
    """快速模式：Measure one implementation with retries and repeats.

    --retry 用于解析失败重试。

    与 torch_npu profiler 对齐的口径：wrapper 在**同一个被采集的进程内**先跑
    --warmup 次预热迭代、再跑 --repeats 次计时迭代，msprof 一把全采；解析时按
    kernel 序列拆分成 (warmup + repeats) 次 run，只对后 repeats 次求平均。

    旧实现把 warmup 放在 msprof 外部的独立进程里执行，对被采集的那个进程毫无
    预热作用，导致采到的必然是冷 run（首个 kernel 被放大 ~47%）。--ext-warmup
    保留了旧行为，默认关闭。

    Returns (duration_us, error, prof_dir).
    """
    prof_dir = None
    impl_abbr = "ref" if mi.impl == "reference" else "asc"
    repeats = max(1, getattr(mi.args, "repeats", 1))
    warmup = max(0, getattr(mi.args, "warmup", 0))
    ext_warmup = max(0, getattr(mi.args, "ext_warmup", 0))

    for _ in range(1 + mi.args.retry):
        wrapper = _generate_wrapper_script(_WrapperConfig(
            mi.out_dir, mi.case_idx, mi.impl, mi.args.seed, mi.device_id,
            warmup, mi.jsonl_case, repeats))
        tag = getattr(mi.args, "prof_tag", None)
        if tag:
            if not _PROF_TAG_RE.match(tag):
                raise ValueError(
                    f"--prof-tag 只允许 [A-Za-z0-9_.-]+，收到 {tag!r}；"
                    f"该值会拼进临时目录名并在随后被 rmtree 清理，禁止路径分隔符")
            tag_suffix = f"_{tag}"
        else:
            tag_suffix = ""
        tmpdir = (f"/tmp/msprof_quick_{impl_abbr}_{mi.out_dir.name}_c{mi.case_idx}"
                  + tag_suffix)
        _cleanup_prof_dirs(tmpdir)
        prof_dir, err = _run_msprof_quick(wrapper, tmpdir, mi.device_id, ext_warmup)
        if not prof_dir:
            continue

        duration, _op_name, parse_err = _parse_msprof_duration_quick(
            prof_dir, warmup, repeats)
        if duration is None:
            err = f"{parse_err} (app log: {tmpdir}/app_output.log)"
            continue

        return duration, None, prof_dir

    return None, err, prof_dir


@dataclass
class _CompareSummaryInput:
    """封装 _compute_compare_summary 的汇总计算参数。

    Args:
        out_dir: 算子输出目录（Path）
        rows: 逐 case 的测量结果列表
        speedups: 有效 speedup 值列表
        ref_times: 参考实现耗时列表（us）
        asc_times: AscendC 实现耗时列表（us）
        n_cases: case 总数
        args: argparse.Namespace（需含 warmup, repeats, seed 属性）
        device_id: NPU 设备 ID
        device_src: 设备来源描述（cli/env/auto）
    """
    out_dir: Path
    rows: list
    speedups: list
    ref_times: list
    asc_times: list
    n_cases: int
    args: argparse.Namespace
    device_id: int
    device_src: str


def _compute_compare_summary(csi: _CompareSummaryInput):
    """Compute the summary statistics dict for compare mode."""
    speedup_stats = _compute_speedup_stats(csi.speedups)
    timing_stats = _compute_timing_stats(csi.ref_times, csi.asc_times)
    return {
        "task": csi.out_dir.name,
        "task_dir": str(csi.out_dir),
        "n_cases_total": csi.n_cases,
        **speedup_stats,
        **timing_stats,
        "warmup": csi.args.warmup,
        "repeats": csi.args.repeats,
        "seed": csi.args.seed,
        "device_id": csi.device_id,
        "device_select_source": csi.device_src,
        "timing_method": "msprof.op_summary.Task_Duration",
        "per_case": csi.rows,
    }


def _compute_speedup_stats(speedups: list) -> dict:
    """Compute speedup statistics from a list of speedup values.

    Returns a dict with n_cases_valid and geomean/mean/median/min/max speedup.
    """
    if not speedups:
        return {
            "n_cases_valid": 0,
            "geomean_speedup": None,
            "mean_speedup": None,
            "median_speedup": None,
            "min_speedup": None,
            "max_speedup": None,
        }
    return {
        "n_cases_valid": len(speedups),
        "geomean_speedup": statistics.geometric_mean(speedups),
        "mean_speedup": statistics.mean(speedups),
        "median_speedup": statistics.median(speedups),
        "min_speedup": min(speedups),
        "max_speedup": max(speedups),
    }


def _compute_timing_stats(ref_times: list, asc_times: list) -> dict:
    """Compute timing statistics for reference and AscendC implementations.

    Returns a dict with mean/median/geomean/total for ref and asc, plus total_speedup.
    """
    if ref_times:
        ref_stats = {
            "mean_ref_us": statistics.mean(ref_times),
            "median_ref_us": statistics.median(ref_times),
            "geomean_ref_us": statistics.geometric_mean(ref_times),
            "total_ref_us": sum(ref_times),
        }
    else:
        ref_stats = {"mean_ref_us": None, "median_ref_us": None,
                     "geomean_ref_us": None, "total_ref_us": None}

    if asc_times:
        asc_stats = {
            "mean_asc_us": statistics.mean(asc_times),
            "median_asc_us": statistics.median(asc_times),
            "geomean_asc_us": statistics.geometric_mean(asc_times),
            "total_asc_us": sum(asc_times),
        }
    else:
        asc_stats = {"mean_asc_us": None, "median_asc_us": None,
                     "geomean_asc_us": None, "total_asc_us": None}

    total_speedup = None
    if ref_times and asc_times:
        asc_sum = sum(asc_times)
        if asc_sum > 0:
            total_speedup = sum(ref_times) / asc_sum

    return {**ref_stats, **asc_stats, "total_speedup": total_speedup}


def _log_and_save_compare_reports(summary, out_dir, speedups, n_cases):
    """Log summary results and save JSON/log/Markdown reports."""
    LOGGER.info("-" * 100)
    if speedups:
        LOGGER.info("--- Speedup ---")
        LOGGER.info(f"  Geomean : {summary['geomean_speedup']:.2f}x  ← 主指标")
        LOGGER.info(f"  Mean    : {summary['mean_speedup']:.2f}x")
        LOGGER.info(f"  Median  : {summary['median_speedup']:.2f}x")
        LOGGER.info(f"  Min/Max : {summary['min_speedup']:.2f}x / {summary['max_speedup']:.2f}x")
        LOGGER.info(f"  Valid   : {len(speedups)}/{n_cases}")
    else:
        LOGGER.info("--- Speedup ---")
        LOGGER.info("  (N/A: 无双侧均有效的 case —— --impl 单边采集或双侧测量失败)")
    # 单边采集（--impl ref/asc）时只渲染已测到的一侧，未测的一侧不出现
    if summary.get("mean_ref_us") is not None:
        LOGGER.info("--- Task Duration (us) ---")
        LOGGER.info(
            f"  Ref  mean/median/geomean/total : {summary['mean_ref_us']:.2f} / "
            f"{summary['median_ref_us']:.2f} / {summary['geomean_ref_us']:.2f} / "
            f"{summary['total_ref_us']:.2f}"
        )
    if summary.get("mean_asc_us") is not None:
        LOGGER.info(
            f"  Asc  mean/median/geomean/total : {summary['mean_asc_us']:.2f} / "
            f"{summary['median_asc_us']:.2f} / {summary['geomean_asc_us']:.2f} / "
            f"{summary['total_asc_us']:.2f}"
        )
    if summary.get("total_speedup") is not None:
        LOGGER.info(f"  Total speedup (Σref/Σasc) : {summary['total_speedup']:.2f}x")
    LOGGER.info("=" * 100)

    # 保存 JSON 报告
    json_path = out_dir / "performance.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"\n[INFO] JSON report saved to: {json_path}")

    # 保存打屏日志
    log_path = out_dir / "performance.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(_report_compare_to_text(summary))
    LOGGER.info(f"[INFO] Console report saved to: {log_path}")

    # 保存 Markdown 报告
    md_path = out_dir / "perf_report.md"
    md = _report_compare_to_markdown(summary)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    LOGGER.info(f"[INFO] Markdown report saved to: {md_path}")


def _load_compare_cases(out_dir):
    """Load test cases from jsonl or json files in the output directory."""
    cases, case_source = _load_cases_jsonl(out_dir)
    if not cases:
        cases, case_source = _load_cases_json(out_dir)
    return cases, case_source


def _log_compare_header(out_dir, args):
    """Log the compare mode header."""
    LOGGER.info("=" * 100)
    LOGGER.info(f"Kernel-level Performance (msprof): {out_dir.name}  "
          f"(warmup={args.warmup}, repeats={args.repeats}, seed={args.seed})")
    LOGGER.info("=" * 100)
    LOGGER.info(f"{'Case':<5} {'Shape':<35} {'dtype':<10} {'Ref(us)':>12} {'Asc(us)':>12} {'Speedup':>10}")
    LOGGER.info("-" * 100)


_PROF_TAG_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
# 与 _measure_one_impl* 中硬编码的 /tmp/msprof_* 前缀保持一致；
# 不用 tempfile.gettempdir()，避免 TMPDIR 环境变量改写导致误拒清理。
_TMP_ROOT = os.path.realpath("/tmp")


def _cleanup_prof_dirs(*dirs):
    """Remove profiling directories if they exist.

    防御：删除前先 realpath 解析，确认目标位于临时目录根下且不是根本身，
    才允许删除。prof_tag 等用户输入会拼进目录名，防止路径逃逸导致误删。
    """
    for d in dirs:
        if not d:
            continue
        real = os.path.realpath(d)
        if real == _TMP_ROOT or os.path.commonpath([real, _TMP_ROOT]) != _TMP_ROOT:
            LOGGER.warning("Refusing to remove %s: not under %s", real, _TMP_ROOT)
            continue
        if os.path.isdir(real):
            shutil.rmtree(real, ignore_errors=True)


@dataclass
class _LoopAccumulator:
    """对比/快速模式共用的逐 case 统计累积器。

    Args:
        args: argparse.Namespace（需含 keep_prof 属性）
        rows: 逐 case 的测量结果列表
        speedups: 有效 speedup 值列表
        ref_times: 参考实现耗时列表（us）
        asc_times: AscendC 实现耗时列表（us）

    record() 统一完成双侧耗时累计、speedup 统计、日志输出、rows 登记与
    PROF 目录清理，对比/快速两个循环共用同一份逻辑。
    """
    args: argparse.Namespace
    rows: List[Dict[str, Any]] = field(default_factory=list)
    speedups: List[float] = field(default_factory=list)
    ref_times: List[float] = field(default_factory=list)
    asc_times: List[float] = field(default_factory=list)

    def record(self, idx, shape, dtype, ref_result, asc_result):
        """累计单个 case 的测量结果。

        ref_result/asc_result 为 _measure_one_impl(_quick) 返回的
        (duration_us, error, prof_dir) 三元组。

        单侧失败或 --impl 单边采集时，仍累计已测到的一侧耗时，生成单边汇总；
        speedup 仅在双侧都有效时统计。
        """
        ref_us, ref_err, ref_prof_dir = ref_result
        asc_us, asc_err, asc_prof_dir = asc_result

        if ref_us is not None:
            self.ref_times.append(ref_us)
        if asc_us is not None and asc_us > 0:
            self.asc_times.append(asc_us)
        if ref_us is not None and asc_us is not None and asc_us > 0:
            sp = ref_us / asc_us
            self.speedups.append(sp)
            LOGGER.info(f"{idx:<5} {shape:<35} {dtype:<10} {ref_us:>12.2f} {asc_us:>12.2f} {sp:>9.3f}x")
        else:
            LOGGER.info(f"{idx:<5} {shape:<35} {dtype:<10} "
                  f"{'N/A' if ref_us is None else f'{ref_us:.2f}':>12} "
                  f"{'N/A' if asc_us is None else f'{asc_us:.2f}':>12} "
                  f"{'N/A':>10}  (ref_err={ref_err}, asc_err={asc_err})")

        self.rows.append({
            "case": idx, "shape": shape, "dtype": dtype,
            "ref_us": ref_us, "asc_us": asc_us,
            "speedup": (ref_us / asc_us) if (ref_us and asc_us and asc_us > 0) else None,
            "ref_error": ref_err,
            "asc_error": asc_err,
            "ref_prof_dir": ref_prof_dir,
            "asc_prof_dir": asc_prof_dir,
        })

        if not self.args.keep_prof:
            _cleanup_prof_dirs(ref_prof_dir, asc_prof_dir)


def _run_compare_loop(out_dir, cases, n_cases, args, device_id):
    """Run the measurement loop over all cases. Returns rows and stats."""
    acc = _LoopAccumulator(args)

    for idx in range(n_cases):
        shape, dtype = _extract_shape_dtype_from_jsonl(cases[idx]) if cases else ("?", "?")
        jsonl_case = cases[idx] if cases else None

        if _case_has_empty_tensor(jsonl_case):
            LOGGER.info(f"{idx:<5} {shape:<35} {dtype:<10} "
                  f"{'--':>12} {'--':>12} "
                  f"{'skip':>10}  (empty tensor)")
            acc.rows.append({
                "case": idx, "shape": shape, "dtype": dtype,
                "ref_us": None, "asc_us": None, "speedup": None,
                "ref_error": None, "asc_error": None,
                "skipped": "empty_tensor",
            })
            continue

        ref_mi = _MeasureInput(out_dir, idx, "reference", args, device_id, jsonl_case)
        asc_mi = _MeasureInput(out_dir, idx, "ascendc", args, device_id, jsonl_case)
        ref_result = _measure_one_impl(ref_mi)
        asc_result = _measure_one_impl(asc_mi)

        acc.record(idx, shape, dtype, ref_result, asc_result)

    return acc.rows, acc.speedups, acc.ref_times, acc.asc_times


def _run_quick_loop(out_dir, cases, n_cases, args, device_id):
    """快速模式：Run the measurement loop over all cases (只跑 1 轮 msprof).

    Returns rows and stats.
    """
    acc = _LoopAccumulator(args)

    sel = getattr(args, "cases", None)
    if sel:
        idx_list = [int(x) for x in str(sel).split(",") if x.strip() != ""]
    else:
        idx_list = list(range(n_cases))

    for idx in idx_list:
        shape, dtype = _extract_shape_dtype_from_jsonl(cases[idx]) if cases else ("?", "?")
        jsonl_case = cases[idx] if cases else None

        if _case_has_empty_tensor(jsonl_case):
            LOGGER.info(f"{idx:<5} {shape:<35} {dtype:<10} "
                  f"{'--':>12} {'--':>12} "
                  f"{'skip':>10}  (empty tensor)")
            acc.rows.append({
                "case": idx, "shape": shape, "dtype": dtype,
                "ref_us": None, "asc_us": None, "speedup": None,
                "ref_error": None, "asc_error": None,
                "skipped": "empty_tensor",
            })
            continue

        # 按 --impl 选择实现；单侧失败或单边采集时仍累计已测到的一侧耗时。
        which = getattr(args, "impl", "both")
        if which in ("both", "ref"):
            ref_mi = _MeasureInput(out_dir, idx, "reference", args, device_id, jsonl_case)
            ref_result = _measure_one_impl_quick(ref_mi)
        else:
            ref_result = (None, "skipped (--impl)", None)
        if which in ("both", "asc"):
            asc_mi = _MeasureInput(out_dir, idx, "ascendc", args, device_id, jsonl_case)
            asc_result = _measure_one_impl_quick(asc_mi)
        else:
            asc_result = (None, "skipped (--impl)", None)

        acc.record(idx, shape, dtype, ref_result, asc_result)

    return acc.rows, acc.speedups, acc.ref_times, acc.asc_times


def run_compare_mode(args):
    """执行对比模式：model.py vs model_new_ascendc.py"""
    out_dir = Path(args.output_dir).resolve()

    device_id, device_src = _select_device_id(args)
    LOGGER.info(f"[INFO] Using NPU device {device_id} (source={device_src})")

    cases, case_source = _load_compare_cases(out_dir)
    n_cases = len(cases)
    if case_source:
        LOGGER.info(f"[INFO] Loaded {n_cases} cases from {case_source}")

    _log_compare_header(out_dir, args)
    rows, speedups, ref_times, asc_times = _run_compare_loop(
        out_dir, cases, n_cases, args, device_id)

    csi = _CompareSummaryInput(
        out_dir, rows, speedups, ref_times, asc_times, n_cases, args, device_id, device_src)
    summary = _compute_compare_summary(csi)
    _log_and_save_compare_reports(summary, out_dir, speedups, n_cases)


def run_quick_mode(args):
    """执行快速模式：model.py vs model_new_ascendc.py（只跑 1 轮 msprof，不采集 7 个 metrics）"""
    out_dir = Path(args.output_dir).resolve()

    device_id, device_src = _select_device_id(args)
    LOGGER.info(f"[INFO] Using NPU device {device_id} (source={device_src})")
    LOGGER.info("[INFO] Quick mode: 1-round profiling (no aic-metrics)")

    cases, case_source = _load_compare_cases(out_dir)
    n_cases = len(cases)
    if case_source:
        LOGGER.info(f"[INFO] Loaded {n_cases} cases from {case_source}")

    max_cases = getattr(args, "max_cases", 0)
    if max_cases and max_cases < n_cases:
        n_cases = max_cases
        LOGGER.info(f"[INFO] --max-cases: limiting to first {n_cases} cases")

    LOGGER.info(f"[INFO] in-process warmup={args.warmup}, timed repeats={args.repeats}, "
                f"ext-warmup={getattr(args, 'ext_warmup', 0)}, impl={getattr(args, 'impl', 'both')}")
    _log_compare_header(out_dir, args)
    rows, speedups, ref_times, asc_times = _run_quick_loop(
        out_dir, cases, n_cases, args, device_id)

    csi = _CompareSummaryInput(
        out_dir, rows, speedups, ref_times, asc_times, n_cases, args, device_id, device_src)
    summary = _compute_compare_summary(csi)
    summary["timing_method"] = "msprof.quick.Task_Duration"
    summary["profiling_mode"] = "quick"
    _log_and_save_compare_reports(summary, out_dir, speedups, n_cases)


# ============================================================================
# 批量模式：扫描多个算子目录，汇总报告
# ============================================================================

def _is_valid_table_row(line: str) -> bool:
    return line.startswith('|') and 'Level' not in line and '---' not in line and len(line) > 5


def _extract_trace_table_rows(trace_file_path: str) -> List[str]:
    if not os.path.exists(trace_file_path):
        return []
    try:
        with open(trace_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return []
    start_idx = content.find('## 汇总表报告')
    if start_idx == -1:
        return []
    section_content = content[start_idx:]
    lines = section_content.split('\n')
    valid_rows = []
    for line in lines:
        line = line.strip()
        if _is_valid_table_row(line):
            valid_rows.append(line)
    return valid_rows


def _load_performance_json(op_dir: Path) -> Optional[Dict[str, Any]]:
    perf_json = op_dir / "performance.json"
    if perf_json.exists():
        try:
            with open(perf_json, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            LOGGER.warning("Failed to load performance.json from %s: %s", op_dir, e)
    return None


def _build_batch_md_summary_table(op_results):
    """Build the batch summary table markdown lines."""
    md_lines = []
    if op_results:
        md_lines.append("## 性能汇总")
        md_lines.append("")
        md_lines.append(
            "| 算子名称 | 用例数 | 有效用例 | 几何平均加速比 | 平均加速比 | 状态 |"
        )
        md_lines.append("| -------- | ------ | -------- | -------------- | ---------- | ---- |")
        for op in op_results:
            data = op["data"]
            name = op["name"]
            n_total = data.get("n_cases_total", 0)
            n_valid = data.get("n_cases_valid", 0)
            geo = data.get("geomean_speedup")
            mean = data.get("mean_speedup")
            geo_str = f"{geo:.3f}" if geo is not None else "N/A"
            mean_str = f"{mean:.3f}" if mean is not None else "N/A"
            status = "✅" if geo is not None and geo > 1 else "⚠️" if geo is not None else "❌"
            md_lines.append(f"| {name} | {n_total} | {n_valid} | {geo_str} | {mean_str} | {status} |")
        md_lines.append("")
    return md_lines


def _build_batch_md_per_op_details(op_results):
    """Build per-operator detail tables in markdown."""
    md_lines = []
    for op in op_results:
        data = op["data"]
        name = op["name"]
        md_lines.append(f"## {name}")
        md_lines.append("")
        if data.get("per_case"):
            md_lines.append("| Case | Shape | DType | 自定义算子(us) | 标杆(us) | 加速比 |")
            md_lines.append("| ---- | ----- | ----- | ------------- | -------- | -------------- |")
            for case in data["per_case"]:
                shape = case.get("shape", "?")
                dtype = case.get("dtype", "?")
                ref = case.get("ref_us")
                asc = case.get("asc_us")
                sp = case.get("speedup")
                ref_str = f"{ref:.2f}" if ref is not None else "N/A"
                asc_str = f"{asc:.2f}" if asc is not None else "N/A"
                sp_str = f"{sp:.3f}" if sp is not None else "N/A"
                md_lines.append(f"| {case['case']} | {shape} | {dtype} | {asc_str} | {ref_str} | {sp_str} |")
            md_lines.append("")
    return md_lines


def _build_batch_md_trace_section(trace_rows):
    """Build trace table section in markdown if rows exist."""
    if not trace_rows:
        return []
    md_lines = [
        "## Trace 汇总表",
        "",
        ("| Level | Problem ID | 算子名称 | 算子类型 | 编译通过 | 精度正确 | "
         "PyTorch 参考延迟 | 生成AscendC代码延迟 | 加速比 | 最终状态 | "
         "精度正确 | 性能0.6x pytorch | 性能0.8x pytorch |"),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    md_lines.extend(trace_rows)
    md_lines.append("")
    return md_lines


def _generate_batch_md_report(args, op_results, trace_rows, base_dir):
    """Generate and save the batch Markdown report."""
    md_lines = []
    md_lines.append("# 📊 算子批量性能汇总报告")
    md_lines.append("")
    md_lines.append(f"- **扫描目录**: {base_dir}")
    md_lines.append(f"- **算子总数**: {len(op_results)}")
    md_lines.append(f"- **生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append("")

    md_lines.extend(_build_batch_md_summary_table(op_results))
    md_lines.extend(_build_batch_md_per_op_details(op_results))
    md_lines.extend(_build_batch_md_trace_section(trace_rows))

    md_path = Path(args.output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    LOGGER.info("Batch markdown report saved to: %s", md_path)


def _generate_batch_json_report(args, op_results, base_dir):
    """Generate and save the batch JSON summary."""
    batch_summary = {
        "base_dir": str(base_dir),
        "n_operators": len(op_results),
        "operators": [
            {
                "name": op["name"],
                "n_cases_total": op["data"].get("n_cases_total", 0),
                "n_cases_valid": op["data"].get("n_cases_valid", 0),
                "geomean_speedup": op["data"].get("geomean_speedup"),
                "mean_speedup": op["data"].get("mean_speedup"),
                "mean_ref_us": op["data"].get("mean_ref_us"),
                "mean_asc_us": op["data"].get("mean_asc_us"),
            }
            for op in op_results
        ],
        "generated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(batch_summary, f, indent=2, ensure_ascii=False)
    LOGGER.info("Batch JSON summary saved to: %s", json_path)


def run_batch_mode(args):
    """执行批量模式：扫描 base_dir 下所有子目录，汇总性能报告。"""
    base_dir = Path(args.batch).resolve()

    if not base_dir.is_dir():
        raise ValueError("'%s' is not a directory." % base_dir)

    # 收集所有子目录的 performance.json
    op_results = []
    for subdir in sorted(base_dir.iterdir()):
        if not subdir.is_dir():
            continue
        perf_data = _load_performance_json(subdir)
        if perf_data:
            op_results.append({
                "name": subdir.name,
                "data": perf_data,
                "dir": subdir,
            })

    # 同时收集 trace.md 中的表格行（兼容旧 batch_report.py 功能）
    trace_rows = []
    for subdir in sorted(base_dir.iterdir()):
        if not subdir.is_dir():
            continue
        trace_file = subdir / "trace.md"
        if trace_file.exists():
            rows = _extract_trace_table_rows(str(trace_file))
            trace_rows.extend(rows)

    LOGGER.info("Found %d operators with performance.json in %s", len(op_results), base_dir)

    if args.output_md:
        _generate_batch_md_report(args, op_results, trace_rows, base_dir)

    if args.output_json:
        _generate_batch_json_report(args, op_results, base_dir)


# ============================================================================
# 主入口
# ============================================================================

def _run_standard_mode(args):
    """Execute standard mode: parse PROF_GROUP and generate summary."""
    group_dir = os.path.abspath(args.prof_group_dir)
    ops_dir = os.path.abspath(args.ops_dir)

    if not os.path.isdir(group_dir):
        raise ValueError("'%s' is not a directory." % group_dir)
    if not os.path.isdir(ops_dir):
        raise ValueError("'%s' is not a directory." % ops_dir)

    merged = merge_metric_rows(group_dir, args.op_name)
    if not merged.get("Op Name"):
        raise ValueError("no op_summary_*.csv rows discovered under %s" % group_dir)

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


def _build_arg_parser():
    """构建命令行参数解析器（标准/对比/批量三种模式）。"""
    parser = argparse.ArgumentParser(description="msprof 解析 & 归档 & 对比测试脚本（统一入口）")

    # 标准模式参数
    parser.add_argument("prof_group_dir", nargs="?", help="PROF_GROUP_<timestamp> directory")
    parser.add_argument("ops_dir", nargs="?", help="Operator directory")
    parser.add_argument("--op-name", default=None, help="Exact Op Name to pick in op_summary.csv")
    parser.add_argument("--round-name", default=None, help="Override round directory name")

    # 对比模式参数
    parser.add_argument("--compare", action="store_true", help="启用对比模式（8 轮采集：7 metrics + sample）")
    parser.add_argument("--quick", action="store_true", help="启用快速模式（1 轮采集：只获取 kernel 时间，不采集 7 个 aic-metrics）")
    parser.add_argument("--output-dir", dest="output_dir", help="算子输出目录（对比模式/快速模式）")
    parser.add_argument("--warmup", type=int, default=3,
                        help="quick 模式：被采集进程内的预热迭代次数（解析时丢弃）")
    parser.add_argument("--repeats", type=int, default=1,
                        help="quick 模式：被采集进程内的计时迭代次数（解析时取均值）")
    parser.add_argument("--ext-warmup", dest="ext_warmup", type=int, default=0,
                        help="旧行为：在 msprof 外部另起进程预热的次数（对被采集进程无预热效果，默认 0）")
    parser.add_argument("--impl", choices=["both", "ref", "asc"], default="both",
                        help="quick 模式只采集指定实现（ref=参考实现，asc=AscendC 实现）")
    parser.add_argument("--max-cases", dest="max_cases", type=int, default=0,
                        help="quick 模式只跑前 N 个 case（0=全部，用于冒烟测试）")
    parser.add_argument("--cases", default=None,
                        help="quick 模式只跑指定 case 下标，逗号分隔，如 0,4,9,18（优先于 --max-cases）")
    parser.add_argument("--prof-tag", dest="prof_tag", default=None,
                        help="PROF 临时目录后缀，用于同一 case 多次独立采集互不覆盖")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--retry", type=int, default=2, help="单 case 解析失败重试次数")
    parser.add_argument("--device", type=int, default=None, help="NPU 设备 id")
    parser.add_argument("--keep-prof", action="store_true", help="保留 msprof 原始 PROF 目录")

    # 批量模式参数
    parser.add_argument("--batch", metavar="BASE_DIR", help="启用批量模式，指定根目录")
    parser.add_argument("--output-md", help="批量模式 Markdown 输出路径")
    parser.add_argument("--output-json", help="批量模式 JSON 输出路径")
    return parser


def _run_or_exit(run_fn, args) -> int:
    """执行模式函数，ValueError（参数/输入不合法）时打日志并返回退出码 1。"""
    try:
        run_fn(args)
    except ValueError as e:
        LOGGER.error("%s", e)
        return 1
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args()

    # 模式路由
    if args.compare:
        if not args.output_dir:
            parser.error("--compare 模式必须指定 --output-dir")
        run_compare_mode(args)
        return 0
    if args.quick:
        if not args.output_dir:
            parser.error("--quick 模式必须指定 --output-dir")
        run_quick_mode(args)
        return 0
    if args.batch:
        return _run_or_exit(run_batch_mode, args)
    if not args.prof_group_dir or not args.ops_dir:
        parser.error("标准模式需要 prof_group_dir 和 ops_dir 参数")
    return _run_or_exit(_run_standard_mode, args)


if __name__ == "__main__":
    sys.exit(main())
