# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""benchmark 采集解析修复的单元测试：坏行剔除（做法 B）、msprof 旧格式兜底、失败分类与证据保留。

背景：level4 批量复测中 profiler 采集/导出偶发失败（多进程并发压 profiler 时
EventQueue 为空 / 导出中断），旧实现把这些 case 与真实性能失败混为一谈，且
缺 kernel_details.csv 时无条件删目录丢失证据。本文件覆盖以下修复点：
1. _aggregate_kernel_rows：以中位数为基准剔除异常采样（阈值=max(median×K, FLOOR)），
   段内求和、跨迭代取中位数兜底，防止 1.34s 级假采样污染聚合值；
2. _parse_msprof_fallback：kernel_details.csv 缺失或解析失败时回退 op_summary/task_time，
   模式 B 型采集失败（数据在、导出缺）可恢复；
3. ProfilerCollectError：采集/导出失败标记 PROFILER_COLLECT_FAIL，与真实性能失败区分；
4. 失败路径的目录清理受 keep_res 控制，保留失败证据。

运行环境仅需 CPU + numpy + pandas（无需 NPU、无需 torch）。
"""
import os
import shutil

import numpy as np
import pandas as pd
import pytest

import benchmark
# 被测的是模块内部聚合函数：以 import 别名引入，避免在类外访问受保护成员
from benchmark import _aggregate_kernel_rows as aggregate_kernel_rows


def _write_profile(tmp_path, rows, subdir="prof", filename="kernel_details.csv"):
    prof = tmp_path / subdir
    prof.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(prof / filename, index=False)
    return str(prof)


def _row(name, dur_us):
    return {"Name": name, "Duration(us)": dur_us}


def _write_op_summary(tmp_path, rows, subdir="prof"):
    prof = tmp_path / subdir
    prof.mkdir(exist_ok=True)
    pd.DataFrame(rows, columns=["Op Name", "Task Duration(us)"]).to_csv(
        prof / "op_summary_20260815.csv", index=False)
    return str(prof)


def _write_task_time(tmp_path, rows, subdir="prof"):
    prof = tmp_path / subdir
    prof.mkdir(exist_ok=True)
    pd.DataFrame(rows, columns=["kernel_name", "task_time(us)"]).to_csv(
        prof / "task_time_20260815.csv", index=False)
    return str(prof)


# ---------------- 1. 坏行剔除（做法 B） ----------------

def test_outlier_sampling_removed():
    """1.34s 级假采样被 median×K 阈值剔除，段内求和不被污染。"""
    durations = np.array([2.0] * 80 + [1_340_000.0], dtype=float)  # 5 迭代 × 16 发射 + 1 个异常行

    avg_us, launch_count = aggregate_kernel_rows("K", durations, active_count=5)

    assert launch_count == 16
    assert avg_us == pytest.approx(32.0)  # 16 发射 × 2us，而非 268ms 级


def test_floor_threshold_when_median_tiny():
    """中位数≈0 时阈值取绝对下限 FLOOR_US，仍能剔除跨数量级的假采样。"""
    durations = np.array([0.01] * 20 + [15.0], dtype=float)  # active=5，15us 为假采样

    avg_us, launch_count = aggregate_kernel_rows("K", durations, active_count=5)

    assert launch_count == 4  # round(21/5)
    assert avg_us == pytest.approx(0.04)  # 4 发射 × 0.01us，15us 假采样被剔除


def test_no_outlier_unchanged():
    """无坏行时结果等于段内求和，不误删正常采样。"""
    durations = np.array([2.0] * 80, dtype=float)

    avg_us, launch_count = aggregate_kernel_rows("K", durations, active_count=5)

    assert avg_us == pytest.approx(32.0)
    assert launch_count == 16


# ---------------- 2. msprof 旧格式兜底 ----------------

def test_op_summary_fallback(tmp_path):
    """缺 kernel_details.csv 时回退 op_summary_*.csv（Op Name / Task Duration(us)）。"""
    path = _write_op_summary(tmp_path, [("K", 1.0)] * 10, subdir="m")  # 5 迭代 × 2 发射

    operators, total_ms = benchmark.parse_operator_latency(path, active_count=5)

    assert operators is not None
    assert operators["K"]["avg_us"] == pytest.approx(2.0)  # 2 发射 × 1us
    assert operators["K"]["launch_count"] == 2
    assert total_ms == pytest.approx(0.002)


def test_task_time_fallback(tmp_path):
    """无 op_summary 时回退 task_time_*.csv（kernel_name / task_time(us)）。"""
    path = _write_task_time(tmp_path, [("K", 1.0)] * 10, subdir="m")

    operators, total_ms = benchmark.parse_operator_latency(path, active_count=5)

    assert operators is not None
    assert operators["K"]["avg_us"] == pytest.approx(2.0)
    assert operators["K"]["launch_count"] == 2


def test_fallback_no_data_returns_none(tmp_path):
    """旧格式存在但无有效 kernel 数据时返回 (None, None)，不抛异常。"""
    prof = tmp_path / "m"
    prof.mkdir()
    pd.DataFrame(columns=["Op Name", "Task Duration(us)"]).to_csv(
        prof / "op_summary_20260815.csv", index=False)

    operators, total_ms = benchmark.parse_operator_latency(str(prof), active_count=5, keep_res=True)

    assert operators is None
    assert total_ms is None


def test_malformed_kernel_details_falls_back(tmp_path):
    """kernel_details.csv 能读但缺 Name 列（导出截断）时回退 op_summary，不崩溃。"""
    prof = tmp_path / "m"
    prof.mkdir()
    pd.DataFrame({"Foo": [1, 2, 3]}).to_csv(prof / "kernel_details.csv", index=False)
    pd.DataFrame([("K", 1.0)] * 10, columns=["Op Name", "Task Duration(us)"]).to_csv(
        prof / "op_summary_20260815.csv", index=False)

    operators, total_ms = benchmark.parse_operator_latency(str(prof), active_count=5, keep_res=True)

    assert operators is not None
    assert operators["K"]["avg_us"] == pytest.approx(2.0)
    assert total_ms == pytest.approx(0.002)


def test_main_vs_fallback_equivalent(tmp_path):
    """同一份数据：主路径（kernel_details）与兜底（op_summary）结果一致。"""
    main = _write_profile(tmp_path, [_row("K", d) for d in (10, 20, 30, 40, 50, 60)], subdir="eq")
    pd.DataFrame([("K", float(d)) for d in (10, 20, 30, 40, 50, 60)],
                 columns=["Op Name", "Task Duration(us)"]).to_csv(
        os.path.join(main, "op_summary_20260815.csv"), index=False)

    ops_main, ms_main = benchmark.parse_operator_latency(main, active_count=3, keep_res=True)

    fb = tmp_path / "fb"
    shutil.copytree(main, fb)
    os.remove(os.path.join(str(fb), "kernel_details.csv"))
    ops_fb, ms_fb = benchmark.parse_operator_latency(str(fb), active_count=3, keep_res=True)

    assert set(ops_main) == set(ops_fb)
    assert ops_fb["K"]["avg_us"] == pytest.approx(ops_main["K"]["avg_us"])  # 均为 70.0
    assert ms_fb == pytest.approx(ms_main)


# ---------------- 3. 采集失败分类 ----------------

def test_profiler_collect_error_classification():
    """采集/导出失败抛 ProfilerCollectError（RuntimeError 子类），消息含 PROFILER_COLLECT_FAIL。"""
    assert issubclass(benchmark.ProfilerCollectError, RuntimeError)
    err = benchmark.ProfilerCollectError(
        "[用例 1/40] PROFILER_COLLECT_FAIL: 无法从 profiler 提取有效时延数据")
    assert "PROFILER_COLLECT_FAIL" in str(err)


# ---------------- 4. 失败路径证据保留 ----------------

def test_failure_path_cleanup_gated_by_keep_res(tmp_path):
    """彻底失败（无可解析数据）：keep_res=True 保留目录，False 清理。"""
    keep = tmp_path / "keep"
    keep.mkdir()
    operators, _ = benchmark.parse_operator_latency(str(keep), active_count=3, keep_res=True)
    assert operators is None
    assert keep.exists(), "keep_res=True 失败路径也应保留证据"

    drop = tmp_path / "drop"
    drop.mkdir()
    benchmark.parse_operator_latency(str(drop), active_count=3)
    assert not drop.exists(), "keep_res=False 失败路径应清理"
