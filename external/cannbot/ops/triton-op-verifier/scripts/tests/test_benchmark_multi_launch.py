# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""benchmark.parse_operator_latency 的单元测试：验证一个 forward 内多次启动同一 kernel 的时延计算。

修复目标：旧实现假定每个 Name 每次 forward 只发射 1 次（每组取末尾 active_count 行、sum/active_count），
当单个 forward 多次启动同一 kernel（发射次数 L>1）时会把单次调用耗时低估为 1/L。
新实现按 L = len(group)/active_count 反推发射次数，再以中位数为基准剔除异常采样、
按迭代段内求和、跨迭代取中位数，避免低估与坏行污染聚合值。

运行环境仅需 CPU torch + pandas（无需 NPU）：用合成的 kernel_details.csv 直接调用解析函数。
"""
import os

import pandas as pd
import pytest

import benchmark


def _write_profile(tmp_path, rows, subdir="prof"):
    """把 rows 写入 tmp_path/<subdir>/kernel_details.csv，返回 profile 目录路径。"""
    prof = tmp_path / subdir
    prof.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(prof / "kernel_details.csv", index=False)
    return str(prof)


def _row(name, dur_us):
    return {"Name": name, "Duration(us)": dur_us}


def test_l1_baseline(tmp_path):
    """L=1：单次调用耗时 = active 阶段各行之和 / active_count，launch_count=1。"""
    path = _write_profile(tmp_path, [_row("K", 10), _row("K", 20), _row("K", 30)])

    operators, total_ms = benchmark.parse_operator_latency(path, active_count=3)

    assert operators["K"]["avg_us"] == 20.0
    assert operators["K"]["launch_count"] == 1
    assert total_ms == 0.02


def test_l2_undercount_fixed(tmp_path):
    """回归：L=2 时旧实现只取末尾 active 行（低估一半），新实现取 L×active 行。"""
    path = _write_profile(tmp_path, [_row("K", d) for d in (10, 20, 30, 40, 50, 60)])

    operators, total_ms = benchmark.parse_operator_latency(path, active_count=3)

    # 3 次调用 × 每次 2 次发射：单次调用 = (10+20+30+40+50+60)/3 = 70
    assert operators["K"]["avg_us"] == 70.0
    assert operators["K"]["launch_count"] == 2
    # 旧实现只取末尾 3 行会得到 50，此处断言已修复
    assert operators["K"]["avg_us"] != 50.0
    assert total_ms == 0.07


def test_mixed_launch_counts(tmp_path):
    """多 Name 混合发射次数：各自 avg_us/launch_count 正确，total_ms = Σ(avg_us)/1000。"""
    rows = (
        [_row("A", 5), _row("A", 6)]  # L=1
        + [_row("B", 1), _row("B", 2), _row("B", 3), _row("B", 4)]  # L=2
        + [_row("C", 1), _row("C", 1), _row("C", 1), _row("C", 2), _row("C", 2), _row("C", 2)]  # L=3
    )
    path = _write_profile(tmp_path, rows)

    operators, total_ms = benchmark.parse_operator_latency(path, active_count=2)

    assert operators["A"]["avg_us"] == 5.5 and operators["A"]["launch_count"] == 1
    assert operators["B"]["avg_us"] == 5.0 and operators["B"]["launch_count"] == 2
    assert operators["C"]["avg_us"] == 4.5 and operators["C"]["launch_count"] == 3
    assert total_ms == (5.5 + 5.0 + 4.5) / 1000


def test_missing_csv_returns_none(tmp_path):
    """无 kernel_details.csv 时返回 (None, None)，并清理 profile 目录。"""
    empty = tmp_path / "nofile"
    empty.mkdir()

    operators, total_ms = benchmark.parse_operator_latency(str(empty), active_count=3)

    assert operators is None
    assert total_ms is None
    assert not empty.exists()


def test_keep_res_controls_cleanup(tmp_path):
    """keep_res=True 保留 profiling 目录；默认 keep_res=False 清理。"""
    path = _write_profile(tmp_path, [_row("K", 10)] * 3, subdir="keep")
    operators, _ = benchmark.parse_operator_latency(path, active_count=3, keep_res=True)
    assert operators is not None
    assert os.path.isdir(path), "keep_res=True 应保留 profiling 目录"

    path2 = _write_profile(tmp_path, [_row("K", 10)] * 3, subdir="drop")
    benchmark.parse_operator_latency(path2, active_count=3)
    assert not os.path.isdir(path2), "keep_res=False 应清理 profiling 目录"


def test_non_integer_launch_warns(caplog, tmp_path):
    """CSV 混入非 active 行（如残留 warmup 行）时 L 非整数：取整、告警、不崩溃。"""
    path = _write_profile(tmp_path, [_row("K", 10)] * 5)  # active=3 下 len=5 → L=1.667

    with caplog.at_level("WARNING", logger="triton_op_verifier.benchmark"):
        operators, _ = benchmark.parse_operator_latency(path, active_count=3)

    assert operators["K"]["launch_count"] == 2  # round(1.667)
    # 新实现只保留末尾可整除段（5%3=2 行边界污染被丢弃）→ 3 行 × 10us / 3 次迭代 = 10
    assert operators["K"]["avg_us"] == 10.0
    assert any("非整数" in r.message for r in caplog.records)
