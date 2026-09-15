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
"""
从 ops-profiling 的 performance.json 生成 baseline_evaluation.json
供 lingxi-evo 进化流程使用。

用法:
    python3 generate_baseline_eval.py \
        --perf-json /path/to/performance.json \
        --baseline-path /path/to/baseline_kernel \
        --output /path/to/output/baseline_evaluation.json \
        [--op-name NAME] [--timestamp TIMESTAMP]

输出字段与 evolution-report 兼容，包含:
    - baseline_time_us: 基准耗时（evolution-report 必需）
    - avg_speedup / best_speedup / worst_speedup: 加速比统计
    - ops_profiling_result: 完整的 performance.json 数据
"""

import argparse
import json
import logging
import os
import sys

LOGGER = logging.getLogger(__name__)


def generate_baseline_eval(perf_path: str, baseline_path: str, op_name: str = "", timestamp: str = "") -> dict:
    """从 performance.json 生成 baseline_evaluation.json 内容。"""

    if not os.path.exists(perf_path):
        raise FileNotFoundError(f"performance.json not found: {perf_path}")

    with open(perf_path, "r", encoding="utf-8") as f:
        perf = json.load(f)

    per_case = perf.get("per_case", [])
    valid_cases = [c for c in per_case if c.get("speedup") is not None]

    if valid_cases:
        avg_speedup = sum(c["speedup"] for c in valid_cases) / len(valid_cases)
        best_speedup = max(c["speedup"] for c in valid_cases)
        worst_speedup = min(c["speedup"] for c in valid_cases)
        baseline_time_us = sum(c.get("asc_us", 0) for c in valid_cases) / len(valid_cases)
        pytorch_time_us = sum(c.get("ref_us", 0) for c in valid_cases) / len(valid_cases)
    else:
        avg_speedup = best_speedup = worst_speedup = 0.0
        baseline_time_us = pytorch_time_us = 0.0

    first_case = per_case[0] if per_case else {}
    shape = first_case.get("shape", [128])
    dtype = first_case.get("dtype", "float32")

    n_valid = perf.get("n_cases_valid", 0)
    n_total = perf.get("n_cases_total", 0)
    geomean = perf.get("geomean_speedup", 0)

    baseline_eval = {
        "baseline_kernel_path": baseline_path,
        "compilation_success": n_valid > 0,
        "precision_passed": n_valid > 0,
        "match_rate": 100.0 if n_valid > 0 else 0.0,
        "avg_speedup": avg_speedup,
        "best_speedup": best_speedup,
        "worst_speedup": worst_speedup,
        "base_time_ms_avg": pytorch_time_us / 1000.0,
        "gen_time_ms_avg": baseline_time_us / 1000.0,
        "baseline_time_us": baseline_time_us,
        "notes": f"Baseline from ops-profiling: {n_valid}/{n_total} cases valid, geomean={geomean:.3f}x",
        "test_case": {
            "inputs": [{"name": "x", "shape": shape, "dtype": dtype}],
            "scalar_args": {}
        },
        "ops_profiling_result": perf
    }

    return baseline_eval


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Generate baseline_evaluation.json from ops-profiling performance.json")
    parser.add_argument("--perf-json", required=True, help="Path to performance.json from ops-profiling")
    parser.add_argument("--baseline-path", required=True, help="Path to baseline kernel directory")
    parser.add_argument("--output", required=True, help="Output path for baseline_evaluation.json")
    parser.add_argument("--op-name", default="", help="Operator name (optional, for notes)")
    parser.add_argument("--timestamp", default="", help="Timestamp (optional, for notes)")

    args = parser.parse_args()

    try:
        result = generate_baseline_eval(args.perf_json, args.baseline_path, args.op_name, args.timestamp)

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        LOGGER.info("[OK] baseline_evaluation.json saved to %s", args.output)
        LOGGER.info("     avg_speedup=%.3fx, baseline_time_us=%.2f",
                    result['avg_speedup'], result['baseline_time_us'])
        sys.exit(0)

    except FileNotFoundError as e:
        LOGGER.error("[ERROR] %s", e)
        sys.exit(1)
    except Exception as e:
        LOGGER.error("[ERROR] Failed to generate baseline_evaluation.json: %s", e)
        sys.exit(2)


if __name__ == "__main__":
    main()
