# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

#!/usr/bin/env python3
"""JSONL 合并工具 —— 将 baseline 和 fused 的逐 step JSONL 合并为 metrics JSON 数组。

只做数据准备，不生成 HTML。Agent 用此脚本拿到 metrics 数组后，
复制 assets/validation_report_template.html 并替换 REPORT_DATA。

用法:
  python3 scripts/merge_jsonl.py baseline.jsonl rmsnorm.jsonl
  python3 scripts/merge_jsonl.py baseline.jsonl rmsnorm.jsonl --threshold 0.01

输出格式（匹配模板 REPORT_DATA.metrics）:
  [{"step": 0, "phase": "wait", "base_loss": 12.59, "fused_loss": 12.59, "base_time": 2.92, "fused_time": 2.74}, ...]

同时输出 thresholds 信息（stderr），方便 agent 填入 REPORT_DATA.thresholds。
"""

from __future__ import annotations

import argparse
import json
import sys


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_num}: JSON 解析失败: {exc}") from exc
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="合并 baseline 和 fused JSONL 为 metrics 数组")
    parser.add_argument("baseline", help="baseline JSONL 路径")
    parser.add_argument("fused", help="fused JSONL 路径")
    parser.add_argument("--threshold", type=float, default=0.01, help="精度阈值（rel diff 小数，默认 0.01=1%%）")
    args = parser.parse_args()

    bl = load_jsonl(args.baseline)
    fu = load_jsonl(args.fused)

    bl_map = {r["step"]: r for r in bl}
    fu_map = {r["step"]: r for r in fu}
    all_steps = sorted(set(bl_map) | set(fu_map))

    metrics = []
    max_rel = 0.0
    first_fail = None
    for step in all_steps:
        b = bl_map.get(step, {})
        f = fu_map.get(step, {})
        bl_loss = b.get("loss")
        fu_loss = f.get("loss")
        bl_time = b.get("step_time")
        fu_time = f.get("step_time")

        if bl_loss is not None and fu_loss is not None and bl_loss != 0:
            rel = abs(fu_loss - bl_loss) / abs(bl_loss)
            if rel > max_rel:
                max_rel = rel
            if rel > args.threshold and first_fail is None:
                first_fail = step

        metrics.append({
            "step": step,
            "phase": f.get("phase") or b.get("phase", ""),
            "base_loss": bl_loss,
            "fused_loss": fu_loss,
            "base_time": bl_time,
            "fused_time": fu_time,
        })

    json.dump(metrics, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    sys.stderr.write(f"thresholds:\n")
    sys.stderr.write(f"  max_rel_diff: {max_rel:.6f}\n")
    sys.stderr.write(f"  first_fail_step: {first_fail if first_fail is not None else 'null'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
