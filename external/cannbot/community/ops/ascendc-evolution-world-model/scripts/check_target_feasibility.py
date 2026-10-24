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
check_target_feasibility.py — Init 阶段目标可达性预检（P6）

检查 `target_time_us = baseline_time_us / target_speedup` 是否低于设备 launch 地板。
设备地板来自 OKF proven_solutions 卡的 `device_floor_us` frontmatter（如 x24 卡
"910B 每 kernel launch 设备地板 ~4.3-5.6μs"）。目标时间低于地板 → 物理不可达，
输出 alarm 供 lingxi-evo 写入 open_questions（不硬阻断进化）。

用法（Init 阶段执行，skill 根目录下）:
    python3 scripts/check_target_feasibility.py \
        --baseline-eval output/{op}_evo_{ts}/baseline_evaluation.json \
        --target-speedup 2.0 \
        [--op-family attention] [--op-name NAME]

输出（stdout JSON）:
    {feasible, target_time_us, device_floor_us, alarm, notes}
"""

from __future__ import annotations
import argparse
import json
import logging
import re
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
# scripts → skill 根 → skills 根（仓内 community/ops 或安装后 .claude/skills，
# 两种布局下兄弟 skill 均并列于同一父目录）
SKILLS_ROOT = SCRIPT_DIR.parents[1]
OKF_PROVEN = (
    SKILLS_ROOT / "ascendc-evolution-knowledge/references/okf/runbooks/proven-solutions"
)


def _parse_floor(value: str) -> float:
    """解析 device_floor_us frontmatter："4.3-5.6" 取上限 5.6；数值直接取。"""
    m = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", value)
    if m:
        return max(float(m.group(1)), float(m.group(2)))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def load_device_floor(op_family: str = "") -> float:
    """扫描 OKF proven-solutions 卡 frontmatter，收集 device_floor_us，取最保守（最大）值。

    设备 launch 地板是跨算子通用的（所有 kernel 都有该固定开销）。优先取覆盖目标
    op_family 的卡；族匹配不到时回退到全局最大地板，避免 attention 等非 matmul 族
    因知识卡缺失而跳过可行性判定。
    """
    floors_all = []
    floors_family = []
    if OKF_PROVEN.exists():
        for md in sorted(OKF_PROVEN.glob("*.md")):
            text = md.read_text(encoding="utf-8", errors="ignore")
            fm = re.search(r"^---\n(.*?)\n---", text, re.S | re.M)
            if not fm:
                continue
            body = fm.group(1)
            m = re.search(r"device_floor_us:\s*[\"']?([^\"'\n]+)[\"']?", body)
            if not m:
                continue
            val = _parse_floor(m.group(1).strip())
            if val <= 0:
                continue
            floors_all.append(val)
            fam_m = re.search(r"op_families:\s*\[([^\]]*)\]", body)
            if not op_family or not fam_m or op_family in fam_m.group(1):
                floors_family.append(val)
    if floors_family:
        return max(floors_family)
    return max(floors_all) if floors_all else 0.0


def check_feasibility(
    baseline_time_us: float,
    target_speedup: float,
    op_family: str = "",
    device_floor_us: float = 0.0,
) -> dict:
    """计算目标可达性。"""
    if baseline_time_us <= 0:
        return {
            "feasible": None,
            "target_time_us": None,
            "device_floor_us": device_floor_us,
            "alarm": "baseline_time_us 缺失或非正，无法预检",
            "notes": "",
        }
    if target_speedup <= 0:
        return {
            "feasible": None,
            "target_time_us": None,
            "device_floor_us": device_floor_us,
            "alarm": "target_speedup 非正，无法预检",
            "notes": "",
        }

    target_time_us = baseline_time_us / target_speedup
    if device_floor_us <= 0:
        return {
            "feasible": None,
            "target_time_us": round(target_time_us, 2),
            "device_floor_us": None,
            "alarm": "未找到设备地板知识（OKF proven_solutions 无 device_floor_us），跳过可行性判定",
            "notes": "target_time_us = baseline_time_us/target_speedup",
        }

    feasible = target_time_us >= device_floor_us
    alarm = ""
    if not feasible:
        alarm = (
            f"目标 {target_speedup}x 需 {target_time_us:.2f}μs，低于设备 launch 地板 "
            f"{device_floor_us:.2f}μs，物理不可达（见 OKF x24 proven_solution 上限认知）"
        )
    return {
        "feasible": feasible,
        "target_time_us": round(target_time_us, 2),
        "device_floor_us": round(device_floor_us, 2),
        "alarm": alarm,
        "notes": "baseline_time_us/target_speedup vs 设备 launch 地板",
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Init 阶段目标可达性预检（P6）")
    parser.add_argument(
        "--baseline-eval", required=True, help="baseline_evaluation.json 路径"
    )
    parser.add_argument(
        "--target-speedup", type=float, required=True, help="目标加速比（vs PyTorch）"
    )
    parser.add_argument(
        "--op-family", default="", help="算子族（用于匹配地板知识卡，可选）"
    )
    parser.add_argument(
        "--device-floor-us",
        type=float,
        default=0.0,
        help="显式指定设备地板 μs（可选，覆盖知识库检索）",
    )
    args = parser.parse_args()

    baseline_time_us = 0.0
    try:
        with open(args.baseline_eval, "r", encoding="utf-8") as f:
            be = json.load(f)
        baseline_time_us = float(be.get("baseline_time_us", 0) or 0)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        LOGGER.error("[ERROR] 无法读取 baseline_eval: %s", e)
        return 2

    floor = args.device_floor_us or load_device_floor(args.op_family)
    result = check_feasibility(
        baseline_time_us, args.target_speedup, args.op_family, floor
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
