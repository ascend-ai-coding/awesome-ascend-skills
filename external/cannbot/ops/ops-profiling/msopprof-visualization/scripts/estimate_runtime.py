#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

cli_logger = logging.getLogger(__name__ + ".cli")
cli_logger.propagate = False
if not cli_logger.handlers:
    _cli_handler = logging.StreamHandler(sys.stdout)
    _cli_handler.setFormatter(logging.Formatter("%(message)s"))
    cli_logger.addHandler(_cli_handler)

PRESET_PHASES = {
    "fast": ["preflight", "details", "memory_detail", "raw_data"],
    "core": ["preflight", "details", "roofline", "timeline", "memory_detail", "raw_data"],
    "complete": [
        "preflight", "details", "roofline", "timeline", "source",
        "warp_stall", "instruction_timeline", "memory_detail", "raw_data",
        "onchip_memory",
    ],
    "deep": [
        "preflight", "details", "roofline", "timeline", "source",
        "warp_stall", "instruction_timeline", "memory_detail", "raw_data",
        "onchip_memory", "timeline_detail", "kernel_scale",
    ],
}


def _load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("cannot read timing JSON: %s: %s", path, exc)
        return None
    if not isinstance(data, dict) or not isinstance(data.get("phases"), list):
        logger.error("timing JSON must contain a phases list.")
        return None
    return data


def _phase_map(data: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    out: Dict[str, Mapping[str, Any]] = {}
    for item in data.get("phases", []):
        if isinstance(item, Mapping) and item.get("phase"):
            out[str(item["phase"])] = item
    return out


def _sum_seconds(phases: Mapping[str, Mapping[str, Any]], names: Iterable[str]) -> float:
    total = 0.0
    for name in names:
        item = phases.get(name, {})
        status = str(item.get("status", ""))
        if status in {"aliased", "reused", "unavailable", "skipped"}:
            continue
        try:
            total += float(item.get("elapsed_seconds", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
    return total


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Estimate preset runtime from an observed timing_summary.json.")
    parser.add_argument("--timing", required=True, type=Path)
    parser.add_argument("--visualization-seconds", type=float, default=0.0,
                        help="Optional renderer cost to add to every preset.")
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown")
    args = parser.parse_args()

    data = _load(args.timing)
    if data is None:
        return 1
    phases = _phase_map(data)
    estimates = {}
    for preset, names in PRESET_PHASES.items():
        seconds = _sum_seconds(phases, names) + max(0.0, args.visualization_seconds)
        estimates[preset] = round(seconds, 3)

    baseline = estimates.get("complete", 0.0)
    rows = []
    for preset in ["fast", "core", "complete", "deep"]:
        seconds = estimates[preset]
        saved = max(0.0, baseline - seconds)
        reduction = (saved / baseline * 100.0) if baseline else 0.0
        rows.append({
            "preset": preset,
            "estimated_seconds": seconds,
            "saved_vs_complete_seconds": round(saved, 3),
            "reduction_vs_complete_percent": round(reduction, 1),
        })

    result = {
        "schema": "msopprof-runtime-estimate/v1",
        "source": str(args.timing),
        "basis": "observed phase durations; aliased/reused/unavailable phases count as zero replay time",
        "rows": rows,
    }
    if args.format == "json":
        cli_logger.info(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        cli_logger.info("| Preset | Estimated | Saved vs complete | Reduction |")
        cli_logger.info("|---|---:|---:|---:|")
        for row in rows:
            cli_logger.info(
                f"| {row['preset']} | {row['estimated_seconds']:.3f} s | "
                f"{row['saved_vs_complete_seconds']:.3f} s | "
                f"{row['reduction_vs_complete_percent']:.1f}% |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
