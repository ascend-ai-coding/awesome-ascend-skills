#!/usr/bin/env python3
# coding: utf-8
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
Analyze core usage rate per leafHash from swimlane data.

Compares actual used cores against theoretical core count from hardware,
to determine whether cores are fully utilized before applying merge optimizations.

Usage:
    python3 analyze_core_usage.py <output_dir> [--device-id N]

Output:
    Per-leafHash core usage report with:
    - psgId: subgraph id
    - core_type: AIC or AIV
    - tasks: total task count
    - used/total: actual used cores / theoretical core count (usage %)
    - avg(us): average task duration
    - total(us): total task duration
    - status: FULL or NOT FULL
    - suggestion: optimization suggestion (fill cores first, or proceed to merge)

Theoretical core count is obtained from torch.npu.get_device_properties(),
which returns chip-level cube_core_num (AIC) and vector_core_num (AIV),
corresponding to SoC::GetAICCoreNum() / SoC::GetAIVCoreNum() in platform.h.
"""

import logging
import argparse
import json
import os
import re
import sys
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(message)s')


def get_theoretical_cores(device_id):
    """Get theoretical core count from hardware via torch.npu.

    Returns (aic_cores, aiv_cores) or (None, None) on failure.
    """
    try:
        import torch
        import torch_npu
        prop = torch.npu.get_device_properties(device_id)
        return prop.cube_core_num, prop.vector_core_num
    except Exception as e:
        logging.warning(f"Failed to get theoretical cores from torch.npu: {e}")
        logging.warning("Falling back to trace metadata (may underestimate).")
        return None, None


def parse_swimlane(events):
    """Parse traceEvents to extract per-leafHash core usage and timing.

    Returns:
        tid_to_core: dict mapping tid -> core name
        leafhash_cores: dict mapping leafHash -> set of core names
        leafhash_info: dict mapping leafHash -> {psgId, cnt, total_dur}
    """
    tid_to_core = {}
    for ev in events:
        if ev.get("ph") == "M" and ev.get("name") == "thread_name":
            tid_to_core[ev.get("tid")] = ev.get("args", {}).get("name", "")

    leafhash_cores = defaultdict(set)
    leafhash_info = {}

    for ev in events:
        if ev.get("ph") != "X":
            continue
        hint = ev.get("args", {}).get("event-hint", "")
        m = re.search(r"leafHash:(\d+)", hint)
        if not m:
            continue
        lh = m.group(1)
        if lh == "0":
            continue

        core_name = tid_to_core.get(ev.get("tid"), "")
        name_parts = ev.get("name", "").replace("()", "").split("-")
        psg_id = name_parts[-1] if len(name_parts) >= 5 else "?"
        dur = ev.get("dur", 0)

        leafhash_cores[lh].add(core_name)
        if lh not in leafhash_info:
            leafhash_info[lh] = {"psgId": psg_id, "cnt": 0, "total_dur": 0}
        leafhash_info[lh]["cnt"] += 1
        leafhash_info[lh]["total_dur"] += dur

    return tid_to_core, leafhash_cores, leafhash_info


def main():
    parser = argparse.ArgumentParser(
        description="Analyze per-leafHash core usage rate from swimlane data.")
    parser.add_argument("output_dir",
                        help="Path to output directory containing merged_swimlane.json")
    parser.add_argument("--device-id", type=int, default=0,
                        help="NPU device ID for querying theoretical core count (default: 0)")
    args = parser.parse_args()

    base_dir = args.output_dir.rstrip("/")

    swimlane_path = os.path.join(base_dir, "merged_swimlane.json")
    if not os.path.exists(swimlane_path):
        logging.error(f"{swimlane_path} not found")
        sys.exit(1)

    with open(swimlane_path, "r") as f:
        data = json.load(f)
    events = data.get("traceEvents", [])

    tid_to_core, leafhash_cores, leafhash_info = parse_swimlane(events)

    # Get theoretical core count (guaranteed int after fallback)
    aic_theoretical, aiv_theoretical = get_theoretical_cores(args.device_id)

    if aic_theoretical is None or aiv_theoretical is None:
        all_real = {tid_to_core[tid] for tid in tid_to_core
                    if "Fake" not in tid_to_core.get(tid, "")}
        aic_theoretical = len([c for c in all_real if "AIC" in c])
        aiv_theoretical = len([c for c in all_real if "AIV" in c])

    logging.info(f"Theoretical cores: AIC={aic_theoretical}, AIV={aiv_theoretical}")

    # Print table header
    header = (f'{"psgId":>5} | {"type":>4} | {"tasks":>5} | '
              f'{"used/total (usage%)":>22} | {"avg(us)":>8} | {"total(us)":>9} | '
              f'{"status":>9} | suggestion')
    sep = "-" * len(header)

    logging.info("")
    logging.info(header)
    logging.info(sep)

    not_full_items = []
    full_items = []

    sorted_lhs = sorted(leafhash_info, key=lambda x: leafhash_info[x]["total_dur"], reverse=True)
    for lh in sorted_lhs:
        info = leafhash_info[lh]
        real_cores = {c for c in leafhash_cores[lh] if "Fake" not in c}
        if not real_cores:
            continue

        is_aic = any("AIC" in c for c in real_cores)
        ct = "AIC" if is_aic else "AIV"
        theoretical = aic_theoretical if is_aic else aiv_theoretical
        if theoretical == 0:
            continue
        used = len(real_cores)
        avg = info["total_dur"] / info["cnt"] if info["cnt"] > 0 else 0
        usage_pct = used / theoretical * 100
        is_full = used >= theoretical

        cores_str = f"{used}/{theoretical} ({usage_pct:.0f}%)"
        status = "FULL" if is_full else "NOT FULL"

        if is_full:
            suggestion = "can merge (L1Reuse/NBuffer)"
            full_items.append(info["psgId"])
        else:
            suggestion = "FILL CORES FIRST (reduce TileShape)"
            not_full_items.append((info["psgId"], ct, used, theoretical, info["total_dur"]))

        logging.info(
            f'{info["psgId"]:>5} | {ct:>4} | {info["cnt"]:>5} | {cores_str:>22} | '
            f'{avg:>8.1f} | {info["total_dur"]:>9.1f} | {status:>9} | {suggestion}')

    logging.info("")
    logging.info("=" * len(header))
    logging.info("Summary:")
    logging.info(f"  NOT FULL (fill cores first): {len(not_full_items)} leafHash(es)")
    for psg_id, ct, used, total, dur in not_full_items:
        logging.info(f"    - psgId={psg_id} ({ct}): {used}/{total} cores, total={dur:.1f}us")
    logging.info(f"  FULL (can merge): {len(full_items)} leafHash(es)")
    for psg_id in full_items:
        logging.info(f"    - psgId={psg_id}")
    logging.info("")

    if not_full_items:
        logging.info("Next step: For NOT FULL leafHashes, use leafhash_to_code.py to map to frontend code,")
        logging.info("           then adjust set_cube_tile_shapes() to increase task count.")
    else:
        logging.info("Next step: All cores are FULL. Proceed to merge tuning (L1Reuse / CubeNBuffer / VecNBuffer).")


if __name__ == "__main__":
    main()
