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
"""Update S2P1_path_list.json with reachability/group data and validate.

Usage:
  python3 update_path_list.py \
    --path-list S2P1_path_list.json \
    --param-def S2P2_param_def.json \
    --reach-data S2P2_reachability_data.json \
    --op-name resize_bilinear_v2 \
    --platform "npu_arch=DAV_3510, soc_version=ASCEND950, chip_model=Ascend950PR, core_count=64, ub_size=248"

Updates path_list.json in-place, then runs 6 validation checks.
Exit 0 on success, exit 1 on validation failure.
"""

import json
import argparse
import logging
import sys
from collections import Counter

_logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(format="%(message)s")
    parser = argparse.ArgumentParser(description="Update path_list with reachability data")
    parser.add_argument("--path-list", required=True, help="Path to S2P1_path_list.json")
    parser.add_argument("--param-def", required=True, help="Path to S2P2_param_def.json")
    parser.add_argument("--reach-data", required=True, help="Path to S2P2_reachability_data.json")
    parser.add_argument("--op-name", required=True, help="Operator name")
    parser.add_argument("--platform", required=True, help="Platform string")
    args = parser.parse_args()

    with open(args.path_list, encoding="utf-8") as f:
        path_list = json.load(f)
    with open(args.param_def, encoding="utf-8") as f:
        param_def = json.load(f)
    with open(args.reach_data, encoding="utf-8") as f:
        reach_data = json.load(f)

    reach_map = {p["id"]: p for p in reach_data["paths"]}

    path_list["op_name"] = args.op_name
    path_list["platform"] = args.platform
    path_list["groups"] = reach_data["groups"]

    for path in path_list["paths"]:
        pid = path["id"]
        if pid not in reach_map:
            continue
        rd = reach_map[pid]
        path["reachability"] = rd["reachability"]
        if "group" in rd:
            path["group"] = rd["group"]
        if "dead_reason" in rd:
            path["dead_reason"] = rd["dead_reason"]

    with open(args.path_list, "w", encoding="utf-8") as f:
        json.dump(path_list, f, indent=2, ensure_ascii=False)

    errors = validate(path_list, param_def)
    if errors:
        _logger.info("Validation FAILED:")
        for e in errors:
            _logger.info("  %s", e)
        sys.exit(1)

    counts = Counter(p["reachability"] for p in path_list["paths"])
    _logger.info("Validation PASSED:")
    _logger.info("  Total paths: %d", len(path_list['paths']))
    for status, count in sorted(counts.items()):
        _logger.info("    %s: %d", status, count)
    _logger.info("  Groups: %s", path_list['groups'])
    sys.exit(0)


def validate(path_list, param_def):
    errors = []
    paths = path_list["paths"]
    valid_statuses = {"reachable", "api_dead", "api_warn", "dead", "disputed"}

    # [1] reachability 全覆盖
    for p in paths:
        r = p.get("reachability")
        if r is None or r not in valid_statuses:
            errors.append(f"[1] {p['id']}: reachability={r} invalid")

    # [2] 数量等式
    counts = Counter(p.get("reachability") for p in paths)
    total_counted = sum(counts.get(s, 0) for s in valid_statuses)
    if total_counted != len(paths):
        errors.append(f"[2] count mismatch: {total_counted} != {len(paths)}")

    # [3] reachable 必有 group
    for p in paths:
        if p.get("reachability") == "reachable" and not p.get("group"):
            errors.append(f"[3] {p['id']}: reachable but no group")

    # [4] groups 列表一致
    s1_groups = set(path_list.get("groups", []))
    s2_groups = set(g["id"] for g in param_def["groups"])
    if s1_groups != s2_groups:
        errors.append(f"[4] groups mismatch: S1={s1_groups} S2={s2_groups}")

    # [5] 无空 group
    referenced = set(p["group"] for p in paths if p.get("reachability") == "reachable" and p.get("group"))
    for gid in s2_groups:
        if gid not in referenced:
            errors.append(f"[5] empty group: {gid}")

    # [6] per_dtype 路径覆盖
    for g in param_def["groups"]:
        gid = g["id"]
        reachable_paths = set(p["id"] for p in paths if p.get("reachability") == "reachable" and p.get("group") == gid)
        all_covered = set()
        per_dtype_paths = {}
        for dtype, entries in g["per_dtype"].items():
            covered = set(e["path"] for e in entries)
            per_dtype_paths[dtype] = covered
            all_covered |= covered
            extra = covered - reachable_paths
            if extra:
                errors.append(f"[6] {gid}/{dtype}: extra paths {extra}")
        if all_covered != reachable_paths:
            missing = reachable_paths - all_covered
            if missing:
                errors.append(f"[6] {gid}: missing paths across all dtypes: {missing}")

    return errors


if __name__ == "__main__":
    main()
