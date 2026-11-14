#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Analyze swimlane data for PyPTO operator performance tuning.

Extracts per-leafHash statistics and merge tuning guidance from:
  - merged_swimlane.json: task durations, core types, and hashOrder-hint

The hashOrder-hint in each event's args provides merge info:
  - l1ReuseInfo hashOrder/subGraphCount  -> cube_l1_reuse_setting
  - cubeMergeInfo hashOrder/subGraphCount -> cube_nbuffer_setting
  - vecMergeInfo hashOrder/subGraphCount  -> vec_nbuffer_setting

Usage:
    python3 analyze_swimlane.py <output_dir> --outer-loops N

outer-loops 必须由用户手动计算后传入，不可自动检测。
计算方式：阅读 kernel 函数中的 pypto.loop() 嵌套结构，
每层循环的迭代次数 = 循环变量的取值范围，
outer_loops = 所有非最内层 pypto.loop() 的迭代次数乘积。
（最内层循环通常是带 unroll_list 的那个，不参与 outer_loops 计算）

Output columns:
  #        - rank by total time
  leafHash - leaf function hash
  min/max/avg/total(us) - duration statistics
  core     - AIC or AIV (from swimlane tid metadata)
  hashOrder - merge group key (from hashOrder-hint)
  subGCnt  - subGraphCount, same-structure subgraph count (from hashOrder-hint)
  t/iter   - subGraphCount / outer_loops, subgraphs per root function per iteration
  root_name - root function name
  compute_ops - compute opcodes (excludes COPY_IN/OUT, SYNC, PHASE)

outer_loops is auto-detected as GCD of all subGraphCount values.
Override with --outer-loops if needed.

如何确定 outer-loops：
1. 阅读 kernel 函数中的 pypto.loop() 嵌套结构
2. 每层循环的迭代次数 = 循环变量的取值范围
3. outer_loops = 所有非最内层 pypto.loop() 的迭代次数乘积
   （最内层循环通常是带 unroll_list 的那个，不参与 outer_loops 计算）
4. 如果有多个不嵌套的循环序列，需要根据 subGraphCount 与各循环次数的匹配关系确定哪些子图属于哪个循环

⚠️ 如果 auto 值不正确（如 GCD=1），必须手动计算并传入 --outer-loops。

Merge Tuning Guide:
  Groups by hashOrder for each merge type, uses t/iter to guide granularity.
"""

import argparse
import json
import logging
import re

logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_hashorder_hint(hint_text):
    result = {}
    for line in hint_text.split("\n"):
        line = line.strip()
        m = re.match(r"(\w+) hashOrder: (\S+), subGraphCount: (\d+)", line)
        if m:
            result[m.group(1)] = {
                "hashOrder": m.group(2),
                "subGraphCount": int(m.group(3)),
            }
    return result


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze swimlane data: leafHash stats, hashOrder, merge tuning guide"
    )
    parser.add_argument("output_dir", help="Output directory with swimlane data")
    parser.add_argument(
        "--outer-loops",
        type=int,
        default=1,
        help="Outer loop iteration count (default=1; "
        "manually calculate from kernel code for correct t/iter)",
    )
    return parser.parse_args()


def _load_json(path):
    with open(path, "r") as json_file:
        return json.load(json_file)


def _map_tid_to_core(events):
    tid_core = {}
    for ev in events:
        if ev.get("ph") == "M" and ev.get("name") == "thread_name":
            name = ev.get("args", {}).get("name", "")
            tid = ev.get("tid", 0)
            if "AIC" in name:
                tid_core[tid] = "AIC"
            elif "AIV" in name:
                tid_core[tid] = "AIV"
            else:
                tid_core[tid] = name
    return tid_core


def _collect_leaf_data(events, tid_core):
    leaf_stats = {}
    leaf_hashorder = {}
    leaf_root = {}

    for ev in events:
        if ev.get("ph") != "X":
            continue
        args_data = ev.get("args", {})
        hint = args_data.get("event-hint", "")

        m = re.search(r"leafHash:(\d+)", hint)
        if not m:
            continue
        lh = m.group(1)
        if lh == "0":
            continue

        m_rh = re.search(r"rootHash:(\d+)", hint)
        root_hash = m_rh.group(1) if m_rh else ""

        ho_hint = args_data.get("hashOrder-hint", "")
        ho_info = parse_hashorder_hint(ho_hint) if ho_hint else {}

        dur = ev.get("dur", 0)
        ct = tid_core.get(ev.get("tid", 0), "?")

        if lh not in leaf_stats:
            leaf_stats[lh] = {"durs": [], "core": ct}
        leaf_stats[lh]["durs"].append(dur)

        if lh not in leaf_hashorder and ho_info:
            leaf_hashorder[lh] = ho_info

        if lh not in leaf_root and root_hash:
            leaf_root[lh] = root_hash
    return leaf_stats, leaf_hashorder, leaf_root


def _extract_program_metadata(prog):
    root_name = {}
    for func in prog["functions"]:
        if func.get("graphtype") == 2:
            h = str(func.get("hash", ""))
            magic = func.get("func_magicname", func.get("funcmagic", ""))
            root_name[h] = magic.replace("TENSOR_LOOP_", "")

    leaf_ops = {}
    skip_ops = {
        "COPY_IN",
        "COPY_OUT",
        "SYNC_SRC",
        "SYNC_DST",
        "PHASE1",
        "PHASE2",
        "PHASE3",
    }
    for func in prog["functions"]:
        if func.get("graphtype") == 3:
            h = str(func.get("hash", ""))
            opcodes = [op.get("opcode", "?") for op in func.get("operations", [])]
            compute = [opcode for opcode in opcodes if opcode not in skip_ops]
            leaf_ops[h] = "+".join(compute) if compute else "(pure copy)"
    return root_name, leaf_ops


def _log_leaf_stats(leaf_data, program_data, outer_loops):
    leaf_stats = leaf_data["stats"]
    leaf_hashorder = leaf_data["hashorder"]
    leaf_root = leaf_data["root"]
    root_name = program_data["root_name"]
    leaf_ops = program_data["leaf_ops"]
    logging.info(
        f"{'#':<3} {'leafHash':<22} "
        f"{'min(us)':>9} {'max(us)':>9} {'avg(us)':>9} {'total(us)':>10} "
        f"{'core':>4} {'hashOrder':<12} {'subGCnt':>7} {'t/iter':>6} "
        f"{'root_name':<42} {'compute_ops'}"
    )
    logging.info("-" * 240)

    sorted_items = sorted(leaf_stats.items(), key=lambda x: -sum(x[1]["durs"]))
    for i, (lh, st) in enumerate(sorted_items, 1):
        durs = st["durs"]
        ho_info = leaf_hashorder.get(lh, {})
        ct = st["core"]

        if ct == "AIC":
            ho = ho_info.get("l1ReuseInfo", {}).get("hashOrder", "?")
            sgc = ho_info.get("l1ReuseInfo", {}).get("subGraphCount", 0)
        else:
            ho = ho_info.get("vecMergeInfo", {}).get("hashOrder", "?")
            sgc = ho_info.get("vecMergeInfo", {}).get("subGraphCount", 0)

        t_per_iter = sgc / outer_loops if isinstance(sgc, int) and sgc > 0 else 0

        rh = leaf_root.get(lh, "")
        rn = root_name.get(rh, "?")
        if len(rn) > 40:
            rn = rn[:37] + "..."
        ops = leaf_ops.get(lh, "?")
        if len(ops) > 100:
            ops = ops[:97] + "..."

        sgc_str = str(sgc) if sgc > 0 else "?"
        tpi_str = f"{t_per_iter:.1f}" if sgc > 0 else "?"
        average = sum(durs) / len(durs)
        logging.info(
            f"{i:<3} {lh:<22} "
            f"{min(durs):>9.2f} {max(durs):>9.2f} {average:>9.2f} {sum(durs):>10.2f} "
            f"{ct:>4} {ho:<12} {sgc_str:>7} {tpi_str:>6} {rn:<42} {ops}"
        )

    logging.info(f"\nouter_loops={outer_loops}")


def build_merge_stats(leaf_stats, leaf_hashorder, info_key):
    merge_stats = {}
    for leaf_hash, stats in leaf_stats.items():
        info = leaf_hashorder.get(leaf_hash, {}).get(info_key)
        if not info:
            continue
        hash_order = info["hashOrder"]
        if hash_order not in merge_stats:
            merge_stats[hash_order] = {
                "subGraphCount": info["subGraphCount"],
                "durs": [],
            }
        merge_stats[hash_order]["durs"].extend(stats["durs"])
    return merge_stats


def _format_merge_suggestion(hash_order, tasks_per_iteration, is_vector):
    values = [value for value in [2, 4, 8, 16] if value <= tasks_per_iteration * 4]
    values_text = "/".join(str(value) for value in values)
    if is_vector:
        return (
            f"\n    -> integer key: {{-1: {values_text}}} (global)"
            f"\n    -> func key:    "
            f'{{"DEFAULT": 1, "{hash_order}": {values_text}}} (specific)'
        )

    default_text = "/".join(str(value) for value in values[:2])
    return (
        f"\n    -> integer key: {{-1: {values_text}}} (global)"
        f"\n    -> func key:    "
        f'{{"DEFAULT": {default_text}, "{hash_order}": {values_text}}} (specific)'
    )


def _log_merge_stats(title, merge_stats, outer_loops, is_vector=False):
    if not merge_stats:
        return

    logging.info(f"\n{title}:")
    ordered_hashes = sorted(
        merge_stats,
        key=lambda hash_order: -sum(merge_stats[hash_order]["durs"]),
    )
    for hash_order in ordered_hashes:
        stats = merge_stats[hash_order]
        subgraph_count = stats["subGraphCount"]
        tasks_per_iteration = subgraph_count / outer_loops
        average = sum(stats["durs"]) / len(stats["durs"])
        line = (
            f"  hashOrder={hash_order}: subGraphCount={subgraph_count}, "
            f"t/iter={tasks_per_iteration:.0f}, avg={average:.2f}us"
        )
        if tasks_per_iteration > 1:
            line += _format_merge_suggestion(hash_order, tasks_per_iteration, is_vector)
        logging.info(line)


def _log_merge_guide(leaf_stats, leaf_hashorder, outer_loops):
    logging.info(f"\n{'=' * 80}")
    logging.info("Merge Tuning Guide (hashOrder = merge key)")
    logging.info(f"{'=' * 80}")
    categories = (
        ("[AIC] cube_l1_reuse_setting", "l1ReuseInfo", False),
        ("[AIC] cube_nbuffer_setting", "cubeMergeInfo", False),
        ("[AIV] vec_nbuffer_setting", "vecMergeInfo", True),
    )
    for title, info_key, is_vector in categories:
        merge_stats = build_merge_stats(leaf_stats, leaf_hashorder, info_key)
        _log_merge_stats(title, merge_stats, outer_loops, is_vector)


def main():
    args = _parse_args()
    base_dir = args.output_dir.rstrip("/")
    data = _load_json(f"{base_dir}/merged_swimlane.json")
    events = data.get("traceEvents", data)

    tid_core = _map_tid_to_core(events)
    leaf_stats, leaf_hashorder, leaf_root = _collect_leaf_data(events, tid_core)
    leaf_data = {
        "stats": leaf_stats,
        "hashorder": leaf_hashorder,
        "root": leaf_root,
    }
    if not leaf_hashorder:
        logging.warning(
            "No hashOrder-hint found in merged_swimlane.json. "
            "Please ensure the runtime supports hashOrder-hint output."
        )

    program = _load_json(f"{base_dir}/program.json")
    root_name, leaf_ops = _extract_program_metadata(program)
    program_data = {"root_name": root_name, "leaf_ops": leaf_ops}
    _log_leaf_stats(leaf_data, program_data, args.outer_loops)
    _log_merge_guide(leaf_stats, leaf_hashorder, args.outer_loops)


if __name__ == "__main__":
    main()
