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

import logging
import argparse
import json
import re

logging.basicConfig(level=logging.INFO, format='%(message)s')


def parse_hashorder_hint(hint_text):
    result = {}
    for line in hint_text.split('\n'):
        line = line.strip()
        m = re.match(r'(\w+) hashOrder: (\S+), subGraphCount: (\d+)', line)
        if m:
            result[m.group(1)] = {
                'hashOrder': m.group(2),
                'subGraphCount': int(m.group(3))
            }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Analyze swimlane data: leafHash stats, hashOrder, merge tuning guide")
    parser.add_argument("output_dir", help="Output directory with swimlane data")
    parser.add_argument("--outer-loops", type=int, default=1,
                        help="Outer loop iteration count (default=1; " \
                             "manually calculate from kernel code for correct t/iter)")
    args = parser.parse_args()

    base_dir = args.output_dir.rstrip("/")

    with open(f"{base_dir}/merged_swimlane.json", "r") as f:
        data = json.load(f)
    events = data.get("traceEvents", data)

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

    if not leaf_hashorder:
        logging.warning("No hashOrder-hint found in merged_swimlane.json. "
                        "Please ensure the runtime supports hashOrder-hint output.")

    with open(f"{base_dir}/program.json", "r") as f:
        prog = json.load(f)

    root_name = {}
    for func in prog["functions"]:
        if func.get("graphtype") == 2:
            h = str(func.get("hash", ""))
            magic = func.get("func_magicname", func.get("funcmagic", ""))
            root_name[h] = magic.replace("TENSOR_LOOP_", "")

    leaf_ops = {}
    skip_ops = {"COPY_IN", "COPY_OUT", "SYNC_SRC", "SYNC_DST", "PHASE1", "PHASE2", "PHASE3"}
    for func in prog["functions"]:
        if func.get("graphtype") == 3:
            h = str(func.get("hash", ""))
            opcodes = [op.get("opcode", "?") for op in func.get("operations", [])]
            compute = [o for o in opcodes if o not in skip_ops]
            leaf_ops[h] = "+".join(compute) if compute else "(pure copy)"

    outer_loops = args.outer_loops

    logging.info(f"{'#':<3} {'leafHash':<22} "
          f"{'min(us)':>9} {'max(us)':>9} {'avg(us)':>9} {'total(us)':>10} "
          f"{'core':>4} {'hashOrder':<12} {'subGCnt':>7} {'t/iter':>6} "
          f"{'root_name':<42} {'compute_ops'}")
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
        logging.info(f"{i:<3} {lh:<22} "
              f"{min(durs):>9.2f} {max(durs):>9.2f} {sum(durs)/len(durs):>9.2f} {sum(durs):>10.2f} "
              f"{ct:>4} {ho:<12} {sgc_str:>7} {tpi_str:>6} {rn:<42} {ops}")

    logging.info(f"\nouter_loops={outer_loops}")

    def build_merge_stats(info_key):
        merge_stats = {}
        for lh, st in leaf_stats.items():
            ho_info = leaf_hashorder.get(lh, {})
            info = ho_info.get(info_key)
            if not info:
                continue
            ho = info['hashOrder']
            sgc = info['subGraphCount']
            if ho not in merge_stats:
                merge_stats[ho] = {'subGraphCount': sgc, 'durs': [], 'core': st['core']}
            merge_stats[ho]['durs'].extend(st['durs'])
        return merge_stats

    l1_stats = build_merge_stats('l1ReuseInfo')
    cube_stats = build_merge_stats('cubeMergeInfo')
    vec_stats = build_merge_stats('vecMergeInfo')

    logging.info(f"\n{'=' * 80}")
    logging.info("Merge Tuning Guide (hashOrder = merge key)")
    logging.info(f"{'=' * 80}")

    if l1_stats:
        logging.info(f"\n[AIC] cube_l1_reuse_setting:")
        for ho in sorted(l1_stats.keys(), key=lambda x: -sum(l1_stats[x]['durs'])):
            st = l1_stats[ho]
            sgc = st['subGraphCount']
            tpi = sgc / outer_loops
            avg = sum(st['durs']) / len(st['durs'])
            line = f"  hashOrder={ho}: subGraphCount={sgc}, t/iter={tpi:.0f}, avg={avg:.2f}us"
            if tpi > 1:
                vals = [v for v in [2, 4, 8, 16] if v <= tpi * 4]
                vals_str = '/'.join(str(v) for v in vals)
                default_str = '/'.join(str(v) for v in vals[:2])
                line += f"\n    -> integer key: {{-1: {vals_str}}} (global)"
                line += (
                    f"\n    -> func key:    "
                    f"{{\"DEFAULT\": {default_str}, \"{ho}\": {vals_str}}} (specific)"
                )
            logging.info(line)

    if cube_stats:
        logging.info(f"\n[AIC] cube_nbuffer_setting:")
        for ho in sorted(cube_stats.keys(), key=lambda x: -sum(cube_stats[x]['durs'])):
            st = cube_stats[ho]
            sgc = st['subGraphCount']
            tpi = sgc / outer_loops
            avg = sum(st['durs']) / len(st['durs'])
            line = f"  hashOrder={ho}: subGraphCount={sgc}, t/iter={tpi:.0f}, avg={avg:.2f}us"
            if tpi > 1:
                vals = [v for v in [2, 4, 8, 16] if v <= tpi * 4]
                vals_str = '/'.join(str(v) for v in vals)
                default_str = '/'.join(str(v) for v in vals[:2])
                line += f"\n    -> integer key: {{-1: {vals_str}}} (global)"
                line += (
                    f"\n    -> func key:    "
                    f"{{\"DEFAULT\": {default_str}, \"{ho}\": {vals_str}}} (specific)"
                )
            logging.info(line)

    if vec_stats:
        logging.info(f"\n[AIV] vec_nbuffer_setting:")
        for ho in sorted(vec_stats.keys(), key=lambda x: -sum(vec_stats[x]['durs'])):
            st = vec_stats[ho]
            sgc = st['subGraphCount']
            tpi = sgc / outer_loops
            avg = sum(st['durs']) / len(st['durs'])
            line = f"  hashOrder={ho}: subGraphCount={sgc}, t/iter={tpi:.0f}, avg={avg:.2f}us"
            if tpi > 1:
                vals = [v for v in [2, 4, 8, 16] if v <= tpi * 4]
                vals_str = '/'.join(str(v) for v in vals)
                line += f"\n    -> integer key: {{-2: 1, -1: {vals_str}}} (global)"
                line += (
                    f"\n    -> func key:    "
                    f"{{\"DEFAULT\": 1, \"{ho}\": {vals_str}}} (specific)"
                )
            logging.info(line)


if __name__ == "__main__":
    main()
