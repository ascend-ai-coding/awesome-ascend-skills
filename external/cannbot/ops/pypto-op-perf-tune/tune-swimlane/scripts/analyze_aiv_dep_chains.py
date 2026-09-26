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
analyze_aiv_dep_chains.py - AIV 依赖链分析工具（sg_set_scope 合图优化）

从 dyn_topo.txt 中提取 AIV(vector) 任务之间的依赖链路，
用于指导 sg_set_scope 合图优化的插入位置。

用法:
    python3 analyze_aiv_dep_chains.py <output_dir>
    python3 analyze_aiv_dep_chains.py <output_dir> --json result.json

输入:
    output_dir/dyn_topo.txt  — 任务动态拓扑（含 successors 依赖）
    output_dir/program.json  — 程序编译数据（可选，用于标注操作类型）

原理:
    dyn_topo.txt 中同一 taskId 可能有多行（不同 seqNo），代表同一逻辑任务
    被多次调度执行。本脚本按 (seqNo, taskId) 作为唯一键构建依赖图，
    并按行数（实际执行次数）统计链路出现次数。
"""

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')

COL_W = 22


def parse_dyn_topo(path):
    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 11:
                continue
            succs = [int(x) for x in row[10:] if x.strip().isdigit()]
            rows.append({
                "seqNo": int(row[0]),
                "taskId": int(row[1]),
                "opmagic": int(row[4]),
                "leafHash": row[6],
                "coreType": int(row[7]),
                "psgId": int(row[8]),
                "successors": succs,
            })
    return rows


def build_aiv_graph(rows):
    keyed = {}
    for r in rows:
        key = (r["seqNo"], r["taskId"])
        if key not in keyed:
            keyed[key] = {
                "key": key,
                "taskId": r["taskId"],
                "seqNo": r["seqNo"],
                "opmagic": r["opmagic"],
                "leafHash": r["leafHash"],
                "coreType": r["coreType"],
                "psgId": r["psgId"],
                "successors": r["successors"],
            }

    aiv_keys = {k for k, v in keyed.items() if v["coreType"] == 0}
    aic_keys = {k for k, v in keyed.items() if v["coreType"] == 1}

    aiv_succ = defaultdict(list)
    aiv_pred = defaultdict(set)
    has_cube_succ = set()

    for k in aiv_keys:
        task = keyed.get(k)
        if task is None:
            continue
        sn = task["seqNo"]
        seen_succ = set()
        for s_tid in task["successors"]:
            s_key = (sn, s_tid)
            if s_key in aic_keys:
                has_cube_succ.add(k)
            if s_key in aiv_keys and s_key not in seen_succ:
                seen_succ.add(s_key)
                aiv_succ[k].append(s_key)
                aiv_pred[s_key].add(k)

    lh_has_cube = set()
    for k in has_cube_succ:
        t = keyed.get(k)
        if t is not None:
            lh_has_cube.add(t["leafHash"])

    starts = sorted(k for k in aiv_keys if not aiv_pred[k])
    return keyed, aiv_succ, starts, lh_has_cube


def build_tree(key, aiv_succ, keyed, on_path=None, stop_set=None):
    if on_path is None:
        on_path = set()
    if key in on_path:
        return None
    on_path.add(key)
    t = keyed[key]
    children = []
    if stop_set is None or t["leafHash"] not in stop_set:
        for s in sorted(aiv_succ.get(key, [])):
            c = build_tree(s, aiv_succ, keyed, on_path, stop_set)
            if c:
                children.append(c)
    on_path.discard(key)
    return {
        "leafHash": t["leafHash"],
        "opmagic": t["opmagic"],
        "psgId": t["psgId"],
        "children": children,
    }


def tree_to_levels(tree):
    levels = []
    cur = [tree]
    while cur:
        lvl = []
        nxt = []
        for node in cur:
            lvl.append({
                "leafHash": node["leafHash"],
                "opmagic": node["opmagic"],
                "psgId": node["psgId"],
                "n_children": len(node["children"]),
            })
            nxt.extend(node["children"])
        levels.append(lvl)
        cur = nxt
    return levels


def level_sig(levels):
    return tuple(
        tuple(sorted((n["leafHash"], n["opmagic"]) for n in lvl))
        for lvl in levels
    )


def render_chain(levels):
    lines = []
    for i, lvl in enumerate(levels):
        is_last = i == len(levels) - 1

        hashes = [n["leafHash"] for n in lvl]
        if len(hashes) == 1:
            lines.append(hashes[0])
        else:
            lines.append("".join(f"{h:<{COL_W}s}" for h in hashes).rstrip())

        if is_last:
            if len(lvl) > 1:
                anns = [f"(op={n['opmagic']})" for n in lvl]
                lines.append("".join(f"{a:<{COL_W}s}" for a in anns).rstrip())
            continue

        nxt = levels[i + 1]
        nc = [n["n_children"] for n in lvl]

        if len(lvl) == 1:
            c = nc[0]
            if c == 1:
                lines.append("  │")
                lines.append("  ▼")
            elif c == 2:
                lines.append("  ├──────────────┐")
                lines.append("  ▼              ▼")
            else:
                segs = []
                for j in range(c):
                    col = j * COL_W
                    segs.append(" " * col + "▼")
                lines.append("\n".join(segs))
        else:
            if all(x == 2 for x in nc):
                conns = "         ".join("├────────┐" for _ in lvl)
                lines.append(f"  {conns}")
                downs = "        ".join("▼        ▼" for _ in lvl)
                lines.append(f"  {downs}")
            elif all(x <= 1 for x in nc):
                total = sum(nc)
                downs = "  ".join("▼" for _ in range(total))
                lines.append(f"  {downs}")
            else:
                parts = []
                for n in nc:
                    if n == 2:
                        parts.append("├────┐")
                    elif n == 1:
                        parts.append("  ▼  ")
                    else:
                        parts.append("     ")
                lines.append("  ".join(parts))
                dparts = []
                for n in nc:
                    dparts.append("▼    ▼"[:n * 2 - 1] if n > 0 else "")
                lines.append("  ".join(dparts))

    return "\n".join(lines)


def load_leaf_info(path):
    if not path.exists():
        return {}
    with open(path) as f:
        prog = json.load(f)
    info = {}
    for func in prog.get("functions", []):
        h = func.get("hash")
        if h is not None:
            ops = []
            for op in func.get("operations", []):
                ops.append(op.get("opcode"))
            skip_set = {"PHASE1", "PHASE2", "SYNC_SRC", "SYNC_DST"}
            key_ops = []
            for o in ops:
                if o not in skip_set:
                    key_ops.append(o)
            info[str(h)] = {
                "name": func.get("func_magicname", ""),
                "ops": key_ops,
            }
    return info


def extract_core_ops(ops):
    skip_set = {
        "PHASE1", "PHASE2", "SYNC_SRC", "SYNC_DST",
        "COPY_IN", "COPY_OUT", "BAR.V", "BAR.M", "EXPAND",
    }
    core = []
    for o in ops:
        if o not in skip_set and not o.startswith("L1_TO_L0"):
            core.append(o)
    return core


def infer_label(ops):
    core = extract_core_ops(ops)
    is_cube = any(o in ("A_MUL_B", "A_MULACC_B") for o in core)
    tag = "cube" if is_cube else "vec"
    if core:
        return f"[{tag}] {'+'.join(core[:12])}"
    return f"[{tag}] (no compute ops)"


def count_start_rows(starts, keyed, rows):
    start_lh_set = defaultdict(set)
    for k in starts:
        t = keyed[k]
        start_lh_set[t["leafHash"]].add(k)

    lh_row_count = {}
    for r in rows:
        if r["coreType"] == 0:
            lh = r["leafHash"]
            k = (r["seqNo"], r["taskId"])
            if k in start_lh_set.get(lh, set()):
                lh_row_count[lh] = lh_row_count.get(lh, 0) + 1

    return lh_row_count


def collect_leaf_hashes(levels):
    seen = []
    seen_set = set()
    for lvl in levels:
        for n in lvl:
            if n["leafHash"] not in seen_set:
                seen_set.add(n["leafHash"])
                seen.append(n)
    return seen


def print_chain_detail(lvls, leaf_info):
    seen_h = set()
    for lvl in lvls:
        for n in lvl:
            if n["leafHash"] not in seen_h:
                seen_h.add(n["leafHash"])
                li = leaf_info.get(n["leafHash"], {})
                lb = infer_label(li.get("ops", [])) if li else ""
                logging.info(f"  {n['leafHash']}: op={n['opmagic']}, psg={n['psgId']}, {lb}")


def main():
    parser = argparse.ArgumentParser(
        description="AIV 依赖链分析工具（sg_set_scope 合图优化）"
    )
    parser.add_argument("output_dir", help="包含 dyn_topo.txt 的输出目录路径")
    parser.add_argument("--json", default=None, help="输出 JSON 文件路径（可选）")
    args = parser.parse_args()

    d = Path(args.output_dir)
    topo = d / "dyn_topo.txt"
    prog = d / "program.json"

    if not topo.exists():
        logging.error(f"错误: {topo} 不存在")
        sys.exit(1)

    rows = parse_dyn_topo(topo)
    keyed, aiv_succ, starts, lh_has_cube = build_aiv_graph(rows)
    leaf_info = load_leaf_info(prog)

    aiv_count = sum(1 for r in rows if r["coreType"] == 0)

    # === Part 1: 原始链路 ===
    logging.info("=" * 80)
    logging.info("AIV 依赖链分析（sg_set_scope 合图优化）")
    logging.info("=" * 80)
    logging.info(f"\n数据: {topo}")
    logging.info(f"总任务: {len(rows)}, AIV: {aiv_count}, 起点: {len(starts)}")
    if lh_has_cube:
        logging.info(f"后继含 cube 的 leafHash: {sorted(lh_has_cube)}")

    chains = []
    for k in starts:
        tree = build_tree(k, aiv_succ, keyed)
        if tree:
            lvls = tree_to_levels(tree)
            sig = level_sig(lvls)
            chains.append((k, lvls, sig))

    groups = defaultdict(list)
    for k, lvls, sig in chains:
        groups[sig].append((k, lvls))

    logging.info(f"去重后链路: {len(groups)} 条")

    json_chains = []
    json_suggestions = []
    ch = ord("A")
    for _, items in groups.items():
        label = chr(ch)
        ch += 1
        lvls = items[0][1]

        lh_row_count = count_start_rows([k for k, _ in items], keyed, rows)
        start_lh = keyed[items[0][0]]["leafHash"]
        count = lh_row_count.get(start_lh, len(items))

        if len(lvls) == 1 and lvls[0][0]["n_children"] == 0:
            n = lvls[0][0]
            logging.info(f"\n链路{label}（{count}次）")
            logging.info(f"{n['leafHash']}  (孤立，无依赖)")
            if n["leafHash"] in leaf_info:
                li = leaf_info[n["leafHash"]]
                logging.info(f"  opmagic={n['opmagic']}, psgId={n['psgId']}, {infer_label(li['ops'])}")
        else:
            logging.info(f"\n链路{label}（{count}次）")
            logging.info(render_chain(lvls))
            print_chain_detail(lvls, leaf_info)

        if args.json:
            json_chains.append({
                "label": label,
                "count": count,
                "levels": [
                    [{"leafHash": n["leafHash"], "opmagic": n["opmagic"], "psgId": n["psgId"]}
                     for n in lvl]
                    for lvl in lvls
                ],
            })

    # === Part 2: 优化建议（cube 边界截断） ===
    logging.info(f"\n{'=' * 80}")
    logging.info("sg_set_scope 优化建议")
    logging.info(f"{'=' * 80}")
    logging.info("\n规则: 遇到后继含 cube 的 AIV 节点时保留该节点，但不继续展开")
    logging.info("      截断后 >=2 节点且 psgId 有变化的链段建议 sg_set_scope 合并\n")

    if not lh_has_cube:
        logging.info("  所有 AIV 节点均无 cube 后继，无需截断")
        logging.info("  建议对完整链路中 psgId 有变化的段进行 sg_set_scope 合并")
    else:
        cut_chains = []
        for k in starts:
            tree = build_tree(k, aiv_succ, keyed, stop_set=lh_has_cube)
            if tree:
                lvls = tree_to_levels(tree)
                sig = level_sig(lvls)
                cut_chains.append((k, lvls, sig))

        cut_groups = defaultdict(list)
        for k, lvls, sig in cut_chains:
            cut_groups[sig].append((k, lvls))

        sugg_idx = 0
        for _, items in cut_groups.items():
            lvls = items[0][1]
            lh_row_count = count_start_rows([k for k, _ in items], keyed, rows)
            start_lh = keyed[items[0][0]]["leafHash"]
            count = lh_row_count.get(start_lh, len(items))

            all_nodes = collect_leaf_hashes(lvls)
            unique_psgids = set(n["psgId"] for n in all_nodes)
            can_merge = len(all_nodes) >= 2 and len(unique_psgids) >= 2

            cut_points = []
            for lvl in lvls:
                for n in lvl:
                    if n["leafHash"] in lh_has_cube:
                        cut_points.append(n["leafHash"])

            if not can_merge and not cut_points:
                continue

            sugg_idx += 1
            psg_path = " → ".join(str(n["psgId"]) for n in all_nodes)
            logging.info(
                f"  建议 {sugg_idx}: 截断后 {len(all_nodes)} 个节点, "
                f"{count} 次, psgId 变化: {psg_path}"
            )

            if len(lvls) == 1 and lvls[0][0]["n_children"] == 0:
                logging.info(f"    {lvls[0][0]['leafHash']}")
            else:
                for line in render_chain(lvls).split("\n"):
                    logging.info(f"    {line}")

            for n in all_nodes:
                li = leaf_info.get(n["leafHash"], {})
                lb = infer_label(li.get("ops", [])) if li else ""
                cube_mark = " [✂ cube边界]" if n["leafHash"] in lh_has_cube else ""
                logging.info(f"    {n['leafHash']}: psg={n['psgId']}, {lb}{cube_mark}")

            if can_merge:
                psg_path = []
                for n in all_nodes:
                    if not psg_path or psg_path[-1] != n["psgId"]:
                        psg_path.append(n["psgId"])
                logging.info(f"    → 建议: 用 sg_set_scope 包裹 psgId {' → '.join(str(p) for p in psg_path)} 的 vector 操作")
            else:
                logging.info(f"    → 单节点或 psgId 无变化, 无需 sg_set_scope")

            if cut_points:
                logging.info(f"    ✂ 截断点 (后继含 cube): {cut_points}")

            logging.info("")

            if args.json:
                json_suggestions.append({
                    "id": sugg_idx,
                    "count": count,
                    "can_merge": can_merge,
                    "psgid_transition": [n["psgId"] for n in all_nodes],
                    "cut_points": cut_points,
                    "levels": [
                        [{"leafHash": n["leafHash"], "opmagic": n["opmagic"], "psgId": n["psgId"]}
                         for n in lvl]
                        for lvl in lvls
                    ],
                })

    if args.json:
        out = Path(args.json)
        with open(out, "w") as f:
            json.dump({"chains": json_chains, "suggestions": json_suggestions}, f, indent=2, ensure_ascii=False)
        logging.info(f"\nJSON 已写入: {out}")

    logging.info(f"\n{'=' * 80}")
    logging.info("分析完成")


if __name__ == "__main__":
    main()
