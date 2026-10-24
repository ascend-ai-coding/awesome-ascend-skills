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
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

COL_W = 22
LEAF_INFO_SKIP_OPS = frozenset({"PHASE1", "PHASE2", "SYNC_SRC", "SYNC_DST"})


def parse_dyn_topo(path):
    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 11:
                continue
            succs = [int(x) for x in row[10:] if x.strip().isdigit()]
            rows.append(
                {
                    "seqNo": int(row[0]),
                    "taskId": int(row[1]),
                    "opmagic": int(row[4]),
                    "leafHash": row[6],
                    "coreType": int(row[7]),
                    "psgId": int(row[8]),
                    "successors": succs,
                }
            )
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
            lvl.append(
                {
                    "leafHash": node["leafHash"],
                    "opmagic": node["opmagic"],
                    "psgId": node["psgId"],
                    "n_children": len(node["children"]),
                }
            )
            nxt.extend(node["children"])
        levels.append(lvl)
        cur = nxt
    return levels


def level_sig(levels):
    return tuple(
        tuple(sorted((n["leafHash"], n["opmagic"]) for n in lvl)) for lvl in levels
    )


def _format_level_hashes(level):
    hashes = [node["leafHash"] for node in level]
    if len(hashes) == 1:
        return hashes[0]
    return "".join(f"{leaf_hash:<{COL_W}s}" for leaf_hash in hashes).rstrip()


def _format_last_annotations(level):
    if len(level) <= 1:
        return None
    annotations = [f"(op={node['opmagic']})" for node in level]
    return "".join(f"{annotation:<{COL_W}s}" for annotation in annotations).rstrip()


def _single_node_connections(child_count):
    if child_count == 1:
        return ["  │", "  ▼"]
    if child_count == 2:
        return ["  ├──────────────┐", "  ▼              ▼"]
    arrows = [" " * (index * COL_W) + "▼" for index in range(child_count)]
    return ["\n".join(arrows)]


def _mixed_connections(child_counts):
    connectors = []
    for child_count in child_counts:
        if child_count == 2:
            connectors.append("├────┐")
        elif child_count == 1:
            connectors.append("  ▼  ")
        else:
            connectors.append("     ")
    arrows = [
        "▼    ▼"[: child_count * 2 - 1] if child_count > 0 else ""
        for child_count in child_counts
    ]
    return ["  ".join(connectors), "  ".join(arrows)]


def _multi_node_connections(level, child_counts):
    if all(child_count == 2 for child_count in child_counts):
        connectors = "         ".join("├────────┐" for _ in level)
        arrows = "        ".join("▼        ▼" for _ in level)
        return [f"  {connectors}", f"  {arrows}"]
    if all(child_count <= 1 for child_count in child_counts):
        arrows = "  ".join("▼" for _ in range(sum(child_counts)))
        return [f"  {arrows}"]
    return _mixed_connections(child_counts)


def render_chain(levels):
    lines = []
    for index, level in enumerate(levels):
        lines.append(_format_level_hashes(level))
        if index == len(levels) - 1:
            annotation = _format_last_annotations(level)
            if annotation is not None:
                lines.append(annotation)
            continue

        child_counts = [node["n_children"] for node in level]
        if len(level) == 1:
            lines.extend(_single_node_connections(child_counts[0]))
        else:
            lines.extend(_multi_node_connections(level, child_counts))
    return "\n".join(lines)


def load_leaf_info(path):
    if not path.exists():
        return {}
    with open(path) as f:
        prog = json.load(f)
    info = {}
    for func in prog.get("functions", []):
        leaf_hash = func.get("hash")
        if leaf_hash is None:
            continue
        raw_operations = func.get("operations", [])
        operations = [operation.get("opcode") for operation in raw_operations]
        info[str(leaf_hash)] = {
            "name": func.get("func_magicname", ""),
            "ops": [op for op in operations if op not in LEAF_INFO_SKIP_OPS],
        }
    return info


def extract_core_ops(ops):
    skip_set = {
        "PHASE1",
        "PHASE2",
        "SYNC_SRC",
        "SYNC_DST",
        "COPY_IN",
        "COPY_OUT",
        "BAR.V",
        "BAR.M",
        "EXPAND",
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
                logging.info(
                    f"  {n['leafHash']}: op={n['opmagic']}, psg={n['psgId']}, {lb}"
                )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="AIV 依赖链分析工具（sg_set_scope 合图优化）"
    )
    parser.add_argument("output_dir", help="包含 dyn_topo.txt 的输出目录路径")
    parser.add_argument("--json", default=None, help="输出 JSON 文件路径（可选）")
    return parser.parse_args()


def _require_topology(output_dir):
    topo = output_dir / "dyn_topo.txt"
    if not topo.exists():
        raise FileNotFoundError(f"错误: {topo} 不存在")
    return topo


def _build_chain_groups(starts, aiv_succ, keyed, stop_set=None):
    groups = defaultdict(list)
    for key in starts:
        tree = build_tree(key, aiv_succ, keyed, stop_set=stop_set)
        if tree is None:
            continue
        levels = tree_to_levels(tree)
        groups[level_sig(levels)].append((key, levels))
    return groups


def _log_analysis_header(topo, rows, starts, lh_has_cube):
    aiv_count = sum(1 for row in rows if row["coreType"] == 0)
    logging.info("=" * 80)
    logging.info(
        "AIV Dependency Chain Analysis (sg_set_scope graph merge optimization)"
    )
    logging.info("=" * 80)
    logging.info(f"\nData: {topo}")
    logging.info(f"Total tasks: {len(rows)}, AIV: {aiv_count}, Starts: {len(starts)}")
    if lh_has_cube:
        logging.info(f"leafHash with cube successors: {sorted(lh_has_cube)}")


def _chain_count(items, graph):
    levels = items[0][1]
    leaf_counts = count_start_rows(
        [key for key, _ in items], graph["keyed"], graph["rows"]
    )
    start_leaf_hash = graph["keyed"][items[0][0]]["leafHash"]
    return levels, leaf_counts.get(start_leaf_hash, len(items))


def _is_isolated(levels):
    return len(levels) == 1 and levels[0][0]["n_children"] == 0


def _json_levels(levels):
    return [
        [
            {
                "leafHash": node["leafHash"],
                "opmagic": node["opmagic"],
                "psgId": node["psgId"],
            }
            for node in level
        ]
        for level in levels
    ]


def _log_original_chain(label, count, levels, leaf_info):
    logging.info(f"\nChain {label} ({count} times)")
    if _is_isolated(levels):
        node = levels[0][0]
        logging.info(f"{node['leafHash']}  (isolated, no dependencies)")
        if node["leafHash"] in leaf_info:
            leaf = leaf_info[node["leafHash"]]
            logging.info(
                f"  opmagic={node['opmagic']}, psgId={node['psgId']}, "
                f"{infer_label(leaf['ops'])}"
            )
    else:
        logging.info(render_chain(levels))
        print_chain_detail(levels, leaf_info)


def _process_original_groups(groups, graph, leaf_info, include_json):
    json_chains = []
    for index, items in enumerate(groups.values()):
        label = chr(ord("A") + index)
        levels, count = _chain_count(items, graph)
        _log_original_chain(label, count, levels, leaf_info)
        if include_json:
            json_chains.append(
                {
                    "label": label,
                    "count": count,
                    "levels": _json_levels(levels),
                }
            )
    return json_chains


def _log_suggestion_header():
    logging.info(f"\n{'=' * 80}")
    logging.info("sg_set_scope optimization suggestions")
    logging.info(f"{'=' * 80}")
    logging.info(
        "\nRule: when an AIV node has cube successors, keep the node but do not expand further"
    )
    logging.info(
        "      Chain segments with >=2 nodes after truncation and psgId "
        "changes are suggested to merge with sg_set_scope\n"
    )


def _collect_cut_points(levels, lh_has_cube):
    cut_points = []
    for level in levels:
        for node in level:
            if node["leafHash"] in lh_has_cube:
                cut_points.append(node["leafHash"])
    return cut_points


def _unique_psg_path(nodes):
    path = []
    for node in nodes:
        if not path or path[-1] != node["psgId"]:
            path.append(node["psgId"])
    return path


def _log_suggestion(suggestion, leaf_info, lh_has_cube):
    index = suggestion["id"]
    count = suggestion["count"]
    levels = suggestion["levels"]
    nodes = suggestion["nodes"]
    can_merge = suggestion["can_merge"]
    cut_points = suggestion["cut_points"]
    psg_path = " → ".join(str(node["psgId"]) for node in nodes)
    logging.info(
        f"  Suggestion {index}: {len(nodes)} nodes after truncation, "
        f"{count} times, psgId changes: {psg_path}"
    )

    if _is_isolated(levels):
        logging.info(f"    {levels[0][0]['leafHash']}")
    else:
        for line in render_chain(levels).split("\n"):
            logging.info(f"    {line}")

    for node in nodes:
        leaf = leaf_info.get(node["leafHash"], {})
        label = infer_label(leaf.get("ops", [])) if leaf else ""
        cube_mark = " [✂ cube boundary]" if node["leafHash"] in lh_has_cube else ""
        logging.info(f"    {node['leafHash']}: psg={node['psgId']}, {label}{cube_mark}")

    if can_merge:
        merge_path = " → ".join(str(psg_id) for psg_id in _unique_psg_path(nodes))
        logging.info(
            f"    → Suggestion: wrap the vector operations of psgId {merge_path} with sg_set_scope"
        )
    else:
        logging.info("    → Single node or psgId unchanged, no sg_set_scope needed")
    if cut_points:
        logging.info(f"    ✂ Cut points (cube successors): {cut_points}")
    logging.info("")


def _process_suggestion_groups(groups, graph, leaf_info, include_json):
    suggestions = []
    for items in groups.values():
        levels, count = _chain_count(items, graph)
        nodes = collect_leaf_hashes(levels)
        can_merge = len(nodes) >= 2 and len({node["psgId"] for node in nodes}) >= 2
        cut_points = _collect_cut_points(levels, graph["cube_hashes"])
        if not can_merge and not cut_points:
            continue

        suggestion_id = len(suggestions) + 1
        suggestion = {
            "id": suggestion_id,
            "count": count,
            "can_merge": can_merge,
            "nodes": nodes,
            "levels": levels,
            "cut_points": cut_points,
        }
        _log_suggestion(suggestion, leaf_info, graph["cube_hashes"])
        suggestions.append(
            {
                "id": suggestion_id,
                "count": count,
                "can_merge": can_merge,
                "psgid_transition": [node["psgId"] for node in nodes],
                "cut_points": cut_points,
                "levels": _json_levels(levels),
            }
        )
    return suggestions if include_json else []


def _write_json(path, chains, suggestions):
    output_path = Path(path)
    with open(output_path, "w") as output_file:
        json.dump(
            {"chains": chains, "suggestions": suggestions},
            output_file,
            indent=2,
            ensure_ascii=False,
        )
    logging.info(f"\nJSON written to: {output_path}")


def main():
    args = _parse_args()
    output_dir = Path(args.output_dir)
    topo = _require_topology(output_dir)
    rows = parse_dyn_topo(topo)
    keyed, aiv_succ, starts, lh_has_cube = build_aiv_graph(rows)
    leaf_info = load_leaf_info(output_dir / "program.json")
    graph = {"keyed": keyed, "rows": rows, "cube_hashes": lh_has_cube}

    _log_analysis_header(topo, rows, starts, lh_has_cube)
    groups = _build_chain_groups(starts, aiv_succ, keyed)
    logging.info(f"Deduped chains: {len(groups)}")
    json_chains = _process_original_groups(groups, graph, leaf_info, bool(args.json))

    _log_suggestion_header()
    json_suggestions = []
    if not lh_has_cube:
        logging.info("  No AIV node has cube successors, no truncation needed")
        logging.info(
            "  For full chains with psgId changes, merge the segments with sg_set_scope"
        )
    else:
        cut_groups = _build_chain_groups(starts, aiv_succ, keyed, stop_set=lh_has_cube)
        json_suggestions = _process_suggestion_groups(
            cut_groups, graph, leaf_info, bool(args.json)
        )

    if args.json:
        _write_json(args.json, json_chains, json_suggestions)
    logging.info(f"\n{'=' * 80}")
    logging.info("Analysis complete")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        logging.error("%s", exc)
        raise SystemExit(1) from None
