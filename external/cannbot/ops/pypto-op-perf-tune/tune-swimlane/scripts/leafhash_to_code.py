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
leafhash_to_code.py - leafHash → 前端代码行映射工具

从 program.json 中提取每个 leafHash 对应的前端代码文件和行号，
用于 sg_set_scope 优化时定位具体插入位置。

用法:
    python3 leafhash_to_code.py <output_dir>
    python3 leafhash_to_code.py <output_dir> --leafhash 3907163356593077760
    python3 leafhash_to_code.py <output_dir> --json result.json

输入:
    output_dir/program.json  — 程序编译数据（含 operations 的 file/line）
    output_dir/dyn_topo.txt  — 任务动态拓扑（可选，用于 rootIndex 和 psgId）
"""

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')

SKIP_OPS = frozenset([
    "PHASE1", "PHASE2", "SYNC_SRC", "SYNC_DST",
])


def load_program_info(prog_path):
    """从 program.json 提取每个 leafHash 的代码行映射。"""
    with open(prog_path) as f:
        prog = json.load(f)

    info = {}
    for func in prog.get("functions", []):
        h = str(func.get("hash", ""))

        all_ops = []
        core_ops = []
        file_lines = {}

        for op in func.get("operations", []):
            opcode = op.get("opcode", "")
            all_ops.append(opcode)

            if opcode not in SKIP_OPS:
                core_ops.append(opcode)

            f_val = op.get("file", "")
            l_val = op.get("line", "")
            if f_val and l_val:
                if f_val not in file_lines:
                    file_lines[f_val] = []
                file_lines[f_val].append((int(l_val), opcode))

        info[h] = {
            "name": func.get("func_magicname", ""),
            "all_ops": all_ops,
            "core_ops": core_ops,
            "file_lines": file_lines,
            "incasts": func.get("incasts", []),
            "outcasts": func.get("outcasts", []),
            "rawtensor_symbols": [
                t.get("symbol", "") for t in func.get("rawtensors", [])
                if t.get("symbol")
            ],
            "subfunc_symbols": [
                t.get("symbol", "")
                for t in (func.get("subfunc_param") or {}).get("tensors") or []
                if t.get("symbol")
            ],
        }
    return info


def load_dyn_topo_info(topo_path):
    """从 dyn_topo.txt 提取每个 leafHash 的 rootIndex 和 psgId。"""
    if not topo_path.exists():
        return {}

    lh_info = {}
    with open(topo_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 11:
                continue
            lh = row[6]
            if lh not in lh_info:
                lh_info[lh] = {
                    "rootIndex": int(row[2]),
                    "psgIds": set(),
                    "coreType": int(row[7]),
                    "count": 0,
                }
            lh_info[lh]["psgIds"].add(int(row[8]))
            lh_info[lh]["count"] += 1
    return lh_info


def extract_compute_ops(ops):
    """过滤掉框架指令，保留实际计算指令。"""
    skip = frozenset([
        "PHASE1", "PHASE2", "SYNC_SRC", "SYNC_DST",
        "COPY_IN", "COPY_OUT", "BAR.V", "BAR.M", "EXPAND",
    ])
    return [o for o in ops if o not in skip and not o.startswith("L1_TO_L0")]


def get_code_range(file_lines):
    """对每个源文件，返回涉及的行号范围和对应操作。"""
    ranges = {}
    for fpath, lines in file_lines.items():
        if not lines:
            continue
        sorted_lines = sorted(lines, key=lambda x: x[0])
        min_line = sorted_lines[0][0]
        max_line = sorted_lines[-1][0]
        line_ops = defaultdict(list)
        for ln, opc in sorted_lines:
            line_ops[ln].append(opc)
        ranges[fpath] = {
            "min": min_line,
            "max": max_line,
            "lines": dict(line_ops),
        }
    return ranges


def format_code_location(code_ranges):
    """格式化代码位置信息。"""
    parts = []
    for fpath, r in code_ranges.items():
        fname = fpath.rsplit("/", 1)[-1] if "/" in fpath else fpath
        if r["min"] == r["max"]:
            parts.append(f"{fname}:{r['min']}")
        else:
            parts.append(f"{fname}:{r['min']}-{r['max']}")
    return ", ".join(parts) if parts else "(no source info)"


def main():
    parser = argparse.ArgumentParser(
        description="leafHash → 前端代码行映射工具"
    )
    parser.add_argument("output_dir", help="包含 merged_swimlane.json 的输出目录路径")
    parser.add_argument("--leafhash", default=None,
                        help="只显示指定 leafHash 的映射（可选）")
    parser.add_argument("--json", default=None, help="输出 JSON 文件路径（可选）")
    args = parser.parse_args()

    d = Path(args.output_dir)
    prog_path = d / "program.json"
    topo_path = d / "dyn_topo.txt"

    if not prog_path.exists():
        logging.error(f"错误: {prog_path} 不存在")
        sys.exit(1)

    prog_info = load_program_info(prog_path)
    topo_info = load_dyn_topo_info(topo_path)

    filter_lh = args.leafhash

    logging.info("=" * 80)
    logging.info("leafHash → 前端代码行映射")
    logging.info("=" * 80)
    logging.info(f"\n数据: {d}")
    logging.info(f"program.json 函数数: {len(prog_info)}")
    if topo_info:
        logging.info(f"dyn_topo.txt leafHash 数: {len(topo_info)}")

    results = {}

    for h, info in sorted(prog_info.items()):
        if filter_lh and h != filter_lh:
            continue

        compute_ops = extract_compute_ops(info["core_ops"])
        if not compute_ops:
            is_cube = False
        else:
            is_cube = any(o in ("A_MUL_B", "A_MULACC_B") for o in compute_ops)

        code_ranges = get_code_range(info["file_lines"])
        loc_str = format_code_location(code_ranges)

        ti = topo_info.get(h, {})
        root_idx = ti.get("rootIndex", "?")
        psg_ids = sorted(ti.get("psgIds", set()))
        task_cnt = ti.get("count", 0)
        core_type = ti.get("coreType", "?")
        ct_name = {0: "VEC", 1: "CUBE", 4: "FAKE"}.get(core_type, "?")

        has_cube_succ = False

        logging.info(f"\n--- {h} ---")
        logging.info(f"  类型: {ct_name}{' (cube)' if is_cube else ''}")
        logging.info(f"  函数: {info['name']}")
        if task_cnt:
            logging.info(f"  执行次数: {task_cnt}, rootIndex: {root_idx}, psgId: {psg_ids}")
        logging.info(f"  计算指令: {' + '.join(compute_ops) if compute_ops else '(无)'}")
        logging.info(f"  代码位置: {loc_str}")

        if info["subfunc_symbols"]:
            logging.info(f"  子函数 tensor: {info['subfunc_symbols']}")

        for fpath, r in code_ranges.items():
            fname = fpath.rsplit("/", 1)[-1] if "/" in fpath else fpath
            logging.info(f"  {fname} 行号详情:")
            for ln in sorted(r["lines"]):
                ops_str = ", ".join(r["lines"][ln])
                logging.info(f"    L{ln}: {ops_str}")

        results[h] = {
            "name": info["name"],
            "compute_ops": compute_ops,
            "is_cube": is_cube,
            "code_ranges": {
                fpath: {"min": r["min"], "max": r["max"], "lines": r["lines"]}
                for fpath, r in code_ranges.items()
            },
            "rootIndex": root_idx,
            "psgIds": psg_ids,
            "taskCount": task_cnt,
            "coreType": core_type,
            "subfunc_symbols": info["subfunc_symbols"],
            "rawtensor_symbols": info["rawtensor_symbols"],
        }

    if args.json:
        out = Path(args.json)
        with open(out, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logging.info(f"\nJSON 已写入: {out}")

    logging.info(f"\n{'=' * 80}")
    logging.info("映射完成")


if __name__ == "__main__":
    main()
