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
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

SKIP_OPS = frozenset(
    [
        "PHASE1",
        "PHASE2",
        "SYNC_SRC",
        "SYNC_DST",
    ]
)


def _collect_file_lines(operations):
    file_lines = defaultdict(list)
    for operation in operations:
        file_path = operation.get("file", "")
        line = operation.get("line", "")
        if file_path and line:
            file_lines[file_path].append((int(line), operation.get("opcode", "")))
    return dict(file_lines)


def _program_entry(func):
    operations = func.get("operations", [])
    all_ops = [operation.get("opcode", "") for operation in operations]
    return {
        "name": func.get("func_magicname", ""),
        "all_ops": all_ops,
        "core_ops": [opcode for opcode in all_ops if opcode not in SKIP_OPS],
        "file_lines": _collect_file_lines(operations),
        "incasts": func.get("incasts", []),
        "outcasts": func.get("outcasts", []),
        "rawtensor_symbols": [
            tensor.get("symbol", "")
            for tensor in func.get("rawtensors", [])
            if tensor.get("symbol")
        ],
        "subfunc_symbols": [
            tensor.get("symbol", "")
            for tensor in (func.get("subfunc_param") or {}).get("tensors") or []
            if tensor.get("symbol")
        ],
    }


def load_program_info(prog_path):
    """从 program.json 提取每个 leafHash 的代码行映射。"""
    with open(prog_path) as f:
        prog = json.load(f)

    info = {}
    for func in prog.get("functions", []):
        info[str(func.get("hash", ""))] = _program_entry(func)
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
    skip = frozenset(
        [
            "PHASE1",
            "PHASE2",
            "SYNC_SRC",
            "SYNC_DST",
            "COPY_IN",
            "COPY_OUT",
            "BAR.V",
            "BAR.M",
            "EXPAND",
        ]
    )
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


def _parse_args():
    parser = argparse.ArgumentParser(description="leafHash → 前端代码行映射工具")
    parser.add_argument("output_dir", help="包含 merged_swimlane.json 的输出目录路径")
    parser.add_argument(
        "--leafhash", default=None, help="只显示指定 leafHash 的映射（可选）"
    )
    parser.add_argument("--json", default=None, help="输出 JSON 文件路径（可选）")
    return parser.parse_args()


def _require_program(path):
    prog_path = path / "program.json"
    if not prog_path.exists():
        raise FileNotFoundError(f"错误: {prog_path} 不存在")
    return prog_path


def _log_input_summary(output_dir, prog_info, topo_info):
    logging.info("=" * 80)
    logging.info("leafHash → Frontend Code Line Mapping")
    logging.info("=" * 80)
    logging.info(f"\nData: {output_dir}")
    logging.info(f"program.json function count: {len(prog_info)}")
    if topo_info:
        logging.info(f"dyn_topo.txt leafHash count: {len(topo_info)}")


def _leaf_details(leaf_hash, info, topo_info):
    compute_ops = extract_compute_ops(info["core_ops"])
    code_ranges = get_code_range(info["file_lines"])
    topo = topo_info.get(leaf_hash, {})
    core_type = topo.get("coreType", "?")
    return {
        "compute_ops": compute_ops,
        "is_cube": any(op in ("A_MUL_B", "A_MULACC_B") for op in compute_ops),
        "code_ranges": code_ranges,
        "location": format_code_location(code_ranges),
        "rootIndex": topo.get("rootIndex", "?"),
        "psgIds": sorted(topo.get("psgIds", set())),
        "taskCount": topo.get("count", 0),
        "coreType": core_type,
        "coreTypeName": {0: "VEC", 1: "CUBE", 4: "FAKE"}.get(core_type, "?"),
    }


def _log_leaf(leaf_hash, info, details):
    cube_suffix = " (cube)" if details["is_cube"] else ""
    logging.info(f"\n--- {leaf_hash} ---")
    logging.info(f"  Type: {details['coreTypeName']}{cube_suffix}")
    logging.info(f"  Function: {info['name']}")
    if details["taskCount"]:
        logging.info(
            f"  Execution count: {details['taskCount']}, "
            f"rootIndex: {details['rootIndex']}, psgId: {details['psgIds']}"
        )
    compute_text = (
        " + ".join(details["compute_ops"]) if details["compute_ops"] else "(none)"
    )
    logging.info(f"  Compute ops: {compute_text}")
    logging.info(f"  Code location: {details['location']}")

    if info["subfunc_symbols"]:
        logging.info(f"  Subfunction tensor: {info['subfunc_symbols']}")
    for file_path, code_range in details["code_ranges"].items():
        file_name = file_path.rsplit("/", 1)[-1] if "/" in file_path else file_path
        logging.info(f"  {file_name} line details:")
        for line in sorted(code_range["lines"]):
            operations = ", ".join(code_range["lines"][line])
            logging.info(f"    L{line}: {operations}")


def _result_entry(info, details):
    return {
        "name": info["name"],
        "compute_ops": details["compute_ops"],
        "is_cube": details["is_cube"],
        "code_ranges": {
            file_path: {
                "min": code_range["min"],
                "max": code_range["max"],
                "lines": code_range["lines"],
            }
            for file_path, code_range in details["code_ranges"].items()
        },
        "rootIndex": details["rootIndex"],
        "psgIds": details["psgIds"],
        "taskCount": details["taskCount"],
        "coreType": details["coreType"],
        "subfunc_symbols": info["subfunc_symbols"],
        "rawtensor_symbols": info["rawtensor_symbols"],
    }


def _collect_results(prog_info, topo_info, filter_leaf_hash):
    results = {}
    for leaf_hash, info in sorted(prog_info.items()):
        if filter_leaf_hash and leaf_hash != filter_leaf_hash:
            continue

        details = _leaf_details(leaf_hash, info, topo_info)
        _log_leaf(leaf_hash, info, details)
        results[leaf_hash] = _result_entry(info, details)
    return results


def main():
    args = _parse_args()
    output_dir = Path(args.output_dir)
    prog_info = load_program_info(_require_program(output_dir))
    topo_info = load_dyn_topo_info(output_dir / "dyn_topo.txt")
    _log_input_summary(output_dir, prog_info, topo_info)
    results = _collect_results(prog_info, topo_info, args.leafhash)

    if args.json:
        out = Path(args.json)
        with open(out, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logging.info(f"\nJSON written to: {out}")

    logging.info(f"\n{'=' * 80}")
    logging.info("Mapping complete")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        logging.error("%s", exc)
        raise SystemExit(1) from None
