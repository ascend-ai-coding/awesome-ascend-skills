# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
import json
import os
from pathlib import Path

from src.snapshot_construction.code_graph.code_map_0309 import CodeMap
from src.snapshot_construction.code_graph.clang_parser import ClangParser
from src.logger import info, debug, warning, error


# -------------------------------------------------------
# 功能一：每个文件单独的 code_map
# -------------------------------------------------------


def build_file_maps(local_path, output_dir):
    files_dir = os.path.join(output_dir, "files")
    os.makedirs(files_dir, exist_ok=True)

    parser = ClangParser(local_path)
    root = Path(local_path)

    for file in root.rglob("*"):
        if file.suffix not in [".cpp", ".c", ".h", ".hpp"]:
            continue

        p = str(file).replace("\\", "/")
        if "/tests/" in p or "/examples/" in p:
            continue

        nodes, edges = parser.parse_file(str(file))

        data = {"file": str(file), "nodes": nodes, "edges": edges}

        safe_name = file.name
        out_path = os.path.join(files_dir, f"{safe_name}_map.json")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# -------------------------------------------------------
# 功能二：所有 function 的汇总 json
# -------------------------------------------------------


def build_functions_summary(graph, output_dir):
    nodes = graph.nodes
    edges = graph.edges

    function_ids = set()
    function_path = {}

    for n in nodes:
        if n.get("type") == "Function":
            fid = n["id"]
            function_ids.add(fid)
            if "path" in n and n["path"]:
                function_path[fid] = n["path"]

    calls_map = {}
    called_by_map = {}
    returns_map = {}

    for e in edges:
        etype = e.get("type")
        src = e.get("source")
        tgt = e.get("target")

        if etype == "calls":
            if src in function_ids:
                calls_map.setdefault(src, [])
                if tgt not in calls_map[src]:
                    calls_map[src].append(tgt)
            if tgt in function_ids:
                called_by_map.setdefault(tgt, [])
                if src not in called_by_map[tgt]:
                    called_by_map[tgt].append(src)

        elif etype == "returns":
            if src in function_ids:
                returns_map.setdefault(src, [])
                if tgt not in returns_map[src]:
                    returns_map[src].append(tgt)

    result = []

    for fid in sorted(function_ids):
        entry = {
            "id": fid,
            "path": function_path.get(fid, None),
            "calls": calls_map.get(fid, []),
            "called_by": called_by_map.get(fid, []),
            "returns": returns_map.get(fid, []),
        }
        result.append(entry)

    out_path = os.path.join(output_dir, "functions.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


# -------------------------------------------------------
# 主流程
# -------------------------------------------------------


def main(operators, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    total = len(operators)
    success = 0
    skipped = 0

    for i, local_path in enumerate(operators, 1):
        if not os.path.isdir(local_path):
            debug(f"[{i}/{total}] SKIP (not found): {local_path}")
            skipped += 1
            continue

        debug(f"[{i}/{total}] Processing: {local_path}")

        try:
            # 1. 整体 code_map
            graph = CodeMap(local_path)
            graph.build()

            code_map_path = os.path.join(output_dir, "code_map.json")
            graph.export(code_map_path)

            debug(f"           -> code_map:  {code_map_path}")

            # 2. 每个文件 code_map
            build_file_maps(local_path, output_dir)
            debug(f"           -> file_maps: {output_dir}/files/")

            # 3. function 汇总
            build_functions_summary(graph, output_dir)
            debug(f"           -> functions: {output_dir}/functions.json")

            success += 1

        except Exception as e:
            error(f"           -> ERROR: {e}")
            skipped += 1

    info(f"Done. success={success}, skipped={skipped}, total={total}")


if __name__ == "__main__":
    # -------------------------------------------------------
    # 配置
    # -------------------------------------------------------

    OUTPUT_DIR = "../../../out/code_map_result"

    #  直接写本地路径（要分析的算子目录）
    operators = [
        "../../../data/ops-math/math/abs",
        "../../../data/ops-math/math/add",
    ]
    main(operators, output_dir=OUTPUT_DIR)
