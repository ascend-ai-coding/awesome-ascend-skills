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
"""Normalize op_name from _def.cpp OP_ADD registration.

Subcommands:
  find       Search for operator path by op_name via OP_ADD reverse lookup
  normalize  Normalize op_name from a known operator path

Usage:
  python3 normalize_op_name.py find --op-name lamb_next_mv --search-root /path/to/workspace
  python3 normalize_op_name.py normalize --op-path /path/to/lamb_next_m_v
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

_logger = logging.getLogger(__name__)


def camel_to_snake(name):
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def extract_op_add(def_cpp_path):
    pattern = re.compile(r'OP_ADD\s*\(\s*(\w+)\s*\)')
    matches = []
    with open(def_cpp_path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            m = pattern.search(line)
            if m:
                matches.append((m.group(1), lineno))
    return matches


def _is_def_cpp(path):
    return 'op_host' in path and path.endswith('_def.cpp')


def find_def_cpp_files(search_root):
    result = []
    for root, _, files in os.walk(search_root):
        result.extend(os.path.join(root, f) for f in files if _is_def_cpp(os.path.join(root, f)))
    return result


def find_op_dir(op_name, search_root):
    def_files = find_def_cpp_files(search_root)
    candidates = []
    for def_path in def_files:
        op_adds = extract_op_add(def_path)
        for op_type, lineno in op_adds:
            normalized = camel_to_snake(op_type)
            if normalized == op_name:
                op_host_dir = os.path.dirname(def_path)
                op_dir = os.path.dirname(op_host_dir)
                candidates.append({
                    "op_path": op_dir,
                    "op_type": op_type,
                    "source": f"{os.path.basename(def_path)}:{lineno}",
                    "has_op_kernel": os.path.isdir(os.path.join(op_dir, "op_kernel"))
                })
    valid = [c for c in candidates if c["has_op_kernel"]]
    if valid:
        return valid
    return candidates


def cmd_find(args):
    results = find_op_dir(args.op_name, args.search_root)
    if not results:
        return {"op_path": None, "error": "not found"}, False
    if len(results) == 1:
        r = results[0]
        return {"op_path": r["op_path"], "op_type": r["op_type"], "source": r["source"]}, True
    return {
        "candidates": [{"op_path": r["op_path"], "op_type": r["op_type"], "source": r["source"]} for r in results],
        "count": len(results)
    }, True


def cmd_normalize(args):
    op_path = args.op_path
    op_host = os.path.join(op_path, "op_host")
    if not os.path.isdir(op_host):
        return {"op_name": None, "normalized": False, "error": "op_host/ not found"}, False
    def_files = [os.path.join(op_host, f) for f in os.listdir(op_host) if f.endswith('_def.cpp')]
    if not def_files:
        return {"op_name": None, "normalized": False, "error": "no *_def.cpp found"}, False
    all_adds = []
    for df in def_files:
        for op_type, lineno in extract_op_add(df):
            all_adds.append((op_type, os.path.basename(df), lineno))
    if not all_adds:
        return {"op_name": None, "normalized": False, "error": "OP_ADD not found"}, False
    dir_name = os.path.basename(os.path.normpath(op_path))
    if len(all_adds) == 1:
        op_type, fname, lineno = all_adds[0]
    else:
        best = None
        best_dist = float('inf')
        for op_type, fname, lineno in all_adds:
            sn = camel_to_snake(op_type)
            dist = sum(a != b for a, b in zip(sn, dir_name)) + abs(len(sn) - len(dir_name))
            if dist < best_dist:
                best_dist = dist
                best = (op_type, fname, lineno)
        op_type, fname, lineno = best
    normalized_name = camel_to_snake(op_type)
    return {
        "op_name": normalized_name,
        "op_type": op_type,
        "dir_name": dir_name,
        "normalized": normalized_name != dir_name,
        "source": f"{fname}:{lineno}"
    }, True


def main():
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logging.basicConfig(handlers=[_handler])
    parser = argparse.ArgumentParser(description="Normalize op_name from OP_ADD registration")
    sub = parser.add_subparsers(dest="command", required=True)

    p_find = sub.add_parser("find", help="Find operator path by op_name via OP_ADD reverse lookup")
    p_find.add_argument("--op-name", required=True, help="Operator name (snake_case)")
    p_find.add_argument("--search-root", required=True, help="Root directory to search")

    p_norm = sub.add_parser("normalize", help="Normalize op_name from a known operator path")
    p_norm.add_argument("--op-path", required=True, help="Operator directory path")

    args = parser.parse_args()
    if args.command == "find":
        result, ok = cmd_find(args)
    else:
        result, ok = cmd_normalize(args)
    _logger.info(json.dumps(result))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
