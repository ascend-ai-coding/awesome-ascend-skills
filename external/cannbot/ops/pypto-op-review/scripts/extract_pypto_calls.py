#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
List PyPTO API call sites in a Python file (line number + call expression).

Used for line-by-line / op-by-op review when debugging PyPTO-specific errors.
Resolves: import pypto; pypto.matmul(...), pypto.frontend.jit(...), import pypto as p; p.view(...)
Also records: from pypto import matmul; matmul(...)  (tagged as short-import).

Usage:
  python3 extract_pypto_calls.py <path/to/kernel.py>
  python3 extract_pypto_calls.py <path/to/kernel.py> --json
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Set


@dataclass
class CallSite:
    line: int
    col: int
    call: str
    kind: str  # "attr" | "short_import"


def _attr_chain(func: ast.AST) -> Optional[List[str]]:
    if isinstance(func, ast.Name):
        return [func.id]
    if isinstance(func, ast.Attribute):
        inner = _attr_chain(func.value)
        if inner is None:
            return None
        return inner + [func.attr]
    return None


def _collect_pypto_imports(tree: ast.AST) -> "tuple[Set[str], Set[str]]":
    """Return (pypto module roots, short `from pypto import ...` names)."""
    pypto_roots: Set[str] = set()
    short_from_pypto: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            pypto_roots.update(alias.asname or "pypto" for alias in node.names if alias.name == "pypto")
        elif isinstance(node, ast.ImportFrom) and node.module == "pypto":
            short_from_pypto.update(alias.asname or alias.name for alias in node.names if alias.name != "*")
    return pypto_roots, short_from_pypto


def _collect_call_sites(tree: ast.AST, pypto_roots: Set[str], short_from_pypto: Set[str]) -> List[CallSite]:
    sites: List[CallSite] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        chain = _attr_chain(func)
        if chain and chain[0] in pypto_roots:
            sites.append(CallSite(line=node.lineno, col=node.col_offset, call=f"{'.'.join(chain)}(...)", kind="attr"))
        elif isinstance(func, ast.Name) and func.id in short_from_pypto:
            sites.append(CallSite(line=node.lineno, col=node.col_offset, call=f"{func.id}(...)", kind="short_import"))
    return sites


def extract_pypto_calls(source: str, path: str) -> List[CallSite]:
    tree = ast.parse(source, filename=path)
    pypto_roots, short_from_pypto = _collect_pypto_imports(tree)
    sites = _collect_call_sites(tree, pypto_roots, short_from_pypto)
    return sorted(sites, key=lambda s: (s.line, s.col))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Extract pypto call sites for op-by-op debugging.")
    ap.add_argument("path", type=Path, help="Python source file")
    ap.add_argument("--json", action="store_true", help="JSON lines output")
    args = ap.parse_args()

    p = args.path
    if not p.is_file():
        logging.error("Not a file: %s", p)
        sys.exit(1)

    src = p.read_text(encoding="utf-8", errors="replace")
    try:
        sites = extract_pypto_calls(src, str(p))
    except SyntaxError as e:
        logging.error("SyntaxError: %s", e)
        sys.exit(2)

    if args.json:
        logging.info(json.dumps([asdict(s) for s in sites], indent=2))
        return

    logging.info("# %s — %d pypto-related call site(s)\n", p, len(sites))
    for i, s in enumerate(sites, 1):
        tag = f"[{s.kind}]"
        logging.info("%4d  L%5d  %-14s  %s", i, s.line, tag, s.call)


if __name__ == "__main__":
    main()
