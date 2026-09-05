#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Count effective lines in a PyPTO golden function.

Used by Stage 3 architect to compute decomposition complexity (see
``pypto-op-design/SKILL.md`` Round 0). "Effective lines" excludes:
  - blank lines
  - pure comment lines
  - docstrings (top-level string-expression inside the function body)
  - import statements (rarely appear inside the function body anyway)
  - decorator lines (attached to the function, not part of the body)

Only the bodies of functions whose name ends with ``_golden`` are counted.
If the file contains multiple ``*_golden`` functions (e.g. a tiled and a
non-tiled variant), the script counts the LARGEST one — that is the
"main" reference. This avoids double-counting when a file holds
equivalent variants.

Usage::

    python count_golden_lines.py <path/to/op_golden.py>

Output (stdout): one integer — the effective line count.

Exit code: 0 on success; 1 on parse failure or no ``*_golden`` function.
"""

from __future__ import annotations

import ast
import logging
import sys
from pathlib import Path

_LOGGER = logging.getLogger("count_golden_lines")


def _is_docstring(stmt: ast.stmt) -> bool:
    """Detect a docstring-style expression statement."""
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _count_item(item: object) -> int:
    """Effective-line contribution of one element of an AST list field.

    A nested statement recurses; an ``ExceptHandler`` contributes its body (the
    handler clause itself is not an ``ast.stmt``); anything else counts as 0.
    """
    if isinstance(item, ast.stmt):
        return _count_stmt(item)
    if isinstance(item, ast.ExceptHandler):
        return sum(_count_stmt(sub) for sub in item.body)
    return 0


def _count_stmt(stmt: ast.stmt) -> int:
    """Count effective lines in a single statement, recursing into bodies."""
    if _is_docstring(stmt):
        return 0
    total = 1  # the statement itself
    # Recurse into compound bodies (If / For / While / With / Try / ...)
    for _field, value in ast.iter_fields(stmt):
        if isinstance(value, list):
            total += sum(_count_item(item) for item in value)
    return total


def count_effective_lines(source_path: Path) -> int:
    """Return the effective line count of the largest *_golden function.

    Raises ValueError if the file cannot be parsed as Python, or LookupError
    if it does not contain any ``*_golden`` function. The caller (main) is
    responsible for translating these into a SystemExit when invoked as a
    script.
    """
    src = source_path.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        raise ValueError(f"cannot parse {source_path}: {exc}") from exc

    counts: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.endswith("_golden"):
            counts.append(sum(_count_stmt(s) for s in node.body))

    if not counts:
        raise LookupError(f"no *_golden function found in {source_path}")
    return max(counts)


def _configure_stdout_logger() -> None:
    """Send INFO records to stdout with a bare format, so the script's output
    can still be captured by shell pipes (e.g. ``LINES=$(... count_golden_lines.py op.py)``)."""
    if _LOGGER.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _LOGGER.addHandler(handler)
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python count_golden_lines.py <path/to/op_golden.py>")
    path = Path(sys.argv[1])
    if not path.is_file():
        raise SystemExit(f"error: file not found: {path}")
    try:
        effective = count_effective_lines(path)
    except (ValueError, LookupError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    _configure_stdout_logger()
    _LOGGER.info("%d", effective)


if __name__ == "__main__":
    main()
