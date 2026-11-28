#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Estimate development complexity from a Python golden without executing it.

Counts lexical statements, loops and operation call sites in the selected
function. Names are syntax hints: einsum is reported separately because it is
not necessarily a matrix product. Helpers are not expanded. The module count
is a starting point for design, not a correctness check or a kernel count.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

_LOGGER = logging.getLogger(__name__)


class Signals(ast.NodeVisitor):
    def __init__(self):
        self.counts = dict(
            statements=0, matmul_calls=0, einsum_calls=0, reduction_calls=0, loops=0
        )

    def generic_visit(self, node):
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            return  # Nested definitions are not part of the selected function body.
        if isinstance(node, ast.stmt):
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                return
            self.counts["statements"] += 1
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While, ast.comprehension)):
            self.counts["loops"] += 1
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            self.counts["matmul_calls"] += 1
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if name in {"matmul", "mm", "bmm"}:
                self.counts["matmul_calls"] += 1
            elif name == "einsum":
                self.counts["einsum_calls"] += 1
            elif name in {
                "sum",
                "mean",
                "amax",
                "amin",
                "prod",
                "softmax",
                "logsumexp",
            }:
                self.counts["reduction_calls"] += 1
        super().generic_visit(node)


def estimate(source: str, function: str | None = None) -> dict:
    tree = ast.parse(source)
    candidates = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if function is None and not node.name.endswith("_golden"):
            continue
        if function is not None and node.name != function:
            continue
        candidates.append(node)
    if len(candidates) != 1:
        names = ", ".join(n.name for n in candidates) or "none"
        raise ValueError(f"select one golden with --function; candidates: {names}")
    node = candidates[0]
    signals = Signals()
    for stmt in node.body:
        signals.visit(stmt)
    s = signals.counts
    # Heuristic work units: ~30 statements or ~3 operation sites. Loops are
    # reported for review, not treated as proof of loop-carried state.
    operations = (s["matmul_calls"] + s["einsum_calls"] + s["reduction_calls"]) / 3
    score = max(min(s["statements"] / 30, max(1, operations) + 1), operations)
    suggested = (
        1
        if score < 1.3
        else max(1, min(math.floor(score + 0.5), math.ceil(s["statements"] / 12)))
    )
    return {
        "function": node.name,
        "signals": s,
        "suggested_module_count": suggested,
        "limitations": "Lexical counts only; review helper calls, aliases and state dependencies.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("golden", type=Path)
    parser.add_argument("--function", help="top-level function name when ambiguous")
    args = parser.parse_args()
    try:
        result = estimate(args.golden.read_text(encoding="utf-8"), args.function)
    except (OSError, SyntaxError, ValueError) as exc:
        _LOGGER.error("error: %s", exc)
        return 1
    _LOGGER.info(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
