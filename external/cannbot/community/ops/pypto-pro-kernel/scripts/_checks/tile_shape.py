# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check literal TileType rank without importing the framework.

The installed TileType contract requires two physical dimensions, independently
of GM tensor rank. Only literal list/tuple arity is known here; element values,
star expansion, named shapes, alignment and capacity remain unchecked.
"""

from __future__ import annotations

import ast
from pathlib import Path

from .source import Finding, apply_claim, attribute_path, counterexample_claims, read

RULE = "tile-shape-must-be-2d"


def _literal_shape(node):
    shape = next(
        (kw.value for kw in node.keywords if kw.arg == "shape"),
        node.args[0] if node.args else None,
    )
    if isinstance(shape, (ast.List, ast.Tuple)) and not any(
        isinstance(e, ast.Starred) for e in shape.elts
    ):
        return shape
    return None


def scope_counts(path: Path) -> tuple[int, int]:
    source = read(path)
    nodes = [
        node
        for node in ast.walk(source.tree)
        if isinstance(node, ast.Call)
        and source.pl is not None
        and attribute_path(node.func) == f"{source.pl}.TileType"
    ]
    checked = sum(_literal_shape(node) is not None for node in nodes)
    return checked, len(nodes) - checked


def check(path: Path) -> list[Finding]:
    source = read(path)
    if source.pl is None:
        return []
    by_function: dict[ast.AST, list[Finding]] = {}
    top_level: list[Finding] = []

    class Visitor(ast.NodeVisitor):
        owner: ast.AST | None = None

        def visit_FunctionDef(self, node):
            previous = self.owner
            self.owner = node
            by_function[node] = []
            self.generic_visit(node)
            self.owner = previous

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node):
            if attribute_path(node.func) == f"{source.pl}.TileType":
                shape = _literal_shape(node)
                if shape is not None and len(shape.elts) != 2:
                    finding = Finding(
                        source.path,
                        node.lineno,
                        RULE,
                        f"TileType.shape 有 {len(shape.elts)} 维，必须为二维；"
                        "一维 GM 输出也使用物理对齐的 [1, padded_cols] tile，"
                        "再设置二维有效区。元素值和对齐由框架继续检查",
                    )
                    if self.owner is None:
                        top_level.append(finding)
                    else:
                        by_function[self.owner].append(finding)
            self.generic_visit(node)

    Visitor().visit(source.tree)
    findings = top_level
    for function, found in by_function.items():
        findings += apply_claim(
            source, function, RULE, found, counterexample_claims(function)
        )
    return findings
