# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Catch setters on directly traceable fixed-valid-shape tiles without PyPTO.

Track local constructors, aliases and group slots in jit functions. Helper returns,
parameters, nonliteral valid shapes and bindings changed by control flow stay unknown.
"""

import ast
from pathlib import Path

from .source import (
    Finding,
    apply_claim,
    attribute_path,
    counterexample_claims,
    has_decorator,
    read,
)

RULE = "fixed-validshape-mutation"


def _argument(call, name, position=0):
    return next(
        (kw.value for kw in call.keywords if kw.arg == name),
        call.args[position] if len(call.args) > position else None,
    )


def _fixed_dimension(node):
    if not isinstance(node, (ast.List, ast.Tuple)) or len(node.elts) != 2:
        return False
    for element in node.elts:
        try:
            value = ast.literal_eval(element)
        except (ValueError, TypeError):
            continue
        if type(value) is int and value >= 0:
            return True
    return False


def check(path: Path) -> list[Finding]:
    source = read(path)
    if source.pl is None:
        return []
    findings = []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.fixed = set()
            self.found = []

        def resolve(self, node):
            if isinstance(node, ast.Name):
                return node.id in self.fixed
            if isinstance(node, ast.Subscript):
                return self.resolve(node.value)
            if isinstance(node, ast.Call):
                name = attribute_path(node.func)
                if name == f"{source.pl}.TileType":
                    valid = next(
                        (kw.value for kw in node.keywords if kw.arg == "valid_shape"),
                        None,
                    )
                    return _fixed_dimension(valid)
                if name in {f"{source.pl}.make_tile", f"{source.pl}.make_tile_group"}:
                    return self.resolve(_argument(node, "type"))
                if isinstance(node.func, ast.Attribute) and node.func.attr in {
                    "next",
                    "current",
                    "get",
                }:
                    return self.resolve(node.func.value)
            return False

        def visit_Assign(self, node):
            self.visit(node.value)
            fixed = self.resolve(node.value)
            for target in node.targets:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store):
                        self.fixed.discard(name.id)
                if isinstance(target, ast.Name) and fixed:
                    self.fixed.add(target.id)

        def visit_AnnAssign(self, node):
            if node.value is not None:
                self.visit_Assign(ast.Assign(targets=[node.target], value=node.value))

        def visit_AugAssign(self, node):
            self.generic_visit(node)
            if isinstance(node.target, ast.Name):
                self.fixed.discard(node.target.id)

        def visit_Call(self, node):
            if attribute_path(
                node.func
            ) == f"{source.pl}.set_validshape" and self.resolve(
                _argument(node, "tile")
            ):
                self.found.append(
                    Finding(
                        source.path,
                        node.lineno,
                        RULE,
                        "此 tile 的 valid_shape 含固定维度，不能调用 set_validshape；"
                        "保留固定有效区并删除多余 setter，或需要运行期变更时改为 "
                        "[-1, -1]/省略 valid_shape。SDK 要求两个维度均动态",
                    )
                )
            self.generic_visit(node)

        def visit_control(self, node):
            assigned = {
                child.id
                for child in ast.walk(node)
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
            }
            stable = self.fixed - assigned
            for field in ("test", "iter"):
                value = getattr(node, field, None)
                if value is not None:
                    self.visit(value)
            for block in (node.body, node.orelse):
                self.fixed = stable.copy()
                for statement in block:
                    self.visit(statement)
            self.fixed = stable

        visit_If = visit_For = visit_While = visit_control

        def visit_FunctionDef(self, node):
            pass  # Nested kernels are checked in their own local scope below.

        visit_AsyncFunctionDef = visit_FunctionDef

    for function in ast.walk(source.tree):
        if not has_decorator(function, source.pl, "jit"):
            continue
        visitor = Visitor()
        for statement in function.body:
            visitor.visit(statement)
        findings += apply_claim(
            source, function, RULE, visitor.found, counterexample_claims(function)
        )
    return findings
