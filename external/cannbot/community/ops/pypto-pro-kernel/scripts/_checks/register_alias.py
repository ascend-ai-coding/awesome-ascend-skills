# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Advisory for mutable VF register aliases with later sibling reads.

This is a source-level audit, not an error rule: intentional sharing is legal.
Track direct producers and aliases in straight-line/loop bodies and individual
if arms, skipping branch joins and helper returns. No PyPTO execution or type inference.
"""

import ast
from pathlib import Path

from .source import Finding, attribute_path, read, vector_functions

RULE = "vf-shared-register-write"
# Known register-producing instructions only; unknown APIs remain unclassified.
PRODUCERS = {
    "full",
    "load_align",
    "move",
    "add",
    "adds",
    "sub",
    "mul",
    "muls",
    "div",
    "max",
    "min",
    "select",
    "gather",
    "astype",
    "arange",
    "reduce_max",
    "reduce_min",
    "reduce_sum",
}
# Older Python can check ordinary kernels without having newer AST node classes.
UNTRACKED_BLOCKS = tuple(
    getattr(ast, name) for name in ("Try", "With", "Match") if hasattr(ast, name)
)


def check(path: Path) -> list[Finding]:
    source = read(path)
    if not source.vf:
        return []
    findings = []
    for function in vector_functions(source):
        bindings = {}
        seen = set()
        reads = {}

        class Reads(ast.NodeVisitor):
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Load):
                    reads.setdefault(node.id, []).append(node.lineno)

            def visit_FunctionDef(self, node):
                pass

            visit_AsyncFunctionDef = visit_FunctionDef

        collector = Reads()
        for statement in function.body:
            collector.visit(statement)

        def producer(value):
            name = attribute_path(value.func) if isinstance(value, ast.Call) else None
            return name in {f"{source.vf}.{op}" for op in PRODUCERS}

        def walk(statements):
            for stmt in statements:
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                ):
                    target, value = stmt.targets[0].id, stmt.value
                    if producer(value):
                        identity = bindings.setdefault(target, object())
                        siblings = sorted(
                            name
                            for name, token in bindings.items()
                            if name != target
                            and token is identity
                            and any(
                                line > stmt.end_lineno for line in reads.get(name, [])
                            )
                        )
                        if siblings and identity not in seen:
                            seen.add(identity)
                            findings.append(
                                Finding(
                                    source.path,
                                    stmt.lineno,
                                    RULE,
                                    f"VF 写入 `{target}` 会同时改变当前别名 {', '.join(siblings)}，且后面仍有这些名称的读取；"
                                    "确认是否有意共享。独立状态应在首次绑定时分别用 vf.full 或新名称的全掩码 vf.move 创建，"
                                    "普通 b = a 不是副本。此项仅提示核查，不判定算法错误",
                                    advisory=True,
                                )
                            )
                    elif isinstance(value, ast.Name) and value.id in bindings:
                        bindings[target] = bindings[value.id]
                    else:
                        bindings.pop(target, None)
                elif isinstance(stmt, (ast.For, ast.While)):
                    # VF writes reuse identity; ordinary bindings across an
                    # iteration may differ, so invalidate those before entry.
                    uncertain = {
                        node.targets[0].id
                        for node in ast.walk(stmt)
                        if isinstance(node, ast.Assign)
                        and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name)
                        and not producer(node.value)
                    }
                    for name in uncertain:
                        bindings.pop(name, None)
                    if isinstance(stmt, ast.For):
                        for node in ast.walk(stmt.target):
                            if isinstance(node, ast.Name):
                                bindings.pop(node.id, None)
                    walk(stmt.body)
                    for name in uncertain:
                        bindings.pop(name, None)
                elif isinstance(stmt, ast.If):
                    # Inspect each arm with its own incoming bindings, without
                    # propagating an arm's aliases into its sibling or the join.
                    incoming = bindings.copy()
                    for arm in (stmt.body, stmt.orelse):
                        bindings.clear()
                        bindings.update(incoming)
                        walk(arm)
                    bindings.clear()
                    bindings.update(incoming)
                    for node in ast.walk(stmt):
                        if isinstance(node, ast.Name) and isinstance(
                            node.ctx, ast.Store
                        ):
                            bindings.pop(node.id, None)
                elif isinstance(stmt, UNTRACKED_BLOCKS):
                    # Avoid pretending to resolve branch joins or context effects.
                    for node in ast.walk(stmt):
                        if isinstance(node, ast.Name) and isinstance(
                            node.ctx, ast.Store
                        ):
                            bindings.pop(node.id, None)
                elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    for node in ast.walk(stmt):
                        if isinstance(node, ast.Name) and isinstance(
                            node.ctx, ast.Store
                        ):
                            bindings.pop(node.id, None)

        walk(function.body)
    return findings
