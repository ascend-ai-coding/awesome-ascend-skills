# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Reject a visibly unseeded destination of the two VF fused accumulator ops.

This checks value bindings, not register types or predicate coverage. Parameters,
helper results and possible branch/loop bindings remain unknown rather than
being rejected. It does not import PyPTO or prove initialization on every path.
"""

from __future__ import annotations

import ast
from pathlib import Path

from .source import (
    Finding,
    apply_claim,
    attribute_path,
    counterexample_claims,
    read,
    vector_functions,
)

RULE = "vf-dst-read-before-write"
FUSED_OPS = {"mul_dst_add", "mul_add_dst"}


def _nodes(node):
    yield node
    if isinstance(
        node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
    ):
        return
    for child in ast.iter_child_nodes(node):
        yield from _nodes(child)


def _target_names(target):
    return {
        node.id
        for node in ast.walk(target)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }


def _assignment(node):
    if isinstance(node, ast.Assign):
        return node.targets, node.value
    if isinstance(node, ast.AnnAssign) and node.value is not None:
        return [node.target], node.value
    return [], None


def inspect(path: Path) -> tuple[list[Finding], tuple[int, int]]:
    source = read(path)
    if not source.vf:
        return [], (0, 0)
    fused = {f"{source.vf}.{name}" for name in FUSED_OPS}
    findings = []
    checked, unverified = set(), set()

    def is_fused(value):
        return isinstance(value, ast.Call) and attribute_path(value.func) in fused

    def loop_bindings(loop):
        # A seed in another iteration/branch is possible, but not proved here.
        bound = set()
        for node in _nodes(loop):
            targets, value = _assignment(node)
            if targets and not is_fused(value):
                for target in targets:
                    bound.update(_target_names(target))
        return bound

    def walk(statements, bound, found):
        bound = bound.copy()
        for stmt in statements:
            targets, value = _assignment(stmt)
            if targets:
                if (
                    len(targets) == 1
                    and isinstance(targets[0], ast.Name)
                    and is_fused(value)
                ):
                    checked.add(id(value))
                    target = targets[0].id
                    if target not in bound:
                        found.append(
                            Finding(
                                source.path,
                                stmt.lineno,
                                RULE,
                                f"`{target}` 首次赋值调用 `{attribute_path(value.func)}`，但该操作还读取目标寄存器的旧值；"
                                "先用同一个目标名称 load/full/move 所需初值，再原位赋值。载入另一名称不会初始化此目标",
                            )
                        )
                for target in targets:
                    bound.update(_target_names(target))
            elif isinstance(stmt, ast.If):
                bound |= walk(stmt.body, bound, found) | walk(stmt.orelse, bound, found)
            elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
                incoming = bound | loop_bindings(stmt)
                if isinstance(stmt, (ast.For, ast.AsyncFor)):
                    incoming |= _target_names(stmt.target)
                bound |= walk(stmt.body, incoming, found)
                bound |= walk(stmt.orelse, bound, found)
            else:
                # Complex blocks are not interpreted; their writes stay unknown.
                bound |= {
                    node.id
                    for node in _nodes(stmt)
                    if isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Store)
                    and not isinstance(stmt, ast.AnnAssign)
                }
        return bound

    for function in vector_functions(source):
        candidates = {
            id(node)
            for stmt in function.body
            for node in _nodes(stmt)
            if is_fused(node)
        }
        args = function.args
        bound = {arg.arg for arg in args.posonlyargs + args.args + args.kwonlyargs}
        bound |= {arg.arg for arg in (args.vararg, args.kwarg) if arg is not None}
        if source.vf in bound or any(
            isinstance(node, ast.Name)
            and node.id == source.vf
            and isinstance(node.ctx, ast.Store)
            for stmt in function.body
            for node in _nodes(stmt)
        ):
            unverified |= candidates
            continue
        found = []
        walk(function.body, bound, found)
        unverified |= candidates - checked
        findings += apply_claim(
            source, function, RULE, found, counterexample_claims(function)
        )
    return findings, (len(checked), len(unverified))


def check(path: Path) -> list[Finding]:
    return inspect(path)[0]


def scope_counts(path: Path) -> tuple[int, int]:
    return inspect(path)[1]
