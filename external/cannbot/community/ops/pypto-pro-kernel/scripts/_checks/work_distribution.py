# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Advise on repeated worker-cyclic loops; do not infer runtime load balance.

Recognize direct block IDs and singly assigned local aliases inside one JIT
function. Only nested native ranges starting at worker or base + worker are
considered. Helpers, nonlinear mappings and multiply assigned aliases are
outside scope. A retained group owner can be an intentional reuse tradeoff.
"""

import ast
from collections import defaultdict
from pathlib import Path

from .source import Finding, attribute_path, has_decorator, read

RULE = "nested-worker-restart"


def _nodes(node):
    for child in ast.iter_child_nodes(node):
        if isinstance(
            child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
        ):
            continue
        yield child
        yield from _nodes(child)


def check(path: Path) -> list[Finding]:
    source = read(path)
    findings = []
    for fn in ast.walk(source.tree):
        if not has_decorator(fn, source.pl, "jit"):
            continue
        nodes = list(_nodes(fn))
        parameters = {
            a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
        }
        parameters.update(
            a.arg for a in (fn.args.vararg, fn.args.kwarg) if a is not None
        )
        if source.pl in parameters or any(
            isinstance(n, ast.Name)
            and isinstance(n.ctx, ast.Store)
            and n.id == source.pl
            for n in nodes
        ):
            continue
        bindings = defaultdict(list)
        for node in nodes:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    for name in ast.walk(target):
                        if isinstance(name, ast.Name):
                            bindings[name.id].append(
                                node.value if isinstance(target, ast.Name) else None
                            )
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and isinstance(
                node.target, ast.Name
            ):
                bindings[node.target.id].append(
                    node.value if isinstance(node, ast.AnnAssign) else None
                )
            elif isinstance(node, ast.For):
                for name in ast.walk(node.target):
                    if isinstance(name, ast.Name):
                        bindings[name.id].append(None)

        def resolve(node, seen=frozenset()):
            if (
                isinstance(node, ast.Name)
                and node.id not in seen
                and node.id not in parameters
            ):
                values = bindings.get(node.id, [])
                if len(values) == 1 and values[0] is not None:
                    return resolve(values[0], seen | {node.id})
            return node

        def is_worker(node):
            value = resolve(node)
            return (
                isinstance(value, ast.Call)
                and not value.args
                and not value.keywords
                and attribute_path(value.func) == f"{source.pl}.get_block_idx"
            )

        def restarted_start(node):
            value = resolve(node)
            return is_worker(value) or (
                isinstance(value, ast.BinOp)
                and isinstance(value.op, ast.Add)
                and (is_worker(value.left) or is_worker(value.right))
            )

        class Visitor(ast.NodeVisitor):
            def __init__(self):
                self.ranges = []

            def visit_FunctionDef(self, node):
                pass

            visit_AsyncFunctionDef = visit_FunctionDef
            visit_Lambda = visit_FunctionDef
            visit_ClassDef = visit_FunctionDef

            def visit_For(self, node):
                call = node.iter
                native = (
                    isinstance(call, ast.Call)
                    and attribute_path(call.func) == f"{source.pl}.range"
                )
                if (
                    native
                    and self.ranges
                    and len(call.args) == 3
                    and restarted_start(call.args[0])
                ):
                    findings.append(
                        Finding(
                            source.path,
                            node.lineno,
                            RULE,
                            f"嵌套工作循环每次外层迭代（第 {self.ranges[-1]} 行）都从同一个 block ID 开始分配。"
                            "按实际分组长度和 M/N tile 数统计每核工作量，并对照跨组连续编号；"
                            "检查本核无任务时的预载与槽位复用。保留组内复用可能合理，"
                            "此提示不证明负载不均，也不要求改成全局分配。",
                            advisory=True,
                        )
                    )
                if native:
                    self.ranges.append(node.lineno)
                self.generic_visit(node)
                if native:
                    self.ranges.pop()

        visitor = Visitor()
        for statement in fn.body:
            visitor.visit(statement)
    return findings
