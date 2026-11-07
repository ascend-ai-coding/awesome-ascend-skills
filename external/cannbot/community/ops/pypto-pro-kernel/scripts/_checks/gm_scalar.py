# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Bounded advisory for dense per-worker scalar writes to GM Tensor outputs."""

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

RULE = "gm-scalar-dense-output"


def _argument(call, name, position):
    return next(
        (kw.value for kw in call.keywords if kw.arg == name),
        call.args[position] if len(call.args) > position else None,
    )


def check(path: Path):
    source = read(path)
    if not source.pl:
        return []
    findings = []

    def native_call(node, name):
        return (
            isinstance(node, ast.Call)
            and attribute_path(node.func) == f"{source.pl}.{name}"
        )

    for function in ast.walk(source.tree):
        if not has_decorator(function, source.pl, "jit"):
            continue
        arguments = (
            function.args.posonlyargs + function.args.args + function.args.kwonlyargs
        )
        if source.pl in {arg.arg for arg in arguments}:
            continue
        tensors = set()
        for arg in arguments:
            annotation = arg.annotation
            if not (
                isinstance(annotation, ast.Subscript)
                and attribute_path(annotation.value) == f"{source.pl}.Tensor"
                and isinstance(annotation.slice, ast.Tuple)
                and annotation.slice.elts
            ):
                continue
            shape = annotation.slice.elts[0]
            if isinstance(shape, (ast.List, ast.Tuple)) and len(shape.elts) == 1:
                tensors.add(arg.arg)
        rebound = {
            node.id
            for statement in function.body
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        if source.pl in rebound:
            continue
        tensors -= rebound
        found = []

        class Visitor(ast.NodeVisitor):
            def __init__(self):
                self.owners = set()

            def visit_For(self, node):
                previous = self.owners.copy()
                if isinstance(node.target, ast.Name):
                    self.owners.discard(node.target.id)
                    iterator = node.iter
                    if (
                        native_call(iterator, "range")
                        and len(iterator.args) == 3
                        and native_call(iterator.args[0], "get_block_idx")
                        and native_call(iterator.args[2], "get_block_num")
                        and not any(
                            isinstance(child, ast.Name)
                            and child.id == node.target.id
                            and isinstance(child.ctx, ast.Store)
                            for statement in node.body
                            for child in ast.walk(statement)
                        )
                    ):
                        self.owners.add(node.target.id)
                self.generic_visit(node)
                self.owners = previous

            def visit_Call(self, node):
                if native_call(node, "setval"):
                    target, offset = (
                        _argument(node, "container", 0),
                        _argument(node, "offset", 1),
                    )
                    if (
                        isinstance(target, ast.Name)
                        and target.id in tensors
                        and (
                            native_call(offset, "get_block_idx")
                            or isinstance(offset, ast.Name)
                            and offset.id in self.owners
                        )
                    ):
                        found.append(
                            Finding(
                                source.path,
                                node.lineno,
                                RULE,
                                "GM Tensor 的相邻结果由 block 索引或按行轮转分工直接 pl.setval 写回；"
                                "实际多 worker 的标量路径可能互相覆盖，不能由逻辑索引不同推断物理写回独立。"
                                "核对执行核数、dtype 和地址条件；紧密小结果优先在 UB 形成后 pl.store。"
                                "适用边界见 references/patterns/vf-small-results.md；这不是对 DMA 写回的限制。",
                                advisory=True,
                            )
                        )
                self.generic_visit(node)

            def visit_FunctionDef(self, node):
                pass

            visit_AsyncFunctionDef = visit_FunctionDef

        visitor = Visitor()
        for statement in function.body:
            visitor.visit(statement)
        findings += apply_claim(
            source, function, RULE, found, counterexample_claims(function)
        )
    return findings
