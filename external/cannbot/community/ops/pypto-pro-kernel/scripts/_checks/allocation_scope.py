# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Reject unavailable native allocation APIs without importing the source."""

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

RULE = "unsupported-allocation-api"


def inspect(path: Path):
    source = read(path)
    if not source.pl:
        return [], (0, 0)
    allocations = {
        f"{source.pl}.{name}" for name in ("mem_space", "allocate_tile_group")
    }
    calls = {
        id(node)
        for node in ast.walk(source.tree)
        if isinstance(node, ast.Call) and attribute_path(node.func) in allocations
    }
    checked = set()
    findings = []
    # Module/closure rebinding can change the meaning of the imported namespace.
    rebound = any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Store)
        and node.id == source.pl
        for node in ast.walk(source.tree)
    )
    for function in ast.walk(source.tree):
        if rebound or not has_decorator(function, source.pl, "jit"):
            continue
        arguments = (
            function.args.posonlyargs + function.args.args + function.args.kwonlyargs
        )
        arguments += [arg for arg in (function.args.vararg, function.args.kwarg) if arg]
        if any(arg.arg == source.pl for arg in arguments):
            continue
        found = []

        class Visitor(ast.NodeVisitor):
            def visit_Call(self, node):
                if attribute_path(node.func) in allocations:
                    checked.add(id(node))
                    found.append(
                        Finding(
                            source.path,
                            node.lineno,
                            RULE,
                            "当前框架不支持 mem_space / allocate_tile_group；使用原生 "
                            'make_tile_group(type=..., addrs=..., mutex_ids="auto", depth=N)，'
                            "并显式核对各槽位的地址、容量和跨核事件。",
                        )
                    )
                self.generic_visit(node)

            def visit_FunctionDef(self, node):
                pass

            visit_AsyncFunctionDef = visit_FunctionDef
            visit_ClassDef = visit_FunctionDef
            visit_Lambda = visit_FunctionDef

        visitor = Visitor()
        for statement in function.body:
            visitor.visit(statement)
        findings += apply_claim(
            source, function, RULE, found, counterexample_claims(function)
        )
    return findings, (len(checked), len(calls - checked))


def check(path: Path):
    return inspect(path)[0]


def scope_counts(path: Path):
    return inspect(path)[1]
