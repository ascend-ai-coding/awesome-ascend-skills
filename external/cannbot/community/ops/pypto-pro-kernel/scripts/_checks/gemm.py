# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Keep tile layout attributes off native GEMM sugar calls; no framework imports."""

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

RULE = "gemm-compact-is-tile-option"


def check(path: Path):
    source = read(path)
    if not source.pl:
        return []
    findings = []
    for function in ast.walk(source.tree):
        if not has_decorator(function, source.pl, "jit"):
            continue
        arguments = (
            function.args.posonlyargs + function.args.args + function.args.kwonlyargs
        )
        bound = {arg.arg for arg in arguments}
        bound.update(
            node.id
            for node in ast.walk(function)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        )
        if source.pl in bound:
            continue
        found = []

        class Visitor(ast.NodeVisitor):
            def visit_Call(self, node):
                name = attribute_path(node.func)
                if name in {f"{source.pl}.gemm", f"{source.pl}.gemm_mx"}:
                    for keyword in node.keywords:
                        if keyword.arg == "compact":
                            found.append(
                                Finding(
                                    source.path,
                                    keyword.value.lineno,
                                    RULE,
                                    f"{name} 不接收 compact 参数；compact=1 属于被收窄的 Tile/TileGroup 声明。"
                                    "在声明处设置物理布局，在 GEMM 调用处用 m/n/k 表达有效范围；"
                                    "规则依据行为版本 872df763f 的语法糖签名。",
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
    return findings
