# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Flag possible runtime control inside SIMD loops without predicting codegen.

This is a performance advisory. Parameters, loop indices and their local scalar
expressions may be runtime values; factory captures and literals are left alone.
Call-site specialization and compiler loop recognition remain native checks.
"""

import ast
from pathlib import Path

from .source import Finding, read, vector_functions
from .vf_scope import _simd_vf

RULE = "vf-loop-runtime-control"


def _body_nodes(node):
    for child in ast.iter_child_nodes(node):
        if isinstance(
            child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
        ):
            continue
        yield child
        yield from _body_nodes(child)


def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def check(path: Path) -> list[Finding]:
    source = read(path)
    findings = []
    for function in vector_functions(source):
        if not _simd_vf(function, source.pl):
            continue
        nodes = list(_body_nodes(function))
        runtime = {
            a.arg
            for a in function.args.posonlyargs
            + function.args.args
            + function.args.kwonlyargs
        }
        for node in nodes:
            if isinstance(node, (ast.For, ast.AsyncFor)):
                runtime.update(_names(node.target))
        # A conservative dependency closure, not a type or path-sensitive analysis.
        assignments = [n for n in nodes if isinstance(n, (ast.Assign, ast.AnnAssign))]
        changed = True
        while changed:
            before = len(runtime)
            for node in assignments:
                if node.value is not None and _names(node.value) & runtime:
                    targets = (
                        node.targets if isinstance(node, ast.Assign) else [node.target]
                    )
                    for target in targets:
                        if isinstance(target, ast.Name):
                            runtime.add(target.id)
            changed = len(runtime) != before

        class Visitor(ast.NodeVisitor):
            loop_depth = 0

            def visit_FunctionDef(self, node):
                pass

            visit_AsyncFunctionDef = visit_FunctionDef
            visit_Lambda = visit_FunctionDef
            visit_ClassDef = visit_FunctionDef

            def visit_For(self, node):
                self.loop_depth += 1
                self.generic_visit(node)
                self.loop_depth -= 1

            visit_AsyncFor = visit_For

            def visit_While(self, node):
                # A conditional loop is itself repeated runtime control.
                self.loop_depth += 1
                self._condition(node)
                self.generic_visit(node)
                self.loop_depth -= 1

            def _condition(self, node):
                dependencies = _names(node.test) & runtime
                if self.loop_depth and dependencies:
                    findings.append(
                        Finding(
                            source.path,
                            node.lineno,
                            RULE,
                            "SIMD 循环中的条件依赖形参、循环索引或其派生值 "
                            f"({', '.join(sorted(dependencies))})，可能阻止硬件循环/地址生成优化。"
                            "核对调用处是否已常量化及生成的 VF 指令；优先把块级判断移到 VF 外并选择静态变体，"
                            "行有效范围可比较循环界限，逐 lane 条件使用 VF 谓词。"
                            "这是性能提示，不表示分支非法、同步缺失或必然变慢。",
                            advisory=True,
                        )
                    )

            def visit_If(self, node):
                self._condition(node)
                self.generic_visit(node)

            visit_IfExp = visit_If

        visitor = Visitor()
        for statement in function.body:
            visitor.visit(statement)
    return findings
