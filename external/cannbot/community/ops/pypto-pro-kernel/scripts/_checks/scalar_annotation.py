# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Reject proven dtype aliases used as complete JIT scalar annotations."""

import ast
from collections import Counter
from pathlib import Path

from .register_dtype import DTYPES
from .source import (
    Finding,
    apply_claim,
    attribute_path,
    counterexample_claims,
    has_decorator,
    read,
)

RULE = "jit-scalar-dtype-alias"


def check(path: Path):
    source = read(path)
    if not source.pl:
        return []
    # Any additional binding anywhere in the file keeps the name unknown.
    bindings = Counter()
    for node in ast.walk(source.tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bindings[node.id] += 1
        elif isinstance(node, ast.arg):
            bindings[node.arg] += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bindings[node.name] += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bindings[alias.asname or alias.name.split(".")[0]] += 1
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bindings[node.name] += 1
    if bindings[source.pl.split(".")[0]] != 1:
        return []
    aliases = {}
    supported = {f"{source.pl}.{name}" for name in DTYPES | {"DT_BOOL"}}
    for statement in source.tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target, value = statement.targets[0], statement.value
            if isinstance(target, ast.Name) and bindings[target.id] == 1:
                dtype = attribute_path(value)
                if dtype in supported:
                    aliases[target.id] = dtype
    findings = []
    for function in ast.walk(source.tree):
        if not has_decorator(function, source.pl, "jit"):
            continue
        found = []
        arguments = (
            function.args.posonlyargs + function.args.args + function.args.kwonlyargs
        )
        for argument in arguments:
            annotation = argument.annotation
            if isinstance(annotation, ast.Name) and annotation.id in aliases:
                found.append(
                    Finding(
                        source.path,
                        annotation.lineno,
                        RULE,
                        f"参数 {argument.arg} 的完整注解 {annotation.id} 是 dtype 别名；"
                        f"请写 {aliases[annotation.id]}。行为版本 872df763f 的标量注解解析不展开该别名；"
                        "Tensor/Ptr 内的 dtype 位置是另一条解析路径，那里可使用 dtype 别名。",
                    )
                )
        findings += apply_claim(
            source, function, RULE, found, counterexample_claims(function)
        )
    return findings
