# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check scalar annotations and review version-bound postponed entry dispatch."""

import ast
import hashlib
from functools import lru_cache
from collections import Counter
from pathlib import Path

from .paths import PYPTO_ROOT
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
ENTRY_RULE = "jit-postponed-entry"
# The complete recognizer, not a substring shared by unrelated dispatch paths.
ENTRY_GUARD_SHA256 = "13e19e83667c3f7171b915363f46ad619a591b4d102485a9187d01e109c19078"


@lru_cache(maxsize=4)
def entry_guard(payload):
    try:
        text = payload.decode("utf-8")
        tree = ast.parse(text)
    except (UnicodeError, SyntaxError):
        return False
    recognizers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_is_tile_kernel"
    ]
    return (
        len(recognizers) == 1
        and hashlib.sha256(
            ast.get_source_segment(text, recognizers[0]).encode()
        ).hexdigest()
        == ENTRY_GUARD_SHA256
    )


def postponed_entry(source, function, arguments, pypto_root):
    future = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in source.tree.body
    )
    strings, tensor_like = [], False
    for argument in arguments:
        annotation = argument.annotation
        if annotation is None:
            continue
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            text = annotation.value
        elif future:
            text = ast.unparse(annotation)
        else:
            # Another eager annotation can select native dispatch on its own.
            return []
        strings.append(text)
        try:
            expression = ast.parse(text, mode="eval").body
        except SyntaxError:
            continue
        if isinstance(expression, ast.Subscript):
            tensor_like |= attribute_path(expression.value) in {
                f"{source.pl}.Tensor",
                f"{source.pl}.Ptr",
            }
    if not tensor_like or any(
        text.split(".")[-1].startswith("DT_") for text in strings
    ):
        return []
    selected = Path(pypto_root) / "python/pypto_pro/runtime/jit.py"
    try:
        verified = entry_guard(selected.read_bytes())
    except OSError:
        verified = False
    if not verified:
        return []
    return [
        Finding(
            source.path,
            function.lineno,
            ENTRY_RULE,
            f"`{function.name}` 的参数注解均延迟为字符串；所选 JIT 识别器不会把这些 Tensor/Ptr 拼写选为 native 入口 "
            f"(recognizer SHA256 {ENTRY_GUARD_SHA256})。先检查导入后入口的实际类型与注解，"
            "再做 native capture；可选用 eager Tensor 注解或显式 pl.DT_* dtype 拼写。"
            "这不是对 future annotations 或 Tensor 内 dtype 别名的一般禁令；包装器或运行时注解改写需另行验证。",
            advisory=True,
        )
    ]


def check(path: Path, *, pypto_root=PYPTO_ROOT):
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
        findings += postponed_entry(source, function, arguments, pypto_root)
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
