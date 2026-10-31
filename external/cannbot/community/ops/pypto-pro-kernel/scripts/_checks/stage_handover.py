# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Find directly visible stage groups whose cross-core handover needs review.

This is a source advisory, not a scheduler or a read/write analysis. It follows
direct group declarations and named stage arguments in the same lexical scope.
An unmarked group may intentionally share another group's events; only the
generated program and actual pipe/lifetime proof can establish that ordering.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

from .source import Finding, attribute_path, has_decorator, read

RULE = "stage-handover-unverified"


def body_nodes(node):
    """Walk one function's statements, excluding nested function/class bodies."""
    for child in ast.iter_child_nodes(node):
        if isinstance(
            child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            continue
        yield child
        yield from body_nodes(child)


def stages_in(scope, pl):
    result = {}
    for fn in getattr(scope, "body", []):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for decorator in fn.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if attribute_path(target) == f"{pl}.pipeline.stage":
                result[fn.name] = fn
    return result


def check(path: Path) -> list[Finding]:
    source = read(path)
    if not source.pl:
        return []
    parents = {
        child: parent
        for parent in ast.walk(source.tree)
        for child in ast.iter_child_nodes(parent)
    }
    findings = []
    for kernel in ast.walk(source.tree):
        if not has_decorator(kernel, source.pl, "jit"):
            continue
        scopes = []
        scope = parents.get(kernel)
        while scope is not None:
            if isinstance(scope, (ast.Module, ast.FunctionDef)):
                scopes.append(scope)
            scope = parents.get(scope)
        stages = {}
        for scope in reversed(scopes):
            stages.update(stages_in(scope, source.pl))
        stages.update(stages_in(kernel, source.pl))
        groups = {}
        nodes = list(body_nodes(kernel))
        for node in nodes:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target, call = node.targets[0], node.value
            if not isinstance(target, ast.Name) or not isinstance(call, ast.Call):
                continue
            if attribute_path(call.func) != f"{source.pl}.make_tile_group":
                continue
            events = {
                k.arg
                for k in call.keywords
                if k.arg in ("fwd_ids", "bwd_ids")
                and not (isinstance(k.value, ast.Constant) and k.value.value is None)
                and not (
                    isinstance(k.value, (ast.List, ast.Tuple)) and not k.value.elts
                )
            }
            marked = events == {"fwd_ids", "bwd_ids"}
            if not marked:
                groups[target.id] = node.lineno
        users = defaultdict(set)
        for section in nodes:
            if not isinstance(section, ast.With):
                continue
            kinds = {
                attribute_path(item.context_expr.func)
                for item in section.items
                if isinstance(item.context_expr, ast.Call)
            }
            kind = next(
                (k for k in ("cube", "vector") if f"{source.pl}.section_{k}" in kinds),
                None,
            )
            if kind is None:
                continue
            for node in body_nodes(section):
                if not isinstance(node, ast.Call) or not isinstance(
                    node.func, ast.Name
                ):
                    continue
                stage = stages.get(node.func.id)
                if stage is None:
                    continue
                parameters = stage.args.posonlyargs + stage.args.args
                arguments = dict(zip((p.arg for p in parameters), node.args))
                arguments.update((k.arg, k.value) for k in node.keywords if k.arg)
                used = {
                    n.id
                    for n in body_nodes(stage)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                }
                for parameter, argument in arguments.items():
                    if (
                        parameter in used
                        and isinstance(argument, ast.Name)
                        and argument.id in groups
                    ):
                        users[argument.id].add(kind)
        for name, kinds in users.items():
            if kinds == {"cube", "vector"}:
                findings.append(
                    Finding(
                        source.path,
                        groups[name],
                        RULE,
                        f"{name} 被 Cube/Vector stage 的参数实际引用，但未显式声明 fwd_ids / bwd_ids。"
                        "核对读写角色及生成的 set/wait、实际管线与复用时机；auto_mutex 不能独自建立跨核交接。"
                        "若有意共用另一组事件或手写同步，应证明完整的前向/反向依赖；本提示不判定数据竞争。",
                        advisory=True,
                    )
                )
    return findings
