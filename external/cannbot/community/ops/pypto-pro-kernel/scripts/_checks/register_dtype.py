# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check VF destination dtype changes, narrowing stores and scalar self-broadcasts.

Only inspect named results of a small set of register constructors/conversions.
Resolve literal pl.DT_* and single-assignment module aliases, not helper returns,
local/closure dtype bindings, implicit result types or control-flow joins.
Self-broadcasts of parameters or non-VF bindings are advisories because their
source kind may be unknown; established register broadcasts remain legal.
"""

import ast
from collections import Counter
from pathlib import Path

from .source import (
    Finding,
    PL_MODULE,
    apply_claim,
    attribute_path,
    counterexample_claims,
    read,
    vector_functions,
)

RULE = "vf-register-dtype-change"
BROADCAST_RULE = "vf-broadcast-reuses-binding"
NARROW_STORE_RULE = "vf-narrow-store-layout"
TYPED_RESULTS = {"full", "arange", "astype", "bit_cast", "pack", "unpack"}
DATA_ONLY_RESULTS = TYPED_RESULTS - {"pack", "unpack"}
DTYPES = {
    "DT_FP16",
    "DT_BF16",
    "DT_FP32",
    "DT_INT8",
    "DT_UINT8",
    "DT_INT16",
    "DT_UINT16",
    "DT_INT32",
    "DT_UINT32",
    "DT_INT64",
    "DT_UINT64",
}


class Bindings(ast.NodeVisitor):
    """Find lexical bindings, excluding child scopes."""

    def __init__(self):
        self.names = Counter()

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.names[node.id] += 1

    def visit_FunctionDef(self, node):
        self.names[node.name] += 1

    visit_AsyncFunctionDef = visit_ClassDef = visit_FunctionDef

    def visit_Lambda(self, node):
        pass

    def visit_Import(self, node):
        for alias in node.names:
            self.names[alias.asname or alias.name.split(".")[0]] += 1

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.names[alias.asname or alias.name] += 1

    def visit_ExceptHandler(self, node):
        if node.name:
            self.names[node.name] += 1
        self.generic_visit(node)


def bound(statements):
    collector = Bindings()
    for statement in statements:
        collector.visit(statement)
    return collector.names


def _narrow_store_findings(source, function, dtype, blocked):
    """Recognize a direct ZERO half-width cast followed by an ordinary store.

    The predicate dtype is evidence of the source lane width, not a full type
    proof. Raw interleaved-container probes may be intentional, so this is an
    advisory. Reassignments, aliases and interprocedural flows stay unknown.
    """
    assignments, stores = [], []

    class Visitor(ast.NodeVisitor):
        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                assignments.append((node.targets[0].id, node.value, node.lineno))
            self.generic_visit(node)

        def visit_Call(self, node):
            if attribute_path(node.func) == f"{source.vf}.store_align":
                stores.append(node)
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            pass

        visit_AsyncFunctionDef = visit_ClassDef = visit_Lambda = visit_FunctionDef

    visitor = Visitor()
    for statement in function.body:
        visitor.visit(statement)
    writes = bound(function.body)
    masks, narrowed = {}, {}
    for name, call, line in assignments:
        if writes[name] != 1 or not isinstance(call, ast.Call):
            continue
        options = {kw.arg: kw.value for kw in call.keywords}
        if None in options:
            continue
        operation = attribute_path(call.func)
        if operation in {f"{source.vf}.create_mask", f"{source.vf}.update_mask"}:
            masks[name] = (dtype(options.get("dtype"), blocked), line)
        elif operation == f"{source.vf}.astype" and len(call.args) >= 2:
            predicate = call.args[1]
            mask = masks.get(predicate.id) if isinstance(predicate, ast.Name) else None
            target = dtype(options.get("dtype"), blocked)
            layout = options.get("layout")
            zero = (
                layout is None
                or attribute_path(layout) == f"{source.pl}.CastLayout.ZERO"
            )
            if (
                mask
                and mask[0] in {"DT_FP32", "DT_INT32", "DT_UINT32"}
                and mask[1] < line
            ):
                if target in {"DT_FP16", "DT_BF16"} and zero:
                    narrowed[name] = line
    found = []
    for call in stores:
        if len(call.args) < 2 or not isinstance(call.args[1], ast.Name):
            continue
        name = call.args[1].id
        if name not in narrowed or call.lineno <= narrowed[name]:
            continue
        options = {kw.arg: kw.value for kw in call.keywords}
        if None in options:
            continue
        dist = options.get("dist")
        normal = dist is None or (isinstance(dist, ast.Constant) and dist.value is None)
        normal |= attribute_path(dist) in {
            f"{source.pl}.StoreDist.NORM",
            f"{source.pl}.StoreDist.NORM_B16",
        }
        if normal:
            found.append(
                Finding(
                    source.path,
                    call.lineno,
                    NARROW_STORE_RULE,
                    f"`{name}` 来自第 {narrowed[name]} 行使用 32-bit predicate 的 FP16/BF16 ZERO 转换；"
                    "直接 NORM 写出可能把交错空位一起导出。稠密数值输出需核对完整转换链："
                    "成对 de_interleave 后按目标位宽写出，或用 PACK_B32 按 b32 容器 mask 写出。"
                    "若有意观察原始交错容器，请验证其字节契约；本检查不推断源 Tile dtype。",
                    advisory=True,
                )
            )
    return found


def check(path: Path) -> list[Finding]:
    source = read(path)
    if not source.pl or not source.vf:
        return []
    module_bindings = bound(source.tree.body)
    aliases = {}

    def dtype(node, blocked=()):
        if isinstance(node, ast.Name) and node.id not in blocked:
            return aliases.get(node.id)
        name = attribute_path(node)
        prefix = source.pl + "."
        if (
            name
            and name.startswith(prefix)
            and name[slice(len(prefix), None)] in DTYPES
        ):
            return name[slice(len(prefix), None)]
        return None

    for statement in source.tree.body:
        if isinstance(statement, ast.ImportFrom) and statement.module == PL_MODULE:
            for alias in statement.names:
                name = alias.asname or alias.name
                if alias.name in DTYPES and module_bindings[name] == 1:
                    aliases[name] = alias.name
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            name = statement.targets[0].id
            value = dtype(statement.value)
            if value and module_bindings[name] == 1:
                aliases[name] = value
    for node in ast.walk(source.tree):
        if isinstance(node, ast.Global):
            for name in node.names:
                aliases.pop(name, None)

    parents = {
        child: parent
        for parent in ast.walk(source.tree)
        for child in ast.iter_child_nodes(parent)
    }
    findings = []
    for function in vector_functions(source):
        blocked = set()
        scope = function
        while scope is not None:
            if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                blocked.update(bound(scope.body))
                arguments = scope.args
                blocked.update(
                    arg.arg
                    for arg in arguments.posonlyargs
                    + arguments.args
                    + arguments.kwonlyargs
                )
                blocked.update(
                    arg.arg for arg in (arguments.vararg, arguments.kwarg) if arg
                )
            scope = parents.get(scope)
        if (
            source.pl.split(".")[0] in blocked
            or source.vf in blocked
            or module_bindings[source.pl.split(".")[0]] > 1
            or module_bindings[source.vf] > 1
        ):
            continue
        local_findings = []
        broadcast_findings = []
        parameters = {
            arg.arg
            for arg in function.args.posonlyargs
            + function.args.args
            + function.args.kwonlyargs
        }

        def scan(statements):
            known = {}
            registers = set()
            prior_bindings = set(parameters)
            for statement in statements:
                if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                    target, value = statement.targets[0], statement.value
                elif (
                    isinstance(statement, ast.AnnAssign) and statement.value is not None
                ):
                    target, value = statement.target, statement.value
                else:
                    target = value = None
                if isinstance(target, ast.Name):
                    operation = (
                        attribute_path(value.func)
                        if isinstance(value, ast.Call)
                        else None
                    )
                    if (
                        operation == f"{source.vf}.full"
                        and value.args
                        and isinstance(value.args[0], ast.Name)
                        and value.args[0].id == target.id
                        and target.id in prior_bindings
                        and target.id not in registers
                    ):
                        broadcast_findings.append(
                            Finding(
                                source.path,
                                statement.lineno,
                                BROADCAST_RULE,
                                f"`{target.id}` is already a parameter or non-VF binding. If it is a scalar, "
                                "broadcast into a fresh register name: reusing it can shadow the scalar with "
                                "an uninitialized register. dtype= does not redeclare an existing destination. "
                                "Otherwise establish that the source is an initialized register.",
                                advisory=True,
                            )
                        )
                    if operation and operation.startswith(source.vf + "."):
                        registers.add(target.id)
                    elif isinstance(value, ast.Name) and value.id in registers:
                        registers.add(target.id)
                    else:
                        registers.discard(target.id)
                    prior_bindings.add(target.id)
                    result_type = None
                    data_only = operation in {
                        f"{source.vf}.{name}" for name in DATA_ONLY_RESULTS
                    }
                    if operation in {f"{source.vf}.{name}" for name in TYPED_RESULTS}:
                        result_type = dtype(
                            next(
                                (
                                    kw.value
                                    for kw in value.keywords
                                    if kw.arg == "dtype"
                                ),
                                None,
                            ),
                            blocked,
                        )
                        first = value.args[0] if value.args else None
                        if isinstance(first, ast.Name) and first.id in known:
                            data_only |= known[first.id][2]
                    if result_type:
                        previous = known.get(target.id)
                        # pack/unpack also preserve MaskReg identity; their dtype
                        # keywords alone do not establish a RegTensor<T> change.
                        if (
                            previous
                            and previous[0] != result_type
                            and (previous[2] or data_only)
                        ):
                            local_findings.append(
                                Finding(
                                    source.path,
                                    statement.lineno,
                                    RULE,
                                    f"VF 目的 `{target.id}` 的已知类型为 {previous[0]}（来源第 {previous[1]} 行），"
                                    f"这里的显式结果类型变成 {result_type}；VF 调用赋值会写入既有目的，"
                                    "请用新名称接收不同 dtype 的转换结果",
                                )
                            )
                        if previous is None:
                            known[target.id] = (
                                result_type,
                                statement.lineno,
                                data_only,
                            )
                    elif isinstance(value, ast.Name) and value.id in known:
                        known[target.id] = known[value.id]
                    else:
                        known.pop(target.id, None)
                    continue
                for name in bound([statement]):
                    known.pop(name, None)
                    registers.discard(name)
                for field in ("body", "orelse", "finalbody"):
                    nested = getattr(statement, field, None)
                    if isinstance(nested, list) and not isinstance(
                        statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                    ):
                        scan(nested)
                if isinstance(statement, ast.Try):
                    for handler in statement.handlers:
                        scan(handler.body)

        scan(function.body)
        findings += apply_claim(
            source, function, RULE, local_findings, counterexample_claims(function)
        )
        findings += apply_claim(
            source,
            function,
            BROADCAST_RULE,
            broadcast_findings,
            counterexample_claims(function),
        )
        findings += apply_claim(
            source,
            function,
            NARROW_STORE_RULE,
            _narrow_store_findings(source, function, dtype, blocked),
            counterexample_claims(function),
        )
    return findings
