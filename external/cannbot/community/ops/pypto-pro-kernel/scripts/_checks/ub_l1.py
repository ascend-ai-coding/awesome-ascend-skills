# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Bounded source facts for local transfers; no imports or guessed tile defaults.

Only direct native aliases, simple constants, local tile declarations and group
slots are followed. Unknown helpers, rebindings and control-flow joins stay unknown.
The odd-NZ move rejection is gated by the selected frontend's exact guard source.
"""

from __future__ import annotations

import ast
import hashlib
import operator
from functools import lru_cache
from pathlib import Path

from .paths import PYPTO_ROOT
from .source import (
    Finding,
    apply_claim,
    attribute_path,
    counterexample_claims,
    has_decorator,
    read,
)

LAYOUT = "ub-l1-layout-mismatch"
PITCH = "ub-nz-move-pitch"
PITCH_GUARD = "a2f24a23e00671444b04c20db67f7d0fe67ba5432f600df3239c39e8566f0910"
_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}


def pitch_guard(pypto_root):
    path = pypto_root / "python/pypto_pro/ir/op/block_ops.py"
    try:
        return _pitch_guard(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return False


@lru_cache(maxsize=4)
def _pitch_guard(payload):
    tree = ast.parse(payload)
    return any(
        hashlib.sha256(ast.get_source_segment(payload, n).encode()).hexdigest()
        == PITCH_GUARD
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and any(
            isinstance(c, ast.Constant)
            and isinstance(c.value, str)
            and c.value.startswith("move: a Vec-to-Vec move into an NZ Vec")
            for c in ast.walk(n)
        )
    )


def argument(call, name, position=None):
    return next(
        (kw.value for kw in call.keywords if kw.arg == name),
        call.args[position]
        if position is not None and len(call.args) > position
        else None,
    )


def value(node, env, pl):
    if isinstance(node, ast.Constant) and type(node.value) in (int, str, bool):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id)
    name = attribute_path(node) if node is not None else None
    if (
        name
        and pl
        and (
            name.startswith(pl + ".DT_")
            or name.startswith(pl + ".MemorySpace.")
            or name in {pl + "." + k for k in ("ND", "DN", "NZ", "ZN", "ZZ")}
        )
    ):
        return name.rsplit(".", 1)[-1]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [value(n, env, pl) for n in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = value(node.operand, env, pl)
        return -v if type(v) is int else None
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        a, b = value(node.left, env, pl), value(node.right, env, pl)
        if type(a) is int and type(b) is int and abs(a) < 2**31 and abs(b) < 2**31:
            try:
                return _BINARY[type(node.op)](a, b)
            except ZeroDivisionError:
                return None
    if isinstance(node, ast.Call):
        name = attribute_path(node.func)
        if name == f"{pl}.TileType" and not any(k.arg is None for k in node.keywords):
            return {
                key: value(argument(node, field), env, pl)
                for key, field in (
                    ("space", "target_memory"),
                    ("layout", "layout"),
                    ("dtype", "dtype"),
                    ("shape", "shape"),
                    ("valid_shape", "valid_shape"),
                )
            }
        if name in {
            f"{pl}.make_tile",
            f"{pl}.make_tile_group",
            f"{pl}.allocate_tile_group",
        }:
            return value(argument(node, "type", 0), env, pl)
        if isinstance(node.func, ast.Attribute) and node.func.attr in {
            "current",
            "next",
            "get",
        }:
            candidate = value(node.func.value, env, pl)
            return (
                candidate
                if isinstance(candidate, dict) and "space" in candidate
                else None
            )
    if isinstance(node, ast.Subscript):
        candidate = value(node.value, env, pl)
        return (
            candidate if isinstance(candidate, dict) and "space" in candidate else None
        )
    return None


def transfer_issues(api, src, dst, guard):
    """Shared descriptor checks for source facts and captured native operands."""
    issues = []
    if src.get("space") == "Vec" and dst.get("space") == "Mat":
        if src.get("layout") == "ND" and dst.get("layout") in {"NZ", "ZN"}:
            issues.append(
                (
                    LAYOUT,
                    True,
                    "UB→L1 copies physical bytes; an ND source and NZ/ZN destination do not perform layout conversion. "
                    "Verify the producer's address map; for logical ND values, pack NZ in UB before the handover. "
                    "A deliberately prepacked ND view needs separate evidence.",
                )
            )
    shape = dst.get("shape")
    if (
        api == "move"
        and src.get("space") == dst.get("space") == "Vec"
        and dst.get("layout") == "NZ"
        and isinstance(shape, list)
        and len(shape) == 2
        and type(shape[0]) is int
        and shape[0] > 0
        and shape[0] % 16
    ):
        issues.append(
            (
                PITCH,
                not guard,
                "Vec→Vec move into this NZ tile uses a non-16-aligned physical row count. "
                + (
                    "The selected frontend rejects this pitch mismatch. "
                    if guard
                    else "The selected frontend's pitch guard is unrecognized; native parsing is required. "
                )
                + "Use an aligned conversion buffer, or the complete direct VF-to-NZ recipe; "
                "changing logical rows or blindly rounding an existing direct-store pitch is not a repair.",
            )
        )
    return issues


def inspect(path: Path, *, pypto_root=PYPTO_ROOT):
    source = read(path)
    facts, results = [], []
    coverage = {"resolved": 0, "unverified": 0}
    if not source.pl:
        return results, facts, coverage
    # A module binding changed anywhere is not a compile-time constant here.
    stores = {}
    for node in source.tree.body:
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                stores[child.id] = stores.get(child.id, 0) + 1
    constants = {}
    rebound = set()
    namespace_imports = 0
    for node in ast.walk(source.tree):
        if isinstance(node, ast.Import):
            namespace_imports += sum(
                (a.asname or a.name) == source.pl for a in node.names
            )
        if isinstance(node, ast.ImportFrom):
            namespace_imports += sum(
                (a.asname or a.name) == source.pl for a in node.names
            )
        if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(
            node.ctx, ast.Store
        ):
            target = node
            while isinstance(target, (ast.Attribute, ast.Subscript)):
                target = target.value
            if isinstance(target, ast.Name):
                rebound.add(target.id)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr
            in {
                "append",
                "extend",
                "insert",
                "pop",
                "clear",
                "update",
                "remove",
                "sort",
                "reverse",
            }
        ):
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id != source.pl
            ):
                rebound.add(node.func.value.id)
    for node in source.tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and stores[target.id] == 1
                    and target.id not in rebound
                ):
                    resolved = value(node.value, constants, source.pl)
                    if not isinstance(resolved, dict):
                        constants[target.id] = resolved
    guard = None

    def inspect_function(function):
        findings = []
        if not has_decorator(function, source.pl, "jit"):
            return findings
        arguments = (
            function.args.posonlyargs + function.args.args + function.args.kwonlyargs
        )
        local_names = {a.arg for a in arguments} | {
            n.id
            for n in ast.walk(function)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }
        shadowed = (
            source.pl in local_names
            or stores.get(source.pl, 0)
            or namespace_imports != 1
            or source.pl in rebound
        )
        env = {k: v for k, v in constants.items() if k not in local_names}
        found = []

        def expression(node, bindings, uncertain=False):
            nonlocal guard
            if isinstance(
                node, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                return
            if isinstance(node, ast.Call) and attribute_path(node.func) in {
                f"{source.pl}.move",
                f"{source.pl}.insert",
            }:
                api = node.func.attr
                src = (
                    value(argument(node, "src_tile", 1), bindings, source.pl)
                    if not shadowed
                    else None
                )
                dst = (
                    value(argument(node, "dst_tile", 0), bindings, source.pl)
                    if not shadowed
                    else None
                )
                src = src if isinstance(src, dict) else {}
                dst = dst if isinstance(dst, dict) else {}
                known = bool(src.get("space") and dst.get("space"))
                if (
                    not known
                    or src.get("space") == "Vec"
                    and dst.get("space") in {"Vec", "Mat"}
                ):
                    complete = known and all(
                        t.get("layout") is not None
                        and t.get("dtype") is not None
                        and isinstance(t.get("shape"), list)
                        and len(t["shape"]) == 2
                        and all(type(d) is int for d in t["shape"])
                        for t in (src, dst)
                    )
                    coverage["resolved" if complete else "unverified"] += 1
                    facts.append(
                        {
                            "file": str(path.resolve()),
                            "line": node.lineno,
                            "end_line": node.end_lineno,
                            "function": function.name,
                            "source_expr": ast.unparse(argument(node, "src_tile", 1))
                            if argument(node, "src_tile", 1) is not None
                            else None,
                            "destination_expr": ast.unparse(
                                argument(node, "dst_tile", 0)
                            )
                            if argument(node, "dst_tile", 0) is not None
                            else None,
                            "api": api,
                            "source": src,
                            "destination": dst,
                            "offset": value(
                                argument(node, "offset", 2), bindings, source.pl
                            ),
                            "scope": "source descriptors; producer bytes, helpers and synchronization are unverified",
                        }
                    )
                    if guard is None:
                        guard = pitch_guard(pypto_root)
                    for rule, advisory, message in transfer_issues(
                        api, src, dst, guard
                    ):
                        if uncertain and not advisory:
                            advisory = True
                            message += (
                                " The containing control flow is unresolved; confirm"
                                " the active specialization with native preflight."
                            )
                        found.append(
                            Finding(path, node.lineno, rule, message, advisory=advisory)
                        )
            for child in ast.iter_child_nodes(node):
                expression(child, bindings, uncertain)

        def statements(body, bindings, uncertain=False):
            for node in body:
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    bindings.pop(node.name, None)
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    rhs = node.value
                    if rhs is not None:
                        expression(rhs, bindings, uncertain)
                    resolved = (
                        value(rhs, bindings, source.pl) if rhs is not None else None
                    )
                    targets = (
                        node.targets if isinstance(node, ast.Assign) else [node.target]
                    )
                    for target in targets:
                        for n in ast.walk(target):
                            if isinstance(n, ast.Name):
                                bindings.pop(n.id, None)
                        if isinstance(target, ast.Name):
                            bindings[target.id] = resolved
                        elif (
                            isinstance(target, (ast.Tuple, ast.List))
                            and isinstance(resolved, list)
                            and len(target.elts) == len(resolved)
                        ):
                            for item, item_value in zip(target.elts, resolved):
                                if isinstance(item, ast.Name):
                                    bindings[item.id] = item_value
                elif isinstance(node, ast.With):
                    for item in node.items:
                        expression(item.context_expr, bindings, uncertain)
                        if item.optional_vars:
                            for n in ast.walk(item.optional_vars):
                                if isinstance(n, ast.Name):
                                    bindings.pop(n.id, None)
                    statements(node.body, bindings, uncertain)
                elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                    if isinstance(node, ast.If):
                        condition = value(node.test, bindings, source.pl)
                        if type(condition) in (int, bool):
                            statements(
                                node.body if condition else node.orelse,
                                bindings,
                                uncertain,
                            )
                            continue
                    changed = {
                        n.id
                        for n in ast.walk(node)
                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                    }
                    for name in changed:
                        bindings.pop(name, None)
                    for field in ("test", "iter"):
                        expr = getattr(node, field, None)
                        if expr is not None:
                            expression(expr, bindings, uncertain)
                    for block in [
                        node.body,
                        node.orelse,
                        getattr(node, "finalbody", []),
                    ] + [h.body for h in getattr(node, "handlers", [])]:
                        statements(block, bindings.copy(), True)
                elif isinstance(node, ast.AugAssign):
                    expression(node.value, bindings, uncertain)
                    if isinstance(node.target, ast.Name):
                        bindings.pop(node.target.id, None)
                else:
                    expression(node, bindings, uncertain)

        statements(function.body, env)
        claims = counterexample_claims(function)
        for rule in (LAYOUT, PITCH):
            findings += apply_claim(
                source, function, rule, [f for f in found if f.rule == rule], claims
            )
        return findings

    for candidate in ast.walk(source.tree):
        results += inspect_function(candidate)
    return results, facts, coverage


def check(path, *, pypto_root=PYPTO_ROOT):
    return inspect(path, pypto_root=pypto_root)[0]
