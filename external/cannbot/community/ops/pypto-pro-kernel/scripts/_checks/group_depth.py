# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check explicit group slot declarations only where native identity is known."""

import ast
import hashlib
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

RULE = "tile-group-depth-required"
GUARD = "14636575aa826887641b7f5389f9c2fb55a6858438186f9676bb0ab8c1f3783b"
PARSER = "python/pypto_pro/language/parser/_buffer_parser.py"


def guard_verified(pypto_root):
    try:
        text = (pypto_root / PARSER).read_text(encoding="utf-8")
        return _guard(text)
    except (OSError, SyntaxError):
        return False


@lru_cache(maxsize=4)
def _guard(text):
    return any(
        hashlib.sha256(ast.get_source_segment(text, node).encode()).hexdigest() == GUARD
        for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.FunctionDef)
        and node.name == "_parse_mutex_ids_and_depth"
    )


def inspect(path: Path, *, pypto_root=PYPTO_ROOT):
    source = read(path)
    name = f"{source.pl}.make_tile_group"
    calls = [
        node
        for node in ast.walk(source.tree)
        if isinstance(node, ast.Call)
        and source.pl
        and attribute_path(node.func) == name
    ]
    coverage = {
        "checked": 0,
        "unverified": len(calls),
        "guard_verified": guard_verified(pypto_root),
    }
    imports = [
        alias
        for node in source.tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "pypto_pro.language" and alias.asname == source.pl
    ]
    if not imports:
        return [], coverage
    # Skip the file if a visible binding could replace the imported namespace.
    for node in ast.walk(source.tree):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and node.id == source.pl
            or isinstance(node, ast.arg)
            and node.arg == source.pl
            or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == source.pl
            or isinstance(node, ast.ExceptHandler)
            and node.name == source.pl
            or isinstance(node, ast.alias)
            and (node.asname or node.name) == source.pl
            and node not in imports
            or isinstance(node, ast.Attribute)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and (attribute_path(node) or "").startswith(source.pl + ".")
        ):
            return [], coverage

    def declarations(body):
        for statement in body:
            if isinstance(statement, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                break
            if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Expr)):
                call = statement.value
                if isinstance(call, ast.Call) and attribute_path(call.func) == name:
                    yield call
            elif isinstance(statement, ast.With) and all(
                isinstance(item.context_expr, ast.Call)
                and attribute_path(item.context_expr.func)
                in {source.pl + ".section_cube", source.pl + ".section_vector"}
                for item in statement.items
            ):
                yield from declarations(statement.body)

    findings = []
    for function in ast.walk(source.tree):
        if not has_decorator(function, source.pl, "jit"):
            continue
        found = []
        for call in declarations(function.body):
            if call.args or any(keyword.arg is None for keyword in call.keywords):
                continue
            args = {keyword.arg: keyword.value for keyword in call.keywords}
            depth = args.get("depth")
            if depth is not None and not (
                isinstance(depth, ast.Constant) and depth.value is None
            ):
                coverage["checked"] += 1
                continue
            mutex = args.get("mutex_ids")
            literal = isinstance(mutex, ast.Constant) and mutex.value in ("auto", None)
            sequence = isinstance(mutex, (ast.List, ast.Tuple))
            if not literal and not sequence:
                continue
            coverage["checked"] += 1
            if sequence and mutex.elts:
                continue
            found.append(
                Finding(
                    source.path,
                    call.lineno,
                    RULE,
                    'Explicit mutex_ids="auto", None or an empty sequence needs a positive compile-time depth; '
                    "choose the intended slots and verify placement/lifetimes. "
                    + (
                        "The selected parser guard is verified."
                        if coverage["guard_verified"]
                        else "The selected parser guard is unknown; verify this requirement natively."
                    ),
                    advisory=not coverage["guard_verified"],
                )
            )
        findings += apply_claim(
            source, function, RULE, found, counterexample_claims(function)
        )
    coverage["unverified"] -= coverage["checked"]
    return findings, coverage


def check(path, *, pypto_root=PYPTO_ROOT):
    return inspect(path, pypto_root=pypto_root)[0]
