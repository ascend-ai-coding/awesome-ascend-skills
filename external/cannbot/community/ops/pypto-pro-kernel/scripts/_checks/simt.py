# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Rule: every thread in a block must reach `pl.simt.syncthreads()` uniformly.

| Rule | Evidence | What happens if missed |
| --- | --- | --- |
| `syncthreads-must-be-uniform` | Docstring `_simt_api.py`: parsing allows it. | **Hang**: some threads wait forever. |

Original docstring:

> The barrier must be reached uniformly by every thread in the block. It **cannot** be placed
> inside runtime ``if``, ``for``, or ``while`` control flow.

**Parsing does not block this at all** (measured 2026-09-28): `parse_target_program` accepts `syncthreads()` inside `if
tid > 0:` and compilation is also silent. Some threads reach the barrier while others skip it, leaving the former
waiting for a signal that never arrives.

This is the kind of case described by the criterion in §3 of [skill
constraints](../../SKILL.md#constraints-and-workspace-policy): **the documentation cannot correct it (the docs already
state the rule), the symptom is a hang, and it can be checked statically → put the check in the skill's `scripts/`.**

Control: `threadfence_block()` / `threadfence()` **may** appear in runtime control flow (explicitly allowed by their
docstring and passed measurement), so this rule applies only to `syncthreads`. Banning both would reject correct
kernels.

## Criterion

Inside a function decorated with `@pl.vector_function(mode="simt", ...)`, report `pl.simt.syncthreads()` when it appears
inside `if` / `for` / `while` / `with` / `try`.

## Scope Limits

- **Inspect only functions with `mode="simt"`.** Other `vector_function` forms do not contain `syncthreads`.
- **Do not decide whether a condition is runtime or compile-time.** A compile-time constant such as `if THREADS > 8:` is
folded during parsing and, literally, is not a violation under the docstring. But statically distinguishing runtime from
compile-time conditions requires constant propagation, which belongs to the parser ([skill
constraints](../../SKILL.md#constraints-and-workspace-policy): do not reimplement it). **Report both; use a `反例：`
declaration to waive the compile-time case.** This scope note is necessary because the broad rule can produce false
positives.
- **Do not analyze across functions.** This check cannot see a `syncthreads` inside a called helper when the branch is
in the caller.
"""

from __future__ import annotations

import ast
from pathlib import Path

from .source import (
    Finding,
    Source,
    apply_claim,
    attribute_path,
    counterexample_claims,
    read,
)

RULE = "syncthreads-must-be-uniform"

SYNCTHREADS = "syncthreads"
SIMT_MODE = "simt"

_BRANCHING = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Try,
)


def check(path: Path) -> list[Finding]:
    source = read(path)
    findings: list[Finding] = []
    for function in _simt_functions(source):
        found = _syncthreads_under_branches(source, function)
        findings += apply_claim(
            source, function, RULE, found, counterexample_claims(function)
        )
    return findings


def _simt_functions(source: Source) -> list[ast.FunctionDef]:
    """Return functions decorated with `@pl.vector_function(mode="simt", ...)`, sorted by line number.

    Recognize only the literal `"simt"`. If a variable is used, the mode cannot be determined statically and is not
    reported; a “possible” violation would make the tool noisy.
    """
    found = [
        node
        for node in ast.walk(source.tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _is_simt(node, source.pl)
    ]
    return sorted(found, key=lambda fn: fn.lineno)


def _is_simt(fn: ast.AST, alias: str | None) -> bool:
    if alias is None or not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    for decorator in fn.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue  # Without parentheses, `@pl.vector_function` cannot receive mode.
        if attribute_path(decorator.func) != f"{alias}.vector_function":
            continue
        for keyword in decorator.keywords:
            if (
                keyword.arg == "mode"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == SIMT_MODE
            ):
                return True
    return False


def _syncthreads_under_branches(
    source: Source, function: ast.FunctionDef
) -> list[Finding]:
    """Find `syncthreads()` calls in the function body that are nested under branches.

    Use a **depth-aware** traversal rather than `ast.walk`: `walk` does not show which node contains another, and this
    rule depends on nesting relationships.
    """
    target = None if source.pl is None else f"{source.pl}.simt.{SYNCTHREADS}"
    findings: list[Finding] = []

    def visit(node: ast.AST, enclosing: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            inside = enclosing
            if isinstance(child, _BRANCHING):
                inside = type(child).__name__
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue  # Nested definitions are handled in their own pass.
            if (
                enclosing is not None
                and isinstance(child, ast.Call)
                and attribute_path(child.func) == target
            ):
                findings.append(
                    Finding(
                        source.path,
                        child.lineno,
                        RULE,
                        f"`{target}()` 在 `{enclosing}` 里——屏障必须被 block 里每个线程一致地到达,"
                        f"否则一部分线程会等一个永远不来的信号（挂死）。"
                        f"解析期和编译期都不拦；栅栏 `threadfence_block` / `threadfence` 不受此限",
                    )
                )
            visit(child, inside)

    visit(function, None)
    return findings
