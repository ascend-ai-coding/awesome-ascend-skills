# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared source inspection for import aliases, decorators, call sequences, and declarations in docstrings.

**Read the AST only; do not import the checked module.** This is deliberate: the checks must run on machines without
`pypto_pro` installed (the development machine lacks it) and must not consume board time. The tradeoff is that runtime
values are unavailable—for example, when `gather_mode=` receives a variable rather than a literal, the checker cannot
decide; it emits a finding under the “report unknowns” rule in `references/tooling/checks.md`. Do not guess.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PL_MODULE = "pypto_pro.language"
VF_CLASS = "Vf"

# Counterexample declaration format: `反例：<rule-id> —— <why it remains>`. The dash is two U+2014 characters, matching
# other repository docs.
# Allow leading indentation—docstrings are indented, and requiring a flush-left marker would make examples ugly.
_COUNTEREXAMPLE = re.compile(
    r"^[ \t]*反例：([a-z][a-z0-9-]*)[ \t]*——[ \t]*(\S.*)$", re.MULTILINE
)


@dataclass(frozen=True)
class Finding:
    """One finding. String form is `file:line: rule: message`, with a clickable line number.

    `waived` means a violation is permitted by a `反例：` declaration—it is real and intentionally retained. It is still
    printed, but does not count as a failure; a hidden waiver is no waiver at all.
    """

    path: Path
    lineno: int
    rule: str
    message: str
    waived: bool = False
    advisory: bool = False

    def __str__(self) -> str:
        mark = "[反例] " if self.waived else "[需核查] " if self.advisory else ""
        return f"{self.path}:{self.lineno}: {mark}{self.rule}: {self.message}"

    def waive(self, reason: str) -> "Finding":
        return Finding(
            self.path,
            self.lineno,
            self.rule,
            f"{self.message}（声明的理由：{reason}）",
            waived=True,
            advisory=self.advisory,
        )


@dataclass(frozen=True)
class Call:
    """One `<alias>.<name>(...)` call."""

    lineno: int
    col: int
    name: str
    node: ast.Call


@dataclass(frozen=True)
class Source:
    path: Path
    tree: ast.Module
    pl: Optional[str]
    """Alias from `import pypto_pro.language as pl`, or None if that import is absent."""
    vf: Optional[str]
    """Alias from `from pypto_pro.language import Vf as vf`, or None if that import is absent."""


def read(path: Path) -> Source:
    text = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    return Source(
        path=Path(path), tree=tree, pl=_module_alias(tree), vf=_vf_alias(tree)
    )


def _module_alias(tree: ast.Module) -> Optional[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == PL_MODULE:
                    return alias.asname or alias.name
    return None


def _vf_alias(tree: ast.Module) -> Optional[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == PL_MODULE:
            for alias in node.names:
                if alias.name == VF_CLASS:
                    return alias.asname or alias.name
    return None


def has_decorator(fn: ast.AST, alias: Optional[str], name: str) -> bool:
    """Recognize both `@alias.name` and `@alias.name(...)` forms (`@pl.jit()` has parentheses; `@pl.vector_function`
    does not).
    """
    if alias is None or not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    for decorator in fn.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if (
            isinstance(target, ast.Attribute)
            and target.attr == name
            and isinstance(target.value, ast.Name)
            and target.value.id == alias
        ):
            return True
    return False


def auto_mutex_off(fn: ast.AST, alias: Optional[str]) -> bool:
    """Treat `auto_mutex` as disabled only when `@pl.jit(auto_mutex=False)` is explicit.

    The signature in `runtime/jit.py` is `auto_mutex: bool = True`. Bare `@pl.jit()` also inserts mutexes, so an omitted
    switch does not disable automatic synchronization.

    A variable such as `auto_mutex=FLAG` cannot be resolved statically; treat uncertainty as not-disabled—return False
    and run the rule normally. Missing a violation is worse than allowing a real defect.
    """
    if alias is None or not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    for decorator in fn.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        if attribute_path(decorator.func) != f"{alias}.jit":
            continue
        for keyword in decorator.keywords:
            if (
                keyword.arg == "auto_mutex"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is False
            ):
                return True
    return False


def vector_functions(source: Source) -> list[ast.FunctionDef]:
    """Return all `@pl.vector_function` functions in this file, sorted by line number; include nested definitions."""
    found = [
        node
        for node in ast.walk(source.tree)
        if has_decorator(node, source.pl, "vector_function")
    ]
    return sorted(found, key=lambda fn: fn.lineno)


def calls(node: ast.AST, prefix: Optional[str]) -> list[Call]:
    """Return `<prefix>.<name>(...)` calls sorted by source location (line, column).

    `prefix` may contain dots: `"pl"` matches only `pl.load(...)`; `"pl.system"` matches `pl.system.bar_all(...)`. They
    do not subsume one another; match by the **number of components** in the attribute chain, so `pl.system.x` is not
    treated as an API named `system` in the `pl` namespace. Measurement found this distinction because the initial index
    omitted all 56 `pl.system.*` calls in examples.

    `ast.walk` is breadth-first, not source-ordered, so sort explicitly. **Source order is not evaluation order**: an
    inner call nested in another call's argument is evaluated first. `nested_pairs` reports that form; this function
    does not handle it.
    """
    if prefix is None:
        return []
    depth = prefix.count(".") + 1
    found = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        path = attribute_path(child.func)
        if path is None:
            continue
        parts = path.split(".")
        if len(parts) == depth + 1 and ".".join(parts[:-1]) == prefix:
            found.append(Call(child.lineno, child.col_offset, parts[-1], child))
    return sorted(found, key=lambda call: (call.lineno, call.col))


def used_names(node: ast.AST, prefix: Optional[str]) -> set[str]:
    """Return names used as `<prefix>.<name>`: calls **plus bare decorators without parentheses**.

    `@pl.jit()` has parentheses and appears as an AST `Call`, so `calls()` finds it; `@pl.vector_function` has no
    parentheses and appears as a bare `Attribute`, which a Call-only scan would miss. Conversely, do not include every
    `alias.<attr>`—that would flood the index with constants such as `pl.DT_FP32` / `pl.ND` / `pl.MaskPattern.ALL`.
    """
    if prefix is None:
        return set()
    depth = prefix.count(".") + 1
    names = {call.name for call in calls(node, prefix)}
    for child in ast.walk(node):
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for decorator in child.decorator_list:
            path = attribute_path(decorator)
            if path is None:
                continue
            parts = path.split(".")
            if len(parts) == depth + 1 and ".".join(parts[:-1]) == prefix:
                names.add(parts[-1])
    return names


def nested_pairs(items: list[Call]) -> list[tuple[Call, Call]]:
    """Return `(outer, inner)` pairs where an inner `Call` node is inside the outer call's subtree.

    Callers should report a nonempty result and stop judging; source-location order is reversed for this form.
    """
    pairs = []
    for outer in items:
        inside = {id(child) for child in ast.walk(outer.node)} - {id(outer.node)}
        pairs += [(outer, inner) for inner in items if id(inner.node) in inside]
    return pairs


def attribute_path(node: ast.AST) -> Optional[str]:
    """Convert `pl.SqueezeMode.STORE_REG` to `"pl.SqueezeMode.STORE_REG"`; return None unless the node is a pure
    attribute chain.
    """
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def first_docstring_line(node: ast.AST) -> Optional[str]:
    """Return the first docstring line after removing leading blank lines and surrounding whitespace; return None if
    there is no docstring.
    """
    text = ast.get_docstring(node, clean=False)
    if text is None:
        return None
    return text.lstrip("\n").split("\n")[0].strip()


def counterexample_claims(node: ast.AST) -> dict[str, str]:
    """Return counterexamples declared in this function's docstring as `{rule-id: reason}`.

    The example library **deliberately** retains violations (for example, `vf_unalign`'s “two squeezes without an
    intervening store,” which parses but is not run on board). A declaration lets the checker waive the finding, while
    also requiring that the violation **really exists**—if the declaration goes stale, report it, or an invalid waiver
    could silently hide a later bug.
    """
    text = ast.get_docstring(node, clean=False) or ""
    return {rule: reason.strip() for rule, reason in _COUNTEREXAMPLE.findall(text)}


def apply_claim(
    source: "Source",
    function: ast.AST,
    rule: str,
    found: list[Finding],
    claims: dict[str, str],
) -> list[Finding]:
    """Waive a finding when `反例：<rule>` is declared, but require the violation to **really exist**.

    The second half is critical: waivers go stale. If the counterexample becomes compliant but the declaration remains,
    it could hide a real future violation.

    This logic lives here rather than in one rule module because both `hardware` and `synchronisation` use it, and
    waiver semantics must have one definition.
    """
    if rule not in claims:
        return found
    if found:
        return [finding.waive(claims[rule]) for finding in found]
    name = getattr(function, "name", "<module>")
    return [
        Finding(
            source.path,
            function.lineno,
            rule,
            f"`{name}` 声明了 `反例：{rule}` 但没有检出违规"
            f"——豁免失效了，删掉声明或把反例恢复回去",
        )
    ]
