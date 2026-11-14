# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Two rules whose violations are reported only by hardware; parsing and compilation do not block them.

| Rule | Evidence | What happens if missed |
| --- | --- | --- |
| `squeeze-alternation` | Official `squeeze_store_unalign.md:43`: must alternate. | **Hang**; the card must be reset. |
| `clear-spr-before-squeeze` | Board measurement; `references/api/vf.md` §2.9. | Silently writes the wrong byte count. |

**The evidence strength differs and must be stated.** The official documentation explicitly states the first rule in its
body; the second appears in the official docs **only in a call example** as `vf.clear_spr()`
(`squeeze_store_unalign.md:61`), without saying in the body that this is required—the same evidence pattern as the
`addc` predeclaration requirement ([skill constraints](../../SKILL.md#constraints-and-workspace-policy)). The framework
docstring now includes the rule (`_vf_api.py:288`), and that wording in PyPTO `c2f7141f4` derives from the board
measurement, so it is **not independent evidence**.

Both forms pass parsing completely (third measurement in §9.5: two consecutive squeezes without an intervening store
parse successfully).

## Scope Limits

- Inspect calls only within one `@pl.vector_function` body. A squeeze inside a called helper, or a squeeze and its store
split across two functions, is invisible here.
- When one call is nested in another call's argument on the same line, source order does not equal evaluation
order—report it without judging it.
- When relevant calls appear under `if` / `with` / `try`, branch order is not unique—report without judging.
- Account for loop bodies separately: AR must have no unconsumed byte count on loop entry or exit.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional

from .source import (
    Call,
    Finding,
    apply_claim,
    attribute_path,
    calls,
    counterexample_claims,
    nested_pairs,
    read,
    vector_functions,
    Source,
)

ALTERNATION = "squeeze-alternation"
CLEAR_SPR = "clear-spr-before-squeeze"
UNALIGN_BUDGET = "unalign-reg-budget"

SQUEEZE = "squeeze"
SQUEEZE_STORE = "squeeze_store_unalign"
CLEAR = "clear_spr"
UNALIGN_DECL = "unalign_reg_for_store"

UNALIGN_LIMIT = 4
"""Maximum simultaneously live `UnalignReg` count from the constraint note in official `load.md`; measurement found that
4 pass and 5 crash."""

_LOOPS = (ast.For, ast.AsyncFor, ast.While)
_UNORDERED = (ast.If, ast.With, ast.AsyncWith, ast.Try)


def check(path: Path) -> list[Finding]:
    source = read(path)
    findings: list[Finding] = []
    for function in vector_functions(source):
        claims = counterexample_claims(function)
        for rule, found in (
            (ALTERNATION, _alternation(source, function)),
            (CLEAR_SPR, _clear_spr(source, function)),
            (UNALIGN_BUDGET, _unalign_budget(source, function)),
        ):
            findings += apply_claim(source, function, rule, found, claims)
    return findings


def store_reg_mode(call: Call, pl_alias: Optional[str]) -> Optional[bool]:
    """Whether this `vf.squeeze` writes AR: `True` writes, `False` does not, and `None` is unknown.

    `pl_alias` is only a hint; matching uses the final two components of the attribute chain. Both
    `pl.SqueezeMode.STORE_REG` and `SqueezeMode.STORE_REG` are recognized because either import form may be used.
    """
    node: Optional[ast.expr] = None
    for keyword in call.node.keywords:
        if keyword.arg is None:
            return None  # `**kwargs` expansion; cannot determine statically.
        if keyword.arg == "gather_mode":
            node = keyword.value
    if node is None:
        positional = [arg for arg in call.node.args if not isinstance(arg, ast.Starred)]
        if len(positional) != len(call.node.args):
            return None  # `*args` expansion; cannot determine statically.
        if len(positional) < 3:
            return True  # A5 codegen defaults to MODE_STORED
        node = positional[2]  # squeeze(src, preg, gather_mode)
    if isinstance(node, ast.Constant) and node.value is None:
        return None
    path = attribute_path(node)
    if path is None:
        return None
    tail = path.split(".")[-2:]
    if tail == ["SqueezeMode", "STORE_REG"]:
        return True
    if tail == ["SqueezeMode", "NO_STORE_REG"]:
        return False
    return None


def _relevant(node: ast.AST, source: Source) -> list[Call]:
    return [
        call for call in calls(node, source.vf) if call.name in (SQUEEZE, SQUEEZE_STORE)
    ]


class _Alternation:
    """Visit statements in source order and track one state: whether AR has an unconsumed byte count."""

    def __init__(self, source: Source) -> None:
        self.source = source
        self.findings: list[Finding] = []
        self.pending: Optional[Call] = None
        self.blocked = False

    def _say(self, lineno: int, message: str) -> None:
        self.findings.append(Finding(self.source.path, lineno, ALTERNATION, message))

    def statements(self, body: list[ast.stmt]) -> None:
        for statement in body:
            if self.blocked:
                return
            if isinstance(statement, _LOOPS):
                self._loop(statement)
            elif isinstance(statement, _UNORDERED) and _relevant(
                statement, self.source
            ):
                self.blocked = True
                self._say(
                    statement.lineno,
                    f"`vf.{SQUEEZE}` 这条链出现在 `{type(statement).__name__.lower()}` 里"
                    "——分支顺序不唯一，这条规则不判这种形状；把整条链放在同一段直线代码里",
                )
            elif isinstance(
                statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue  # Nested definitions are handled separately by vector_functions().
            else:
                self._flat(statement)

    def _flat(self, statement: ast.stmt) -> None:
        items = _relevant(statement, self.source)
        nested = nested_pairs(items)
        if nested:
            outer, inner = nested[0]
            self.blocked = True
            self._say(
                outer.lineno,
                f"`vf.{inner.name}`（{inner.lineno} 行）嵌在 `vf.{outer.name}` 的实参里"
                "——源码顺序不等于求值顺序，这条规则不判这种形状；拆成两条语句",
            )
            return
        for call in items:
            self._one(call)

    def _one(self, call: Call) -> None:
        if call.name == SQUEEZE_STORE:
            if self.pending is None:
                self._say(
                    call.lineno,
                    f"`vf.{SQUEEZE_STORE}` 前面没有未被消费的 `vf.{SQUEEZE}(STORE_REG)`"
                    "——它的推进量从 AR 读，AR 这一轮没被写过就是陈旧值",
                )
            self.pending = None
            return
        mode = store_reg_mode(call, self.source.pl)
        if mode is None:
            self.blocked = True
            self._say(
                call.lineno,
                f"`vf.{SQUEEZE}` 的 `gather_mode` 不是 `SqueezeMode` 字面量，"
                "静态判不了它写不写 AR；写成字面量，或把这段拆出去",
            )
            return
        if mode is False:
            return  # NO_STORE_REG does not touch AR and does not participate in alternation.
        if self.pending is not None:
            self._say(
                call.lineno,
                f"两个 `vf.{SQUEEZE}(STORE_REG)` 之间没有 `vf.{SQUEEZE_STORE}`"
                f"（上一个在 {self.pending.lineno} 行）——AR 里没被消费的字节数被覆盖，"
                "官方 `squeeze_store_unalign.md` 说这会导致硬件挂死",
            )
        self.pending = call

    def _loop(self, statement: ast.stmt) -> None:
        """Check loop bodies independently: every iteration must balance its own AR state because there is only one
        AR."""
        items = _relevant(statement, self.source)
        if not items:
            return
        if self.pending is not None:
            self._say(
                statement.lineno,
                f"循环体里用了 `vf.{SQUEEZE}` 这条链，而进循环时 AR 里还有 "
                f"{self.pending.lineno} 行那次 `vf.{SQUEEZE}(STORE_REG)` 没被消费"
                "——循环外写一次、循环内每轮消费，第二轮起读的是陈旧值",
            )
            # Keep pending so a store in the loop body can consume it normally. Do not report
            # “no preceding unconsumed squeeze” again inside the body; that follows from this issue.
        self.statements(statement.body)
        if self.pending is not None:
            self._say(
                self.pending.lineno,
                f"循环体里的 `vf.{SQUEEZE}(STORE_REG)` 没在同一轮被 "
                f"`vf.{SQUEEZE_STORE}` 消费——下一轮会覆盖 AR",
            )
            self.pending = None
        for clause in getattr(statement, "orelse", []):
            self.statements([clause])


def _alternation(source: Source, function: ast.FunctionDef) -> list[Finding]:
    scan = _Alternation(source)
    scan.statements(function.body)
    if scan.pending is not None and not scan.blocked:
        scan.findings.append(
            Finding(
                source.path,
                scan.pending.lineno,
                ALTERNATION,
                f"`vf.{SQUEEZE}(STORE_REG)` 写了 AR 但这个 `@pl.vector_function` 体里没有"
                f"后续的 `vf.{SQUEEZE_STORE}` 消费它",
            )
        )
    return scan.findings


def _clear_spr(source: Source, function: ast.FunctionDef) -> list[Finding]:
    writers = [
        call
        for call in calls(function, source.vf)
        if call.name == SQUEEZE and store_reg_mode(call, source.pl) is True
    ]
    if not writers:
        return []
    first = writers[0]
    cleared = [call for call in calls(function, source.vf) if call.name == CLEAR]
    if any((call.lineno, call.col) < (first.lineno, first.col) for call in cleared):
        return []
    return [
        Finding(
            source.path,
            first.lineno,
            CLEAR_SPR,
            f"第一个 `vf.{SQUEEZE}(STORE_REG)` 之前没有 `vf.{CLEAR}()`"
            "——AR 是特殊寄存器，进 `@pl.vector_function` 时不保证为空，"
            "陈旧字节数会被当推进量用，谁都不报错",
        )
    ]


def _unalign_lifetimes(
    source: Source, function: ast.FunctionDef
) -> list[tuple[str, int, int]]:
    """For each `name = vf.unalign_reg_for_store()`, return `(name, declaration line, last-mentioned line)`.

    “Last mentioned” is the final line where the name appears in the function body; this is not full liveness analysis.
    Conservatism is appropriate here: if the name appears later, it must remain live.
    """
    if source.vf is None:
        return []
    declared: dict[str, int] = {}
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target, value = node.targets[0], node.value
        if not isinstance(target, ast.Name) or not isinstance(value, ast.Call):
            continue
        func = value.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == UNALIGN_DECL
            and isinstance(func.value, ast.Name)
            and func.value.id == source.vf
        ):
            declared.setdefault(target.id, node.lineno)

    last: dict[str, int] = dict(declared)
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and node.id in declared:
            last[node.id] = max(last[node.id], node.lineno)
    return [
        (name, line, last[name])
        for name, line in sorted(declared.items(), key=lambda i: i[1])
    ]


def _unalign_budget(source: Source, function: ast.FunctionDef) -> list[Finding]:
    """The number of simultaneously live `UnalignReg` values must not exceed `UNALIGN_LIMIT`.

    **Count overlapping live ranges, not declarations.** This distinction was measured on 2026-09-27, Ascend950PR:

    | Shape | Result |
    | --- | --- |
    | Four live for the whole function | Pass. |
    | Five live for the whole function | **bisheng segfaults in the Greedy Register Allocator**, with no diagnostic. |
    | Five, each declared and consumed immediately, with disjoint live ranges | **Pass.** |
    | Four live values plus one intervening `vf.load` (which declares one internally) | **Pass.** |

    The third row is why this rule does not count declarations: that would falsely reject runnable code. The fourth row
    is why it does not count `vf.load` / `vf.store`: their internal register is declared and consumed immediately, which
    the allocator handles.
    """
    spans = _unalign_lifetimes(source, function)
    if len(spans) <= UNALIGN_LIMIT:
        return []
    for line in range(
        min(start for _, start, _ in spans), max(end for _, _, end in spans) + 1
    ):
        live = [name for name, start, end in spans if start <= line <= end]
        if len(live) > UNALIGN_LIMIT:
            return [
                Finding(
                    source.path,
                    line,
                    UNALIGN_BUDGET,
                    f"这一行有 {len(live)} 个 `vf.{UNALIGN_DECL}` 同时活跃"
                    f"（{'、'.join(live)}），上限是 {UNALIGN_LIMIT}"
                    "——超了会让 bisheng 在 Greedy Register Allocator 里 segfault，"
                    "没有任何说明寄存器上限的诊断；把用完的那几个提前用掉，"
                    "让活跃期不重叠",
                )
            ]
    return []
