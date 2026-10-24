# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Report bare tiles relying on automatic mutex insertion.

The default and explicit True both enable auto_mutex; only tile-group metadata
supplies mutex IDs. A real-parser probe on pypto 13d4573c064d (CANN 9.2.0,
Python 3.11.15, 2026-09-28) parsed a load/store pair under @pl.jit, @pl.jit(),
explicit True and False, with bare/group tiles. The first three emitted two
lock/unlock pairs for a group and none for a bare tile; False emitted none.

Unknown expressions and expanded decorator options are not judged. Any manual
synchronization silences this rule; its adequacy remains the compiler's concern.
"""

from __future__ import annotations

import ast
from pathlib import Path

from .source import (
    Finding,
    Source,
    apply_claim,
    attribute_path,
    calls,
    counterexample_claims,
    read,
)

RULE = "auto-mutex-needs-tile-group"

MAKE_TILE = "make_tile"
MAKE_TILE_GROUP = "make_tile_group"
AUTO_MUTEX = "auto_mutex"

_MANUAL_SYNC_PREFIXES = ("sync_", "bar_", "mutex_")
"""Prefixes treated as “manual synchronization” in `pl.system.*`. Match prefixes instead of enumerating names:
`sync_src` / `sync_dst` /
`bar_all` / `bar_vector` / `mutex_lock` / `mutex_unlock` form a family, and missing a new member is worse than accepting
one too many."""


def check(path: Path) -> list[Finding]:
    source = read(path)
    findings: list[Finding] = []
    for function in _auto_mutex_kernels(source):
        found = _bare_tiles_without_sync(source, function)
        findings += apply_claim(
            source, function, RULE, found, counterexample_claims(function)
        )
    return findings


def _auto_mutex_kernels(source: Source) -> list[ast.FunctionDef]:
    """Functions where automatic mutex is enabled, sorted by line number.

    Nested definitions count too: example kernels often live inside factory functions (as do the four pad-value variants
    in `dma_pad`),
    Looking only at module-level definitions would miss them—the first case this rule caught was a factory-created
    kernel.
    """
    found = [
        node
        for node in ast.walk(source.tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _auto_mutex_decorator(node, source.pl) is not None
    ]
    return sorted(found, key=lambda fn: fn.lineno)


def _auto_mutex_decorator(fn: ast.AST, alias: str | None) -> ast.AST | None:
    """Return the jit decorator when auto_mutex is defaulted or explicitly True."""
    if alias is None or not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    for decorator in fn.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if attribute_path(target) != f"{alias}.jit":
            continue
        if not isinstance(decorator, ast.Call):
            return decorator
        if any(kw.arg is None for kw in decorator.keywords):
            return None
        option = next(
            (kw.value for kw in decorator.keywords if kw.arg == AUTO_MUTEX), None
        )
        if option is None or (
            isinstance(option, ast.Constant) and option.value is True
        ):
            return decorator
    return None


def _bare_tiles_without_sync(
    source: Source, function: ast.FunctionDef
) -> list[Finding]:
    bare = [call for call in calls(function, source.pl) if call.name == MAKE_TILE]
    if not bare:
        return []
    if _has_manual_sync(source, function):
        return []
    grouped = [
        call for call in calls(function, source.pl) if call.name == MAKE_TILE_GROUP
    ]
    extra = (
        "；同一个函数里另有 make_tile_group，但组外的 tile 不会因此被同步"
        if grouped
        else ""
    )
    return [
        Finding(
            source.path,
            call.lineno,
            RULE,
            f"`auto_mutex=True` 推不出裸 `{source.pl}.{MAKE_TILE}` 的依赖"
            f"（要 `{MAKE_TILE_GROUP}` 的 `mutex_ids`），"
            f"解析后一条同步都不会插，结果时序相关且不报错{extra}",
        )
        for call in bare
    ]


def _has_manual_sync(source: Source, function: ast.FunctionDef) -> bool:
    system = None if source.pl is None else f"{source.pl}.system"
    return any(
        call.name.startswith(_MANUAL_SYNC_PREFIXES) for call in calls(function, system)
    )
