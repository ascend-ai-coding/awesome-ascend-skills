# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Admissibility of a kernel normalisation, decided statically.

Normalisation is the INIT step that turns a tile call written with named
constants into one written with integer literals:

    TILE_B, TILE_D = 16, 64
    pypto.set_vec_tile_shapes(TILE_B, TILE_D)   ->   pypto.set_vec_tile_shapes(16, 64)

`apply` tunes a tile call only when every argument is an integer literal --
correctly, because a name like `TILE_B` also drives the view's extent, the loop
bound and the host wrapper's padding, so rewriting the call alone would desync
them. The consequence is that whether the deterministic core owns the tile at all
depends on how the generated kernel happened to spell one call. In one campaign
a kernel that used literals had its tiles searched, while a kernel that used
names produced zero trials.

The rewrite itself is a semantic judgement -- which uses of `TILE_B` are the
view's extent and which are the tile's -- so a model writes it. This module is
the deterministic half: it decides whether what the model wrote is admissible.

The line, and it is the one claim the paper has to defend:

    normalisation exposes parameters and does not change values.

That is checkable here, before any device time is spent. Resolve the symbolic
call's arguments against the module's constant bindings and compare them to the
literals the model wrote. Equal, and no value moved. Different, and the model
optimised while it was refactoring -- which is a different thing, has to earn its
place through the search, and must not be smuggled into the baseline the whole
run is measured against.

What this module deliberately does NOT decide: whether the rewrite preserves
semantics. Constant folding is not enough to know that, and the golden test
already answers it exactly. Two gates, each doing the thing it can actually do.
"""

import ast

from . import apply

_TILE_FUNCS = ("set_cube_tile_shapes", "set_vec_tile_shapes")


def _func_name(func):
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _int_pairs(target, value):
    """(name, int) pairs bound by one target of a simple assignment."""
    if isinstance(target, ast.Name) and _is_int(value):
        return [(target.id, value.value)]
    if not (
        isinstance(target, ast.Tuple)
        and isinstance(value, ast.Tuple)
        and len(target.elts) == len(value.elts)
    ):
        return []
    return [
        (t.id, v.value)
        for t, v in zip(target.elts, value.elts)
        if isinstance(t, ast.Name) and _is_int(v)
    ]


def constant_env(src):
    """Module-level integer bindings, as {name: value}.

    Only literal integers at module scope, and only from simple assignments --
    `TILE_B = 16` and `TILE_B, TILE_D = 16, 64`. A binding whose right-hand side
    is an expression (`TILE_D = D // 8`) is left out rather than evaluated: this
    module exists to prove a value did not change, and a proof that depends on
    reimplementing Python's arithmetic is not one. An unresolvable name simply
    makes the check fail closed, which is the correct outcome -- it means the
    normalisation cannot be verified, not that it is wrong.
    """
    env = {}
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return env
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            env.update(_int_pairs(target, node.value))
    return env


def _is_int(n):
    return (
        isinstance(n, ast.Constant)
        and isinstance(n.value, int)
        and not isinstance(n.value, bool)
    )


def _arg_value(node, env):
    """The integer this argument denotes, or None when it cannot be resolved.

    Literals and names bound to literals only. `TILE_B * 2` resolves to None on
    purpose -- see `constant_env`.
    """
    if _is_int(node):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id)
    return None


def tile_calls(src):
    """[(func_name, [resolved args])] for every tile call, in source order.

    A cube call's arguments are lists, so its entry is a list of lists. An
    argument that cannot be resolved appears as None, which is what makes the
    comparison in `check` fail closed rather than silently pass.
    """
    tree = ast.parse(src)
    env = constant_env(src)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _func_name(node.func)
        if name not in _TILE_FUNCS:
            continue
        args = []
        for a in node.args:
            if isinstance(a, ast.List):
                args.append([_arg_value(e, env) for e in a.elts])
            else:
                args.append(_arg_value(a, env))
        found.append((node.lineno, name, args))
    found.sort()
    return [(name, args) for _, name, args in found]


def tunable_count(src):
    """How many tile sites BO can actually move. 0 means the lever is dead."""
    try:
        return len([s for s in apply.tunable_sites(src) if s.kind in ("cube", "vec")])
    except SyntaxError:
        return 0


def _arg_reason(i, j, before, after):
    """Why argument `j` of tile call `i` is not an admissible normalisation, or None."""
    if before is None:
        return (
            f"tile call {i} arg {j}: original value could not be resolved "
            f"to an integer, so the rewrite cannot be verified. Bind it to "
            f"a module-level integer literal, or leave the call alone"
        )
    if after is None:
        return f"tile call {i} arg {j}: normalised value is still not a literal integer"
    if before != after:
        return (
            f"tile call {i} arg {j}: value changed {before} -> {after}. "
            f"Normalisation exposes parameters; it does not tune them"
        )
    return None


def check(before_src, after_src):
    """Is `after_src` an admissible normalisation of `before_src`?

    Returns {"ok": bool, "reasons": [...], "before": {...}, "after": {...}}.
    Every reason names what failed, because a rejected normalisation has to be
    fixable by reading the message.
    """
    report = {"ok": False, "reasons": []}

    try:
        after_calls = tile_calls(after_src)
    except SyntaxError as e:
        report["reasons"].append(f"normalised source does not parse: {e}")
        return report
    try:
        before_calls = tile_calls(before_src)
    except SyntaxError as e:
        report["reasons"].append(f"original source does not parse: {e}")
        return report

    n_before, n_after = tunable_count(before_src), tunable_count(after_src)
    report["before"] = {"tile_calls": len(before_calls), "tunable": n_before}
    report["after"] = {"tile_calls": len(after_calls), "tunable": n_after}

    # 1. The same calls, in the same order. A normalisation that adds or removes
    #    a tile call has restructured the kernel, which is a search action.
    if len(before_calls) != len(after_calls):
        report["reasons"].append(
            f"tile call count changed: {len(before_calls)} -> {len(after_calls)}; "
            f"normalisation may not add or remove tile calls"
        )
        return report
    for i, ((bn, _), (an, _)) in enumerate(zip(before_calls, after_calls)):
        if bn != an:
            report["reasons"].append(f"tile call {i}: {bn} -> {an}")

    # 2. Every value identical. This is the bright line.
    for i, ((_, ba), (_, aa)) in enumerate(zip(before_calls, after_calls)):
        if len(ba) != len(aa):
            report["reasons"].append(
                f"tile call {i}: arity {len(ba)} -> {len(aa)}; a normalisation "
                f"does not change how many dimensions a tile has"
            )
            continue
        for j, (b, a) in enumerate(zip(ba, aa)):
            reason = _arg_reason(i, j, b, a)
            if reason:
                report["reasons"].append(reason)

    # 3. It has to actually expose something, or it is churn with a device cost.
    if n_after <= n_before:
        report["reasons"].append(
            f"tunable tile sites did not increase ({n_before} -> {n_after}); "
            f"the lever can see no more than it could before"
        )

    report["ok"] = not report["reasons"]
    return report
