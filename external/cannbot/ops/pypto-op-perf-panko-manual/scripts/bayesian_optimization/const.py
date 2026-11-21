# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Which integer constants the deterministic core may tune, and where they sit.

`apply` tunes tile-call arguments. A constant that sets a loop trip count is
invisible to it, so the model has to sweep that number by hand, one structural
delta and one full evaluation per point, for what is a single BO dimension.

Selection is two-staged and the order matters. See
`references/tunable_constants.md` for the reasoning and the calibration.

  1. Read DIRECTLY inside a @jit annotation or a torch.* call -> part of the
     operator's declared interface. Excluded.
  2. Of the rest, admit only what reaches a `pypto.loop` trip count, directly or
     through a derivation. Reaching a loop bound means the constant sets an
     iteration count, which is what a tiling parameter does and what a declared
     extent does not.
  3. Everything else is left alone. The rule is default-deny.

Derivation is followed in stage 2 and NOT in stage 1: a constant typically
reaches its loop bound one derivation removed, while a constant whose derived
name reaches an annotation is not itself part of the interface.

Taint is propagated by name and ignores scope, so a name reused in two functions
over-taints. For stage 2 that is the conservative direction only in the sense of
admitting more; the incumbent-value check and the golden are what bound the cost.
"""

import ast

_LOOP = "loop"


def _fname(func):
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _is_jit(fn):
    for d in getattr(fn, "decorator_list", []):
        node = d.func if isinstance(d, ast.Call) else d
        if _fname(node) == "jit":
            return True
    return False


def _int_lit(n):
    return (
        isinstance(n, ast.Constant)
        and isinstance(n.value, int)
        and not isinstance(n.value, bool)
    )


def _offsets(src):
    starts, pos = [], 0
    for line in src.splitlines(keepends=True):
        starts.append(pos)
        pos += len(line)
    return starts


def _span(starts, node):
    return (
        starts[node.lineno - 1] + node.col_offset,
        starts[node.end_lineno - 1] + node.end_col_offset,
    )


def _assign_pairs(node):
    """(name node, value node) pairs this assignment binds to integer literals."""
    if _int_lit(node.value):
        return [(t, node.value) for t in node.targets if isinstance(t, ast.Name)]
    if not isinstance(node.value, ast.Tuple):
        return []
    pairs = []
    for t in node.targets:
        if isinstance(t, ast.Tuple) and len(t.elts) == len(node.value.elts):
            pairs += [
                (a, b)
                for a, b in zip(t.elts, node.value.elts)
                if isinstance(a, ast.Name) and _int_lit(b)
            ]
    return pairs


def bindings(tree, starts):
    """`NAME = <int literal>` at any scope -> {name: (value, start, end)}.

    First binding wins. A name bound twice is not a single site and tuning it
    would move only one of the two, so later bindings are dropped rather than
    guessed at.
    """
    out, seen = {}, set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for t, v in _assign_pairs(node):
            if t.id in seen:
                out.pop(t.id, None)
                continue
            seen.add(t.id)
            out[t.id] = (v.value, *_span(starts, v))
    return out


def _parents(tree):
    out = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            out[child] = node
    return out


def _contexts(node, parents, jit_nodes):
    """The contexts a Name read sits inside."""
    flags, cur = set(), node
    while cur in parents:
        cur = parents[cur]
        if cur in jit_nodes:
            flags.add("graph")
        if isinstance(cur, (ast.arg, ast.AnnAssign)):
            flags.add("annot")
        if isinstance(cur, ast.Call):
            n = _fname(cur.func)
            if n == _LOOP:
                flags.add("loop")
            elif (
                isinstance(cur.func, ast.Attribute)
                and isinstance(cur.func.value, ast.Name)
                and cur.func.value.id == "torch"
            ):
                flags.add("torch")
    return flags


def _names_bound(node):
    """Every name this assignment binds."""
    out = set()
    for t in node.targets:
        for nm in ast.walk(t):
            if isinstance(nm, ast.Name):
                out.add(nm.id)
    return out


def _reads_tainted(node, tainted):
    """Does this assignment read a name the root constant reaches?"""
    return any(
        isinstance(n, ast.Name) and n.id in tainted for n in ast.walk(node.value)
    )


def _taint(tree, root):
    """Every name reachable from `root` through assignments, to a fixed point."""
    tainted, changed = {root}, True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not _reads_tainted(node, tainted):
                continue
            fresh = _names_bound(node) - tainted
            tainted |= fresh
            changed = changed or bool(fresh)
    return tainted


def analyse(src):
    """{name: (value, start, end, direct_flags, via_flags)} for every int constant."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {}
    starts = _offsets(src)
    binds = bindings(tree, starts)
    if not binds:
        return {}
    parents = _parents(tree)
    jit_nodes = set()
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_jit(fn):
            jit_nodes.update(ast.walk(fn))
    reads = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    ]

    out = {}
    for name, (val, s, e) in binds.items():
        via_names = _taint(tree, name) - {name}
        direct, via = set(), set()
        for n in reads:
            if n.id == name:
                direct |= _contexts(n, parents, jit_nodes)
            elif n.id in via_names:
                via |= _contexts(n, parents, jit_nodes)
        out[name] = (val, s, e, direct, via - direct)
    return out


def admitted(src):
    """{name: (value, start, end)} for the constants the core may tune."""
    out = {}
    for name, (val, s, e, direct, via) in analyse(src).items():
        if "annot" in direct or "torch" in direct:
            continue
        if "loop" in direct or "loop" in via:
            out[name] = (val, s, e)
    return out
