# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Is this candidate the incumbent, spelled differently?

A run can keep a single candidate out of a whole budget, and have it be this:

    for _ in pypto.loop(N, submit_before_loop=False):     # candidate
    for _ in pypto.loop(N):                               # incumbent

`False` is the parameter's own default. The two programs are the same program.
It is accepted because `J` is a function of the measured latency and this draw
came back lower than the incumbent's -- the same semantics measured three times
can span tens of percent. So the delivered kernel is no longer the byte-sequence
Stage 7 started with, for no reason at all.

The eval cache could not catch it: it keys on `code_hash` over raw bytes, and
these are different bytes.

What this module claims, and it is deliberately narrow
------------------------------------------------------
Passing a keyword argument its own declared default is the same call as omitting
it. That is Python's calling convention, not a claim about the compiler: the
tracer downstream receives an identical value either way, so there is nothing
for it to generate differently. Comments and formatting are handled for free by
comparing ASTs rather than text.

What it deliberately does NOT claim
-----------------------------------
That commutative operands may be reordered. The same run measured
`mul(t4, x_tile)` at 13.82 against `mul(x_tile, t4)` at 9.64 and concluded the
gap was noise -- but concluding it and establishing it are different things, and
canonicalising operand order would assert that this toolchain emits identical
code for both. Nothing here has measured that. A false positive in this module
DELETES a candidate the search would otherwise have tried, which is the
expensive direction to be wrong in, so the module only folds away things whose
equivalence is a property of Python.

`defaults` is supplied by the caller, from `inspect.signature` on the real
callables. Without it the comparison still catches reformatting and comment
edits, and folds nothing -- absent knowledge produces a weaker check, never a
wrong one.
"""

import ast
import inspect

# Only literals are compared. A default that is a mutable or computed object
# (`None` aside) cannot be matched against source text without evaluating it,
# and evaluating candidate source is not something this is willing to do.
_LITERAL = (bool, int, float, str, type(None))


def _literal_defaults(obj):
    """{kwarg: default} for the literal-valued keyword defaults of one callable.

    Empty when the object has no signature to read -- a builtin, a C extension,
    anything `inspect` declines. That is an answer rather than an error: a call
    whose defaults cannot be read simply folds nothing.
    """
    try:
        sig = inspect.signature(obj)
    except (AttributeError, TypeError, ValueError):
        return {}
    kw = {}
    for p in sig.parameters.values():
        if p.default is not inspect.Parameter.empty and isinstance(p.default, _LITERAL):
            kw[p.name] = p.default
    return kw


def defaults_for(*modules):
    """{call name: {kwarg: default}} for every callable the modules expose.

    Keyed by the ATTRIBUTE name, because that is what the AST gives for
    `pypto.loop(...)`. A name defined by two modules keeps the first, which is
    why the caller passes the more specific module first.
    """
    out = {}
    for mod in modules:
        if mod is None:
            continue
        for name in dir(mod):
            if name.startswith("_") or name in out:
                continue
            obj = getattr(mod, name, None)
            if not callable(obj):
                continue
            kw = _literal_defaults(obj)
            if kw:
                out[name] = kw
    return out


def _call_name(node, root=None):
    """The attribute name of a call, but ONLY for calls rooted at `root`.

    `pypto.loop(...)` and `pypto.frontend.jit(...)` qualify; a bare `loop(...)`
    or a `helper.op(...)` does not. Without the root test a local helper that
    happens to share a name with a pypto callable -- `helper.op(flag=False)`
    against pypto's own `op(flag=False)` -- would have its keyword folded on the
    strength of an unrelated signature, which is a false positive of exactly the
    kind this module says it refuses to risk.
    """
    fn = node.func
    if not isinstance(fn, ast.Attribute):
        return ""
    if root is not None:
        base = fn
        while isinstance(base, ast.Attribute):
            base = base.value
        if not (isinstance(base, ast.Name) and base.id == root):
            return ""
    return fn.attr


class _FoldDefaults(ast.NodeTransformer):
    """Drop `k=v` where `v` is the literal `k` already defaults to."""

    def __init__(self, defaults, root):
        self.defaults = defaults
        self.root = root
        self.folded = []

    def visit(self, node):
        """Dispatch by hand, so the handler can carry a Python name.

        `ast.NodeVisitor` dispatches on `"visit_" + type(node).__name__`, which
        forces a CamelCase method. Overriding `visit` keeps the same traversal --
        `generic_visit` still calls back into it for every child -- while the
        handler is named like everything else here.
        """
        if isinstance(node, ast.Call):
            return self._fold_call(node)
        return super().visit(node)

    def _fold_call(self, node):
        self.generic_visit(node)
        name = _call_name(node, self.root)
        known = self.defaults.get(name)
        if not known:
            return node
        keep = []
        for kw in node.keywords:
            named_default = (
                kw.arg is not None
                and kw.arg in known
                and isinstance(kw.value, ast.Constant)
            )
            # `is` on the types too: `False == 0` and `True == 1` in Python, so
            # a plain `==` would fold `run_mode=0` against a default of `False`,
            # which are not the same request.
            restates_it = named_default and (
                type(kw.value.value) is type(known[kw.arg])
                and kw.value.value == known[kw.arg]
            )
            if restates_it:
                self.folded.append(f"{name}({kw.arg}={kw.value.value!r})")
                continue
            keep.append(kw)
        node.keywords = keep
        return node


def canonical(src, defaults=None, root="pypto"):
    """(dump, folded) -- the program's AST with default-valued keywords removed.

    Raises SyntaxError if `src` does not parse; callers treat that as "cannot
    decide", not as "different".
    """
    tree = ast.parse(src)
    f = _FoldDefaults(defaults or {}, root)
    tree = f.visit(tree)
    ast.fix_missing_locations(tree)
    return ast.dump(tree), f.folded


def same_program(candidate_src, incumbent_src, defaults=None, root="pypto"):
    """(is_same, why). `why` names what was folded, for the log.

    False whenever either side fails to parse: an undecidable comparison must
    let the candidate through to be measured, never silently delete it.
    """
    try:
        a, folded = canonical(candidate_src, defaults, root)
        b, _ = canonical(incumbent_src, defaults, root)
    except (SyntaxError, ValueError, RecursionError):
        return False, ""
    if a != b:
        return False, ""
    if folded:
        return True, "restates a default: " + ", ".join(sorted(set(folded)))
    return True, "identical once comments and formatting are removed"
