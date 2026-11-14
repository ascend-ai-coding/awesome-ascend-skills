# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Deterministic PER-SITE config applier: rewrite each tile-shape call site independently.

Per-site is the DEFAULT: different tile calls in a kernel are independent knobs. A kernel may set
one tile for its [BT,D] vec ops and another for its [BT,BT] ops, or use a smaller cube tile in an
unaligned branch — forcing every site to the same tile would both miscompile such kernels and
shrink the search. So the space is built PER discovered site; callers may optionally GROUP sites
that should move together (apply.group_identical) as an efficiency lever, but nothing is tied by
default.

Discovery is AST-based (robust to `pypto.` prefixes, whitespace, nested brackets, multi-arg cube
calls) and doubles as a parse check. Each site gets a stable id `"<kind>#<ordinal>"` in source
order (e.g. cube#0, cube#1, vec#0..vec#4). Rewrites replace only the call's argument list, so the
`pypto.set_*` prefix and everything else in the file are preserved byte-for-byte.

Tunable kinds: "cube" (set_cube_tile_shapes) and "vec" (set_vec_tile_shapes at ANY arity from one
to four integer arguments, exposed as d0..dk). A call whose arguments are not all integer literals
is "vec_dyn" / "cube_dyn": DISCOVERED but left untouched, because rewriting `set_vec_tile_shapes(M,
K)` with numbers would change the program's meaning rather than its tiling.

Materialize-absent: BO must explore the FULL numeric space, not only the literals the Stage-6
generator happened to emit. So discovery ALSO surfaces universally-insertable numeric knobs that are
MISSING — the runtime_options scalars (device_sched_mode / stitch_function_max_num) absent from a
`@jit(runtime_options={...})`, and `unroll_list` absent from a `pypto.loop(...)`. These become
zero-width "insertion" sites (start == end). Their current value is the sentinel UNSET (= leave it
out), which is always in the domain, so the original kernel is a reachable point and BO can also try
inserting the knob at real values — every candidate still gated by E(x), so an unsupported insert is
simply dropped. Absent CUBE/VEC tiles are not materialized: there is no deterministic place to insert
a tile the generator never wrote (that is a structural decision, left to PANKO).
"""

import ast
import re
from dataclasses import dataclass, field
from collections import namedtuple

from . import const, space

# `loop_depth` = how many enclosing loops the call sits in. It is the cheapest
# available proxy for "how much work flows through this site", and it separates
# the two cases that matter: a tile that governs the loop body, and one that
# runs once on the way in. Defaults to 0 so a Site built positionally by older
# callers still constructs.
# `trip` is the loop's iteration count when `pypto.loop(<int literal>, ...)`
# writes one, else None. master's rule for unroll is "the largest value must not
# exceed the trip count", and this is the only place that number exists.
Site = namedtuple(
    "Site",
    "id kind ordinal func_text start end loop_depth loop_span trip",
    defaults=(0, None, None),
)

_TILE_FUNCS = ("set_cube_tile_shapes", "set_vec_tile_shapes")
_TUNABLE = ("cube", "vec", "global", "unroll", "pass", "const")
MAX_VEC_ARITY = 4  # set_vec_tile_shapes(d0[, d1[, d2[, d3]]])
# How a jit decorator can be spelled at the call site.
_JIT_NAMES = ("jit", "frontend.jit", "pypto.frontend.jit")
_RUNTIME_KNOBS = tuple(
    space.GLOBAL_DOMAINS
)  # runtime_options keys BO tunes (single source: space)
_PASS_KNOBS = tuple(
    space.PASS_DOMAINS
)  # pass_options keys BO tunes (compiler merge passes)
UNSET = "__unset__"  # absent-knob sentinel: "do not write this knob"


def _func_name(func):
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _int_const(n):
    return isinstance(n, ast.Constant) and isinstance(n.value, int)


def _signed_int(n):
    """Value of an integer literal, sign included, or None if `n` is not one.

    `-1` is not an `ast.Constant`: the parser builds `UnaryOp(USub, Constant(1))`,
    because the minus is an operator applied to a literal. `_int_const` therefore
    rejects it, and `{-1: 4}` -- the standard shape of every pass knob the
    reference writes, and the one every recorded kernel carries -- fell through
    to `p_named` and was never offered as a tunable site. The knob was also
    marked SET, so no insertion site was offered either: an existing pass
    configuration was simply invisible to the search.
    """
    if isinstance(n, ast.Constant) and isinstance(n.value, int):
        return n.value
    if not (isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd))):
        return None
    signed_literal = isinstance(n.operand, ast.Constant) and isinstance(
        n.operand.value, int
    )
    if not signed_literal:
        return None
    return -n.operand.value if isinstance(n.op, ast.USub) else n.operand.value


def _int_list(n):
    """[<int>, <int>, ...] — a tile-dim list of integer literals."""
    return (
        isinstance(n, ast.List) and bool(n.elts) and all(_int_const(e) for e in n.elts)
    )


def _line_starts(src):
    """(lineno, col) -> absolute offset. Tile-call lines are ASCII, so char==byte offset holds."""
    starts = [0]
    for line in src.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    return lambda ln, col: starts[ln - 1] + col


def _func_spans(tree, off):
    """{function name: byte span of its body}, for every `def` in the module."""
    spans = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.body:
            spans[n.name] = (
                off(n.body[0].lineno, n.body[0].col_offset),
                off(n.body[-1].end_lineno, n.body[-1].end_col_offset),
            )
    return spans


def _enclosing_func(spans, pos):
    """Name of the innermost function whose body contains `pos`, or None."""
    best, width = None, None
    for name, (a, b) in spans.items():
        if a <= pos < b and (width is None or b - a < width):
            best, width = name, b - a
    return best


def _inherited_depth(tree, off, loops, spans):
    """{function name: the loop depth its body inherits from its callers}.

    Lexical depth alone answers "is this tile written inside a `for`", but the
    filter in `tunable_sites` reads it as "does this tile run more than once",
    and those two differ the moment a kernel factors its inner work into a
    helper. An attention kernel commonly does exactly that: its matmuls and its
    softmax live in stage helpers called from inside the loop, so they read
    depth 0 while placeholder tiles written inline in the loop body read depth
    2 -- and the filter then drops precisely the sites that carry the kernel,
    leaving BO the ones that do not. Blocks charge device evaluations for no
    feasible candidate, while the real wins sit on sites the lever cannot see.

    Following the call graph is what makes the number mean what the filter
    reads. Depth is the MAXIMUM over call paths, because the safe direction is
    to keep a site: keeping a cold one costs the sampler a dimension, dropping a
    hot one costs the whole lever.
    """
    edges = {}  # callee -> [(caller or None, lexical depth of the call)]
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        name = _func_name(n.func)
        if name not in spans:
            continue
        pos = off(n.lineno, n.col_offset)
        edges.setdefault(name, []).append(
            (_enclosing_func(spans, pos), sum(1 for a, b in loops if a <= pos < b))
        )

    def depth(name, seen):
        # A cycle contributes no bound we can read, so it stops here rather than
        # recursing. Kernels are not recursive; this only keeps the walk total.
        if name in seen:
            return 0
        seen = seen | {name}
        return max(
            (d + (depth(c, seen) if c else 0) for c, d in edges.get(name, ())),
            default=0,
        )

    return {name: depth(name, frozenset()) for name in spans}


@dataclass
class _Found:
    """What one pass over the AST turned up, before any of it becomes a Site.

    Each list is a different thing the rewriter can do, and they are collected
    together because they come from the same walk: a tile call to sweep, a knob
    whose value can be replaced, a knob missing from a dict that exists, a kwarg
    that has to be written whole.
    """

    tiles: list = field(
        default_factory=list
    )  # (start, end, func_text, kind, depth, span)
    g_present: list = field(
        default_factory=list
    )  # runtime_options scalar: (vstart, vend, knob)
    g_absent: list = field(
        default_factory=list
    )  # runtime_options knob missing: (pos, knob)
    u_present: list = field(
        default_factory=list
    )  # unroll_list=[...]: (start, end, trip)
    u_absent: list = field(
        default_factory=list
    )  # loop with no unroll_list: (pos, trip)
    p_present: list = field(
        default_factory=list
    )  # pass_options {-1: N}: (vstart, vend, knob)
    p_in_dict: list = field(
        default_factory=list
    )  # dict exists, knob missing: (pos, knob)
    p_no_kwarg: list = field(default_factory=list)  # no pass_options kwarg: (pos, knob)
    p_named: set = field(default_factory=set)  # knob present in a form we do not sweep


def _tile_kind(node, name):
    """Which of the four tile kinds this call is, tunable or not.

    Only literal-arg tiles are tunable; one driven by variables (`[M, K]`) must
    NOT be swept, because rewriting it with numbers would change what the kernel
    computes. Those become `*_dyn`.
    """
    if name == "set_cube_tile_shapes":
        # Tunable only in the standard m/k/n form: exactly three axis lists, each
        # a 2-element [L0, L1] integer pair. A 3-element k axis ([kL0, kAL1,
        # kBL1]) or a split_k variant is left as cube_dyn rather than mis-parsed
        # -- the six-knob reader assumes two values per axis.
        lit = len(node.args) >= 3 and all(
            isinstance(a, ast.List) and len(a.elts) == 2 and _int_list(a)
            for a in node.args[:3]
        )
        return "cube" if lit else "cube_dyn"
    if 1 <= len(node.args) <= MAX_VEC_ARITY and all(_int_const(a) for a in node.args):
        # Any arity from one to four. The 2-argument form was the only tunable one
        # until now, which left two shapes untouched that the recorded runs use
        # constantly: the 1-argument `set_vec_tile_shapes(TILE)`, where the
        # distance between a small and a large extent is enormous, and the
        # 3-argument form. How many knobs a site has is a property of the
        # kernel.
        return "vec"
    return "vec_dyn"


def _collect_tile(node, name, ctx, found):
    off, src, loops, inherited, fspans = ctx
    cstart = off(node.lineno, node.col_offset)
    cend = off(node.end_lineno, node.end_col_offset)
    fstart = off(node.func.lineno, node.func.col_offset)
    fend = off(node.func.end_lineno, node.func.end_col_offset)
    enclosing = [(a, b) for a, b in loops if a <= cstart < b]
    # A tile inside a helper runs as often as the helper is called, so the depth
    # a site reports is its own lexical depth plus whatever its function inherits
    # from the call graph.
    depth = len(enclosing) + inherited.get(_enclosing_func(fspans, cstart), 0)
    # The INNERMOST enclosing loop body, as a byte span. Two tile calls that share
    # it run in the same loop body and hand tiles to each other; that is the
    # relation `tie_groups` needs and nothing else in the parser records it.
    span = min(enclosing, key=lambda ab: ab[1] - ab[0]) if enclosing else None
    found.tiles.append(
        (cstart, cend, src[fstart:fend], _tile_kind(node, name), depth, span)
    )


def _collect_runtime_options(kw, off, found):
    present = set()
    for k, v in zip(kw.value.keys, kw.value.values):
        if not (isinstance(k, ast.Constant) and k.value in _RUNTIME_KNOBS):
            continue
        present.add(k.value)
        if isinstance(v, ast.Constant) and isinstance(v.value, int):
            found.g_present.append(
                (
                    off(v.lineno, v.col_offset),
                    off(v.end_lineno, v.end_col_offset),
                    k.value,
                )
            )
    insert = off(kw.value.end_lineno, kw.value.end_col_offset) - 1  # just before "}"
    for knob in _RUNTIME_KNOBS:
        if knob not in present:
            found.g_absent.append((insert, knob))


def _collect_pass_options(kw, off, found):
    present = set()
    for kk, vv in zip(kw.value.keys, kw.value.values):
        if not (isinstance(kk, ast.Constant) and kk.value in _PASS_KNOBS):
            continue
        present.add(kk.value)
        # Only the integer-key form `{-1: N}` is tunable. A dict with string keys
        # names individual hashOrders, which is a structural choice about WHICH
        # subgraphs to merge; sweeping a scalar over it would be meaningless, and
        # the reference forbids mixing the two forms in one dict anyway.
        one_entry = isinstance(vv, ast.Dict) and len(vv.keys) == 1
        both_ints = one_entry and (
            _signed_int(vv.keys[0]) is not None
            and _signed_int(vv.values[0]) is not None
        )
        if both_ints:
            found.p_present.append(
                (
                    off(vv.lineno, vv.col_offset),
                    off(vv.end_lineno, vv.end_col_offset),
                    kk.value,
                )
            )
        else:
            found.p_named.add(kk.value)
    insert = off(kw.value.end_lineno, kw.value.end_col_offset) - 1
    for knob in _PASS_KNOBS:
        if knob not in present:
            found.p_in_dict.append((insert, knob))


def _is_unroll_list(kw):
    return (
        kw.arg == "unroll_list"
        and isinstance(kw.value, ast.List)
        and all(
            isinstance(e, ast.Constant) and isinstance(e.value, int)
            for e in kw.value.elts
        )
    )


def _collect_keywords(node, off, trip, found):
    """(has_pass_options, loop_has_unroll) for one call's keywords."""
    has_pass_options = loop_has_unroll = False
    for kw in node.keywords:
        # global JIT knobs, carried in the runtime_options dict -- e.g. the
        # device_sched_mode entry
        if (
            kw.arg == "runtime_options"
            and isinstance(kw.value, ast.Dict)
            and kw.value.keys
        ):
            _collect_runtime_options(kw, off, found)
        elif kw.arg == "pass_options" and isinstance(kw.value, ast.Dict):
            _collect_pass_options(kw, off, found)
            has_pass_options = True
        # loop unroll schedule: the unroll_list keyword on a pypto loop
        elif _is_unroll_list(kw):
            found.u_present.append(
                (
                    off(kw.value.lineno, kw.value.col_offset),
                    off(kw.value.end_lineno, kw.value.end_col_offset),
                    trip,
                )
            )
            loop_has_unroll = True
    return has_pass_options, loop_has_unroll


def _literal_trip(node):
    """A `pypto.loop`'s trip count when its first argument is an integer literal."""
    if not node.args or not isinstance(node.args[0], ast.Constant):
        return None
    value = node.args[0].value
    return value if isinstance(value, int) else None


def _walk_calls(tree, ctx):
    """One pass over every call, gathering everything a rewrite could act on."""
    off = ctx[0]
    found = _Found()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _func_name(node.func)
        if name in _TILE_FUNCS:
            _collect_tile(node, name, ctx, found)
        is_loop = name == "loop"
        trip = _literal_trip(node) if is_loop else None
        has_pass_options, loop_has_unroll = _collect_keywords(node, off, trip, found)
        takes_arguments = bool(node.args or node.keywords)
        if is_loop and not loop_has_unroll and takes_arguments:
            found.u_absent.append((off(node.end_lineno, node.end_col_offset) - 1, trip))
        if name in _JIT_NAMES and not has_pass_options and node.keywords:
            # The whole kwarg has to be written, not just a key. Every recorded
            # kernel is in this state: `@jit(runtime_options={...})` and nothing
            # else, so a search that only edits an existing dict can never reach
            # these passes.
            for knob in _PASS_KNOBS:
                found.p_no_kwarg.append(
                    (off(node.end_lineno, node.end_col_offset) - 1, knob)
                )
    return found


def _tile_sites(found):
    counters, sites = {}, []
    for cstart, cend, func_text, kind, depth, span in sorted(found.tiles):
        i = counters.get(kind, 0)
        counters[kind] = i + 1
        sites.append(
            Site(
                id=f"{kind}#{i}",
                kind=kind,
                ordinal=i,
                func_text=func_text,
                start=cstart,
                end=cend,
                loop_depth=depth,
                loop_span=span,
            )
        )
    return sites


def _pass_sites(found):
    """`func_text` carries HOW to write the site, because a pass knob has three
    shapes: replace the value, add a key to an existing dict, or write the whole
    `pass_options=` kwarg. One knob is offered in at most one state -- a value
    that exists is not also "missing", and that includes every knob the dict
    already mentions in a form we do not sweep. A `{"func8_0": 4}` names one
    hashOrder and is not swept, but it is still SET: offering an insertion site
    for it would write the key a second time.
    """
    have = {knob for _, _, knob in found.p_present} | found.p_named
    pall = (
        [(vs, ve, knob, "") for (vs, ve, knob) in found.p_present]
        + [
            (pos, pos, knob, ", ")
            for (pos, knob) in found.p_in_dict
            if knob not in have
        ]
        + [
            (pos, pos, knob, ", pass_options={")
            for (pos, knob) in found.p_no_kwarg
            if knob not in have
        ]
    )
    count, sites = {}, []
    for start, end, knob, prefix in sorted(pall):
        i = count.get(knob, 0)
        count[knob] = i + 1
        sites.append(
            Site(
                id=f"pass:{knob}#{i}",
                kind="pass",
                ordinal=i,
                func_text=prefix,
                start=start,
                end=end,
            )
        )
    return sites


def _global_sites(found):
    """A present knob gets the span of its value and an empty func_text; an absent
    one gets a zero-width insertion point whose func_text is the comma and space
    that will precede it. Ordinal per knob, by source position.
    """
    gall = [(vs, ve, knob, "") for (vs, ve, knob) in found.g_present] + [
        (p, p, knob, ", ") for (p, knob) in found.g_absent
    ]
    count, sites = {}, []
    for start, end, knob, prefix in sorted(gall):
        i = count.get(knob, 0)
        count[knob] = i + 1
        sites.append(
            Site(
                id=f"g:{knob}#{i}",
                kind="global",
                ordinal=i,
                func_text=prefix,
                start=start,
                end=end,
            )
        )
    return sites


def _unroll_sites(found):
    uall = [(ls, le, "", t) for (ls, le, t) in found.u_present] + [
        (p, p, ", ", t) for (p, t) in found.u_absent
    ]
    return [
        Site(
            id=f"unroll#{i}",
            kind="unroll",
            ordinal=i,
            func_text=prefix,
            start=start,
            end=end,
            trip=trip,
        )
        for i, (start, end, prefix, trip) in enumerate(sorted(uall))
    ]


def _const_sites(src):
    """Integer constants that set a loop trip count. Selection is const's, and is
    default-deny: a constant reaching the operator's declared interface is
    excluded, and only what reaches a loop bound is admitted.
    """
    sites = []
    admitted = sorted(const.admitted(src).items())
    for i, (name, (_v, start, end)) in enumerate(admitted):
        sites.append(
            Site(
                id=f"const:{name}",
                kind="const",
                ordinal=i,
                func_text="",
                start=start,
                end=end,
            )
        )
    return sites


def discover_sites(src):
    """Return all tile-call sites (cube / vec / vec3) in source order, each with a stable id.

    Raises SyntaxError if src doesn't parse (a useful guard before spending an evaluation).
    """
    tree = ast.parse(src)
    off = _line_starts(src)
    # Byte spans of every loop body. Both kinds count: `for idx in pypto.loop(...)`
    # and the plain `for n_off in range(0, N, NC)` a model writes to chunk an axis --
    # the second is a Python loop unrolled at trace time, and a tile inside it still
    # governs every chunk.
    loops = []
    for n in ast.walk(tree):
        if not isinstance(n, (ast.For, ast.AsyncFor, ast.While)) or not n.body:
            continue
        loops.append(
            (
                off(n.body[0].lineno, n.body[0].col_offset),
                off(n.body[-1].end_lineno, n.body[-1].end_col_offset),
            )
        )
    fspans = _func_spans(tree, off)
    ctx = (off, src, loops, _inherited_depth(tree, off, loops, fspans), fspans)
    found = _walk_calls(tree, ctx)
    return (
        _tile_sites(found)
        + _pass_sites(found)
        + _global_sites(found)
        + _unroll_sites(found)
        + _const_sites(src)
    )


def tie_groups(sites, anchor):
    """Vec sites whose LEADING tile dimension must move together, as id lists.

    The row extent is a hand-off, not a per-site choice. Several vec scopes
    inside one loop body pass tiles to each other; give two of them different
    leading extents and every iteration pays a re-block between them. A mixed
    set of leading extents is reliably among the worst configurations of a run,
    and the cost tracks the mixture rather than the row count: holding the tile
    width fixed and varying only coherence is enough to produce it.

    Two restrictions:

    * only sites with arity >= 2 are tied. A `[rows, 1]` reduction-output leg is
      a single-argument call and is free to take its own extent; tying it to the
      wide sites can delete the best point.
    * only the LEADING field is tied, never the whole site. The claim is about
      the row extent; nothing establishes that two wide sites want the same
      width, and tying `d1` as well would remove configurations on a claim
      nothing supports.
    """
    by_span = {}
    for s in sites:
        if s.kind != "vec" or s.loop_span is None:
            continue
        # The anchor is what declares a site's arity. Without an entry for this
        # site there is no arity to read, and `vec_fields({})` falls back to two
        # dimensions -- which would tie the single-argument reduction legs to the
        # wide scopes and delete the best configuration this kernel has. Unknown
        # arity means do not tie.
        a = (anchor or {}).get(s.id)
        if not a or len(space.vec_fields(a)) < 2:
            continue
        by_span.setdefault(s.loop_span, []).append(s.id)
    return [g for g in by_span.values() if len(g) > 1]


def tunable_sites(src):
    """Sites BO can tune, with the once-only ones dropped.

    A tile call outside every loop runs once. Tuning it cannot move a latency
    that a loop body dominates, but it costs the sampler the same number of
    dimensions as the site that does -- and TPE spends its draws on all of
    them equally.

    A kernel typically carries both kinds: a site that sets the tile for a
    cast executed once before the loop, and a site that sets the tile for the
    chain inside it, executed many times over far larger tensors. The work
    ratio between the two can run to several orders of magnitude, and the
    sampler draws both on every trial -- so a large share of the vector search
    is spent on a tile that cannot move the latency.

    Dropped only when something deeper exists. A kernel written without loops
    has all its sites at depth 0, and those are the only sites it has.
    """
    sites = [s for s in discover_sites(src) if s.kind in _TUNABLE]
    # Loop depth separates a TILE that governs a loop body from one that runs
    # once on the way in. It says nothing about the others: a `runtime_options`
    # or `pass_options` knob lives on the decorator and an `unroll_list` on the
    # loop header, so all three are depth 0 by construction. Filtering them the
    # same way silently switched off the global, unroll and pass knobs on every
    # kernel that has a loop -- which is every kernel that matters.
    tiles = [s for s in sites if s.kind in ("cube", "vec")]
    if any(s.loop_depth > 0 for s in tiles):
        keep = {id(s) for s in tiles if s.loop_depth == 0}
        sites = [s for s in sites if id(s) not in keep]
    return sites


def _cube_args(c):
    # pypto.set_cube_tile_shapes(m, k, n) takes the m / k / n AXES, each a
    # [L0, L1] pair (confirmed against the installed pypto: the docstring example
    # `set_cube_tile_shapes([16,16],[256,512],[128,128])` round-trips as
    # m=[16,16], k=[256,512], n=[128,128]). An earlier version read these three
    # lists as operand buffers [mL0,kL0],[kL0,nL0],[mL0,nL0]; that mis-parsed the
    # tile, wrote back malformed pairs that pypto rejected as `Invalid L1/L0
    # relation` (the third list [mL0,nL0] became n=[nL0=mL0, nL1=nL0]), and never
    # tuned L1 at all -- the axis the hand-optimised 3768us kernel wins on.
    return (
        f"[{c['mL0']}, {c['mL1']}], [{c['kL0']}, {c['kL1']}], [{c['nL0']}, {c['nL1']}]"
    )


def _vec_args(v):
    return ", ".join(str(v[f]) for f in space.vec_fields(v))


# `_site_text` returns one of these instead of replacement text: `_SKIP` for a
# knob the config leaves UNSET (which stays out of the program), `_MERGED` for a
# pass knob folded into a shared `pass_options=` insertion.
_SKIP = object()
_MERGED = object()


def _pass_text(s, cfg, absent, kwarg_inserts):
    """A pass knob: `{-1: N}` in place, a new key in an existing dict, or merged
    into one `pass_options=` insertion for a decorator that carries none."""
    v = cfg.get("value", UNSET)
    if v == UNSET:
        return _SKIP
    knob = space.pass_knob(s.id)
    body = f"{{-1: {v}}}"
    if not absent:
        return body  # replace {-1: N}
    if s.func_text.startswith(", pass_options={"):
        kwarg_inserts.setdefault(s.start, {})[knob] = v  # merged by the caller
        return _MERGED
    return f'{s.func_text}"{knob}": {body}'  # key into the dict


def _global_text(s, cfg, absent):
    """A global knob, left out entirely when UNSET -- the original,
    always-reachable point."""
    v = cfg["value"]
    if v == UNSET:
        return _SKIP
    return f'{s.func_text}"{space.global_knob(s.id)}": {v}' if absent else str(v)


def _unroll_text(s, cfg, absent):
    """An unroll list, as the list literal or as the whole keyword."""
    v = cfg["value"]
    if v == UNSET:
        return _SKIP
    arr = "[" + ", ".join(x.strip() for x in v.split(",")) + "]"
    return f"{s.func_text}unroll_list={arr}" if absent else arr


def _site_text(s, cfg, kwarg_inserts):
    """The replacement text for one site, dispatched on its kind."""
    absent = (
        s.start == s.end
    )  # zero-width -> INSERT a missing knob (func_text = ", " prefix)
    if s.kind == "cube":
        return f"{s.func_text}({_cube_args(cfg)})"
    if s.kind == "vec":
        return f"{s.func_text}({_vec_args(cfg)})"
    if s.kind == "pass":
        return _pass_text(s, cfg, absent, kwarg_inserts)
    if s.kind == "global":
        return _global_text(s, cfg, absent)
    if s.kind == "const":
        return str(cfg["value"])
    return _unroll_text(s, cfg, absent)


def apply(src, site_configs):
    """Rewrite each site named in `site_configs` with its own tile. Returns (new_src, n_applied).

    Sites absent from site_configs (e.g. vec3, or a site the caller chose not to tune) are left
    untouched. Edits are applied right-to-left so earlier spans keep their offsets.

    Returns the number of SITES written, which is not the number of edits: every
    pass knob on a decorator that carries no `pass_options` at all shares one
    insertion offset, and they are merged into a single `pass_options={...}`.
    """
    edits = []
    # insertion offset -> {knob: value}. One decorator, one kwarg. Emitting an
    # edit per knob wrote `pass_options=` twice and the candidate did not even
    # parse ("keyword argument repeated"), so with both knobs in the domain the
    # search could produce a program no compiler would ever see.
    kwarg_inserts = {}
    n_sites = 0
    for s in discover_sites(src):
        if s.id not in site_configs or s.kind not in _TUNABLE:
            continue
        text = _site_text(s, site_configs[s.id], kwarg_inserts)
        if text is _SKIP:
            continue
        n_sites += 1
        if text is not _MERGED:
            edits.append((s.start, s.end, text))
    # Sorted by knob so the same selection always produces the same bytes; the
    # code hash is the search's identity for a candidate.
    for pos, knobs in kwarg_inserts.items():
        body = ", ".join(f'"{k}": {{-1: {v}}}' for k, v in sorted(knobs.items()))
        edits.append((pos, pos, f", pass_options={{{body}}}"))
    for start, end, text in sorted(edits, reverse=True):
        src = src[:start] + text + src[end:]
    return src, n_sites


def _read_site(kind, nums):
    """One site's current values, as the config dict its kind uses."""
    if kind == "cube":
        # Six values in source order: m=[nums0,nums1], k=[nums2,nums3],
        # n=[nums4,nums5]. The first of each pair is L0, the second L1.
        return {
            "mL0": nums[0],
            "mL1": nums[1],
            "kL0": nums[2],
            "kL1": nums[3],
            "nL0": nums[4],
            "nL1": nums[5],
        }
    if kind in ("global", "const"):
        return {"value": nums[0]}
    if kind == "pass":
        # The span is the inner dict `{-1: N}`, so the first number is the -1 key
        # and the LAST is the value. Reading nums[0] here fed the warm start a -1;
        # it never showed because the signed-literal bug meant no pass site with a
        # present value was ever discovered.
        return {"value": nums[-1]}
    if kind == "unroll":
        return {"value": ",".join(str(x) for x in nums)}
    # d0..dk, one per argument. The arity is read off the call, so a 1-argument
    # site has one knob and a 4-argument site has four.
    return {f"d{i}": n for i, n in enumerate(nums)}


def current_config(src):
    """Read each tunable site's CURRENT tile values into a per-site config (BO warm-start seed).

    cube call [mL0,mL1],[kL0,kL1],[nL0,nL1] -> {mL0,mL1,kL0,kL1,nL0,nL1};
    vec call (M,N) -> {M,N}.
    """
    cfg = {}
    for s in tunable_sites(src):
        if s.start == s.end:  # absent knob (insertion site) -> currently unset
            cfg[s.id] = {"value": UNSET}
            continue
        nums = [int(x) for x in re.findall(r"-?\d+", src[slice(s.start, s.end)])]
        cfg[s.id] = _read_site(s.kind, nums)
    return cfg


def uniform_config(src, cube, vec):
    """Build a per-site config that sets EVERY tunable site the same (cube dict / vec dict).

    Handy for warm-starting BO from a global config such as the current production tiles, without
    committing the search to uniformity (BO still explores each site independently thereafter).
    Global (runtime_options) sites keep their CURRENT value so the warm-start config is complete.
    """
    cur = current_config(src)
    out = {}
    for s in tunable_sites(src):
        if s.kind == "cube":
            out[s.id] = dict(cube)
        elif s.kind == "vec":
            # Keep the site's own arity: a uniform vec dict may carry more or
            # fewer dimensions than this particular call has arguments, and
            # writing the wrong number of them would not parse.
            out[s.id] = {f: vec.get(f, cur[s.id][f]) for f in cur[s.id]}
        else:  # global
            out[s.id] = dict(cur[s.id])
    return out


def group_identical(src):
    """Opt-in efficiency lever: partition tunable sites into groups whose CURRENT call text is
    identical, so the caller can tie them and cut search dimensionality. Returns a list of id-lists
    (every tunable site appears in exactly one group). Default search does NOT group.
    """
    groups = {}
    for s in tunable_sites(src):
        seg = src[slice(s.start, s.end)]
        key = (
            ("g", space.global_knob(s.id), seg) if s.kind == "global" else (s.kind, seg)
        )
        groups.setdefault(key, []).append(s.id)
    return list(groups.values())
