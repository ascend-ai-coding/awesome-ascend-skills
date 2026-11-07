# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""What the core can say about a tile, and what it must be told.

A tile is bounded by the hardware, not by the program's views: in PyPTO a tile
shape and a view shape are independent quantities. So there are two rules, and
only two.

    trailing dimension is 16-aligned        (in `space`; a compiler rule,
                                             recorded in ERR_CONFIG_ALIGNMENT)
    footprint x dtype x live tiles <= UB    (here)

Neither was enforced. `_vec_feasible` checked the alignment and stopped; the
harness constructed `HW(calibrated=False)`, which switched off even the checks
that existed. A ceiling seed larger than the buffer can hold at the kernel's
live-tile count then reaches the device, returns s=0, and records a PENALTY at
the region of interest -- teaching the surrogate to avoid a neighbourhood that
was never actually bad.

The one fact the core cannot read
--------------------------------
Live tile count. How many tile-shaped tensors are simultaneously resident is a
dataflow property, and after the model has fused, flattened or reordered
anything it is a property of a program the extractor has never seen. On a
fused chain the optimizer may derive four live tiles where `ub_occupancy`
assumes two.

The model may report that number, with its reasoning, and may report nothing
else. It may not narrow a range, name a maximum, or otherwise touch the edges.
Without it the footprint bound is OFF -- stated in the notes rather than guessed
at, because a guess here is a guess in an unknown direction.

Which direction to fail
-----------------------
Wide. A bound that is too generous is caught per candidate: every configuration
is re-checked and an impossible one is rejected before it reaches the device. A
bound that is too tight cannot be detected at all -- BO behaves optimally inside
whatever box it is given, and an optimum outside the box leaves no trace anywhere
in the run.

This module previously ignored its own advice. It derived per-dimension domains
from the view's extents, on the assumption that a tile dimension must divide the
view it is cut from. That assumption is false, and the kernels say so: a tile's
leading dimension can be many times the view's. The rule collapsed such an axis
to a single element and capped another far below the extent the hand-optimised
version runs at. A search inside those boxes would have converged cleanly and
reported nothing wrong. It was removed rather than narrowed.
"""

import ast
import os
import sys

# `pypto_ast` and `feasibility` are siblings of this package, one directory up.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# One home, shared with `predicates`: see pypto_ast. Not optional -- every tile
# this module counts depends on knowing which calls are pypto's.
from pypto_ast import pypto_roots  # noqa: E402

try:
    import feasibility  # noqa: E402
except Exception:  # pragma: no cover
    feasibility = None


# Unsupplied means the capacity rule is OFF, not that residency is two. A guess
# here is a guess in an unknown direction: too high and the bound is too tight,
# which narrows the domain -- the one error that leaves no trace.
#
# Since the divisor rules turn out to constrain only the axes where the view
# genuinely exceeds the tile, capacity is the bound that does most of the work,
# and it is off until the optimizer supplies `--live-tiles`. A run whose blocks
# never carry that number is a run where the core is checking alignment and
# little else; `block` reports `capacity_elems: null` so it is visible rather
# than inferred from a disappointing latency.
DEFAULT_LIVE_TILES = 0


# Calls that are NOT vector ops governed by a `set_vec_tile_shapes` directive.
# A matmul is a cube op (its footprint lives in L0, checked separately); a view
# is an addressing construct, not a tile allocation; a loop / range is control
# flow. An assignment whose right-hand side calls one of these does not add a
# vec tile to the current site's residency.
_NON_VEC_CALLS = frozenset(
    (
        "matmul",
        "view",
        "loop",
        "range",
        "assemble",
        "load",
        "store",
        "set_cube_tile_shapes",
        "set_vec_tile_shapes",
    )
)

# Methods that produce a SCALAR, not a tile: `(M - off).min(TILE_M)`, `.max()`,
# `x.shape[0]`. An assignment calling one of these is index / trip-count
# arithmetic on SymbolicScalars, not a vector op, so it adds no UB tile.
_SCALAR_METHODS = frozenset(("min", "max", "ceil", "floor", "shape", "item"))


def _read_text(path, errors="replace"):
    """The file's text, with the handle closed before the caller sees it."""
    with open(path, encoding="utf-8", errors=errors) as f:
        return f.read()


def _rhs_calls(node):
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            yield f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")


def _produces_a_tile(rhs):
    """Does this right-hand side create a tile-shaped tensor (as opposed to a
    SymbolicScalar)? A `pypto.<op>(...)` call that is not an index/scalar helper
    does: view, matmul, and every vector op return a tile. `(M-off).min(...)`,
    `x.shape[0]` and bare arithmetic do not.
    """
    if isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Attribute):
        return rhs.func.attr not in _SCALAR_METHODS
    return False


def _is_pypto_call(call, roots):
    node = call.func
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name) and node.id in roots


def _is_jit(fn):
    """A `@...jit(...)`-decorated def -- the kernel, as opposed to the wrapper."""
    for d in getattr(fn, "decorator_list", []):
        node = d.func if isinstance(d, ast.Call) else d
        while isinstance(node, ast.Attribute):
            if node.attr == "jit":
                return True
            node = node.value
        if isinstance(node, ast.Name) and node.id == "jit":
            return True
    return False


def _is_vector_op(rhs, tiles, roots=frozenset({"pypto"})):
    """Does this right-hand side allocate a VECTOR tile in UB?

    Two forms, and the discriminator against scalar arithmetic is `tiles` -- the
    set of names already known to hold a tile (results of view / matmul / vector
    ops). A blind operand count cannot tell `gate_tile * sigmoid_g` from
    `idx * TILE_M`: both are BinOps. The tile set can -- the first references a
    tile, the second references only loop scalars.

      * a `pypto.<vecop>(...)` call -- an Attribute call ON PYPTO that is neither a
        cube / addressing op (`_NON_VEC_CALLS`) nor a scalar helper
        (`_SCALAR_METHODS`);
      * a BinOp / UnaryOp that references at least one name in `tiles`.

    Over-counting residency is the dangerous direction -- it shrinks the per-site
    cap and rejects legal tiles -- so anything not positively a tile op is out.
    The call test used to check the attribute name alone, so a host-side
    `torch.cat` or `torch.add` was a vector op holding UB it never touches.
    """
    if isinstance(rhs, ast.Call):
        f = rhs.func
        if isinstance(f, ast.Attribute) and _is_pypto_call(rhs, roots):
            return f.attr not in _SCALAR_METHODS and f.attr not in _NON_VEC_CALLS
        return False
    if isinstance(rhs, (ast.BinOp, ast.UnaryOp)):
        return any(isinstance(n, ast.Name) and n.id in tiles for n in ast.walk(rhs))
    return False


def _rhs_tensor_operands(node):
    """Distinct tensor NAMES read on the right-hand side of a vector op, + result.

    `(operands + 1)` is the `tensor_count` the design docs give: 2 for a unary op,
    3 for a binary one. Module refs (`pypto` in `pypto.cast`) and `pypto.DT_*`
    attributes are not operands, so `sigmoid(gate)` is one operand and
    `gate * sigmoid` is two. Called only for statements `_is_vector_op` has
    already accepted.
    """
    bases = {
        a.value.id
        for a in ast.walk(node)
        if isinstance(a, ast.Attribute) and isinstance(a.value, ast.Name)
    }
    names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    return len(names - bases)


def _ordered_statements(scopes):
    """The tile setters and the assignments, in source order.

    `ast.walk` does not preserve statement order across nesting, so they are
    collected with their positions and sorted once.
    """
    stmts = []
    for scope in scopes:
        for node in ast.walk(scope):
            if isinstance(node, ast.Call) and _func_attr(node) == "set_vec_tile_shapes":
                stmts.append((node.lineno, node.col_offset, "setter", node))
            elif isinstance(node, ast.Assign):
                stmts.append((node.lineno, node.col_offset, "assign", node))
    stmts.sort(key=lambda t: (t[0], t[1]))
    return stmts


def _assigned_names(node):
    """The plain names this assignment binds."""
    out = set()
    for tgt in node.targets:
        if isinstance(tgt, ast.Name):
            out.add(tgt.id)
    return out


def vec_residency(src):
    """Peak vec-tile residency per `set_vec_tile_shapes` site, from the source.

    Returns `{vec_site_id: n}` where n is the largest `operands + 1` over the
    elementwise ops the site governs -- the number of tile-shaped tensors the UB
    must hold at once for that directive. This replaces the single `--live-tiles`
    number the optimizer used to supply by judgement, which drifted between 4, 6
    and 8 across runs of the same kernel and set the search space with it.

    Site ids match `apply`: `set_vec_tile_shapes` calls are numbered vec#0,
    vec#1, ... in source order. A directive governs the elementwise statements
    that follow it until the next `set_vec_tile_shapes`; matmul / view / loop
    statements in between belong to the cube directive or to control flow and are
    not counted (see `_NON_VEC_CALLS`).

    Scoped to the KERNEL. This walked the whole module, so statements in the
    host-side torch wrapper -- `torch.cat`, `torch.add`, an `empty_like` feeding
    a BinOp -- were charged to whichever tile directive happened to precede them
    in the file. They hold no unified buffer, so the estimate came out high, the
    per-site cap came out small, and legal tiles were rejected before they
    reached the device. It also meant editing the wrapper moved the kernel's
    search space. A module with no jit function is not a PyPTO kernel at all, so
    that case keeps the whole-module walk rather than silently returning nothing.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {}
    roots = pypto_roots(tree)
    kernels = [
        fn
        for fn in ast.walk(tree)
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_jit(fn)
    ]
    scopes = kernels or [tree]
    res, ordinal, active = {}, -1, None
    tiles = set()  # names known to hold a tile-shaped tensor
    # ast.walk does not preserve statement order across nesting, so collect the
    # setters and assignments with their positions and sort once.
    for _, _, kind, node in _ordered_statements(scopes):
        if kind == "setter":
            ordinal += 1
            active = f"vec#{ordinal}"
            res.setdefault(active, 1)
            continue
        rhs = node.value
        # A statement contributes to residency only if it is a vector op. It
        # contributes a tile NAME regardless, so a later BinOp can tell a tile
        # operand from a loop scalar.
        is_vec = active is not None and _is_vector_op(rhs, tiles, roots)
        if is_vec:
            n = _rhs_tensor_operands(rhs) + 1
            if n > res.get(active, 1):
                res[active] = n
        if is_vec or _produces_a_tile(rhs):
            tiles |= _assigned_names(node)
    return res


def _func_attr(call):
    f = call.func
    return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")


def vec_ladder_cap(domain, site_id, field, fields):
    """The largest value `field` may take before it overflows the UB even with
    every other dimension of this vec site at its smallest legal rung.

    The UB rule is joint -- `prod(dims) * dtype * residency <= UB` -- so no single
    dimension has a fixed bound. But a rung is safe to DROP if it overflows when
    all the OTHER dimensions are minimal: no feasible tile can contain it. That
    bound is `UB / (dtype * residency * prod of the other fields' minima)`, where
    the trailing (contiguous, 16-aligned) field's minimum is 16 and the rest are
    1. Returns None when the footprint bound is off, so the caller keeps the full
    ladder. Removing only these rungs discards no reachable configuration; it
    stops TPE spending its startup draws on tiles a hundredfold over the buffer.
    """
    ub = (domain or {}).get("ub_bytes")
    db = (domain or {}).get("dtype_bytes")
    r = ((domain or {}).get("vec_residency") or {}).get(site_id)
    if not (ub and db and r):
        return None
    others = 1
    for j, f in enumerate(fields):
        if f == field:
            continue
        others *= 16 if j == len(fields) - 1 else 1
    return ub // (db * max(1, r) * others)


def read_views(src):
    """View shapes as integers, at ANY rank, resolving named bindings.

    Two departures from `feasibility.extract`, and each has a reason.

    **Rank.** `feasibility` keeps `len(shape) >= 2`, which is right for the rule
    it feeds and wrong here: it makes a flattened kernel's view invisible, and a
    flatten is exactly the case this domain exists to constrain. Once a kernel
    has been flattened the only remaining knob IS the chunk size, cut from a 1-D
    view.

    **Symbols.** `feasibility` reads views literal-only, deliberately:

        resolving its symbols would feed the rejection rule a footprint the
        kernel never pays

    That reasoning is about FOOTPRINT, and this is not a footprint. Whether a
    tile dimension divides a view's extent does not depend on whether the view is
    resident. And literal-only is not a conservative choice here, it is an empty
    one: every level1 kernel writes `pypto.view(x, [TILE_B, D], ...)`, so a
    literal-only reader returns nothing on all six and the divisor rules never
    fire once.

    A name bound to two different values is dropped by `_int_env`, so the shape
    never depends on which binding happened to be walked last.

    Reading this here rather than widening `feasibility` keeps the baseline arms,
    which share that module, byte-identical.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    env = _int_env(tree)
    views = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name != "view":
            continue
        # The SHAPE is argument 1, positionally. Scanning args[1:] for "the first
        # list that resolves" reads the OFFSET as a shape the moment the real
        # shape is symbolic -- `pypto.view(x, [TILE_B, D], [0, 0])` yields a view
        # of [0, 0], which is worse than reading nothing: it is a fabricated
        # extent that suppresses the "no literal view" note and then constrains
        # the domain to divisors of zero.
        if len(node.args) < 2:
            continue
        shape = _int_list(node.args[1], env)
        if shape:
            views.append(shape)
    return views


def _int_bindings(target, value):
    """(name, int) pairs bound by one target, tuple unpacking included."""
    if isinstance(target, ast.Name) and _is_int_const(value):
        return [(target.id, value.value)]
    if not (
        isinstance(target, (ast.Tuple, ast.List))
        and isinstance(value, (ast.Tuple, ast.List))
        and len(target.elts) == len(value.elts)
    ):
        return []
    return [
        (t.id, v.value)
        for t, v in zip(target.elts, value.elts)
        if isinstance(t, ast.Name) and _is_int_const(v)
    ]


def _bind_ints(node, env, bad):
    """Fold one assignment's integer bindings into `env`, naming clashes in `bad`."""
    for t in node.targets:
        for name, value in _int_bindings(t, node.value):
            if name in env and env[name] != value:
                bad.add(name)
            env[name] = value


def _int_env(tree):
    """NAME -> int for unambiguous integer bindings anywhere in the module.

    Tuple unpacking included: `TILE_B, TILE_D = 16, 256` is how these kernels
    bind their tile constants, and `feasibility._int_env` -- which this used to
    delegate to -- reads simple assignments only, so both names came back
    unresolved and the view with them.

    A name bound to two different values is dropped, so a shape never depends on
    which binding happened to be walked last.
    """
    env, bad = {}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            _bind_ints(node, env, bad)
    return {k: v for k, v in env.items() if k not in bad}


def _is_int_const(n):
    return (
        isinstance(n, ast.Constant)
        and isinstance(n.value, int)
        and not isinstance(n.value, bool)
    )


def _int_list(node, env=None):
    if not isinstance(node, ast.List) or not node.elts:
        return None
    out = []
    for e in node.elts:
        if _is_int_const(e):
            out.append(e.value)
        elif isinstance(e, ast.Name) and (env or {}).get(e.id) is not None:
            out.append(env[e.id])
        else:
            return None
    return out


def _prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


def derive(op_file, live_tiles=None, ub_budget_kb=192):
    """What the program says about the tiles it can hold.

    Returns:
        {"capacity_elems": int or None,     the footprint bound, in elements
         "live_tiles": int, "dtype_bytes": int,
         "views": [...],                    for change detection ONLY
         "per_site": {},                    always empty; see below
         "notes": [str]}

    `per_site` exists so callers stay stable, and it is always empty. An earlier
    version derived per-dimension domains from the view's extents -- the tile
    dimension had to divide it -- and that was simply wrong about PyPTO: a tile
    shape and a view shape are independent. The kernels say so plainly once you
    stop assuming otherwise: a generated kernel can set a leading tile dimension
    many times larger than the view it is cut from.

    The rule survived one review because it looked conservative and its failures
    looked like caution. It collapsed a leading axis to a single element and
    capped a trailing axis far below the extent the hand-optimised version runs
    at -- and a search inside those boxes would have converged
    cleanly, reported nothing unusual, and never once indicated that the optimum
    was outside. That is the failure mode this module's header warns about, built
    into the module itself.

    What is left is what was true all along: the tile is bounded by the hardware,
    not by the view. Alignment (in `space`) and footprint (here).
    """
    report = {
        "per_site": {},
        "capacity_elems": None,
        "views": [],
        "live_tiles": int(live_tiles or DEFAULT_LIVE_TILES),
        "dtype_bytes": 4,
        "notes": [],
    }
    if feasibility is None:
        report["unreadable"] = True
        report["notes"].append("feasibility unavailable; no footprint bound")
        return report
    try:
        info = feasibility.extract(op_file)
    except Exception as e:  # pragma: no cover
        report["unreadable"] = True
        report["notes"].append(f"extract failed ({e}); no footprint bound")
        return report
    if "parse_error" in info:
        report["unreadable"] = True
        report["notes"].append("kernel does not parse; no footprint bound")
        return report

    src = _read_text(op_file)
    # Kept for `changed()` only. A view is not a constraint on the tile, but a
    # view that moves is a program whose access pattern moved, which is a reason
    # to re-tune even though it is not a reason to reject anything.
    report["views"] = read_views(src)
    report["dtype_bytes"] = int(info.get("dtype_bytes") or 4)

    # Per-vec-site UB budget, in BYTES, derived from the source rather than
    # supplied by hand. Each vec directive may hold a different tile size and
    # governs a different number of simultaneously-resident tiles, so the bound
    # is per site: `prod(dims) * dtype * residency <= UB`. `verify` applies it.
    report["ub_bytes"] = int(ub_budget_kb) * 1024 if ub_budget_kb else 0
    report["vec_residency"] = vec_residency(src)
    # `capacity_elems` and `live_tiles` are retained as a single-number summary
    # for `changed()` / logging only -- computed from the WORST site so the
    # fingerprint still moves when the footprint envelope moves. The real gate is
    # per-site in `verify`; nothing reads this scalar as a bound any more.
    db = report["dtype_bytes"]
    peak = max(report["vec_residency"].values(), default=0)
    report["live_tiles"] = peak
    if db > 0 and peak > 0 and report["ub_bytes"]:
        report["capacity_elems"] = report["ub_bytes"] // (db * peak)
    else:
        report["notes"].append(
            "no vec site found, so the footprint bound is off (alignment only)"
        )
    return report


def fingerprint(domain):
    """A comparable summary of the space, ignoring everything that is not it.

    Change 4 rests on this: when a structural action lands, the previously
    located tile optimum describes a program that no longer exists -- but only
    when the SPACE moved. `lever.structural_signature` is the obvious test and
    the wrong one, because it hashes the program with the tile literals erased,
    so a renamed variable or an added comment changes it and a block would fire
    after every non-tile action.

    What actually invalidates a tile optimum is a change to the inputs of the
    derivation: the view extents, the dtype, the residency. Those and nothing
    else are what this fingerprint carries.

    Deliberately NOT including `per_site`. Since an axis is only constrained when
    the view genuinely exceeds the running tile, the per-dimension domains depend
    on the tile's current value -- so a block that improved the tile would change
    them, fire a retune, run another block, and loop.
    """
    return repr(
        (
            sorted(tuple(v) for v in (domain.get("views") or [])),
            domain.get("capacity_elems"),
            domain.get("dtype_bytes"),
        )
    )


def changed(before, after):
    """Did the tile space move? (bool, one-line reason).

    A program that could not be read is UNKNOWN, not changed. Reporting a move
    there would fire a block on every syntactically broken candidate -- and a
    broken candidate is about to fail its golden test anyway, so the block would
    be spent tuning a kernel that is on its way to being reverted.
    """
    if before.get("unreadable") or after.get("unreadable"):
        return False, "the program could not be read; no verdict"
    if fingerprint(before) == fingerprint(after):
        return False, "the derived tile domain is unchanged"
    bits = []
    if (before.get("views") or []) != (after.get("views") or []):
        bits.append(f"views {before.get('views')} -> {after.get('views')}")
    if before.get("capacity_elems") != after.get("capacity_elems"):
        bits.append(
            f"capacity {before.get('capacity_elems')} -> "
            f"{after.get('capacity_elems')} elements"
        )
    return True, "; ".join(bits) or "the derived tile domain changed"


def verify(domain, config, dtype_bytes=4):
    """Is `config` inside `domain`? (ok, reason).

    Called on every candidate, which is what makes a too-wide domain harmless.

    The UB footprint rule is per vec SITE: a directive's tile must satisfy
    `prod(dims) * dtype * residency <= UB`, where `residency` is how many tiles
    that directive keeps live at once (`vec_residency`, from the source) and UB
    is the 192 KB unified buffer. Each site has its own tile size and its own
    residency, so the check cannot be collapsed to one number -- doing so is what
    made the old bound reject `(1,64)` beside `(128,192)` for a total that was
    never how the buffer is spent. Cube sites carry no `d*` dims and are checked
    against L0A/B/C in `space` instead.
    """
    db = domain.get("dtype_bytes") or dtype_bytes
    ub = domain.get("ub_bytes")
    res = domain.get("vec_residency") or {}
    for sid, c in config.items():
        dims = [v for k, v in c.items() if k.startswith("d")]
        if not (dims and ub):
            continue
        live = res.get(sid, 1)
        need = _prod(dims) * db * live
        if need > ub:
            return False, (
                f"{sid}: {_prod(dims)} elements x {db}B x {live} live "
                f"= {need}B > {ub}B UB"
            )
    return True, ""
