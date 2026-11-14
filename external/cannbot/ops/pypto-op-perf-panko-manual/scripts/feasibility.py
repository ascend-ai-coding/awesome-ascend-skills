#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""feasibility.py --- reject hardware-infeasible candidates before they reach the device.

Ported from the numeric-feasibility idea in scripts/bayesian_optimization/bayesian.space.py on
improve/bo-config-search, and extended in the one way PANKO needs: Bayesian
optimisation owns its parameter space and can hand the checker a structured
config, whereas the optimiser here proposes free-form source edits. So the
tile parameters have to be recovered from the candidate file first.

What it catches, taken from the failures the 2026-07-28 campaign actually hit:

    F7A002 ALLOC_FAILED   TILE_B=2, view [2,16384] FP32:
                          2 rows = 128KB, plus 128x256 vec tiles = 128KB,
                          against a ~192KB unified-buffer budget
    ERR_CONFIG_ALIGNMENT  kL0 not 16-aligned
    ERR_CONFIG_TILE       mL0 > mL1, or mL1 % mL0 != 0

DESIGN RULE, and the reason this is safe to run on every candidate:

    reject ONLY when the violation is certain.

An unknown dtype is assumed to be the *smallest* plausible one, an unknown L1
envelope disables the L1 rules entirely, and anything the extractor cannot
parse is treated as feasible. A false rejection silently deletes a real
optimisation from the search space, which is far more expensive than the
evaluation a false acceptance wastes.

Usage:
    ok, reason = check("custom/<op>/<op>_impl.py")
    python3 feasibility.py <impl.py> [--ub-kb 192] [--l1-kb 192] [--m_l1 64] [--k_l1 64]
"""

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from typing import Optional
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jsonio  # noqa: E402

# One home for "is this module installed" and "did the runtime refuse": see
# chip_profile, which owns them because its refusals are load-bearing.
from chip_profile import optional as _optional, quietly as _quietly  # noqa: E402

# The dtype names a kernel spells ARE `pypto.DataType` members, so `pypto.bytes_of`
# is the table and this file must not hold a second one. What stays below is the
# OFFLINE fallback, for the test suite and for a box with no pypto installed, and
# it is partial on purpose: a name absent from the table is not read at all, so
# `extract` falls through to its default width, overestimates the footprint, and
# rejects legal tiles before any device sees them. `DT_FLOAT` is a CANN spelling
# with no `DataType` member of its own, so it is merged in either way.
DTYPE_BYTES_OFFLINE = {
    "DT_FP32": 4,
    "DT_FLOAT": 4,
    "DT_FP16": 2,
    "DT_BF16": 2,
    "DT_INT8": 1,
    "DT_UINT8": 1,
    "DT_INT32": 4,
}
_DTYPE_TABLE = None


def dtype_bytes_table():
    """`{dtype name: width in bytes}`, from pypto when it can be imported.

    Cached because importing pypto loads the runtime's shared objects and this is
    asked once per candidate.

    A width of zero or less is dropped rather than stored. The sub-byte members
    (`DT_INT4`, `DT_FP4_E2M1X2`) have no whole-byte width, and a 0 in this table
    would make a tile of them look free.
    """
    global _DTYPE_TABLE
    if _DTYPE_TABLE is not None:
        return _DTYPE_TABLE
    table = dict(DTYPE_BYTES_OFFLINE)
    pypto = _optional("pypto")
    members = getattr(pypto, "DataType", None) if pypto is not None else None
    width_of = getattr(pypto, "bytes_of", None) if pypto is not None else None
    if members is not None and callable(width_of):
        for member in members:
            width = _quietly(width_of, member)
            if isinstance(width, int) and width > 0:
                table[member.name] = width
    _DTYPE_TABLE = table
    return table


@dataclass
class HW:
    """Chip envelope. Unknown fields switch off the rules that need them."""

    ub_budget_kb: int = 192  # unified buffer available to vector tiles
    # 512 KB fixed hardware on A3. This said 192 -- an old conservative estimate
    # that `bayesian_optimization/space.py` had already corrected on its own copy
    # while this one kept the stale number, so the two gates in one run disagreed
    # by a factor of 2.7 and a legal cube tile could be rejected here.
    l1_budget_kb: int = 512  # L1 available to the two matmul operands
    m_l1: Optional[int] = None  # None => skip the L0/L1 subdivision rules
    k_l1: Optional[int] = None
    n_l1: Optional[int] = None
    n_buffers: int = 2  # double buffering; 1 if the kernel does not


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------
def _int_env(tree):
    """NAME -> int, for simple integer assignments anywhere in the module.

    Real kernels write `TILE_B = 32` and then `set_vec_tile_shapes(TILE_B, 256)`,
    so a literal-only reader sees a one-element tile and silently under-reports
    the footprint. A name bound to two different values is dropped: the estimate
    must not depend on which binding happened to be walked last.
    """
    seen = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        v = node.value
        if not (
            isinstance(v, ast.Constant)
            and isinstance(v.value, int)
            and not isinstance(v.value, bool)
        ):
            continue
        for t in node.targets:
            if isinstance(t, ast.Name):
                seen.setdefault(t.id, set()).add(v.value)
    return {k: next(iter(s)) for k, s in seen.items() if len(s) == 1}


def _as_int(node, env):
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    ):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id)
    return None


def _const_list(node, env=None):
    env = env or {}
    if isinstance(node, (ast.List, ast.Tuple)):
        out = []
        for e in node.elts:
            v = _as_int(e, env)
            if v is None:
                return None  # symbolic entry: not statically known
            out.append(v)
        return out
    return None


def _read_vec_tiles(node, env, info):
    """`set_vec_tile_shapes(...)`, ALL OR NOTHING.

    Dropping the arguments that do not resolve and keeping the rest fabricates a
    tile the kernel never had, and it fabricates a SMALL one, which is the
    direction that matters:

        A kernel writes `set_vec_tile_shapes(1, H)` with `H = x1.shape[1]`.
        Keeping the resolvable half records a 1-element tile, so ub_occupancy
        reads 0.0 on a kernel whose real tile is kilobytes wide, and the
        symptom fires on a fabricated number.

    A partially-readable call is a call we cannot read.
    """
    dims = [_as_int(a, env) for a in node.args]
    if dims and all(d is not None for d in dims):
        info["vec_tiles"].append(dims)
    else:
        info["unresolved_vec"] = info.get("unresolved_vec", 0) + 1


def _read_cube_tiles(node, env, info):
    """`set_cube_tile_shapes(...)`, on the same all-or-nothing terms."""
    groups = [_const_list(a, env) for a in node.args]
    if groups and all(g for g in groups):
        info["cube_tiles"].append(groups)
    else:
        info["unresolved_cube"] = info.get("unresolved_cube", 0) + 1


def _read_view(node, info):
    """`view(...)`, literal-only on purpose.

    A view is a window into global memory and is not necessarily resident in UB;
    resolving its symbols would feed the rejection rule a footprint the kernel
    never pays, and start refusing candidates that run perfectly well.
    """
    for a in node.args[1:]:
        shp = _const_list(a)
        if shp and len(shp) >= 2:
            info["views"].append(shp)
            return


def _read_unrolls(node, env, info):
    """Any `unroll_list=` keyword on the call."""
    for kw in node.keywords or []:
        if kw.arg == "unroll_list":
            v = _const_list(kw.value, env)
            if v:
                info["unrolls"].append(v)


def _read_call(node, env, info):
    """One call, dispatched on the name being called."""
    fn = node.func
    name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
    if name == "set_vec_tile_shapes":
        _read_vec_tiles(node, env, info)
    elif name == "set_cube_tile_shapes":
        _read_cube_tiles(node, env, info)
    elif name == "view":
        _read_view(node, info)
    _read_unrolls(node, env, info)


def extract(path):
    """Recover the tile parameters a candidate actually sets.

    Returns a dict; any key may be missing, and a missing key disables the
    rules that depend on it rather than failing the candidate.
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        src = f.read()
    info = {"vec_tiles": [], "cube_tiles": [], "views": [], "unrolls": []}

    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return {**info, "parse_error": str(e)}

    env = _int_env(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            _read_call(node, env, info)

    # dtype: assume the SMALLEST width seen, so an unknown or mixed-precision
    # kernel is never rejected on a footprint we may have overestimated.
    # `_` is in the class because `DT_FP8E4M3` and `DT_FP4_E2M1X2` carry one and
    # the old pattern stopped at it, so those names were never looked up.
    table = dtype_bytes_table()
    widths = [w for w in (table.get(m) for m in re.findall(r"DT_[A-Z0-9_]+", src)) if w]
    info["dtype_bytes"] = min(widths) if widths else 2
    return info


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------
def _vec_rules(info, hw):
    db = info["dtype_bytes"]

    for dims in info["vec_tiles"]:
        n = dims[-1]
        if n % 16 != 0 and n > 16:
            return False, f"vec tile inner dim {n} is not 16-aligned"

    # Unified-buffer pressure: the staged view plus the vector tiles must fit.
    # This is the check that would have caught the TILE_B=2 failures.
    if info["views"] and info["vec_tiles"]:
        view_kb = max(_prod(v) for v in info["views"]) * db / 1024.0
        tile_kb = max(_prod(t) for t in info["vec_tiles"]) * db / 1024.0
        total = view_kb + tile_kb * hw.n_buffers
        if total > hw.ub_budget_kb:
            return False, (
                f"UB estimate {total:.0f}KB "
                f"(view {view_kb:.0f}KB + {hw.n_buffers}x tile "
                f"{tile_kb:.0f}KB) > budget {hw.ub_budget_kb}KB"
            )
    return True, ""


def _cube_rules(info, hw):
    db = info["dtype_bytes"]
    for groups in info["cube_tiles"]:
        flat = []
        for g in groups:
            flat.extend(g)
        for d in flat:
            if d % 16 != 0:
                return False, f"cube tile dim {d} is not 16-aligned"
        if hw.m_l1 is None or hw.k_l1 is None:
            continue  # uncalibrated: alignment only
        m, k = groups[0][0], groups[0][1]
        if m > hw.m_l1:
            return False, f"mL0={m} > mL1={hw.m_l1}"
        if hw.m_l1 % m != 0:
            return False, f"mL1={hw.m_l1} not divisible by mL0={m}"
        if k > hw.k_l1:
            return False, f"kL0={k} > kL1={hw.k_l1}"
        if hw.k_l1 % k != 0:
            return False, f"kL1={hw.k_l1} not divisible by kL0={k}"
        n = groups[1][1] if len(groups) > 1 and len(groups[1]) > 1 else k
        est_kb = (m * k + k * n) * db / 1024.0
        if est_kb > hw.l1_budget_kb:
            return False, (f"L1 estimate {est_kb:.0f}KB > budget {hw.l1_budget_kb}KB")
    return True, ""


def _prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


def ub_occupancy(path, hw=None):
    """Fraction of the unified buffer the vector tiles are estimated to occupy.

    Returns a float in (0, inf) or None when it cannot be estimated. This is the
    other half of a number the gate has always computed and only ever used in one
    direction. `check` rejects a candidate whose footprint exceeds the budget; it
    has never had anything to say about a candidate that uses 4% of it, and a
    tile four times too small is invisible to a search that reads only latency.
    Nothing in a profiler trace says "the buffer is nearly empty".

    Deliberately narrower than the rejection rule: only the vector tiles, which
    are unambiguously UB-resident, times the buffer count. The staged view is
    excluded because it is a window into global memory and the kernel may never
    pay for it in UB. An overestimate here would push the search away from tile
    sizes that are in fact affordable, which is the opposite of the point.

        occupancy = n_buffers * max_tile_bytes / ub_budget_bytes

    A generated kernel's pre-optimization tile commonly scores a few hundredths
    here, while the best kernel a search reaches scores several tenths.
    """
    hw = hw or HW()
    info = extract(path)
    # No readable tile -> None, never 0.0. A kernel whose tiles are all driven
    # by symbols is one we cannot see, not one whose buffer is empty, and the
    # difference decides whether a symptom fires.
    if "parse_error" in info or not info.get("vec_tiles"):
        return None
    db = info["dtype_bytes"]
    tile_kb = max(_prod(t) for t in info["vec_tiles"]) * db / 1024.0
    if hw.ub_budget_kb <= 0:
        return None
    return round(tile_kb * hw.n_buffers / hw.ub_budget_kb, 4)


def _l1_kb(groups, db):
    """L1 footprint of one cube call: the two operands the gate already charges.

    L0A is [m_l0, k_l0] and L0B is [k_l0, n_l0]; L0C is an accumulator and is not
    staged through the same budget. This is the identical expression used by
    `_cube_rules` to REJECT -- read in the other direction.
    """
    m, k = groups[0][0], groups[0][1]
    n = groups[1][1] if len(groups) > 1 and len(groups[1]) > 1 else k
    return (m * k + k * n) * db / 1024.0


def l1_occupancy(path, hw=None):
    """Fraction of L1 the cube operands are estimated to occupy, or None.

    The cube counterpart of `ub_occupancy`, and it exists for the same reason:
    the rejection rule has always computed this number and has only ever used it
    in one direction. `check` refuses a cube tile whose operands exceed the
    budget; it has never had anything to say about one that uses 2% of it.

    None for a kernel with no readable cube call, which is the honest answer for
    an operator that has no matmul at all -- a vector kernel has no L1 symptom,
    not a bad one.

    Unlike the UB estimate this one discriminates between kernels: a
    matmul-heavy kernel fills a large fraction of L1, while a kernel whose cube
    tiles are small never comes close to it -- and the second is exactly the
    kernel with L1 headroom still to take.
    """
    hw = hw or HW()
    info = extract(path)
    if "parse_error" in info or not info.get("cube_tiles"):
        return None
    if hw.l1_budget_kb <= 0:
        return None
    db = info["dtype_bytes"]
    kb = max(_l1_kb(g, db) for g in info["cube_tiles"])
    return round(kb / hw.l1_budget_kb, 4)


def check(path, hw=None):
    """(ok, reason). ok=True whenever the violation is not certain."""
    hw = hw or HW()
    info = extract(path)
    if "parse_error" in info:
        return True, ""  # the compiler will reject it anyway
    ok, why = _vec_rules(info, hw)
    if not ok:
        return False, why
    return _cube_rules(info, hw)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("impl")
    ap.add_argument("--ub-kb", type=int, default=192)
    ap.add_argument("--l1-kb", type=int, default=192)
    ap.add_argument("--mL1", dest="m_l1", type=int, default=None)
    ap.add_argument("--kL1", dest="k_l1", type=int, default=None)
    ap.add_argument("--n-buffers", type=int, default=2)
    ap.add_argument("--show", action="store_true", help="dump what was extracted")
    a = ap.parse_args()
    hw = HW(
        ub_budget_kb=a.ub_kb,
        l1_budget_kb=a.l1_kb,
        m_l1=a.m_l1,
        k_l1=a.k_l1,
        n_buffers=a.n_buffers,
    )
    if a.show:
        jsonio.emit(extract(a.impl), indent=2)
    ok, why = check(a.impl, hw)
    jsonio.emit({"feasible": ok, "reason": why})
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
