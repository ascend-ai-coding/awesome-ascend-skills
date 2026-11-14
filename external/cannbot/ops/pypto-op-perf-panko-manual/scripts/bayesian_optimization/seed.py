# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Startup points chosen to COVER the operating space, not to guess a good one.

Why not random
--------------
TPE's startup draws are uniform in PARAMETER space, and most of that volume is
either rejected or absurd. Two draws from one recorded block:

    {mL0:192, kL0:32, nL0:32, ...}  ->  22,198 us   (8.5x the incumbent)
    {mL0:512, kL0:64, nL0:32, ...}  ->  14,114 us

Both reached the device, and in FEATURE space they are nearly the same point --
a starved N dimension, an empty L0C, far too many trips. Two of a block's four
to eight measurements went to one operating regime.

Why not a model either
----------------------
The obvious repair is to seed where some computable quantity predicts a win.
Every such quantity tried on this benchmark has been refuted by measurement:
filling L0 (three configurations, all tie-or-worse), operand reuse per loaded
element (improved monotonically while latency got monotonically worse), padded
work against the core count, power-of-two alignment. One survived -- filling
L1 -- on a single operator. Seeding by a predictor built on that record would
repeat the mistake it is made of.

So this module predicts nothing. It measures FEATURES that are arithmetic
rather than theory -- how full each buffer is, how many loop trips there are,
how those trips divide across the cores -- and picks points that are as far
apart in that space as possible. Whatever the true relationship turns out to
be, the measurements span it, and the device answers the question instead of us.

Two facts here are arithmetic and are used as such, not as models:
`trips < cube_cores` leaves cores with no work at all, and a tile occupying a
small fraction of a buffer runs the loop proportionally more times.

Cost
----
Candidates are filtered by the existing static gate BEFORE selection, so a
startup point can no longer be spent on a configuration the harness would have
rejected anyway. Nothing here touches the device.
"""

import random

from dataclasses import dataclass
from typing import Any

from . import space

# Fallback only. The live values ride on `hw` (space.HW.cube_cores /
# vector_cores), supplied by the harness from the run's chip envelope, so a chip
# with different counts does not silently score against these.
# Ascend 910B3, confirmed against a trace: 20 cube lanes (AIC_0..19) and 40
# vector lanes (AIV_20..59). Used only to express "does every core get work",
# which is division, not a performance claim.
CUBE_CORES = 20
VECTOR_CORES = 40


def _cube_features(c, hw):
    """Buffer occupancies for one cube tile, each in [0, 1] against its own limit."""
    db = hw.dtype_bytes
    a = space.ceil16(c["mL0"]) * space.ceil16(c["kL0"]) * db
    b = space.ceil16(c["kL0"]) * space.ceil16(c["nL0"]) * db
    cc = space.ceil16(c["mL0"]) * space.ceil16(c["nL0"]) * 4
    l1 = (c["mL1"] * c["kL1"] + c["kL1"] * c["nL1"]) * db
    return {
        "l0a": a / (hw.l0a_kb * 1024),
        "l0b": b / (hw.l0b_kb * 1024),
        "l0c": cc / (hw.l0c_kb * 1024),
        "l1": l1 / (hw.l1_budget_kb * 1024),
        # How many L0 steps one L1 staging serves on each axis. Small means
        # the operand is re-staged often; this is a count, not a verdict.
        "m_steps": c["mL1"] / max(1, c["mL0"]),
        "k_steps": c["kL1"] / max(1, c["kL0"]),
        "n_steps": c["nL1"] / max(1, c["nL0"]),
    }


def _vec_features(v, hw, domain, site_id):
    """UB occupancy of one vector tile, and the trip count it implies."""
    dims = [v[f] for f in space.vec_fields(v)]
    n = 1
    for d in dims:
        n *= max(1, d)
    res = ((domain or {}).get("vec_residency") or {}).get(site_id) or 1
    ub = (domain or {}).get("ub_bytes") or (192 * 1024)
    cap = (domain or {}).get("capacity_elems")
    f = {"ub": min(1.0, n * hw.dtype_bytes * res / ub)}
    if cap:
        trips = cap / max(1, n)
        f["trips"] = trips
        # Below one trip per core, some cores are handed nothing. Division, not
        # a model: it is true whatever the kernel does with the work it gets.
        f["waves"] = trips / getattr(hw, "vector_cores", VECTOR_CORES)
    return f


def features(cfg, hw, domain=None):
    """Flat feature vector for a per-site config. Keys are stable, values finite."""
    out = {}
    for sid, v in sorted(cfg.items()):
        if not isinstance(v, dict) or "value" in v:
            # A global / unroll / pass site holds a scalar under "value", not a
            # tile. `vec_fields` will happily invent dimension names for it, and
            # reading them raises. There is no buffer occupancy to compute for a
            # scheduling mode or a merge granularity.
            continue
        if "mL0" in v:
            out.update({f"{sid}.{k}": x for k, x in _cube_features(v, hw).items()})
        elif space.vec_fields(v):
            vec = _vec_features(v, hw, domain, sid)
            out.update({f"{sid}.{k}": x for k, x in vec.items()})
    return out


def _normalise(vectors):
    """Scale each feature to [0, 1] over the candidate set.

    Without this, `trips` (thousands) would drown every occupancy (0..1) and the
    selection would be a trip-count sweep wearing a disguise.
    """
    if not vectors:
        return []
    keys = sorted({k for v in vectors for k in v})
    lo = {k: min(v.get(k, 0.0) for v in vectors) for k in keys}
    hi = {k: max(v.get(k, 0.0) for v in vectors) for k in keys}
    span = {k: (hi[k] - lo[k]) or 1.0 for k in keys}
    return [[(v.get(k, lo[k]) - lo[k]) / span[k] for k in keys] for v in vectors]


def _dist2(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


# Squared distance in the normalised feature space below which two candidates
# describe the same operating point, so measuring both would spend a device
# trial to learn what the first one already said.
#
# A guard, not a fix for anything observed: selected seeds normally sit orders
# of magnitude further apart than this bound.
# Farthest-point selection only crowds when `k` approaches the number of
# distinct regimes the domain can express, which is where a small `k` will not
# take it. Kept because that condition is a property of the caller's budget,
# not of this space.
_MIN_SEP2 = 0.01


def _hold_l0(site, anchor, rng):
    """One cube config holding the incumbent's L0 and drawing only the L1 multipliers.

    None when the incumbent does not name an L0 on every axis: there is then no
    granularity to hold fixed, and this family has nothing to say.
    """
    a = (anchor or {}).get(site.id, {})
    c = {}
    for ax in ("m", "k", "n"):
        l0 = a.get(f"{ax}L0")
        if not l0:
            return None
        x = rng.choice(space.mult_dom(l0, a.get(f"{ax}L1")))
        c[f"{ax}L0"], c[f"{ax}L1"] = l0, l0 * x
    return c


def _keep_incumbent_vec(cfg, sites, anchor):
    """The vector sites keep the incumbent's values.

    This family exists to vary ONE thing, and mixing a vec draw into it would
    make a measurement that separates nothing.
    """
    for s in sites:
        if s.kind == "cube":
            continue
        a = (anchor or {}).get(s.id)
        if a:
            cfg[s.id] = dict(a)


def _l1_only(sites, anchor, rng, n):
    """Cube points that HOLD the incumbent's L0 and move only the L1 multipliers.

    This slice has no other generator. Cube proposals split into two modes that
    do not overlap:

      * exploration -- L0 drawn wide, and because a large L0 arrives together
        with a large multiplier, most of those proposals are rejected by the
        static gate on L0A / L0C / L1 capacity.
      * concentration -- L0 returns to the incumbent, and so do the multipliers,
        so only a handful of multiplier triples are ever proposed.

    Between them lies "same granularity, different re-staging count". Many
    configurations pass the gate there and a campaign visits almost none of
    them, even though that region holds real gains -- some of which a later
    source rewrite only pays out on top of.

    `_sample_space` cannot reach here: it draws L0 uniformly from a ten-value
    ladder per axis, so all three axes land on the incumbent about once in a
    thousand draws, and a pool of 400 holds such a point 0.4 times.

    Nothing here claims a large L1 is good. `_cube_features` already scores
    `m_steps` / `k_steps` -- the L1/L0 ratios -- so the machinery to tell these
    points apart existed; what was missing was any point to tell apart.
    """
    cubes = [s for s in sites if s.kind == "cube"]
    if not cubes:
        return []
    out = []
    for _ in range(n):
        cfg = {}
        for s in cubes:
            c = _hold_l0(s, anchor, rng)
            if c is None:
                return []
            cfg[s.id] = c
        _keep_incumbent_vec(cfg, sites, anchor)
        out.append(cfg)
    return out


def _rand_cube(a, rng):
    """A cube draw from the same domains `space.suggest` uses."""
    c = {}
    for ax in ("m", "k", "n"):
        l0 = rng.choice(space.admit(space.CUBE_CHOICES, a.get(f"{ax}L0")))
        x = rng.choice(space.mult_dom(a.get(f"{ax}L0"), a.get(f"{ax}L1")))
        c[f"{ax}L0"], c[f"{ax}L1"] = l0, l0 * x
    return c


def _rand_vec(site_id, a, domain, rng):
    """A vec draw, or {} when the incumbent names no fields to draw."""
    fields = space.vec_fields(a)
    if not fields:
        return {}
    return {
        f: rng.choice(space.vec_domain(site_id, f, fields, a, domain)) for f in fields
    }


def _sample_space(sites, anchor, domain, n, rng):
    """Random draws from the same domains `space.suggest` uses.

    Sampling then filtering is deliberate: the domains are what the sampler will
    actually see, so a point selected here is one TPE could also have proposed.
    """
    out = []
    for _ in range(n):
        cfg = {}
        for s in sites:
            a = (anchor or {}).get(s.id, {})
            drawn = (
                _rand_cube(a, rng)
                if s.kind == "cube"
                else _rand_vec(s.id, a, domain, rng)
            )
            if drawn:
                cfg[s.id] = drawn
        if cfg:
            out.append(cfg)
    return out


# Below this many knobs the domains are small enough that the sampler's own
# ladder walks them, and a covering point is a trial taken away from that walk.
# Measured, not assumed: on a one-knob vector kernel whose optimum is a narrow
# spike, spending three trials on coverage made the block stagnate at 15 trials
# having never drawn the optimum, where the unmodified block reached it at 17.
# Coverage buys information when the space has more dimensions than the budget
# can walk; it costs information when it does not.
MIN_DIMS = 5


def n_dims(sites, anchor=None):
    """How many knobs the sampler will draw.

    A cube site is six -- an L0 and an L1 multiplier on each of m, k, n. A
    vector site is one per tile DIMENSION, and that count is a property of the
    kernel, not a constant: `set_vec_tile_shapes` takes one to four arguments
    and the recorded kernels use all of them. Assuming two was wrong in both
    directions -- it under-counts a four-argument tile (4 knobs read as 2, so a
    space that needs covering does not get it) and over-counts three
    single-argument sites (3 read as 6, so a space that the sampler's own
    ladder walks is charged for coverage it cannot use). The second is the
    failure the block tests caught.

    The arity comes from `anchor`, which is the configuration currently in the
    file. Without it, fall back to two.
    """
    n = 0
    for s in sites:
        if s.kind == "cube":
            n += 6
        elif s.kind in ("global", "unroll"):
            n += 1
        else:
            fields = space.vec_fields((anchor or {}).get(s.id, {}))
            n += len(fields) or 2
    return n


@dataclass
class CoverOptions:
    """How `cover` draws its candidates: where from, and what to leave out."""

    domain: Any = None
    seed: int = 0
    pool: int = 400  # candidates sampled before the k are picked
    exclude: Any = ()  # configs already queued, not to repeat


def cover(sites, hw, anchor, k, opts=None):
    """`k` configs spanning the feature space, PLUS one L1-only point on a cube
    kernel. With `anchor` as the notional first point.

    Greedy farthest-point: repeatedly take the candidate whose nearest already
    chosen point is furthest away. The anchor seeds the set because it is the
    one configuration whose latency is already known, so every later point is
    chosen to differ from something real rather than from an arbitrary origin.

    Returns configs EXCLUDING the anchor -- the caller already has that one.
    Infeasible candidates are dropped first, so these cost device time only for
    configurations the harness would have accepted.

    ADDITIVE, not a reallocation, and the measurement is why. Taking the L1-only
    slot out of `k` costs the high-occupancy corner: over twenty seeds the
    fraction of selections reaching `l0a > 0.5` went 6/20 -> 1/20 at k=3, 9/20 ->
    6/20 at k=4. Both slices are part of the space this module exists to span, so
    robbing one for the other is not coverage; k=3 is simply too small to hold
    both. The extra point is drawn only when the kernel HAS a cube site, so a
    vector-only kernel pays nothing and gets nothing -- and on a cube kernel it
    costs one device trial per block.
    """
    opts = opts or CoverOptions()
    if k <= 0 or not sites or n_dims(sites, anchor) < MIN_DIMS:
        return []
    rng = random.Random(opts.seed)
    seen = {_key(c) for c in opts.exclude}
    if anchor:
        seen.add(_key(anchor))

    def _keep(rows):
        out = []
        for cfg in rows:
            key = _key(cfg)
            if key in seen or not space.static_feasible(cfg, hw)[0]:
                continue
            seen.add(key)
            out.append(cfg)
        return out

    # ONE slot reserved for the L1-only family, because farthest-point cannot
    # reach it: those points hold the incumbent's L0, so their L0A / L0B / L0C
    # occupancies equal the anchor's exactly and only the ratio features move.
    # They sit next to the anchor by construction, and a rule that maximises
    # distance from what is already chosen will never take one however many are
    # in the pool -- measured: putting them in the pool changed nothing, 0 of 60
    # across twenty seeds.
    #
    # The family is kept OUT of the farthest-point candidate set entirely, not
    # merely picked first from inside it. Sixty-four points all sharing the
    # anchor's L0 move the min/max that `_normalise` scales by, and that shifted
    # the free walk enough to lose the high-occupancy corner: with them in the
    # set the six selected points topped out at l0a = 0.50 while the pool held
    # twenty-four above it. The free walk must see exactly what it saw before,
    # one slot smaller.
    # Its OWN random stream. Sharing `rng` would consume draws before
    # `_sample_space` runs, so the free walk would see a different pool than it
    # saw without this family -- and "additive" has to mean the free selection is
    # bit-for-bit what it was, not merely the same size.
    l1_cands = _keep(
        _l1_only(sites, anchor, random.Random(opts.seed ^ 0x5EED), min(opts.pool, 64))
    )
    # Drawn, not maximised. Taking the member FURTHEST from the anchor picks the
    # same corner every time -- twenty seeds returned four multiplier triples,
    # all at the top of the ladder, which is the crowding this module exists to
    # remove reproduced one level down. One slot cannot cover forty-seven
    # configurations; it can sample a different one per block, and a campaign
    # fires six to nine blocks.
    reserved = [random.Random(opts.seed ^ 0x5EED).choice(l1_cands)] if l1_cands else []

    cands = _keep(_sample_space(sites, anchor, opts.domain, opts.pool, rng))
    if not cands:
        return reserved

    base = [features(anchor, hw, opts.domain)] if anchor else []
    vecs = _normalise(base + [features(c, hw, opts.domain) for c in cands])
    chosen_vecs = vecs[: len(base)]
    pool_vecs = vecs[slice(len(base), None)]

    picked, avail = [], list(range(len(cands)))
    while avail and len(picked) < k:
        if chosen_vecs:
            i = max(
                avail, key=lambda j: min(_dist2(pool_vecs[j], c) for c in chosen_vecs)
            )
        else:
            i = avail[0]
        picked.append(cands[i])
        chosen_vecs.append(pool_vecs[i])
        avail.remove(i)
        # Farthest-point walks to the extremes first and then, having reached a
        # corner, is free to pick a second point in the SAME corner: the run
        # that motivated this module chose two seeds at l0a=0.09/l0b=0.09/
        # l0c=0.01, which is the duplicate-operating-point waste it exists to
        # remove. Drop what is now indistinguishable from something chosen.
        avail = [
            j
            for j in avail
            if min(_dist2(pool_vecs[j], c) for c in chosen_vecs) > _MIN_SEP2
        ]
    # Reserved point first: if the block's budget is cut short, the one that no
    # other mechanism can produce should not be the one dropped.
    return reserved + picked


def _key(cfg):
    return tuple(
        sorted(
            (sid, tuple(sorted(v.items())))
            for sid, v in cfg.items()
            if isinstance(v, dict)
        )
    )
