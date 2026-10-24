# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Per-action Bayesian optimization over one kernel's tile sites.

The difference from `bo_stage7_init`, which runs one closed BO loop at INIT over
the whole numeric space and then hands off: this is a lever the search pulls
*inside* an action's refinement loop. Every time a tile action is selected the
core asks the study for one candidate, writes it, and reports the measurement
back. The two designs answer different questions and the earlier one is not
replaced.

Three reasons it belongs inside the loop rather than in front of it:

  * the optimal tile depends on the structure. After the model fuses two ops or
    changes a layout the tile chosen at INIT is stale, and a two-phase design has
    no way to revisit it.
  * budget. `run_bo` defaults to 40 trials; the whole run gets 90 device runs.
    Inside the loop BO spends nothing extra, because the refinement loop was
    going to burn those candidates on model guesses anyway.
  * the core ends up owning the number, which is what PANKO claims and what, for
    tile shapes, it has not been doing.

An Optuna study lives in one process and a PANKO run is a sequence of dispatches,
so the study is rebuilt from `search_state.json` on every call. `ask` / `tell`
rather than `study.optimize`, because the evaluation happens in the optimizer's
own cycle, not here.
"""

import hashlib
import math
import re

from dataclasses import dataclass
from typing import Any

from . import apply, space

PENALTY = 1e12  # same sentinel as driver: any real latency is far below

# The four catalogue actions this lever serves, each with the site kind it is
# DEFINED over. Twenty of the forty-eight actions carry a number, but only these
# four are tile shapes, and tile shapes are the only lever where the payoff has
# been measured: on the operators recorded so far, the diff from the root
# program to the best program contains nothing else.
#
# The scope is enforced, not documentation. The lever used to tune every cube AND
# vec site whichever of the four was selected, so picking the Cube action could
# change a Vector tile and the gain it earned was booked to the Cube action. The
# catalogue's own wording ("Cube TileShape", "Vector tile shapes") was then a
# claim about the action that the search did not honour.
SCOPE_OF = {"F-9": "cube", "S-11": "cube", "F-10": "vec", "S-12": "vec"}
TILE_ACTIONS = set(SCOPE_OF)


def scope_of(catalog_id):
    """The site kind an action may move. Unknown ids get None, meaning "all tile
    sites" -- the union search, which is what `block` does deliberately.
    """
    return SCOPE_OF.get(catalog_id)


def _scoped(sites, scope):
    return [s for s in sites if scope is None or s.kind == scope]


# `basic-block-optimization.md`, the Vector TileShape section, rule 3: the data block
# should sit between 16 and 64 KB. Only the floor is taken from that document.
# Real kernels beat its ceiling -- RMSNorm is fastest at 168 KB -- so the upper
# end is not a bound we can trust.


FLOOR_KB = 16

# The ceiling comes from the buffer instead. `feasibility.py` charges the unified
# buffer `view + n_buffers x tile`, so a tile alone can never usefully exceed
# UB/2 whatever the view costs. The old ceiling seed was "the top of the domain
# that `space` accepts", and `space` does not model the view: against a large
# view that proposes a tile whose total charge is well over the buffer, which
# the harness gate then refuses. Free, but a wasted seed out of three.
#
# UB is the RUN'S, not a constant. The default below is half of one SKU's buffer,
# and another SKU's is larger -- so hardcoding it would refuse tiles that chip
# holds, the same defect the chip envelope exists to remove, surviving in the one
# module that had its own copy of the number. It is used only when no envelope
# reaches here.
CEIL_KB = 96


def ceil_kb(hw=None):
    ub = getattr(hw, "ub_budget_kb", 0) or 0
    return int(ub // 2) if ub else CEIL_KB


# Seed family, expressed in FOOTPRINT rather than in per-dimension scale. A
# factor here multiplies the tile's byte count; each dimension moves by the
# factor's n-th root, so a 1-D and a 4-D tile sweep the same range of footprints.
#
# Scaling each dimension by 2 instead, as the first version did, multiplies a 2-D
# tile's bytes by 4 and a 4-D tile's by 16. That can leave nothing at all
# between 4 KB and 64 KB, so the floor seed (smallest >= 16 KB) and the ceiling
# seed (largest <= 96 KB) collide on one point and the ceiling falls back to
# 4 KB -- a "ceiling" sixteen times smaller than the floor.
#
# The steps are x1.5 rather than x2, for a reason that is arithmetic rather than
# aesthetic. Both seeds are drawn from the band FLOOR_KB..CEIL_KB, which is
# 16..96 KB: 2.58 octaves. At x2 that band holds at most three rungs, and holds
# fewer as soon as one is deduplicated against the current footprint. Run against
# the fifteen Stage-6 kernels in this repository, x2 gave:
#
#   8 kernels   generated at 64 KB, the DESIGN.md cube default. 64x2 = 128 KB is
#               outside the band, so the only in-band neighbour was DOWNWARD and
#               the "ceiling" came out at 36 KB, below the current tile.
#   4 kernels   one in-band rung, so a floor seed and no ceiling at all.
#
# At x1.5 the band holds five to six rungs, 64 KB reaches 96 KB, and every one of
# the fifteen gets both seeds on the correct sides.


BYTE_FACTORS = (
    1 / 64.0,
    1 / 48.0,
    1 / 32.0,
    1 / 24.0,
    1 / 16.0,
    1 / 12.0,
    1 / 8.0,
    1 / 6.0,
    1 / 4.0,
    1 / 3.0,
    1 / 2.0,
    2 / 3.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    8.0,
    12.0,
    16.0,
    24.0,
    32.0,
    48.0,
    64.0,
    96.0,
    128.0,
    192.0,
    256.0,
)

_NUM = re.compile(r"-?\d+")


# --------------------------------------------------------------------------- keys
def structural_signature(src):
    """Hash of the program with every tile literal erased.

    The ratchet restores the global best before each action, so a tile lever
    selected twice may be looking at two different programs. Observations taken
    on the old structure are not evidence about the new one. This changes when
    the structure changes and not when a tile value does, which is exactly the
    condition for reusing a study.

    Only the tile CALL's arguments are erased, which is exactly what
    `apply.apply` rewrites, so a BO step never changes the signature. That
    holds at every arity: a 1-argument call and a 4-argument call are both erased
    down to their `<vec>` marker.

    What is still NOT erased is a tile driven through a symbol
    (`set_vec_tile_shapes(TILE)` with `TILE = 8192` bound elsewhere). Its binding
    site is an ordinary assignment. `apply` classifies that call `vec_dyn` and
    leaves it alone, so the two limitations still cancel -- but a literal
    1-argument call, `set_vec_tile_shapes(8192)`, is now tuned.
    """
    try:
        sites = apply.discover_sites(src)
    except SyntaxError:
        return "unparseable"
    out, cur = [], 0
    for s in sorted(sites, key=lambda x: x.start):
        out.append(src[slice(cur, s.start)])
        out.append(f"<{s.kind}>")
        cur = s.end
    out.append(src[cur:])
    return hashlib.sha256("".join(out).encode("utf-8")).hexdigest()[:12]


def lever_key(catalog_id, src):
    """Keyed by SCOPE, so the two actions over one space share one study.

    F-9 and S-11 are both "tune the Cube tile"; keyed by action they kept
    separate studies of the same space and re-explored it from scratch, paying
    device evaluations twice for one surrogate. The structural signature still
    isolates them per program, which is the separation that has evidence behind
    it. An unknown id keys by itself, so nothing silently pools into a scope.

    The format changed, so records written under the old `<action>|<sig>` keys
    no longer match and their studies start cold. That is deliberate: those
    trials were drawn from the unscoped space and replaying them into a scoped
    study would describe points it cannot propose.
    """
    return f"{scope_of(catalog_id) or catalog_id}|{structural_signature(src)}"


# ------------------------------------------------------------------------- seeding
# Site-id prefixes that are NOT tile sites. They matter here because the
# `pass_options` encoding collides with the vec encoding: a pass site is stored
# as {"d0": <dict key>, "d1": <dict value>}, so `cube_l1_reuse_setting={0: 8}`
# is shaped exactly like a two-dimensional vec tile of (0, 8). A shape-only test
# cannot tell them apart, and scaling that "tile" as a footprint hands 0 to
# `math.log` inside `_snap` -- which killed the entire block, before its first
# trial, on any kernel carrying a pass option. Kinds are what actually separate
# these (`apply.tunable_sites().kind`), and the id carries the kind.
_NON_TILE_PREFIXES = ("pass:", "g:", "unroll")


def _is_tile(c, sid=""):
    if sid.startswith(_NON_TILE_PREFIXES):
        return False
    return any(k.startswith("d") for k in c) or "mL0" in c


def _tile_bytes(cfg, dtype_bytes):
    """Elements in the largest site, times the dtype. Cube sites report their L0A.

    Arity-agnostic: the vec footprint is the product of however many dimensions
    the call carries, which is the same quantity `feasibility.py` charges to the
    unified buffer.
    """
    n = 0
    for c in cfg.values():
        fields = [k for k in c if k.startswith("d")]
        if fields:
            p = 1
            for f in fields:
                p *= c[f]
            n = max(n, p)
        elif "mL0" in c:
            n = max(n, c["mL0"] * c["kL0"])
    return n * dtype_bytes


def _snap(value, domain):
    """Nearest ladder rung on a log scale.

    Seeds have to be points the sampler could also have drawn: Optuna refuses to
    replay an observation whose value is outside the categorical domain, so a
    seed built by arbitrary arithmetic cannot be added to the study at all.
    Snapping is what keeps a scaled tile inside the ladder.
    """
    return min(domain, key=lambda c: (abs(math.log(c) - math.log(max(value, 1e-9))), c))


def _cube_l1(before, after, ax):
    """L1 for `ax` once its L0 has moved, holding the multiplier the kernel had.

    A cube axis is DRAWN as (L0, L1/L0) exactly so that `0 < L0 <= L1 and
    L1 % L0 == 0` holds by construction, and `_dist` rebuilds that multiplier's
    domain from the ANCHOR -- `L1_MULTS` plus whatever ratio the kernel itself
    carries. A generator that scales L0 and leaves L1 where it was therefore
    invents a ratio the study cannot replay: shrinking a 128/128 axis to 16
    records a multiplier of 8 against a domain of (1, 2, 3, 4), and `_rebuild`
    raises ValueError on the NEXT call to `ask` -- the lever is dead for the
    rest of the run, one call after a seed it accepted.

    This never fired while the lever could see only one site per kernel. It
    became reachable the moment `tunable_sites` stopped dropping the tiles
    written inside helper functions, because that is what first put a cube site
    and a vec site in the same draw.
    """
    l0, l1 = before.get(f"{ax}L0"), before.get(f"{ax}L1")
    if not l0 or not l1 or l1 % l0:
        return l1
    return after[f"{ax}L0"] * (l1 // l0)


def _scaled(cfg, byte_factor):
    """Multiply every tile site's FOOTPRINT by `byte_factor`, then snap to ladders.

    Every dimension moves, not the leading one only. With the leading dimension
    pinned at 1 -- 67 of the 313 recorded tile writes, the most common case there
    is -- a leading-only family contains exactly one point and the lever has
    nothing to propose. Each dimension takes the factor's n-th root so the shape
    keeps the aspect ratio the generator chose while the footprint moves by the
    requested amount; the sampler explores aspect ratio afterwards.
    """
    out = {}
    for sid, c in cfg.items():
        d = dict(c)
        fields = sorted((k for k in c if k.startswith("d")), key=lambda k: int(k[1:]))
        if fields:
            f = byte_factor ** (1.0 / len(fields))
            for i, k in enumerate(fields):
                dom = (
                    space.VEC_TRAILING_CHOICES
                    if i == len(fields) - 1
                    else space.VEC_CHOICES
                )
                d[k] = _snap(c[k] * f, space.admit(dom, c[k]))
        elif "mL0" in c:
            f = byte_factor**0.5  # the L0A footprint is mL0 x kL0
            for k in ("mL0", "kL0", "nL0"):
                d[k] = _snap(c[k] * f, space.admit(space.CUBE_CHOICES, c[k]))
            for ax in ("m", "k", "n"):
                d[f"{ax}L1"] = _cube_l1(c, d, ax)
        out[sid] = d
    return out


def _rung(value, domain, step):
    """The value `step` rungs from `value` on `domain`, clamped at both ends."""
    d = sorted(domain)
    i = min(
        range(len(d)),
        key=lambda j: (abs(math.log(d[j]) - math.log(max(value, 1e-9))), d[j]),
    )
    return d[max(0, min(len(d) - 1, i + step))]


def _nudged(cur, idx, step):
    """Move ONE dimension by `step` rungs, leaving the others where they are.

    Uniform scaling cannot make a fine footprint change to a multi-dimensional
    tile. Every dimension snaps to a x1.5 ladder, so moving all of them together
    shifts a 2-D footprint by x2.25 at best and a 3-D one by x3.4. On the eight
    Stage-6 kernels generated at the 64 KB DESIGN.md cube default that put the
    whole 64..96 KB range out of reach: the nearest in-band neighbour was
    downward, and the "ceiling" seed came out at 36 KB, below the current tile.
    Widening BYTE_FACTORS to x1.5 steps did not help, because the limit is the
    ladder the dimensions snap to, not the factors.

    One dimension, one rung, is a x1.5 footprint step at any arity.

    Returns None when the nudge changes nothing (already clamped at an end).
    """
    out, changed = {}, False
    for sid, c in cur.items():
        d = dict(c)
        fields = sorted((k for k in c if k.startswith("d")), key=lambda k: int(k[1:]))
        if fields and idx < len(fields):
            f = fields[idx]
            dom = (
                space.VEC_TRAILING_CHOICES
                if idx == len(fields) - 1
                else space.VEC_CHOICES
            )
            d[f] = _rung(c[f], space.admit(dom, c[f]), step)
            changed |= d[f] != c[f]
        elif "mL0" in c and idx < 3:
            ax = ("m", "k", "n")[idx]
            k = f"{ax}L0"
            d[k] = _rung(c[k], space.admit(space.CUBE_CHOICES, c[k]), step)
            changed |= d[k] != c[k]
            d[f"{ax}L1"] = _cube_l1(c, d, ax)
        out[sid] = d
    return out if changed else None


def _family(cur, dtype_bytes):
    """[(tile_bytes, config)] around the current tile, deduplicated, small to large.

    Two generators, because they cover different scales. The uniform scalings
    reach far -- 1/64 to 256 of the current footprint, which is what a kernel
    generated at 0.3 KB needs to get to 16 KB. The single-dimension nudges are
    fine, x1.5 per step, which is what a kernel generated at 64 KB needs to reach
    96 KB without overshooting to 144.
    """
    seen = {_tile_bytes(cur, dtype_bytes)}
    fam = []

    def add(cfg):
        if cfg is None:
            return
        nb = _tile_bytes(cfg, dtype_bytes)
        if nb in seen:
            return
        seen.add(nb)
        fam.append((nb, cfg))

    for f in BYTE_FACTORS:
        add(_scaled(cur, f))
    for idx in range(apply.MAX_VEC_ARITY):
        for step in (1, -1, 2, -2):
            add(_nudged(cur, idx, step))
    fam.sort(key=lambda x: x[0])
    return fam


def seed_points(src, hw, dtype_bytes=4, current_latency=None, scope=None):
    """Three grounded starting observations instead of ten random trials.

    TPE takes `n_startup_trials = 10` uniformly random draws before its
    acquisition step does anything, and actions in the recorded runs close after
    9 to 23 refinements. The default would spend most of a lever's life sampling
    at random, and an action that closes at 9 would never reach BO at all.

        current   the global best is running it and its latency is recorded,
                  so this observation costs nothing
        floor     the smallest scaling of the current shape whose tile clears
                  the documented 16 KB lower bound
        ceiling   the largest scaling whose tile still fits the buffer, at
                  UB / 2 buffers for the chip this run was initialised for. The
                  static gate runs before the device, so a refusal here is also
                  free.

    `scope` restricts the seeds to one site kind, for the four catalogue actions
    that are defined over one. None keeps every tile site, which is the union
    search `block` asks for on purpose.

    Returns [(config, value_or_None, label)]; None means "not measured yet".

    Both bounds are on the tile FOOTPRINT, not on any one dimension, which is
    what makes them arity-independent: a 1-argument tile and a 4-argument tile
    are bounded by the same buffer.
    """
    cur = apply.current_config(src)
    cur = {
        k: v
        for k, v in cur.items()
        if _is_tile(v, k) and (scope is None or k.startswith(scope + "#"))
    }
    if not cur:
        return []
    out = [(cur, current_latency, "current")]
    fam = _family(cur, dtype_bytes)
    if not fam:
        return out

    # Both seeds are drawn from the SAME legal band, so the ceiling can never
    # come out below the floor and neither can be a footprint the buffer cannot
    # hold. Bounding the floor from above matters as much as bounding the
    # ceiling: without it the floor seed can come out many times larger than
    # the buffer holds, which the harness gate then refuses.
    lo, hi = FLOOR_KB * 1024, ceil_kb(hw) * 1024
    band = [
        (nb, cfg)
        for nb, cfg in fam
        if lo <= nb <= hi and space.static_feasible(cfg, hw)[0]
    ]
    if band:
        out.append((band[0][1], None, "floor"))
        if len(band) > 1:
            out.append((band[-1][1], None, "ceiling"))
    return out


# --------------------------------------------------------------------------- study
def _rebuild(record, sites, groups, anchor, seed=0):
    """Reconstruct the Optuna study from the trials persisted in the state file."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        # 3, not the default 10. Two of the three seeds are already observations
        # by the time the sampler is consulted, so the acquisition step is doing
        # work from the fourth candidate rather than the eleventh.
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=3),
    )
    for t in record.get("trials", []):
        if t.get("value") is None:
            continue
        study.add_trial(
            optuna.trial.create_trial(
                params=t["params"],
                distributions={k: _dist(k, sites, groups, anchor) for k in t["params"]},
                value=float(t["value"]),
            )
        )
    return study


def _dist(name, sites, groups, anchor):
    """The distribution a persisted param was drawn from. space.suggest names
    parameters `<rep_site_id>.<field>`, and every field is categorical.
    """
    import optuna

    sid, field = name.rsplit(".", 1)
    kind = space.kind_of(sites, sid)
    a = (anchor or {}).get(sid, {})
    if kind == "cube":
        # Cube params are `<axis>L0` (a tile value) and `<axis>L1x` (the L1/L0
        # multiplier). They are drawn from different ladders, so the replayed
        # distribution has to match by suffix, and the anchor's own value keeps a
        # warm start reachable.
        ax = field[0]
        if field.endswith("L1x"):
            base = space.mult_dom(a.get(f"{ax}L0"), a.get(f"{ax}L1"))
            val = (
                a.get(f"{ax}L1", 0) // a["{}L0".format(ax)]
                if a.get(f"{ax}L0")
                and a.get(f"{ax}L1")
                and a[f"{ax}L1"] % a[f"{ax}L0"] == 0
                else None
            )
            return optuna.distributions.CategoricalDistribution(space.admit(base, val))
        base = space.CUBE_CHOICES
    elif field.startswith("d"):
        # Which rung ladder depends on the position, and the position depends on
        # the arity, which only the anchor knows. Trailing dimension: 16-aligned
        # values only, matching what `suggest` drew from.
        fields = space.vec_fields(a)
        base = space.VEC_TRAILING_CHOICES if field == fields[-1] else space.VEC_CHOICES
    else:
        base = space.GLOBAL_DOMAINS.get(space.global_knob(sid), [])
    return optuna.distributions.CategoricalDistribution(space.admit(base, a.get(field)))


@dataclass
class AskOptions:
    """What the lever needs beyond the program and the chip: the element width
    the tiles are measured in, what the incumbent costs, and the draw's seed.
    """

    dtype_bytes: int = 4
    current_latency: Any = None
    seed: int = 0


def _fill_seeds(rec, sites, pts):
    """Spend the seed points: record the ones already measured, queue the rest.

    A seed that carries a value was measured by somebody else, so it becomes an
    observation without ever occupying the device.
    """
    rec["seeded"] = True
    rec["queue"] = []
    for cfg, val, label in pts:
        if val is not None:
            rec["trials"].append(
                {
                    "params": space.flatten(cfg, sites),
                    "value": float(val),
                    "label": label,
                    "status": "ok",
                    "free": True,
                }
            )
        else:
            rec["queue"].append({"config": cfg, "label": label})


def _next_seed(rec, sites, scope):
    """The next queued seed as (config, meta), or None when the queue is empty."""
    queue = rec.get("queue") or []
    if not queue:
        return None
    item = queue.pop(0)
    rec["pending"] = {
        "params": space.flatten(item["config"], sites),
        "label": item["label"],
    }
    return item["config"], {
        "label": item["label"],
        "scope": scope,
        "trial": len(rec["trials"]) + 1,
    }


def _draw_one_site(study, ctx):
    """ONE SITE PER DRAW, the rest held at the anchor.

    The gate is an AND over every site, so a draw that moves all of them is
    accepted with probability p**n and the lever dies exponentially in a kernel's
    size. The per-site acceptance rate p is well below 1 -- lowest for a tile
    with many arguments -- so a 2-site kernel clears 32 draws most of the time,
    a 6-site kernel almost never, and a kernel carrying a dozen sites would need
    astronomically many draws for an even chance. That is what `no feasible
    candidate in 32 draws` was: not an exhausted neighbourhood, an impossible
    one. Moving one site at a time makes the acceptance rate p rather than p**n,
    and it is also what a win tends to look like -- one axis moved alone.

    Interactions are not lost, they are deferred: the seed phase still scales
    every site together, and the anchor advances whenever a candidate is kept, so
    the next site is drawn against the winner.

    Returns (trial, config, site id).
    """
    rec, sites, anchor = ctx["rec"], ctx["sites"], ctx["anchor"]
    site = sites[rec.get("cursor", 0) % len(sites)]
    rec["cursor"] = rec.get("cursor", 0) + 1
    held = {site.id: dict(anchor.get(site.id, {}))}
    trial = study.ask(
        {k: _dist(k, sites, None, anchor) for k in space.flatten(held, [site])}
    )
    cfg = dict(anchor)
    cfg.update(space.unflatten(trial.params, [site]))
    return trial, cfg, site.id


def _gate(cfg, hw, dtype_bytes):
    """The free gates a draw must clear. Returns (ok, status, reason).

    The widened ladder spans 1 to 16384 per dimension, so the product reaches 268M
    elements: without the footprint bound the sampler proposes 144 MB tiles.
    `space` cannot catch them -- it validates alignment and the L1 relations, not
    the footprint -- and the harness gate would, but only after charging the
    action a stagnation count each time, so an action could close at K=7 having
    measured nothing at all. The bound is the buffer's: this run's UB over 2
    buffers.
    """
    ok, reason = space.static_feasible(cfg, hw)
    if not ok:
        return False, "infeasible_static", reason
    cap = ceil_kb(hw)
    nb = _tile_bytes(cfg, dtype_bytes)
    if nb > cap * 1024:
        return (
            False,
            "over_buffer",
            (
                f"tile {nb / 1024:.0f}KB > {cap}KB "
                f"({getattr(hw, 'ub_budget_kb', CEIL_KB * 2)}KB UB / 2 buffers)"
            ),
        )
    return True, "ok", reason


def _tpe_candidate(rec, sites, src, scope, hw, opts):
    """Draw from the TPE surrogate until the gate accepts, penalising rejections."""
    anchor = apply.current_config(src)
    in_scope = {s.id for s in sites}
    ctx = {"rec": rec, "sites": sites, "anchor": anchor}
    study = _rebuild(rec, sites, None, anchor, opts.seed)
    tried_sites = []
    for _ in range(32):  # gate rejections are free; keep asking
        trial, cfg, site_id = _draw_one_site(study, ctx)
        tried_sites.append(site_id)
        ok, status, reason = _gate(cfg, hw, opts.dtype_bytes)
        if ok:
            rec["pending"] = {"params": trial.params, "label": "tpe"}
            # Feasibility is judged on the WHOLE program -- the buffer is shared,
            # so a cube tile's legality depends on what the vec sites hold -- but
            # what goes back is the scoped subset. The caller writes what it is
            # handed, and an action that cannot express a change outside its
            # scope cannot make one by accident.
            return (
                {k: v for k, v in cfg.items() if k in in_scope},
                {"label": "tpe", "scope": scope, "trial": len(rec["trials"]) + 1},
            )
        rec["trials"].append(
            {
                "params": trial.params,
                "value": PENALTY,
                "label": "tpe",
                "status": status,
                "reason": reason,
                "free": True,
            }
        )
        study.tell(trial, PENALTY)
    # Name the sites, so a refusal is a fact about the kernel rather than about
    # the lever. Thirty-two single-site draws covering every site and finding
    # nothing means the neighbourhood really is closed; the old message could
    # not distinguish that from the combinatorial collapse above.
    return None, {
        "reason": "no feasible candidate in 32 draws",
        "scope": scope,
        "sites_tried": sorted(set(tried_sites)),
    }


def ask(state_bo, catalog_id, src, hw, opts=None):
    """Next tile config for this lever, or None when there is nothing to tune.

    Returns (config, meta). `meta["label"]` is "current"/"floor"/"ceiling" while
    the seeds are being spent and "tpe" afterwards. The caller writes the config
    with apply and records the outcome through `tell`.
    """
    opts = opts or AskOptions()
    key = lever_key(catalog_id, src)
    scope = scope_of(catalog_id)
    rec = state_bo.setdefault(key, {"trials": [], "seeded": False, "pending": None})
    try:
        sites = apply.tunable_sites(src)
    except SyntaxError:
        return None, {"reason": "unparseable"}
    sites = _scoped([s for s in sites if s.kind in ("cube", "vec")], scope)
    if not sites:
        # Naming the scope matters. "No tunable tile site" on a kernel that has
        # vec sites and no cube ones reads as a broken lever; "no cube tile site"
        # is the Cube action correctly declining a kernel with no Cube work, and
        # the optimizer should retire it rather than retry.
        return None, {
            "reason": f"no {scope} tile site" if scope else "no tunable tile site",
            "scope": scope,
        }

    # --- seeds first -------------------------------------------------------
    if not rec["seeded"]:
        _fill_seeds(
            rec,
            sites,
            seed_points(src, hw, opts.dtype_bytes, opts.current_latency, scope),
        )
    seeded = _next_seed(rec, sites, scope)
    if seeded:
        return seeded

    # --- then the sampler --------------------------------------------------
    return _tpe_candidate(rec, sites, src, scope, hw, opts)


def tell(state_bo, catalog_id, src, s, p):
    """Report a measurement for the candidate `ask` last returned."""
    key = lever_key(catalog_id, src)
    rec = state_bo.get(key)
    if not rec or not rec.get("pending"):
        return False
    pend = rec.pop("pending")
    rec["trials"].append(
        {
            "params": pend["params"],
            "label": pend["label"],
            "value": float(p) if (s and p and p > 0) else PENALTY,
            "status": "ok" if (s and p and p > 0) else "incorrect",
        }
    )
    return True
