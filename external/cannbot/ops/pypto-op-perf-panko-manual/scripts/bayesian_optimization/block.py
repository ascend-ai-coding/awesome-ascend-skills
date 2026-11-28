# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""A tile-tuning block: the core owns its own loop and returns one configuration.

The lever this replaces asked the study for a single candidate, handed it back to
the optimizer, and was asked again on the next cycle. That design put the tile
values inside the ratchet one at a time, which cost it twice. `stagnation_K`
counted BO trials, so an action closed after seven of them and TPE never warmed
up. And every intermediate value was judged by the ratchet on its own, so a tile
that only pays off together with a structural change could not be reached: the
structure is submitted at the stale tile, loses, and is reverted before the tile
that would have vindicated it is ever tried. A hand-optimised kernel is
typically exactly that pair -- a structural rewrite AND the tile that suits it,
neither of which wins alone.

A block runs the whole loop against a fixed base program and submits only the
winner. What reaches the ratchet is the pair.

    request  -> ask / apply / E(x) / tell,  n times, base restored between trials
             -> best_config, best_latency, and a summary

Two entry points, one mechanism: a tile action being selected, and the optimizer
asking for tuning after a structural change. The loop is identical; only what
happens afterwards differs.

Stopping
--------
Seven consecutive non-improving trials -- `stagnation_K`, the same number the
rest of the harness uses -- and nothing else. A trial cap looks like prudence
against the noise floor, on the reasoning that a large fraction of
"improvements" are draws rather than gains and a stagnation counter could be
reset indefinitely. It cannot be: the bar a trial must beat is the running
minimum, which only ever falls, so under pure noise the chance that trial n sets
a new record is about 1/n and seven consecutive non-records arrive quickly. A cap
would truncate real improvement to guard against a runaway the arithmetic rules
out.

The remaining evaluation budget bounds the block, but that is an accounting
limit and not a quality rule: a block may not start a trial it cannot pay for.

Budget
------
Every trial is a device evaluation and is charged as one. The caller charges the
losing trials and caches their bytes so they can never be paid for twice; the
winner is deliberately left uncached so the ordinary `record` path charges it and
runs the ratchet. The ratchet therefore stays in exactly one place.
"""

import hashlib
import os
import sys

# Aliased. `domain` and `seed` are also parameter names in this file, and a
# parameter shadows a module for the whole function body -- the bare import
# turned `domain_mod.derive(...)` into an UnboundLocalError and
# `seed_mod.cover(...)` into one too. The `bo_` prefix used to keep them apart
# by accident; dropping it removed the accident, so the separation is
# explicit now.
from dataclasses import dataclass
from typing import Any

from . import apply, driver, lever, space
from . import domain as domain_mod, seed as seed_mod


def _read_text(path, errors="replace"):
    """The file's text, with the handle closed before the caller sees it."""
    with open(path, encoding="utf-8", errors=errors) as f:
        return f.read()


def seeds_for(src, hw, dtype_bytes=4, current_latency=None):
    """Warm-start configurations for a block, cheapest evidence first.

    `lever.seed_points` returns [(config, value_or_None, label)]: the current
    tile (already measured, so it costs no device time), the smallest scaling of
    it that clears the documented 16 KB floor, and the largest that still fits
    the buffer. Three grounded observations instead of the sampler's ten random
    draws, which matters because a block that converges in nine trials would
    otherwise spend all of them warming up.
    """
    pts = lever.seed_points(src, hw, dtype_bytes, current_latency)
    warm = [cfg for cfg, val, _ in pts if val is None]
    measured = [(cfg, val) for cfg, val, _ in pts if val is not None]
    # The incumbent is returned SEPARATELY, and this is not tidying.
    #
    # `run_bo` takes its anchor from `warm_start[0]`, and the anchor is what the
    # categorical domains are built around and what the capacity `floor` guard
    # measures. The incumbent is the only seed that carries a latency -- it is
    # already running -- so in production it lands in `measured` and never in
    # `warm`, and the anchor silently became the FLOOR seed: the smallest scaling
    # of the current shape that clears 16 KB.
    #
    # Three mechanisms then guard the wrong program. `floor` is ~16 KB instead of
    # the incumbent's footprint, so a capacity refusal at 24 KB is accepted as a
    # ceiling and everything above it is rejected for free -- including the tile
    # the kernel is running. `space.suggest` anchors the ladders on the floor
    # seed, so the incumbent's own values need not even be in the domain. And the
    # learned L1 relation is checked against the floor seed too.
    #
    # A recorded block shows the shape of it exactly: 42 trials, 35 rejected for
    # free, 7 reaching the device, ONE measurement, best unchanged at the preopt
    # -- and `ceiling=24576`, well under a [128,128] fp32 cube tile. The block was
    # searching a corner the incumbent is not in and could not have won from
    # there. `best_was_incumbent=true, winner_hash=null` in the earlier runs is
    # the same failure reported as a result.
    cur = next((cfg for cfg, _, lbl in pts if lbl == "current"), None)
    return warm, measured, cur


def memory_key(action_id, domain):
    """Which earlier block's observations this one may reuse.

    Keyed by the DERIVED TILE DOMAIN, not by the program text. This module used
    `lever.structural_signature`, which hashes the program with the tile
    literals erased -- unchanged by a tile value and changed by anything else at
    all. The repository already argues that this is the wrong test, in
    `domain.fingerprint`:

        `lever.structural_signature` is the obvious test and the wrong one,
        because it hashes the program with the tile literals erased, so a
        renamed variable or an added comment changes it [...] What actually
        invalidates a tile optimum is a change to the inputs of the derivation.

    That argument was made for the retune trigger and never carried across to
    the memory, so every non-tile edit threw the study away. The two questions
    are different -- "should another block run?" versus "are these measurements
    still admissible?" -- but they have the same answer, and it is the domain.

    WHAT THIS FIXES. Keyed on the source hash, the great majority of tile
    blocks begin at `replayed=0`, and a large share of those cold starts have no
    tile-space change behind them at all: a `pass_options` line, a reordered
    statement, a renamed intermediate. The ratchet rewrites the incumbent after
    every kept candidate, so the source hash moves constantly while the space the
    study searches stands still. Those blocks now inherit. So does the case the
    optimizer writes down and does not get -- an extent revisited later, which
    the source hash delivered only when the whole file happened to be
    byte-identical modulo tiles.

    WHAT THIS DOES NOT FIX, stated because it is the larger half. The other four
    fifths are an action whose own delta moves the view extent -- a granularity
    sweep walking TILE_B through a dozen values. There the domain genuinely
    differs, this key misses on purpose, and it should: latency is not comparable
    across extents, so replaying a measurement taken over ten times the work
    would teach the surrogate that a bigger block is a worse tile. That case
    needs a work-normalised objective and joint tile/extent search. It is not a
    key problem and this change does not pretend to address it.
    """
    fp = domain_mod.fingerprint(domain)
    return f"{action_id}|{hashlib.sha256(fp.encode('utf-8')).hexdigest()[:12]}"


@dataclass
class BlockOptions:
    """How one block runs, as one value. See driver.SearchOptions -- same
    reason, one level up: `run` takes the program, the objective and the
    envelope, and this.
    """

    stagnation_k: int = 7  # non-improving trials that close the block
    max_trials: Any = None  # the remaining evaluation budget
    current_latency: Any = None  # what the incumbent measured, if known
    dtype_bytes: int = 4
    on_trial: Any = None
    seed: int = 0
    domain: Any = None
    memory: Any = None
    startup_min_feasible: int = 5
    startup_cover: int = 3  # space-covering points before TPE draws
    refusals: Any = None


def _search_options(opts, warm, incumbent, sites):
    """The driver's options for one block, translated from BlockOptions."""
    return driver.SearchOptions(
        warm_start=warm,
        anchor=incumbent,
        n_trials=opts.max_trials if opts.max_trials is not None else 10**6,
        seed=opts.seed,
        on_trial=opts.on_trial,
        stagnation_limit=opts.stagnation_k,
        domain=opts.domain,
        dtype_bytes=opts.dtype_bytes,
        memory=opts.memory,
        startup_min_feasible=opts.startup_min_feasible,
        # Run level, not structure level: what the compiler will not accept about
        # a tile call outlives the program shape that first asked it.
        refusals=opts.refusals,
        # The row extent of vec scopes inside one loop body is a hand-off, not a
        # per-site choice. Left untied, the sampler draws mismatched extents,
        # and those are reliably among the worst configurations of a run,
        # against an otherwise identical coherent baseline.
        tie=apply.tie_groups(sites, incumbent),
    )


def _seeded_warm_starts(src, hw, opts, sites):
    """The warm-start set for one block: the incumbent scaled up and down, plus
    space-covering points, filtered against the derived domain.

    `warm` starts as the incumbent scaled up and down -- two points on one line
    through the space. Everything else the block measures before TPE has a
    surrogate comes from uniform draws over the PARAMETER domains, and those
    cluster: one recorded block spent two of its six measurements on mL0 192 /
    kL0 32 / nL0 32 at 22,198 us, and mL0 512 / kL0 64 / nL0 32 (14,114 us),
    which are the same operating regime described twice.

    `seed.cover` replaces the random startup with points chosen to sit far
    apart in the space of things that can be computed without the device --
    buffer occupancies, trip counts, trips per core. It asserts nothing about
    which is fast: every predictor tried on this benchmark has been refuted by
    measurement. It only guarantees the device is asked a different question
    each time.
    """
    warm, measured, incumbent = seeds_for(
        src, hw, opts.dtype_bytes, opts.current_latency
    )
    if opts.startup_cover:
        warm = warm + seed_mod.cover(
            sites,
            hw,
            incumbent,
            opts.startup_cover,
            seed_mod.CoverOptions(
                domain=opts.domain,
                seed=opts.seed,
                exclude=warm + ([incumbent] if incumbent else []),
            ),
        )
    # Seeds are drawn from the current shape by scaling, so one can land outside
    # the derived domain. Dropping it is right -- it would be rejected on the
    # first draw anyway -- but the incumbent is never dropped, because a domain
    # that excludes the running program is describing a different kernel.
    if opts.domain:
        kept = []
        for c in warm:
            if domain_mod.verify(opts.domain, c, opts.dtype_bytes)[0]:
                kept.append(c)
        warm = kept
    return warm, measured, incumbent


def run(op_file, objective, hw, opts=None):
    """Run one block against the program currently in `op_file`.

    objective : site_configs -> (s, p). In production this is a
                `evaluator.RealEvaluator`, which applies each config on top of
                a base captured at construction and restores that base
                afterwards, so trials never contaminate each other. In tests it
                is a plain function.
    max_trials: the remaining evaluation budget. None means unbounded, which is
                only correct in tests.
    startup_cover: how many space-covering points to measure before TPE's own
                draws. 0 restores the previous behaviour (scaling seeds only).

    Returns the `run_bo` dict, plus `reason` when no trial could be run at all.
    """
    opts = opts or BlockOptions()
    src = _read_text(op_file)
    try:
        sites = [s for s in apply.tunable_sites(src) if s.kind in ("cube", "vec")]
    except SyntaxError as e:
        return _empty(f"kernel does not parse: {e}")
    if not sites:
        # The single most consequential outcome, so it is named rather than
        # returned as an empty result: a kernel whose tile call is written with
        # named constants has no tunable site, and normalisation at INIT is what
        # fixes it. Silence here is what a whole recorded run looked like.
        return _empty(
            "no tunable tile site; the tile call is not written with "
            "integer literals (normalise at INIT)"
        )
    if opts.max_trials is not None and opts.max_trials <= 0:
        return _empty("no evaluation budget remaining")

    warm, measured, incumbent = _seeded_warm_starts(src, hw, opts, sites)
    result = driver.run_bo(
        objective,
        sites,
        hw,
        _search_options(opts, warm, incumbent, sites),
    )
    # An already-measured configuration is evidence and costs nothing, so it
    # competes with what the block found rather than being discarded.
    for cfg, val in measured:
        if result["best_latency"] is None or val < result["best_latency"]:
            result["best_config"], result["best_latency"] = cfg, float(val)
            result["best_was_incumbent"] = True
    result.setdefault("best_was_incumbent", False)
    return result


def apply_best(src, best_config):
    """(new_src, n_sites_rewritten) for the winning configuration.

    `apply` rewrites every tile site or none. A coder that changed the kernel
    body but not the wrapper's padding arithmetic once shipped a kernel that
    failed verification, the action was scored unproductive and closed, and
    the path it had opened was lost to that one bookkeeping mistake.
    """
    if not best_config:
        return src, 0
    return apply.apply(src, best_config)


def _empty(reason):
    return {
        "best_config": None,
        "best_latency": None,
        "n_trials": 0,
        "n_feasible": 0,
        "stopped_early": False,
        "history": [],
        "best_was_incumbent": False,
        "aborted": None,
        "fault_kind": None,
        "learned_ceiling": None,
        "env_faults": 0,
        "failures": 0,
        "replayed": 0,
        "memory": {},
        "refusals": [],
        "reason": reason,
    }


def _merge_bounds(op_file):
    """Per-pass subgraph ceilings read off the newest trace, or {}.

    N subgraphs cannot be merged in groups larger than N, so the profiler's own
    `subGraphCount` bounds the merge granularities worth proposing -- and it can
    rule a pass out entirely: when every `cubeMergeInfo` group in the trace
    holds exactly one subgraph, that domain collapses to "do not merge" without
    spending a trial to discover it.

    Best-effort and silent. A missing trace means the documented ladder stands;
    it must not mean the block refuses to run. The trace is tens of megabytes,
    so this is a per-block cost, not a per-trial one.
    """
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import swimlane

        trace = swimlane.find_trace(os.path.dirname(op_file or ""), os.getcwd())
        return swimlane.subgraph_counts(trace) if trace else {}
    except Exception:
        return {}


# space.HW's field name for each chip-envelope key. The two differ because the
# envelope is named after the chip's own datasheet and HW after what the static
# check calls the same quantity.
_HW_FROM_ENVELOPE = (
    ("ub_budget_kb", "ub_kb"),
    ("l1_budget_kb", "l1_kb"),
    ("l0a_kb", "l0a_kb"),
    ("l0b_kb", "l0b_kb"),
    ("l0c_kb", "l0c_kb"),
    ("cube_cores", "cube_cores"),
    ("vector_cores", "vector_cores"),
)


def hw_for(op_file, live_tiles=None, envelope=None):
    """The static envelope and the derived domain a block validates against.

    Returns (hw, domain). `live_tiles` is the one fact the core cannot read: how
    many tile-shaped tensors are simultaneously resident is a dataflow property,
    and after the model has fused or flattened anything it is a property of a
    program the extractor has never seen. It is supplied by the optimizer as a
    fact and never as a bound.

    `calibrated` stays False because the op-specific L1 numbers (mL1/kL1/nL1)
    come from a DESIGN this module does not read. The capacity rule is not behind
    that flag -- it needs only the program and the chip.

    `envelope` is the chip's buffer capacities, supplied by the caller so that
    the tile search and the static feasibility gate validate against the SAME
    numbers. They did not: this function passed none, so the search used the
    dataclass defaults while `--ub-kb` / `--l1-kb` moved only the other gate.
    None keeps the defaults, for a caller that has no envelope to hand.
    """
    env = envelope or {}
    domain = domain_mod.derive(op_file, live_tiles, ub_budget_kb=env.get("ub_kb", 192))
    bounds = _merge_bounds(op_file)
    if bounds:
        domain["subgraph_counts"] = bounds
    kw = {field: env[key] for field, key in _HW_FROM_ENVELOPE if key in env}
    hw = space.HW(
        calibrated=False,
        dtype_bytes=domain.get("dtype_bytes", 4),
        n_live_tiles=domain.get("live_tiles", 0),
        **kw,
    )
    return hw, domain
