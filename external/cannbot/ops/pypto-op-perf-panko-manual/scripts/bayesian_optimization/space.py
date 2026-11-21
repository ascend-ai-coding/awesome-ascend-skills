# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Search space + STATIC hardware feasibility for the numeric config subspace.

MVP levers: cube L0 tile (m_l0/k_l0/n_l0) and vector tile (M/N).

Why a static pre-check exists
-----------------------------
In PANKO the LLM proposed tile numbers blind, so a large fraction of evals were spent
compiling configs that the compiler rejected instantly:
  - ERR_CONFIG_ALIGNMENT  "kL0(8) must be aligned to 16"        (u21)
  - ERR_CONFIG_TILE       "mL0=128 > mL1=64 ..."                (u9)
  - ERR_CONFIG_TILE       "kL0=128 > kL1a=64"                   (u60)
  - ALLOC_FAILED          L1 ~1MB > budget                      (u62)
static_feasible() reproduces these rules so BO never wastes an evaluation on a config the
hardware would reject. It is a CHEAP gate; numerical correctness is still decided later by the
frozen evaluator E(x) (golden compare), never here.
"""

from dataclasses import dataclass
from typing import Any, Optional

# Discrete domains (ordinal). TPE handles categoricals well.
#
# These were not chosen a priori. Across the 313 `set_vec_tile_shapes` calls the
# recorded arms wrote, the leading dimension took
#
#     1  2  4  8  12  16  24  32  40  48  64  96  128  256
#
# and the trailing dimension took
#
#     16  32  64  128  160  256  512  1024  2048  4096  8192  10240  12288  16384
#
# The previous domains -- M in [4..128], N in [16..256] -- excluded 99 of those
# 313 writes, a third of everything the models actually tried. The single most
# common leading value, 1 (67 writes), was not in the domain at all, and
# neither were the large trailing extents that a wide elementwise kernel needs.
#
# One ladder now serves every dimension, because the arity is no longer fixed:
# `set_vec_tile_shapes` is tuned at 1, 2, 3 or 4 arguments and there is no
# position that is always "M". Spacing is x2 with a 1.5x rung between, so the
# ladder spans 1 to 16384 in 25 steps rather than 15 doublings.
VEC_CHOICES = [
    1,
    2,
    4,
    8,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    192,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
    12288,
    16384,
]

# The trailing dimension is the contiguous one and the compiler requires it to be
# 16-aligned (`vec_N not 16-aligned` in the recorded ERR_CONFIG_ALIGNMENT
# failures). Drawing 1, 2, 4 or 8 there is always rejected, so they are removed
# from that position rather than sampled and thrown away. Every trailing value
# ever observed is in this list.
VEC_TRAILING_CHOICES = [c for c in VEC_CHOICES if c % 16 == 0]

# Cube dims must be 16-aligned. Widened in the same spirit: the recorded runs use
# up to 256 and the L1 relation rules, not the domain, are what should bound it.
CUBE_CHOICES = [16, 32, 48, 64, 96, 128, 192, 256, 384, 512]

# L1 is drawn as L0 x multiplier, never independently. Drawing L0 and L1 from the
# same ladder made ~93% of cube configs violate the compiler's per-axis rule
# (0 < L0 <= L1 and L1 % L0 == 0 must hold on all three axes at once -- 7% of
# independent pairs), so almost every device draw was rejected before it could
# measure anything. A multiplier makes the rule hold by construction while keeping
# a FIXED Optuna domain (a per-L0 choice set would be a dynamic domain, which
# Optuna forbids). The config still stores the actual L1 value (m_l1 = mL0 * x), so
# nothing downstream changes.
#
# The range stops at 4 on purpose: L1 holds BOTH operands (A = m_l1*k_l1, B =
# k_l1*n_l1) against a 512 KB buffer, so a multiplier of 8+ blows it on almost any
# L0 (128 x 8 = 1024, and a 1024-square L1 tile alone is 4 MB), and those draws
# were measured to be ~all of the cube L1-buffer rejections. 1..4 spans the useful
# ratios -- the reference and the hand-tuned kernel both sit at 1-2.
L1_MULTS = [1, 2, 3, 4]


def mult_dom(anchor_l0, anchor_l1):
    """Multiplier ladder, plus the incumbent's own L1/L0 ratio so a warm start is
    always reachable.
    """
    base = list(L1_MULTS)
    if anchor_l0 and anchor_l1 and anchor_l1 % anchor_l0 == 0:
        m = anchor_l1 // anchor_l0
        if m not in base:
            base = sorted(set(base) | {m})
    return base


# Global JIT runtime_options scalars (kind="global" sites). Single source of truth for the tunable
# knob names (apply discovers exactly these keys). Domains are hardware-plausible starting sets;
# an invalid value simply fails the correctness gate, so BO learns the feasible ones. Adjust per chip.
GLOBAL_DOMAINS = {
    "device_sched_mode": [0, 1, 2],
    "stitch_function_max_num": [16, 32, 64, 128, 256],
}

# pypto.loop(..., unroll_list=[...]) options. Optuna categoricals must be scalars, so each list is
# encoded as a comma-joined string ("64,16,4" <-> [64, 16, 4]); apply parses it back. An option
# that doesn't fit a given loop simply fails the correctness gate, so BO drops it. (A multi-value
# unroll is admissible at Stage 7 and refused before it.)
UNROLL_CHOICES = ["1", "2", "4", "8", "64,16,4"]  # kept for callers; see unroll_domain

# The five above are five points of a family, not the family: every schedule the
# master Stage-7 skill documents is a strictly-descending list of powers of two,
# and most members of that family are unreachable from a five-item enumeration.
# Generating the family is what makes the space searchable rather than guessed.
#
# Two bounds on it, and both are real rather than tidy:
#   LENGTH. Each value generates one compilation path -- which is why a
#   multi-value unroll is refused before Stage 7 at all -- so the list length IS
#   compile time, which is wall clock, which is budget.
#   TOP. The largest value must not exceed the trip count. Only usable when the
#   trip count is a literal; `pypto.loop(batch)` is a SymbolicScalar and nothing
#   static can bound it. Absent a literal the ladder stays full -- too wide is
#   caught by measurement, too narrow is invisible.
UNROLL_LADDER = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
UNROLL_MAX_SLOTS = 5


def unroll_domain(trip=None):
    """Descending power-of-two unroll schedules, as the comma-joined strings
    `apply` writes. 1023 of them unbounded; a literal `trip` cuts it hard
    (trip=64 leaves 119).
    """
    import itertools

    top = [v for v in UNROLL_LADDER if trip is None or v <= trip] or [1]
    return [
        ",".join(str(v) for v in combo)
        for n in range(1, UNROLL_MAX_SLOTS + 1)
        for combo in itertools.combinations(sorted(top, reverse=True), n)
    ]


# Compiler merge passes, set through `pass_options` on the JIT decorator rather
# than `runtime_options`. They were invisible to this search until now, and one
# of them is worth having: a small `cube_nbuffer_setting` granularity can beat
# the unset kernel, while the value the catalog recommends for its sibling can
# fail to allocate at all and has to be shrunk by hand. A knob whose good value
# is found by hand is a knob the deterministic core should have been holding.
#
# The domain is not a guess. `tune-swimlane/references/merge-optimization.md`
# defines the value: "Value (N) is the merge granularity, every N homogeneous
# subgraphs merge into one; 1 means no merging", with "common values 1/2/4/8/16".
# So these are COUNTS of subgraphs, and the ladder is the documented one.
MERGE_GRANULARITY = [1, 2, 4, 8, 16]

# Integer-key form only: `{-1: N}` applies one granularity to every subgraph.
# The string-key form (`{"func8_0": N}`) targets one hashOrder, which is a
# structural judgement about which subgraphs to merge -- the model's job, not a
# scalar sweep -- and the same reference forbids mixing the two in one dict.
PASS_DOMAINS = {
    "cube_nbuffer_setting": MERGE_GRANULARITY,
    "cube_l1_reuse_setting": MERGE_GRANULARITY,
}


def merge_domain(knob, subgraph_count=None):
    """Granularities worth trying for a merge pass.

    `subgraph_count` is the `subGraphCount` the profiler reports for that pass's
    hashOrder, and it is a hard bound: N subgraphs cannot be merged in groups
    larger than N. The two passes do not even have the same ceiling on the same
    kernel: a trace routinely reports very different `subGraphCount` values for
    `cubeMergeInfo` and for `l1ReuseInfo`. Same shape as
    the tile domains, which the buffer sizes bound: how many there are decides
    how many can be grouped.

    None (no trace read) leaves the documented ladder intact.
    """
    dom = PASS_DOMAINS.get(knob, MERGE_GRANULARITY)
    if subgraph_count:
        dom = [n for n in dom if n <= subgraph_count] or [1]
    return dom


def pass_knob(site_id):
    """`pass:cube_nbuffer_setting#0` -> `cube_nbuffer_setting`."""
    return site_id.split(":", 1)[1].split("#")[0]


def global_knob(site_id):
    """'g:<knob>#<i>' -> '<knob>'. Global site ids encode which runtime_options key they tune."""
    return site_id[2:].rsplit("#", 1)[0]


@dataclass
class HW:
    """Chip / kernel envelope the static check validates against.

    m_l1/k_l1/n_l1 are the L1 tile dims the kernel is compiled with (each L0 tile dim must divide its
    L1 dim). l1_budget_kb is the usable L1 for the two matmul operands. These are PER-KERNEL, read
    from the op's DESIGN and passed in at dispatch (BOINIT --m_l1/--k_l1/...). If you don't have them,
    leave calibrated=False: the check keeps only the universal alignment rule and defers the rest to
    the correctness gate, so BO still works on ANY op — just less eval-efficiently.
    nL1=None skips the n_l0/n_l1 relation (kernels that don't L1-tile the N dim).
    """

    m_l1: int = 64
    k_l1: int = 64
    # The L1 buffer, holding both matmul operands' L1 tiles at once. 512 KB fixed
    # hardware (knowledge base: "L1=512KB"); this was 192 under an old conservative
    # estimate that has no basis now that the real buffer is checked.
    l1_budget_kb: int = 512
    dtype_bytes: int = 4  # L1 staging is fp16/bf16 but estimate conservatively in fp32
    head_dim: int = 128  # bounds the vector tail tile
    n_l1: Optional[int] = None  # L1 N-dim tile; None => no n_l0/n_l1 constraint
    calibrated: bool = (
        True  # False => skip op-specific L1/subdivision checks (keep alignment
    )
    #          only) so an un-calibrated op still explores via the gate
    # The unified-buffer capacity rule. Deliberately NOT behind `calibrated`:
    # `calibrated` gates the checks that need an op's DESIGN (m_l1/k_l1/n_l1), and
    # these two need only the program and the chip. Leaving the footprint bound
    # switched off is what let a 16384-element tile reach the device against a
    # buffer that holds about 12288.
    ub_budget_kb: int = 192
    n_live_tiles: int = 0  # 0 => unknown, rule off. Supplied by the optimizer
    #      as a FACT about residency, never as a bound.
    # The three L0 matmul-operand buffers. Unlike m_l1/k_l1/n_l1 these are FIXED
    # HARDWARE, not per-kernel, so they are checked regardless of `calibrated`:
    # the L0 tile the kernel sets must fit its buffer no matter what the L1
    # envelope is. Sizes from the knowledge base (pass.md: "UB=192KB, L0A=64KB,
    # L0B=64KB, L0C=128KB, L1=512KB. L0A holds mL0 by kL0, L0B holds kL0 by nL0,
    # and L0C
    # holds mL0xnL0 (always fp32 accumulator). Each dim is ceil-aligned to 16.
    l0a_kb: int = 64
    l0b_kb: int = 64
    l0c_kb: int = 128
    # Core counts, on the same object as the buffers so every module reads one
    # envelope. `seed` asks "does every core get work", which is a question about
    # the chip and not about the kernel. 20/40 are 910B3, confirmed against a
    # trace (AIC_0..19, AIV_20..59); an A5 has different ones, which is why the
    # harness supplies these rather than each module keeping its own copy.
    cube_cores: int = 20
    vector_cores: int = 40


# ------------------------------------------------------------------------------- grouping helpers
def _rep(sites, groups):
    """Map each site id -> its representative id. groups=None => every site is its own rep
    (per-site independent, the default). groups is a list of id-lists sharing one config.
    """
    rep = {s.id: s.id for s in sites}
    for grp in groups or []:
        head = grp[0]
        for sid in grp:
            rep[sid] = head
    return rep


def _tie_rep(sites, tie):
    """site id -> the id whose LEADING field it shares. Unlisted sites map to themselves.

    Distinct from `_rep`/`groups`, which makes a whole group ONE knob. Here only
    the first tile dimension is shared and every other dimension stays a knob of
    its own site -- see `apply.tie_groups` for why the evidence supports the
    narrower relation and not the wider one.
    """
    rep = {s.id: s.id for s in sites}
    for grp in tie or []:
        for sid in grp:
            if sid in rep:
                rep[sid] = grp[0]
    return rep


def _lead_key(rep, sid, field, fields):
    """Optuna param name for `field` of `sid`: the tie representative's, for the
    leading field of a tied site; the site's own otherwise.
    """
    if fields and field == fields[0]:
        return f"{rep.get(sid, sid)}.{field}"
    return f"{sid}.{field}"


def kind_of(sites, sid):
    for s in sites:
        if s.id == sid:
            return s.kind
    return None


# ------------------------------------------------------------------------------- per-site space
def admit(base, val):
    """Base domain, guaranteeing the anchor (current) value is a legal choice. Optuna categoricals
    must contain the enqueued warm-start value, and real kernels use values outside our static sets
    (device_sched_mode=3, a 40-row tile, a 10240-wide tile, ...) — so always include the current
    one. The ladder is a sampling grid, never a constraint on what the kernel may already hold.
    """
    return base if (val is None or val in base) else sorted(set(base) | {val})


CONST_SCALES = [1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8]


def const_domain(anchor):
    """Ladder for a `const` site, anchored on the value the kernel is running.

    There is no static set to draw from: a loop-trip constant is a token in one
    kernel, and what is a sane extent for one operator is absurd for the next.
    Scaling the incumbent is the only anchor available that does not need a
    hardware model, and it matches how the L1 multiplier is already handled.

    The domain does not have to be correct, only to CONTAIN the good values.
    Anything illegal announces itself on the device for one evaluation and the
    learned ceiling and carried refusals then generalise it.
    """
    if not isinstance(anchor, int) or anchor < 1:
        return [anchor] if anchor is not None else [1]
    return sorted({max(1, int(anchor * s)) for s in CONST_SCALES})


def vec_fields(site_config):
    """The dimension field names of one vec site, in argument order.

    A vec site is `set_vec_tile_shapes(d0, ..., dk)` with one to four integer
    arguments, so the number of knobs is a property of the kernel rather than of
    this module. The anchor (the values the kernel currently holds) is what
    declares the arity; everything downstream reads it from here. Falling back to
    two dimensions keeps the common case working when no anchor is supplied.
    """
    fs = sorted((k for k in site_config if k.startswith("d")), key=lambda k: int(k[1:]))
    return fs or ["d0", "d1"]


def vec_domain(site_id, field, fields, anchor_cfg, domain=None):
    """The values one vec dimension may take: the derived domain, else the ladder.

    `domain` comes from `domain.derive` -- divisors of the view's extent, so a
    leading dimension of 24 against a 16-row view is never drawn rather than
    drawn, written, shipped to the device and rejected there. A field the
    derivation could not constrain is absent from `domain` and keeps its ladder,
    because the safe direction to fail on a domain is WIDE: too wide is caught
    per candidate, too narrow is invisible.
    """
    base = VEC_TRAILING_CHOICES if field == fields[-1] else VEC_CHOICES
    derived = (domain or {}).get("per_site", {}).get(site_id, {}).get(field)
    ladder = list(derived if derived else base)
    # Drop rungs that overflow the UB even with every other dimension at its
    # smallest -- no feasible tile can use them, so sampling them only burns TPE's
    # startup draws (the ladder reaches 16384 against a cap that can be near 1024).
    # This removes no reachable config; `dom` still re-admits the anchor, so the
    # running kernel's value is a legal choice even if it sits above the cap.
    # Aliased: the bare import rebound the PARAMETER `domain` to the module, so
    # the derived domain was lost and the module itself was passed in its place.
    from . import domain as domain_mod

    cap = domain_mod.vec_ladder_cap(domain, site_id, field, fields)
    if cap:
        pruned = [v for v in ladder if v <= cap]
        if pruned:  # never empty the ladder
            ladder = pruned
    return admit(ladder, (anchor_cfg or {}).get(field))


def tied_lead_domain(tie, sites, anchor=None, domain=None):
    """{tie_head: [choices]} -- ONE domain for the leading field a tie group shares.

    A tie group shares one Optuna parameter name (`_lead_key`), and Optuna
    requires one name to carry one distribution for the whole study. But
    `vec_domain` derives its ladder cap from the SITE's unified-buffer residency,
    and members of a tie group need not have the same residency: when two tied
    scopes hold a different number of live tiles, `vec_domain` hands the very
    same parameter two different caps. The first trial then died with

        ValueError: CategoricalDistribution does not support dynamic value space

    and took the entire block with it, before a single measurement -- the tie was
    introduced to save a re-block per iteration and instead disabled the lever on
    any kernel whose tied scopes differ in residency.

    The shared ladder is the INTERSECTION of the members' ladders: the only set
    every member can legally hold. Each member's anchor is then unioned back in,
    for the same reason `dom` does it -- the value the kernel is running must
    stay drawable even when it sits above another member's cap.
    """
    kind = {s.id: s.kind for s in sites}
    out = {}
    for grp in tie or []:
        shared, anchors = None, []
        for sid in grp:
            if kind.get(sid) != "vec":
                continue
            a = (anchor or {}).get(sid) or {}
            fields = vec_fields(a)
            if not fields:
                continue
            d = set(vec_domain(sid, fields[0], fields, a, domain))
            shared = d if shared is None else (shared & d)
            if a.get(fields[0]) is not None:
                anchors.append(a[fields[0]])
        if shared is None:
            continue
        out[grp[0]] = sorted(shared | set(anchors))
    return out


@dataclass
class DrawOptions:
    """What shapes one draw beyond the trial and the sites: which sites move
    together, what the kernel currently holds, and what the domain allows.
    """

    groups: Any = None  # site ids sharing one knob
    anchor: Any = None  # the running config, always re-admitted
    domain: Any = None
    tie: Any = None  # vec scopes that must keep one extent


def _draw_cube(trial, r, a):
    """Six knobs: L0 per axis, and L1 as L0 x multiplier.

    L1 is a real tunable -- the hand-optimised kernel wins by raising it from 128
    to 256, an axis the old three-knob reader never touched. Drawing the
    multiplier (not L1 itself) keeps `0 < L0 <= L1 and L1 % L0 == 0` true by
    construction, so the cube feasible fraction goes from ~7% to ~100% instead of
    the search burning its draws on illegal pairs. Config stores the real
    L1 = L0 x mult.
    """
    cube = {}
    for ax in ("m", "k", "n"):
        l0 = trial.suggest_categorical(
            f"{r}.{ax}L0", admit(CUBE_CHOICES, a.get(f"{ax}L0"))
        )
        x = trial.suggest_categorical(
            f"{r}.{ax}L1x", mult_dom(a.get(f"{ax}L0"), a.get(f"{ax}L1"))
        )
        cube[f"{ax}L0"], cube[f"{ax}L1"] = l0, l0 * x
    return cube


def _draw_scalar(trial, r, a, dom):
    """One `value` knob drawn from `dom`, with the anchor always admitted."""
    return {
        "value": trial.suggest_categorical(f"{r}.value", admit(dom, a.get("value")))
    }


def _draw_vec(trial, r, a, ctx):
    """Every dimension of the call is a knob, however many there are.

    The old space moved the leading one only, on the reading that
    `basic-block-optimization.md` rules 4 and 5 pin the trailing axis. The
    recorded runs say otherwise: with the leading value at 1 -- the single most
    common case, 67 of 313 writes -- all the tuning happens in the trailing
    dimension, from 64 to 16384. A leading-only sweep is a no-op on exactly those
    kernels.

    The leading field may be shared with the other vec scopes in the same loop
    body -- see `apply.tie_groups`. Sharing the PARAM NAME is what makes it one
    knob to Optuna: two sites drawing `vec#0.d0` get the same value by
    construction, and the sampler sees one dimension instead of two that must
    agree.
    """
    lead, lead_dom = ctx["lead"], ctx["lead_dom"]
    fields = vec_fields(a)
    head = lead.get(r, r)
    return {
        f: trial.suggest_categorical(
            _lead_key(lead, r, f, fields),
            (
                lead_dom[head]
                if (f == fields[0] and head in lead_dom)
                else vec_domain(r, f, fields, a, ctx["opts"].domain)
            ),
        )
        for f in fields
    }


def _draw_site(trial, site, r, ctx):
    """One draw for one representative site, dispatched on its kind."""
    opts = ctx["opts"]
    a = (opts.anchor or {}).get(r, {})
    if site.kind == "cube":
        return _draw_cube(trial, r, a)
    if site.kind == "global":
        return _draw_scalar(trial, r, a, GLOBAL_DOMAINS[global_knob(r)])
    if site.kind == "pass":
        counts = (opts.domain or {}).get("subgraph_counts", {})
        return _draw_scalar(
            trial, r, a, merge_domain(pass_knob(r), counts.get(pass_knob(r)))
        )
    if site.kind == "const":
        return _draw_scalar(trial, r, a, const_domain(a.get("value")))
    if site.kind == "unroll":
        return _draw_scalar(trial, r, a, unroll_domain(getattr(site, "trip", None)))
    return _draw_vec(trial, r, a, ctx)


def _apply_tie(cfg, tie):
    """Write the shared leading value back over every member of a tied group.

    A tied site is drawn independently (it is its own `rep`), so without this the
    group agrees in Optuna's params and disagrees in the config that reaches the
    kernel.
    """
    for grp in tie or []:
        head = next((cfg.get(g) for g in grp if g in cfg), None)
        if not head:
            continue
        f0 = next(iter(vec_fields(head)), None)
        if f0 is None:
            continue
        for sid in grp:
            if sid in cfg and f0 in cfg[sid]:
                cfg[sid][f0] = head[f0]


def suggest(trial, sites, opts=None):
    """Draw a PER-SITE config from an Optuna trial: {site_id: tile_dict} for every tunable site.

    Default (groups=None) samples each site independently. Grouped sites share one draw (their
    representative's params), so a group is one knob. `anchor` (typically current_config) keeps each
    site's current value in its domain, so warm-start never fails and BO can pick the current value.

    `tie` is the narrower relation: id-lists whose LEADING tile dimension is one
    knob while every other dimension stays per-site. Used for vec scopes that
    share a loop body, where a mismatched row extent costs a re-block on every
    iteration.
    """
    opts = opts or DrawOptions()
    rep = _rep(sites, opts.groups)
    ctx = {
        "opts": opts,
        "lead": _tie_rep(sites, opts.tie),
        # One shared parameter name demands one shared distribution; see
        # `tied_lead_domain`.
        "lead_dom": tied_lead_domain(opts.tie, sites, opts.anchor, opts.domain),
    }
    drawn = {}
    cfg = {}
    for s in sites:
        r = rep[s.id]
        if r not in drawn:
            drawn[r] = _draw_site(trial, s, r, ctx)
        cfg[s.id] = dict(drawn[r])
    _apply_tie(cfg, opts.tie)
    return cfg


def _cube_to_params(r, c):
    """A cube config {m_l0,m_l1,...} -> Optuna params {r.m_l0, r.mL1x, ...}, encoding
    L1 as the multiplier mL1 / mL0 to match how `suggest` draws it.
    """
    out = {}
    for ax in ("m", "k", "n"):
        l0, l1 = c.get(f"{ax}L0"), c.get(f"{ax}L1")
        out[f"{r}.{ax}L0"] = l0
        if l0 and l1 and l1 % l0 == 0:
            out[f"{r}.{ax}L1x"] = l1 // l0
        elif l1 is not None:
            out[f"{r}.{ax}L1x"] = max(1, round((l1 or l0) / l0)) if l0 else 1
    return out


def _params_to_cube(pre, params):
    """The inverse of the above: flat params keyed by representative site back to
    per-axis cube values, with each L1 dim recovered as its L0 dim times the
    stored multiplier.
    """
    c = {}
    for ax in ("m", "k", "n"):
        l0 = params.get(f"{pre}{ax}L0")
        x = params.get(f"{pre}{ax}L1x")
        if l0 is not None:
            c[f"{ax}L0"] = l0
            c[f"{ax}L1"] = l0 * (x if x is not None else 1)
    return c


def flatten(site_configs, sites, groups=None, tie=None):
    """Per-site config -> flat Optuna params (for enqueue_trial / warm-start), keyed by representative.

    Must produce exactly the names `suggest` draws, or an enqueued warm start
    describes a point the study cannot match.
    """
    rep = _rep(sites, groups)
    lead = _tie_rep(sites, tie)
    params = {}
    for s in sites:
        r = rep[s.id]
        cfg = site_configs[s.id]
        if s.kind == "cube":
            # L1 is encoded as a multiplier so the params match the fixed domain
            # `suggest` draws from; the config keeps the actual L1 value.
            params.update(_cube_to_params(r, cfg))
            continue
        # Field-agnostic for the rest: the config carries whatever knobs the site
        # has, which for a vec site is however many arguments the call was written
        # with.
        fields = vec_fields(cfg) if s.kind == "vec" else []
        for field, v in cfg.items():
            params[_lead_key(lead, r, field, fields)] = v
    return params


def unflatten(params, sites, groups=None, tie=None):
    """Flat Optuna params -> per-site config dict."""
    rep = _rep(sites, groups)
    lead = _tie_rep(sites, tie)
    cfg = {}
    for s in sites:
        pre = rep[s.id] + "."
        if s.kind == "cube":
            cfg[s.id] = _params_to_cube(pre, params)
            continue
        own = {}
        for k, v in params.items():
            if k.startswith(pre):
                own[k[slice(len(pre), None)]] = v
        # The leading field of a tied site lives under the representative's name,
        # so the prefix scan above does not see it.
        # `vec_fields` returns the dimensions in argument order, so the leading
        # one is always `d0`; it is named here rather than inferred because at
        # this point there is no config to read the arity from.
        head = lead.get(s.id, s.id)
        if head != s.id and f"{head}.d0" in params:
            own["d0"] = params[f"{head}.d0"]
        cfg[s.id] = own
    return cfg


# ------------------------------------------------------------------------------- feasibility
def ceil16(v):
    return ((int(v) + 15) // 16) * 16


def _cube_feasible(c, hw):
    # DESIGN.md checklist: L0<=L1 AND L1 % L0 == 0 per dim; kL0 16-aligned; operand footprint.
    #
    # The same four rules are pypto's own, in
    # `tools/scripts/tuner/tuner.py: HeuristicTile.is_good_tiling`. They are not
    # imported because `tools/` is not part of the installed package -- the wheel
    # ships `pypto/lib/scripts/` and no tuner -- so the agreement is pinned by a
    # test instead: `tests/test_pypto_reuse.py` restates them and asserts that
    # nothing the tuner rejects passes here. This gate is deliberately the
    # stricter of the two, and `HeuristicTile` takes its capacities from
    # hardcoded defaults where this one is handed a derived envelope.
    if c["kL0"] % 16 != 0:
        return False, f"kL0={c['kL0']} not 16-aligned"

    # The L0 operand buffers are fixed hardware, so they bound the tile whether or
    # not the L1 envelope is calibrated -- and this is the class that was reaching
    # the device: `Alloc tensor size [73728] exceeds MEM_L0B size [65536]` is a
    # kL0xnL0 tile overflowing L0B, paid for as a full evaluation because nothing
    # checked it statically. mL0 is not 16-constrained for alignment, but its
    # buffer still ceil-aligns it. L0C is the fp32 accumulator regardless of input
    # dtype.
    m_l0, k_l0, n_l0 = c["mL0"], c["kL0"], c["nL0"]
    db = hw.dtype_bytes
    for elems, cap_kb, name in (
        (ceil16(m_l0) * ceil16(k_l0) * db, hw.l0a_kb, "L0A"),
        (ceil16(k_l0) * ceil16(n_l0) * db, hw.l0b_kb, "L0B"),
        (ceil16(m_l0) * ceil16(n_l0) * 4, hw.l0c_kb, "L0C"),
    ):
        if cap_kb and elems > cap_kb * 1024:
            return False, (
                f"{name} tile {elems}B > {cap_kb}KB "
                f"(mL0={m_l0}, kL0={k_l0}, nL0={n_l0} at "
                f"{db if name != 'L0C' else 4}B)"
            )

    # The L1 tile is now IN the config (m_l1/k_l1/n_l1), a real tunable rather than a
    # DESIGN number the harness never had. The compiler's per-axis invariant is
    # `0 < L0 <= L1 and L1 % L0 == 0`; `suggest` draws L1 as a multiple of L0 so
    # this holds by construction, but a warm start or an off-ladder anchor can
    # still violate it, so it is checked (this is the `Invalid L1/L0 relation`
    # error, now caught statically instead of on the device).
    for ax in ("m", "k", "n"):
        l0, l1 = c.get(f"{ax}L0"), c.get(f"{ax}L1")
        if l0 is None or l1 is None:
            continue
        if l0 > l1 or l1 % l0 != 0:
            return False, (
                f"Invalid L1/L0 relation: {ax}L0={l0}, {ax}L1={l1}, "
                f"require {ax}L0 <= {ax}L1 and {ax}L1 % {ax}L0 == 0"
            )

    # L1 holds BOTH matmul operands' L1 tiles at once: A = m_l1*k_l1, B = k_l1*n_l1.
    # 512 KB per the knowledge base. Uses the L1 dims, not the L0 ones.
    m_l1, k_l1, n_l1 = c.get("mL1"), c.get("kL1"), c.get("nL1")
    if None not in (m_l1, k_l1, n_l1) and hw.l1_budget_kb:
        l1_bytes = (ceil16(m_l1) * ceil16(k_l1) + ceil16(k_l1) * ceil16(n_l1)) * db
        if l1_bytes > hw.l1_budget_kb * 1024:
            return False, (
                f"L1 tile {l1_bytes}B > {hw.l1_budget_kb}KB "
                f"(A={m_l1}x{k_l1} + B={k_l1}x{n_l1} at {db}B)"
            )
    return True, ""


def _vec_feasible(v, hw):
    """Alignment on the contiguous (trailing) dimension, whatever the arity.

    The leading dimensions are row counts and carry no alignment rule; only the
    innermost one is the vectorised axis. With one argument that axis is the only
    argument.
    """
    fields = vec_fields(v)
    trail = v.get(fields[-1])
    if trail is None:
        return True, ""
    # A trailing extent of 1 is a degenerate reduction-output leg, not a
    # vectorised axis: there is nothing to align when the axis is one element
    # wide. It is excluded from VEC_TRAILING_CHOICES, so the sampler never draws
    # it; it reaches this gate only through `dom` re-admitting the value the
    # kernel is ALREADY running. Rejecting it there threw away the warm start
    # on a kernel whose reduction-output legs run and pass golden with a
    # trailing extent of 1 -- and with no feasible observation TPE had nothing
    # to model, so the block returned best_config: null after 16 draws and 0
    # device trials. A rule that rejects a running kernel is not describing the
    # hardware.
    if trail != 1 and trail % 16 != 0:
        return False, f"trailing vec dim {trail} not 16-aligned"
    # The UB footprint bound is NOT here. It is per-vec-site and needs the
    # residency `domain.vec_residency` reads from the source, so it lives in
    # `domain.verify`, which runs on every candidate right after this. The old
    # `hw.n_live_tiles` bound was a single number applied per site with the wrong
    # arithmetic (it pre-divided the buffer by a hand-supplied residency and then
    # compared each site alone); it is gone rather than left to disagree.
    if not hw.calibrated:
        return True, ""
    if hw.head_dim and trail > 4 * hw.head_dim:
        return False, (
            f"trailing vec dim {trail} >> head_dim={hw.head_dim} (unusable tail tile)"
        )
    return True, ""


def static_feasible(site_configs, hw):
    """(bool, reason) over ALL sites. False => reject WITHOUT evaluating; reason names the site.

    hw is a single HW (shared L1 envelope) or a dict {site_id: HW} for per-site envelopes.
    A config is cube-typed if it has 'mL0', else vec-typed.
    """
    for sid, c in site_configs.items():
        h = hw[sid] if isinstance(hw, dict) else hw
        if "value" in c and not any(k.startswith("d") for k in c):
            ok, reason = (
                True,
                "",
            )  # global scalar: validated by the correctness gate, not statically
        elif "mL0" in c:
            ok, reason = _cube_feasible(c, h)
        else:
            ok, reason = _vec_feasible(c, h)
        if not ok:
            return False, f"{sid}: {reason}"
    return True, ""


# No hardcoded per-op envelopes live here — BO is operator-agnostic. Callers pass the envelope from
# the op's DESIGN (BOINIT --m_l1/--k_l1/...) or omit it (HW(calibrated=False), the default). Recording
# a specific op's numbers in this general module is exactly the coupling we avoid.
