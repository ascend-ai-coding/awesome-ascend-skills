#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""panko_harness.py — deterministic control core for Stage 7 PANKO (Arch A).

The world model is the persisted tree in `custom/<op>/optimization/search_state.json`.
This harness owns only the DETERMINISTIC parts of the loop — selection (argmax pypto_action_priority),
stagnation counting, j, tree-edit application, stop conditions, and state I/O. It runs
no LLM and needs no API key. The optimizer (the only reader of pypto-op-perf-panko-manual)
calls these subcommands each cycle; the reasoning (propose δ, rescore V, decide
Insert/Update/Prune, run the eval) stays with the optimizer.

All subcommands read/write <op_dir>/optimization/search_state.json and print a JSON
result to stdout. See references/state_schema.md for the schema.

  J(x) = s * (P_ref / p) * 100        # success stop when best_J >= 100
  V_pypto := pypto_action_priority = w_stage(stage) * prior_gain * novelty(tried) * V
    # pypto_action_priority (short symbol V_pypto) is OUR composite selection score — NOT from
    # the PANKO paper. V is PANKO's per-node value/belief and is only one of its factors.
"""

import argparse
import ast
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import evaluation_session

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jsonio  # noqa: E402

# The package needs `optuna`, which is a real dependency of the tile search and
# of nothing else here. Absent, the search still runs with
# `bayesian_optimization_tile_search` off -- that is the ablation arm, and the
# only supported way to run without it. `cmd_init` refuses the combination of
# the flag ON and the package missing, so the two can never disagree silently.
#
# ONLY ModuleNotFoundError, and only for optuna. `except Exception` swallowed a
# circular import, a syntax error in this package, and an optuna API change
# alike, set every module to None, and let the run continue reporting
# `tile_search: true` while the numbers were being chosen by the model instead.
# Those are bugs and they must reach a traceback.
OPTUNA_MIN_VERSION = (2, 0)  # `create_trial` / `add_trial`; developed against 4.9


def _read_text(path, errors="replace"):
    """The file's text, with the handle closed before the caller sees it."""
    with open(path, encoding="utf-8", errors=errors) as f:
        return f.read()


def _read_bytes(path):
    """The file's bytes, with the handle closed before the caller sees them."""
    with open(path, "rb") as f:
        return f.read()


def is_missing_optuna(exc):
    """Is this import failure the optional dependency, or a bug in this package?

    Only a ModuleNotFoundError naming optuna is the former. A circular import, a
    syntax error in this package and an optuna API change all raise ImportError
    or a ModuleNotFoundError naming something else, and each of those is a
    defect that has to reach a traceback rather than silently turn the tile
    search off.
    """
    return (
        isinstance(exc, ModuleNotFoundError)
        and (getattr(exc, "name", "") or "").split(".")[0] == "optuna"
    )


try:
    import bayesian_optimization as bayesian

    _BAYESIAN_UNAVAILABLE = None
except ModuleNotFoundError as _e:  # pragma: no cover
    if not is_missing_optuna(_e):
        raise  # this package is broken, not a missing optional dep
    # One name for the whole package, so the guards downstream are `bayesian is
    # None` rather than eleven separate module flags. The previous form bound
    # `= None, shutil` -- a TUPLE -- to three of them, so every `is None` guard
    # read False on the baseline arms.
    bayesian = None
    _BAYESIAN_UNAVAILABLE = f"optuna is not importable ({_e})"


def bayesian_capability():
    """Whether the tile search can actually run, and on what.

    Reported at INIT and persisted, so a run's report names the algorithm that
    ran rather than the one that was configured.
    """
    if bayesian is None:
        return {
            "available": False,
            "optuna_version": None,
            "reason": _BAYESIAN_UNAVAILABLE,
        }
    try:
        import optuna  # noqa: PLC0415

        version = getattr(optuna, "__version__", "")
        parts = tuple(int(x) for x in version.split(".")[:2] if x.isdigit())
    except Exception as e:  # pragma: no cover
        return {
            "available": False,
            "optuna_version": None,
            "reason": f"optuna imported but unusable: {e}",
        }
    if parts and parts < OPTUNA_MIN_VERSION:
        return {
            "available": False,
            "optuna_version": version,
            "reason": (
                f"optuna {version} < "
                f"{'.'.join(str(x) for x in OPTUNA_MIN_VERSION)}; "
                "the tile search needs trial.create_trial / study.add_trial"
            ),
        }
    return {"available": True, "optuna_version": version, "reason": None}


try:  # optional: absent -> the gate is a no-op
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import chip_profile
    import feasibility
    import predicates
except Exception:  # pragma: no cover
    # BOTH names, or the "gate is a no-op" fallback is a NameError instead: the
    # first reference is `cmd_init`'s `if (predicates and ...)`, so the run would
    # fail to start on exactly the configuration this handler exists for.
    chip_profile = feasibility = predicates = None

STAGE_ORDER = ["frontend", "swimlane", "incore"]
# How many DISTINCT structures a refusal must be seen on before it is carried to
# a structure it has not been seen on. 1 restores the old behaviour (carry from
# the first sighting); a value above 2 buys more evidence at the price of paying
# for the refusal on that many structures first.
REFUSAL_GLOBAL_AFTER = 2


# The chip envelope every static rule is validated against: buffer capacities in
# KB and the core counts the seed generator uses to ask "does every core get
# work". ONE dict, because there were two.
#
# `feasibility.HW` and `bayesian_optimization.space.HW` each carried their own
# defaults and they disagreed: L1 was 512 in one and 192 in the other, and
# space.py's own comment says the 192 "was an old conservative estimate that has
# no basis now that the real buffer is checked". Only the feasibility side was
# reachable from the CLI, so `--l1-kb` moved one envelope while the tile search
# kept using the other. On A3, today, a legal cube tile can be rejected by the
# stale number.
#
# TRANSCRIBED, and no longer the default: `init` DERIVES the envelope from the
# platform ini (see chip_profile.py) and refuses rather than falling back here.
# These remain for one case only -- a state file written before the field
# existed, where nothing better is knowable -- and `source` says so wherever the
# envelope is reported.
#
# They were transcribed faithfully from one SKU, which is not the point: they are
# right for that SKU only, and already wrong on another one in the same family,
# which carries these same five buffers and a different core count. Nor would a
# per-generation constant help -- two SKUs of one generation can share every
# buffer size and differ several fold in core count.
CHIP_ENVELOPE = {
    "ub_kb": 192,
    "l1_kb": 512,
    "l0a_kb": 64,
    "l0b_kb": 64,
    "l0c_kb": 128,
    "cube_cores": 20,
    "vector_cores": 40,
    "source": "transcribed:910B3",  # becomes the ini path once derived
}


def chip_envelope(st=None):
    """The envelope for this run. From the state when it has one, so a resumed
    campaign keeps validating against what it started with.
    """
    env = dict(CHIP_ENVELOPE)
    if st:
        env.update(
            {k: v for k, v in (st.get("chip_envelope") or {}).items() if k in env}
        )
    return env


DEFAULT_CONFIG = {
    # Refinement allowance per action: how many candidates may fail to beat the
    # running global best before the action is closed. This is a hyperparameter
    # of the search loop, not a claim either method makes, so both arms must use
    # the same value or the comparison measures the setting. 7 is the PANKO
    # baseline's own default; adopting it rather than our own removes any charge
    # that it was tuned in our favour.
    #
    # It was 3, which favours breadth. On the level-1 operators, where the
    # payoff is concentrated in one direction that needs many attempts to tune,
    # 3 closed the productive action after three near misses: several separate
    # actions can each produce a candidate within 10% of the best and be retired
    # anyway.
    "stagnation_K": 7,
    # Static rejections are counted separately from stagnation and bounded
    # separately. `stagnation_K` counts MEASUREMENTS that failed to improve;
    # a rejection is not a measurement. This bound exists only so that an action
    # whose whole neighbourhood is infeasible terminates: it spends no budget,
    # so eval_budget would never end it. Set well above the five rejections that
    # is the most any single action accumulated across the recorded runs, because
    # a rejection costs one dispatch rather than one device run.
    "infeasible_K": 16,
    "staged_prior": True,  # soft-staging toggle (staged-prior vs free-tree)
    "wall_clock_limit_s": 86400,  # 24 h; the existing Stage 7 stops at 12 h
    "eval_timeout_s": 300,  # 5 min per candidate (enforced by the optimizer's `timeout`)
    # Evaluations per operator; None means unlimited. Raised from 90: the recorded
    # campaigns kept ending on the budget rather than on a search decision, so
    # the number was measuring the cap and not the method. 300 lets the
    # deterministic stops (stagnation, frontier exhaustion, wall clock) be the
    # ones that actually fire.
    "eval_budget": 300,
    # None = DISABLED. The counter still advances and is still reported; it simply
    # no longer halts the search. Rationale: in the 2026-07 campaign this limit
    # halted every PANKO run at 23-67 evaluations of a 120 budget while the
    # PANKO baseline ran on, which makes the budget accounting incomparable
    # across methods. Set an int to restore the old behaviour (Sec. 7.4 ablation).
    "global_stagnation_limit": None,
    # False = advance stages cyclically but never halt on a no-gain cycle. With
    # stage_patience=2 over three stages, `converged` is reachable in six
    # unproductive closes (~18 evaluations), well inside the budget.
    "stage_convergence": False,
    "stage_min_gain": 0.01,  # a close is "productive" only if it cuts best latency by >= 1%
    # Firings of a tile BO lever that are exempt from the novelty penalty. A
    # sweep is not repetition; see novelty(). 0 restores the old behaviour and
    # is the setting for the bayesian_optimization_tile_search=false ablation arm.
    "bayesian_optimization_free_trials": 0,
    # A tile BO block explores at random until this many FEASIBLE configurations
    # are measured, then hands the search to TPE. Counting feasible measurements
    # rather than trials keeps TPE's surrogate off the 1-2 real points a
    # mostly-rejected block used to model on.
    "bayesian_optimization_startup_min_feasible": 5,
    # Distinct structures a compiler refusal must be observed on before it is
    # carried to one it has not been seen on. See `refusals_for_structure`:
    # the signature is stable across occurrences of the same refusal, which is
    # not the same as establishing that two refusals sharing it share a cause.
    # 1 restores the old carry-from-first-sighting behaviour.
    "bayesian_optimization_refusal_global_after": REFUSAL_GLOBAL_AFTER,
    "stage_patience": 2,  # advance to the next stage after this many non-productive closes
    # Tile shapes are the one lever where the number, not the structure, decides
    # the outcome, and the number has always been the model's guess. A generated
    # kernel carries whatever tile DESIGN.md wrote down, and the right value can
    # be orders of magnitude away from it, yet a model-driven search can spend a
    # whole budget without touching it -- because nothing in a profiler trace
    # says "this tile is 256 bytes". False restores that: the model proposes
    # numbers again.
    "bayesian_optimization_tile_search": True,
    "symptom_boost": 1.3,  # >1 enables deterministic symptom->action V re-boost (1.0 = off)
    "static_feasibility": True,  # reject hardware-infeasible candidates before the device (False = off)
    "action_preconditions": True,  # hold back actions whose structural precondition is unmet (False = off)
    "conversion_actions": True,  # include the actions that CREATE preconditions (False = off)
    # Fraction of the run budget a tile BO block may consume, counted across all
    # blocks. None = DISABLED: a block is bounded only by the remaining run
    # budget and by its own stagnation rule. Rationale: the share existed so
    # blocks could not crowd out structural search, but tile numbers are exactly
    # the lever the model cannot propose, and capping them at half the budget
    # retired blocks while they were still improving. Set a float (0.5 was the
    # previous value) to restore the old behaviour.
    "block_budget_share": None,
}

# Deterministic symptom index (a subset of the existing Stage-7 index) over metrics we ALREADY
# measure per eval (core-util %, bubble %). At INIT and at every close we re-point open-node
# beliefs (V) at the CURRENT bottleneck -- no extra eval -- complementing the optimizer's evolve.
# The `bubble` threshold is the one number here with almost no evidence behind
# it. It was chosen while the metric was unobtainable: `bubble_analysis.log` was
# never written (a local `sys.exit(0)` in pypto's draw_swim_lane.py sat in front
# of the writer), the stdout fallback never matches, and every recorded value
# was a 0.0 the harness itself had supplied. So 10.0 was never tested against a
# reading.
#
# The first genuine readings, taken after that patch was moved, sit BELOW the
# threshold -- so on a well-packed kernel the symptom would still not fire. That
# may be correct (such a kernel need not be scheduler-bound) or the threshold
# may simply be too high; a handful of points cannot tell. Collect readings
# across operators before moving it, and do not treat 10.0 as validated.
SYMPTOM_THRESH = {
    "bubble": 10.0,
    "util": 50.0,  # bubble% ABOVE 10 ; core-util% BELOW 50
    "occupancy": 0.5,  # UB fraction BELOW 0.5
    "l1_occupancy": 0.5,  # L1 fraction BELOW 0.5
    # --- per-pipe, from measure_latency._pipes_from_log ---
    # `util` above is a mean over AIC and AIV together, and with 20
    # cube cores against 40 vector ones the vector pipe carries two
    # thirds of it. Measured kernels sit outside every threshold
    # here while a pipe is the bottleneck. One whose vector pipe is
    # saturated and whose cube pipe is idle reads above 50 and
    # raises nothing at all; one where both pipes are idle raises
    # `util`, whose action set is about occupancy and contains no
    # way to overlap two pipes -- S-15 lives under `bubble`, and
    # `bubble` counts only wait_schedule while nearly all of that
    # stall is wait_predecessor.
    # Two of these four come from the master Stage-7 skill, which
    # grades the aggregate utilisation on a scale this project
    # already uses elsewhere: below 50 is a symptom worth acting on
    # (analyze_perf.py, and optimization_catalog's 症状B, which is
    # where `util: 50.0` above came from), 70 and up is "scheduling
    # and utilisation are about right", and 80 is 分满核 -- the
    # cores are full (tileshape-deep-tuning §1.1, and the "利用率 >
    # 80% 且 气泡率 < 10%" that the perf-tune skill repeats three
    # times). Applying its own scale per pipe leaves the measured
    # kernels classified exactly as before -- a saturated vector
    # pipe is full by any of these lines and an idle one is idle by
    # any of them -- so this replaces a guess with a cited number
    # and changes no verdict.
    "pipe_saturated": 80.0,  # ... while aiv_util is AT OR ABOVE this
    "pipe_idle": 50.0,  # BOTH pipes below this ...
    # The other two are ours: master measures neither a per-pipe
    # ratio nor the predecessor wait, so there is nothing to cite
    # and these stay open until campaign data settles them. Note
    # pred_stall reads 75-91% on both kernels measured so far, a
    # very different range from the bubble it mirrors, so borrowing
    # bubble's 10/20 bands for it would fire on nearly everything.
    #
    # STILL UNTESTED, and the record says so precisely: across the
    # three per-pipe readings taken to date NEITHER of these two has
    # decided a verdict. Every ratio measured falls on the same side
    # of 0.5, and every predecessor wait measured falls far above
    # 50, so `pipe_saturated` and `pipe_idle` -- the two that ARE
    # cited -- did all the separating. Moving either number on this
    # evidence would be inventing precision. What would settle them
    # is a kernel whose ratio lands between 0.3 and 0.7, or whose
    # predecessor wait lands between 30 and 70; until one appears,
    # treat a verdict that turns on these as unsupported.
    "cube_starved": 0.5,  # aic_util / aiv_util BELOW this ...
    "pred_stall": 50.0,
}  # ... and predecessor-wait% ABOVE this
SYMPTOM_ACTIONS = {
    # A symptom names an AXIS, not a direction. Where a presence directive and its
    # removal counterpart both bear on the diagnosis, BOTH are boosted and the
    # device decides which way to move: REMOVING submit_before_loop is the
    # large win on one kernel while SETTING it is a heavy loss on another.
    # Boosting only the positive prescribes a direction the evidence does not
    # support.
    # bubble high -> fill idle bubbles: larger tiles / unroll, merge loops, stitch, vec-merge, sched, fuse/pipeline
    "bubble": {
        "F-2",
        "F-3",
        "F-6",
        "F-8",
        "S-4",
        "S-5",
        "S-20",
        "S-9",
        "S-10",
        "S-14",
        "S-15",
    },
    # util low -> raise occupancy: more tasks (granularity/tiling), core-fill, cube merges, memory cuts
    "util": {
        "F-1",
        "F-7",
        "F-9",
        "F-10",
        "S-1",
        "S-2",
        "S-6",
        "S-18",
        "S-7",
        "S-19",
        "S-11",
        "S-12",
        "S-16",
        "S-17",
        "F-17",
        "F-18",
    },
    # UB under-occupied -> make the tile bigger. Only the actions that resize a
    # tile or the loop it implies; nothing here is about scheduling or fusion,
    # because the diagnosis is specifically "the buffer is nearly empty".
    "occupancy": {"F-3", "F-9", "F-10", "F-11", "F-19", "S-11", "S-12"},
    # L1 under-occupied -> the cube operands are far below what fits. Three
    # kinds of response and nothing else: enlarge the cube tile (F-9, S-11),
    # make the residency worth more by reusing or merging (S-6, S-7, S-8), or
    # concede that this shape does not belong on the Cube pipe at all (I-11,
    # "the Cube tile is mostly padding"). S-2 is deliberately ABSENT: it shrinks
    # L0/L1 tiles to raise the task count, which is the opposite prescription.
    "l1_occupancy": {"F-9", "S-6", "S-18", "S-7", "S-19", "S-8", "S-11", "I-11"},
    # Cube idle while Vector is saturated -> the work is on the wrong pipe, and
    # no amount of retiling moves it. Either put some of it on the Cube (F-20
    # turns a reduction into a matmul against a ones vector) or make the vector
    # side cheaper: halve its bytes (F-18), stop materialising what it has to
    # walk (F-17, I-10), or keep the intermediate on-chip instead of round-
    # tripping it (S-14, S-17). I-11 is deliberately ABSENT and is the inverse
    # prescription -- it moves a matmul ONTO the pipe that is already full.
    #
    # F-22 used to be here, and the note said it was the sharpest evidence the
    # set had: a kernel diagnosed cube_starved correctly, fixed by moving a
    # per-tile layout copy off the vector pipe and ONTO THE HOST. That reading
    # does not survive the scope this search now has. The copy did not get
    # cheaper; it left the region `p` measures. Against a frozen wrapper the
    # move is not available, and the gain was a boundary effect rather than a
    # kernel one.
    # The diagnosis stays right; the prescription for it is not PANKO's to give.
    "cube_starved": {"F-20", "F-18", "F-17", "I-10", "S-14", "S-17"},
    # Both pipes idle and each waiting on the other -> a serial recurrence, not
    # an occupancy problem. Overlap it (S-15, I-8), give the Cube something to
    # hold across the wait (S-7, S-8), or cut the chains into more independent
    # work (F-7, S-2). Note S-2 shrinks tiles to raise the task count, which is
    # right here and wrong under `l1_occupancy`; the two never fire together.
    # I-12 sits beside I-8 deliberately: on a pipe-serialized kernel the
    # winning move can be to REMOVE the directive, so naming only I-8 here
    # would boost the harmful direction on exactly the kernels this symptom is
    # written for.
    "pipe_serialized": {"S-15", "I-8", "I-12", "S-7", "S-19", "S-8", "F-7", "S-2"},
}

# The per-pipe readings, carried on `m` beside `util`/`bubble`. Absent means not
# measured: the installed draw_swim_lane.py has a latency-only fast path that
# exits before writing bubble_analysis.log, so these are missing far more often
# than `util` is, and a missing reading must never read as a zero.
PIPE_METRICS = ("aic_util", "aiv_util", "pred_stall", "aic_bubble", "aiv_bubble")


# --------------------------------------------------------------------------- I/O
def _sdir(op_dir):
    return os.path.join(op_dir, "optimization")


def _spath(op_dir):
    return os.path.join(_sdir(op_dir), "search_state.json")


def _rpath(op_dir, op):
    return os.path.join(_sdir(op_dir), f"{op}_optimization.md")


# `bo` was the package name and the key prefix until the abbreviation was spelled
# out. State files outlive a rename -- a search in progress is resumed by the next
# `select` -- so the old spellings are accepted on read and written back in the new
# one by the next `save`. Dropped once no in-flight run can predate the rename.
_RENAMED_STATE_KEYS = {
    "bo_memory": "bayesian_optimization_memory",
    "bo_refusals": "bayesian_optimization_refusals",
    "bo_studies": "bayesian_optimization_studies",
}
_RENAMED_CONFIG_KEYS = {
    "tile_bo": "bayesian_optimization_tile_search",
    "bo_free_trials": "bayesian_optimization_free_trials",
    "bo_startup_min_feasible": "bayesian_optimization_startup_min_feasible",
}


def migrate_keys(st):
    """Accept a state file written before the `bo` -> `bayesian_optimization` rename.

    The new spelling wins if both are present, which can only happen if a file was
    hand-edited; the old key is dropped either way so `save` cannot write it back.
    """
    for holder, table in (
        (st, _RENAMED_STATE_KEYS),
        (st.get("config") or {}, _RENAMED_CONFIG_KEYS),
    ):
        for old_key, new_key in table.items():
            if old_key in holder:
                value = holder.pop(old_key)
                holder.setdefault(new_key, value)
    return st


def load(op_dir):
    with open(_spath(op_dir), encoding="utf-8") as f:
        return migrate_keys(json.load(f))


def save(op_dir, st):
    os.makedirs(_sdir(op_dir), exist_ok=True)
    st["progress"]["wall_clock_s"] = int(time.time() - st["progress"]["_start_ts"])
    # Restamp on every write. A state resumed under a different build is otherwise
    # indistinguishable from one produced entirely by the build recorded at init.
    cv = _code_version()
    if cv and st.get("code_version") and cv != st["code_version"]:
        st.setdefault("code_version_history", [])
        if cv not in st["code_version_history"]:
            st["code_version_history"].append(cv)
    if cv:
        st["code_version"] = cv
    tmp = _spath(op_dir) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)
    os.replace(tmp, _spath(op_dir))


def _fpath(op_dir):
    return os.path.join(_sdir(op_dir), "faults.jsonl")


def append_log(op_dir, op, line):
    with open(_rpath(op_dir, op), "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def append_fault(op_dir, rec):
    """One line per environment fault, in the operator's optimization/ folder.

    Kept out of search_state.json on purpose: this is raw device output, it is
    written while a block is still running, and it must survive a process that
    dies before it can save. Append-only JSONL does all three; the state file
    does none of them.

    It exists because the one signal that can end a whole run was the one signal
    nothing kept. A campaign can stop early reporting `env_faults=3` with the
    markers behind those three already discarded three times over -- the
    evaluator dropped the detail, the driver's history was never persisted, and
    the block's `on_trial` returned before recording anything. Afterwards
    there was no way to tell a held device from a mistyped path, which are
    opposite problems.
    """
    os.makedirs(_sdir(op_dir), exist_ok=True)
    with open(_fpath(op_dir), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# Where this skill actually is, resolved from this file rather than from the
# caller's working directory.
#
# SKILL.md used to spell the harness and the action catalogue as
# `cannbot-skills/ops/pypto-op-perf-panko-manual/...`, which resolves only when the cwd
# happens to be a pypto-gym clone. Quickstart installs the plugin into any
# project, where the skill arrives as `.opencode/skills/pypto-op-perf-panko-manual` (a
# symlink, so the files are reachable) and no `cannbot-skills/ops/` tree exists
# at all. The agent could find the skill and then fail on its first harness call.
#
# `realpath` follows that symlink, so the layout is the source tree's either
# way, and neither the client's directory convention nor the cwd is hardcoded.
SKILL_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def skill_path(*parts):
    """A path inside this skill, wherever it was installed."""
    return os.path.join(SKILL_ROOT, *parts)


DEFAULT_CATALOG = skill_path("references", "action_catalog.json")


def out(obj):
    """One JSON directive on the protocol channel. See jsonio."""
    jsonio.emit(obj)


class Refusal(Exception):
    """A command refusing to proceed, carrying the exit code it ends on.

    Every refusal in this harness has already written its JSON -- the directive
    the optimizer reads -- and what is left is the process's exit code. Raising
    `SystemExit` from the command itself would bury an exit inside a function
    that this module's own tests import and call, where stopping the interpreter
    is not what the caller asked for. The code travels as data instead, and
    `main` is the single place that turns it into an exit.
    """

    def __init__(self, code):
        super().__init__(f"refused with exit code {code}")
        self.code = code


# ------------------------------------------------- code snapshots (harness-owned)
# Greedy accumulation is enforced HERE, not by the LLM: the harness snapshots every kept
# candidate under nodes/<code_hash>.py and deterministically restores the running global best
# into the working <op>_impl.py before each new action, rolling back non-improving candidates.
# This makes "keep the win, build the next idea on it" hold end-to-end and prevents the
# best code from ever being lost (which the agent-driven scheme could not guarantee).
def _ndir(op_dir):
    return os.path.join(_sdir(op_dir), "nodes")


def _npath(op_dir, code_hash):
    return os.path.join(_ndir(op_dir), f"{code_hash}.py")


def _file_hash(path):
    """sha256 of the file, or None. The same digest `record` is handed."""
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except (OSError, TypeError):
        return None


_PYPTO_DEFAULTS = None


def _pypto_defaults():
    """Declared defaults of the pypto callables, or {} when pypto is absent.

    Absent is the normal case off-device (tests, the baseline arms), and it is
    handled by degrading rather than by guessing: with no defaults the semantic
    comparison still catches reformatting and comment edits and folds nothing.
    """
    global _PYPTO_DEFAULTS
    if _PYPTO_DEFAULTS is None:
        if bayesian is None:
            _PYPTO_DEFAULTS = {}
        else:
            try:  # pragma: no cover
                import pypto  # noqa: PLC0415

                _PYPTO_DEFAULTS = bayesian.semantic.defaults_for(
                    getattr(pypto, "frontend", None), pypto
                )
            except Exception:
                _PYPTO_DEFAULTS = {}
    return _PYPTO_DEFAULTS


def _semantic_noop(st, op_dir, op_file, parent=None):
    """(True, why) when the candidate on disk IS the program it is judged against.

    The base is the one `_restore` would roll back to: the action's running best
    if it has produced one, otherwise the global best. That is exactly the
    program the ratchet compares against, so it is the one the candidate has to
    differ from to be a candidate at all.

    Two guards, and the first is what makes this usable at all. EVERY rejection
    path in this harness restores the base into the working file -- `record`'s
    duplicate and non-improving branches, the static gate, `select` -- so at most
    points in a run the file IS the base, byte for byte. Comparing them then
    reports "semantic no-op" about a candidate the coder has not written yet.
    Byte-equality is a different condition with a different meaning (nothing was
    edited), and it belongs to the caller, not here.
    """
    if bayesian is None or not op_file or not os.path.exists(op_file):
        return False, ""
    r = st["progress"].get("refine") or {}
    # The running best belongs to ONE action. Judging action B's candidate
    # against action A's local best compares it to a program it was never built
    # on top of.
    if parent is not None and r.get("action") != parent:
        r = {}
    best = r.get("best") or {}
    base_hash = best.get("code_hash") or _best_hash(st)
    snap = _npath(op_dir, base_hash) if base_hash else None
    if not snap or not os.path.exists(snap):
        return False, ""
    try:
        cand = _read_text(op_file)
        base = _read_text(snap)
    except OSError:
        return False, ""
    if cand == base:
        return False, ""  # nothing was edited; not this gate's business
    return bayesian.semantic.same_program(cand, base, _pypto_defaults())


def _op_file(st, op_dir):
    """The working impl file to snapshot/restore. Prefer the explicit path stored at init;
    fall back to <op_dir>/<op>_impl.py (the standard convention).
    """
    f = st.get("op_file")
    if f:
        return f
    op = st.get("op")
    return os.path.join(op_dir, f"{op}_impl.py") if op else None


def _best_hash(st):
    bn = st.get("best_node")
    return next((c["code_hash"] for c in st["nodes"]["closed"] if c["id"] == bn), None)


def _snapshot(op_dir, op_file, code_hash):
    """Persist current op_file bytes -> nodes/<code_hash>.py. True iff it is on disk.

    Reports failure rather than raising. Callers on the restore contract --
    `init`'s baseline above all -- must REFUSE on a False: every later candidate
    is applied on top of a snapshot, so a missing one means the search is
    accumulating onto whatever the caller happened to leave in the working file.
    """
    if not op_file or not code_hash or not os.path.exists(op_file):
        return False
    try:
        os.makedirs(_ndir(op_dir), exist_ok=True)
        shutil.copyfile(op_file, _npath(op_dir, code_hash))
    except OSError:
        return False
    return True


def _restore(op_dir, op_file, code_hash):
    """Restore nodes/<code_hash>.py -> op_file (deterministic roll-back to a known-good state)."""
    if not op_file or not code_hash:
        return False
    snap = _npath(op_dir, code_hash)
    if not os.path.exists(snap):
        return False
    try:
        shutil.copyfile(snap, op_file)
    except OSError:
        return False
    return True


# ------------------------------------------------------------------- scoring
def compute_j(s, p, p_ref):
    if not s or not p or p <= 0:
        return 0.0
    return round(100.0 * p_ref / p, 4)


def w_stage(stage, active_stage, staged_prior):
    """Annealed stage prior. Not a hard gate: cross-stage moves stay legal, just
    down-weighted. Favors the active stage; upstream (already-explored) < downstream.
    """
    if not staged_prior:
        return 1.0
    d = STAGE_ORDER.index(stage) - STAGE_ORDER.index(active_stage)
    return {0: 1.0, 1: 0.6, 2: 0.3, -1: 0.4, -2: 0.25}.get(d, 0.2)


def clamp_belief(v):
    """Belief into [0, 1], which is where every argument about selection assumes
    it lives.

    `symptom_boost` was clamped here from the start; the model's own `insert`
    and `update` were not, and a recorded run carried V up to 1.4. That matters
    because the staged prior is an ordering claim: an action outside the active
    stage scores at most 0.6 * 0.9 = 0.54 against 0.9 for an untried action of
    maximal prior inside it, and the claim holds only while belief is bounded by
    one. Unbounded, a belief above 1.667 inverts it. The bound is now enforced
    where the value enters rather than assumed everywhere it is read.
    """
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, round(x, 6)))


def novelty(tried, free=0):
    """1/(1+tried) — the frontier's guard against re-proposing the same intent.

    `free` exempts a node's first `free` firings, then resumes the usual decay
    from the excess so there is no cliff at the boundary. It exists for the
    tile BO lever and for nothing else: novelty asks "have you tried this
    before?", which is the right question for a qualitative intent and the
    wrong one for a lever whose every firing is a different point in a
    continuous space.

    Without it the lever is outbid by anything untried from its third firing
    (0.9 * 1/3 * 1.0 = 0.30 loses to a fresh 0.6 * 1.0 * 0.6 = 0.36), which is
    why a component designed around ~40 trials was getting three.
    """
    t = max(0, int(tried))
    return 1.0 / (1.0 + max(0, t - max(0, int(free))))


def _bayes_free_trials(st):
    """0 unless the tile lever is on, so the ablation arm is untouched."""
    cfg = st.get("config", {})
    if not cfg.get("bayesian_optimization_tile_search", False):
        return 0
    return int(cfg.get("bayesian_optimization_free_trials", 0) or 0)


def _is_bayes_lever(node):
    """A frontier node the tile BO lever drives, rather than the model."""
    if bayesian is None:
        return False
    return node.get("catalog_id") in getattr(bayesian.lever, "TILE_ACTIONS", ())


def _block_metrics(st, code_hash):
    """Diagnostics a block recorded for `code_hash`, or {}.

    The winner is promoted by an ordinary `record` call so the ratchet lives
    in one place, but that call is made by the optimizer, which is told not to
    re-measure -- so it has no reading to pass. The block does: it measured
    this program on the device. Recovering it here keeps the promotion
    deterministic and costs no device time.
    """
    if not code_hash:
        return {}
    hit = (st.get("eval_cache") or {}).get(code_hash) or {}
    m = hit.get("m") or {}
    return {k: v for k, v in m.items() if v is not None}


def _metrics(util, bubble, fallback=None, pipes=None):
    """`m` for an eval record: only the readings that exist.

    An absent key means "not measured"; 0.0 means "measured, and it was zero".
    `fallback` supplies values the caller knows but the CLI was not given --
    used for a block winner, whose program the block already measured.
    `pipes` carries the PIPE_METRICS under the same rule.
    """
    m = dict(fallback or {})
    if util is not None:
        m["util"] = util
    if bubble is not None:
        m["bubble"] = bubble
    for k in PIPE_METRICS:
        v = (pipes or {}).get(k)
        if v is not None:
            m[k] = v
    return m


def pypto_action_priority(
    node, active_stage, staged_prior, bayesian_optimization_free_trials=0
):
    """OUR composite selection score (not from the PANKO paper): the paper's per-node
    value V modulated by our staged prior, catalog prior_gain, and novelty.

    `bayesian_optimization_free_trials` is passed through to novelty for tile-lever nodes only, so
    a continuous sweep is not scored as repetition. Every other node keeps the
    original behaviour exactly.
    """
    belief = float(node.get("V", 0.0))
    pg = float(node.get("prior_gain", 0.0))
    free = bayesian_optimization_free_trials if _is_bayes_lever(node) else 0
    return round(
        w_stage(node["stage"], active_stage, staged_prior)
        * pg
        * novelty(node.get("tried", 0), free)
        * belief,
        6,
    )


def _occupancies(op_file):
    """(UB fraction, L1 fraction) for the kernel currently on disk; either may be
    None. Never raises: a symptom that can fail the run is worse than a symptom
    that stays quiet.

    None is not zero. A kernel whose tiles are all driven by symbols cannot be
    read, and a kernel with no matmul has no L1 tile to read -- neither is a
    kernel whose buffer is empty. Reporting 0.0 for those would fire the symptom
    on a fabricated number, which is what happened before `extract` learned to
    drop a partially-readable tile call instead of keeping the half it
    understood.
    """
    if not op_file or not os.path.exists(op_file):
        return None, None
    try:
        return (feasibility.ub_occupancy(op_file), feasibility.l1_occupancy(op_file))
    except Exception:
        return None, None


def _cli_pipes(a, prefix=""):
    """PIPE_METRICS off the parsed CLI namespace, skipping what was not given."""
    return {
        k: getattr(a, prefix + k, None)
        for k in PIPE_METRICS
        if getattr(a, prefix + k, None) is not None
    }


def _pipe_symptoms(m):
    """Which of the two per-pipe symptoms the readings on `m` support.

    Both need BOTH pipes present: a kernel with no `AIC_` rows is not a cube
    kernel starving, it is a vector kernel doing what it is (a pure elementwise
    kernel has no AIC rows at all and a pure matmul none on AIV -- neither is
    defective, and a bare aic/aiv ratio would fire on both).

    The two are mutually exclusive by construction, and that is the point: they
    prescribe opposite things.

        cube_starved     the cube pipe is idle AND the vector pipe is
                         saturated, so the work is on the wrong pipe and
                         overlapping the two buys almost nothing.
        pipe_serialized  neither pipe is saturated and most of the cube's wait
                         is on its predecessor, so the schedule is the cost and
                         overlap is what pays.
        neither          both pipes busy and balanced: no symptom, which is the
                         check that the pair does not fire on a healthy program.

    A ratio alone cannot tell the first two apart: two kernels can have clearly
    different ratios and BOTH sit below the 0.5 line, so the ratio returns the
    same verdict for a kernel on the wrong pipe and a kernel with a serial
    schedule. Whether the busy pipe is actually busy is what separates
    them, and that test is `pipe_saturated` / `pipe_idle` -- the two thresholds
    taken from master, not the two that are ours.
    """
    aic, aiv = (m or {}).get("aic_util"), (m or {}).get("aiv_util")
    if aic is None or aiv is None:
        return []
    try:
        aic, aiv = float(aic), float(aiv)
    except (TypeError, ValueError):
        return []
    if aiv <= 0.0:
        return []
    starved = aic < SYMPTOM_THRESH["cube_starved"] * aiv
    if starved and aiv >= SYMPTOM_THRESH["pipe_saturated"]:
        return ["cube_starved"]
    pred = (m or {}).get("pred_stall")
    if (
        max(aic, aiv) < SYMPTOM_THRESH["pipe_idle"]
        and pred is not None
        and float(pred) > SYMPTOM_THRESH["pred_stall"]
    ):
        return ["pipe_serialized"]
    return []


def _symptom_reboost(st, util, bubble, op_file=None, pipes=None):
    """Deterministic symptom->action bias: re-point OPEN-node beliefs (V) at the current bottleneck
    of the best kernel. Only NUDGES matched actions up (bounded to 1.0); never lowers, so it
    complements -- not overwrites -- evolve. Returns the active symptoms (for logging).

    THREE symptoms, and the third is not a measurement. `occupancy` is read off
    the source by the deterministic core: the fraction of the unified buffer the
    vector tiles are estimated to hold. It is here because the other two cannot
    see it. A tile four times smaller than the buffer affords pays for itself on
    every loop iteration, and a profiler trace of that kernel looks unremarkable
    -- utilisation can be perfectly healthy while the buffer sits nearly empty.
    Nothing in `(s, p, m)` says so, which is why a search that reads only latency
    spends evaluations sampling tile shapes it could have computed.

    This is the symptom the recorded runs needed and did not have. Generated
    kernels routinely carry a tile of a few kilobytes against an affordable
    footprint tens of times larger, and the distance from such a tile to the
    best one is not small. INIT logged `seeded_symptoms=[]` on kernels like
    that, because neither profiler metric crosses its threshold on a kernel
    whose buffer is nearly empty.

    Because it comes from the source and not the device, this symptom is
    available BEFORE the first measurement and survives a run whose profiler
    output is missing.
    """
    boost = float(st["config"].get("symptom_boost", 1.0))
    if boost <= 1.0:
        return []
    try:
        util = float(util or 0.0)
        bubble = float(bubble or 0.0)
    except (TypeError, ValueError):
        util = bubble = 0.0

    active = []
    occ, l1 = _occupancies(op_file)
    if occ is not None and occ < SYMPTOM_THRESH["occupancy"]:
        active.append("occupancy")
    if l1 is not None and l1 < SYMPTOM_THRESH["l1_occupancy"]:
        active.append("l1_occupancy")
    if util > 0.0 or bubble > 0.0:  # a real measurement to read
        if bubble > SYMPTOM_THRESH["bubble"]:
            active.append("bubble")
        if 0.0 < util < SYMPTOM_THRESH["util"]:
            active.append("util")
    active += _pipe_symptoms(pipes)
    if not active:
        return []
    ids = set().union(*(SYMPTOM_ACTIONS[s] for s in active))
    for n in st["nodes"]["open"]:
        if n.get("catalog_id") in ids:
            n["V"] = min(1.0, round(float(n.get("V", 0.0)) * boost, 6))
    return active


def _code_version():
    """Which harness produced this state. A run whose provenance is not
    recorded cannot be compared against another run: the same state file can
    come from a build with or without symptom anchoring, greedy accumulation,
    or the frozen evaluator, and nothing in the state says which.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    rev = ""
    git = shutil.which("git")
    if git:
        try:
            rev = (
                subprocess.check_output(
                    [git, "-C", here, "rev-parse", "--short", "HEAD"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
        except (OSError, subprocess.SubprocessError):
            # Not a work tree, or git cannot read it. The VERSION file below is
            # the documented fallback for a skill tree that was copied rather
            # than cloned, so an empty revision is the handled case.
            rev = ""
    if rev:
        return rev
    # The skill tree is sometimes copied outside a work tree. A VERSION file
    # written at copy time is the documented fallback.
    try:
        with open(os.path.join(here, "VERSION"), encoding="utf-8") as f:
            v = f.read().strip()
            return v or None
    except OSError:
        return None


def _admit(st, op_dir):
    """Recompute the kernel's structure and move newly-qualified actions from
    `held` back into `open`.

    Called at init and after every improvement, because the structure is not
    fixed: an action that introduces a matmul makes the whole cube family
    reachable, and holding rather than deleting is what preserves that path.
    Returns the ids that were admitted, for the log.
    """
    if predicates is None or not st["config"].get("action_preconditions", True):
        return []
    preds = predicates.compute(_op_file(st, op_dir) or "", op_dir)
    st["predicates"] = preds
    pruned = set(st["nodes"].get("pruned", []))
    admitted = []
    keep = []
    for n in st["nodes"].get("held", []):
        if n["id"] in pruned:  # the optimizer retired this while it was reachable
            keep.append(n)
            continue
        if predicates.satisfies(n.get("requires"), preds):
            st["nodes"]["open"].append(n)
            admitted.append(n.get("catalog_id") or n["id"])
        else:
            keep.append(n)
    st["nodes"]["held"] = keep
    return admitted


def _gstag(st):
    """'n/limit', or 'n (limit off)' when global stagnation no longer halts."""
    lim = st["config"].get("global_stagnation_limit")
    n = st["progress"]["global_stagnation"]
    return f"{n}/{lim}" if lim else f"{n} (limit off)"


def _next_id(st, prefix):
    st["_seq"] = st.get("_seq", 0) + 1
    return f"{prefix}{st['_seq']}"


# ----------------------------------------------------------------- subcommands
def _domain_state(st, op_file, live_tiles=None):
    """The derived tile domain for `op_file`, or None when BO is not in play."""
    if bayesian is None or not st["config"].get(
        "bayesian_optimization_tile_search", False
    ):
        return None
    if not op_file or not os.path.exists(op_file):
        return None
    return bayesian.domain.derive(op_file, live_tiles or st.get("live_tiles"))


def _retune_verdict(st, op_file):
    """Has the delta just applied moved the tile space out from under the optimum?

    Change 4, and the whole of it. When a structural action lands -- a flatten, a
    change of view extents, a fusion that changes residency -- the previously
    located tile describes a program that no longer exists, and submitting the
    structure at that stale tile is how a winning pair gets reverted. An
    optimizer wrote this down twice and had nowhere to send the work.

    The obvious test is `bayesian.lever.structural_signature`, and it is the wrong one:
    it hashes the program with tile literals erased, so a renamed variable
    changes it and a block would fire after every non-tile action. What actually
    invalidates the optimum is a change to the inputs of the derivation, so that
    is what is compared. A rename leaves them alone; u50's 1-D flatten does not.

    Deterministic, AST-only, no device. The case it misses -- a rewrite that
    leaves extents and residency alone yet moves the optimum -- is the model's to
    judge, through an explicit request that costs the action a `tried`.
    """
    baseline = st.get("domain_fingerprint")
    now = _domain_state(st, op_file)
    if now is None or baseline is None:
        return {"retune": False}
    moved, why = bayesian.domain.changed(baseline, now)
    if not moved:
        return {"retune": False}
    return {
        "retune": True,
        "retune_reason": why,
        "hint": "the tile this program inherited was chosen for a different "
        "space. Run `block --parent <action_id>` BEFORE record, so the "
        "candidate the ratchet judges is the (structure, tile) pair. "
        "Supply --live-tiles if the rewrite changed how many tiles are "
        "resident.",
    }


def _block_budget(pr, cfg, a):
    """How many device trials this block may spend.

    A run-level share, so blocks cannot crowd out structural search. Not a
    quality rule: within a block the only stopping rule is stagnation. None (the
    default) disables the share: the block is then bounded only by the remaining
    run budget and by stagnation.
    """
    budget = cfg.get("eval_budget")
    remaining = (budget - pr["evals_used"]) if budget else None
    share = cfg.get("block_budget_share", DEFAULT_CONFIG["block_budget_share"])
    if budget and share is not None:
        spent = pr.get("block_evals", 0)
        allowance = int(budget * share) - spent
        remaining = max(0, min(remaining, allowance))
    if a.max_trials:
        remaining = a.max_trials if remaining is None else min(remaining, a.max_trials)
    return remaining


def _block_context(a, st, cfg):
    """Everything the block carries from its setup into its bookkeeping.

    The memory is keyed on the derived tile space, not the program text, so an
    edit that leaves the space alone keeps the study and a genuine reshape starts
    from zero. Without it, repeated blocks on one action each begin with
    `ceiling=None`, re-deriving the same footprint bound the run has already
    paid an evaluation to learn.

    Refusals are scoped to the structure until they have been EARNED on more
    than one. A run can buy the same refusal several times over several program
    shapes -- one oversized `set_vec_tile_shapes` raising PadLocalBuffer.Tensor
    in the block for each view, one device trial each -- and carrying it from
    the first sighting would have saved the rest. It also would have been a
    guess: see `refusals_for_structure`. Two sightings on different structures
    is the evidence, so a run now pays twice and saves the remainder.
    """
    op_file = _op_file(st, a.op_dir)
    with open(op_file, encoding="utf-8", errors="replace") as f:
        base_src = f.read()
    hw, domain = bayesian.block.hw_for(op_file, a.live_tiles, chip_envelope(st))
    structure = _structure(st, a.op_dir)
    refusal_rows = stored_refusals(st)
    return {
        "op_file": op_file,
        "base_src": base_src,
        "hw": hw,
        "domain": domain,
        "remaining": _block_budget(st["progress"], cfg, a),
        "mem_key": bayesian.block.memory_key(a.parent, domain),
        "memories": st.setdefault("bayesian_optimization_memory", {}),
        "structure": structure,
        "refusal_rows": refusal_rows,
        "refusals": refusals_for_structure(refusal_rows, structure),
        "meter": {"device_trials": 0, "winner_runs": 0},
        # Per-trial diagnostics, persisted. A block reporting "24 trials, 1
        # feasible" used to keep nothing about the other 23, so neither the
        # failure class nor the capacity markers could be checked against what
        # the device actually said.
        "trials": [],
        "faults": [],
        "winners": {},
        "winner_hash": None,
        "charged": 0,
        # Wall-clock bounds on the block. The CANN host log is the only other
        # record of a device fault and it is keyed by time; without these there is
        # nothing to correlate it against.
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
    }


def _fault_record(st, a, rec):
    """The env-fault log entry: the marker and the device's own last words."""
    return {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "op": st["op"],
        "action": a.parent,
        "marker": rec.get("reason"),
        "kind": rec.get("kind"),
        "retries": rec.get("retries", 0),
        "config": rec.get("config"),
        "detail": (rec.get("detail") or "")[:2000],
    }


def _block_recorder(a, st, ctx):
    """Build the per-trial callback: count what touched the device, and cache the
    bytes it measured.

    An environment fault is not a trial. The device was unusable, so nothing was
    learned about the candidate and nothing may be charged for it -- the first
    recorded block paid five evaluations for `MAP_REG_ADDR_FAILED` and scored them
    as five bad tile shapes.

    Two separate things, and conflating them undercounts. The cache is keyed by
    bytes, so a configuration drawn twice occupies one entry -- but
    `RealEvaluator` ran the test command and the profiler both times, so the
    device was paid for twice. The meter counts device trials; the cache
    deduplicates bytes. A static rejection reached neither and is charged nothing,
    which is the existing rule for `feasible`.
    """
    cache = st.setdefault("eval_cache", {})

    def on_trial(rec):
        """One trial, as the driver saw it."""
        if rec.get("status") == "env_fault":
            # Not charged and not a trial -- but written down, immediately, with
            # the marker and the device's own last words. See `append_fault`.
            f = _fault_record(st, a, rec)
            ctx["faults"].append(f)
            append_fault(a.op_dir, f)
            return
        # Every status the driver returns BEFORE calling the objective. Missing
        # one here is expensive twice over: the trial is charged an evaluation it
        # never spent, and `cache[h] = {"s": 0, "p": None}` records a program that
        # was never run as measured-and-incorrect -- after which `probe` reports a
        # hit and `record` rejects those bytes as a duplicate, so a tile the gate
        # merely guessed about is unreachable for the rest of the run.
        #
        # `over_learned_l1` was missing, which is the whole point of that gate:
        # the same file's `_FREE` map already calls it free, so the block logged
        # `charged=N ... free=l1:M` about the trials it had just charged for.
        if rec.get("status") in _FREE:
            # Not charged, not cached -- but RECORDED. Those were the same
            # `return` before, so a block could report `free=static:15` in its log
            # line while the state held no trace of why any of the fifteen were
            # rejected. The gate that rejects three quarters of the draws is
            # exactly the one whose reasons have to be auditable: the bound it
            # enforces is the harness's own model of the buffer, not something the
            # compiler reports, and nobody can check a model whose verdicts are
            # not written down.
            ctx["trials"].append(
                {
                    "status": rec.get("status"),
                    "config": rec.get("config"),
                    "latency": None,
                    "reason": rec.get("reason", ""),
                }
            )
            return
        ctx["meter"]["device_trials"] += 1
        t = {
            "status": rec.get("status"),
            "config": rec.get("config"),
            "latency": rec.get("latency"),
            "reason": (rec.get("reason") or "")[:400],
        }
        if rec.get("metrics"):
            t["metrics"] = rec["metrics"]
        ctx["trials"].append(t)
        new_src, n = bayesian.block.apply_best(ctx["base_src"], rec.get("config"))
        if not n:
            return
        h = hashlib.sha256(new_src.encode("utf-8")).hexdigest()
        ctx["winners"].setdefault(h, 0)
        ctx["winners"][h] += 1
        cache.setdefault(
            h,
            {
                "s": 1 if rec.get("latency") else 0,
                "p": rec.get("latency"),
                "m": dict(rec.get("metrics") or {}),
                "via": "block",
            },
        )

    return on_trial


def _block_search(a, st, ctx, cfg):
    """Hand the tile space to the core and let it own the loop.

    `best_latency_us` is a measurement OF A PROGRAM, and the block seeds it as a
    free observation of the tile config now in the file. That is only true while
    the file IS that program. A block is normally fired right after the coder
    applied a delta -- the case it exists for is `retune: true`, i.e. the structure
    just moved -- so passing the number through attaches a latency taken on the
    old program to a structure that has never been measured.

    It surfaces as `best_latency` equal to the preopt with `best_was_incumbent:
    true` and `winner_hash: null`, which reads like a clean "the incumbent held"
    and is not one. A recorded run hit exactly this: the u1 block on TILE_M=1024
    returned best_latency 48.98, the TILE_M=512 number, while the only config it
    had actually measured on the new structure was 148.46. The optimizer caught it
    by hand, spent another evaluation re-running E(x), and got 47.04 -- so the
    block had reported the action as break-even when it was an improvement.

    Narrower than `memory_key`, deliberately. The memory asks whether an old
    measurement is admissible EVIDENCE, and the answer is the tile domain. This
    asks whether `best_latency_us` may be quoted as THIS program's incumbent, and
    that is only true of the exact bytes.
    """
    ev = bayesian.evaluator.RealEvaluator(
        op=st["op"],
        op_dir=a.op_dir,
        op_file=ctx["op_file"],
        device=st.get("device_id", ""),
        test_command=a.test_command,
        lint_root=(a.lint_root or None),
        eval_timeout_s=cfg.get("eval_timeout_s", 300),
        evaluation_mode=getattr(a, "evaluation_mode", "subprocess"),
        # Overridable so tests can assert the schedule without waiting it out.
        # The default is the production one and campaigns are told to pass
        # nothing else.
        fault_backoff_s=tuple(cfg.get("fault_backoff_s", (30, 90))),
    )
    same_program = hashlib.sha256(
        ctx["base_src"].encode("utf-8")
    ).hexdigest() == _best_hash(st)
    hw = ctx["hw"]
    return bayesian.block.run(
        ctx["op_file"],
        ev,
        hw,
        bayesian.block.BlockOptions(
            stagnation_k=cfg.get("stagnation_K", 7),
            max_trials=ctx["remaining"],
            current_latency=(
                st["progress"].get("best_latency_us") if same_program else None
            ),
            on_trial=_block_recorder(a, st, ctx),
            domain=ctx["domain"],
            # `bayesian.block.run` defaults this to 4, while `hw_for` derives the real
            # element width from the kernel and defaults it to 2. Leaving it
            # unthreaded meant every block on a bf16/fp16 kernel computed its 16 KB
            # floor seed and 96 KB ceiling seed at twice the true width, so it
            # searched the bottom half of the band it believed it was searching --
            # while bayesian.domain.verify used the correct width, so the two gates
            # disagreed by exactly 2x.
            dtype_bytes=hw.dtype_bytes,
            # Keep drawing at random until this many FEASIBLE trials are measured
            # before TPE takes over, so its surrogate is fitted on grounded data
            # rather than on the 1-2 feasible points a mostly-rejected block used to
            # hand it.
            startup_min_feasible=cfg.get(
                "bayesian_optimization_startup_min_feasible", 5
            ),
            memory=ctx["memories"].get(ctx["mem_key"]),
            refusals=ctx["refusals"],
        ),
    )


def _block_apply_winner(st, ctx, result):
    """Write the winning tile config into the op_file, if the block found one."""
    if not result.get("best_config") or result.get("best_was_incumbent"):
        return
    new_src, n = bayesian.block.apply_best(ctx["base_src"], result["best_config"])
    if not n:
        return
    winner_hash = hashlib.sha256(new_src.encode("utf-8")).hexdigest()
    with open(ctx["op_file"], "w", encoding="utf-8") as f:
        f.write(new_src)
    # Left OUT of the cache on purpose: `record` must see it as new, so it charges
    # the one remaining evaluation and runs the ratchet.
    st.setdefault("eval_cache", {}).pop(winner_hash, None)
    ctx["winner_hash"] = winner_hash


def _block_fold(a, st, ctx, result, cfg):
    """Fold the block's result back into the state: memory, refusals, budget, record."""
    if result.get("memory"):
        ctx["memories"][ctx["mem_key"]] = result["memory"]
    # Truthiness, not `is not None`. `bayesian.block._empty()` returns `refusals: []`
    # for every early return -- no tunable site, no budget left, a source that does
    # not parse -- and "no budget left" is a routine end-of-campaign condition. An
    # `is not None` guard let any of them wipe the whole history.
    if result.get("refusals"):
        # MERGE, never overwrite. `result["refusals"]` is the driver folding its
        # new rows into the ones it was handed, and it was handed only the subset
        # admissible on this structure -- assigning it back would drop every
        # refusal earned on every other structure.
        st["bayesian_optimization_refusals"] = merge_refusals(
            ctx["refusal_rows"],
            result["refusals"],
            ctx["structure"],
            cfg.get("bayesian_optimization_refusal_global_after", REFUSAL_GLOBAL_AFTER),
        )
    if ctx["winner_hash"]:
        st["domain_fingerprint"] = ctx["domain"]
    # Every device trial is charged. One is handed to `record` along with the
    # winner, so the total across block+record equals the trials that ran.
    # Never negative. The winner is left uncached so `record` charges it once, and
    # this subtracts it here so it is not charged twice -- but a RESUMED block can
    # return a winner drawn from its predecessor's replayed observations while
    # every fresh draw was rejected for free, i.e. device_trials == 0 with a winner
    # set. That refunded an evaluation the run had already spent.
    device_trials = ctx["meter"]["device_trials"]
    ctx["charged"] = max(0, device_trials - (1 if ctx["winner_hash"] else 0))
    st["progress"]["evals_used"] += ctx["charged"]
    st["progress"]["block_evals"] = st["progress"].get("block_evals", 0) + device_trials
    st.setdefault("blocks", []).append(
        {
            "action": a.parent,
            "reason": a.reason,
            "live_tiles": a.live_tiles,
            "granted": ctx["remaining"],
            "n_trials": result.get("n_trials"),
            "n_feasible": result.get("n_feasible"),
            "stopped_early": result.get("stopped_early"),
            "aborted": result.get("aborted"),
            "fault_kind": result.get("fault_kind"),
            "fault_markers": _fault_markers(ctx),
            "env_faults": result.get("env_faults"),
            "failures": result.get("failures"),
            "learned_ceiling": result.get("learned_ceiling"),
            "best_latency": result.get("best_latency"),
            "best_config": result.get("best_config"),
            "device_trials": device_trials,
            "charged_now": ctx["charged"],
            "winner_hash": ctx["winner_hash"],
            "started_at": ctx["started_at"],
            "ended_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "trials": ctx["trials"],
        }
    )


def _fault_markers(ctx):
    """The distinct device fault markers this block saw."""
    return sorted({f["marker"] for f in ctx["faults"] if f.get("marker")})


def _block_log(a, st, ctx, result):
    """One trajectory line for the block."""
    append_log(
        a.op_dir,
        st["op"],
        f"- block[{a.parent}]: {result.get('n_trials')} trials "
        f"({result.get('n_feasible')} feasible), best={result.get('best_latency')} "
        f"stopped_early={result.get('stopped_early')} "
        f"device_trials={ctx['meter']['device_trials']} charged={ctx['charged']} "
        f"env_faults={result.get('env_faults')} "
        f"failures={result.get('failures')} "
        f"replayed={result.get('replayed')} "
        # Why the free rejections happened. Without the split, a block
        # that rejected 35 of 42 trials before the device reports the
        # same line whether its gates are working or walled off its own
        # incumbent, and the second is not diagnosable from the log.
        f"free={_free_rejects(result)} "
        f"ceiling={result.get('learned_ceiling')}"
        # Surfaced in the trajectory, not only in domain.notes, because
        # it has now been forgotten in every recorded block. Without it
        # the footprint bound is OFF and the block pays a device
        # evaluation for each oversized tile instead of rejecting it for
        # free -- two per block in the last run, and the same ceiling
        # relearned from scratch each time.
        + (
            " | LIVE-TILES NOT SUPPLIED: footprint bound OFF, "
            "oversized tiles cost a device evaluation each"
            if not a.live_tiles
            else f" live_tiles={a.live_tiles}"
        )
        + (f" | ABORTED: {result['aborted']}" if result.get("aborted") else "")
        + (f" | {result['reason']}" if result.get("reason") else "")
        + (f" | reason: {a.reason}" if a.reason else ""),
    )


def _block_payload(a, ctx, result):
    """What the optimizer reads back from a block."""
    winner_hash = ctx["winner_hash"]
    out(
        {
            "ran": bool(result.get("n_trials")),
            "best_config": result.get("best_config"),
            "best_latency": result.get("best_latency"),
            "code_hash": winner_hash,
            "n_trials": result.get("n_trials"),
            "n_feasible": result.get("n_feasible"),
            "stopped_early": result.get("stopped_early"),
            "aborted": result.get("aborted"),
            "fault_kind": result.get("fault_kind"),
            "fault_markers": _fault_markers(ctx),
            "env_faults": result.get("env_faults"),
            "failures": result.get("failures"),
            "replayed": result.get("replayed"),
            "learned_ceiling": result.get("learned_ceiling"),
            "best_was_incumbent": result.get("best_was_incumbent"),
            "device_trials": ctx["meter"]["device_trials"],
            "charged": ctx["charged"],
            "granted": ctx["remaining"],
            "reason": result.get("reason"),
            "domain_notes": ctx["domain"].get("notes"),
            "capacity_elems": ctx["domain"].get("capacity_elems"),
            "live_tiles": a.live_tiles,
            "warning": (
                None
                if a.live_tiles
                else "--live-tiles was not supplied, so the footprint bound is OFF "
                "and every oversized tile costs a device evaluation. It is a "
                "FACT about the program (how many tile-shaped tensors are "
                "resident at once), not a bound: count them and pass it."
            ),
            "note": (
                "the winner is already written to the op_file and NOT yet charged; "
                "call `record` with this code_hash, s=1 and best_latency as p. "
                "Do not dispatch the coder and do not re-measure."
            )
            if winner_hash
            else None,
        }
    )


def cmd_block(a):
    """Run one tile-tuning block: the core owns the loop and returns one config.

    Fired two ways, one mechanism. A tile action is selected, or the optimizer
    asks after a structural change has made the previous optimum describe a
    program that no longer exists. What differs is only what happens afterwards.

    Budget. Every trial is a device evaluation. The losers are charged here and
    their bytes are cached so they can never be paid for twice; the winner is
    deliberately left uncached, so the ordinary `record` path charges it and runs
    the ratchet. The ratchet stays in exactly one place, and what reaches it is
    the pair (structure, tile) rather than a tile on its own.
    """
    st = load(a.op_dir)
    cfg = st["config"]
    if bayesian is None or not cfg.get("bayesian_optimization_tile_search", False):
        out(
            {
                "ran": False,
                "reason": "bayesian_optimization_tile_search is off, or the "
                "bayesian_optimization package is absent",
            }
        )
        return
    op_file = _op_file(st, a.op_dir)
    if not op_file or not os.path.exists(op_file):
        out(
            {
                "ran": False,
                "reason": f"op_file not found: {op_file!r} (cwd={os.getcwd()!r})",
            }
        )
        return
    ctx = _block_context(a, st, cfg)
    # Before the first evaluation of this block, not after it: see confirm_chip.
    bad = confirm_chip(st, st.get("device_id", ""))
    if bad is not None:
        save(a.op_dir, st)
        out(dict(bad, ran=False, refused=True))
        raise Refusal(3)
    # Kept so `stop` can ask the device a question of its own rather than end the
    # run on an observation a block made some time ago. Nothing else reads it.
    st["test_command"] = a.test_command
    # `_domain_state` reads `st["live_tiles"]`, and nothing ever wrote it. So the
    # fingerprint this block stores is derived WITH the residency count (capacity
    # bound on) while every later `feasible` re-derives it WITHOUT (capacity bound
    # off) -- the two can never match, and `_retune_verdict` reported
    # `capacity 12288 -> None elements` as a real move. Every block that produced
    # a winner therefore made the next `feasible` demand another block.
    if a.live_tiles:
        st["live_tiles"] = a.live_tiles
    result = _block_search(a, st, ctx, cfg)
    _block_apply_winner(st, ctx, result)
    _block_fold(a, st, ctx, result, cfg)
    save(a.op_dir, st)
    _block_log(a, st, ctx, result)
    _block_payload(a, ctx, result)


def cmd_normalize(a):
    """Static verdict on a kernel normalisation, before any device time.

    Normalisation runs at INIT, before preopt is measured, and it is the step
    that decides whether the deterministic core owns the tile at all: `bayesian.apply`
    tunes a call only when every argument is an integer literal, so a kernel
    written `set_vec_tile_shapes(TILE_B, TILE_D)` produces zero BO trials no
    matter what else is fixed.

    Three gates decide admissibility. This is the one that costs nothing:

        this command      values unchanged, no call added or removed,
                          tunable sites actually increased
        the golden test   semantics preserved
        preopt neutrality same latency within the noise floor

    It is deliberately not a semantics check -- the golden test already answers
    that exactly, and constant folding cannot. It is the check that the rewrite
    did not TUNE while it refactored, which nothing else can see: a normalisation
    that quietly changes 64 to 8192 would pass both other gates and would move
    the baseline every speedup in the run is divided by.
    """
    if bayesian is None:
        out(
            {
                "ok": False,
                "reasons": ["bayesian_optimization package not available on this arm"],
            }
        )
        raise Refusal(2)
    for path in (a.before, a.after):
        if not os.path.exists(path):
            out({"ok": False, "reasons": [f"not found: {path}"]})
            raise Refusal(2)
    before = _read_text(a.before)
    after = _read_text(a.after)
    report = bayesian.normalize.check(before, after)
    if a.op_dir and os.path.isdir(a.op_dir):
        append_log(
            a.op_dir,
            a.op or "",
            f"- normalize: ok={report['ok']} "
            f"tunable {report.get('before', {}).get('tunable')} -> "
            f"{report.get('after', {}).get('tunable')}"
            + ("" if report["ok"] else " | " + "; ".join(report["reasons"])),
        )
    out(report)
    if not report["ok"]:
        raise Refusal(1)


def _init_code_version(a):
    """Provenance is a precondition, not a nicety.

    A state file with no code_version cannot be attributed to a build, so two
    runs cannot be compared and neither can be reported. Refuse to start rather
    than produce a log that is silently unusable.
    """
    cv = _code_version()
    if cv is None and not getattr(a, "allow_unversioned", False):
        out(
            {
                "initialized": False,
                "error": "code_version_unavailable",
                "hint": "run the script from inside its git work tree, or write a "
                "VERSION file next to it, or pass --allow-unversioned",
            }
        )
        raise Refusal(2)
    return cv


def _init_capability(cfg):
    """BO capability preflight.

    The flag says which algorithm this run claims to use, so it must not be
    possible to claim TPE and get the model's guesses: with the flag ON and the
    package missing, every tile proposal would come back empty and the coder would
    pick the numbers, while the config, the log and the report all still said the
    search was on.

    Refusing is right and stopping here is not. Without the tile search this is no
    longer PANKO, and WHICH of the two ways out to take -- install the dependency,
    or run the other method deliberately -- is the user's call, not the harness's
    and not the agent's. So the refusal carries the question to put to them.
    `ask_user` is the marker the skill keys off; an agent that answers it on the
    user's behalf has substituted its own judgement for a decision about what the
    run is.
    """
    capability = bayesian_capability()
    if (
        not cfg.get("bayesian_optimization_tile_search", True)
        or capability["available"]
    ):
        return capability
    floor = ".".join(str(x) for x in OPTUNA_MIN_VERSION)
    out(
        {
            "initialized": False,
            "refused": True,
            "error": "bayesian_optimization_unavailable",
            "bayesian_optimization": capability,
            "ask_user": {
                "question": (
                    f"PANKO needs optuna (>= {floor}) to search tile values, and it is "
                    f"not importable here ({capability['reason']}). Tile numbers are the "
                    "one lever the model cannot propose, so without it this is a "
                    "different method rather than a slower one. How do you want to "
                    "proceed?"
                ),
                "options": [
                    {
                        "id": "install",
                        "label": "Install optuna and run PANKO",
                        "action": f"python3 -m pip install 'optuna>={floor}', then re-run init",
                    },
                    {
                        "id": "ablation",
                        "label": "Run Stage 7 without the tile search",
                        "action": (
                            "re-run init with --config "
                            "'{\"bayesian_optimization_tile_search\": false}' -- the "
                            "structural search still runs; tile values stay as the "
                            "kernel has them unless the coder changes them"
                        ),
                    },
                    {
                        "id": "stepwise",
                        "label": "Abandon PANKO and use the stepwise Stage 7 path",
                        "action": "report back to the orchestrator; no PANKO state is written",
                    },
                ],
            },
            "hint": "relay the question to the user; do not choose for them",
        }
    )
    raise Refusal(3)


def _init_envelope(a):
    """The chip envelope is DERIVED, not assumed.

    Every static rule -- the UB footprint bound, the L0/L1 capacity checks, the
    core-count seeding -- is validated against it, so running one chip's numbers
    on another rejects legal candidates before any device sees them, and leaves no
    evidence that it did. The review is explicit that silently using the A3
    defaults is the thing not to do.
    """
    envelope, why = _resolve_envelope(a)
    if envelope is not None:
        return envelope
    out(
        {
            "initialized": False,
            "refused": True,
            "error": "chip_envelope_unresolved",
            "reason": why,
            "ask_user": {
                "question": (
                    "PANKO validates every candidate against this chip's buffer "
                    "capacities and core counts, and could not derive them here: "
                    f"{why}. Another chip's numbers would reject legal tiles "
                    "before the device ever saw them, so there is no safe "
                    "default. How do you want to proceed?"
                ),
                "options": [
                    {
                        "id": "cann_env",
                        "label": "Make the platform ini reachable",
                        "action": "source the CANN set_env.sh, or set "
                        "ASCEND_TOOLKIT_HOME, then re-run init",
                    },
                    {
                        "id": "name_soc",
                        "label": "Name the SoC explicitly",
                        "action": "PANKO_SOC_NAME=<your SoC name>; the "
                        "harness then reads <platform_config>/<name>.ini",
                    },
                    {
                        "id": "explicit",
                        "label": "Supply the envelope yourself",
                        "action": '--chip-envelope \'{"ub_kb":192,"l1_kb":512,'
                        '"l0a_kb":64,"l0b_kb":64,"l0c_kb":128,'
                        '"cube_cores":20,"vector_cores":40}\' -- recorded '
                        "as explicit:cli so the report says the numbers "
                        "were given rather than read",
                    },
                ],
            },
            "hint": "relay the question to the user; do not choose for them",
        }
    )
    raise Refusal(3)


def _init_baseline_hash(a, op_file):
    """The baseline hash is COMPUTED from the file, not taken on the caller's word.

    Everything the run does afterwards is anchored to this snapshot: `select`
    restores it, `record` rolls back to it, `stop` delivers from it. A hash that
    names no bytes turns that chain into "whatever is in the working file right
    now", and which program a candidate was built on then depends on the caller's
    path and on residue from earlier runs.
    """
    preopt_hash = _file_hash(op_file)
    if preopt_hash is None:
        out(
            {
                "initialized": False,
                "refused": True,
                "error": "op_file_unreadable",
                "op_file": op_file,
                "hint": "PANKO snapshots and restores this file on every cycle; it "
                "must exist and be readable before the search starts. Pass "
                "--op-file if the kernel is not at <op_dir>/<op>_impl.py.",
            }
        )
        raise Refusal(3)
    if a.preopt_hash and a.preopt_hash != preopt_hash:
        # Not a warning. The caller believes it measured one program and the file
        # on disk is another, so the preopt latency this run normalises every
        # speedup against was taken on a kernel that is not the baseline.
        out(
            {
                "initialized": False,
                "refused": True,
                "error": "preopt_hash_mismatch",
                "op_file": op_file,
                "given": a.preopt_hash,
                "actual": preopt_hash,
                "hint": "--preopt-hash does not match the bytes of op_file. Re-measure "
                "the file as it stands, or point --op-file at the kernel that "
                "was measured. Omit --preopt-hash to let the harness derive it.",
            }
        )
        raise Refusal(3)
    return preopt_hash


def _init_progress(a, opening):
    """The progress counters as `init` starts them."""
    return {
        "active_stage": "frontend",
        "stage_stagnation": 0,
        "evals_used": 1,
        "wall_clock_s": 0,
        "_start_ts": time.time(),
        "global_stagnation": 0,
        "best_J": opening["j0"],
        "preopt_us": a.preopt_p,
        "best_latency_us": a.preopt_p,
        "best_speedup_vs_preopt": 1.0,
        "cycle_productive": False,
        "converged": False,
        "static_rejected": 0,
        "duplicate_rejected": 0,
        "frontier_emptied_by_prune": 0,
        "expansion_requests": 0,
        "expansion_required": False,
        "stop_reason": None,
        "refine": None,
    }


def _init_nodes(a, opening, preopt_m):
    """The node tree at init: the preopt program, already measured and closed."""
    return {
        "closed": [
            {
                "id": "root",
                "parent": None,
                "stage": "frontend",
                "code_hash": opening["preopt_hash"],
                "J": opening["j0"],
                "eval": {"s": a.preopt_s, "p": a.preopt_p, "m": preopt_m},
            }
        ],
        "open": [],
        "pruned": [],
    }


def _init_state(a, cfg, opening):
    """The state file as `init` writes it, before any action is seeded into it."""
    preopt_m = _metrics(
        getattr(a, "preopt_util", None),
        getattr(a, "preopt_bubble", None),
        pipes=_cli_pipes(a, "preopt_"),
    )
    st = {
        "op": a.op,
        "device_id": a.device,
        "target": {"metric": "latency_us", "P_ref": a.p_ref, "lower_is_better": True},
        "config": cfg,
        "_seq": 0,
        "progress": _init_progress(a, opening),
        "nodes": _init_nodes(a, opening, preopt_m),
        "best_node": "root",
        "op_file": (
            a.op_file or None
        ),  # working <op>_impl.py the harness snapshots/restores
        # The preopt reading was accepted at INIT, used once for symptom anchoring
        # and then dropped, so every later close re-read an empty `m` on the root.
        "eval_cache": {
            opening["preopt_hash"]: {"s": a.preopt_s, "p": a.preopt_p, "m": preopt_m}
        },
        "code_version": opening["cv"],
        # What actually ran, not what was asked for. A report that names the
        # algorithm has to read this rather than the config flag.
        "bayesian_optimization": opening["capability"],
        # What every static rule in this run is validated against. Persisted so a
        # resumed campaign keeps the envelope it started with, and so a report
        # can say which chip the numbers were admissible for.
        "chip_envelope": opening["envelope"],
        # The host wrapper as it stands at the baseline. Frozen for the run:
        # see `wrapper_signature`.
        "wrapper_signature": wrapper_signature(opening["source"]),
    }
    st["nodes"]["held"] = []
    return st


def _seed_frontier(st, a, cfg, seeds):
    """Seed the frontier from the action catalog. Returns the predicate reading."""
    preds = (
        predicates.compute(a.op_file or "", a.op_dir)
        if (predicates and cfg.get("action_preconditions", True))
        else None
    )
    st["predicates"] = preds
    for sd in seeds:
        if sd.get("conversion") and not cfg.get("conversion_actions", True):
            continue
        node = {
            "id": _next_id(st, "u"),
            "parent": "root",
            "stage": sd["stage"],
            "catalog_id": sd.get("id"),  # keep the F-x/S-x/I-x id for symptom matching
            "delta": sd["delta"],
            "V": float(sd.get("prior_gain", 0.5)),
            "prior_gain": float(sd.get("prior_gain", 0.5)),
            "tried": 0,
            "requires": sd.get("requires", []),
        }
        if preds is not None and not predicates.satisfies(node["requires"], preds):
            st["nodes"]["held"].append(node)  # held, never deleted
        else:
            st["nodes"]["open"].append(node)
    return preds


def _anchoring_warning(a, cfg):
    """Whether symptom anchoring is on, and the warning it earns when it cannot fire.

    Silent no-op is the dangerous case: without metrics the two PROFILE symptoms
    cannot fire at all, so a whole campaign can run with anchoring configured on
    and never once applied to a bottleneck.

    `occupancy` is unaffected -- it is read off the source -- so this is a warning
    about two of the three symptoms rather than all of them, and it is worded that
    way. A run that logs this warning AND a non-empty seeded_symptoms is a run
    where only the source-derived symptom is live.
    """
    anchoring = float(cfg.get("symptom_boost", 1.0)) > 1.0
    warning = None
    if anchoring and not (
        getattr(a, "preopt_util", 0.0) or getattr(a, "preopt_bubble", 0.0)
    ):
        warning = (
            "symptom_boost is enabled but no --preopt-util/--preopt-bubble "
            "were supplied; the two PROFILE symptoms (bubble, util) cannot "
            "fire until record is called with --util/--bubble. The "
            "source-derived `occupancy` symptom is unaffected"
        )
    return anchoring, warning


def _init_snapshot(a, op_file, preopt_hash):
    """Persist the baseline. Emits and raises when it cannot be written.

    Called before `save`, so a refused init leaves no state behind: a half-built
    run whose baseline is missing is worse than no run, because `select` would
    restore nothing and every candidate would stack on the last one.
    """
    if _snapshot(a.op_dir, op_file, preopt_hash):
        return
    out(
        {
            "initialized": False,
            "refused": True,
            "error": "baseline_snapshot_failed",
            "op_file": op_file,
            "snapshot": _npath(a.op_dir, preopt_hash),
            "hint": "the preopt snapshot could not be written; check the "
            "permissions on optimization/nodes/ and the free space there.",
        }
    )
    raise Refusal(3)


def _init_announce(a, st, opening, report):
    """Write the opening line of the trajectory log and emit init's payload."""
    warning, lint_warning = report["warning"], report["lint_root_warning"]
    n_folds, chip_note = report["n_folds"], report["chip_mismatch"]
    append_log(
        a.op_dir,
        a.op,
        f"- init: code_version={st.get('code_version')} "
        f"anchoring={report['anchoring']} "
        f"seeded_symptoms={report['seeded_symptoms']} "
        f"open={len(st['nodes']['open'])} held={len(st['nodes'].get('held', []))} "
        f"predicates={report['preds']} semantic_fold_calls={n_folds}"
        + (
            "  <- pypto not importable: the semantic no-op check compares "
            "ASTs only and cannot fold a restated default"
            if not n_folds
            else ""
        )
        + (f" | WARNING: {warning}" if warning else "")
        + (f" | LINT-ROOT: {lint_warning}" if lint_warning else ""),
    )
    save(a.op_dir, st)
    out(
        {
            "initialized": True,
            "best_J": opening["j0"],
            "open": len(st["nodes"]["open"]),
            "held": len(st["nodes"].get("held", [])),
            "predicates": report["preds"],
            "preopt_snapshot": True,
            "preopt_hash": opening["preopt_hash"],
            "bayesian_optimization": opening["capability"],
            # So a caller never has to guess where the skill's own files are.
            "skill_root": SKILL_ROOT,
            "catalog": a.catalog,
            "chip_envelope": st["chip_envelope"],
            # Present only when the chip in the slot is not the chip the envelope
            # describes. The run is initialised either way; the first block is what
            # stops.
            **({"chip_mismatch": chip_note} if chip_note else {}),
            "seeded_symptoms": report["seeded_symptoms"],
            "code_version": st.get("code_version"),
            "anchoring": report["anchoring"],
            "semantic_fold_calls": n_folds,
            "warning": warning,
            "lint_root_warning": lint_warning,
        }
    )


def cmd_init(a):
    cfg = dict(DEFAULT_CONFIG)
    if a.config:
        cfg.update(json.loads(a.config))
    opening = {
        "cv": _init_code_version(a),
        "capability": _init_capability(cfg),
        "envelope": _init_envelope(a),
    }
    op_file = a.op_file or os.path.join(a.op_dir, f"{a.op}_impl.py")
    opening["preopt_hash"] = _init_baseline_hash(a, op_file)
    with open(op_file, encoding="utf-8", errors="replace") as f:
        opening["source"] = f.read()
    seeds = []
    if a.catalog:
        with open(a.catalog, encoding="utf-8") as f:
            seeds = json.load(f)
    opening["j0"] = compute_j(a.preopt_s, a.preopt_p, a.p_ref)
    st = _init_state(a, cfg, opening)
    preds = _seed_frontier(st, a, cfg, seeds)
    # The space the preopt's tile was chosen for. Every later delta is compared
    # against this, and against whatever replaced it when a candidate was kept.
    st["domain_fingerprint"] = _domain_state(st, _op_file(st, a.op_dir))
    seeded_symptoms = _symptom_reboost(
        st,
        getattr(a, "preopt_util", 0.0),
        getattr(a, "preopt_bubble", 0.0),
        _op_file(st, a.op_dir),
        _cli_pipes(a, "preopt_"),
    )  # first-move symptom bias
    _init_snapshot(a, op_file, opening["preopt_hash"])
    save(a.op_dir, st)
    evaluation_session.clear_terminal_marker(a.op_dir)
    append_log(
        a.op_dir,
        a.op,
        f"# {a.op} — PANKO optimization\n\n## Baseline\n"
        f"- preopt latency: {a.preopt_p} us | P_ref: {a.p_ref} us | j0: {opening['j0']}\n"
        f"- device: {a.device}\n\n## Trajectory\n",
    )
    report = {"preds": preds, "seeded_symptoms": seeded_symptoms}
    report["anchoring"], report["warning"] = _anchoring_warning(a, cfg)
    report["lint_root_warning"] = _check_lint_root(getattr(a, "lint_root", ""))
    # Say out loud whether the semantic-no-op fold is live. It reads the real
    # `pypto` signatures, so off-device it degrades to comparing ASTs and folds
    # nothing -- and a guard that is silently off in the environment where it can
    # be tested, and on only where it cannot, is how `bubble` stayed 0.0 for an
    # entire campaign. A run that logs `folds=0` and rejects no no-ops has not
    # shown there were none.
    report["n_folds"] = len(_pypto_defaults())
    st["semantic_fold_calls"] = report["n_folds"]
    # Ask the device now, while someone is watching. `init` must not REFUSE on
    # what it hears -- it has to stay runnable with no device, and reading
    # another box's ini deliberately is a legal thing to do -- but a mismatch
    # surfaced here is one surfaced before the campaign is configured around it
    # rather than at the first block. The halt itself lives at the first
    # evaluation: see `confirm_chip`.
    report["chip_mismatch"] = confirm_chip(st, a.device)
    _init_announce(a, st, opening, report)


def _check_lint_root(lint_root):
    """Is `--lint-root` a directory `python -m pypto_op_lint` can be run from?

    A wrong one is not a small mistake. It makes the lint gate raise
    `No module named pypto_op_lint` on EVERY trial, the evaluator cannot tell
    that from a device that has gone away, and the block aborts on consecutive
    environment faults having measured nothing. A run can lose its first block
    exactly this way -- a few faults with `fault_kind: invocation`, the repo
    root passed where `cannbot-skills/plugins-official/
    pypto-op-orchestrator/hooks/pypto-op-lint` was wanted -- and an
    `os.path.isdir` at init catches it before the device is touched.

    Returns "" when there is nothing to say. This WARNS rather than refuses:
    a valid layout the check does not recognise must not stop a campaign.
    """
    if not lint_root:
        return ""
    if not os.path.isdir(lint_root):
        return (
            f"not a directory: {lint_root!r} -- the lint gate will fault on every trial"
        )
    if not (
        os.path.exists(os.path.join(lint_root, "pypto_op_lint"))
        or os.path.exists(os.path.join(lint_root, "pypto_op_lint.py"))
    ):
        return (
            f"{lint_root!r} holds no importable `pypto_op_lint` -- "
            f"`python -m pypto_op_lint` will fail on every trial and the "
            f"block will abort on invocation faults. Point --lint-root at "
            f"the directory CONTAINING the package "
            f"(usually <repo>/cannbot-skills/plugins-official/"
            f"pypto-op-orchestrator/hooks/pypto-op-lint), not at the repo root"
        )
    return ""


_FREE = {
    "infeasible_static": "static",
    "over_learned_ceiling": "ceiling",
    "duplicate_config": "dup",
    # A configuration the compiler refused in an EARLIER block, on a
    # different structure. Free by the same argument as `ceiling`: the
    # device already answered this question and the answer does not depend
    # on the program shape that asked it.
    "known_compile_failure": "refused",
}


def _free_rejects(result):
    """Counts of the trials rejected without touching the device, by cause."""
    c = {}
    for h in result.get("history", []):
        k = _FREE.get(h.get("status"))
        if k:
            c[k] = c.get(k, 0) + 1
    return "/".join(f"{k}:{v}" for k, v in sorted(c.items())) or "0"


def _structure(st, op_dir):
    """The current program's structural signature, or None if unavailable."""
    if bayesian is None:
        return None
    f = _op_file(st, op_dir)
    try:
        return bayesian.lever.structural_signature(_read_text(f, errors="strict"))
    except Exception:
        return None


def _live(st, op_dir):
    """The frontier minus actions retired on the structure now in the file.

    A measurement disproves an action ON THE PROGRAM IT WAS TAKEN ON. A sweep
    of a loop granularity can find a sharp interior optimum and prune the other
    granularity actions -- correctly, for the program in front of it. A later
    action that merges two matmuls into one then wants a granularity that sweep
    had already measured and retired. Because the prune was permanent, every
    later structural action is explored at the old geometry, and the
    restructuring that needed the other one can never show its value. The same
    happens to a cube tile: several independent mechanisms agreeing on one value
    is still one structure's evidence, generalised past the structure that
    produced it.

    So a prune is scoped to `structural_signature` -- the program hash with tile
    literals erased, which moves when the structure moves and not when a tile
    value does. While the structure is unchanged the action stays retired; once
    something restructures the kernel its evidence is stale and it returns to the
    frontier. It returns as an ORDINARY candidate: `pypto_action_priority` ranks
    it against everything else with whatever prior_gain and `tried` count it
    carries. Nothing is promoted, nothing is ordered by fiat.

    `prune_permanent` stays permanent, for the case that is not about structure
    at all -- u12 (NZ weight format) is inexpressible without changing the
    operator's frozen layout, and no restructuring makes it expressible.
    """
    sig = _structure(st, op_dir)
    return [
        n
        for n in st["nodes"]["open"]
        if not (n.get("pruned_on") and n["pruned_on"] == sig)
    ]


def _refuse_if_action_open(st):
    """An open refinement has to be finished before another action is chosen.

    Without this, making `close` refuse achieves nothing: the optimizer can
    leave an action at n=1 simply by selecting a different one and never
    coming back, which is the same abandonment the refusal exists to prevent,
    and it also leaves `refine` pointing at an action whose candidates the
    next `record` would be attributed to.
    """
    r = st["progress"].get("refine") or {}
    if not r.get("action"):
        return
    why = _refuse_early_close(st, r["action"])
    if why:
        out(
            {
                "selected": None,
                "refused": True,
                "reason": f"action {r['action']} is still open: {why}",
                "open_action": r["action"],
                "n": r.get("n", 0),
            }
        )
        raise Refusal(3)


def _out_expansion_required(st):
    # Not a stop. The frontier is the model's to extend.
    out(
        {
            "selected": None,
            "reason": "expansion_required",
            "declared_exhausted": st["progress"].get("expansion_declined"),
            "hint": "reflect over the trajectory and call `evolve` with at least one "
            "insert. If there is genuinely nothing left to try, call `evolve` "
            'with exhausted:"<reason>"; only then may the search stop.',
        }
    )


def _restore_or_refuse(a, restore_hash, op_f):
    """Restore the running global best into the working file, or refuse.

    Refuse rather than hand out an action. The response says "apply your
    delta on top of restore_code_hash", and the file is not that program,
    so the candidate's gain would be measured against a base nobody
    recorded and its parent link in the tree would be a fiction. Returning
    `restored: false` and continuing made that the caller's problem, and
    nothing downstream checks it.
    """
    restored = _restore(a.op_dir, op_f, restore_hash)
    if restored or _file_hash(op_f) == restore_hash:
        return restored
    out(
        {
            "selected": None,
            "refused": True,
            "error": "restore_failed",
            "restore_code_hash": restore_hash,
            "op_file": op_f,
            "snapshot": _npath(a.op_dir, restore_hash) if restore_hash else None,
            "reason": (
                "the global best has no snapshot on disk"
                if restore_hash
                else "no global best code_hash is recorded"
            ),
            "hint": "optimization/nodes/<code_hash>.py is the search's memory of "
            "every kept program. Without it the next candidate would be "
            "built on whatever is in the working file.",
        }
    )
    raise Refusal(3)


def cmd_select(a):
    st = load(a.op_dir)
    _refuse_if_action_open(st)
    active = st["progress"]["active_stage"]
    staged = st["config"]["staged_prior"]
    frontier = _live(st, a.op_dir)
    if not frontier:
        _out_expansion_required(st)
        return
    free = _bayes_free_trials(st)
    best = max(frontier, key=lambda n: pypto_action_priority(n, active, staged, free))
    # greedy accumulation: restore the running global best into the working file NOW, so the
    # optimizer applies this action's delta ON TOP of the accumulated best (deterministic —
    # not left to the agent). restore_code_hash is still returned for visibility/logging.
    restore_hash = _best_hash(st)
    op_f = _op_file(st, a.op_dir)
    restored = _restore_or_refuse(a, restore_hash, op_f)
    # `already_measured` is the whole point: the file the coder is about to edit IS
    # the global best and its bytes are already in the cache. Submitting it unchanged
    # is the single most common way budget was wasted (see cmd_record).
    payload = {
        "selected": {
            "id": best["id"],
            "parent": best["parent"],
            "stage": best["stage"],
            "delta": best["delta"],
            "V": best["V"],
            "pypto_action_priority": pypto_action_priority(best, active, staged, free),
            "restore_code_hash": restore_hash,
            "restored": bool(restored),
            "already_measured": restore_hash,
            "known_hashes": len(st.get("eval_cache", {})),
        }
    }
    proposal = _tile_bayes_propose(st, a.op_dir, best)
    if proposal:
        payload["bayesian_optimization"] = proposal
    save(a.op_dir, st)
    out(payload)


def _bayes_next(st, op_dir, action_id):
    """The lever's next value for an action that is continuing.

    Returns None for every action that is not a tile lever -- `_tile_bayes_propose`
    already makes that decision -- so the ordinary refinement loop is untouched.
    """
    for group in ("open", "held"):
        for n in st["nodes"].get(group) or []:
            if isinstance(n, dict) and n.get("id") == action_id:
                return _tile_bayes_propose(st, op_dir, n)
    return None


def _tile_bayes_propose(st, op_dir, node):
    """For a tile action, choose the next value and write it. Returns the record
    to attach to `select`, or None when this action is not a BO lever.

    The core writes the file itself rather than handing an intent to the coder.
    That is not a shortcut, it removes a failure mode: a tile can be bound at
    two sites, the wrapper's padding arithmetic and the kernel, and a coder that
    changes one and not the other ships a kernel that fails verification. The
    action is then scored as unproductive and closed, and the path it had opened
    is lost to a single bookkeeping mistake. `bayesian.apply` rewrites every
    site or none.

    `bayesian.apply` is a script, not an agent, so the rule that only the coder writes
    kernel code is untouched -- the harness already writes files in `_snapshot`
    and `_restore`.
    """
    if bayesian is None or not st["config"].get(
        "bayesian_optimization_tile_search", False
    ):
        return None
    if node.get("catalog_id") not in bayesian.lever.TILE_ACTIONS:
        return None
    op_file = _op_file(st, op_dir)
    if not op_file or not os.path.exists(op_file):
        # Loud, not silent. `op_file` is stored at init as a path relative to the
        # server's working directory; if the harness is invoked from anywhere
        # else it resolves to nothing, the lever quietly does not fire, and the
        # run looks completely normal while the core owns none of the numbers.
        return {
            "applied": False,
            "reason": f"op_file not found: {op_file!r} (cwd={os.getcwd()!r}). "
            f"bayesian_optimization_tile_search is ON but the lever cannot read the kernel.",
        }
    src = _read_text(op_file)
    studies = st.setdefault("bayesian_optimization_studies", {})
    # The same 2x `cmd_block` threads below, in the one path that never got it.
    # `bayesian.space.HW` defaults `dtype_bytes` to 4 and so does `bayesian.lever.ask`, while
    # `hw_for` reads the real operand width off the kernel -- so built bare, the
    # lever sized a bf16 kernel's L0A/L0B/L1 AND its own tile-footprint ceiling
    # at fp32. That is not conservative, it is wrong: it rejects kernels that
    # RAN. A delivered cube tile of kL0=128, nL0=256 is exactly 64 KB of L0B at
    # 2 B and 128 KB at 4 B -- legal as written, illegal if read as fp32.
    #
    # The cost is not one axis. A draw holds every site but one at the anchor,
    # and the gate is an AND over all of them, so an anchor the gate calls
    # illegal fails every draw that does not move the offending site -- the vec
    # tiles and the other cube sites become unreachable, and the only proposals
    # left are ones that shrink the offending site below its delivered value,
    # which it can then never climb back to. Meanwhile the block path, correct
    # at 2 B, kept working: the same search_state.json says "at 2B" under
    # `blocks` and "at 4B" under `bayesian_optimization_studies`.
    hw, domain = bayesian.block.hw_for(op_file, st.get("live_tiles"), chip_envelope(st))
    cfg, meta = bayesian.lever.ask(
        studies,
        node["catalog_id"],
        src,
        hw,
        bayesian.lever.AskOptions(
            dtype_bytes=domain.get("dtype_bytes", 4),
            current_latency=st["progress"].get("best_latency_us"),
        ),
    )
    if cfg is None:
        return {"applied": False, "reason": meta.get("reason", "no candidate")}
    new_src, n = bayesian.apply.apply(src, cfg)
    if not n or new_src == src:
        return {"applied": False, "reason": "no site rewritten"}
    with open(op_file, "w", encoding="utf-8") as f:
        f.write(new_src)
    return {
        "applied": True,
        "config": cfg,
        "label": meta.get("label"),
        "trial": meta.get("trial"),
        "sites": n,
        "code_hash": hashlib.sha256(new_src.encode("utf-8")).hexdigest(),
        "note": "already written to the kernel; do NOT dispatch the coder, "
        "just evaluate and record",
    }


def _tile_bayes_tell(st, op_dir, parent_id, s, p):
    """Report the measurement to the study that proposed it. Silent when the
    action is not a BO lever: every other action still goes through the model.
    """
    if bayesian is None or not st["config"].get(
        "bayesian_optimization_tile_search", False
    ):
        return
    node = next((n for n in st["nodes"]["open"] if n["id"] == parent_id), None)
    cid = (node or {}).get("catalog_id")
    if cid not in bayesian.lever.TILE_ACTIONS:
        return
    op_file = _op_file(st, op_dir)
    if not op_file or not os.path.exists(op_file):
        return
    src = _read_text(op_file)
    bayesian.lever.tell(
        st.setdefault("bayesian_optimization_studies", {}), cid, src, s, p
    )


def _record_duplicate(a, st, pr, cache):
    """Byte-identical resubmission.

    Two ways this happens, both observed: the coder is handed back its own local
    best by the roll-back in `_apply_ratchet` and edits nothing, or it is handed
    the global best by `select` and edits nothing. Either way the bytes were
    already measured, so this is not a candidate. Reject it WITHOUT touching
    r["n"] -- counting it as a regression let an action be closed by re-proposing
    its own best three times.

    Returns True when it has emitted a rejection and the caller must stop.
    """
    if a.code_hash not in cache:
        return False
    pr["duplicate_rejected"] = pr.get("duplicate_rejected", 0) + 1
    r0 = _refine_for(pr, a.parent)
    # A run of duplicates is the only honest signal that an action has no
    # unmeasured value left, and it is one the harness observes rather than one
    # the optimizer declares. `close` now refuses while n < stagnation_K, so an
    # action whose axis really is a three-point space would otherwise deadlock:
    # it cannot produce a new candidate, cannot advance n, and cannot be closed.
    # Three consecutive duplicates release it.
    #
    # Deliberately NOT in SKILL.md, and deliberately not reported in this
    # response. A documented "three duplicates ends an action" is a cheap exit the
    # optimizer can take whenever refining gets tedious, which is the behaviour
    # the enforcement exists to stop. It has to be reachable by an action that has
    # genuinely run out, and invisible to one that has not.
    r0["dup_streak"] = r0.get("dup_streak", 0) + 1
    # Roll back, exactly as the non-improving branch does. This branch used to
    # return with the working file still holding the rejected trial's bytes -- and
    # a duplicate is by definition a program that was already measured and did NOT
    # become the best, so what stayed on disk was a loser. Every candidate after
    # it was then built on that base. Observed twice in one run: a BO trial
    # rejected as a duplicate left the pre-loop vector tile at the study's (4,64)
    # instead of the incumbent's (1,64), so two "single-variable" candidates in
    # fact changed two sites, and the optimizer's own account of what it had
    # tested was wrong.
    _rollback(a, st, r0)
    append_log(
        a.op_dir,
        st["op"],
        f"- refine[{a.parent}] cand {a.code_hash[:8]}: DUPLICATE rejected "
        f"(already measured; no budget charged, n unchanged at {r0.get('n', 0)}) "
        f"| duplicates so far: {pr['duplicate_rejected']}",
    )
    out(
        {
            "duplicate": True,
            "improved": False,
            "cached": cache[a.code_hash],
            "J": cache[a.code_hash].get("J"),
            "n": r0.get("n", 0),
            "stop_refine": False,
            "hint": "these bytes were already evaluated -- call `probe` before E(x); "
            "if the coder cannot change this file, the action does not apply",
        }
    )
    return True


def _record_semantic_noop(a, st, pr, j):
    """Backstop for the gate in `feasible`.

    That is where a semantic no-op is caught for free; this is where it is stopped
    from being KEPT, for runs that reach `record` without the gate. The
    measurement has already been paid by the time we get here, so this cannot save
    the evaluation -- only the delivered kernel. Same treatment as a
    byte-duplicate, and for the same reason: it is the same program, so it is not
    a candidate and `n` must not move.

    Returns True when it has emitted a rejection and the caller must stop.
    """
    # Only when the file on disk IS the bytes being recorded. `record` is told a
    # `code_hash` and reads a file, and nothing forces the two to agree -- a block
    # winner is promoted through here, `select` restores a snapshot, and the tests
    # drive the state machine with hashes that name no file at all. Judging a
    # candidate by a file that is not it would reject on the strength of whatever
    # happened to be on disk.
    same, why = (
        (False, "")
        if _file_hash(_op_file(st, a.op_dir)) != a.code_hash
        else _semantic_noop(st, a.op_dir, _op_file(st, a.op_dir), a.parent)
    )
    if not same:
        return False
    pr["semantic_noop_rejected"] = pr.get("semantic_noop_rejected", 0) + 1
    r0 = _refine_for(pr, a.parent)
    # `dup_streak`, exactly as the byte-duplicate branch. This IS a duplicate --
    # the same program, missed by the byte hash -- and without it the branch had
    # no termination path at all: `n` must not move (nothing new was measured), so
    # an action that can only resubmit its own program could neither advance nor
    # be closed.
    r0["dup_streak"] = r0.get("dup_streak", 0) + 1
    _rollback(a, st, r0)
    append_log(
        a.op_dir,
        st["op"],
        f"- refine[{a.parent}] cand {a.code_hash[:8]}: SEMANTIC NO-OP rejected "
        f"({why}); n unchanged at {r0.get('n', 0)}, "
        f"dup_streak={r0['dup_streak']}",
    )
    out(
        {
            "semantic_noop": True,
            "improved": False,
            "J": j,
            "n": r0.get("n", 0),
            "stop_refine": False,
            "reason": f"this candidate is the incumbent program: {why}",
            "hint": "call `feasible` before E(x) -- this would have been caught "
            "without spending the measurement",
        }
    )
    return True


def _record_wrapper_drift(a, st, pr):
    """`feasible` is the cheap gate and it is not mandatory, so the ratchet checks
    too. A measurement taken on a changed wrapper is not comparable with the rest
    of the campaign and must not be allowed to set the best. Emits and raises."""
    drift = _wrapper_drift(st, _op_file(st, a.op_dir))
    if not drift:
        return
    pr["wrapper_drift_rejected"] = pr.get("wrapper_drift_rejected", 0) + 1
    _restore(a.op_dir, _op_file(st, a.op_dir), _best_hash(st))
    save(a.op_dir, st)
    append_log(
        a.op_dir,
        st["op"],
        f"- wrapper-drift[{a.parent}] cand {a.code_hash[:8]}: "
        f"measurement discarded, working file rolled back",
    )
    out(
        {
            "improved": False,
            "refused": True,
            "error": "wrapper_drift",
            "reason": drift,
            "wrapper_frozen": False,
            "hint": "PANKO optimizes inside the @jit kernel. This measurement "
            "is not comparable with the campaign's and was discarded.",
        }
    )
    raise Refusal(3)


def _apply_ratchet(a, st, j, m_eval, op_file):
    """Gate keep + stagnation on the RUNNING GLOBAL BEST, not a fresh per-action
    optimum.

    A candidate is only "kept" (and n reset) if it beats the running best;
    otherwise it is a regression -> revert + n++. This is the existing Stage-7
    rule -- keep anything that gains, roll back anything that loses -- and it
    stops an action from burning budget below the global best.

    Both branches touch the snapshot chain, and a failure in either one is
    invisible in the response the optimizer reads. `select` and `stop` refuse on a
    broken chain, but a refinement submits candidate after candidate for the SAME
    action without passing through either, so a failure here would go unnoticed
    until the next select -- with every candidate in between built on the wrong
    base.

    Returns (refine record, improved, bar, chain_error).
    """
    pr = st["progress"]
    r = _refine_for(pr, a.parent)
    # A real measurement, so the action is still producing new programs. The
    # duplicate run that would otherwise release it from stagnation_K starts over.
    # Only an UNBROKEN run counts, which is what makes it evidence of an exhausted
    # axis rather than of an occasional repeat.
    r["dup_streak"] = 0
    bar = (
        r["best"]["J"] if r["best"] is not None else r.get("threshold_J", pr["best_J"])
    )
    improved = bool(a.s) and j > bar
    chain_error = None
    if improved:
        r["best"] = {"code_hash": a.code_hash, "J": j, "p": a.p, "m": m_eval}
        r["n"] = 0
        if not _snapshot(
            a.op_dir, op_file, a.code_hash
        ):  # persist the new running best
            # The bytes are still on disk, so nothing is lost yet -- but the next
            # reverted candidate could not be rolled back to them, which is the
            # promise that "the best code can never be lost" rests on.
            chain_error = "snapshot_failed"
        # The kept program is now the one later deltas are judged against, so the
        # space it defines becomes the reference. Without this the verdict would
        # keep firing against the preopt's space long after it was superseded.
        st["domain_fingerprint"] = _domain_state(st, op_file) or st.get(
            "domain_fingerprint"
        )
    else:
        r["n"] += 1
        # roll the working file back to the action's current best (or the global best if this
        # action has produced nothing yet), so the next candidate is applied on the right base.
        base_hash = r["best"]["code_hash"] if r["best"] else _best_hash(st)
        if (
            not _restore(a.op_dir, op_file, base_hash)
            and _file_hash(op_file) != base_hash
        ):
            # The loser is still in the working file. Left unsaid, the next
            # candidate for this action is applied on top of a program the ratchet
            # has already rejected, and its measured gain is against a base
            # nothing recorded.
            chain_error = "restore_failed"
    return r, improved, bar, chain_error


def _halt_on_broken_chain(a, st, payload, chain_error, op_file):
    """The snapshot chain is broken. Emit the measurement and stop the run.

    Called AFTER `save`: the measurement was paid for and belongs in the state and
    the cache whatever happens to the files. The non-zero exit is what stops the
    run, because the next candidate would be built on the wrong program.
    """
    payload.update(
        {
            "refused": True,
            "error": chain_error,
            "code_hash": a.code_hash,
            "op_file": op_file,
            "hint": "optimization/nodes/ is the search's memory of "
            "every program it may return to; the measurement "
            "is recorded, but the next candidate cannot be "
            "built until this is repaired.",
        }
    )
    append_log(
        a.op_dir,
        st["op"],
        f"- refine[{a.parent}] cand {a.code_hash[:8]}: {chain_error} "
        f"-- the snapshot chain is broken, run halted",
    )
    out(payload)
    raise Refusal(3)


def cmd_record(a):
    """One candidate evaluated inside local refinement of action `parent`."""
    st = load(a.op_dir)
    pr = st["progress"]
    j = compute_j(a.s, a.p, st["target"]["P_ref"])
    cache = st.setdefault(
        "eval_cache", {}
    )  # memoize E(x) by code_hash (write-once = deterministic)

    if _record_duplicate(a, st, pr, cache):
        return
    if _record_semantic_noop(a, st, pr, j):
        return
    _record_wrapper_drift(a, st, pr)

    pr["evals_used"] += 1
    # A block winner is promoted through this path deliberately (so the ratchet
    # runs in exactly one place), but the block measured it and the CLI call
    # that promotes it carries no reading. Recover it from the block record
    # rather than writing a zero over a real measurement.
    m_eval = _metrics(
        getattr(a, "util", None),
        getattr(a, "bubble", None),
        _block_metrics(st, a.code_hash),
        _cli_pipes(a),
    )
    cache[a.code_hash] = {"s": a.s, "p": a.p, "m": m_eval}
    # Only now. Above this line the candidate may have been a byte-identical
    # resubmission, which is not a measurement of anything: telling the study a
    # value for a program it did not propose would poison the surrogate with a
    # point the search never actually visited.
    _tile_bayes_tell(st, a.op_dir, a.parent, a.s, a.p)
    op_file = _op_file(st, a.op_dir)
    r, improved, bar, chain_error = _apply_ratchet(a, st, j, m_eval, op_file)
    pr["refine"] = r
    stagnation_k = st["config"]["stagnation_K"]
    stop_refine = (r["n"] >= stagnation_k) or (
        r["best"] is not None and r["best"]["J"] >= 100
    )
    # One ask per measurement, not one per action. Asked only at select, the
    # lever proposes once and the remaining refinements fall back to the coder
    # guessing at tile numbers -- so `stagnation_K` would count the model's
    # guesses rather than BO trials, and six evaluations in seven would be
    # spent on exactly the thing the lever exists to replace.
    proposal = None if stop_refine else _bayes_next(st, a.op_dir, a.parent)
    save(a.op_dir, st)
    verdict = "kept" if improved else ("tied" if (a.s and j == bar) else "reverted")
    append_log(
        a.op_dir,
        st["op"],
        f"- refine[{a.parent}] cand {a.code_hash[:8]}: s={a.s} p={a.p} j={j} "
        f"-> {verdict} (n={r['n']})",
    )
    # The lever's next value rides back with the verdict, so the optimizer
    # evaluates it without another select and without the coder.
    record_payload = {
        "J": j,
        "improved": bool(improved),
        "n": r["n"],
        "n_static": r.get("n_static", 0),
        "stop_refine": bool(stop_refine),
    }
    if proposal:
        record_payload["bayesian_optimization"] = proposal
    if chain_error:
        _halt_on_broken_chain(a, st, record_payload, chain_error, op_file)
    out(record_payload)


def _refine_and_bound(st, pr, parent, field):
    """Bump one static-rejection counter on the parent's refine record.

    Returns the record (or None, when the candidate has no parent), the new
    value of `field`, and the bound that value is measured against. The three
    static rejections below each count into their own field, so that a run of
    one kind is never mistaken for a run of another.
    """
    r = _refine_for(pr, parent) if parent else None
    if r is not None:
        r[field] = r.get(field, 0) + 1
    bound = st["config"].get("infeasible_K", DEFAULT_CONFIG["infeasible_K"])
    return r, (r.get(field, 0) if r else 0), bound


def _rollback(a, st, r):
    """Roll the working file back to the incumbent, as a measured regression would."""
    _restore(
        a.op_dir,
        _op_file(st, a.op_dir),
        (r["best"]["code_hash"] if r and r.get("best") else _best_hash(st)),
    )
    save(a.op_dir, st)


def _reject_semantic_noop(a, st, pr, op_f, retune):
    """The cheaper and more decisive question, asked before the hardware one: is
    this the program it is being judged against?

    The eval cache answers that for identical BYTES; this answers it for
    identical SEMANTICS. A run can need it: a candidate that merely restates a
    parameter's own default -- `submit_before_loop=False`, say -- wins on a low
    draw of the incumbent's own latency distribution, and the delivered kernel
    changes for no reason.
    Rejecting here rather than at `record` is what makes it free: the device is
    never asked.

    Returns True when it has emitted a rejection and the caller must stop.
    """
    same, why = _semantic_noop(st, a.op_dir, op_f, a.parent)
    if not same:
        return False
    pr["semantic_noop_rejected"] = pr.get("semantic_noop_rejected", 0) + 1
    # Its OWN counter, not `n_static`. A semantic no-op is not a hardware
    # infeasibility, and sharing the bound made the two indistinguishable in
    # the state and let a run of them close an action on `infeasible_K` --
    # recorded as no successful candidate at n=1 of stagnation_K=7, and then
    # retired on that. The separate bound releases `close` (see the `n_noop`
    # escape in `_refuse_early_close`) for an action that can only resubmit the
    # program it already has. `_refine_for` creates the record here rather than
    # waiting for `record`, so the bound also covers an action whose FIRST
    # candidate is a no-op -- previously that action had no `refine` to count
    # into and was left to the `dup_streak` backstop, which needs a measurement
    # to run at all.
    r, n_noop, infeasible_k = _refine_and_bound(st, pr, a.parent, "n_noop")
    _rollback(a, st, r)
    append_log(
        a.op_dir,
        st["op"],
        f"- semantic no-op[{a.parent}]: {why} "
        f"(no device time; n and n_static unchanged, n_noop={n_noop}/{infeasible_k})",
    )
    out(
        {
            "feasible": False,
            "semantic_noop": True,
            "reason": f"this candidate is the incumbent program: {why}",
            # The hardware gate did NOT run -- this was rejected before it. On the
            # `static_feasibility: false` ablation arm that gate is off entirely,
            # and reporting `checked: true` there would describe a check the arm
            # does not perform.
            "checked": False,
            **retune,
            # `.get`, because a `refine` dict is not guaranteed to carry `n`: the
            # duplicate branch of `record` creates one holding nothing but
            # `dup_streak` when it fires before the action's first measurement.
            "n": (r.get("n", 0) if r else 0),
            "n_static": (r.get("n_static", 0) if r else 0),
            "n_noop": n_noop,
            "stop_refine": n_noop >= infeasible_k,
            "stop_cause": ("semantic_noop_K" if n_noop >= infeasible_k else None),
            "hint": "the delta has to change what the program DOES. Passing a "
            "keyword its own default value is the same call as omitting "
            "it, so there is nothing here for the device to measure.",
        }
    )
    return True


def _reject_wrapper_drift(a, st, pr, op_f, retune):
    """Before the capacity check and before any device time: a candidate that
    moved work into the wrapper is not a candidate, whatever its tiles do.

    Returns True when it has emitted a rejection and the caller must stop.
    """
    drift = _wrapper_drift(st, op_f)
    if not drift:
        return False
    pr["wrapper_drift_rejected"] = pr.get("wrapper_drift_rejected", 0) + 1
    r, n_static, infeasible_k = _refine_and_bound(st, pr, a.parent, "n_static")
    _rollback(a, st, r)
    append_log(
        a.op_dir,
        st["op"],
        f"- wrapper-drift[{a.parent}]: rejected before the device "
        f"(n_static={n_static}/{infeasible_k})",
    )
    out(
        {
            "feasible": False,
            "reason": drift,
            "checked": True,
            "wrapper_frozen": False,
            **retune,
            "n": (r["n"] if r else 0),
            "n_static": n_static,
            "stop_refine": n_static >= infeasible_k,
            "stop_cause": ("infeasible_K" if n_static >= infeasible_k else None),
            "hint": "PANKO optimizes inside the @jit kernel. Edit the kernel, "
            "not the wrapper.",
        }
    )
    return True


def _reject_infeasible(a, st, pr, why, retune):
    """The capacity gate said no. Count it against `infeasible_K`, not stagnation.

    `n` is untouched on purpose: see `cmd_feasible`'s docstring. Only the
    separate bound can end a refinement here, and when it does it is a different
    event from stagnation, so it is logged as one. The refine record is created
    here when this is the action's first event, so an action whose opening
    neighbourhood is entirely infeasible can still reach the bound -- it used to
    need a measurement first, and an all-infeasible action never gets one.
    """
    pr["static_rejected"] = pr.get("static_rejected", 0) + 1
    r, n_static, infeasible_k = _refine_and_bound(st, pr, a.parent, "n_static")
    stop_refine = n_static >= infeasible_k
    _rollback(a, st, r)
    append_log(
        a.op_dir,
        st["op"],
        f"- infeasible[{a.parent}]: {why} (no device time; "
        f"n={r['n'] if r else 0} unchanged, n_static={n_static}/{infeasible_k})"
        + ("  <- closing on infeasible_K, not stagnation" if stop_refine else ""),
    )
    out(
        {
            "feasible": False,
            "reason": why,
            "checked": True,
            **retune,
            "n": (r["n"] if r else 0),
            "n_static": n_static,
            "stop_refine": stop_refine,
            "stop_cause": ("infeasible_K" if stop_refine else None),
            "static_rejected": pr["static_rejected"],
        }
    )


def cmd_feasible(a):
    """Static hardware feasibility, checked before the candidate reaches the NPU.

    A rejection here costs no device time, so it does NOT consume eval_budget,
    and it does NOT advance the action's stagnation counter either.

    It used to advance it, and that was wrong twice over. `n` is a count of
    *measurements that failed to improve*, which is evidence about the action.
    A static rejection is evidence about one candidate: the gate never ran the
    program, so nothing was learned about whether the action helps. Worse, the
    two are not independent -- the gate fires hardest exactly where an action is
    pushing a tile or a view against the buffer, which is where the interesting
    candidates are.

    It already cost us an action. One recorded run had an action (F-1, increase
    task granularity) accumulate seven static rejections against the buffer
    budget and close on the stagnation limit having never reached the device:

        - infeasible[u1]: UB estimate over budget (n=7)
        - close[u1] -> (no candidate beat the best) | stagnation 1

    Rejections are counted separately, in `n_static`, against `infeasible_K`.
    That bound exists only to guarantee termination: an action whose whole
    neighbourhood is infeasible would otherwise loop forever, spending no budget
    and so never hitting eval_budget either. It is set far above anything the
    recorded runs ever accumulated in one action (five), because the cost of a
    rejection is one dispatch rather than one device run.

    Rejecting is deliberately conservative -- see feasibility.py. A false
    rejection deletes a real optimisation from the search space permanently,
    while a false acceptance costs one evaluation.
    """
    st = load(a.op_dir)
    pr = st["progress"]
    op_f = a.op_file or _op_file(st, a.op_dir)
    retune = _retune_verdict(st, op_f)

    if _reject_semantic_noop(a, st, pr, op_f, retune):
        return

    if not st["config"].get("static_feasibility", True) or feasibility is None:
        out({"feasible": True, "reason": "", "checked": False, **retune})
        return
    # Same envelope the tile search validates against. `--ub-kb` / `--l1-kb`
    # still override, for a caller measuring a chip the state does not name.
    env = chip_envelope(st)
    hw = feasibility.HW(
        ub_budget_kb=(a.ub_kb or env["ub_kb"]),
        l1_budget_kb=(a.l1_kb or env["l1_kb"]),
        m_l1=a.m_l1,
        k_l1=a.k_l1,
    )
    if _reject_wrapper_drift(a, st, pr, op_f, retune):
        return
    ok, why = feasibility.check(op_f, hw)
    if ok:
        out({"feasible": True, "reason": "", "checked": True, **retune})
        return
    _reject_infeasible(a, st, pr, why, retune)


def cmd_probe(a):
    """Cache lookup by code_hash. The optimizer calls this BEFORE running the (expensive) frozen
    E(x): a hit means these exact bytes were already evaluated, so it reuses (s,p) and skips
    compile+measure. Kills redundant re-evals of a coder no-op (unchanged bytes) or a re-proposed best.
    """
    st = load(a.op_dir)
    hit = st.get("eval_cache", {}).get(a.code_hash)
    out(
        {
            "hit": hit is not None,
            "s": (hit or {}).get("s"),
            "p": (hit or {}).get("p"),
            "m": (hit or {}).get("m", {}),
        }
    )


def stored_refusals(st):
    """Stored refusals, as rows that carry their own scope.

    Rows written before this field existed have no `structures` and are treated
    as evidence about no structure at all, so they match nothing and are
    re-earned. That is the safe direction: the cost is one repeated device trial,
    where trusting them is a tile deleted from the space of a program they were
    never observed on.
    """
    rows = []
    for r in st.get("bayesian_optimization_refusals") or []:
        if not r.get("params") or not r.get("sig"):
            continue
        rows.append(
            {
                "params": r["params"],
                "sig": r["sig"],
                "structures": list(r.get("structures") or []),
                "scope": r.get("scope") or "structure",
            }
        )
    return rows


def refusals_for_structure(rows, structure):
    """The subset a block on `structure` may act on.

    `error_signature` already refuses to sign a capacity failure or a golden
    mismatch, so what reaches here is a front-end pass rejecting the tile call.
    That was carried to every structure on the reasoning that it is a fact about
    the tile -- but the signature keeps only ErrCode + Enum + PassName, and
    `F4FFFF` is the generic FeError code that this repo's own experience table
    maps to several causes. `error_signature`'s docstring says as much: it
    establishes that the signature is stable across occurrences of the same
    refusal, NOT that two refusals sharing one share a cause. So a refusal
    earned where a tile met one structure could delete that tile from the space
    of a program that no longer has it, permanently and without a measurement.

    Structure-scoped by default therefore, and promoted only on evidence: a
    refusal seen on REFUSAL_GLOBAL_AFTER distinct structures has been shown to
    survive a restructure rather than assumed to.
    """
    return [
        {"params": r["params"], "sig": r["sig"]}
        for r in rows
        if r["scope"] == "global" or structure in r["structures"]
    ]


def merge_refusals(rows, observed, structure, promote_after=REFUSAL_GLOBAL_AFTER):
    """Fold this block's refusals into the stored rows, stamped with `structure`."""
    index = {(json.dumps(r["params"], sort_keys=True), r["sig"]): r for r in rows}
    for obs in observed or []:
        if not obs.get("params") or not obs.get("sig"):
            continue
        key = (json.dumps(obs["params"], sort_keys=True), obs["sig"])
        row = index.get(key)
        if row is None:
            row = {
                "params": obs["params"],
                "sig": obs["sig"],
                "structures": [],
                "scope": "structure",
            }
            index[key] = row
            rows.append(row)
        if structure and structure not in row["structures"]:
            row["structures"].append(structure)
        if len(row["structures"]) >= promote_after:
            row["scope"] = "global"
    return rows


def wrapper_signature(src):
    """Hash of everything in the file that is NOT a `@...jit` kernel.

    The host wrapper is FROZEN for the whole run, and this is what enforces it.

    `p` counts AICore time inside the jit kernel and nothing else. That is a
    sound objective exactly while the uncounted part is identical between the
    baseline and the candidate, because then it cancels in the comparison. An
    action that moves work across the boundary breaks the condition: the kernel
    gets cheaper, the wrapper gets more expensive, `p` improves, and the ratchet
    keeps a candidate that is slower to run. The upstream review put it as
    kernel 100us -> 80us with 30us added to the wrapper, scored as a 20us win.

    Two of the catalogue's strongest recorded results were that shape -- F-21's
    fusion and F-22's layout copy. Neither made the work cheaper; both moved it
    out of the region `p` measures. Removing those actions is what makes the objective
    sound, and checking the signature is what makes the removal enforceable:
    the coder is handed free-form intents, and a rule that lives only in a
    document is a rule the search cannot rely on.

    Hashed over the AST rather than the bytes, so a comment, a blank line or a
    reformat outside the kernel is invisible and only a change to what the host
    actually DOES trips it. An added module-level import trips it too: that is
    deliberate erring, since the alternative is deciding which statements are
    "work" and a module constant feeding the wrapper's padding is work.

    Top-level jit functions are what this excludes. A kernel nested inside a
    class or a factory would be hashed as wrapper, which would make every tile
    rewrite look like drift -- no kernel in this repository is written that way,
    and the check says so rather than guessing.

    Returns None when the file does not parse -- callers treat that as "cannot
    tell" and say so, rather than passing a candidate the check never ran on.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None

    def _is_kernel(n):
        return isinstance(
            n, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and _decorated_jit(n)

    shell = [n for n in tree.body if not _is_kernel(n)]
    dumped = ast.dump(ast.Module(body=shell, type_ignores=[]))
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def _decorated_jit(node):
    for d in getattr(node, "decorator_list", []):
        f = d.func if isinstance(d, ast.Call) else d
        while isinstance(f, ast.Attribute):
            if f.attr == "jit":
                return True
            f = f.value
        if isinstance(f, ast.Name) and f.id == "jit":
            return True
    return False


def _wrapper_drift(st, path):
    """ "" if the wrapper still matches the baseline, else why it does not."""
    want = st.get("wrapper_signature")
    if not want:
        return ""  # pre-dates the check; nothing to compare
    try:
        got = wrapper_signature(_read_text(path))
    except OSError:
        return ""
    if got is None:
        return (
            "the candidate does not parse, so the wrapper freeze could not be checked"
        )
    if got != want:
        return (
            "the host wrapper differs from the baseline. PANKO optimizes the "
            "time inside the @jit kernel, and that number is only comparable "
            "while everything outside it is unchanged -- work moved into the "
            "wrapper leaves the measured region and scores as a win without "
            "being one. Restore the wrapper and put the change inside the "
            "kernel, or run this as a different experiment"
        )
    return ""


def _resolve_envelope(a):
    """(envelope, reason). None means it could not be derived and init must ask.

    Explicit beats derived: a caller who names the numbers is measuring a chip
    deliberately, and that is recorded as such so the report cannot be read as
    "these were read from the installed runtime" when they were not.
    """
    given = getattr(a, "chip_envelope", "")
    if given:
        try:
            supplied = json.loads(given)
        except ValueError as e:
            return None, f"--chip-envelope is not valid JSON: {e}"
        missing = [k for k in CHIP_ENVELOPE if k != "source" and k not in supplied]
        if missing:
            return None, ("--chip-envelope is missing " + ", ".join(sorted(missing)))
        env = {k: supplied[k] for k in CHIP_ENVELOPE if k != "source"}
        env["source"] = "explicit:cli"
        return env, ""
    if chip_profile is None:
        return None, "chip_profile.py is not importable next to the harness"
    env, why = chip_profile.resolve(device=getattr(a, "device", ""))
    if env is None:
        return None, why
    out_env = {k: env[k] for k in CHIP_ENVELOPE if k != "source" and k in env}
    out_env["source"] = env.get("source", "")
    out_env["soc"] = env.get("soc", "")
    out_env["soc_via"] = env.get("soc_via", "")
    return out_env, ""


def _refine_for(pr, action):
    """The refinement record for `action`, created if this is its first event.

    Every counter that bounds a refinement has to survive the first event that
    moves it, and two of those events happen BEFORE any measurement: a static
    infeasibility and a semantic no-op, both decided by `feasible` without a
    device. The record used to be created only by `record`, and the pre-device
    branches incremented `n_static` / `n_noop` only `if r and r["action"] ==
    parent` -- so an action whose whole opening neighbourhood was rejected had
    nowhere to put them. Both read 0 forever and `infeasible_K` could never
    fire, which is the opposite of what SKILL.md promises: the optimizer keeps
    selecting an action that has never produced a measurable candidate.

    Reported from a run with `infeasible_K=2` and three certainly-out-of-bounds
    candidates: `static_rejected` reached 3 while every response still said
    `n_static=0, stop_refine=false`.

    `record` names the action it is recording for, so it always gets a record.
    `feasible` does not: its `--parent` defaults to empty, and such a call
    belongs to no refinement, so those two call sites skip this entirely rather
    than open a record keyed by "".
    """
    r = pr.get("refine")
    if not r or r.get("action") != action:
        # Fully shaped from the start. The duplicate branch used to leave a
        # headless `{"dup_streak": n}` behind, which every later reader had to
        # repair, and which a pre-device counter would now silently discard.
        r = {
            "action": action,
            "best": None,
            "threshold_J": pr["best_J"],
            "n": 0,
            "n_static": 0,
            "n_noop": 0,
            "dup_streak": 0,
        }
        pr["refine"] = r
    return r


def _refuse_early_close(st, action):
    """None if this action may be closed, else the reason it may not.

    `stagnation_K` is listed among the deterministic terms, and it was not one:
    `record` computed `stop_refine` and `close` never looked at it, so whether an
    action had been tested was the optimizer's call. Two runs of the SAME build
    read the rule in opposite ways -- one wrote "the harness requires one more
    candidate and I do not override its judgment" and closed at seven, the other
    wrote "closed by judgement, not by the counter" and closed six actions at a
    mean of 1.3. An action abandoned after one non-improving candidate has not
    been tested; it has been sampled once against a measurement whose cv is 15%.

    The single exception is an action that produced no candidate at all -- the
    coder found the delta inexpressible in this build, so there is nothing to
    refine and nothing to learn by demanding six more. Refusing that would
    deadlock the run.
    """
    pr = st["progress"]
    r = pr.get("refine") or {}
    if r.get("action") != action:
        return None
    if r.get("best") is None and not r.get("n"):
        return None  # no candidate ever recorded
    stagnation_k = st["config"].get("stagnation_K", 7)
    n = r.get("n", 0)
    if n >= stagnation_k:
        return None
    if r.get("best") and r["best"].get("J", 0) >= 100:
        return None  # success stop
    if r.get("n_static", 0) >= st["config"].get("infeasible_K", 16):
        return None  # closing on infeasible_K
    if r.get("n_noop", 0) >= st["config"].get("infeasible_K", 16):
        # `feasible` reports `stop_refine: true, stop_cause: semantic_noop_K`
        # when an action can only resubmit the program it already has. Without a
        # matching escape here the harness told the optimizer to close and then
        # refused the close, and the action was stuck: `n` cannot advance
        # (nothing was measured), `dup_streak` cannot advance (the bytes differ),
        # and `close` is blocked.
        return None
    if r.get("dup_streak", 0) >= 3:
        # See the duplicate branch of `record`. An action that can only
        # re-propose bytes already in the cache has no unmeasured value left,
        # and that is an observation rather than a claim.
        return None
    return (
        f"n={n} of stagnation_K={stagnation_k}: this action has not been tested. "
        f"Propose the next concrete candidate for it. An axis you believe "
        f"is exhausted still has to be shown exhausted -- submit the "
        f"remaining values and let the duplicate check reject them for free."
    )


def _refuse_close(a, st, pr):
    """The toll on the entrance to `close`: an action may not be closed before it
    has been refined. Emits and raises when it refuses."""
    why = _refuse_early_close(st, a.action)
    if not why:
        return
    pr["early_close_refused"] = pr.get("early_close_refused", 0) + 1
    save(a.op_dir, st)
    out(
        {
            "closed": None,
            "refused": True,
            "reason": why,
            "n": (pr.get("refine") or {}).get("n", 0),
            "stagnation_K": st["config"].get("stagnation_K", 7),
        }
    )
    raise Refusal(3)


def _retire_spent_action(a, st, r):
    """Retire an action that paid `stagnation_K` measurements and produced nothing.

    THIS is what `tried` is for. The action was selected, refined and measured,
    and it produced nothing -- so `novelty = 1/(1+tried)` should now push it down
    the frontier. It used to stay here at tried=0 and be re-selected at full
    priority.

    Halving the priority is not enough. Several actions can each run the full
    `stagnation_K` measurements, every one closed "no successful candidate", and
    all of them stay in `open` at tried=1. The selector then re-picks a spent
    lever with a handful of evaluations left, leaving the last untried lever
    class never tested at all, and the optimizer has to prune them BY HAND.

    So retire it, on exactly the terms `evolve`'s conditional prune uses:
    `pruned_on` scopes the retirement to the structure now in the file, and
    `_live` revives the action automatically once a later delta changes that
    structure. "No successful candidate after stagnation_K measurements" is the
    strongest reason to retire an action against a program, not a reason to leave
    it selectable at half price. It is NOT permanent, because the evidence is
    about geometry that a restructure invalidates.

    `_refuse_early_close` still guards the entrance, so this cannot be used to
    skip the toll, only to record that the toll was paid for nothing. It lets
    several OTHER cases through, and none of them has paid for a retirement: an
    action with no refinement at all (a close aimed at an id that was never
    selected), a `refine` belonging to a different action, the documented escape
    hatch for a delta the coder found inexpressible at n=0, and the
    `infeasible_K` / `semantic_noop_K` / `dup_streak` releases. Retiring on any
    of those would be exactly the free prune this change exists to make
    unnecessary -- and it would be recorded as "no successful candidate", i.e. as
    `stagnation_K` measurements that never happened.

    Returns (retired, why_not).
    """
    stagnation_k = st["config"].get("stagnation_K", DEFAULT_CONFIG["stagnation_K"])
    paid = bool(r and r.get("action") == a.action and r.get("n", 0) >= stagnation_k)
    # AFTER the roll-back, not before. `_structure` reads the file on disk, and
    # the working file at this point may still hold the last rejected candidate
    # -- which is a program the run has just discarded. Scoping the retirement to
    # that structure retires the action against something no longer on disk, so
    # `_live` shows it again on the very next `select` while the payload has
    # already told the optimizer not to prune it.
    _restore(a.op_dir, _op_file(st, a.op_dir), _best_hash(st))
    sig = _structure(st, a.op_dir) if paid else None
    retired = False
    for n in st["nodes"]["open"]:
        if n["id"] == a.action:
            n["tried"] = n.get("tried", 0) + 1
            # No signature means no scope. `_live` skips a falsy `pruned_on`, so
            # writing None would report a retirement that hides nothing. Falsy
            # here means `bayesian.lever` is absent or the file could not be read
            # -- a file that does not PARSE yields the string "unparseable",
            # which is a real scope that stops matching as soon as the file
            # parses again. Leave the rest to `tried`.
            if sig:
                n["pruned_on"] = sig
                retired = True
    # Retiring the last live action empties the frontier, and until now only
    # `evolve` could do that -- so the counter that exists to catch a run halting
    # far under budget on an empty frontier could not see this route.
    if retired and not _live(st, a.op_dir):
        st["progress"]["frontier_emptied_by_prune"] = (
            st["progress"].get("frontier_emptied_by_prune", 0) + 1
        )
    # Three distinct reasons, and they were being reported as two. An action that
    # paid in full but is no longer in `open` got "no structural signature" when
    # the signature was fine, and an action whose `refine` belongs to somebody
    # else got that other action's `n` quoted back.
    if not paid:
        n_paid = r.get("n", 0) if (r and r.get("action") == a.action) else 0
        why_not = f"n={n_paid} of stagnation_K={stagnation_k}, so nothing was disproved"
    elif not sig:
        why_not = "no structural signature to scope the retirement to"
    else:
        why_not = f"{a.action} is not on the open frontier"
    return retired, why_not


def _close_without_candidate(a, st, pr, r):
    """Nothing beat the running best: count a stagnation, retire the action if it
    paid for one, and leave the working file AT the global best (records already
    rolled back; `_retire_spent_action` restores again defensively)."""
    pr["global_stagnation"] += 1
    pr["stage_stagnation"] += 1
    retired, why_not = _retire_spent_action(a, st, r)
    pr["refine"] = None
    save(a.op_dir, st)
    append_log(
        a.op_dir,
        st["op"],
        f"- close[{a.action}] -> (no candidate beat the best) "
        f"| stagnation {_gstag(st)}"
        + (
            " | retired against the current structure (revives if it changes)"
            if retired
            else f" | NOT retired: {why_not}"
        ),
    )
    out(
        {
            "closed": None,
            "note": "no successful candidate for this action",
            "retired": retired,
            "hint": (
                "this action is retired against the structure now in the file and "
                "`select` will not offer it again until a delta changes that "
                "structure -- do NOT prune it by hand"
                if retired
                else "the action stays selectable at reduced priority: retirement "
                "requires stagnation_K measurements AND a readable structure"
            ),
        }
    )


def _promote_new_best(a, st, pr, b, cid):
    """This action's best beat the global best. Re-point the frontier at it and
    re-analyse. Returns the symptoms that were re-boosted."""
    prev_p = pr["best_latency_us"]
    pr["best_J"] = b["J"]
    st["best_node"] = cid
    pr["best_latency_us"] = b["p"]
    pr["best_speedup_vs_preopt"] = (
        round(pr["preopt_us"] / b["p"], 4) if b["p"] else None
    )
    pr["global_stagnation"] = 0
    for n in st["nodes"][
        "open"
    ]:  # greedy accumulate: stack subsequent actions on the new best
        n["parent"] = cid
    # re-analysis: the bottleneck shifts as we optimize -> re-point open beliefs (V) at the NEW
    # best kernel's current symptom, reusing the util/bubble we just measured (no extra eval).
    # `.get(k)` -> None when the key is absent, which is the point: an
    # unmeasured node must not be read as an idle one.
    reboosted = _symptom_reboost(
        st,
        (b.get("m") or {}).get("util"),
        (b.get("m") or {}).get("bubble"),
        _op_file(st, a.op_dir),
        (b.get("m") or {}),
    )
    admitted = _admit(st, a.op_dir)  # the new best may have new structure
    if admitted:
        append_log(
            a.op_dir,
            st["op"],
            f"- admit: preconditions now met, {len(admitted)} action(s) "
            f"returned to the frontier: {admitted}",
        )
    gain = (prev_p - b["p"]) / prev_p if prev_p else 0.0
    if gain >= st["config"].get("stage_min_gain", 0.01):
        pr["stage_stagnation"] = 0
        pr["cycle_productive"] = True  # productive (>=1%): keep mining this stage
    else:
        pr["stage_stagnation"] += 1  # marginal (<1%): counts toward advancing
    return reboosted


def _advance_stage(st, pr):
    """Stage progression: advance CYCLICALLY (frontend->swimlane->incore->frontend...)
    when the current stage yields no productive (>=1%) gain; a full cycle with no
    >=1% gain => converged."""
    if pr["stage_stagnation"] < st["config"].get("stage_patience", 2):
        return
    i = (STAGE_ORDER.index(pr["active_stage"]) + 1) % len(STAGE_ORDER)
    if i == 0:  # wrapped incore -> frontend = a full cycle completed
        if not pr.get("cycle_productive", False):
            # Recorded either way; only halts when stage_convergence is on.
            pr["cycle_unproductive"] = pr.get("cycle_unproductive", 0) + 1
            if st["config"].get("stage_convergence", True):
                pr["converged"] = True
        pr["cycle_productive"] = False
    pr["active_stage"] = STAGE_ORDER[i]
    pr["stage_stagnation"] = 0


def cmd_close(a):
    """Attach the refinement's best candidate as a CLOSED node; update global best."""
    st = load(a.op_dir)
    pr = st["progress"]
    _refuse_close(a, st, pr)
    r = pr.get("refine")
    if not r or r.get("action") != a.action or r["best"] is None:
        _close_without_candidate(a, st, pr, r)
        return
    b = r["best"]
    act = next((n for n in st["nodes"]["open"] if n["id"] == a.action), None)
    stage = act["stage"] if act else pr["active_stage"]
    reboosted = []
    cid = _next_id(st, "x")
    st["nodes"]["closed"].append(
        {
            "id": cid,
            "parent": act["parent"] if act else "root",
            "stage": stage,
            "code_hash": b["code_hash"],
            "J": b["J"],
            "eval": {"s": 1, "p": b["p"], "m": b["m"]},
        }
    )
    remaining = []
    for n in st["nodes"]["open"]:
        if n["id"] != a.action:
            remaining.append(n)
    st["nodes"]["open"] = remaining  # consumed
    if b["J"] > pr["best_J"]:
        reboosted = _promote_new_best(a, st, pr, b, cid)
    else:
        pr["global_stagnation"] += 1
        pr["stage_stagnation"] += 1
    _advance_stage(st, pr)
    pr["refine"] = None
    save(a.op_dir, st)
    # leave the working file AT this action's best (= the new running best when it improved).
    _restore(a.op_dir, _op_file(st, a.op_dir), b["code_hash"])
    append_log(
        a.op_dir,
        st["op"],
        f"- close[{a.action}] -> {cid}: best j={b['j']} p={b['p']} "
        f"(global best_J={pr['best_J']}) "
        f"| stagnation {_gstag(st)}"
        f" | symptoms {reboosted or '-'}",
    )
    out(
        {
            "closed": cid,
            "best_J": pr["best_J"],
            "best_latency_us": pr["best_latency_us"],
            "reboosted_symptoms": reboosted,
        }
    )


def _apply_inserts(st, op_nodes, edits):
    for ins in edits.get("insert", []):
        op_nodes.append(
            {
                "id": _next_id(st, "u"),
                "parent": ins["parent"],
                "stage": ins["stage"],
                "delta": ins["delta"],
                "V": clamp_belief(ins.get("V", 0.5)),
                "prior_gain": float(ins.get("prior_gain", 0.5)),
                "tried": 0,
            }
        )


def _apply_updates(op_nodes, edits):
    """V only. Revising a belief is not an attempt, and counting it as one
    inverted the whole mechanism: in a recorded run the two actions the optimizer
    had repeatedly identified as its most promising carried tried=3 without ever
    having been evaluated, and were outranked by an inapplicable action at
    tried=0. `tried` is incremented where an action is actually attempted -- at
    `close` -- and nowhere else.
    """
    upd = {u["id"]: u for u in edits.get("update", [])}
    for n in op_nodes:
        if n["id"] in upd:
            n["V"] = clamp_belief(upd[n["id"]]["V"])
    return upd


def _apply_prunes(st, a, op_nodes, edits):
    """(permanent, conditional, ignored_held, unknown, structure).

    Pruning splits by what the optimizer could actually see. A node in `open` had
    its preconditions satisfied and was retired on judgment, so the retirement is
    permanent and `_admit` must honour it. A node in `held` is already governed by
    the deterministic precondition system, and letting the optimizer delete it
    would destroy the path that `held, never deleted` exists to preserve: a later
    action can introduce the structure that makes it applicable.

    `prune` retires an action against the CURRENT structure; `prune_permanent`
    retires it outright. The split exists because two prunes in the same run can
    be different in kind and the harness could not tell them apart: one action
    is disproved by a sweep over the program as it stands, another is
    inexpressible without changing the operator's frozen layout. Only the second
    is a fact about the operator. See `_live`.
    """
    prune = set(edits.get("prune", []))
    forever = set(edits.get("prune_permanent", []))
    open_ids = {n["id"] for n in op_nodes}
    held_ids = {n["id"] for n in st["nodes"].get("held", [])}
    permanent = forever & open_ids
    conditional = (prune & open_ids) - permanent
    ignored_held = (prune | forever) & held_ids
    unknown = (prune | forever) - open_ids - held_ids
    sig = _structure(st, a.op_dir)
    if conditional and sig is None:
        # No signature means no scope, and `_live` skips a node whose `pruned_on`
        # is falsy -- so writing None would report every conditional prune as
        # applied while none of them hid anything. That is the default state on an
        # arm built without `bayesian.lever`, and it fires transiently whenever the
        # file on disk does not parse. Retire them outright and say so, rather
        # than silently discarding what the optimizer asked for.
        permanent |= conditional
        conditional = set()
    for n in op_nodes:
        if n["id"] in conditional:
            # Kept in `open` rather than deleted: the node has to survive for its
            # evidence to expire. `_live` hides it while the structure matches.
            n["pruned_on"] = sig
    st["nodes"]["open"] = [n for n in op_nodes if n["id"] not in permanent]
    st["nodes"]["pruned"] = sorted(set(st["nodes"].get("pruned", [])) | permanent)
    return permanent, conditional, ignored_held, unknown, sig


def _record_frontier(pr, edits, live, pruned_any):
    """Emptying the frontier is termination by another name, and termination is
    not the model's decision. Extending the tree IS its job: `insert` is the world
    model. So an evolve that leaves nothing to select must either offer new
    intents or state, explicitly, that it has none. It has been observed: a run
    whose last evolves inserted nothing and pruned steadily, the last of them
    reasoned as "pruning exhausted actions to focus remaining budget on novel
    directions" while offering no novel direction. The search halted with most
    of its budget unspent.
    """
    if live:
        pr["expansion_required"] = False
        pr["expansion_requests"] = 0
        pr.pop("expansion_declined", None)
        return
    if pruned_any and not edits.get("insert"):
        pr["frontier_emptied_by_prune"] = pr.get("frontier_emptied_by_prune", 0) + 1
    if edits.get("exhausted"):
        pr["expansion_declined"] = str(edits["exhausted"])
        return
    # Counted, never acted on. An automatic decline after N requests would itself
    # be a heuristic termination, which is the thing this branch exists to
    # remove: a run merely slow to expand would be cut short before B.
    # frontier_exhausted therefore fires only on an explicit declaration. Note
    # the backstop is wall_clock, not eval_budget: an empty frontier evaluates
    # nothing, so evals_used never advances. `expansion_requests` in the state
    # names the cause after the fact.
    pr["expansion_requests"] = pr.get("expansion_requests", 0) + 1
    pr["expansion_required"] = True


def _evolve_note(pr, live, ignored_held, unknown):
    """The warnings an evolve appends to its log line, in words."""
    note = ""
    if not live:
        if pr.get("expansion_declined"):
            note += f" | FRONTIER EXHAUSTED, declared: {pr['expansion_declined']}"
        else:
            note += (
                f" | FRONTIER EMPTY and nothing inserted: expansion required "
                f"(request {pr.get('expansion_requests', 0)}). Propose new intents, "
                f'or pass exhausted:"<reason>" to declare there are none.'
            )
    if ignored_held:
        note += (
            f" | {len(ignored_held)} prune(s) ignored: still held by unmet "
            f"preconditions, not the optimizer's to delete: {sorted(ignored_held)}"
        )
    if unknown:
        note += f" | {len(unknown)} prune(s) named no known node: {sorted(unknown)}"
    return note


def cmd_evolve(a):
    """Apply the optimizer's tree edits: {insert:[...], update:[...], prune:[ids]}."""
    st = load(a.op_dir)
    edits = json.loads(a.edits)
    op_nodes = st["nodes"]["open"]
    _apply_inserts(st, op_nodes, edits)
    upd = _apply_updates(op_nodes, edits)
    # Pruning splits by what the optimizer could actually see. A node in `open`
    # had its preconditions satisfied and was retired on judgment, so the retirement
    # is permanent and _admit must honour it. A node in `held` is already governed
    # by the deterministic precondition system, and letting the optimizer delete it
    # would destroy the path that `held, never deleted` exists to preserve: a later
    # action can introduce the structure that makes it applicable.
    # `prune` retires an action against the CURRENT structure; `prune_permanent`
    # retires it outright. The split exists because two prunes in the same run
    # can be different in kind and the harness could not tell them apart: one
    # action is disproved by a sweep over the program as it stands, another is
    # inexpressible without changing the operator's frozen layout. Only the
    # second is a fact about the operator. See `_live`.
    permanent, conditional, ignored_held, unknown, sig = _apply_prunes(
        st, a, op_nodes, edits
    )

    # Emptying the frontier is termination by another name, and termination is
    # not the model's decision. Extending the tree IS its job: `insert` is the
    # world model. So an evolve that leaves nothing to select must either offer
    # new intents or state, explicitly, that it has none. It has been observed:
    # a run whose last evolves inserted nothing and pruned steadily, the last of
    # them reasoned as "pruning exhausted actions to focus remaining budget on
    # novel directions" while offering no novel direction. The search halted
    # with most of its budget unspent.
    pr = st["progress"]
    # What `select` will actually see. A conditionally-pruned node stays in
    # `open` so its evidence can expire, so `open` is no longer the frontier and
    # asking it here would report a frontier that select cannot choose from.
    live = _live(st, a.op_dir)
    _record_frontier(pr, edits, live, permanent or conditional)
    # (stage progression is handled in cmd_close: cyclic advance on <stage_min_gain diminishing returns)
    save(a.op_dir, st)
    reasons = "; ".join(edits.get("reasons", [])) or "(no reason given)"
    note = _evolve_note(pr, live, ignored_held, unknown)
    append_log(
        a.op_dir,
        st["op"],
        f"- evolve: +{len(edits.get('insert', []))} insert, "
        f"{len(upd)} update, {len(conditional)} prune "
        f"({len(permanent)} permanent) | reasons: {reasons}{note}",
    )
    out(
        {
            "open": len(live),
            "active_stage": st["progress"]["active_stage"],
            "pruned_permanently": sorted(permanent),
            # Retired against this structure only, and named so the optimizer can
            # see that its own evidence has an expiry date.
            "pruned_on_structure": sorted(conditional),
            "structure": sig,
            "prune_ignored_held": sorted(ignored_held),
            "prune_unknown": sorted(unknown),
            "pruned_total": len(st["nodes"]["pruned"]),
            "expansion_required": bool(pr.get("expansion_required")),
            "expansion_declined": pr.get("expansion_declined"),
        }
    )


def _delivered_check(st, op_dir):
    """Does the file on disk actually hold the program the run reports as best?

    A run can answer `frontier_exhausted` reporting one best latency and ship a
    slightly different kernel: the two differ by one line of `pass_options`, the
    snapshot of the winner is intact under `nodes/`, and nothing anywhere says
    the delivered artefact is not the reported one. The number is right and the
    deliverable is wrong, which is the worst combination -- it survives every
    check that reads the state file and fails only when somebody diffs the
    kernel by hand.

    So the last thing the harness says about a run includes the hash of what it
    actually left on disk, next to the hash it claims to have found.
    """
    best = _best_hash(st)
    f = _op_file(st, op_dir)
    on_disk = None
    if f and os.path.exists(f):
        on_disk = hashlib.sha256(_read_bytes(f)).hexdigest()
    return {
        "best_code_hash": best,
        "delivered_code_hash": on_disk,
        "delivered_is_best": bool(best) and on_disk == best,
    }


def _chip_mismatch_report(expected, live, how, device, env):
    """The halt payload for an envelope derived for another chip."""
    return {
        "error": "chip_mismatch",
        "reason": (
            f"the envelope was derived for {expected} but device "
            f"{device!r} reports {live} (via {how}). The buffer "
            "capacities and core counts PANKO validates every candidate "
            "against are the wrong chip's."
        ),
        "expected_soc": expected,
        "live_soc": live,
        "envelope_source": env.get("source", ""),
        "ask_user": {
            "question": (
                f"PANKO was initialised for {expected}, and the device it is "
                f"about to measure on is {live}. Continuing would filter the "
                "search space with another chip's limits, rejecting legal tiles "
                "before the device ever saw them. How do you want to proceed?"
            ),
            "options": [
                {
                    "id": "re_init",
                    "label": f"Re-initialise for {live}",
                    "action": "delete optimization/search_state.json and re-run "
                    "init on this device; any search so far is scored "
                    "against the wrong envelope and does not carry over",
                },
                {
                    "id": "other_device",
                    "label": f"Point PANKO at a {expected}",
                    "action": "re-run with --device set to a device that holds the "
                    "chip this run was initialised for",
                },
            ],
        },
        "hint": "relay the question to the user; do not choose for them",
    }


def confirm_chip(st, device, probe=None):
    """The chip the envelope describes, re-checked against the chip in the slot.

    Two gates, because they answer at different moments. `init` resolves a NAME
    and derives the envelope from that name's ini; it must stay runnable with no
    device, or PANKO cannot be initialised offline and its own suite cannot run.
    Only a device evaluation can say whether the name was the right one, so that
    is where the name is checked -- before the first one, not after eighty.

    A mismatch is a halt and not a warning. The envelope decides which tiles are
    even legal, so the damage from a wrong one is candidates deleted before the
    device sees them, leaving nothing in the record that contradicts the result.
    A warning would be printed into a log nobody reads until the numbers look
    strange.

    "unknown" proceeds. The envelope came from a named SoC either way, and a box
    without pyACL is still a box PANKO can search; what it must not do is claim
    the check passed. The verdict is recorded, so a report can say whether the
    numbers were confirmed against silicon or only against a name.
    """
    env = st.get("chip_envelope") or {}
    expected = env.get("soc") or ""
    ask = probe or (chip_profile.confirm if chip_profile is not None else None)
    if ask is None:
        return None
    verdict, live, how = ask(expected, device)
    env["confirmed"] = {
        "verdict": verdict,
        "live_soc": live,
        "via": how,
        "at": int(time.time()),
    }
    st["chip_envelope"] = env
    if verdict != "mismatch":
        return None
    return _chip_mismatch_report(expected, live, how, device, env)


def _device_probe(st, run_cmd=None):
    """Is the device usable NOW? Asked before any run is ended for it being busy.

    A block's faults are observations from the moment that block ran. Ending a
    campaign on them asserts something about the present from evidence about the
    past, and the cost is plain: a few faults inside one block can end a
    campaign a fraction of the way into its budget, most of it never spent, on a
    device nobody has asked since.

    The probe re-runs the correctness command and looks for a DEVICE marker in
    the output. Whether the kernel passes is beside the point -- a golden
    mismatch is a usable device saying no, which is exactly what the probe wants
    to distinguish from a device that will not answer at all.
    """
    cmd = st.get("test_command")
    if not cmd:
        return {
            "ran": False,
            "usable": False,
            "note": "no test_command recorded; the block's observation stands",
        }
    # The same shell-free runner the evaluator uses, so the probe re-runs the
    # command exactly as the block did -- see `evaluator.command_steps`.
    runner = run_cmd or (lambda c, t: bayesian.evaluator.run_command(c, t)[0])
    try:
        text = runner(cmd, st["config"].get("eval_timeout_s", 300))
    except Exception as e:  # timeout, OSError, anything
        return {"ran": False, "usable": False, "note": f"probe could not run: {e}"}
    markers = getattr(bayesian.evaluator, "DEVICE_FAULT_MARKERS", ())
    hit = next((m for m in markers if m in text), None)
    return {"ran": True, "usable": hit is None, "marker": hit}


def _device_verdict(st, a):
    """(reason, probe) for a run whose last block aborted on environment faults.

    A device another process is holding is not a search outcome, and it is the
    one stop the optimizer must not have to argue for: every block started on a
    contended device spends a few evaluations rediscovering it. Read off the
    last block's own abort rather than taken on the agent's word.

    Two things happen before that abort ends the run. A fault whose marker says
    the command never STARTED is a bug in this harness, not a busy device, and
    recording it as contention is how a mistyped path becomes a line in a
    results table. And a device that was busy some minutes ago may not be busy
    now, so it is asked.
    """
    last_block = (st.get("blocks") or [None])[-1]
    if not (last_block and "environment faults" in (last_block.get("aborted") or "")):
        return None, None
    if last_block.get("fault_kind") == "invocation":
        return "harness_fault", None
    evaluation_session.close_session(a.op_dir)
    probe = _device_probe(st)
    return (None if probe["usable"] else "device_unavailable"), probe


def _stop_condition(st, a):
    """The first halt condition that has fired, or None."""
    pr, cfg = st["progress"], st["config"]
    if pr["best_J"] >= 100:
        return "success"
    if pr.get("converged"):
        return "converged"
    if pr["wall_clock_s"] >= cfg["wall_clock_limit_s"]:
        return "wall_clock"
    if cfg.get("eval_budget") and pr["evals_used"] >= cfg["eval_budget"]:
        return "eval_budget"
    if (
        not _live(st, a.op_dir)
        and not pr.get("refine")
        and pr.get("expansion_declined")
    ):
        return "frontier_exhausted"
    if (
        cfg.get("global_stagnation_limit")
        and pr["global_stagnation"] >= cfg["global_stagnation_limit"]
    ):
        return "global_stagnation"
    return None


def _deliver_best(st, a):
    """Put the global best in the working file, or refuse the stop.

    `stop_reason` is sticky and the Stage-7 completion gate reads it, so writing
    it while the working file holds some other program would let Stage 7 complete
    on a kernel the search never chose.
    """
    best_hash = _best_hash(st)
    op_f = _op_file(st, a.op_dir)
    if _restore(a.op_dir, op_f, best_hash) or _file_hash(op_f) == best_hash:
        return
    out(
        {
            "stop": False,
            "refused": True,
            "error": "restore_failed",
            "best_code_hash": best_hash,
            "op_file": op_f,
            "reason": (
                "the global best has no snapshot on disk"
                if best_hash
                else "no global best code_hash is recorded"
            ),
            "hint": "the run cannot be closed out on a file that is not the "
            "winner. Restore optimization/nodes/<best_code_hash>.py "
            "into the kernel, then call stop again.",
        }
    )
    raise Refusal(3)


def _record_probe(st, a, probe):
    """Persisted whichever way it went. A run that CONTINUED past a block's
    env-fault abort is as much a decision as one that stopped, and neither is
    reconstructable afterwards without this.
    """
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    st["progress"]["device_probe"] = dict(probe, at=now)
    append_log(
        a.op_dir,
        st["op"],
        f"- device probe after env-fault abort: usable={probe.get('usable')}"
        + (f" marker={probe['marker']}" if probe.get("marker") else "")
        + (f" ({probe['note']})" if probe.get("note") else "")
        + (" -> run continues" if probe.get("usable") else " -> device_unavailable"),
    )


def cmd_stop(a):
    st = load(a.op_dir)
    pr = st["progress"]
    pr["wall_clock_s"] = int(time.time() - pr["_start_ts"])
    reason, probe = _device_verdict(st, a)
    if reason is None:
        reason = _stop_condition(st, a)
    if reason is None and getattr(a, "force", ""):
        # No condition fired, but the caller knows the run is over: the agent
        # session ended without ever calling stop. Recording that is strictly
        # better than leaving stop_reason null, which is indistinguishable from
        # a crash and fails the Stage-7 completion gate. Flagged separately so
        # analysis can exclude forced stops from budget accounting.
        reason = a.force
        pr["forced_stop"] = True
    if reason is not None:
        _deliver_best(st, a)
        pr["stop_reason"] = reason  # sticky; the Stage-7 completion gate reads this
        evaluation_session.close_session(a.op_dir)
    if probe is not None:
        _record_probe(st, a, probe)
    delivered = _delivered_check(st, a.op_dir)
    save(a.op_dir, st)
    out(
        {
            "stop": reason is not None,
            "reason": reason,
            "device_probe": probe,
            **delivered,
            "best_J": pr["best_J"],
            "best_latency_us": pr["best_latency_us"],
            "wall_clock_s": pr["wall_clock_s"],
            "evals_used": pr["evals_used"],
            "static_rejected": pr.get("static_rejected", 0),
            "duplicate_rejected": pr.get("duplicate_rejected", 0),
            "global_stagnation": pr["global_stagnation"],
            "forced": bool(pr.get("forced_stop")),
            "frontier_emptied_by_prune": pr.get("frontier_emptied_by_prune", 0),
            "expansion_requests": pr.get("expansion_requests", 0),
            "expansion_declined": pr.get("expansion_declined"),
        }
    )


def cmd_report(a):
    """Write the FINAL best block — once, only at a real stop.

    'Final best' is a completion marker, so it is gated on a real halt (progress.stop_reason set
    by cmd_stop), the same key the Stage-7 completion gate uses. Without this guard the
    optimizer could call `report` at a dispatch/checkpoint boundary and append a misleading
    '## Final best' mid-run (the search is still improving). No stop yet -> no-op.
    """
    st = load(a.op_dir)
    pr = st["progress"]
    if not pr.get("stop_reason"):
        out(
            {
                "reported": False,
                "note": "no stop yet — Final best is written only at a real halt",
            }
        )
        return
    # `report` is the last thing a run says, so it is where the deliverable is
    # checked. A run can report one best and ship a different kernel while the
    # snapshot of the winner sits on disk the whole time and nothing notices.
    d = _delivered_check(st, a.op_dir)
    if not d["delivered_is_best"]:
        _restore(a.op_dir, _op_file(st, a.op_dir), d["best_code_hash"])
        d = _delivered_check(st, a.op_dir)
        d["repaired"] = True
    append_log(
        a.op_dir,
        st["op"],
        f"\n## Final best\n- latency: {pr['best_latency_us']} us | j: {pr['best_J']} "
        f"| speedup_vs_preopt: {pr['best_speedup_vs_preopt']}x "
        f"| evals: {pr['evals_used']} | wall: {pr['wall_clock_s']}s | stop: {pr['stop_reason']}"
        f"\n- rejected without spending budget: {pr.get('static_rejected', 0)} static, "
        f"{pr.get('duplicate_rejected', 0)} duplicate"
        f"\n- code_version: {st.get('code_version')}"
        f"\n- delivered: {str(d['delivered_code_hash'])[:12]} "
        f"(best {str(d['best_code_hash'])[:12]}) "
        + (
            "OK"
            if d["delivered_is_best"]
            else "MISMATCH — the kernel on disk is NOT the best found"
        )
        + (
            " | repaired from the snapshot"
            if d.get("repaired") and d["delivered_is_best"]
            else ""
        ),
    )
    out(
        {
            "reported": True,
            "best_J": pr["best_J"],
            "stop_reason": pr["stop_reason"],
            **d,
            "static_rejected": pr.get("static_rejected", 0),
            "duplicate_rejected": pr.get("duplicate_rejected", 0),
        }
    )


# --------------------------------------------------------------------------- cli
def _parser_block(sub):
    """Arguments for `block`."""
    q = sub.add_parser("block")
    q.set_defaults(fn=cmd_block)
    q.add_argument("--op-dir", required=True)
    q.add_argument("--parent", required=True, help="the action this block belongs to")
    q.add_argument(
        "--test-command",
        dest="test_command",
        required=True,
        help="shell string that exits 0 iff golden+layout pass",
    )
    q.add_argument(
        "--reason", default="", help="why tuning is needed now; recorded for audit"
    )
    q.add_argument(
        "--live-tiles",
        dest="live_tiles",
        type=int,
        default=0,
        help="how many tile-shaped tensors are simultaneously resident. "
        "A FACT about the program, supplied because the extractor "
        "cannot read it after a fusion or a flatten. Never a bound.",
    )
    q.add_argument(
        "--max-trials",
        dest="max_trials",
        type=int,
        default=0,
        help="upper bound requested by the optimizer; the core clamps it "
        "against the remaining budget and the run-level share",
    )
    q.add_argument("--lint-root", dest="lint_root", default="")
    q.add_argument(
        "--evaluation-mode",
        choices=("subprocess", "resident", "cached", "prepared"),
        default="subprocess",
    )


def _parser_normalize(sub):
    """Arguments for `normalize`."""
    q = sub.add_parser("normalize")
    q.set_defaults(fn=cmd_normalize)
    q.add_argument(
        "--before", required=True, help="the kernel as generated, before normalisation"
    )
    q.add_argument("--after", required=True, help="the normalised kernel")
    q.add_argument("--op-dir", default="", help="optional, to log the verdict")
    q.add_argument("--op", default="")


# The optional preopt metric readings `init` accepts. All type=float,
# default=None, and only the flag name and help text differ.
_PREOPT_METRIC_ARGS = (
    (
        "--preopt-util",
        "preopt_util",
        "preopt core-util %% (optional; enables first-move symptom bias)",
    ),
    (
        "--preopt-bubble",
        "preopt_bubble",
        "preopt bubble %% (optional; enables first-move symptom bias)",
    ),
    (
        "--preopt-aic-util",
        "preopt_aic_util",
        "preopt cube-pipe core-util %% (optional; enables the per-pipe symptoms at INIT)",
    ),
    (
        "--preopt-aiv-util",
        "preopt_aiv_util",
        "preopt vector-pipe core-util %% (optional)",
    ),
    (
        "--preopt-pred-stall",
        "preopt_pred_stall",
        "preopt predecessor-wait %% (optional)",
    ),
    (
        "--preopt-aic-bubble",
        "preopt_aic_bubble",
        "preopt cube-pipe wait-schedule %% (optional)",
    ),
    (
        "--preopt-aiv-bubble",
        "preopt_aiv_bubble",
        "preopt vector-pipe wait-schedule %% (optional)",
    ),
)


def _parser_init(sub):
    """Arguments for `init`."""
    q = sub.add_parser("init")
    q.set_defaults(fn=cmd_init)
    q.add_argument("--op-dir", required=True)
    q.add_argument("--op", required=True)
    q.add_argument("--p-ref", type=float, required=True)
    q.add_argument("--device", default="")
    # Defaults to this skill's own catalogue, so a caller in any working
    # directory gets the right one without being told where it is.
    # Last resort, for a box where the platform ini is unreachable. Recorded as
    # explicit:cli so a report never reads as derived when it was supplied.
    q.add_argument("--chip-envelope", dest="chip_envelope", default="")
    q.add_argument("--catalog", default=DEFAULT_CATALOG)
    q.add_argument("--config", default="")
    q.add_argument("--preopt-s", dest="preopt_s", type=int, default=1)
    q.add_argument("--preopt-p", dest="preopt_p", type=float, required=True)
    q.add_argument("--preopt-hash", dest="preopt_hash", default="")
    for flag, dest, help_text in _PREOPT_METRIC_ARGS:
        q.add_argument(flag, dest=dest, type=float, default=None, help=help_text)
    q.add_argument(
        "--op-file",
        dest="op_file",
        default="",
        help="working <op>_impl.py the harness snapshots/restores (default: <op-dir>/<op>_impl.py)",
    )
    q.add_argument(
        "--lint-root",
        dest="lint_root",
        default="",
        help="the same --lint-root the blocks will be given; checked here so a "
        "wrong one is reported before it faults every trial",
    )
    q.add_argument(
        "--allow-unversioned",
        dest="allow_unversioned",
        action="store_true",
        help="start even when the harness build cannot be identified (CI only; "
        "a run started this way must not be reported)",
    )


def _parser_select(sub):
    """Arguments for `select`."""
    q = sub.add_parser("select")
    q.set_defaults(fn=cmd_select)
    q.add_argument("--op-dir", required=True)


def _parser_record(sub):
    """Arguments for `record`."""
    q = sub.add_parser("record")
    q.set_defaults(fn=cmd_record)
    q.add_argument("--op-dir", required=True)
    q.add_argument("--parent", required=True)
    q.add_argument("--code-hash", dest="code_hash", required=True)
    q.add_argument("--s", type=int, required=True)
    q.add_argument("--p", type=float, default=0.0)
    # None, NOT 0.0. A reading that was never taken and a kernel that genuinely
    # idled are different facts, and defaulting to zero made them the same one:
    # `_symptom_reboost` gates on `util > 0 or bubble > 0`, so every candidate
    # recorded without a reading silently disabled the device half of the
    # symptom index. In one campaign that was every block winner, spine nodes
    # included, and no device symptom fired for the whole run.
    q.add_argument("--util", type=float, default=None)
    q.add_argument("--bubble", type=float, default=None)
    # Per-pipe, from measure_latency (`aic_util` / `aiv_util` / `pred_stall` in
    # its result dict). Optional: they exist only when a bubble_analysis.log was
    # written, and the installed draw_swim_lane.py often skips it.
    q.add_argument(
        "--aic-util",
        dest="aic_util",
        type=float,
        default=None,
        help="cube-pipe core-util %% (optional; enables the per-pipe symptoms)",
    )
    q.add_argument(
        "--aiv-util",
        dest="aiv_util",
        type=float,
        default=None,
        help="vector-pipe core-util %% (optional)",
    )
    q.add_argument(
        "--pred-stall",
        dest="pred_stall",
        type=float,
        default=None,
        help="predecessor-wait %% (optional; separates a serial recurrence from a full pipe)",
    )
    q.add_argument(
        "--aic-bubble",
        dest="aic_bubble",
        type=float,
        default=None,
        help="cube-pipe wait-schedule %% (optional)",
    )
    q.add_argument(
        "--aiv-bubble",
        dest="aiv_bubble",
        type=float,
        default=None,
        help="vector-pipe wait-schedule %% (optional; rises when tasks get too big to fill the cores)",
    )


def _parser_feasible(sub):
    """Arguments for `feasible`."""
    q = sub.add_parser("feasible")
    q.set_defaults(fn=cmd_feasible)
    q.add_argument("--op-dir", required=True)
    q.add_argument("--parent", default="")
    q.add_argument("--op-file", dest="op_file", default="")
    # 0 = use the run's chip envelope. The old literals were a second, stale
    # copy of it: --l1-kb defaulted to 192 against a 512 KB buffer.
    q.add_argument("--ub-kb", dest="ub_kb", type=int, default=0)
    q.add_argument("--l1-kb", dest="l1_kb", type=int, default=0)
    q.add_argument("--mL1", dest="m_l1", type=int, default=None)
    q.add_argument("--kL1", dest="k_l1", type=int, default=None)


def _parser_probe(sub):
    """Arguments for `probe`."""
    q = sub.add_parser("probe")
    q.set_defaults(fn=cmd_probe)
    q.add_argument("--op-dir", required=True)
    q.add_argument("--code-hash", dest="code_hash", required=True)


def _parser_close(sub):
    """Arguments for `close`."""
    q = sub.add_parser("close")
    q.set_defaults(fn=cmd_close)
    q.add_argument("--op-dir", required=True)
    q.add_argument("--action", required=True)


def _parser_evolve(sub):
    """Arguments for `evolve`."""
    q = sub.add_parser("evolve")
    q.set_defaults(fn=cmd_evolve)
    q.add_argument("--op-dir", required=True)
    q.add_argument("--edits", required=True)


def _parser_stop(sub):
    """Arguments for `stop`."""
    q = sub.add_parser("stop")
    q.set_defaults(fn=cmd_stop)
    q.add_argument("--op-dir", required=True)
    q.add_argument(
        "--force",
        default="",
        help="record this stop_reason when no condition fired, e.g. session_ended. "
        "For the campaign driver to call after the agent exits; the agent must "
        "never pass it.",
    )


def _parser_report(sub):
    """Arguments for `report`."""
    q = sub.add_parser("report")
    q.set_defaults(fn=cmd_report)
    q.add_argument("--op-dir", required=True)


def main():
    p = argparse.ArgumentParser(description="Deterministic PANKO control core.")
    sub = p.add_subparsers(dest="cmd", required=True)

    _parser_block(sub)
    _parser_normalize(sub)
    _parser_init(sub)
    _parser_select(sub)
    _parser_record(sub)
    _parser_feasible(sub)
    _parser_probe(sub)
    _parser_close(sub)
    _parser_evolve(sub)
    _parser_stop(sub)
    _parser_report(sub)

    a = p.parse_args()
    try:
        a.fn(a)
    except Refusal as refusal:
        sys.exit(refusal.code)


if __name__ == "__main__":
    main()
