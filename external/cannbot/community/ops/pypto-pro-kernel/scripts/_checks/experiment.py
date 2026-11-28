# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""One best snapshot and one current experiment, with source-bound evidence gates.

This consumes explicit verifier observations; it does not execute kernels, prove
an oracle, infer omitted dependencies or charge a device-request budget.
"""

from __future__ import annotations

import base64
import copy
import fcntl
import json
import math
import os
import re
import tempfile
from pathlib import Path

from . import actions, feedback
from .paths import PYPTO_ROOT

SCHEMA = "pypto-experiment-state/1"
MAX_SOURCE_BYTES = 2 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def read_json(path, *, limit=512 * 1024):
    data = Path(path).read_bytes()
    require(len(data) <= limit, f"JSON exceeds {limit} bytes: {path}")
    result = json.loads(data)
    require(isinstance(result, dict), "Expected a JSON object")
    canonical(result)
    return result


def sha(value):
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
        "Expected a SHA256 hex digest",
    )
    return value


def artifact(value, base):
    require(
        isinstance(value, dict) and isinstance(value.get("path"), str),
        "Evidence needs path and sha256",
    )
    path = (base / value["path"]).resolve()
    expected = sha(value.get("sha256"))
    require(
        feedback.file_identity(path)["sha256"] == expected, f"Evidence changed: {path}"
    )
    return {"path": str(path), "sha256": expected}


def relative_file(name):
    require(
        isinstance(name, str)
        and name
        and not Path(name).is_absolute()
        and ".." not in Path(name).parts
        and str(Path(name)) == name,
        "Source names must be normalized relative file paths",
    )
    return name


def variant(spec, base=None):
    """Hash the actual execution inputs; use the same spec in the verifier driver."""
    require(isinstance(spec, dict), "variant must be an object")
    if base is None:
        base = Path.cwd()
    root = (base / spec["root"]).resolve()
    names = spec["files"]
    require(
        isinstance(names, list)
        and 0 < len(names) <= 64
        and len(set(names)) == len(names),
        "Declare 1..64 unique source/dependency files",
    )
    entry = spec["entry"]
    require(
        isinstance(entry, str) and ":" in entry and entry.split(":", 1)[0] in names,
        "entry must be a declared relative file:symbol",
    )
    require(
        isinstance(spec.get("configuration"), dict),
        "Declare the executed configuration, even if empty",
    )
    files, total = {}, 0
    for name in sorted(names):
        relative_file(name)
        path = (root / name).resolve()
        require(root in path.parents, "Source dependency escapes its root")
        data = path.read_bytes()
        total += len(data)
        require(
            total <= MAX_SOURCE_BYTES,
            "Source snapshot exceeds 2 MiB; select executable dependencies only",
        )
        files[name] = {
            "sha256": feedback.digest(data),
            "content": base64.b64encode(data).decode(),
        }
    result = {"entry": entry, "configuration": spec["configuration"], "files": files}
    result["sha256"] = variant_digest(result)
    return result


def variant_digest(value):
    return feedback.digest(
        canonical(
            {
                "entry": value["entry"],
                "configuration": value["configuration"],
                "files": {
                    name: item["sha256"] for name, item in value["files"].items()
                },
            }
        )
    )


def check_snapshot(value):
    require(value["sha256"] == variant_digest(value), "Variant identity changed")
    total = 0
    for name, item in value["files"].items():
        relative_file(name)
        data = base64.b64decode(item["content"], validate=True)
        total += len(data)
        require(feedback.digest(data) == item["sha256"], "Snapshot bytes changed")
    require(total <= MAX_SOURCE_BYTES, "Snapshot exceeds 2 MiB")


def _scope(scope):
    require(isinstance(scope, dict), "scope must be an object")
    cases = scope.get("cases")
    require(
        isinstance(cases, dict)
        and 0 < len(cases) <= 100
        and all(
            isinstance(k, str) and k and isinstance(v, dict) and v
            for k, v in cases.items()
        ),
        "scope.cases needs 1..100 named, concrete workloads",
    )
    timing = scope.get("timing_cases")
    require(
        isinstance(timing, list)
        and timing
        and len(timing) == len(set(timing))
        and set(timing) <= set(cases),
        "timing_cases must be a nonempty subset of cases",
    )
    require(
        scope.get("metric") in {"native_task_sum_us", "synchronized_wall_us"},
        "Unsupported target metric",
    )
    require(
        isinstance(scope.get("numerical_contract"), dict)
        and scope["numerical_contract"],
        "Freeze the numerical contract and oracle identity in scope.numerical_contract",
    )
    sha(scope["numerical_contract"].get("oracle_sha256"))
    return scope


def _number(value, minimum=0, maximum=math.inf):
    return (
        type(value) in (int, float)
        and math.isfinite(value)
        and minimum <= value <= maximum
    )


def _policy(value):
    defaults = {"min_pairs": 5, "min_improvement": 0.02, "min_win_fraction": 0.7}
    require(
        isinstance(value, dict) and not set(value) - set(defaults),
        "Unknown comparison policy field",
    )
    policy = {**defaults, **value}
    require(
        type(policy["min_pairs"]) is int and policy["min_pairs"] >= 2,
        "min_pairs must be at least 2",
    )
    require(_number(policy["min_improvement"], 0, 0.99), "Invalid min_improvement")
    require(_number(policy["min_win_fraction"], 0.5, 1), "Invalid min_win_fraction")
    return policy


def _framework(pypto_root):
    selected = feedback.identity(pypto_root)["selected_source"]
    require(selected["complete"], "Selected framework source identity is incomplete")
    return selected["files"]


def initialize(spec, base, pypto_root):
    require(
        spec.get("schema") == "pypto-experiment-plan/1",
        "Unsupported experiment plan schema",
    )
    maximum = spec.get("max_trials")
    require(
        type(maximum) is int and maximum > 0,
        "max_trials must be a positive selection budget",
    )
    return {
        "schema": SCHEMA,
        "scope": _scope(spec["scope"]),
        "policy": _policy(spec.get("policy", {})),
        "toolchain": artifact(spec["toolchain"], base),
        "framework_sources": _framework(pypto_root),
        "max_trials": maximum,
        "trials": 0,
        "best": {
            "variant": variant(spec["baseline"], base),
            "verified": False,
            "evidence": None,
            "correctness": None,
            "performance": None,
        },
        "current": None,
        "phase": "baseline_unverified",
        "next_action": "Validate the complete baseline call on every acceptance case with the frozen oracle.",
    }


def select(state, spec, base):
    require(
        spec.get("schema") == "pypto-experiment-selection/1",
        "Unsupported selection schema",
    )
    require(
        state["best"]["verified"], "Verify the baseline before selecting a candidate"
    )
    require(
        state["trials"] < state["max_trials"],
        "Selection budget exhausted; retain best and report unresolved directions",
    )
    hypothesis = spec.get("hypothesis", {})
    require(isinstance(hypothesis, dict), "hypothesis must be an object")
    for key in ("observation", "change", "prediction", "reject_if"):
        require(
            isinstance(hypothesis.get(key), str) and 0 < len(hypothesis[key]) <= 2000,
            f"hypothesis.{key} must describe this comparison",
        )
    require(
        isinstance(spec.get("changes_wrapper"), bool),
        "Declare changes_wrapper explicitly",
    )
    current = variant(spec["variant"], base)
    require(
        current["sha256"] != state["best"]["variant"]["sha256"],
        "Candidate is identical to the best configuration and bytes",
    )
    state["current"] = {
        "variant": current,
        "hypothesis": {
            k: hypothesis[k]
            for k in ("observation", "change", "prediction", "reject_if")
        },
        "changes_wrapper": spec["changes_wrapper"],
        "decision": "awaiting_evidence",
        "evidence": None,
        "correctness": None,
        "comparisons": [],
    }
    state["trials"] += 1
    state["phase"] = "awaiting_evidence"
    state["next_action"] = (
        "Check/parse the snapshot, validate every acceptance case, then collect paired complete-call timings."
    )


def correctness(item, expected, state):
    require(
        isinstance(item, dict) and item.get("variant_sha256") == expected,
        "Executed variant does not match its snapshot",
    )
    data = item.get("correctness", {})
    require(isinstance(data, dict), "correctness must be an object")
    if not data:
        return "unverified"
    require(
        data.get("scope") == "complete_call", "Correctness must cover the complete call"
    )
    require(
        data.get("oracle_sha256")
        == state["scope"]["numerical_contract"]["oracle_sha256"],
        "Numerical oracle changed",
    )
    cases = data.get("cases", {})
    require(
        isinstance(cases, dict)
        and set(cases) <= set(state["scope"]["cases"])
        and all(v in {"passed", "failed", "not_run"} for v in cases.values()),
        "Invalid correctness case statuses",
    )
    if "failed" in cases.values():
        return "failed"
    return (
        "passed"
        if cases and all(cases.get(k) == "passed" for k in state["scope"]["cases"])
        else "unverified"
    )


def paired(data, metric, policy):
    """Validate the existing timer's AB/BA summaries; never manufacture samples."""
    require(
        isinstance(data, dict) and data.get("metric") == metric,
        "Timing metric differs from the frozen target",
    )
    require(
        type(data.get("warmup")) is int and data["warmup"] > 0, "Timing requires warmup"
    )
    repeats = data.get("pairs_per_order")
    require(
        type(repeats) is int and repeats > 0, "Timing requires a positive pair count"
    )
    if metric == "native_task_sum_us":
        counts = data.get("tasks_per_call", {})
        require(
            isinstance(counts, dict)
            and set(counts) == {"baseline", "candidate"}
            and all(type(v) is int and v > 0 for v in counts.values()),
            "Declare complete-call native task counts for both variants",
        )
    orders = data.get("orders", [])
    require(
        isinstance(orders, list)
        and len(orders) == 2
        and all(isinstance(order, dict) for order in orders)
        and {order.get("order") for order in orders} == {"AB", "BA"},
        "Both AB and BA orders are required",
    )
    outcomes = []
    for order in orders:
        require(
            order.get("pairs") == repeats,
            "Pair count differs between summary and orders",
        )
        for side in ("baseline", "candidate"):
            stats = order.get(side, {})
            require(
                isinstance(stats, dict)
                and all(
                    _number(stats.get(k), 1e-300)
                    for k in ("min_us", "median_us", "max_us")
                ),
                "Durations must be finite and positive",
            )
            require(
                stats["min_us"] <= stats["median_us"] <= stats["max_us"],
                "Invalid duration range",
            )
        ratio, wins = order.get("median_ratio"), order.get("candidate_wins")
        require(
            _number(ratio, 1e-300) and type(wins) is int and 0 <= wins <= repeats,
            "Invalid paired ratios/wins",
        )
        require(
            _number(order.get("win_fraction"), 0, 1)
            and math.isclose(order["win_fraction"], wins / repeats),
            "Win fraction disagrees with wins",
        )
        lo = order["candidate"]["min_us"] / order["baseline"]["max_us"]
        hi = order["candidate"]["max_us"] / order["baseline"]["min_us"]
        require(
            lo - 1e-12 <= ratio <= hi + 1e-12,
            "Paired ratio is outside the possible duration range",
        )
        require(
            not (ratio < 1 and wins < math.ceil(repeats / 2))
            and not (ratio >= 1 and wins > repeats / 2),
            "Median ratio contradicts the paired win count",
        )
        if repeats < policy["min_pairs"]:
            outcomes.append("inconclusive")
        elif (
            ratio < 1 - policy["min_improvement"]
            and wins / repeats >= policy["min_win_fraction"]
        ):
            outcomes.append("improved")
        elif ratio > 1 + policy["min_improvement"]:
            outcomes.append("regressed")
        else:
            outcomes.append("inconclusive")
    return outcomes[0] if outcomes[0] == outcomes[1] else "inconclusive"


def observe(state, data, evidence):
    require(
        data.get("schema") == "pypto-experiment-observation/1",
        "Unsupported observation schema",
    )
    require(
        data.get("scope") == state["scope"],
        "Observation workload or numerical contract differs",
    )
    require(
        data.get("toolchain_sha256") == state["toolchain"]["sha256"],
        "Observation toolchain differs",
    )
    baseline = correctness(
        data.get("baseline"), state["best"]["variant"]["sha256"], state
    )
    current = state["current"]
    if current is None:
        require(
            not data.get("candidate") and not data.get("comparisons"),
            "Baseline verification cannot contain a candidate",
        )
        state["best"]["verified"] = baseline == "passed"
        state["best"]["evidence"] = evidence
        state["best"]["correctness"] = copy.deepcopy(
            data["baseline"].get("correctness")
        )
        state["phase"] = "ready" if baseline == "passed" else "baseline_unverified"
        state["next_action"] = (
            "Use measured costs to select one discriminating candidate."
            if baseline == "passed"
            else "Repair or complete baseline numerical validation; no candidate can be retained."
        )
        return
    require(
        current["decision"] != "retained",
        "This candidate is already best; select the next candidate before observing",
    )
    candidate = correctness(data.get("candidate"), current["variant"]["sha256"], state)
    current["evidence"], current["comparisons"] = evidence, []
    current["correctness"] = copy.deepcopy(data["candidate"].get("correctness"))
    if baseline == "failed":
        state["best"]["verified"] = False
        state["best"]["correctness"] = copy.deepcopy(data["baseline"]["correctness"])
        state["best"]["evidence"] = evidence
        state["best"]["performance"] = None
    elif baseline == "passed":
        if not state["best"]["verified"]:
            state["best"]["correctness"] = copy.deepcopy(
                data["baseline"]["correctness"]
            )
            state["best"]["evidence"] = evidence
        state["best"]["verified"] = True
    if "failed" in (baseline, candidate):
        decision, action = (
            "rejected",
            "Resolve numerical failures at the original tolerance; keep the best snapshot.",
        )
    elif "unverified" in (baseline, candidate):
        decision, action = (
            "awaiting_correctness",
            "Complete numerical coverage for both exact snapshots before timing acceptance.",
        )
    else:
        comparisons = data.get("comparisons", [])
        require(
            isinstance(comparisons, list)
            and len(comparisons) <= len(state["scope"]["timing_cases"])
            and all(isinstance(c, dict) for c in comparisons),
            "Invalid comparison count",
        )
        names = [c.get("case") for c in comparisons]
        require(
            len(set(names)) == len(names)
            and set(names) <= set(state["scope"]["timing_cases"]),
            "Unexpected or duplicate timing case",
        )
        outcomes = []
        for comparison in comparisons:
            require(
                comparison.get("scope") == "complete_call"
                and comparison.get("matched_inputs") is True,
                "Timing must compare complete calls with matched inputs/configuration in one session",
            )
            timing = comparison.get("timing")
            outcome = paired(timing, state["scope"]["metric"], state["policy"])
            counts = timing.get("tasks_per_call", {})
            needs_wall = state["scope"]["metric"] == "native_task_sum_us" and (
                current["changes_wrapper"]
                or counts.get("baseline") != counts.get("candidate")
            )
            wall = comparison.get("wall_timing")
            if wall is not None:
                wall_outcome = paired(wall, "synchronized_wall_us", state["policy"])
                if needs_wall and wall_outcome != "improved":
                    outcome = "inconclusive"
            elif needs_wall:
                outcome = "awaiting_wall"
            outcomes.append(outcome)
            current["comparisons"].append(
                {
                    "case": comparison["case"],
                    "outcome": outcome,
                    "timing": timing,
                    "wall_timing": wall,
                }
            )
        if (
            set(names) != set(state["scope"]["timing_cases"])
            or "awaiting_wall" in outcomes
        ):
            decision, action = (
                "awaiting_timing",
                "Collect missing representative AB/BA timings and required paired wall timings.",
            )
        elif "regressed" in outcomes:
            decision, action = (
                "rejected",
                "Keep best; revisit the predicted cost or propose a scoped tradeoff as a new experiment.",
            )
        elif outcomes and all(outcome == "improved" for outcome in outcomes):
            decision, action = (
                "retained",
                "Export best for delivery, or select another cost-driven candidate within the remaining budget.",
            )
            previous = state["best"]["variant"]["sha256"]
            state["best"] = {
                "variant": copy.deepcopy(current["variant"]),
                "verified": True,
                "evidence": evidence,
                "correctness": copy.deepcopy(data["candidate"]["correctness"]),
                "performance": {
                    "against_variant_sha256": previous,
                    "comparisons": copy.deepcopy(current["comparisons"]),
                },
            }
        else:
            decision, action = (
                "inconclusive",
                "Keep best; resolve noise/order dependence or choose a different hypothesis.",
            )
    current["decision"] = state["phase"] = decision
    state["next_action"] = action


def summary(state):
    result = copy.deepcopy(state)
    result["action_guidance"] = actions.for_experiment(state)
    for record in (result["best"], result.get("current")):
        if record:
            for item in record["variant"]["files"].values():
                item.pop("content", None)
    result.pop("framework_sources", None)
    result.pop("payload_sha256", None)
    result["remaining_trials"] = state["max_trials"] - state["trials"]
    if result["remaining_trials"] == 0:
        result["next_action"] += (
            " No further candidate selections remain; unresolved directions stay open."
        )
    result["evidence_limit"] = (
        "Verifier attestations, not independent replay. Only declared files are"
        " snapshotted; device/time/token budgets remain external."
    )
    return result


def _write(path, state):
    state.pop("payload_sha256", None)
    state["payload_sha256"] = feedback.digest(canonical(state))
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(
                state,
                stream,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def run(
    path,
    *,
    init=None,
    selection=None,
    observation=None,
    export_best=None,
    pypto_root=PYPTO_ROOT,
):
    """Serialize updates to one explicit scratch file; reads never rewrite it."""
    path = Path(path).absolute()
    require(
        path.suffix == ".json" and not path.is_symlink(),
        "State must be a nonsymlink .json file in scratch",
    )
    require(
        sum(v is not None for v in (init, selection, observation, export_best)) <= 1,
        "Choose one state action",
    )
    with path.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(
                "Experiment state is in use; retry after the other update"
            ) from exc
        if init:
            require(
                not path.exists(),
                "State already exists; use show/select/observe instead of overwriting best",
            )
            state = initialize(read_json(init), Path(init).resolve().parent, pypto_root)
        else:
            state = read_json(path, limit=8 * 1024 * 1024)
            checksum = state.pop("payload_sha256", None)
            require(
                checksum == feedback.digest(canonical(state))
                and state.get("schema") == SCHEMA,
                "State integrity/schema mismatch",
            )
            require(
                state["framework_sources"] == _framework(pypto_root),
                "Framework source changed; start a newly validated experiment",
            )
            artifact(state["toolchain"], path.parent)
            for record in (state["best"], state.get("current")):
                if record:
                    check_snapshot(record["variant"])
        if selection:
            select(state, read_json(selection), Path(selection).resolve().parent)
        if observation:
            observe(state, read_json(observation), feedback.file_identity(observation))
        if export_best:
            require(
                state["best"]["verified"],
                "Best snapshot has not passed numerical validation",
            )
            export_best = Path(export_best)
            export_best.mkdir(parents=True, exist_ok=False)
            for name, item in state["best"]["variant"]["files"].items():
                target = export_best / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(base64.b64decode(item["content"], validate=True))
        if init or selection or observation:
            _write(path, state)
        return summary(state)
