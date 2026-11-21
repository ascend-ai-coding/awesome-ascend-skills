# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Bounded optimization hypotheses from existing checks and maintained owners."""

from __future__ import annotations

import copy
from pathlib import Path

from . import (
    actions,
    feedback,
    performance_cards,
    query,
    ub_l1,
    vf_control_flow,
    work_distribution,
)
from .experiment import artifact, read_json, require, _number
from .paths import PYPTO_ROOT


def _layout_sites(facts):
    for index, conversion in enumerate(facts):
        src, dst = conversion["source"], conversion["destination"]
        if not (
            conversion["api"] == "move"
            and src.get("space") == dst.get("space") == "Vec"
            and src.get("layout") == "ND"
            and dst.get("layout") == "NZ"
        ):
            continue
        for consumer in facts[index + 1:]:
            if consumer["function"] != conversion["function"]:
                continue
            if (
                consumer["source_expr"] == conversion["destination_expr"]
                and consumer["source"] == dst
                and consumer["destination"].get("space") == "Mat"
            ):
                yield {
                    "file": conversion["file"],
                    "line": conversion["line"],
                    "handover_line": consumer["line"],
                    "observation": "A local ND→NZ move precedes a same-expression UB→L1 transfer.",
                    "unknowns": (
                        "This is a lexical candidate: active branch, producer address"
                        " map, aliases and other consumers are unproved."
                    ),
                }
                break


def _measurements(path, files, pypto_root):
    data = read_json(path)
    require(
        data.get("schema") == "pypto-kernel-costs/1", "Unsupported measured-cost schema"
    )
    expected = {str(p.resolve()): feedback.file_identity(p)["sha256"] for p in files}
    observed = data.get("sources", [])
    require(
        isinstance(observed, list) and len(observed) == len(expected),
        "Measured-cost source set differs",
    )
    actual = {
        str((path.resolve().parent / s["path"]).resolve()): s["sha256"]
        for s in observed
    }
    require(actual == expected, "Measured costs do not match current source bytes")
    selected = feedback.identity(pypto_root)["selected_source"]
    require(
        selected["complete"] and data.get("framework_sources") == selected["files"],
        "Measured costs use different framework sources",
    )
    require(
        data.get("scope") == "complete_call"
        and isinstance(data.get("workload"), dict)
        and data["workload"],
        "Measured costs need the complete-call workload/configuration",
    )
    toolchain = artifact(data["toolchain"], path.resolve().parent)
    profile = data.get("profile", {})
    require(isinstance(profile, dict), "profile must be an object")
    require(
        profile.get("metric") == "native_task_sum_us",
        "Costs must use the native-task timer",
    )
    require(
        type(profile.get("repeats")) is int and profile["repeats"] > 0,
        "Invalid profile sample count",
    )
    require(
        type(profile.get("tasks_per_call")) is int and profile["tasks_per_call"] > 0,
        "Invalid native task count",
    )

    def duration(stats):
        require(
            isinstance(stats, dict)
            and all(
                _number(stats.get(k), 1e-300) for k in ("min_us", "median_us", "max_us")
            )
            and stats["min_us"] <= stats["median_us"] <= stats["max_us"],
            "Invalid measured duration",
        )

    duration(profile)
    tasks = profile.get("per_task", [])
    require(
        isinstance(tasks, list)
        and all(isinstance(t, dict) for t in tasks)
        and len(tasks) == profile["tasks_per_call"]
        and sorted(t["position"] for t in tasks) == list(range(1, len(tasks) + 1)),
        "Incomplete per-task profile",
    )
    for task in tasks:
        duration(task)
    require(
        isinstance(data.get("sites"), list),
        "Declare source-site to task-position mappings explicitly",
    )
    seen = set()
    for site in data["sites"]:
        require(isinstance(site, dict), "Measured source sites must be objects")
        site["file"] = str((path.resolve().parent / site["file"]).resolve())
        require(
            site["file"] in expected
            and type(site.get("line")) is int
            and 1 <= site["line"] <= len(Path(site["file"]).read_text().splitlines())
            and type(site.get("task_position")) is int
            and 1 <= site["task_position"] <= len(tasks),
            "Invalid source-site/task-position mapping",
        )
        key = (site["file"], site["line"])
        require(key not in seen, "Duplicate measured source site")
        seen.add(key)
    return {**data, "toolchain": toolchain, "reference": feedback.file_identity(path)}


def _cost(site, data):
    if data is None:
        return None
    match = next(
        (
            s
            for s in data["sites"]
            if s["file"] == site["file"]
            and s["line"] in {site["line"], site.get("handover_line")}
        ),
        None,
    )
    if not match:
        return None
    profile, position = data["profile"], match["task_position"]
    return {
        "reference": data["reference"],
        "toolchain": data["toolchain"],
        "workload": data["workload"],
        "complete_call_median_us": profile["median_us"],
        "task": next(t for t in profile["per_task"] if t["position"] == position),
        "pipe_counters": next(
            (p for p in profile.get("pipe_counters", []) if p["position"] == position),
            None,
        ),
        "attribution": (
            "Caller-declared source/task mapping; task duration is not this site's"
            " exclusive cost, and counters may overlap."
        ),
    }


def report(
    files,
    *,
    measurements=None,
    limit=3,
    pypto_root=PYPTO_ROOT,
    goal=None,
    cards_root=None,
):
    actions.goal(goal)
    require(type(limit) is int and 1 <= limit <= 5, "Candidate limit must be 1..5")
    files = sorted({Path(p).resolve() for p in files})
    require(
        files and all(p.is_file() and p.suffix == ".py" for p in files),
        "Provide existing Python source files",
    )
    result = feedback.envelope("optimization_candidates", pypto_root, files)
    recipes = {}
    for meta, path, _ in query.sections():
        if meta.get("candidate"):
            recipe = meta["candidate"]
            require(
                set(recipe)
                == {
                    "id",
                    "title",
                    "conditions",
                    "decrease",
                    "increase",
                    "reject_if",
                    "comparison",
                },
                "Invalid maintained candidate metadata",
            )
            require(recipe["id"] not in recipes, "Duplicate maintained candidate ID")
            recipes[recipe["id"]] = (meta, recipe, path)
    data = None
    if measurements:
        try:
            data = _measurements(Path(measurements), files, pypto_root)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result["status"] = "unverified"
            result["diagnostics"].append(
                feedback.diagnostic(
                    "optimization-cost-unverified",
                    "unverified",
                    str(exc),
                    next_action=(
                        "Recapture costs for these sources and toolchain; candidates below have no measured ranking."
                    ),
                )
            )
    sites, unknown_transfers = [], 0
    for file in files:
        for module in (vf_control_flow, work_distribution):
            for finding in module.check(file):
                sites.append(
                    (
                        finding.rule,
                        {
                            "file": str(file),
                            "line": finding.lineno,
                            "observation": finding.message,
                        },
                    )
                )
        _, facts, coverage = ub_l1.inspect(file, pypto_root=pypto_root)
        unknown_transfers += coverage["unverified"]
        sites += [("ub-l1-extra-layout-move", site) for site in _layout_sites(facts)]
    candidates = []
    for pattern, site in sites:
        require(pattern in recipes, f"No maintained recipe for {pattern}")
        meta, recipe, owner = recipes[pattern]
        identity = feedback.file_identity(site["file"])
        candidates.append(
            {
                "id": (
                    f"{pattern}:"
                    f"{feedback.digest((identity['path'] + identity['sha256'] + str(site['line'])).encode())[:16]}"
                ),
                "pattern": pattern,
                "title": recipe["title"],
                "site": site,
                "source": identity,
                "conditions": copy.deepcopy(recipe["conditions"]),
                "prediction": {
                    "decrease": recipe["decrease"],
                    "increase": recipe["increase"],
                    "latency_delta_us": None,
                    "basis": (
                        "Mechanism hypothesis; no current speedup is inferred from a"
                        " source pattern or historical result."
                    ),
                },
                "reject_if": recipe["reject_if"],
                "comparison": recipe["comparison"],
                "measured_cost": _cost(site, data),
                "query": meta["id"],
                "reference": {**feedback.file_identity(owner), "line": meta["line"]},
            }
        )
    candidates.sort(
        key=lambda c: (
            c["measured_cost"] is None,
            -c["measured_cost"]["task"]["median_us"] if c["measured_cost"] else 0,
            c["site"]["file"],
            c["site"]["line"],
            c["pattern"],
        )
    )
    card_coverage = None
    if cards_root is not None:
        cards, card_coverage = performance_cards.collect(
            cards_root,
            references_root=query.REFERENCES_ROOT,
            entries=list(query.sections()),
        )
        snapshot = card_coverage.pop("_files")
        card_coverage.pop("_errors")
        try:
            query.verify_files(snapshot)
        except ValueError as error:
            card_coverage.update(complete=False, source_changed=str(error))
            cards = []
        if not card_coverage["complete"]:
            result["status"] = "unverified"
            result["diagnostics"].append(
                feedback.diagnostic(
                    "optimization-cards-unverified",
                    "unverified",
                    "The configured card source is incomplete; no card applicability or performance is inferred.",
                )
            )
        for candidate in candidates:
            candidate["related_cards"] = [
                {
                    "id": card["id"],
                    "reference": card["reference"],
                    "card": card["card"],
                    "relation": relation,
                    "grouping": "same_mechanism_only"
                    if relation["kind"] == "same_mechanism"
                    else "reference_only",
                    "candidate_equivalence": "not_established",
                    "applicability": "not_established",
                }
                for card in cards
                for relation in card["relations"]
                if relation["target"] == "contract:" + candidate["query"]
            ]
    result["candidates"] = candidates[:limit]
    result["coverage"] = {
        "source_files": len(files),
        "matched_sites": len(candidates),
        "omitted_candidates": max(0, len(candidates) - limit),
        "unresolved_transfers": unknown_transfers,
        "limits": (
            "Three lexical pattern families only. No call-graph, dynamic work counts,"
            " codegen prediction or proof of an optimization ceiling."
        ),
    }
    if card_coverage is not None:
        result["coverage"]["cards"] = card_coverage
    result["next_action"] = (
        "Validate applicability and select one hypothesis; use query for its maintained owner and examples."
        if candidates
        else "No covered pattern matched. Inspect measured task costs and use the optimization workflow."
    )
    actions.for_report(result, goal)
    return result
