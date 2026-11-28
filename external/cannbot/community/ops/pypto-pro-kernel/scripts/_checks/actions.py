# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Action-scoped guidance derived from explicit goals and existing evidence.

This is a read-only projection, not a planner, acceptance gate or knowledge store.
"""

from __future__ import annotations

GOALS = (
    "select_experiment",
    "native_probe",
    "validate_numerics",
    "compare_performance",
)
_REQUIRED = {
    "select_experiment": (
        "For implementation optimization, identify the baseline's exposed parameter"
        " family, expand actual contraction/reduction extents and all addends, and"
        " compare removed/added complete-call work before selecting a local change. A"
        " selector for existing group widths, threads or tiles remains family selection"
        " even when realized ownership changes. Derive a distinct arithmetic/dataflow"
        " candidate where credible, with qualification and a rejection test; preserve requested fixed-family scope."
    ),
    "native_probe": "Use concrete tensor metadata and the matching installation for a source-bound native probe.",
    "validate_numerics": "Run the original evaluator at the original tolerance over the declared acceptance cases.",
    "compare_performance": "Verify the exact variants numerically, then collect matched AB/BA complete-call timing.",
}
_NEXT = {
    "select_experiment": "form_one_hypothesis",
    "native_probe": "native_probe",
    "validate_numerics": "numerical_evaluation",
    "compare_performance": "paired_complete_call_timing",
}
_LATER = {
    "select_experiment": [
        "installed_native_probe",
        "independent_numerical_validation",
        "paired_complete_call_timing",
    ],
    "native_probe": ["independent_numerical_validation", "paired_complete_call_timing"],
    "validate_numerics": ["paired_complete_call_timing"],
    "compare_performance": [],
}


def goal(value):
    if value is not None and (not isinstance(value, str) or value not in GOALS):
        raise ValueError("Unsupported action goal")
    return value


def guidance(requested, *, lookup="not_assessed"):
    goal(requested)
    return {
        "goal": requested,
        "lookup": lookup,
        "readiness": "conditional",
        "next_action": _NEXT[requested],
        "required_now": [_REQUIRED[requested]],
        "later_checks": list(_LATER[requested]),
        "evidence_limit": (
            "Guidance for the requested action only; no installation, numerical,"
            " performance or global-optimality acceptance."
        ),
        "reading_boundary": (
            "Once the current contract and applicability facts are available, take the"
            " next discriminating action. Re-reading declarations cannot replace an execution or measurement."
        ),
    }


def for_report(result, requested):
    """Keep unmatched, conditional and executed evidence distinct."""
    if goal(requested) is None:
        return
    context = [item["id"] for item in result.get("context", [])]
    objects = result.get("knowledge", {}).get("items", [])
    context += [
        item["id"].removeprefix("contract:")
        for item in objects
        if item["kind"] == "contract"
        and "text" in item
        and item["id"].removeprefix("contract:") not in context
    ]
    candidates = result.get("candidates", [])
    known = bool(context or candidates)
    declared = bool(
        result.get("signatures")
        or any(item["evidence"]["kind"] == "source_declaration" for item in objects)
    )
    item = guidance(
        requested,
        lookup="scenario_found"
        if known
        else "declaration_only"
        if declared
        else "unresolved",
    )
    item["owners"] = (
        context or sorted({candidate["query"] for candidate in candidates})[:3]
    )
    if result["status"] == "failed":
        item.update(
            readiness="needs_source_repair",
            next_action="resolve_reported_source_errors",
        )
    elif any(
        d["rule"] == "optimization-cost-unverified"
        for d in result.get("diagnostics", [])
    ):
        item.update(
            readiness="needs_execution_evidence",
            next_action="recapture_source_bound_costs",
            required_now=[
                "Recapture costs for the current source and toolchain before using a measured ranking."
            ],
        )
    elif (
        result.get("stage") == "optimization_candidates"
        and not known
        and requested == "select_experiment"
    ):
        item.update(
            readiness="needs_workload_evidence",
            next_action="inspect_current_costs_and_workload",
            required_now=[
                "No covered pattern matched. Expand actual contractions/reductions,"
                " trace addends and consumers, and count repeated producers, residency"
                " and publications from source. Separate existing-family selection from"
                " a new arithmetic/dataflow candidate; compare removed and added"
                " complete-call work before choosing a local experiment. The detector"
                " is not exhaustive, and a miss is not an API gap or optimization"
                " ceiling. Preserve requested fixed-family scope."
            ],
        )
    elif (
        not known
        and requested != "select_experiment"
        and (declared or result.get("stage") == "optimization_candidates")
    ):
        item["readiness"] = "needs_execution_evidence"
    elif not known:
        item.update(
            readiness="needs_applicability_fact",
            next_action="resolve_one_contract_question",
            required_now=[
                "Resolve the specific missing producer/consumer or contract fact using"
                " the returned conditional routes; no match does not establish absence."
            ],
        )
    elif requested != "select_experiment":
        item["readiness"] = "needs_execution_evidence"
    table = result.get("selected_frontend_store")
    if table:
        item["selected_table"] = {key: table[key] for key in ("status", "reference") if key in table}
        if table["status"] != "observed_source" and result["status"] != "failed":
            item.update(
                readiness="needs_source_fact",
                next_action="resolve_selected_table_or_probe",
                required_now=[
                    "The selected frontend table was not resolved. Inspect its current"
                    " declaration or perform a matching native probe; do not infer support from this query."
                ],
            )
    result["action_guidance"] = item


def for_inventory(result, requested):
    if goal(requested) is None:
        return
    item = guidance(requested)
    item["input_scope"] = (
        "Declared bytes and unauthenticated inline statements; source semantics have not been inferred."
    )
    if result["status"] != "complete":
        item.update(
            readiness="inputs_incomplete",
            next_action="resolve_declared_input_identity",
            required_now=[
                "Resolve missing, changed or incomplete declared inputs before using their evidence."
            ],
        )
    elif requested == "select_experiment":
        item.update(
            next_action="assess_supplied_applicability",
            required_now=[
                "Read the supplied source/workload; for implementation optimization,"
                " identify existing controls, expand actual contractions and trace"
                " every addend before asserting an invariant. Compare"
                " work-elimination/reuse opportunities with local fusion costs, then"
                " choose a falsifiable candidate; an existing-parameter selector alone"
                " is family selection. Preserve requested fixed-family scope. Query"
                " only a named unresolved contract; inline evidence is not another file to search for."
            ],
        )
    else:
        item["readiness"] = "needs_execution_evidence"
    result["action_guidance"] = item


def for_experiment(state):
    """Use the existing state as the only authority for experiment progression."""
    phase = state["phase"]
    if not state["best"]["verified"] or phase == "awaiting_correctness":
        item = guidance("validate_numerics")
    elif phase == "awaiting_evidence":
        item = guidance("native_probe")
        item["next_action"] = "check_parse_and_validate_candidate"
    elif phase == "awaiting_timing":
        item = guidance("compare_performance")
    elif (
        phase == "rejected"
        and "failed"
        in ((state.get("current") or {}).get("correctness") or {})
        .get("cases", {})
        .values()
    ):
        item = guidance("validate_numerics")
        item.update(
            next_action="restore_verified_best_and_resolve_numerical_failure",
            required_now=[
                "Restore/export the verified best; inspect the failed"
                " original-evaluator cases before another performance experiment. Keep"
                " the original tolerance and required arithmetic."
            ],
        )
    elif phase in {"ready", "retained", "rejected", "inconclusive"}:
        item = guidance("select_experiment")
        item["next_action"] = (
            "export_verified_best_and_report_open_directions"
            if state["trials"] >= state["max_trials"]
            else "resolve_timing_uncertainty_or_select_next_hypothesis"
            if phase == "inconclusive"
            else "select_next_hypothesis_from_best"
        )
        if state["trials"] >= state["max_trials"]:
            item["required_now"].append(
                "Export the verified best from this exhausted state. A locally chosen"
                " batch cap does not finish the user's task: continue actionable work"
                " in the next working batch unless an explicit user stop, verified hard"
                " execution limit or genuine blocker prevents it. Preserve cumulative"
                " quotas; never reset a real limit through another state."
            )
    else:
        item = guidance("select_experiment")
        item.update(
            readiness="unknown", next_action="inspect_existing_experiment_phase"
        )
    item["phase"] = phase
    return item
