# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Action readiness never promotes source facts or stale declarations to proof."""

import hashlib
import json
import subprocess
import sys

import pytest

from _checks import actions, experiment, host_io, query, suggest, task_context
from _checks.paths import SKILL_ROOT
from .test_experiment import observation, write
from .test_suggest import GROUP, source


def test_contract_resolution_allows_a_conditional_experiment_without_execution():
    result = query.report("acc-final-output", goal="select_experiment")
    item = result["action_guidance"]
    assert (
        item["next_action"] == "form_one_hypothesis"
        and item["readiness"] == "conditional"
    )
    assert item["selected_table"]["status"] == "observed_source"
    assert (
        item["later_checks"]
        and result["validation"]["performance"] == "not_established"
    )


@pytest.mark.parametrize(
    "goal,action",
    [
        ("native_probe", "native_probe"),
        ("validate_numerics", "numerical_evaluation"),
        ("compare_performance", "paired_complete_call_timing"),
    ],
)
def test_execution_unknowns_select_an_action_not_more_declaration_reads(goal, action):
    result = query.report("acc-final-output", goal=goal)
    assert result["action_guidance"]["next_action"] == action
    assert result["action_guidance"]["readiness"] == "needs_execution_evidence"
    assert (
        result["validation"]["numerical"]
        == result["validation"]["performance"]
        == "not_established"
    )


def test_unknown_contract_and_unbound_declaration_still_need_applicability():
    for text in ("not-a-covered-scenario", "pl.store"):
        item = query.report(text, goal="select_experiment")["action_guidance"]
        assert item["next_action"] == "resolve_one_contract_question"
        assert item["readiness"] == "needs_applicability_fact"


def test_unknown_source_table_is_not_a_positive_capability(tmp_path):
    item = query.report(
        "acc-final-output", pypto_root=tmp_path, goal="select_experiment"
    )["action_guidance"]
    assert item["next_action"] == "resolve_selected_table_or_probe"


def test_source_failure_takes_precedence_over_a_candidate():
    result = {"status": "failed", "context": [{"id": "ub-l1"}], "diagnostics": []}
    actions.for_report(result, "select_experiment")
    assert result["action_guidance"]["next_action"] == "resolve_reported_source_errors"


def test_candidate_advice_uses_existing_conditions_and_keeps_stale_costs_unknown(
    tmp_path,
):
    path = source(tmp_path, GROUP)
    result = suggest.report([path], goal="select_experiment")
    assert result["action_guidance"]["next_action"] == "form_one_hypothesis"
    result = suggest.report(
        [path], measurements=tmp_path / "missing.json", goal="select_experiment"
    )
    assert result["status"] == "unverified"
    assert result["action_guidance"]["next_action"] == "recapture_source_bound_costs"


def test_no_suggestion_is_not_a_request_for_more_api_documentation(tmp_path):
    path = source(tmp_path, "value = 1\n")
    item = suggest.report([path], goal="select_experiment")["action_guidance"]
    assert item["next_action"] == "inspect_current_costs_and_workload"
    assert (
        suggest.report([path], goal="validate_numerics")["action_guidance"][
            "next_action"
        ]
        == "numerical_evaluation"
    )


def test_exact_declaration_can_proceed_to_a_requested_probe():
    result = query.report("pl.store", goal="native_probe")
    assert result["action_guidance"]["next_action"] == "native_probe"
    assert result["action_guidance"]["lookup"] == "declaration_only"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"text": "contract:acc-final-output"},
        {"text": "acc-final-output", "kinds": ["contract"]},
    ],
)
def test_selected_knowledge_contract_reuses_action_guidance(kwargs):
    result = query.report(**kwargs, goal="select_experiment")
    assert not result["context"] and result["knowledge"]["items"][0]["text"]
    assert result["action_guidance"]["lookup"] == "scenario_found"
    assert result["action_guidance"]["next_action"] == "form_one_hypothesis"
    assert result["action_guidance"]["owners"] == ["acc-final-output"]


def test_typed_declaration_and_discovery_keep_different_readiness():
    result = query.report("pl.DT_FP16", goal="native_probe")
    assert (
        not result["signatures"]
        and result["action_guidance"]["lookup"] == "declaration_only"
    )
    assert result["action_guidance"]["readiness"] == "needs_execution_evidence"
    result = query.report("move", kinds=["contract"], goal="select_experiment")
    assert result["action_guidance"]["readiness"] == "needs_applicability_fact"


def test_declared_goal_is_optional_and_stale_inventory_blocks_use(tmp_path):
    path = tmp_path / "candidate.py"
    path.write_text("value = 1\n")
    manifest = write(
        tmp_path / "task.json",
        {
            "schema": "pypto-task-context/1",
            "goal": "select_experiment",
            "artifacts": [
                {
                    "path": path.name,
                    "role": "source",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            ],
        },
    )
    config = host_io.policy(workspace=tmp_path)
    result = task_context.report(manifest, config=config)
    assert result["action_guidance"]["next_action"] == "assess_supplied_applicability"
    assert (
        len((host_io.encode(result) + "\n").encode())
        == result["budget"]["result_bytes"]
    )
    with pytest.raises(ValueError, match="differs"):
        task_context.report(manifest, config=config, goal="native_probe")
    path.write_text("changed = True\n")
    result = task_context.report(manifest, config=config)
    assert (
        result["status"] == "unverified"
        and result["action_guidance"]["readiness"] == "inputs_incomplete"
    )


def test_experiment_projection_follows_existing_phase_and_is_not_written(trial):
    original = trial["path"].read_bytes()
    result = experiment.run(trial["path"])
    assert result["action_guidance"]["goal"] == "validate_numerics"
    assert trial[
        "path"
    ].read_bytes() == original and "action_guidance" not in json.loads(original)
    result = experiment.run(
        trial["path"],
        observation=write(trial["tmp"] / "observed.json", observation(trial)),
    )
    assert (
        result["phase"] == "ready"
        and result["action_guidance"]["next_action"]
        == "select_next_hypothesis_from_best"
    )
    result["phase"] = "awaiting_timing"
    assert actions.for_experiment(result)["goal"] == "compare_performance"
    result.update(phase="inconclusive", trials=result["max_trials"])
    assert (
        actions.for_experiment(result)["next_action"]
        == "export_verified_best_and_report_open_directions"
    )


def test_cli_goal_does_not_execute_and_unknown_goal_is_rejected():
    command = [
        sys.executable,
        str(SKILL_ROOT / "scripts/check.py"),
        "query",
        "acc-final-output",
        "--format",
        "json",
    ]
    run = subprocess.run(
        command + ["--goal", "select_experiment"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert (
        json.loads(run.stdout)["action_guidance"]["next_action"]
        == "form_one_hypothesis"
    )
    assert (
        subprocess.run(
            command + ["--goal", "skip_validation"], capture_output=True
        ).returncode
        == 2
    )
