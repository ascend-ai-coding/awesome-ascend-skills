# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Evaluation separates actual observations, missing telemetry and incompatible trials."""

import json

import pytest

from _checks import evaluation


def observation(variant, **extra):
    return {
        "task_id": "repair",
        "task_sha256": "1" * 64,
        "model": "specified-model",
        "reasoning": "max",
        "budget": {"tool_calls": 10},
        "toolchain": {"source": "fixed"},
        "repeat": 0,
        "variant": variant,
        "skill_sha256": "2" * 64,
        "source_sha256": "3" * 64,
        **extra,
    }


def compare(tmp_path, rows):
    path = tmp_path / "observations.json"
    path.write_text(json.dumps(rows))
    return evaluation.compare_observations(path)


def test_missing_metrics_remain_missing(tmp_path):
    result = compare(
        tmp_path,
        [
            observation("baseline", correct=False),
            observation("candidate", correct=True),
        ],
    )
    pair = result["pairs"][0]
    assert pair["matched"]
    assert pair["observations"]["context_tokens"] == {
        "baseline": None,
        "candidate": None,
    }
    assert pair["observations"]["correct"] == {"baseline": False, "candidate": True}
    assert pair["cost_comparison"]["status"] == "ineligible"


def test_budget_failure_cannot_be_reported_as_a_context_gain(tmp_path):
    gates = {"correct": True, "valid_protocol": True, "telemetry_status": "complete"}
    result = compare(
        tmp_path,
        [
            observation("baseline", **gates, budget_met=False, input_tokens=1000),
            observation("candidate", **gates, budget_met=True, input_tokens=100),
        ],
    )
    pair = result["pairs"][0]
    assert pair["matched"]
    assert pair["observations"]["input_tokens"] == {"baseline": 1000, "candidate": 100}
    assert pair["cost_comparison"]["status"] == "ineligible"
    assert "baseline.budget_met=false" in pair["cost_comparison"]["reasons"]


@pytest.mark.parametrize(
    "missing", ["correct", "budget_met", "valid_protocol", "telemetry_status"]
)
def test_missing_acceptance_or_telemetry_keeps_cost_comparison_unverified(
    tmp_path, missing
):
    gates = {
        "correct": True,
        "budget_met": True,
        "valid_protocol": True,
        "telemetry_status": "complete",
    }
    gates.pop(missing)
    result = compare(
        tmp_path, [observation("baseline", **gates), observation("candidate", **gates)]
    )
    assert result["pairs"][0]["cost_comparison"]["status"] == "unverified"


def test_direction_failure_excludes_an_otherwise_completed_trial(tmp_path):
    gates = {
        "correct": True,
        "budget_met": True,
        "valid_protocol": True,
        "telemetry_status": "complete",
    }
    result = compare(
        tmp_path,
        [
            observation("baseline", **gates, direction_correct=True),
            observation("candidate", **gates, direction_correct=False),
        ],
    )
    assert result["pairs"][0]["cost_comparison"]["status"] == "ineligible"


@pytest.mark.parametrize("status", ["incomplete", "invalid"])
def test_partial_token_totals_are_not_eligible_even_when_supplied(tmp_path, status):
    gates = {
        "correct": True,
        "budget_met": True,
        "valid_protocol": True,
        "input_tokens": 100,
    }
    result = compare(
        tmp_path,
        [
            observation("baseline", **gates, telemetry_status="complete"),
            observation("candidate", **gates, telemetry_status=status),
        ],
    )
    assert result["pairs"][0]["cost_comparison"]["status"] == "unverified"


def test_eligible_tokens_do_not_invent_unexposed_tool_or_peak_counts(tmp_path):
    gates = {
        "correct": True,
        "budget_met": True,
        "valid_protocol": True,
        "telemetry_status": "complete",
    }
    result = compare(
        tmp_path,
        [
            observation("baseline", **gates, input_tokens=1000),
            observation("candidate", **gates, input_tokens=800),
        ],
    )
    eligibility = result["pairs"][0]["cost_comparison"]
    assert eligibility["status"] == "eligible"
    assert "input_tokens" in eligibility["available_metrics"]
    assert not {"tool_calls", "context_tokens"}.intersection(
        eligibility["available_metrics"]
    )


@pytest.mark.parametrize("status", ["estimated", 1, [], {}])
def test_invalid_telemetry_state_is_rejected(tmp_path, status):
    with pytest.raises(ValueError, match="telemetry_status"):
        compare(tmp_path, [observation("baseline", telemetry_status=status)])


@pytest.mark.parametrize(
    "difference",
    [
        {"model": "other"},
        {"budget": {"tool_calls": 20}},
        {"toolchain": {"source": "other"}},
        {"task_sha256": "4" * 64},
        {"repeat": 1},
    ],
)
def test_incompatible_observations_do_not_form_a_pair(tmp_path, difference):
    result = compare(
        tmp_path, [observation("baseline"), observation("candidate", **difference)]
    )
    assert len(result["pairs"]) == 2
    assert not any(p["matched"] for p in result["pairs"])


@pytest.mark.parametrize(
    "bad",
    [
        {"context_tokens": -1},
        {"correct": "passed"},
        {"latency_us": float("nan")},
        {"skill_sha256": "unknown"},
    ],
)
def test_invalid_telemetry_is_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        compare(tmp_path, [observation("baseline", **bad)])


def test_duplicate_trial_does_not_inflate_results(tmp_path):
    with pytest.raises(ValueError):
        compare(tmp_path, [observation("baseline"), observation("baseline")])


def test_task_preparation_is_reproducible_and_never_overwrites(tmp_path):
    a = evaluation.prepare_agent_task(tmp_path / "a")
    b = evaluation.prepare_agent_task(tmp_path / "b")
    assert a["task_sha256"] == b["task_sha256"]
    assert (tmp_path / "a/candidate.py").read_bytes() == (
        tmp_path / "b/candidate.py"
    ).read_bytes()
    with pytest.raises(FileExistsError):
        evaluation.prepare_agent_task(tmp_path / "a")


def events(tmp_path, records, tail=""):
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n" + tail)
    return evaluation.event_metrics(path)


def item(ident, kind, **extra):
    return {"id": ident, "type": kind, **extra}


def test_event_metrics_count_tool_attempts_and_edits_once(tmp_path):
    command = item("a", "command_execution", aggregated_output="结果\n", exit_code=1)
    edit = item("b", "file_change", status="completed")
    result = events(
        tmp_path,
        [
            {"type": "turn.started"},
            {"type": "item.started", "item": item("a", "command_execution")},
            {"type": "item.completed", "item": command},
            {"type": "item.completed", "item": command},
            {"type": "item.completed", "item": edit},
            {"type": "item.completed", "item": item("c", "agent_message", text="done")},
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 1000,
                    "cached_input_tokens": 700,
                    "output_tokens": 80,
                    "reasoning_output_tokens": 50,
                },
            },
        ],
    )
    assert result["status"] == "complete"
    metrics = result["metrics"]
    assert metrics["visible_actions"] == 2 and metrics["tool_calls"] is None
    assert metrics["shell_calls"] == 1 and metrics["file_edits"] == 1
    assert metrics["uncached_input_tokens"] == 300 and metrics["context_tokens"] is None
    assert metrics["command_output_bytes"] == len("结果\n".encode())


def test_pending_tool_and_failed_turn_do_not_look_like_a_cheap_complete_run(tmp_path):
    result = events(
        tmp_path,
        [
            {"type": "turn.started"},
            {"type": "item.started", "item": item("a", "mcp_tool_call")},
            {"type": "turn.failed", "error": "connection lost"},
        ],
    )
    assert result["status"] == "incomplete"
    assert (
        result["metrics"]["visible_actions"] == 1
        and result["metrics"]["pending_tools"] == 1
    )
    assert result["metrics"]["input_tokens"] is None


def test_incomplete_final_line_keeps_usage_unknown(tmp_path):
    result = events(tmp_path, [{"type": "turn.started"}], tail='{"type": "turn.com')
    assert result["status"] == "incomplete" and result["incomplete_event_lines"] == 1
    assert result["metrics"]["input_tokens"] is None


def test_invalid_json_before_final_line_preserves_decode_cause(tmp_path):
    with pytest.raises(ValueError, match="Invalid JSON before") as rejected:
        events(tmp_path, [], tail='invalid\n{"type": "turn.started"}')
    assert isinstance(rejected.value.__cause__, json.JSONDecodeError)


def test_multiple_turns_sum_actual_usage_and_preserve_missing_fields(tmp_path):
    result = events(
        tmp_path,
        [
            {"type": "turn.started"},
            {"type": "turn.completed", "usage": {"input_tokens": 100}},
            {"type": "turn.started"},
            {"type": "turn.completed", "usage": {"input_tokens": 200}},
        ],
    )
    assert result["metrics"]["input_tokens"] == 300
    assert result["metrics"]["cached_input_tokens"] is None
    assert result["metrics"]["uncached_input_tokens"] is None


def test_a_later_failed_turn_invalidates_complete_run_token_totals(tmp_path):
    result = events(
        tmp_path,
        [
            {"type": "turn.started"},
            {"type": "turn.completed", "usage": {"input_tokens": 100}},
            {"type": "turn.started"},
            {"type": "turn.failed"},
        ],
    )
    assert (
        result["status"] == "incomplete" and result["metrics"]["input_tokens"] is None
    )


def test_unknown_host_tool_type_does_not_silently_count_as_zero(tmp_path):
    result = events(
        tmp_path, [{"type": "item.completed", "item": item("a", "new_host_tool")}]
    )
    assert result["metrics"]["visible_actions"] is None
    assert result["unknown_item_types"] == ["new_host_tool"]


@pytest.mark.parametrize(
    "usage",
    [
        {"input_tokens": -1},
        {"input_tokens": True},
        {"input_tokens": 10, "cached_input_tokens": 11},
    ],
)
def test_invalid_host_usage_is_rejected(tmp_path, usage):
    with pytest.raises(ValueError):
        events(
            tmp_path,
            [{"type": "turn.started"}, {"type": "turn.completed", "usage": usage}],
        )


HOOK_NOTICE = "`--dangerously-bypass-hook-trust` is enabled. Enabled hooks may run without review for this invocation."
FEATURE_NOTICE = (
    "Under-development features enabled: skip_host_skill_discovery. "
    "Under-development features are incomplete and may behave unpredictably. "
    "To suppress this warning, set `suppress_unstable_features_warning = true` in /home/user/.codex/config.toml."
)


def test_feature_startup_notice_preserves_mcp_actions_and_token_usage(tmp_path):
    result = events(
        tmp_path,
        [
            {
                "type": "item.completed",
                "item": item("item_0", "error", message=FEATURE_NOTICE),
            },
            {"type": "turn.started"},
            {"type": "item.started", "item": item("call", "mcp_tool_call")},
            {"type": "item.completed", "item": item("call", "mcp_tool_call")},
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 50,
                    "output_tokens": 10,
                    "reasoning_output_tokens": 5,
                },
            },
        ],
    )
    assert result["status"] == "complete"
    assert result["metrics"]["visible_actions"] == result["metrics"]["mcp_calls"] == 1
    assert result["metrics"]["host_notices"] == 1
    assert result["metrics"]["input_tokens"] == 100
    assert result["host_attempt_coverage"] == "not_requested"
    assert result["unknown_item_types"] == []


@pytest.mark.parametrize(
    "message",
    [
        None,
        "Tool failed: " + FEATURE_NOTICE,
        FEATURE_NOTICE.replace("warning", "failure"),
    ],
)
def test_other_error_items_keep_action_accounting_unknown(tmp_path, message):
    result = events(
        tmp_path,
        [
            {"type": "item.completed", "item": item("error", "error", message=message)},
        ],
    )
    assert result["metrics"]["visible_actions"] is None
    assert result["metrics"]["host_notices"] == 0
    assert result["unknown_item_types"] == ["error"]


def test_feature_warning_after_a_tool_is_not_assumed_to_be_startup_metadata(tmp_path):
    result = events(
        tmp_path,
        [
            {"type": "item.completed", "item": item("call", "mcp_tool_call")},
            {
                "type": "item.completed",
                "item": item("error", "error", message=FEATURE_NOTICE),
            },
        ],
    )
    assert result["metrics"]["visible_actions"] is None
    assert result["unknown_item_types"] == ["error"]


def test_feature_notice_does_not_hide_missing_hook_attempt_counters(tmp_path):
    result = events(
        tmp_path,
        [
            {
                "type": "item.completed",
                "item": item("feature", "error", message=FEATURE_NOTICE),
            },
            {
                "type": "item.completed",
                "item": item("hook", "error", message=HOOK_NOTICE),
            },
            {"type": "turn.started"},
            {"type": "item.completed", "item": item("call", "mcp_tool_call")},
            {"type": "turn.completed", "usage": {"input_tokens": 100}},
        ],
    )
    assert result["metrics"]["host_notices"] == 2
    assert result["metrics"]["visible_actions"] is None
    assert result["host_attempt_coverage"] == "unverified_missing_counters"


def hook_events(tmp_path):
    records = [
        {"type": "thread.started", "thread_id": "session-one"},
        {
            "type": "item.completed",
            "item": item("notice", "error", message=HOOK_NOTICE),
        },
        {"type": "turn.started"},
        {
            "type": "item.completed",
            "item": item(
                "command", "command_execution", aggregated_output="ok", exit_code=0
            ),
        },
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 100,
                "cached_input_tokens": 50,
                "output_tokens": 10,
                "reasoning_output_tokens": 5,
            },
        },
    ]
    path = tmp_path / "hook-events.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records))
    return path


def test_hook_notice_without_attempt_counters_leaves_coverage_unknown(tmp_path):
    result = evaluation.event_metrics(hook_events(tmp_path))
    assert result["status"] == "incomplete"
    assert result["metrics"]["visible_actions"] is None
    assert result["metrics"]["input_tokens"] == 100
    assert result["host_attempt_coverage"] == "unverified_missing_counters"


def test_matching_counters_include_refusals_omitted_from_jsonl(tmp_path):
    path = hook_events(tmp_path)
    counters = tmp_path / "counters.json"
    counters.write_text(
        json.dumps(
            {
                "schema": "pypto-host-counters/1",
                "session_id": "session-one",
                "calls": 2,
                "denied": 1,
                "rewritten": 1,
            }
        )
    )
    result = evaluation.event_metrics(path, host_counters=counters)
    assert result["status"] == "complete"
    assert result["metrics"]["visible_actions"] == 2
    assert result["metrics"]["shell_calls"] == 1
    assert result["metrics"]["host_shell_attempts"] == 2
    assert result["metrics"]["host_read_refusals"] == 1
    assert result["metrics"]["tool_calls"] is None
    assert result["host_counters"]["sha256"]


@pytest.mark.parametrize(
    "change",
    [
        {"session_id": "other"},
        {"calls": 0},
        {"denied": 2},
        {"rewritten": -1},
        {"schema": "unknown"},
    ],
)
def test_inconsistent_host_counters_are_rejected(tmp_path, change):
    path = hook_events(tmp_path)
    counters = tmp_path / "counters.json"
    counters.write_text(
        json.dumps(
            {
                "schema": "pypto-host-counters/1",
                "session_id": "session-one",
                "calls": 1,
                "denied": 0,
                "rewritten": 0,
                **change,
            }
        )
    )
    with pytest.raises(ValueError, match="Host counters"):
        evaluation.event_metrics(path, host_counters=counters)
