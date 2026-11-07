# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A first response and an assignee are independently auditable obligations."""

import importlib.util
from pathlib import Path

import pytest

from test_generate_summary_report import complete_state as _complete_state

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/generate_summary_report.py"
SPEC = importlib.util.spec_from_file_location("response_assignment_report", SCRIPT)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def complete_state():
    state = _complete_state()
    state["run"]["issues_listed_total"] = 0
    return state


@pytest.mark.parametrize("field", ["response_status", "assignment_status"])
def test_missing_obligation_is_rejected_even_if_issue_has_a_reply(field):
    state = complete_state()
    state["issues"][0].pop(field)
    assert any(field in error for error in REPORT.validate_state(state))


@pytest.mark.parametrize("field", ["response_status", "assignment_status"])
@pytest.mark.parametrize("value", [[], {}, "made_up_status"])
def test_malformed_status_returns_errors_instead_of_crashing(field, value):
    state = complete_state()
    state["issues"][0][field] = value
    assert any(field in error for error in REPORT.validate_state(state))


def test_reply_without_assignment_is_visible_and_not_complete():
    state = complete_state()
    item = state["issues"][0]
    item.update(
        assignment_status="not_started",
        assignment_reason="按平台规则转交，待执行",
        handling_status="waiting_for_approval",
    )
    item.pop("assignee")
    assert not REPORT.validate_state(state)
    assert "责任人待指派/回查" in "\n".join(
        REPORT.render_results_table(state["issues"])
    )
    item["handling_status"] = "delegated"
    assert any("still pending" in error for error in REPORT.validate_state(state))


def test_assigned_without_reply_is_visible_and_not_resolved():
    state = complete_state()
    item = state["issues"][0]
    item.update(response_status="prepared", handling_status="waiting_for_approval")
    item.pop("response_comment_id")
    assert not REPORT.validate_state(state)
    assert "首响待发送/回查" in REPORT.result_summary_text(item)
    item["resolution_status"] = "resolved"
    assert any("still pending" in error for error in REPORT.validate_state(state))


def test_assignment_success_requires_assignee_readback_and_matching_target():
    state = complete_state()
    item = state["issues"][0]
    item.pop("assignee")
    assert any("GET-confirmed" in error for error in REPORT.validate_state(state))
    item["assignee"] = "different-owner"
    assert any("differs" in error for error in REPORT.validate_state(state))
    item["assignee"] = "OWNER-A"
    assert not REPORT.validate_state(state)


def test_claiming_a_response_requires_its_comment_reference():
    state = complete_state()
    item = state["issues"][0]
    item.pop("response_comment_id")
    item.pop("comments", None)
    assert any("response_comment_id" in error for error in REPORT.validate_state(state))


def test_pending_unknown_owner_requires_reason_and_stays_actionable():
    state = complete_state()
    item = state["issues"][1]
    item.pop("assignment_reason")
    assert any("assignment_reason" in error for error in REPORT.validate_state(state))
    item["assignment_reason"] = "未命中负责人规则，等待用户确认；恢复原指派步骤"
    assert not REPORT.validate_state(state)
    assert "责任人待指派/回查" in REPORT.result_summary_text(item)


def test_required_handoff_cannot_be_hidden_as_not_applicable():
    state = complete_state()
    item = state["issues"][1]
    item.update(
        assignment_status="not_applicable",
        assignment_reason="尚未找到实际 kernel",
        operator_handling_decision="delegate",
    )
    assert any("required delegation" in error for error in REPORT.validate_state(state))


def test_answered_consultation_still_needs_a_platform_owner():
    state = complete_state()
    item = state["issues"][1]
    item.update(
        assignment_status="not_applicable",
        assignment_reason="纯答疑已完整结束，无剩余维护动作",
    )
    assert not REPORT.validate_state(state)
    assert "待指派" in REPORT.result_summary_text(item)
    item["handling_status"] = "completed"
    assert any("still pending" in error for error in REPORT.validate_state(state))


def test_existing_owner_is_preserved_while_first_response_waits_for_approval():
    from pipeline_delivery import _prepared_issue

    entry = {
        "issue": {"assignee": "existing-owner"},
        "response_file_map": {},
        "related_code": [],
    }
    result = {
        "summary": "已准备回复",
        "next_action": "等待准确授权",
        "owner_review": {"status": "not_needed", "reason": "已有负责人"},
        "response_review": {},
        "evidence": ["GET Issue assignee"],
    }
    prepared = _prepared_issue(entry, {}, result, 1)
    assert prepared["assignee"] == "existing-owner"
    assert prepared["assignment_status"] == "verified"
    assert prepared["response_status"] == "prepared"


def test_assignment_plan_is_pending_even_when_candidate_investigation_is_unneeded():
    from pipeline_delivery import _prepared_issue

    entry = {"issue": {}, "response_file_map": {}, "related_code": []}
    result = {
        "summary": "已准备回复和分配稿",
        "next_action": "先回复回查再指派",
        "assignment": {"login": "chen_0715", "reason": "已明确的接单人"},
        "owner_review": {"status": "not_needed", "reason": "无需候选调查"},
        "response_review": {},
        "evidence": ["责任规则"],
    }
    prepared = _prepared_issue(entry, {}, result, 1)
    assert prepared["assignment_status"] == "not_started"
    assert prepared["assignment_plan"]["login"] == "chen_0715"
    assert "责任人待指派/回查" in REPORT.result_summary_text(prepared)


def test_original_attention_list_detects_an_entirely_omitted_issue():
    state = complete_state()
    classification = {
        "issues": [
            {"number": 101, "responsibility": "handle", "bucket": "need_attention"},
            {"number": 3747, "responsibility": "handle", "bucket": "need_attention"},
            {"number": 999, "responsibility": "list-only", "bucket": "need_attention"},
            {"number": 998, "responsibility": "ignore", "bucket": "no_attention"},
        ]
    }
    errors = REPORT.validate_classification_coverage(state, classification)
    assert errors == ["classification: missing handle/need_attention Issues: 3747"]
    classification["issues"][1]["number"] = 102
    assert not REPORT.validate_classification_coverage(state, classification)


def test_original_attention_list_checks_observations_and_deduplicates():
    source = {"number": 3747, "responsibility": "handle", "bucket": "need_attention"}
    errors = REPORT.validate_classification_coverage(
        complete_state(), {"issues": [source], "observations": [source]}
    )
    assert errors == ["classification: missing handle/need_attention Issues: 3747"]


@pytest.mark.parametrize(
    "snapshot", [{}, {"issues": [None]}, {"issues": [], "observations": None}]
)
def test_invalid_classification_evidence_cannot_silently_disable_coverage(snapshot):
    with pytest.raises(ValueError, match="classification"):
        REPORT.validate_classification_coverage(complete_state(), snapshot)


def test_final_report_cli_refuses_a_missing_issue(tmp_path):
    import json
    from test_generate_summary_report import run_report

    source = tmp_path / "classified.json"
    source.write_text(
        json.dumps(
            {
                "issues": [
                    {
                        "number": 3747,
                        "responsibility": "handle",
                        "bucket": "need_attention",
                    }
                ]
            }
        )
    )
    result = run_report(tmp_path, complete_state(), "--classification", str(source))
    assert result.returncode == 2
    assert "missing handle/need_attention Issues: 3747" in result.stderr


def test_final_report_cli_accepts_full_coverage_and_displays_pending_assignment(
    tmp_path,
):
    import json
    from test_generate_summary_report import run_report

    source = tmp_path / "classified.json"
    source.write_text(
        json.dumps(
            {
                "issues": [
                    {
                        "number": n,
                        "responsibility": "handle",
                        "bucket": "need_attention",
                    }
                    for n in [101, 102]
                ]
            }
        )
    )
    result = run_report(tmp_path, complete_state(), "--classification", str(source))
    assert result.returncode == 0, result.stderr
    report = (tmp_path / json.loads(result.stdout)["report_path"]).read_text()
    assert "责任人待指派/回查" in report.split("<details>", 1)[0]


def test_existing_response_evidence_is_preserved_for_assignment_only_work():
    from pipeline_delivery import _prepared_issue

    entry = {"issue": {}, "response_file_map": {}, "related_code": []}
    classification = {
        "latest_maintainer_comment_id": "123",
        "auto_action": {
            "type": "assign_candidate",
            "response_requirement": "satisfied",
        },
    }
    result = {
        "summary": "补齐指派",
        "next_action": "授权后指派",
        "assignment": {"login": "owner", "reason": "PR 作者"},
        "owner_review": {"status": "not_needed", "reason": "已有 PR 作者"},
        "response_review": {},
        "evidence": ["已有实质回复"],
    }
    prepared = _prepared_issue(entry, classification, result, 1)
    assert prepared["response_status"] == "not_applicable_existing"
    assert prepared["response_comment_id"] == "123"
    assert "首响待发送" not in REPORT.result_summary_text(prepared)
    assert "责任人待指派" in REPORT.result_summary_text(prepared)


@pytest.mark.parametrize(
    "level,bucket", [("handle", "need_attention"), ("list-only", "no_attention")]
)
def test_reviewed_missing_owner_remains_actionable_without_kernel_inspection(
    level, bucket
):
    from test_classify_issues import (
        CLASSIFIER as C,
        classification_options,
        comment,
        issue,
    )

    policy = {
        "handle": ["所有非规划 open Issue；明确非 A5 常规算子交给 chen_0715"],
        "list-only": ["范围外仅列举"],
        "ignore": [],
    }
    item = issue(
        number=3747,
        title="A2 Tril 输出异常",
        comments=[comment()],
        operator_routing_required=True,
    )
    item["responsibility_review"] = {
        "level": level,
        "summary": "已核查接单范围；实际 kernel 包待确认不影响接单规则",
        "evidence": ["提出者报告 A2，问题对象为常规 Tril"],
        "policy_digest": C.policy_digest(policy),
    }
    classified = C.classify_with_responsibility(item, classification_options(), policy)
    assert classified["bucket"] == bucket
    assert classified["category"] == "operator_routing_required"
    assert classified["classification_bucket"] == "need_attention"
    assert C.analyze_conversation(item)["latest_maintainer_comment_id"] == "1"


def test_replied_issues_require_owner_even_without_operator_routing_flag():
    from test_classify_issues import (
        CLASSIFIER as C,
        classification_options,
        comment,
        issue,
    )

    policy = {"handle": ["具体问题"], "list-only": [], "ignore": []}
    for assignee, flag, bucket, category in [
        (None, False, "need_attention", "replied_no_owner"),
        ("existing", True, "no_attention", "our_team_replied"),
    ]:
        item = issue(
            assignee=assignee, comments=[comment()], operator_routing_required=flag
        )
        item["responsibility_review"] = {
            "level": "handle",
            "summary": "已核查范围",
            "evidence": ["已知问题事实"],
            "policy_digest": C.policy_digest(policy),
        }
        result = C.classify_with_responsibility(item, classification_options(), policy)
        assert (result["bucket"], result["category"]) == (bucket, category)


def test_reviewed_operator_handoff_does_not_disappear_as_owner_not_needed():
    from pipeline_delivery import _prepared_issue

    entry = {"issue": {}, "response_file_map": {}, "related_code": []}
    classification = {
        "category": "operator_routing_required",
        "latest_maintainer_comment_id": "123",
    }
    result = {
        "summary": "首响已完成，补责任人指派",
        "next_action": "按已确认规则转交",
        "owner_review": {
            "status": "not_needed",
            "reason": "规则已给接单人，无需候选调查",
        },
        "response_review": {},
        "evidence": ["已回复和责任规则"],
    }
    prepared = _prepared_issue(entry, classification, result, 1)
    assert prepared["assignment_status"] == "not_started"
    assert prepared["response_status"] == "not_applicable_existing"
    assert "首响待发送" not in REPORT.result_summary_text(prepared)
    assert "责任人待指派" in REPORT.result_summary_text(prepared)


@pytest.mark.parametrize(
    "response",
    ["pending", "prepared", "failed", "waived_by_user"],
)
@pytest.mark.parametrize("terminal", ["completed", "delegated", "resolved"])
def test_pending_response_or_user_waiver_cannot_satisfy_issue_completion(
    response, terminal
):
    state = complete_state()
    state["issues"][0].update(response_status=response, handling_status=terminal)
    assert any("still pending" in error for error in REPORT.validate_state(state))


@pytest.mark.parametrize("assignment", ["not_started", "failed", "not_applicable"])
def test_completed_batch_cannot_hide_missing_assignment_behind_waiting_issue(
    assignment,
):
    state = complete_state()
    state["run"]["overall_status"] = "completed"
    state["issues"][1].update(assignment_status=assignment)
    assert any("still pending" in error for error in REPORT.validate_state(state))


def test_completed_batch_requires_both_proofs_for_each_handled_issue():
    state = complete_state()
    state["run"]["overall_status"] = "completed"
    for item in state["issues"]:
        item.update(
            handling_status="completed",
            assignment_status="verified",
            assignee=item.get("operator_owner") or "owner-b",
        )
    assert not REPORT.validate_state(state)
    # Explicitly handled no_attention entries must be checked as well.
    item.update(
        bucket="no_attention", handled_in_run=True, response_status="waived_by_user"
    )
    assert any("still pending" in error for error in REPORT.validate_state(state))


def test_completed_batch_cli_requires_full_input_classification(tmp_path):
    from test_generate_summary_report import run_report

    state = complete_state()
    state["run"]["overall_status"] = "completed"
    for item in state["issues"]:
        item.update(
            assignment_status="verified",
            assignee=item.get("operator_owner") or "owner-b",
        )
    result = run_report(tmp_path, state)
    assert result.returncode == 2
    assert "completed run requires --strict --classification" in result.stderr


def test_completed_batch_cli_accepts_all_covered_and_verified_issues(tmp_path):
    import json
    from test_generate_summary_report import run_report

    state = complete_state()
    state["run"]["overall_status"] = "completed"
    for item in state["issues"]:
        item.update(
            assignment_status="verified",
            assignee=item.get("operator_owner") or "owner-b",
        )
    snapshot = tmp_path / "classification.json"
    snapshot.write_text(
        json.dumps(
            {
                "issues": [
                    {
                        "number": item["iid"],
                        "bucket": "need_attention",
                        "responsibility": "handle",
                    }
                    for item in state["issues"]
                ]
            }
        )
    )
    result = run_report(tmp_path, state, "--classification", str(snapshot))
    assert result.returncode == 0, result.stderr
    report = (tmp_path / json.loads(result.stdout)["report_path"]).read_text()
    assert "责任人待指派" not in report.split("<details>", 1)[0]
    assert "| owner-b |" in report


def test_owner_investigation_not_needed_does_not_waive_platform_assignment():
    from pipeline_delivery import _prepared_issue

    entry = {"issue": {}, "response_file_map": {}, "related_code": []}
    result = {
        "summary": "答疑稿已就绪",
        "next_action": "等待回复及负责人授权",
        "owner_review": {"status": "not_needed", "reason": "纯咨询无需调查贡献候选"},
        "response_review": {},
        "evidence": ["Issue 原文"],
    }
    item = _prepared_issue(entry, {}, result, 1)
    assert item["assignment_status"] == "not_started"
    assert "责任人待指派" in REPORT.result_summary_text(item)


def test_self_authored_exemption_skips_reply_preparation():
    from pipeline_delivery import assignment_only

    assert assignment_only(
        {
            "auto_action": {
                "type": "assign_candidate",
                "response_requirement": "exempt_self_authored_pr",
            }
        }
    )
    assert assignment_only(
        {
            "auto_action": {
                "type": "assign_candidate",
                "response_requirement": "satisfied",
            }
        }
    )


@pytest.mark.parametrize("category", ["replied_no_owner", "operator_routing_required"])
def test_existing_reply_with_configured_owner_needs_no_kernel_inspection(
    tmp_path, category
):
    from pipeline_delivery import response_template, validate_response, MaterialError

    classification = {"category": category, "latest_maintainer_comment_id": "123"}
    result = response_template(classification)
    result.update(
        analysis="A2 Tril 按当前非A5常规算子规则由 chen_0715 接单；kernel 包来源待查不影响接单",
        next_action="按已有准确授权指派并回查",
    )
    result["assignment"].update(
        login="chen_0715",
        reason="匹配非A5常规算子责任规则",
        evidence=["当前责任配置与提出者报告 A2，常规 Tril"],
    )
    result["response_review"]["checked_facts"] = [
        "已有有效首响 comment123",
        "已核对当前责任规则",
    ]
    task = {"payload": {"classification": classification}}
    assert validate_response({}, task, result, tmp_path) == []
    assert result["reply"] == ""
    assert result["inspection_files"] == []
    result["assignment"]["evidence"] = []
    with pytest.raises(MaterialError, match="response_materials_incomplete"):
        validate_response({}, task, result, tmp_path)


def test_owner_only_routing_requires_real_comment_evidence():
    from pipeline_delivery import assignment_only

    assert not assignment_only({"category": "replied_no_owner"})


def test_owner_only_routing_loads_responsibility_rules_without_reply_writing():
    from issue_pipeline import task_references

    task = {
        "stage": "response",
        "payload": {
            "classification": {
                "category": "replied_no_owner",
                "latest_maintainer_comment_id": "123",
            }
        },
    }
    assert [Path(p).name for p in task_references(task)] == [
        "automation.md",
        "operator-handoff.md",
    ]


def test_rule_handoff_materializes_assignment_and_reports_both_obligations(tmp_path):
    from pipeline_delivery import (
        response_template,
        validate_response,
        materialize,
        _prepared_issue,
    )
    from pipeline_store import PipelineStore

    classification = {
        "category": "replied_no_owner",
        "latest_maintainer_comment_id": "123",
    }
    result = response_template(classification)
    result.update(
        summary="首响已完成，待执行规则指派",
        analysis="A2 常规 Tril 按当前责任规则交 chen_0715，实际 kernel 来源另行核实",
        next_action="授权后指派并 GET 核验 assignee 和 pending-confirmation",
    )
    result["assignment"].update(
        login="chen_0715",
        reason="匹配当前责任配置的非A5常规算子规则",
        evidence=["责任配置规则与原文 A2/Tril"],
    )
    result["response_review"]["checked_facts"] = ["首响评论123有效", "已匹配责任规则"]
    records = validate_response(
        {}, {"payload": {"classification": classification}}, result, tmp_path
    )
    with PipelineStore(
        tmp_path / ".cannbot/gitcode-issue-handler", "cann/ops-math"
    ) as store:
        files, related = materialize(
            result, records, store, Path("reports/run/issues/issue-3747")
        )
    assert "reply.md" not in files
    assert (tmp_path / files["assign.md"]).read_text() == "/assign @chen_0715\n"
    entry = {
        "issue": {"iid": 3747},
        "response_file_map": files,
        "related_code": related,
    }
    prepared = _prepared_issue(entry, classification, result, 1)
    assert prepared["response_comment_id"] == "123"
    assert prepared["assignment_plan"]["login"] == "chen_0715"
    assert prepared["assignment_status"] == "not_started"
    assert "责任人待指派" in REPORT.result_summary_text(prepared)
    assert "首响待发送" not in REPORT.result_summary_text(prepared)


@pytest.mark.parametrize("with_identity", [False, True])
def test_self_authored_completion_requires_identity_evidence(with_identity):
    state = complete_state()
    item = state["issues"][0]
    item.update(response_status="exempt_self_authored_pr", handling_status="completed")
    item.pop("response_comment_id", None)
    item.pop("comments", None)
    if with_identity:
        item["linked_prs"] = [{"pr_state": "open", "pr_author": item["author"]}]
        assert not REPORT.validate_state(state)
        assert "首响待发送" not in REPORT.result_summary_text(item)
    else:
        assert any(
            "self-authored exemption" in error for error in REPORT.validate_state(state)
        )
    item.update(assignment_status="not_started", assignment_reason="等待指派")
    assert any("still pending" in error for error in REPORT.validate_state(state))
