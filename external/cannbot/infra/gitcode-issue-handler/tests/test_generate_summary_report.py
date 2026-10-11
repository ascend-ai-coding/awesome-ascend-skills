# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_summary_report.py"


COMPLETE_STATE = {
    "run": {
        "run_id": "run-20260811-001",
        "mode": "batch",
        "repository": "cann/ops-math",
        "started_at": "2026-08-11T09:00:00+08:00",
        "completed_at": "2026-08-11T11:00:00+08:00",
        "overall_status": "partial",
        "issues_total": 2,
        "base_ref": "upstream/master",
        "base_commit": "abc123",
        "delivery_mode": "pr",
        "target_remote_branch": "upstream/master",
    },
    "issues": [
        {
            "iid": 101,
            "url": "https://gitcode.com/cann/ops-math/issues/101",
            "title": "修复 Add 精度问题",
            "author": "reporter-a",
            "bucket": "need_attention",
            "category": "our_team_needs_work",
            "reason": "精度结果不一致",
            "problem_summary": "特定 shape 下输出错误",
            "handling_status": "published",
            "resolution_status": "resolution_pending",
            "resolution_metric_reason": "PR 尚未合入",
            "result_summary": "已修复并创建 PR，等待合入",
            "group_id": "g101",
            "operator_owner": "owner-a",
            "root_cause_hypothesis": "尾块处理错误",
            "final_root_cause": "尾块 mask 计算少一位",
            "solution_plan": "修正 mask 并补回归测试",
            "required_environment": {"soc": "Ascend910B"},
            "environment_check": {"status": "matched"},
            "reproduction_status": "stable",
            "reproduction_attempts": ["pytest repro: 3/3 failed before fix"],
            "process_log": [
                {
                    "time": "2026-08-11T09:10:00+08:00",
                    "stage": "triage",
                    "action": "分类并读取 Issue",
                    "result": "进入代码修复路径",
                    "evidence": ["issue #101"],
                },
                {
                    "time": "2026-08-11T09:30:00+08:00",
                    "stage": "diagnose",
                    "action": "定位尾块计算路径",
                    "result": "形成 mask 错误假设",
                    "evidence": ["src/add.cpp"],
                },
                {
                    "time": "2026-08-11T09:45:00+08:00",
                    "stage": "reproduce",
                    "action": "执行最小复现三次",
                    "result": "3/3 稳定失败",
                    "evidence": ["pytest tests/test_add.py"],
                },
                {
                    "time": "2026-08-11T10:00:00+08:00",
                    "stage": "implement",
                    "action": "修正 mask 并补测试",
                    "result": "完成最小代码变更",
                    "evidence": ["src/add.cpp", "tests/test_add.py"],
                },
                {
                    "time": "2026-08-11T10:20:00+08:00",
                    "stage": "validate",
                    "action": "运行相关测试",
                    "result": "测试通过",
                    "evidence": ["pytest tests/test_add.py"],
                },
                {
                    "time": "2026-08-11T10:30:00+08:00",
                    "stage": "deliver",
                    "action": "推送并创建 PR",
                    "result": "PR !88 已创建",
                    "evidence": ["https://gitcode.com/cann/ops-math/merge_requests/88"],
                },
            ],
            "comments": [{"status": "verified", "id": 9001}],
            "evidence": ["tests/test_add.py"],
            "blockers": [],
            "next_action": "owner-a 审核并合入 PR !88",
        },
        {
            "iid": 102,
            "url": "https://gitcode.com/cann/ops-math/issues/102",
            "title": "缺少复现日志",
            "author": "reporter-b",
            "bucket": "need_attention",
            "category": "needs_first_look",
            "reason": "上下文不足",
            "handling_status": "waiting_context",
            "resolution_status": "unresolved",
            "resolution_metric_reason": "等待提交者补充日志",
            "result_summary": "已回评索要最小复现信息",
            "process_log": [
                {
                    "time": "2026-08-11T09:15:00+08:00",
                    "stage": "triage",
                    "action": "分类并读取 Issue",
                    "result": "进入上下文检查",
                    "evidence": ["issue #102"],
                },
                {
                    "time": "2026-08-11T09:20:00+08:00",
                    "stage": "diagnose",
                    "action": "检查描述和评论",
                    "result": "缺少输入 shape 和错误日志",
                    "evidence": [],
                },
            ],
            "comments": [{"status": "verified", "id": 9002}],
            "blockers": ["等待提交者提供输入 shape 和完整日志"],
            "next_action": "收到补充后重新进入环境门禁",
        },
    ],
    "groups": [
        {
            "group_id": "g101",
            "members": [101],
            "theme": "Add 尾块修复",
            "lifecycle_status": "published",
            "branch": "fix/issue-101",
            "changed_files": ["src/add.cpp", "tests/test_add.py"],
            "tests": ["pytest tests/test_add.py: passed"],
            "validation_status": "passed",
            "commit_sha": "deadbeef",
            "pr_url": "https://gitcode.com/cann/ops-math/merge_requests/88",
            "ci_status": "running",
        }
    ],
    "metrics": {
        "resolution_rate": "0%（PR 未合入不计 resolved）",
        "first_response_sla_rate": "100%",
    },
    "internal_blockers": ["Issue #102 等待上下文"],
    "validation_boundaries": ["PR !88 CI 仍在运行"],
    "cleanup": {"cleaned_groups": ["g101"], "retained_worktrees": []},
    "artifacts": {
        "classification": ".cannbot/gitcode-issue-handler/reports/classification.txt",
        "debug": "Authorization: Bearer secret-value",
        "access_token": "must-never-appear",
    },
}


def complete_state() -> dict:
    """Return an isolated copy of the canonical complete run-state fixture."""
    return copy.deepcopy(COMPLETE_STATE)


def run_report(tmp_path: Path, state: dict, *extra: str):
    state_path = tmp_path / "input-state.json"
    state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--state", str(state_path), "--strict", *extra],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )


def owner_analysis():
    return {
        "status": "prepared",
        "operators": ["Add", "OtherOp"],
        "candidates": [{
            "login": "core-dev", "operators": ["Add"],
            "contribution": "implemented tail handling",
            "evidence": ["https://gitcode.com/cann/ops-math/commit/abc"],
            "identity_evidence": ["PR commit author.login = core-dev"],
            "limitation": "historical contributor; ownership unconfirmed",
        }],
        "excluded": ["docs-only contributor"],
        "uncovered_operators": ["OtherOp"],
    }


def test_owner_candidates_are_reported_without_assignment(tmp_path):
    state = complete_state()
    state["issues"][1]["owner_candidate_analysis"] = owner_analysis()
    result = run_report(tmp_path, state)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    report = (tmp_path / payload["report_path"]).read_text()
    assert "[@core-dev](https://gitcode.com/core-dev)" in report
    assert "implemented tail handling" in report
    assert "| 算子 | GitCode账号 | 简述 |" in report
    assert "| OtherOp | 待确认 | 未找到可靠候选 |" in report
    assert "docs-only contributor" not in report
    assert "historical contributor" not in report
    assert "commit/abc" not in report
    saved = json.loads((tmp_path / payload["run_state_path"]).read_text())
    assert "operator_owner" not in saved["issues"][1]
    assert saved["issues"][1]["handling_status"] == "waiting_context"


@pytest.mark.parametrize("invalid", ["missing_identity", "duplicate", "too_many"])
def test_strict_owner_candidates_require_evidence_and_bound(tmp_path, invalid):
    state = complete_state()
    analysis = owner_analysis()
    candidate = analysis["candidates"][0]
    if invalid == "missing_identity":
        candidate.pop("identity_evidence")
    elif invalid == "duplicate":
        duplicate = copy.deepcopy(candidate)
        duplicate["login"] = "CORE-DEV"
        analysis["candidates"].append(duplicate)
    else:
        analysis.pop("operators")
        analysis["candidates"] = [dict(candidate, login=f"dev-{i}") for i in range(6)]
    state["issues"][1]["owner_candidate_analysis"] = analysis
    result = run_report(tmp_path, state)
    assert result.returncode == 2
    assert "owner candidate" in result.stderr


def test_owner_candidates_may_be_empty_when_evidence_is_insufficient(tmp_path):
    state = complete_state()
    state["issues"][1]["owner_candidate_analysis"] = {
        "status": "insufficient_evidence", "candidates": [],
        "unverified_authors": ["identity unresolved"],
    }
    result = run_report(tmp_path, state)
    assert result.returncode == 0, result.stderr
    report = (tmp_path / json.loads(result.stdout)["report_path"]).read_text()
    assert "| 待确认 | 待确认 | 未找到可靠候选 |" in report
    assert "identity unresolved" not in report


def test_owner_candidates_are_bounded_per_operator_and_issue_can_exceed_five(tmp_path):
    state = complete_state()
    analysis = {
        "status": "prepared",
        "operators": ["Add", "Mul"],
        "candidates": [],
        "uncovered_operators": ["Mul"],
    }
    for index in range(5):
        analysis["candidates"].append({
            "login": f"add-dev-{index}", "operators": ["Add"],
            "contribution": "core implementation", "evidence": ["sha"],
            "identity_evidence": ["api"], "limitation": "historical",
        })
    for index in range(5):
        analysis["candidates"].append({
            "login": f"mul-dev-{index}", "operators": ["Mul"],
            "contribution": "core implementation", "evidence": ["sha"],
            "identity_evidence": ["api"], "limitation": "historical",
        })
    state["issues"][1]["owner_candidate_analysis"] = analysis
    result = run_report(tmp_path, state)
    assert result.returncode == 0, result.stderr
    report = (tmp_path / json.loads(result.stdout)["report_path"]).read_text()
    assert report.count("| Add |") == 5
    assert report.count("| Mul |") == 5


def test_strict_owner_candidates_reject_more_than_five_for_one_operator(tmp_path):
    state = complete_state()
    analysis = owner_analysis()
    analysis["operators"] = ["Add"]
    analysis["uncovered_operators"] = []
    analysis["candidates"] = [
        dict(analysis["candidates"][0], login=f"dev-{index}")
        for index in range(6)
    ]
    state["issues"][1]["owner_candidate_analysis"] = analysis
    result = run_report(tmp_path, state)
    assert result.returncode == 2
    assert "Add owner candidates must be at most 5" in result.stderr


def test_strict_owner_candidates_require_explicit_gap_for_target_operator(tmp_path):
    state = complete_state()
    analysis = owner_analysis()
    analysis["uncovered_operators"] = []
    state["issues"][1]["owner_candidate_analysis"] = analysis
    result = run_report(tmp_path, state)
    assert result.returncode == 2
    assert "OtherOp must have candidates or be listed in uncovered_operators" in result.stderr


def test_generates_compact_report_for_handled_issues(tmp_path):
    result = run_report(tmp_path, complete_state())

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    report_path = tmp_path / payload["report_path"]
    latest_path = tmp_path / payload["latest_path"]
    state_path = tmp_path / payload["run_state_path"]
    report = report_path.read_text(encoding="utf-8")

    assert latest_path.read_text(encoding="utf-8") == report
    assert state_path.exists()
    assert payload["report_path"].startswith(
        ".cannbot/gitcode-issue-handler/reports/"
    )
    assert payload["latest_path"] == (
        ".cannbot/gitcode-issue-handler/reports/latest.md"
    )
    saved_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved_state["run"]["report_generated"] is True
    assert saved_state["run"]["report_path"] == payload["report_path"]
    assert "## 本次实际处理的 Issue" in report
    assert "### Issue #101：修复 Add 精度问题" in report
    assert "### Issue #102：缺少复现日志" in report
    assert "已修复并创建 PR，等待合入" in report
    assert "waiting_context" in report
    assert "PR !88" in report
    assert "目标环境" not in report
    assert "清理与保留项" not in report
    assert "状态统计" not in report
    assert "must-never-appear" not in report
    assert "secret-value" not in report


def test_filters_self_assigned_and_classification_only_issues(tmp_path):
    state = complete_state()
    state["run"]["issues_total"] = 3
    state["issues"].append(
        {
            "iid": 103,
            "url": "https://gitcode.com/cann/ops-math/issues/103",
            "title": "自提 Issue",
            "author": "developer",
            "bucket": "no_attention",
            "category": "self_assigned",
            "handling_status": "no_attention",
            "resolution_status": "resolution_pending",
            "result_summary": "已有本人 PR",
            "next_action": "等待本人处理",
            "process_log": [
                {
                    "time": "2026-08-11T09:05:00+08:00",
                    "stage": "comment",
                    "action": "发送 /assign",
                    "result": "已指派",
                    "evidence": ["issue #103"],
                }
            ],
        }
    )

    result = run_report(tmp_path, state)

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    report = (tmp_path / payload["report_path"]).read_text(encoding="utf-8")
    saved_state = json.loads(
        (tmp_path / payload["run_state_path"]).read_text(encoding="utf-8")
    )
    assert payload["issues"] == 2
    assert payload["excluded_observations"] == 1
    assert "Issue #103" not in report
    assert "自提 Issue" not in report
    assert saved_state["run"]["issues_scanned_total"] == 3
    assert saved_state["run"]["issues_total"] == 2
    assert [item["iid"] for item in saved_state["issues"]] == [101, 102]


def test_strict_mode_rejects_issue_without_process_and_result(tmp_path):
    state = complete_state()
    state["issues"][0]["process_log"] = []
    state["issues"][0]["result_summary"] = ""

    result = run_report(tmp_path, state)

    assert result.returncode == 2
    assert "process_log" in result.stderr
    assert "result_summary" in result.stderr
    assert not (
        tmp_path / ".cannbot" / "gitcode-issue-handler" / "reports"
    ).exists()


def test_strict_mode_requires_process_stages_matching_artifacts(tmp_path):
    state = complete_state()
    state["issues"][0]["process_log"] = [
        entry
        for entry in state["issues"][0]["process_log"]
        if entry["stage"] != "validate"
    ]

    result = run_report(tmp_path, state)

    assert result.returncode == 2
    assert "missing stages: validate" in result.stderr


def test_no_issue_run_still_generates_report(tmp_path):
    state = complete_state()
    state["run"]["issues_total"] = 0
    state["run"]["overall_status"] = "no_issues"
    state["issues"] = []
    state["groups"] = []

    result = run_report(tmp_path, state)

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    report = (tmp_path / payload["report_path"]).read_text(encoding="utf-8")
    assert "本次未实际处理任何 Issue" in report


def test_explicit_report_paths_are_never_redirected(tmp_path):
    result = run_report(
        tmp_path,
        complete_state(),
        "--output",
        "custom/result.md",
        "--latest",
        "custom/current.md",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["report_path"] == "custom/result.md"
    assert payload["latest_path"] == "custom/current.md"
    assert (tmp_path / "custom" / "result.md").is_file()
    assert not (
        tmp_path / ".cannbot" / "gitcode-issue-handler" / "reports"
    ).exists()


def test_responsibility_routes_list_only_and_ignore_without_process_details(tmp_path):
    state = complete_state()
    state["issues"][0]["responsibility"] = "handle"
    state["issues"].append({
        "iid": 103,
        "url": "https://gitcode.com/cann/ops-math/issues/103",
        "title": "只列举的 Issue",
        "responsibility": "list-only",
        "responsibility_summary": "记录该请求，等待后续统一排期",
        "handled_in_run": True,
        "owner_candidate_analysis": {"candidates": [{"login": "must-not-show"}]},
        "process_log": [{"stage": "implement", "action": "must-not-show", "result": "x"}],
    })
    state["issues"].append({
        "iid": 104,
        "url": "https://gitcode.com/cann/ops-math/issues/104",
        "responsibility": "ignore",
        "handled_in_run": True,
    })
    state["run"]["issues_total"] = 4

    result = run_report(tmp_path, state)

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    report = (tmp_path / payload["report_path"]).read_text(encoding="utf-8")
    saved = json.loads((tmp_path / payload["run_state_path"]).read_text())
    assert saved["run"]["issues_total"] == 2
    assert saved["run"]["issues_listed_total"] == 1
    assert payload["issues"] == 2
    assert "## 仅列举" in report
    assert "[#103](https://gitcode.com/cann/ops-math/issues/103)：记录该请求，等待后续统一排期" in report
    assert "must-not-show" not in report
    assert "Issue #104" not in report
    assert [item["iid"] for item in saved["issues"]] == [101, 102]
    assert [item["iid"] for item in saved["listed_issues"]] == [103]


def test_list_only_is_idempotent_and_strict_requires_summary(tmp_path):
    state = complete_state()
    listed = {
        "iid": 103,
        "url": "https://gitcode.com/cann/ops-math/issues/103",
        "responsibility": "list-only",
        "responsibility_summary": "仅记录",
    }
    state["issues"].append(copy.deepcopy(listed))
    state["listed_issues"] = [copy.deepcopy(listed)]
    state["run"]["issues_total"] = 3
    first = run_report(tmp_path, state)
    assert first.returncode == 0, first.stderr
    saved = json.loads((tmp_path / json.loads(first.stdout)["run_state_path"]).read_text())
    second = run_report(tmp_path, saved)
    assert second.returncode == 0, second.stderr
    saved_again = json.loads((tmp_path / json.loads(second.stdout)["run_state_path"]).read_text())
    assert [item["iid"] for item in saved_again["listed_issues"]] == [103]

    invalid = complete_state()
    invalid["issues"].append({
        "iid": 105,
        "url": "https://gitcode.com/cann/ops-math/issues/105",
        "responsibility": "list-only",
    })
    invalid["run"]["issues_total"] = 3
    result = run_report(tmp_path, invalid)
    assert result.returncode == 2
    assert "responsibility_summary" in result.stderr


@pytest.mark.parametrize("status,expected", [
    ("verified", "已临时指派 @core-dev，请确认真正负责人"),
    ("failed", "@core-dev 尚未回查成功，不能视为已指派"),
])
def test_provisional_assignment_preserves_candidates_and_confirmation(tmp_path, status, expected):
    state = complete_state()
    issue = state["issues"][1]
    issue.update(assignment_provisional=True, assigned_candidate="core-dev", assignment_status=status,
                 owner_candidate_analysis=owner_analysis())
    result = run_report(tmp_path, state)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    report = (tmp_path / payload["report_path"]).read_text()
    assert expected in report
    assert "供线下确认，未指派" not in report
    assert "[@core-dev](https://gitcode.com/core-dev)" in report
    saved = json.loads((tmp_path / payload["run_state_path"]).read_text())
    assert "operator_owner" not in saved["issues"][1]


def test_fallback_assignment_report_distinguishes_recipient_from_candidates(tmp_path):
    state = complete_state()
    issue = state["issues"][1]
    issue.update(assignment_provisional=True, assigned_candidate="fallback-login",
                 assignment_status="verified", assignment_source="fallback_user")
    analysis = owner_analysis()
    analysis["candidates"] = []
    analysis["uncovered_operators"] = analysis["operators"]
    issue["owner_candidate_analysis"] = analysis
    result = run_report(tmp_path, state)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    report = (tmp_path / payload["report_path"]).read_text()
    assert "无候选，已临时指派兜底接收人 @fallback-login，请确认真正负责人" in report
    assert "[@fallback-login]" not in report
    saved = json.loads((tmp_path / payload["run_state_path"]).read_text())
    assert saved["issues"][1]["owner_candidate_analysis"]["candidates"] == []
    assert "operator_owner" not in saved["issues"][1]


def test_response_stage_assignment_only_is_reported(tmp_path):
    state = complete_state()
    item = state["issues"][0]
    item.update(category="needs_pr_owner_handoff", handled_in_run=True,
                response_status="exempt_self_authored_pr", assignment_source="linked_pr",
                assignment_status="verified", assignment_provisional=True,
                assigned_candidate="reporter-a", result_summary="已补齐PR作者分配",
                response_artifacts={"assign.md": {"path": "reports/run/issues/issue-101/assign.md"}})
    result = run_report(tmp_path, state)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    report = (tmp_path / payload["report_path"]).read_text(encoding="utf-8")
    assert "已补齐PR作者分配" in report
    assert "自提免首响" in report
    assert "[assign.md](<issues/issue-101/assign.md>)" in report


@pytest.mark.parametrize("kind, path", [("directory", "Samples/story"), ("file", "src/shared.cpp")])
def test_related_code_links_and_unlocated_reason_reach_reports(tmp_path, kind, path):
    state = complete_state()
    route = "tree" if kind == "directory" else "blob"
    url = f"https://gitcode.com/cann/cann-samples/{route}/abc123/{path}"
    state["issues"][0]["related_code"] = [
        {"path": path, "url": url, "kind": kind, "revision": "abc123"}
    ]
    state["issues"][1]["related_code_note"] = "缺少问题版本，尚未定位代码"
    result = run_report(tmp_path, state)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    report = (tmp_path / payload["report_path"]).read_text(encoding="utf-8")
    assert f"[{path}](<{url}>)" in report
    assert "版本：abc123" in report
    assert "缺少问题版本，尚未定位代码" in report
    assert (tmp_path / payload["latest_path"]).read_text(encoding="utf-8") == report


def test_default_report_uses_dated_round_metadata_and_internal_state(tmp_path):
    result = run_report(tmp_path, complete_state())
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    folder = (tmp_path / payload['report_path']).parent
    assert folder.name == '20260811-090000-P0800'
    metadata = json.loads((folder / 'run.json').read_text())
    assert metadata['legacy_run_id'] == COMPLETE_STATE['run']['run_id']
    assert metadata['internal_run_id'] and metadata['timezone'] == 'Asia/Shanghai'
    assert Path(payload['run_state_path']).parent.name == '_internal'
    saved = json.loads((tmp_path / payload['run_state_path']).read_text())
    repeated = run_report(tmp_path, saved)
    assert repeated.returncode == 0, repeated.stderr
    assert json.loads(repeated.stdout)['report_path'] == payload['report_path']
