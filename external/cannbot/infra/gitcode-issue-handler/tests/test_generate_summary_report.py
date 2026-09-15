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
