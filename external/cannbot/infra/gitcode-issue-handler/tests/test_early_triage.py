# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Behavioral tests for early routing, preserving existing classification results."""
import copy
import json
from unittest.mock import patch

import pytest
import yaml

from test_classify_issues import CLASSIFIER as C, comment, issue, pull_request

COLLECT_EVIDENCE = getattr(C, "_collect_evidence")


@pytest.fixture(autouse=True)
def issue_pr_api():
    with patch("issue_pr_evidence.api_get", return_value=[]) as get:
        yield get


def inputs(tmp_path, items):
    config = tmp_path / "config.yaml"
    cfg = {
        "repo": "test/repo",
        "responsibility": {"handle": ["公共问题"], "list-only": [], "ignore": []},
        "report_file": str(tmp_path / "report.txt"),
        "cache_dir": str(tmp_path / "cache"),
        "last_check_file": str(tmp_path / "last_check.json"),
    }
    config.write_text(yaml.safe_dump(cfg))
    for i in items:
        i["responsibility_review"] = {
            "level": "handle",
            "summary": "公共问题",
            "evidence": ["工程路径"],
            "policy_digest": C.policy_digest(cfg["responsibility"]),
        }
    path = tmp_path / "input.json"
    # A fixed source window makes repeated classifications one batch.
    path.write_text(json.dumps({"filters": {"since": "2026-09-01T00:00:00Z"}, "issues": items}))
    return ["--config", str(config), "--input", str(path), "--ignore-last-check"], path


def run(argv):
    with patch.object(C, "_write_stdout") as output:
        assert C.main(argv) == 0
    return json.loads(output.call_args.args[0])


def test_closed_filtered_before_any_remote_evidence(tmp_path):
    argv, _ = inputs(tmp_path, [issue(state="closed", comments_count=100)])
    with patch.object(C, "_collect_evidence", side_effect=AssertionError("closed queried")):
        result = run(argv)
    assert result["total"] == 0


def test_pure_planning_skips_remote_evidence(tmp_path, issue_pr_api):
    argv, _ = inputs(tmp_path, [issue(
        title="Roadmap", description="## 详细计划\n计划完善算子支持。", comments_count=20,
        comments_fetch={"status": "error"},
    )])
    with patch.object(C, "fetch_recent_prs", side_effect=AssertionError("planning queried")):
        result = run(argv)
    assert result["issues"][0]["category"] == "planning_record"
    assert result["by_bucket"]["need_attention"] == 0
    issue_pr_api.assert_not_called()


@pytest.mark.parametrize("author,category", [
    ("reporter", "needs_pr_owner_handoff"),
    ("developer", "needs_first_response_with_pr"),
])
def test_old_native_pr_is_found_outside_recent_window(tmp_path, issue_pr_api, author, category):
    argv, _ = inputs(tmp_path, [issue(comments_count=0)])
    issue_pr_api.side_effect = [
        [{"number": 77}],
        pull_request(77, user={"login": author}, updated_at="2026-01-01T00:00:00Z"),
    ]
    with patch.object(C, "fetch_recent_prs", return_value=([], {"complete": True})):
        result = run(argv)
    item = result["issues"][0]
    assert item["category"] == category
    assert item["auto_action"]["candidate"] == author
    assert item["linked_prs"][0]["pr_number"] == 77
    assert issue_pr_api.call_count == 2


def test_direct_pr_failure_blocks_only_affected_issue(tmp_path, issue_pr_api):
    argv, _ = inputs(tmp_path, [issue(number=1, comments_count=0), issue(number=2, comments_count=0)])
    issue_pr_api.side_effect = [C.requests.RequestException("unavailable"), []]
    with patch.object(C, "fetch_recent_prs", return_value=([], {"complete": True})):
        result = run(argv)
    categories = {item["number"]: item["category"] for item in result["issues"]}
    assert categories == {1: "association_scan_incomplete", 2: "needs_first_look"}
    assert result["association_scan"]["linkage_fallback"]["incomplete_issue_numbers"] == ["1"]


def test_settled_assigned_issue_reuses_original_result_without_http(tmp_path):
    answered = issue(
        assignee="maintainer",
        comments=[comment()],
        comments_count=1,
        updated_at="2026-09-16T01:00:00Z",
    )
    argv, path = inputs(tmp_path, [answered])
    with patch.object(C, "fetch_recent_prs", return_value=([], {"complete": True})):
        first = run(argv)
    assert first["issues"][0]["category"] == "our_team_replied"
    raw = json.loads(path.read_text())
    raw["issues"][0].pop("comments")
    path.write_text(json.dumps(raw))
    with patch.object(C, "fetch_recent_prs", side_effect=AssertionError("settled PR scan")):
        second = run(argv)
    assert second["issues"] == first["issues"]
    assert second["comment_fetch"]["classification_cache_hits"] == 1


@pytest.mark.parametrize(
    "change",
    [
        {"assignee": "different"},
        {"assignee": None},
        {"updated_at": "2026-09-16T02:00:00Z"},
        {"comments_count": 2},
        {"issue_state": "挂起"},
        {"followup_watch": {"conversation_state": "awaiting_reporter"}},
        {"labels": [{"name": "需求建议"}]},
        {"issue_type": "Requirement"},
        {"type": "bug"},
        {"category": "planning"},
    ],
)
def test_changed_issue_never_reuses_settled_classification(tmp_path, change):
    argv, path = inputs(
        tmp_path,
        [issue(assignee="maintainer", comments=[comment()], comments_count=1,
               updated_at="2026-09-16T01:00:00Z")],
    )
    with patch.object(C, "fetch_recent_prs", return_value=([], {"complete": True})):
        run(argv)
    raw = json.loads(path.read_text())
    raw["issues"][0].pop("comments")
    raw["issues"][0].update(change)
    path.write_text(json.dumps(raw))
    runtime = C.load_runtime(C.parse_args(argv))
    with patch.object(C, "_batch_prs", side_effect=RuntimeError("fresh evidence required")):
        with pytest.raises(RuntimeError, match="fresh evidence"):
            COLLECT_EVIDENCE(runtime)


def test_new_reporter_turn_is_followup_not_cached_reply(tmp_path):
    argv, path = inputs(
        tmp_path,
        [issue(assignee="maintainer", comments=[comment()], comments_count=1,
               updated_at="2026-09-16T01:00:00Z")],
    )
    with patch.object(C, "fetch_recent_prs", return_value=([], {"complete": True})):
        run(argv)
        raw = json.loads(path.read_text())
        raw["issues"][0]["comments"].append(
            comment("reporter", "补充日志，请继续确认", comment_id=2,
                    created_at="2026-09-16T02:00:00Z")
        )
        raw["issues"][0]["comments_count"] = 2
        raw["issues"][0]["updated_at"] = "2026-09-16T02:00:00Z"
        path.write_text(json.dumps(raw))
        result = run(argv)
    assert result["issues"][0]["category"] == "reporter_followup"
    assert result["issues"][0]["bucket"] == "need_attention"


def test_assignee_without_reply_still_requires_first_response(tmp_path):
    argv, _ = inputs(
        tmp_path,
        [issue(assignee="maintainer", comments_count=0, updated_at="2026-09-16T01:00:00Z")],
    )
    with patch.object(C, "fetch_recent_prs", return_value=([], {"complete": True})) as fetch:
        assert run(argv)["issues"][0]["category"] == "our_team_needs_work"
        assert run(argv)["issues"][0]["category"] == "our_team_needs_work"
    assert fetch.call_count == 2


def test_batch_snapshot_avoids_repeated_pr_scans_and_does_not_cache_incomplete(tmp_path):
    argv, _ = inputs(tmp_path, [issue(comments_count=0)])
    argv += ["--pr-snapshot", str(tmp_path / "batch-prs.json")]
    with patch.object(C, "fetch_recent_prs", return_value=([], {"complete": False})) as fetch:
        run(argv)
        run(argv)
        assert fetch.call_count == 2
    with patch.object(
        C, "fetch_recent_prs", return_value=([], {"complete": True, "pages_requested": 1})
    ) as fetch:
        run(argv)
        second = run(argv)
        assert fetch.call_count == 1
        assert second["association_scan"]["pr_fetch"]["snapshot_hits"] == 1
    with patch.object(C, "fetch_recent_prs", return_value=([], {"complete": True})) as fetch:
        run(argv + ["--no-cache"])
        assert fetch.call_count == 1


def test_partial_classification_does_not_advance_global_cursor(tmp_path):
    argv, _ = inputs(tmp_path, [issue(state="closed")])
    with patch.object(C, "save_last_check") as save:
        run(argv + ["--no-update-last-check"])
        save.assert_not_called()


def test_cache_keeps_current_derived_age_and_sla(tmp_path):
    argv, path = inputs(
        tmp_path,
        [issue(assignee="maintainer", comments=[comment()], comments_count=1,
               updated_at="2026-09-16T01:00:00Z", issue_age_days=1)],
    )
    with patch.object(C, "fetch_recent_prs", return_value=([], {"complete": True})):
        run(argv)
    raw = json.loads(path.read_text())
    raw["issues"][0].pop("comments")
    raw["issues"][0].update(issue_age_days=4, first_response_sla="breached")
    path.write_text(json.dumps(raw))
    with patch.object(C, "fetch_recent_prs", side_effect=AssertionError("no PR query needed")):
        result = run(argv)
    assert result["issues"][0]["issue_age_days"] == 4
    assert result["issues"][0]["first_response_sla"] == "breached"


def test_classification_cache_is_scoped_to_api_endpoint(tmp_path):
    argv, path = inputs(
        tmp_path,
        [issue(assignee="maintainer", comments=[comment()], comments_count=1,
               updated_at="2026-09-16T01:00:00Z")],
    )
    with patch.object(C, "fetch_recent_prs", return_value=([], {"complete": True})):
        run(argv)
    raw = json.loads(path.read_text())
    raw["issues"][0].pop("comments")
    path.write_text(json.dumps(raw))
    cfg_path = tmp_path / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["gitcode_api"] = "https://another.example.test/api/v5"
    cfg_path.write_text(yaml.safe_dump(cfg))
    runtime = C.load_runtime(C.parse_args(argv))
    with patch.object(C, "_batch_prs", side_effect=RuntimeError("fresh site required")):
        with pytest.raises(RuntimeError, match="fresh site"):
            COLLECT_EVIDENCE(runtime)


def test_snapshot_must_cover_requested_window(tmp_path):
    argv, path = inputs(tmp_path, [issue(comments_count=0)])
    argv.remove('--ignore-last-check')
    argv += ["--pr-snapshot", str(tmp_path / "batch-prs.json")]
    with patch.object(C, "fetch_recent_prs", return_value=([], {"complete": True})) as fetch:
        run(argv)
        raw = json.loads(path.read_text())
        raw["filters"]["since"] = "2026-09-02T00:00:00Z"
        path.write_text(json.dumps(raw))
        run(argv)
        assert fetch.call_count == 1
        raw["filters"]["since"] = "2026-08-31T00:00:00Z"
        path.write_text(json.dumps(raw))
        run(argv)
        assert fetch.call_count == 2
