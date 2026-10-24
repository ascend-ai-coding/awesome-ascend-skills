# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Response matrix, historical self-exemption and selection regressions."""
import json
from types import SimpleNamespace

import pytest

from test_classify_issues import CLASSIFIER as C, classification_options, comment, issue
from test_assign_issue import ASSIGN, URL, write_inputs, write_linked_pr
from pr_response_policy import select_response_pr, self_authored


def linked(author="developer", **overrides):
    return {"pr_number": 7, "pr_state": "open", "pr_author": author, **overrides}


@pytest.mark.parametrize("assigned,replied,pr_kind,expected", [
    (False, False, "none", ("need_attention", "needs_first_look")),
    (False, False, "own", ("need_attention", "needs_pr_owner_handoff")),
    (False, False, "other", ("need_attention", "needs_first_response_with_pr")),
    (False, True, "none", ("no_attention", "replied_no_owner")),
    (False, True, "own", ("need_attention", "needs_pr_owner_handoff")),
    (False, True, "other", ("need_attention", "needs_pr_owner_handoff")),
    (True, False, "none", ("need_attention", "our_team_needs_work")),
    (True, False, "own", ("no_attention", "self_assigned")),
    (True, False, "other", ("need_attention", "needs_first_response_with_pr")),
    (True, True, "none", ("no_attention", "our_team_replied")),
    (True, True, "own", ("no_attention", "self_assigned")),
    (True, True, "other", ("no_attention", "our_team_done_with_pr")),
])
def test_response_matrix(assigned, replied, pr_kind, expected):
    prs = [] if pr_kind == "none" else [linked("reporter" if pr_kind == "own" else "developer")]
    source = issue(assignee="owner" if assigned else None, comments=[comment()] if replied else [])
    result = C.classify_one(source, classification_options({"1": prs}))
    assert (result["bucket"], result["category"]) == expected
    action = result["auto_action"]
    assert bool(action) == bool(prs and not assigned)
    if action:
        assert action["policy"] == "auto-response"
        assert action["requires_verified_first_response"] is (pr_kind != "own")


@pytest.mark.parametrize("reply", [False, True])
def test_any_self_authored_pr_takes_priority(reply):
    result = C.classify_one(issue(comments=[comment()] if reply else []), classification_options({
        "1": [linked("other", pr_number=2), linked("reporter", pr_number=99)],
    }))
    assert result["category"] == "needs_pr_owner_handoff"
    assert result["auto_action"]["candidate"] == "reporter"
    assert result["auto_action"]["response_requirement"] == "exempt_self_authored_pr"


def test_pr_authorship_and_selection_share_normalized_identity():
    own = linked(" REPORTER ")
    assert self_authored(" Reporter ", [own])
    assert select_response_pr([linked("other", pr_number=2), own], " Reporter ") == (own, "self_authored")
    assert not self_authored(" ", [linked(" ")])


@pytest.mark.parametrize("pr", [
    linked("reporter", pr_state="closed"),
    linked("reporter", pr_expired=True, expiration_evidence=["作者已明确废弃此方案"]),
])
def test_historical_self_issue_3139_stays_in_unresolved_tracking(pr):
    source = issue(assignee="reporter", comments=[comment(body="/assign @reporter")])
    result = C.classify_one(source, classification_options({"1": [pr]}))
    assert (result["bucket"], result["category"]) == ("no_attention", "self_assigned")
    assert "未闭环" in result["reason"]
    assert result["auto_action"] is None
    # Unassigned historical PRs must not authorize fresh PR-owner assignment.
    source["assignee"] = None
    assert C.classify_one(source, classification_options({"1": [pr]}))["category"] == "needs_only_assign_cmd"


def test_closed_nonself_pr_does_not_erase_response_or_create_assignment():
    result = C.classify_one(issue(assignee="owner", comments=[comment()]), classification_options({
        "1": [linked(pr_state="closed")],
    }))
    assert (result["bucket"], result["category"]) == ("no_attention", "our_team_replied")
    assert result["auto_action"] is None


def test_merged_pr_and_unsubstantiated_expiration_remain_eligible():
    for pr in [linked(pr_state="closed", pr_merged=True), linked(pr_expired=True)]:
        result = C.classify_one(issue(), classification_options({"1": [pr]}))
        assert result["category"] == "needs_first_response_with_pr"


def test_self_assignee_stays_exempt_with_expired_own_and_active_nonself_pr():
    result = C.classify_one(issue(assignee="reporter"), classification_options({
        "1": [linked("reporter", pr_state="closed"), linked("developer", pr_number=8)],
    }))
    assert (result["bucket"], result["category"]) == ("no_attention", "self_assigned")
    assert "未闭环" in result["reason"]
    assert result["auto_action"] is None


@pytest.mark.parametrize("pr", [
    linked("reporter", pr_state="closed"),
    linked("reporter", pr_expired=True, expiration_evidence=["作者已明确废弃此方案"]),
])
def test_historical_own_pr_with_other_assignee_preserves_first_response_exemption(pr):
    result = C.classify_one(issue(assignee="other-owner"), classification_options({"1": [pr]}))
    assert (result["bucket"], result["category"]) == ("no_attention", "self_assigned")
    assert "历史自提豁免" in result["reason"]
    assert "未闭环" in result["reason"]
    assert result["auto_action"] is None


def test_waiting_reporter_does_not_hide_pr_assignment():
    source = issue(comments=[comment()], followup_watch={"conversation_state": "awaiting_reporter"}, issue_state="挂起")
    result = C.classify_one(source, classification_options({"1": [linked()]}))
    assert result["bucket"] == "need_attention"
    assert result["category"] == "needs_pr_owner_handoff"


def test_new_question_survives_historical_self_exemption():
    source = issue(assignee="reporter", comments=[comment(), comment(
        "reporter", "还有一个问题", comment_id=2, created_at="2026-08-15T00:00:00Z",
    )])
    result = C.classify_one(source, classification_options({"1": [linked("reporter", pr_state="closed")]}))
    assert result["category"] == "reporter_followup"


def test_pr_dependent_cache_never_hides_remote_pr_state_change(tmp_path):
    source = issue(assignee="owner", comments_count=1, updated_at="2026-09-16T00:00:00Z")
    result = {"bucket": "no_attention", "category": "our_team_done_with_pr",
              "latest_maintainer_comment_id": 1, "linked_prs": [linked()]}
    source.pop("comments")
    C.save_classification(C.ClassificationCacheEntry(str(tmp_path), "test/repo", source, {}, "v", result))
    assert C.load_settled(str(tmp_path), "test/repo", source, {}, "v") is None


def reviewed_pr(author, number, points):
    return {"pr_author": author, "state": "open", "pr_number": number,
            "pr_url": f"https://gitcode.com/xujiachen8/ops-math/merge_requests/{number}",
            "preexisting_association": True, "covers_issue": True,
            "coverage_review": {"covered_issue_points": points, "evidence": [f"PR !{number} diff"]}}


def test_nonself_selection_uses_issue_coverage_then_stable_tie_order():
    narrow = reviewed_pr("first", 1, ["symptom"])
    broad = reviewed_pr("second", 2, ["symptom", "boundary"])
    assert select_response_pr([narrow, broad], "reporter")[0] == broad
    tie = reviewed_pr("third", 3, ["symptom", "boundary"])
    assert select_response_pr([tie, broad], "reporter")[0] == broad
    assert select_response_pr([broad, narrow], "first")[0] == narrow


@pytest.mark.parametrize("automatic,approved", [(True, False), (False, True), (False, False)])
def test_self_assignment_without_response_is_controlled_by_auto_response(
    tmp_path, monkeypatch, automatic, approved,
):
    paths = write_inputs(tmp_path, auto_response=automatic)
    linked_path = write_linked_pr(tmp_path)
    data = json.loads(linked_path.read_text())
    data["issue_author"] = "test-owner"
    linked_path.write_text(json.dumps(data))
    replies = [
        {"state": "open", "user": {"login": "test-owner"}},
        {"state": "open", "user": {"login": "test-owner"}},
        [{"number": "42"}],
        {"state": "open", "assignee": {"login": "test-owner"}},
    ]
    writes = []
    monkeypatch.setenv("GITCODE_TOKEN", "test-token")
    monkeypatch.setattr(ASSIGN, "api_get", lambda *a, **kw: replies.pop(0))
    monkeypatch.setattr(ASSIGN, "api_patch", lambda *a, **kw: writes.append(kw) or SimpleNamespace(status_code=200))
    kwargs = dict(config_path=paths["config"], issue_url=URL, linked_pr_file=linked_path,
                  result_file=paths["result"], apply=True, assignment_approved=approved, session=object())
    if not automatic and not approved:
        with pytest.raises(ASSIGN.AssignmentError, match="disabled"):
            ASSIGN.execute(**kwargs)
        assert writes == []
        return
    result = ASSIGN.execute(**kwargs)
    assert result["response_status"] == "exempt_self_authored_pr"
    assert result["status"] == "verified"
    assert writes == [{"json_data": {"assignee": "test-owner"}}]
    assert (tmp_path / "issues/issue-42/assign.md").read_text() == "/assign @test-owner\n"
    assert not (tmp_path / "issues/issue-42/reply.md").exists()


def test_multiple_authors_are_selected_by_coverage_in_assignment_preview(tmp_path):
    paths = write_inputs(tmp_path)
    path = tmp_path / "linked.json"
    path.write_text(json.dumps({"target_issue_url": URL, "issue_author": "reporter", "linked_prs": [
        reviewed_pr("narrow", 1, ["symptom"]), reviewed_pr("broad", 2, ["symptom", "boundary"]),
    ]}))
    result = ASSIGN.execute(config_path=paths["config"], issue_url=URL,
                            linked_pr_file=path, result_file=paths["result"])
    assert result["owner"] == "broad"
    assert result["response_status"] == "pending"
    assert result["selection"] == "issue_coverage"


def test_blank_comments_and_unknown_pr_state_are_not_completion_evidence():
    source = issue(assignee="owner", comments=[comment(body="  ")])
    assert C.classify_one(source, classification_options())["category"] == "our_team_needs_work"
    result = C.classify_one(source, classification_options({"1": [linked(pr_state=None)]}))
    assert result["category"] == "association_scan_incomplete"
    assert result["auto_action"] is None


def test_artifacts_keep_each_issue_and_preserve_analysis_between_operations(tmp_path):
    from response_artifacts import save_response_artifacts

    output = tmp_path / "result.json"
    save_response_artifacts(output, URL, "已读取 PR，覆盖全部诉求", reply="方案已在 PR 中实现。")
    directory = tmp_path / "issues/issue-42"
    analysis = directory / "analysis.md"
    analysis.write_text(analysis.read_text() + "\n补充核查：覆盖尾块场景。\n")
    files = save_response_artifacts(output, URL, "自提优先", owner="reporter")
    assert "补充核查：覆盖尾块场景" in analysis.read_text()
    assert "已读取 PR" in analysis.read_text()
    assert (directory / "reply.md").read_text() == "方案已在 PR 中实现。"
    assert set(files) == {"analysis.md", "reply.md", "assign.md"}
    save_response_artifacts(output, URL.replace("42", "43"), "另一条 Issue", owner="other")
    assert (tmp_path / "issues/issue-43/assign.md").read_text() == "/assign @other\n"
    assert (directory / "assign.md").read_text() == "/assign @reporter\n"


def test_nonself_assignment_preview_can_precede_reply_but_apply_cannot(tmp_path, monkeypatch):
    paths = write_inputs(tmp_path, auto_response=True)
    linked_path = write_linked_pr(tmp_path)
    kwargs = dict(config_path=paths["config"], issue_url=URL,
                  linked_pr_file=linked_path, result_file=paths["result"])
    result = ASSIGN.execute(**kwargs)
    assert result["status"] == "preview"
    monkeypatch.setattr(ASSIGN, "api_patch", lambda *a, **kw: pytest.fail("missing reply must not assign"))
    with pytest.raises(ASSIGN.AssignmentError, match="verified or reused"):
        ASSIGN.execute(**kwargs, apply=True)


def test_selected_pr_changed_after_coverage_review_blocks_assignment(tmp_path, monkeypatch):
    paths = write_inputs(tmp_path, auto_response=True)
    linked_path = write_linked_pr(tmp_path, coverage_review={"head_sha": "reviewed"})
    replies = [
        {"state": "open", "user": {"login": "reporter"}},
        {"state": "open", "user": {"login": "test-owner"}, "head": {"sha": "changed"}},
    ]
    monkeypatch.setenv("GITCODE_TOKEN", "test-token")
    monkeypatch.setattr(ASSIGN, "api_get", lambda *a, **kw: replies.pop(0))
    monkeypatch.setattr(ASSIGN, "api_patch", lambda *a, **kw: pytest.fail("stale coverage must not assign"))
    with pytest.raises(ASSIGN.AssignmentError, match="changed after coverage"):
        ASSIGN.execute(config_path=paths["config"], issue_url=URL, linked_pr_file=linked_path,
                       result_file=paths["result"], response_result=paths["response"], apply=True, session=object())
