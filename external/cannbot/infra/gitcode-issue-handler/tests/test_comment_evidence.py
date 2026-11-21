# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Assignment notices must not suppress first responses or reporter follow-ups."""

from datetime import timedelta

import pytest

from test_classify_issues import CLASSIFIER as C, classification_options, comment, issue
from test_auto_close_stale_issues import AUTO_CLOSE, NOW, POLICY, question_issue

NOTICE = (
    "### Notice \n"
    "This issue can not be assigned to ***eyja_fjalla***. "
    "Please try to assign to the repository members."
)
ASSIGN = "/assign [@eyja_fjalla](https://gitcode.com/eyja_fjalla)"


@pytest.mark.parametrize("assignee", [None, "owner"])
@pytest.mark.parametrize("linked_pr", [False, True])
@pytest.mark.parametrize("raw_user", [False, True])
@pytest.mark.parametrize("with_command", [False, True])
def test_assignment_failure_never_satisfies_first_response(
    assignee, linked_pr, raw_user, with_command
):
    comments = [comment("cann-robot", NOTICE, comment_id=194881379, raw_user=raw_user)]
    if with_command:
        comments.insert(0, comment(body=ASSIGN))
    source = issue(assignee=assignee, comments=comments)
    prs = (
        {"1": [{"pr_number": 6532, "pr_state": "open", "pr_author": "developer"}]}
        if linked_pr
        else {}
    )
    result = C.classify_one(source, classification_options(prs))
    assert result["bucket"] == "need_attention"
    if linked_pr:
        assert result["category"] == "needs_first_response_with_pr"
        if result["auto_action"]:
            assert result["auto_action"]["response_requirement"] == "required"
    else:
        expected = (
            ("our_team_only_assign_cmd" if with_command else "our_team_needs_work")
            if assignee
            else ("needs_only_assign_cmd" if with_command else "needs_first_look")
        )
        assert result["category"] == expected
    conversation = C.analyze_conversation(source)
    assert conversation["latest_maintainer_comment_id"] is None
    assert conversation["state"] == "awaiting_maintainer"


@pytest.mark.parametrize(
    "target", ["yourself", "***yourself***", "***eyja_fjalla***", "user.name-42"]
)
def test_assignment_failure_template_accepts_target_account(target):
    body = NOTICE.replace("***eyja_fjalla***", target)
    assert not C.get_effective_comments([comment(" CANN-ROBOT ", body)], "reporter")


@pytest.mark.parametrize(
    "author,body",
    [
        ("maintainer", NOTICE),
        ("cann-robot", "已核对 target 参数描述，文档修订 PR 尚未合入。"),
        ("cann-robot", NOTICE + "\n另外已核对文档，target 描述存在笔误。"),
        ("unknown-bot", NOTICE),
    ],
)
def test_notice_filter_keeps_unrecognized_or_substantive_comments(author, body):
    reply = comment(author, body)
    assert C.get_effective_comments([reply], "reporter") == [reply]


def test_explicit_system_event_is_not_a_reply_even_from_other_account():
    event = {**comment("platform", "assignee changed"), "system": True}
    assert C.get_effective_comments([event], "reporter") == []


def test_issue_3690_real_reply_remains_the_only_response_evidence():
    reply = comment(
        "chensi79",
        "您好，感谢反馈！已核对 target 描述的笔误，PR #6532 尚未合入。",
        comment_id=194918548,
        created_at="2026-10-21T11:03:30+08:00",
        raw_user=True,
    )
    notice = comment(
        "cann-robot",
        NOTICE,
        comment_id=194881379,
        created_at="2026-10-21T09:12:54+08:00",
        raw_user=True,
    )
    source = issue(
        number=3690, assignee="xukaiwei", comments=[reply, notice, comment(body=ASSIGN)]
    )
    assert C.get_effective_comments(source["comments"], "reporter") == [
        reply,
        source["comments"][2],
    ]
    assert C.analyze_conversation(source)["latest_maintainer_comment_id"] == "194918548"
    assert (
        C.classify_one(source, classification_options())["category"]
        == "our_team_replied"
    )


def test_bot_notice_after_reporter_question_cannot_hide_followup():
    source = issue(
        assignee="owner",
        comments=[
            comment(comment_id=10, created_at="2026-10-20T08:00:00Z"),
            comment(
                "reporter",
                "还需要确认另一个参数，请帮忙看下。",
                comment_id=11,
                created_at="2026-10-20T09:00:00Z",
            ),
            comment(
                "cann-robot", NOTICE, comment_id=12, created_at="2026-10-20T10:00:00Z"
            ),
        ],
    )
    result = C.classify_one(source, classification_options())
    assert (result["bucket"], result["category"]) == (
        "need_attention",
        "reporter_followup",
    )
    assert C.analyze_conversation(source)["latest_maintainer_comment_id"] == "10"


def test_previous_false_answer_cache_is_not_reused_after_rule_revision(tmp_path):
    source = issue(
        assignee="owner", comments_count=1, updated_at="2026-10-20T10:00:00Z"
    )
    source.pop("comments")
    cached = {
        "bucket": "no_attention",
        "category": "our_team_replied",
        "latest_maintainer_comment_id": 194881379,
    }
    C.save_classification(
        C.ClassificationCacheEntry(
            str(tmp_path), "cann/ops-math", source, {}, "before-notice-fix", cached
        )
    )
    assert (
        C.load_settled(str(tmp_path), "cann/ops-math", source, {}, "before-notice-fix")
        == cached
    )
    assert (
        C.load_settled(str(tmp_path), "cann/ops-math", source, {}, C._CACHE_REVISION)
        is None
    )


def test_assignment_failure_is_not_an_answer_for_question_auto_close():
    notice = comment(
        "cann-robot", NOTICE, created_at=(NOW - timedelta(hours=72)).isoformat()
    )
    result = AUTO_CLOSE.evaluate_issue(question_issue([notice]), POLICY)
    assert result["eligible"] is False
    assert result["reason"] == "no_handler_reply"


def test_notice_does_not_replace_real_answer_or_restart_auto_close_clock():
    reply = comment(
        "maintainer",
        "请按官方示例传入 target 参数。",
        created_at=(NOW - timedelta(hours=72)).isoformat(),
    )
    notice = comment(
        "cann-robot", NOTICE, created_at=(NOW - timedelta(hours=1)).isoformat()
    )
    result = AUTO_CLOSE.evaluate_issue(question_issue([reply, notice]), POLICY)
    assert result["eligible"] is True
    assert result["handler"] == "maintainer"
    assert result["last_handler_reply_at"] == (NOW - timedelta(hours=72)).isoformat()


def test_notice_cannot_hide_reporter_question_from_auto_close():
    comments = [
        comment(
            "maintainer",
            "请按官方示例传入 target 参数。",
            created_at=(NOW - timedelta(hours=96)).isoformat(),
        ),
        comment(
            "reporter",
            "还有一个参数问题。",
            created_at=(NOW - timedelta(hours=80)).isoformat(),
        ),
        comment(
            "cann-robot", NOTICE, created_at=(NOW - timedelta(hours=72)).isoformat()
        ),
    ]
    result = AUTO_CLOSE.evaluate_issue(question_issue(comments), POLICY)
    assert result["eligible"] is False
    assert result["reason"] == "reporter_replied_after_handler"
