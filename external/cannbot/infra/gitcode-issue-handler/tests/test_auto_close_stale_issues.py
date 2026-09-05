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

import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "auto_close_stale_issues.py"
SPEC = importlib.util.spec_from_file_location("auto_close_stale_issues", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUTO_CLOSE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUTO_CLOSE)

NOW = datetime(2026, 8, 11, 8, 0, tzinfo=timezone.utc)
POLICY = AUTO_CLOSE.ClosePolicy(now=NOW, inactive_hours=48)


def comment(author: str, body: str, created_at: datetime) -> dict:
    return {
        "author": author,
        "body": body,
        "created_at": created_at.isoformat(),
    }


def question_issue(comments: list[dict], **overrides) -> dict:
    issue = {
        "iid": 42,
        "number": 42,
        "title": "[Question|API] 如何调用接口",
        "description": "请问接口如何使用？",
        "labels": ["question"],
        "state": "open",
        "author": "reporter",
        "assignee": "maintainer",
        "comments": comments,
    }
    issue.update(overrides)
    return issue


class TestEligibility:
    def test_exactly_48_hours_is_eligible(self) -> None:
        issue = question_issue(
            [
                comment(
                    "maintainer", "可以按文档中的方式调用。", NOW - timedelta(hours=48)
                )
            ]
        )

        result = AUTO_CLOSE.evaluate_issue(issue, POLICY)

        assert result["eligible"] is True
        assert result["reason"] == "answered_and_inactive"

    def test_reporter_reply_after_handler_blocks_close(self) -> None:
        issue = question_issue(
            [
                comment(
                    "maintainer", "可以按文档中的方式调用。", NOW - timedelta(hours=72)
                ),
                comment("reporter", "还有一个问题。", NOW - timedelta(hours=60)),
            ]
        )

        result = AUTO_CLOSE.evaluate_issue(issue, POLICY)

        assert result["eligible"] is False
        assert result["reason"] == "reporter_replied_after_handler"

    def test_unassigned_issue_with_non_reporter_answer_is_eligible(self) -> None:
        issue = question_issue(
            [
                comment(
                    "maintainer", "可以按文档中的方式调用。", NOW - timedelta(hours=72)
                )
            ],
            assignee=None,
        )

        result = AUTO_CLOSE.evaluate_issue(issue, POLICY)

        assert result["eligible"] is True
        assert result["handler"] == "maintainer"

    def test_latest_non_reporter_answer_restarts_quiet_period(self) -> None:
        issue = question_issue(
            [
                comment("maintainer-a", "这是预期行为。", NOW - timedelta(hours=72)),
                comment(
                    "maintainer-b", "补充一下适用范围。", NOW - timedelta(hours=24)
                ),
            ],
            assignee=None,
        )

        result = AUTO_CLOSE.evaluate_issue(issue, POLICY)

        assert result["eligible"] is False
        assert result["reason"] == "quiet_period_not_reached"

    def test_assign_command_is_not_an_answer(self) -> None:
        issue = question_issue(
            [comment("maintainer", "/assign @maintainer", NOW - timedelta(hours=72))]
        )

        result = AUTO_CLOSE.evaluate_issue(issue, POLICY)

        assert result["eligible"] is False
        assert result["reason"] == "latest_handler_comment_not_answer"

    def test_request_for_more_information_is_not_an_answer(self) -> None:
        issue = question_issue(
            [
                comment(
                    "maintainer",
                    "请补充版本信息和完整日志。",
                    NOW - timedelta(hours=72),
                )
            ]
        )

        result = AUTO_CLOSE.evaluate_issue(issue, POLICY)

        assert result["eligible"] is False
        assert result["reason"] == "latest_handler_comment_not_answer"

    def test_watched_information_request_can_close_after_quiet_period(self) -> None:
        asked_at = NOW - timedelta(hours=72)
        issue = question_issue(
            [comment("maintainer", "请补充版本信息和完整日志。", asked_at)]
        )
        watch = {
            "conversation_state": "awaiting_reporter",
            "last_maintainer_comment_at": asked_at.isoformat(),
            "waiting_since": asked_at.isoformat(),
        }

        result = AUTO_CLOSE.evaluate_issue(issue, POLICY, watch)

        assert result["eligible"] is True
        assert result["reason"] == "watched_awaiting_reporter_inactive"

    def test_reporter_reply_blocks_watched_information_request_close(self) -> None:
        asked_at = NOW - timedelta(hours=72)
        issue = question_issue(
            [
                comment("maintainer", "请补充完整日志。", asked_at),
                comment("reporter", "日志已补充。", NOW - timedelta(hours=60)),
            ]
        )
        watch = {
            "conversation_state": "awaiting_reporter",
            "last_maintainer_comment_at": asked_at.isoformat(),
            "waiting_since": asked_at.isoformat(),
        }

        result = AUTO_CLOSE.evaluate_issue(issue, POLICY, watch)

        assert result["eligible"] is False
        assert result["reason"] == "reporter_replied_after_handler"

    def test_waiting_for_assignee_is_never_auto_closed(self) -> None:
        asked_at = NOW - timedelta(hours=168)
        issue = question_issue(
            [comment("maintainer", "已联系算子责任人，请稍等", asked_at)]
        )
        watch = {
            "conversation_state": "awaiting_assignee",
            "assignee": "operator-owner",
            "last_maintainer_comment_at": asked_at.isoformat(),
            "waiting_since": asked_at.isoformat(),
        }

        result = AUTO_CLOSE.evaluate_issue(issue, POLICY, watch)

        assert result["eligible"] is False
        assert result["reason"] == "followup_watch_not_waiting"

    def test_newer_information_request_overrides_older_answer(self) -> None:
        issue = question_issue(
            [
                comment("maintainer", "这是预期行为。", NOW - timedelta(hours=96)),
                comment("maintainer", "请补充版本信息。", NOW - timedelta(hours=72)),
            ],
            assignee=None,
        )

        result = AUTO_CLOSE.evaluate_issue(issue, POLICY)

        assert result["eligible"] is False
        assert result["reason"] == "latest_handler_comment_not_answer"

    def test_non_question_is_skipped(self) -> None:
        issue = question_issue(
            [comment("maintainer", "已定位。", NOW - timedelta(hours=72))],
            title="[Bug] crash",
            labels=["bug"],
        )

        result = AUTO_CLOSE.evaluate_issue(issue, POLICY)

        assert result["eligible"] is False
        assert result["reason"] == "not_question"

    def test_existing_closure_comment_does_not_look_like_user_reply(self) -> None:
        issue = question_issue(
            [
                comment("maintainer", "这是预期行为。", NOW - timedelta(hours=72)),
                comment(
                    "automation", AUTO_CLOSE.DEFAULT_COMMENT, NOW - timedelta(hours=1)
                ),
            ]
        )

        result = AUTO_CLOSE.evaluate_issue(issue, POLICY)

        assert result["eligible"] is True

    def test_direct_pr_link_is_detected(self) -> None:
        issue = question_issue(
            [],
            description="已提交 https://gitcode.com/cann/ops-math/pull/123",
        )

        assert AUTO_CLOSE.issue_mentions_pr(issue) is True

    def test_native_pr_association_blocks_close(self) -> None:
        issue = question_issue([])

        reason = AUTO_CLOSE.pr_exclusion_reason(
            issue,
            {"42": [{"pr_number": 123}]},
            association_complete=True,
        )

        assert reason == "linked_pr"

    def test_incomplete_pr_scan_blocks_close(self) -> None:
        issue = question_issue([])

        reason = AUTO_CLOSE.pr_exclusion_reason(
            issue,
            {},
            association_complete=False,
        )

        assert reason == "association_scan_incomplete"


class TestCloseWorkflow:
    def test_comment_is_verified_before_issue_is_closed(self) -> None:
        old_reply = comment("maintainer", "这是预期行为。", NOW - timedelta(hours=72))
        closure = comment("automation", AUTO_CLOSE.DEFAULT_COMMENT, NOW)
        raw_issue = {
            "number": 42,
            "title": "[Question|API] 如何调用接口",
            "body": "请问接口如何使用？",
            "labels": ["question"],
            "state": "open",
            "user": {"login": "reporter"},
            "assignee": {"login": "maintainer"},
        }
        issue = question_issue([old_reply])

        with (
            patch.object(
                AUTO_CLOSE.issue_api, "get_single_issue", return_value=raw_issue
            ),
            patch.object(
                AUTO_CLOSE.issue_api,
                "get_issue_comments",
                side_effect=[[old_reply], [old_reply, closure]],
            ),
            patch.object(
                AUTO_CLOSE,
                "api_post",
                return_value=SimpleNamespace(status_code=201),
            ) as api_post,
            patch.object(
                AUTO_CLOSE,
                "api_patch",
                return_value=SimpleNamespace(status_code=200),
            ) as api_patch,
            patch.object(AUTO_CLOSE, "api_get", return_value={"state": "closed"}),
        ):
            api = AUTO_CLOSE.IssueApiContext(
                object(),
                "https://api.example.test",
                "owner",
                "repo",
                "token",
            )
            result = AUTO_CLOSE.close_issue(api, issue, POLICY)

        assert result["status"] == "closed"
        api_post.assert_called_once()
        api_patch.assert_called_once()

    def test_failed_comment_verification_prevents_close(self) -> None:
        old_reply = comment("maintainer", "这是预期行为。", NOW - timedelta(hours=72))
        raw_issue = {
            "number": 42,
            "title": "[Question|API] 如何调用接口",
            "body": "请问接口如何使用？",
            "labels": ["question"],
            "state": "open",
            "user": {"login": "reporter"},
            "assignee": {"login": "maintainer"},
        }

        with (
            patch.object(
                AUTO_CLOSE.issue_api, "get_single_issue", return_value=raw_issue
            ),
            patch.object(
                AUTO_CLOSE.issue_api,
                "get_issue_comments",
                side_effect=[[old_reply], [old_reply]],
            ),
            patch.object(
                AUTO_CLOSE,
                "api_post",
                return_value=SimpleNamespace(status_code=201),
            ),
            patch.object(AUTO_CLOSE, "api_patch") as api_patch,
        ):
            api = AUTO_CLOSE.IssueApiContext(
                object(),
                "https://api.example.test",
                "owner",
                "repo",
                "token",
            )
            result = AUTO_CLOSE.close_issue(api, question_issue([old_reply]), POLICY)

        assert result["status"] == "comment_failed"
        api_patch.assert_not_called()
