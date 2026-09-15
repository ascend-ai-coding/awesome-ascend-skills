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
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import requests
import pytest

HANDLER_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = HANDLER_ROOT / "scripts" / "fetch_issues.py"
SPEC = importlib.util.spec_from_file_location("fetch_issues", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
FETCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FETCHER)
API = FETCHER.RepoApiContext(
    object(), "https://api.example.test", "owner", "repo", "token"
)


def issue(number: int, created_at: datetime) -> dict:
    return {"number": number, "created_at": created_at.isoformat()}


def sample_issues() -> list[dict]:
    return [
        {
            "iid": 1,
            "number": 1,
            "updated_at": "2026-08-11T08:00:00+08:00",
            "comments_count": 1,
        },
        {
            "iid": 2,
            "number": 2,
            "updated_at": "2026-08-11T09:00:00+08:00",
            "comments_count": 1,
        },
    ]


@pytest.mark.parametrize(
    ("comment_count", "expected_pages"),
    ((99, 1), (100, 2), (101, 2)),
)
def test_comment_pagination_counts_real_pages(comment_count, expected_pages) -> None:
    raw = [
        {"id": number, "user": {"login": "author"}, "body": "text"}
        for number in range(comment_count)
    ]

    def page_response(_session, _url, _token, *, params):
        start = (params["page"] - 1) * params["per_page"]
        return raw[start : start + params["per_page"]]

    diagnostics = {"comment_pages": 0}
    with patch.object(FETCHER, "api_get", side_effect=page_response):
        comments = FETCHER.get_issue_comments(
            API,
            42,
            page_diagnostics=diagnostics,
        )

    assert len(comments) == comment_count
    assert diagnostics["comment_pages"] == expected_pages


def test_fetch_cache_resolves_shared_rate_limit_directory(tmp_path) -> None:
    cache_dir = tmp_path / "cache" / "issues"

    assert FETCHER.rate_limit_path(cache_dir) == cache_dir.parent / "gitcode-rate-limit"


def test_since_boundary_stops_pagination():
    since = datetime(2026, 8, 7, tzinfo=FETCHER.TZ_CHINA)
    page = [issue(number, since + timedelta(hours=1)) for number in range(1, 100)]
    page.append(issue(100, since - timedelta(seconds=1)))

    with patch.object(FETCHER, "api_get", return_value=page) as api_get:
        result = FETCHER.get_issues(
            API,
            created_since=since,
        )

    assert len(result) == 99
    assert api_get.call_count == 1


def test_until_boundary_filters_newer_issues():
    start = datetime(2026, 8, 10, tzinfo=FETCHER.TZ_CHINA)
    page = [
        issue(1, start + timedelta(days=1)),
        issue(2, start + timedelta(hours=2)),
        issue(3, start + timedelta(hours=1)),
    ]

    with patch.object(FETCHER, "api_get", return_value=page):
        result = FETCHER.get_issues(
            API,
            created_until_exclusive=start + timedelta(days=1),
        )

    assert [item["number"] for item in result] == [2, 3]


def test_partial_failure_is_cached_and_next_run_resumes(tmp_path: Path):
    first_comment = [{"author": "maintainer", "body": "已受理"}]
    second_comment = [{"author": "maintainer", "body": "已修复"}]
    issues = sample_issues()

    with patch.object(
        FETCHER,
        "get_issue_comments",
        side_effect=[first_comment, requests.HTTPError("rate limited")],
    ):
        first = FETCHER.enrich_issues_with_comments(
            API,
            issues,
            cache_dir=tmp_path,
        )

    assert first["complete"] is False
    assert issues[0]["comments_fetch"]["status"] == "api"
    assert issues[1]["comments_fetch"]["status"] == "error"

    resumed = sample_issues()
    with patch.object(
        FETCHER,
        "get_issue_comments",
        return_value=second_comment,
    ) as get_comments:
        second = FETCHER.enrich_issues_with_comments(
            API,
            resumed,
            cache_dir=tmp_path,
        )

    assert second["complete"] is True
    assert second["cache_hits"] == 1
    assert get_comments.call_count == 1
    assert get_comments.call_args.args[-1] == 2


def test_zero_comment_issue_never_calls_api():
    issue_without_comments = {
        "iid": 3,
        "updated_at": "2026-08-11T10:00:00+08:00",
        "comments_count": 0,
    }
    with patch.object(FETCHER, "get_issue_comments") as get_comments:
        result = FETCHER.enrich_issues_with_comments(
            API,
            [issue_without_comments],
            cache_dir=None,
        )

    get_comments.assert_not_called()
    assert result["skipped"] == 1
    assert (
        issue_without_comments.get("comments_fetch", {}).get("reason") == "no_comments"
    )


def test_updated_scan_includes_closed_issue_and_uses_updated_order() -> None:
    boundary = datetime(2026, 8, 10, tzinfo=FETCHER.TZ_CHINA)
    page = [
        {
            "number": 2535,
            "state": "closed",
            "updated_at": "2026-08-13T09:00:00+08:00",
        }
    ]

    with patch.object(FETCHER, "api_get", return_value=page) as api_get:
        issues, diagnostics = FETCHER.get_updated_issues(API, boundary)

    assert [item["number"] for item in issues] == [2535]
    assert diagnostics["complete"] is True
    params = api_get.call_args.kwargs["params"]
    assert params["state"] == "all"
    assert params["sort"] == "updated"
    assert params["direction"] == "desc"


def test_updated_scan_does_not_claim_completion_at_page_limit() -> None:
    boundary = datetime(2026, 8, 10, tzinfo=FETCHER.TZ_CHINA)
    full_page = [
        {
            "number": number,
            "updated_at": "2026-08-13T09:00:00+08:00",
        }
        for number in range(100)
    ]

    with patch.object(FETCHER, "api_get", return_value=full_page):
        _, diagnostics = FETCHER.get_updated_issues(API, boundary, max_pages=1)

    assert diagnostics["complete"] is False
    assert "page limit" in diagnostics["warnings"][0]


def test_updated_scan_does_not_advance_on_invalid_response_shape() -> None:
    boundary = datetime(2026, 8, 10, tzinfo=FETCHER.TZ_CHINA)

    with patch.object(FETCHER, "api_get", return_value={"unexpected": "shape"}):
        issues, diagnostics = FETCHER.get_updated_issues(API, boundary)

    assert issues == []
    assert diagnostics["complete"] is False
    assert "invalid response shape" in diagnostics["warnings"][0]


def test_merge_sources_deduplicates_and_preserves_watch() -> None:
    merged = FETCHER.merge_issue_sources(
        ("primary", [{"number": 42, "title": "old"}]),
        ("updated", [{"number": 42, "title": "new"}]),
        (
            "watchlist",
            [
                {
                    "number": 42,
                    "followup_watch": {"conversation_state": "awaiting_assignee"},
                }
            ],
        ),
    )

    assert len(merged) == 1
    assert merged[0]["title"] == "new"
    assert merged[0]["fetch_sources"] == ["primary", "updated", "watchlist"]
    assert merged[0]["followup_watch"]["conversation_state"] == ("awaiting_assignee")


def test_normalize_keeps_custom_issue_state() -> None:
    normalized = FETCHER.normalize_issue(
        {
            "number": 42,
            "state": "open",
            "issue_state": "挂起",
            "issue_state_detail": {"id": 917},
        }
    )

    assert normalized["issue_state"] == "挂起"
    assert normalized["issue_state_id"] == 917


def test_created_time_filter_never_drops_followup_sources() -> None:
    old = {
        "number": 42,
        "created_at": "2025-01-01T00:00:00+08:00",
        "fetch_sources": ["updated"],
    }

    result = FETCHER.filter_issues_by_time(
        [old], since=datetime(2026, 8, 10, tzinfo=FETCHER.TZ_CHINA)
    )

    assert result == [old]


def test_batch_keeps_cursor_when_updated_scan_is_incomplete() -> None:
    args = FETCHER.parse_args(
        ["--url", "https://gitcode.com/cann/ops-math", "--token", "token"]
    )
    state = {
        "schema_version": "issue-followup.v1",
        "repository": "cann/ops-math",
        "updated_cursor": "2026-08-10T00:00:00+08:00",
        "issues": {},
    }
    diagnostics = {
        "complete": False,
        "pages_requested": 1,
        "boundary": state["updated_cursor"],
        "warnings": ["page failed"],
    }

    with (
        patch.object(FETCHER, "make_session", return_value=object()),
        patch.object(FETCHER, "get_issues", return_value=[]),
        patch.object(FETCHER, "load_followup_state", return_value=state),
        patch.object(FETCHER, "get_updated_issues", return_value=([], diagnostics)),
        patch.object(FETCHER, "get_watched_issues", return_value=([], {})),
        patch.object(FETCHER, "advance_updated_cursor") as advance,
    ):
        output = FETCHER._batch_output(args, "token")

    assert output["filters"]["follow_up"]["cursor_advanced"] is False
    advance.assert_not_called()


def test_followup_fetch_options_use_config_and_allow_cli_override(
    tmp_path: Path,
) -> None:
    config = tmp_path / "classify.yaml"
    config.write_text(
        "follow_up:\n"
        "  state_file: custom/watch.json\n"
        "  lookback_days: 45\n"
        "  fetch_pages: 12\n",
        encoding="utf-8",
    )
    args = FETCHER.parse_args(
        [
            "--url",
            "https://gitcode.com/cann/ops-math",
            "--config",
            str(config),
            "--follow-up-fetch-pages",
            "3",
        ]
    )

    state_file, lookback, pages = FETCHER._followup_options(args)

    assert state_file == "custom/watch.json"
    assert lookback == 45
    assert pages == 3
