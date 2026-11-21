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


def test_updated_scan_excludes_closed_issue_and_uses_updated_order() -> None:
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

    assert issues == []
    assert diagnostics["complete"] is True
    params = api_get.call_args.kwargs["params"]
    assert params["state"] == "open"
    assert params["sort"] == "updated"
    assert params["direction"] == "desc"


def test_updated_scan_does_not_claim_completion_at_page_limit() -> None:
    boundary = datetime(2026, 8, 10, tzinfo=FETCHER.TZ_CHINA)
    full_page = [
        {
            "number": number,
            "state": "open",
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


def test_fetch_conflicting_target_stops_before_api(tmp_path, monkeypatch):
    import json
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text("repo: cann/math\n")
    with patch.object(FETCHER, "_write_stdout") as output, patch.object(FETCHER, "resolve_token") as token:
        code = FETCHER.main(["--config", str(config), "--url", "https://gitcode.com/user/math"])
    assert code == 2
    assert json.loads(output.call_args.args[0])["status"] == "needs_selection"
    token.assert_not_called()
    assert config.read_text() == "repo: cann/math\n"


def test_fetch_uses_configured_repo_without_url(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GITCODE_URL", raising=False)
    config = tmp_path / "config.yaml"
    config.write_text("repo: cann/math\n")
    with patch.object(FETCHER, "_write_stdout"), patch.object(FETCHER, "resolve_token", return_value="test"), patch.object(FETCHER, "_batch_output", return_value={}) as fetch:
        code = FETCHER.main(["--config", str(config)])
    assert code == 0
    assert fetch.call_args.args[0].url == "https://gitcode.com/cann/math"


def test_fetch_discovers_remote_and_persists_before_fetch(tmp_path, monkeypatch):
    import subprocess
    import yaml
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GITCODE_URL", raising=False)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "remote", "add", "custom-name", "git@gitcode.com:team/math.git"], check=True)
    with patch.object(FETCHER, "_write_stdout"), patch.object(FETCHER, "resolve_token", return_value="test"), patch.object(FETCHER, "_batch_output", return_value={}) as fetch:
        code = FETCHER.main([])
    assert code == 0
    args = fetch.call_args.args[0]
    assert args.url == "https://gitcode.com/team/math"
    assert yaml.safe_load(Path(args.config).read_text())["repo"] == "team/math"


def test_environment_target_conflict_requires_selection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITCODE_URL", "https://gitcode.com/other/math")
    config = tmp_path / "config.yaml"
    config.write_text("repo: team/math\n")
    with patch.object(FETCHER, "_write_stdout"), patch.object(FETCHER, "resolve_token") as token:
        assert FETCHER.main(["--config", str(config)]) == 2
    token.assert_not_called()


@pytest.mark.parametrize("state", ["all", "closed"])
def test_normal_intake_rejects_non_open_state(state):
    with pytest.raises(SystemExit):
        FETCHER.parse_args(["--state", state])


def test_low_level_all_state_fetch_remains_available():
    raw = [{"number": 1, "state": "closed"}]
    with patch.object(FETCHER, "api_get", return_value=raw) as get:
        assert FETCHER.get_issues(API, state="all") == raw
    assert get.call_args.kwargs["params"]["state"] == "all"


def test_single_closed_issue_skips_comments_even_with_refresh(tmp_path):
    args = FETCHER.parse_args([
        "--issue", "https://gitcode.com/owner/repo/issues/42",
        "--refresh-comments", "--cache-dir", str(tmp_path),
    ])
    with (
        patch.object(FETCHER, "make_session", return_value=object()),
        patch.object(FETCHER, "get_single_issue", return_value={
            "number": 42, "state": "closed", "comments": 5,
        }),
        patch.object(FETCHER, "get_issue_comments") as comments,
        patch.object(FETCHER, "load_comments") as cache,
    ):
        output = FETCHER._single_issue_output(args, "token")
    comments.assert_not_called()
    cache.assert_not_called()
    assert output["issues"][0]["comments_fetch"] == {
        "status": "skipped", "reason": "issue_not_open",
    }


def test_watch_refresh_reuses_current_state_but_checks_missing_state():
    watched = {str(n): {"waiting_on": "reporter"} for n in (1, 2, 3, 4)}
    current = [
        {"number": 1, "state": "open"},
        {"number": 2, "state": "closed"},
        {"number": 3},
    ]
    with patch.object(FETCHER, "get_single_issue", side_effect=[
        {"number": 3, "state": "closed"},
        {"number": 4, "state": "open"},
    ]) as get:
        issues, diagnostics = FETCHER.get_watched_issues(
            API, watched, current_issues=current,
        )
    assert [call.args[1] for call in get.call_args_list] == ["3", "4"]
    assert [item["number"] for item in issues] == [1, 4]
    assert diagnostics["reused"] == 2
    assert diagnostics["closed"] == ["2", "3"]
    assert "followup_watch" not in current[0]


def test_closed_watch_is_removed_even_when_updated_scan_is_incomplete(tmp_path):
    import followup_state
    state_file = tmp_path / "watch.json"
    followup_state.watch_issue(
        state_file, "owner/repo", "42", reporter="reporter",
        maintainer_comment_at="2026-08-10T00:00:00Z",
    )
    followup_state.advance_updated_cursor(state_file, "owner/repo", "2026-08-10T00:00:00Z")
    args = FETCHER.parse_args(["--follow-up-state-file", str(state_file)])
    with (
        patch.object(FETCHER, "get_updated_issues", return_value=([], {"complete": False})),
        patch.object(FETCHER, "get_single_issue", return_value={"number": 42, "state": "closed"}),
    ):
        issues, diagnostics = FETCHER._collect_followup_sources(args, API, "owner/repo", [])
    saved = followup_state.load_followup_state(state_file, "owner/repo")
    assert issues == []
    assert saved["issues"] == {}
    assert saved["updated_cursor"] == "2026-08-10T00:00:00Z"
    assert diagnostics["cursor_advanced"] is False


def test_invalid_watch_state_keeps_watch_for_retry(tmp_path):
    import followup_state
    state_file = tmp_path / "watch.json"
    followup_state.watch_issue(
        state_file, "owner/repo", "42", reporter="reporter",
        maintainer_comment_at="2026-08-10T00:00:00Z",
    )
    args = FETCHER.parse_args(["--follow-up-state-file", str(state_file)])
    with (
        patch.object(FETCHER, "get_updated_issues", return_value=([], {"complete": True})),
        patch.object(FETCHER, "get_single_issue", return_value={"number": 42}),
    ):
        issues, diagnostics = FETCHER._collect_followup_sources(args, API, "owner/repo", [])
    assert issues == []
    assert "42" in followup_state.load_followup_state(state_file, "owner/repo")["issues"]
    assert diagnostics["watchlist_refresh"]["complete"] is False


def test_batch_default_uses_open_for_both_scans_and_reuses_watch(tmp_path):
    import followup_state
    state_file = tmp_path / "watch.json"
    followup_state.watch_issue(
        state_file, "owner/repo", "42", reporter="reporter",
        maintainer_comment_at="2026-08-10T00:00:00Z",
    )
    args = FETCHER.parse_args([
        "--url", "https://gitcode.com/owner/repo",
        "--follow-up-state-file", str(state_file),
    ])
    raw = {"number": 42, "state": "open", "updated_at": "2099-08-10T00:00:00Z"}
    with (
        patch.object(FETCHER, "make_session", return_value=object()),
        patch.object(FETCHER, "api_get", return_value=[raw]) as get,
        patch.object(FETCHER, "get_single_issue") as single,
    ):
        output = FETCHER._batch_output(args, "token")
    assert get.call_count == 2
    assert [c.kwargs["params"]["state"] for c in get.call_args_list] == ["open", "open"]
    single.assert_not_called()
    assert output["total"] == 1
    assert output["issues"][0]["fetch_sources"] == ["primary", "updated", "watchlist"]
    assert output["filters"]["follow_up"]["cursor_advanced"] is True


def test_closed_updated_snapshot_overrides_primary_open_watch(tmp_path):
    import followup_state
    state_file = tmp_path / "watch.json"
    followup_state.watch_issue(
        state_file, "owner/repo", "42", reporter="reporter",
        maintainer_comment_at="2026-08-10T00:00:00Z",
    )
    args = FETCHER.parse_args(["--follow-up-state-file", str(state_file)])
    closed = {"number": 42, "state": "closed", "updated_at": "2099-08-10T00:00:00Z"}
    with (
        patch.object(FETCHER, "api_get", return_value=[closed]),
        patch.object(FETCHER, "get_single_issue") as single,
    ):
        issues, diagnostics = FETCHER._collect_followup_sources(
            args, API, "owner/repo", [{"number": 42, "state": "open"}],
        )
    assert issues == []
    assert followup_state.load_followup_state(state_file, "owner/repo")["issues"] == {}
    single.assert_not_called()
    assert diagnostics["watchlist_refresh"]["closed"] == ["42"]


@pytest.mark.parametrize("row", [{"number": 42}, "invalid"])
def test_updated_scan_invalid_row_never_advances_cursor(row):
    boundary = datetime(2026, 8, 10, tzinfo=FETCHER.TZ_CHINA)
    with patch.object(FETCHER, "api_get", return_value=[row]):
        issues, diagnostics = FETCHER.get_updated_issues(API, boundary)
    assert issues == []
    assert diagnostics["complete"] is False


@pytest.mark.parametrize("failure", [requests.HTTPError("unavailable"), {"error": "invalid"}])
def test_primary_scan_preserves_partial_results_and_reports_failure(failure):
    page = [{"number": n, "state": "open"} for n in range(1, 101)]
    diagnostics = {}
    with patch.object(FETCHER, "api_get", side_effect=[page, failure]):
        issues = FETCHER.get_issues(API, diagnostics=diagnostics)
    assert issues == page
    assert diagnostics["complete"] is False
    assert diagnostics["pages_requested"] == 2
    assert diagnostics["warnings"]


def test_primary_scan_full_page_requires_next_page_before_completion():
    page = [{"number": n, "state": "open"} for n in range(1, 101)]
    diagnostics = {}
    with patch.object(FETCHER, "api_get", side_effect=[page, []]):
        assert FETCHER.get_issues(API, diagnostics=diagnostics) == page
    assert diagnostics["complete"] is True
    assert diagnostics["pages_requested"] == 2


@pytest.mark.parametrize("row", ["invalid", {"state": "open"}, {"number": 42}])
def test_primary_scan_invalid_row_is_incomplete(row):
    diagnostics = {}
    with patch.object(FETCHER, "api_get", return_value=[row]):
        FETCHER.get_issues(API, diagnostics=diagnostics)
    assert diagnostics["complete"] is False


def test_defer_cursor_preserves_watch_file_and_proposes_changes(tmp_path):
    import followup_state
    state_file = tmp_path / "watch.json"
    followup_state.watch_issue(
        state_file, "owner/repo", "42", reporter="reporter",
        maintainer_comment_at="2026-08-10T00:00:00Z",
    )
    before = state_file.read_bytes()
    args = FETCHER.parse_args(["--defer-cursor", "--follow-up-state-file", str(state_file)])
    with (
        patch.object(FETCHER, "get_updated_issues", return_value=([], {"complete": True})),
        patch.object(FETCHER, "get_single_issue", return_value={"number": 42, "state": "closed"}),
        patch.object(FETCHER, "advance_updated_cursor") as advance,
        patch.object(FETCHER, "resolve_issue") as resolve,
    ):
        _, diagnostics = FETCHER._collect_followup_sources(args, API, "owner/repo", [])
    assert diagnostics["cursor_deferred"] is True
    assert diagnostics["cursor_advanced"] is False
    assert diagnostics["cursor_proposed"]
    assert diagnostics["closed_watches"] == ["42"]
    assert diagnostics["complete"] is True
    assert state_file.read_bytes() == before
    advance.assert_not_called()
    resolve.assert_not_called()


@pytest.mark.parametrize("updated_complete,watch_complete", [(False, True), (True, False)])
def test_defer_cursor_does_not_propose_after_incomplete_followup(updated_complete, watch_complete):
    args = FETCHER.parse_args(["--defer-cursor"])
    with (
        patch.object(FETCHER, "load_followup_state", return_value={}),
        patch.object(FETCHER, "get_updated_issues", return_value=([], {"complete": updated_complete})),
        patch.object(FETCHER, "get_watched_issues", return_value=([], {"complete": watch_complete})),
        patch.object(FETCHER, "advance_updated_cursor") as advance,
    ):
        _, diagnostics = FETCHER._collect_followup_sources(args, API, "owner/repo", [])
    assert diagnostics["complete"] is False
    assert diagnostics["cursor_proposed"] is None
    advance.assert_not_called()


def test_incremental_boundary_skips_primary_and_overrides_legacy_cursor():
    args = FETCHER.parse_args([
        "--url", "https://gitcode.com/owner/repo", "--defer-cursor",
        "--updated-since", "2026-08-01T00:00:00Z",
    ])
    with (
        patch.object(FETCHER, "make_session", return_value=object()),
        patch.object(FETCHER, "get_issues") as primary,
        patch.object(FETCHER, "load_followup_state", return_value={"updated_cursor": "2026-09-01T00:00:00Z"}),
        patch.object(FETCHER, "get_updated_issues", return_value=([], {"complete": True})) as updated,
        patch.object(FETCHER, "get_watched_issues", return_value=([], {"complete": True})),
    ):
        output = FETCHER._batch_output(args, "token")
    primary.assert_not_called()
    assert updated.call_args.args[1] == FETCHER.parse_iso(args.updated_since)
    assert output["primary_scan"]["skipped"] is True
    assert output["filters"]["follow_up"]["cursor_proposed"]


def test_incomplete_primary_prevents_deferred_cursor_proposal():
    args = FETCHER.parse_args(["--url", "https://gitcode.com/owner/repo", "--defer-cursor"])
    with (
        patch.object(FETCHER, "make_session", return_value=object()),
        patch.object(FETCHER, "api_get", side_effect=requests.HTTPError("failed")),
        patch.object(FETCHER, "_collect_followup_sources", return_value=([], {"cursor_proposed": "later"})),
    ):
        output = FETCHER._batch_output(args, "token")
    assert output["primary_scan"]["complete"] is False
    assert output["filters"]["follow_up"]["cursor_proposed"] is None


def test_atomic_output_replaces_snapshot_and_preserves_stdout(tmp_path):
    import json
    destination = tmp_path / "nested" / "issues.json"
    output = {"issues": [{"iid": 42, "title": "测试"}]}
    with patch.object(FETCHER, "_write_stdout") as stdout:
        FETCHER._write_output(output, str(destination))
    assert json.loads(destination.read_text()) == output
    assert json.loads(stdout.call_args.args[0]) == output
    assert list(destination.parent.iterdir()) == [destination]


def test_atomic_output_failure_preserves_existing_snapshot(tmp_path):
    destination = tmp_path / "issues.json"
    destination.write_text('{"old": true}')
    with (
        patch.object(FETCHER.os, "replace", side_effect=OSError("write failed")),
        patch.object(FETCHER, "_write_stdout") as stdout,
        pytest.raises(OSError),
    ):
        FETCHER._write_output({"new": True}, str(destination))
    assert destination.read_text() == '{"old": true}'
    assert list(tmp_path.iterdir()) == [destination]
    stdout.assert_not_called()


@pytest.mark.parametrize("timestamp", ["invalid", "2026-08-01", "2026-08-01T00:00:00"])
def test_updated_boundary_requires_timezone(timestamp):
    with pytest.raises(SystemExit):
        FETCHER.parse_args(["--updated-since", timestamp])


def test_main_saves_output_snapshot(tmp_path, monkeypatch):
    import json
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text("repo: owner/repo\n")
    snapshot = tmp_path / "snapshot.json"
    output = {"total": 0, "issues": [], "primary_scan": {"complete": True}}
    with (
        patch.object(FETCHER, "resolve_token", return_value="test"),
        patch.object(FETCHER, "_batch_output", return_value=output),
        patch.object(FETCHER, "_write_stdout"),
    ):
        result = FETCHER.main([
            "--config", str(config), "--output", str(snapshot), "--defer-cursor",
        ])
    assert result == 0
    assert json.loads(snapshot.read_text()) == output
