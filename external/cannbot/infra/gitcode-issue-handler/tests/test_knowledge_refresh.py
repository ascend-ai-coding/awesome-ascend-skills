# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

HANDLER_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = HANDLER_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REFRESH = load_module("refresh_issue_knowledge", "refresh_issue_knowledge.py")
QUERY = load_module("knowledge_query_runtime", "knowledge_query.py")


def args(tmp_path: Path, **overrides):
    values = {
        "url": "https://gitcode.com/cann/ops-math",
        "token": "test-token",
        "api_base": "https://api.example.test",
        "workers": 1,
        "request_interval": 0,
        "cache_dir": str(tmp_path / "cache"),
        "include_self_assigned": False,
        "skip_prs": False,
        "force_full": False,
        "ttl_seconds": 900,
        "full_interval_seconds": 7 * 24 * 60 * 60,
        "overlap_seconds": 300,
        "lock_timeout": 0,
        "output": str(tmp_path / "issue-history.json"),
        "report": str(tmp_path / "knowledge-corpus.md"),
        "state_file": str(tmp_path / "issue-knowledge-state.json"),
        "lock_file": str(tmp_path / "refresh.lock"),
        "candidates_per_type": 5,
        "strict": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def issue(number: int, updated_at: str, **overrides):
    payload = {
        "number": number,
        "title": f"MatMul error {number}",
        "body": "precision mismatch",
        "labels": [],
        "state": "open",
        "created_at": "2026-08-01T00:00:00+08:00",
        "updated_at": updated_at,
        "html_url": f"https://gitcode.com/cann/ops-math/issues/{number}",
        "user": {"login": "reporter"},
        "assignee": None,
        "comments": 0,
    }
    payload.update(overrides)
    return payload


def make_corpus(repository="cann/ops-math", generated_at="2026-08-13T10:00:00+08:00"):
    return {
        "metadata": {
            "schema_version": REFRESH.CORPUS_SCHEMA,
            "generated_at": generated_at,
            "source_url": "https://gitcode.com/cann/ops-math",
            "repository": repository,
            "filter": "author != assignee, including missing assignee",
            "trust": "runtime_evidence_only",
        },
        "summary": {},
        "issues": [
            {
                "number": 7,
                "url": "https://gitcode.com/cann/ops-math/issues/7",
                "title": "MatMul precision mismatch",
                "description": "float16 result differs",
                "labels": [],
                "state": "open",
                "created_at": "2026-08-01T00:00:00+08:00",
                "updated_at": "2026-08-13T09:00:00+08:00",
                "author": "reporter",
                "assignee": None,
                "issue_type": "bug",
                "linked_prs": [
                    {
                        "number": 70,
                        "state": "open",
                        "title": "Fix #7",
                        "url": "https://gitcode.com/cann/ops-math/pull/70",
                    }
                ],
                "comments": [{"id": 701, "author": "maintainer", "body": "known"}],
                "evidence_signals": ["linked_change"],
                "handling_outcome": "linked_change",
            }
        ],
    }


def commit_fixture(tmp_path: Path, *, now=None, last_full=None):
    now = now or datetime(2026, 8, 13, 10, tzinfo=REFRESH.issue_api.TZ_CHINA)
    last_full = last_full or now
    corpus = make_corpus(generated_at=REFRESH._iso(now))
    data = REFRESH._json_bytes(corpus)
    corpus_path = tmp_path / "issue-history.json"
    state_path = tmp_path / "issue-knowledge-state.json"
    corpus_path.write_bytes(data)
    state = {
        "schema_version": REFRESH.STATE_SCHEMA,
        "corpus_schema_version": REFRESH.CORPUS_SCHEMA,
        "repository": "cann/ops-math",
        "last_success_at": REFRESH._iso(now),
        "last_full_refresh_at": REFRESH._iso(last_full),
        "cursor_at": REFRESH._iso(now),
        "last_refresh_status": "success",
        "corpus_sha256": hashlib.sha256(data).hexdigest(),
        "source_total": 1,
        "excluded_issue_numbers": [],
        "pr_records": {
            "70": {
                "number": 70,
                "state": "open",
                "title": "Fix #7",
                "url": "https://gitcode.com/cann/ops-math/pull/70",
                "updated_at": "2026-08-12T00:00:00+08:00",
                "head_sha": "abc",
                "issue_refs": ["7"],
            }
        },
        "policy": {"include_self_assigned": False, "skip_prs": False},
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")
    return corpus, state


def append_unchanged_issue(corpus: dict) -> None:
    corpus["issues"].append(
        {
            "number": 8,
            "url": "https://gitcode.com/cann/ops-math/issues/8",
            "title": "Another unchanged issue",
            "description": "kept across merge",
            "labels": [],
            "state": "closed",
            "created_at": "2026-08-01T00:00:00+08:00",
            "updated_at": "2026-08-12T08:00:00+08:00",
            "author": "other",
            "assignee": None,
            "issue_type": "other",
            "linked_prs": [],
            "comments": [{"id": 801, "author": "maintainer", "body": "preserve me"}],
            "evidence_signals": ["other_reply"],
            "handling_outcome": "replied_without_strong_outcome_evidence",
        }
    )


def test_refresh_lock_timeout_preserves_original_blocking_error(tmp_path: Path):
    lock = REFRESH.RefreshLock(tmp_path / "refresh.lock", timeout=0)
    blocking_error = BlockingIOError("lock is already held")

    with (
        patch.object(REFRESH.fcntl, "flock", side_effect=blocking_error),
        patch.object(REFRESH.time, "monotonic", return_value=100.0),
        pytest.raises(TimeoutError, match="knowledge refresh lock is busy") as raised,
    ):
        lock.__enter__()

    assert raised.value.__cause__ is blocking_error
    assert lock.file_obj is None


def test_issue_reference_extractor_is_the_public_pr_parsing_contract():
    text = (
        "Fix #7 twice #7; ignore ##8; see "
        "https://gitcode.com/cann/ops-math/issues/9"
    )

    assert REFRESH.corpus_builder.extract_issue_references(text) == {"7", "9"}
    assert REFRESH._pr_record({"title": text, "body": "Closes #10 and #2"})[
        "issue_refs"
    ] == ["2", "7", "9", "10"]


def test_choose_mode_bootstrap_ttl_incremental_and_periodic_full(tmp_path: Path):
    run_args = args(tmp_path)
    now = datetime(2026, 8, 13, 10, tzinfo=REFRESH.issue_api.TZ_CHINA)
    assert REFRESH.choose_refresh_mode(run_args, None, now) == "full"

    corpus, state = commit_fixture(tmp_path, now=now - timedelta(minutes=5))
    assert REFRESH.choose_refresh_mode(run_args, (corpus, state), now) == "skip_fresh"

    corpus, state = commit_fixture(
        tmp_path,
        now=now - timedelta(hours=1),
        last_full=now - timedelta(days=1),
    )
    assert REFRESH.choose_refresh_mode(run_args, (corpus, state), now) == "incremental"

    corpus, state = commit_fixture(
        tmp_path,
        now=now - timedelta(hours=1),
        last_full=now - timedelta(days=8),
    )
    assert REFRESH.choose_refresh_mode(run_args, (corpus, state), now) == "full"


def test_schema_digest_repository_and_policy_force_full(tmp_path: Path):
    run_args = args(tmp_path)
    now = datetime(2026, 8, 13, 12, tzinfo=REFRESH.issue_api.TZ_CHINA)
    corpus, state = commit_fixture(tmp_path, now=now - timedelta(hours=1))

    state["repository"] = "cann/other"
    assert REFRESH.choose_refresh_mode(run_args, (corpus, state), now) == "full"
    state["repository"] = "cann/ops-math"
    state["policy"]["skip_prs"] = True
    assert REFRESH.choose_refresh_mode(run_args, (corpus, state), now) == "full"

    tampered = make_corpus()
    tampered["issues"][0]["title"] = "tampered after commit"
    (tmp_path / "issue-history.json").write_text(
        json.dumps(tampered, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(REFRESH.SnapshotError, match="digest"):
        REFRESH.load_valid_snapshot(
            tmp_path / "issue-history.json", tmp_path / "issue-knowledge-state.json"
        )


def test_incremental_fetch_uses_overlap_and_stops_at_updated_boundary(tmp_path: Path):
    api = REFRESH.issue_api.RepoApiContext(
        object(), "https://api.example.test", "cann", "ops-math", "token"
    )
    boundary = datetime(2026, 8, 13, 9, tzinfo=REFRESH.issue_api.TZ_CHINA)
    page = [
        issue(1, "2026-08-13T10:00:00+08:00"),
        issue(2, "2026-08-13T08:59:59+08:00"),
    ]
    with patch.object(REFRESH.corpus_builder, "rate_limited_get", return_value=page) as get:
        result = REFRESH._fetch_updated_items(
            api, REFRESH.corpus_builder.RequestLimiter(0), "issues", boundary
        )

    assert [item["number"] for item in result] == [1]
    assert get.call_count == 1
    assert get.call_args.kwargs["params"]["sort"] == "updated"


def test_incremental_order_violation_falls_back_to_full(tmp_path: Path):
    run_args = args(tmp_path)
    scan_started = datetime(2026, 8, 13, 12, tzinfo=REFRESH.issue_api.TZ_CHINA)
    snapshot = commit_fixture(tmp_path, now=scan_started - timedelta(hours=1))
    expected = (make_corpus(), {"last_refresh_mode": "full"})
    with (
        patch.object(
            REFRESH, "_incremental_refresh", side_effect=REFRESH.IncrementalOrderError()
        ),
        patch.object(REFRESH, "_full_refresh", return_value=expected) as full,
    ):
        result = REFRESH.perform_refresh(
            run_args, "incremental", snapshot, scan_started
        )
    assert result == expected
    full.assert_called_once()


def test_self_assignment_transition_removes_issue_without_losing_total(tmp_path: Path):
    run_args = args(tmp_path, skip_prs=True)
    now = datetime(2026, 8, 13, 12, tzinfo=REFRESH.issue_api.TZ_CHINA)
    corpus, state = commit_fixture(tmp_path, now=now - timedelta(hours=1))
    state["policy"]["skip_prs"] = True
    self_assigned = issue(
        7,
        "2026-08-13T11:30:00+08:00",
        assignee={"login": "reporter"},
    )
    with (
        patch.object(REFRESH.issue_api, "resolve_token", return_value="token"),
        patch.object(REFRESH, "_fetch_updated_items", return_value=[self_assigned]),
        patch.object(REFRESH.corpus_builder, "fetch_comments"),
    ):
        refreshed, refreshed_state = REFRESH._incremental_refresh(
            run_args, (corpus, state), now
        )

    assert refreshed["issues"] == []
    assert refreshed["summary"]["source_total"] == 1
    assert refreshed["summary"]["excluded_self_assigned"] == 1
    assert refreshed_state["excluded_issue_numbers"] == ["7"]


def test_incremental_preserves_unchanged_compact_issue_number_comments_and_pr(
    tmp_path: Path,
):
    run_args = args(tmp_path)
    now = datetime(2026, 8, 13, 12, tzinfo=REFRESH.issue_api.TZ_CHINA)
    corpus, state = commit_fixture(tmp_path, now=now - timedelta(hours=1))
    append_unchanged_issue(corpus)
    state["source_total"] = 2
    changed = issue(8, "2026-08-13T11:30:00+08:00", title="Updated issue 8")
    with (
        patch.object(REFRESH.issue_api, "resolve_token", return_value="token"),
        patch.object(
            REFRESH,
            "_fetch_updated_items",
            side_effect=[[changed], []],
        ),
        patch.object(
            REFRESH.corpus_builder,
            "fetch_comments",
            side_effect=lambda issues, *_args, **_kwargs: [
                item.update(comments=[]) for item in issues
            ],
        ),
    ):
        refreshed, _ = REFRESH._incremental_refresh(
            run_args, (corpus, state), now
        )

    by_number = {item["number"]: item for item in refreshed["issues"]}
    assert set(by_number) == {7, 8}
    assert by_number[7]["comments"] == [
        {"id": 701, "author": "maintainer", "body": "known"}
    ]
    assert by_number[7]["linked_prs"][0]["number"] == 70
    assert by_number[8]["title"] == "Updated issue 8"


def test_self_assigned_issue_can_reenter_on_later_increment(tmp_path: Path):
    run_args = args(tmp_path, skip_prs=True)
    now = datetime(2026, 8, 13, 12, tzinfo=REFRESH.issue_api.TZ_CHINA)
    corpus = make_corpus()
    corpus["issues"] = []
    _, state = commit_fixture(tmp_path, now=now - timedelta(hours=1))
    state["policy"]["skip_prs"] = True
    state["excluded_issue_numbers"] = ["7"]
    non_self = issue(
        7,
        "2026-08-13T11:30:00+08:00",
        assignee={"login": "maintainer"},
    )
    with (
        patch.object(REFRESH.issue_api, "resolve_token", return_value="token"),
        patch.object(REFRESH, "_fetch_updated_items", return_value=[non_self]),
        patch.object(
            REFRESH.corpus_builder,
            "fetch_comments",
            side_effect=lambda issues, *_args, **_kwargs: [
                item.update(comments=[]) for item in issues
            ],
        ),
    ):
        refreshed, refreshed_state = REFRESH._incremental_refresh(
            run_args, (corpus, state), now
        )

    assert [item["number"] for item in refreshed["issues"]] == [7]
    assert refreshed_state["excluded_issue_numbers"] == []
    assert refreshed["summary"]["source_total"] == 1
    assert refreshed["summary"]["excluded_self_assigned"] == 0


def test_failure_reuses_verified_snapshot_and_does_not_advance_cursor(
    tmp_path: Path, capsys
):
    run_args = args(tmp_path)
    old_corpus, old_state = commit_fixture(
        tmp_path,
        now=datetime(2026, 8, 13, 8, tzinfo=REFRESH.issue_api.TZ_CHINA),
    )
    old_bytes = Path(run_args.output).read_bytes()
    with (
        patch.object(REFRESH, "parse_args", return_value=run_args),
        patch.object(
            REFRESH, "perform_refresh", side_effect=requests.ConnectionError("offline")
        ),
    ):
        assert REFRESH.main([]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "stale_fallback"
    assert payload["snapshot_usable"] is True
    assert Path(run_args.output).read_bytes() == old_bytes
    new_state = json.loads(Path(run_args.state_file).read_text(encoding="utf-8"))
    assert new_state["cursor_at"] == old_state["cursor_at"]
    assert new_state["last_refresh_status"] == "stale_fallback"


def test_busy_refresh_lock_does_not_mutate_another_writer_state(tmp_path: Path, capsys):
    run_args = args(tmp_path)
    commit_fixture(tmp_path)
    state_path = Path(run_args.state_file)
    old_state = state_path.read_bytes()
    with (
        patch.object(REFRESH, "parse_args", return_value=run_args),
        patch.object(
            REFRESH.RefreshLock,
            "__enter__",
            side_effect=TimeoutError("another refresh owns the lock"),
        ),
    ):
        assert REFRESH.main([]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "stale_fallback"
    assert payload["snapshot_usable"] is True
    assert state_path.read_bytes() == old_state


def test_refresh_rejects_negative_lifecycle_intervals():
    with pytest.raises(SystemExit):
        REFRESH.parse_args(
            [
                "--url",
                "https://gitcode.com/cann/ops-math",
                "--overlap-seconds",
                "-1",
            ]
        )


def test_query_keeps_curated_cards_separate_from_runtime_candidates(tmp_path: Path):
    corpus, _ = commit_fixture(tmp_path)
    loaded, status = QUERY.load_runtime_corpus(
        tmp_path / "issue-history.json", tmp_path / "issue-knowledge-state.json"
    )
    candidates = QUERY.search_runtime(loaded, "MatMul precision", limit=5)

    assert status["status"] == "usable"
    assert candidates[0]["id"] == "runtime:cann/ops-math#7"
    assert candidates[0]["status"] == "provisional"
    assert candidates[0]["confidence"] == "low"
    assert "不能证明" in candidates[0]["warning"]
    assert all("local_path" not in candidate for candidate in candidates)


def test_query_rejects_tampered_runtime_corpus(tmp_path: Path):
    commit_fixture(tmp_path)
    path = tmp_path / "issue-history.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["issues"][0]["title"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded, status = QUERY.load_runtime_corpus(
        path, tmp_path / "issue-knowledge-state.json"
    )
    assert loaded is None
    assert status["status"] == "invalid"
    assert "digest" in status["reason"]


@pytest.mark.parametrize(
    "artifact_name",
    ["issue-history.json", "issue-knowledge-state.json"],
)
def test_query_rejects_malformed_runtime_json(tmp_path: Path, artifact_name: str):
    commit_fixture(tmp_path)
    (tmp_path / artifact_name).write_text("{", encoding="utf-8")

    loaded, status = QUERY.load_runtime_corpus(
        tmp_path / "issue-history.json", tmp_path / "issue-knowledge-state.json"
    )

    assert loaded is None
    assert status["status"] == "invalid"


def test_query_repository_root_points_outside_worktree(tmp_path: Path, monkeypatch):
    repository = tmp_path / "repository"
    runtime_dir = repository / ".cannbot/gitcode-issue-handler/data"
    runtime_dir.mkdir(parents=True)
    corpus, state = commit_fixture(runtime_dir)
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    monkeypatch.chdir(worktree)
    monkeypatch.setenv("ISSUE_HANDLER_REPOSITORY_ROOT", str(repository))
    query_args = argparse.Namespace(
        no_runtime_corpus=False,
        runtime_corpus=None,
        runtime_state=None,
        repository_root=None,
    )

    loaded, status = QUERY._runtime_inputs(query_args)
    assert loaded["issues"] == corpus["issues"]
    assert status["status"] == "usable"


def test_report_title_uses_repository_and_preserves_ops_math_heading():
    metadata = {
        "generated_at": "2026-08-13T12:00:00+08:00",
        "source_url": "https://gitcode.com/cann/ops-math",
        "repository": "cann/ops-math",
        "filter": "author != assignee, including missing assignee",
    }
    summary = REFRESH.corpus_builder.build_summary([], 0, 0)
    report = REFRESH.corpus_builder.render_report(metadata, summary, [])
    assert report.startswith("# ops-math 非自提 Issue 历史分析\n")

    metadata["repository"] = "cann/other-repo"
    metadata["filter"] = "all issues"
    report = REFRESH.corpus_builder.render_report(metadata, summary, [])
    assert report.startswith("# other-repo Issue 历史证据分析\n")
    assert "- 过滤口径：all issues\n" in report
