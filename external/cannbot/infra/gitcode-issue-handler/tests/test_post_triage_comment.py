# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exercise the Handler gate with real classifier categories and executor IO."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GATE = load("post_triage_comment")
CLASSIFIER = load("classify_issues")
CLASSIFIED_ITEM = getattr(CLASSIFIER, "_classified_item")
URL = "https://gitcode.com/cann/ops-math/issues/42"
CONFIG = {"repo": "cann/ops-math", "auto-response": False, "auto-assign": False}


@pytest.fixture(autouse=True)
def isolated_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def classified():
    source = {"number": 42, "url": URL, "author": "reporter", "state": "open", "comments": []}
    options = CLASSIFIER.ClassificationOptions({}, lambda *_: True, True)
    routed = CLASSIFIER.apply_processing_mode(CLASSIFIER.classify_one(source, options), False)
    routed["responsibility"] = "handle"
    return {
        "mode": "batch", "issues": [CLASSIFIED_ITEM(source, routed, {})],
        "association_scan": {
            "pr_fetch": {"complete": True},
            "linkage_fallback": {"complete": True, "incomplete_issue_numbers": []},
        },
        "comment_fetch": {"complete": True, "skipped": 1, "errors": [], "input_hits": 0},
    }


@pytest.mark.parametrize("level", ["pending", "list-only", "ignore", None])
def test_scope_blocks_before_executor(classified, monkeypatch, level):
    classified["issues"][0]["responsibility"] = level
    execute = Mock()
    monkeypatch.setattr(GATE.executor, "execute", execute)
    with pytest.raises(ValueError, match="responsibility"):
        GATE.execute(classified, URL, "reply", "unused", apply=True)
    execute.assert_not_called()


@pytest.mark.parametrize("category", ["self_assigned", "our_team_replied", "replied_no_owner",
    "planning_record", "assigned_requirement", "awaiting_assignee",
    "awaiting_assignee_setup", "association_scan_incomplete", "comment_scan_incomplete",
    "responsibility_review_required", "operator_routing_required"])
def test_single_override_does_not_create_reply_need(classified, category):
    classified["mode"] = "single"
    classified["issues"][0].update(category=category, single_issue_override=True, must_handle=True)
    with pytest.raises(ValueError, match="does not require"):
        GATE.validate_classification(classified, URL)


@pytest.mark.parametrize("target", [URL.replace("cann/", "other/"), URL.replace("42", "43"),
    URL.replace("gitcode.com", "other.test"), URL.replace("cann/", "group/cann/"), URL + "/comments"])
def test_exact_repository_issue_target(classified, target):
    with pytest.raises(ValueError):
        GATE.validate_classification(classified, target)


def test_duplicate_and_number_conflict(classified):
    classified["issues"].append(dict(classified["issues"][0]))
    with pytest.raises(ValueError, match="exactly one"):
        GATE.validate_classification(classified, URL)
    classified["issues"].pop()
    classified["issues"][0]["number"] = 99
    with pytest.raises(ValueError, match="number conflicts"):
        GATE.validate_classification(classified, URL)


def test_no_attention_and_missing_scan_are_blocked(classified):
    classified["issues"][0]["bucket"] = "no_attention"
    with pytest.raises(ValueError, match="does not require"):
        GATE.validate_classification(classified, URL)
    classified["issues"][0]["bucket"] = "need_attention"
    del classified["comment_fetch"]
    with pytest.raises(ValueError, match="comment scan"):
        GATE.validate_classification(classified, URL)


@pytest.mark.parametrize("scan", ["pr_fetch", "linkage_fallback", "comment_fetch"])
@pytest.mark.parametrize("update", [{"complete": False}, {"skipped": "responsibility_gate"}])
def test_scan_blocks_even_if_category_was_manually_changed(classified, scan, update):
    record = classified.get(scan) or classified["association_scan"][scan]
    record.update(update)
    with pytest.raises(ValueError, match="scan"):
        GATE.validate_classification(classified, URL)


@pytest.mark.parametrize("category", sorted(GATE.REPLY_CATEGORIES))
def test_first_response_and_followup_transparently_delegate(classified, monkeypatch, category):
    classified["issues"][0]["category"] = category
    classified["comment_fetch"].update(input_hits=1, cache_hits=1)
    expected = {"status": "reused", "comment_id": 100}
    monkeypatch.setattr(GATE, "_check_live_open", lambda url, session=None: session)
    execute = Mock(return_value=expected)
    monkeypatch.setattr(GATE.executor, "execute", execute)
    session = object()
    assert GATE.execute(
        classified, URL, "body", "result.json", apply=True, session=session,
        config=CONFIG, reply_approved=True,
    ) is expected
    execute.assert_called_once_with(URL, "body", "result.json", apply=True, session=session)


def test_real_preview_no_token_no_result_write(classified, tmp_path, monkeypatch):
    monkeypatch.delenv("GITCODE_TOKEN", raising=False)
    result_file = tmp_path / "result.json"
    result = GATE.execute(classified, URL, "有证据的回复", result_file)
    assert result["status"] == "preview"
    assert result["body"] == "有证据的回复"
    assert not result_file.exists()


def test_apply_requires_explicit_reply_approval_when_auto_response_disabled(
    classified, tmp_path, monkeypatch
):
    execute = Mock()
    monkeypatch.setattr(GATE.executor, "execute", execute)
    with pytest.raises(ValueError, match="reply-approved"):
        GATE.execute(classified, URL, "reply", tmp_path / "result.json", apply=True,
                     config=CONFIG)
    execute.assert_not_called()


def test_apply_with_explicit_reply_approval_is_allowed_when_auto_response_disabled(
    classified, tmp_path, monkeypatch
):
    expected = {"status": "verified"}
    monkeypatch.setattr(GATE, "_check_live_open", lambda url, session=None: session)
    execute = Mock(return_value=expected)
    monkeypatch.setattr(GATE.executor, "execute", execute)
    result = GATE.execute(classified, URL, "reply", tmp_path / "result.json", apply=True,
                          config=CONFIG, reply_approved=True)
    assert result is expected
    execute.assert_called_once()


def test_apply_auto_response_enabled_does_not_require_manual_approval(
    classified, tmp_path, monkeypatch
):
    expected = {"status": "verified"}
    monkeypatch.setattr(GATE, "_check_live_open", lambda url, session=None: session)
    execute = Mock(return_value=expected)
    monkeypatch.setattr(GATE.executor, "execute", execute)
    config = {**CONFIG, "auto-response": True}
    result = GATE.execute(classified, URL, "reply", tmp_path / "result.json", apply=True,
                          config=config)
    assert result is expected
    execute.assert_called_once()


def test_followup_always_requires_manual_approval_even_with_auto_response(
    classified, tmp_path, monkeypatch
):
    classified["issues"][0]["category"] = "reporter_followup"
    execute = Mock()
    monkeypatch.setattr(GATE.executor, "execute", execute)
    with pytest.raises(ValueError, match="reply-approved"):
        GATE.execute(classified, URL, "reply", tmp_path / "result.json", apply=True,
                     config={**CONFIG, "auto-response": True})
    execute.assert_not_called()


def test_apply_rejects_config_for_different_repository(
    classified, tmp_path, monkeypatch
):
    execute = Mock()
    monkeypatch.setattr(GATE.executor, "execute", execute)
    with pytest.raises(ValueError, match="仓库不一致"):
        GATE.execute(classified, URL, "reply", tmp_path / "result.json", apply=True,
                     config={**CONFIG, "repo": "other/project"}, reply_approved=True)
    execute.assert_not_called()


def test_cli_pending_apply_never_reaches_toolkit(classified, tmp_path, monkeypatch, capsys):
    classified["issues"][0]["responsibility"] = "pending"
    path = tmp_path / "classification.json"
    path.write_text(json.dumps(classified))
    main = Mock()
    monkeypatch.setattr(GATE.executor, "main", main)
    assert GATE.main(["--classification", str(path), "--issue-url", URL,
                      "--body-file", "unused", "--result-file", "unused", "--apply"]) == 2
    main.assert_not_called()
    assert "blocked" in capsys.readouterr().err


def test_cli_forwards_apply_and_returns_toolkit_status(classified, tmp_path, monkeypatch):
    Path("body.md").write_text("回复正文", encoding="utf-8")
    path = tmp_path / "classification.json"
    path.write_text(json.dumps(classified))
    monkeypatch.setattr(GATE, "_check_live_open", lambda url, session=None: session)
    main = Mock(return_value=1)
    monkeypatch.setattr(GATE.executor, "main", main)
    assert GATE.main(["--classification", str(path), "--issue-url", URL,
                      "--body-file", "body.md", "--result-file", "result.json", "--apply",
                      "--reply-approved"]) == 1
    main.assert_called_once_with(["--issue", URL, "--body-file", "body.md",
                                 "--result-file", "result.json", "--apply"])


@pytest.mark.parametrize("affected", [42, 99])
@pytest.mark.parametrize("scan", ["linkage_fallback", "comment_fetch"])
def test_partial_scan_blocks_only_affected_issue(classified, scan, affected):
    record = classified.get(scan) or classified["association_scan"][scan]
    record["complete"] = False
    if scan == "linkage_fallback":
        record["incomplete_issue_numbers"] = [affected]
    else:
        record["errors"] = [{"issue_number": affected, "error": "Timeout"}]
    if affected == 42:
        with pytest.raises(ValueError, match="scan is incomplete"):
            GATE.validate_classification(classified, URL)
    else:
        GATE.validate_classification(classified, URL)


def test_real_toolkit_post_readback_and_reuse(classified, tmp_path, monkeypatch):
    monkeypatch.setenv("GITCODE_TOKEN", "test-token")
    comments = []
    session = Mock()

    def get(url, **kwargs):
        payload = ({"login": "me"} if url.endswith("/user") else
                   {"state": "open"} if url.endswith("/issues/42") else comments)
        return Mock(status_code=200, headers={}, json=lambda: payload)

    def post(url, **kwargs):
        comments.append({"id": 7, "body": kwargs["data"]["body"], "user": {"login": "me"}})
        return Mock(status_code=201, json=lambda: {"id": 7})

    session.get.side_effect = get
    session.post.side_effect = post
    path = tmp_path / "result.json"
    result = GATE.execute(
        classified, URL, "reply", path, apply=True, session=session,
        config=CONFIG, reply_approved=True,
    )
    assert result["status"] == "verified"
    assert json.loads(path.read_text())["comment_id"] == 7
    again = GATE.execute(
        classified, URL, "reply", path, apply=True, session=session,
        config=CONFIG, reply_approved=True,
    )
    assert again["status"] == "verified"
    reused = GATE.execute(
        classified, URL, "reply", tmp_path / "reused.json", apply=True,
        session=session, config=CONFIG, reply_approved=True,
    )
    assert reused["status"] == "reused"
    session.post.assert_called_once()


@pytest.mark.parametrize("state", ["closed", "close", "unknown", None])
def test_live_non_open_issue_blocks_stale_classification(classified, tmp_path, monkeypatch, state):
    monkeypatch.setenv("GITCODE_TOKEN", "token")
    monkeypatch.setattr(GATE, "api_get", lambda *a: {"state": state})
    execute = Mock()
    monkeypatch.setattr(GATE.executor, "execute", execute)
    path = tmp_path / "result.json"
    # An earlier response snapshot must not authorize another comment.
    path.write_text(json.dumps({"status": "verified", "post_started": True, "comment_id": 7}))
    original = path.read_text()
    with pytest.raises(ValueError, match="must still be open"):
        GATE.execute(classified, URL, "reply", path, apply=True, session=object(),
                     config=CONFIG, reply_approved=True)
    execute.assert_not_called()
    assert path.read_text() == original


def test_cli_live_closed_issue_blocks_toolkit(classified, tmp_path, monkeypatch, capsys):
    Path("unused").write_text("回复正文", encoding="utf-8")
    monkeypatch.setenv("GITCODE_TOKEN", "token")
    monkeypatch.setattr(GATE, "api_get", lambda *a: {"state": "closed"})
    monkeypatch.setattr(GATE, "make_session", lambda: object())
    monkeypatch.setattr(GATE, "load_handler_config", lambda *a: CONFIG)
    main = Mock()
    monkeypatch.setattr(GATE.executor, "main", main)
    path = tmp_path / "classification.json"
    path.write_text(json.dumps(classified))
    assert GATE.main(["--classification", str(path), "--issue-url", URL,
                      "--body-file", "unused", "--result-file", "unused", "--apply",
                      "--reply-approved"]) == 2
    main.assert_not_called()
    assert "must still be open" in capsys.readouterr().err
