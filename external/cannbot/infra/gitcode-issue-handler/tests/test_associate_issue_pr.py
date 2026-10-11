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
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "associate_issue_pr.py"
SPEC = importlib.util.spec_from_file_location("associate_issue_pr_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ASSOCIATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ASSOCIATE)

ISSUE_URL = "https://gitcode.com/test-org/test-repo/issues/42"
PR_URL = "https://gitcode.com/test-org/test-repo/merge_requests/7"


def write_inputs(tmp_path: Path, *, auto_response: bool = True) -> dict[str, Path]:
    config = tmp_path / "classify_config.yaml"
    config.write_text(
        f"repo: test-org/test-repo\n"
        f"auto-response: {'true' if auto_response else 'false'}\n"
        "auto-assign: false\n"
        f"cache_dir: {tmp_path / 'cache'}\n",
        encoding="utf-8",
    )
    response = tmp_path / "response.json"
    response.write_text(json.dumps({
        "status": "verified",
        "target": {
            "owner": "test-org", "repo": "test-repo",
            "issue_number": "42", "url": ISSUE_URL,
        },
    }), encoding="utf-8")
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({
        "target_issue_url": ISSUE_URL,
        "pr_url": PR_URL,
        "pr_author": "pr-owner",
        "head_sha": "abc123",
        "base_ref": "master",
        "coverage_verified": True,
        "base_compatibility_verified": True,
        "additional_risk_reviewed": True,
        "safe_to_associate": True,
        "unresolved_risks": [],
        "changed_files": ["docs/guide.md"],
        "verification": ["reviewed exact diff"],
    }), encoding="utf-8")
    return {
        "config": config,
        "response": response,
        "evidence": evidence,
        "result": tmp_path / "association-result.json",
        "linked": tmp_path / "linked-pr.json",
    }


def call(paths: dict[str, Path], *, apply: bool = False,
         association_approved: bool = False):
    return ASSOCIATE.execute(
        config_path=paths["config"],
        issue_url=ISSUE_URL,
        pr_url=PR_URL,
        evidence_file=paths["evidence"],
        response_result=paths["response"],
        result_file=paths["result"],
        linked_pr_file=paths["linked"],
        apply=apply,
        association_approved=association_approved,
        session=object(),
    )


def issue_payload():
    return {"state": "open", "user": {"login": "reporter"}}


def pr_payload(**overrides):
    value = {
        "user": {"login": "pr-owner"},
        "state": "open",
        "merged": False,
        "base": {"ref": "master"},
        "head": {"sha": "abc123"},
    }
    value.update(overrides)
    return value


def test_preview_persists_without_http(tmp_path: Path, monkeypatch) -> None:
    paths = write_inputs(tmp_path)
    monkeypatch.setattr(ASSOCIATE, "resolve_token", lambda: pytest.fail("preview must not resolve token"))

    result = call(paths)

    assert result["status"] == "preview"
    assert result["association_status"] == "not_started"
    assert json.loads(paths["result"].read_text())["pr"]["pr_number"] == "7"


@pytest.mark.parametrize("field,value", [
    ("coverage_verified", False),
    ("base_compatibility_verified", False),
    ("additional_risk_reviewed", False),
    ("safe_to_associate", False),
    ("unresolved_risks", ["unknown compatibility"]),
])
def test_safety_evidence_is_mandatory(tmp_path: Path, field: str, value) -> None:
    paths = write_inputs(tmp_path)
    document = json.loads(paths["evidence"].read_text())
    document[field] = value
    paths["evidence"].write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ASSOCIATE.AssociationError):
        call(paths)


def test_apply_posts_once_and_verifies_both_directions(tmp_path: Path, monkeypatch) -> None:
    paths = write_inputs(tmp_path)
    monkeypatch.setenv("GITCODE_TOKEN", "token")
    calls = []
    linked = {"value": False}

    def get(_session, endpoint, _token, **_kwargs):
        calls.append(("get", endpoint))
        if endpoint.endswith("/issues/42"):
            return issue_payload()
        if endpoint.endswith("/pulls/7"):
            return pr_payload()
        if endpoint.endswith("/pulls/7/issues"):
            return [{"number": "42"}] if linked["value"] else []
        if endpoint.endswith("/issues/42/pull_requests"):
            return [{"number": 7}] if linked["value"] else []
        raise AssertionError(endpoint)

    def post(_session, endpoint, _token, *, json_data):
        calls.append(("post", endpoint, json_data))
        linked["value"] = True
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(ASSOCIATE, "api_get", get)
    monkeypatch.setattr(ASSOCIATE, "api_post_json", post)

    result = call(paths, apply=True)

    assert result["status"] == "verified"
    assert result["association_status"] == "verified"
    assert [item[0] for item in calls].count("post") == 1
    linked_record = json.loads(paths["linked"].read_text())["linked_prs"][0]
    assert linked_record["pr_url"] == PR_URL
    assert linked_record["pr_author"] == "pr-owner"
    assert linked_record["association_verified"] is True
    assert linked_record["association_source"] == "current_run"
    assert linked_record["public_cross_reference_verified"] is False
    assert linked_record["covers_issue"] is True


def test_existing_association_is_reused_without_post(tmp_path: Path, monkeypatch) -> None:
    paths = write_inputs(tmp_path)
    monkeypatch.setenv("GITCODE_TOKEN", "token")

    def get(_session, endpoint, _token, **_kwargs):
        if endpoint.endswith("/issues/42"):
            return issue_payload()
        if endpoint.endswith("/pulls/7"):
            return pr_payload()
        if endpoint.endswith("/pulls/7/issues"):
            return [{"number": "42"}]
        if endpoint.endswith("/issues/42/pull_requests"):
            return [{"number": 7}]
        raise AssertionError(endpoint)

    monkeypatch.setattr(ASSOCIATE, "api_get", get)
    monkeypatch.setattr(
        ASSOCIATE, "api_post_json",
        lambda *_args, **_kwargs: pytest.fail("existing association must not POST"),
    )

    result = call(paths, apply=True)

    assert result["status"] == "reused"
    assert result["association_source"] == "preexisting"


def test_duplicate_issue_native_link_uses_verified_public_cross_reference(
    tmp_path: Path, monkeypatch
) -> None:
    paths = write_inputs(tmp_path)
    evidence = json.loads(paths["evidence"].read_text())
    evidence.update({
        "duplicate_issue_url": "https://gitcode.com/test-org/test-repo/issues/41",
        "duplicate_relationship_verified": True,
    })
    paths["evidence"].write_text(json.dumps(evidence), encoding="utf-8")
    response = json.loads(paths["response"].read_text())
    response["comment_id"] = 99
    paths["response"].write_text(json.dumps(response), encoding="utf-8")
    monkeypatch.setenv("GITCODE_TOKEN", "token")

    def get(_session, endpoint, _token, **_kwargs):
        if endpoint.endswith("/issues/42"):
            return issue_payload()
        if endpoint.endswith("/pulls/7"):
            return pr_payload()
        if endpoint.endswith("/pulls/7/issues"):
            return [{"number": "41"}]
        if endpoint.endswith("/issues/42/comments"):
            return [{
                "id": 99,
                "body": f"同一问题见 {evidence['duplicate_issue_url']}，修复见 {PR_URL}",
            }]
        raise AssertionError(endpoint)

    monkeypatch.setattr(ASSOCIATE, "api_get", get)
    monkeypatch.setattr(
        ASSOCIATE, "api_post_json",
        lambda *_args, **_kwargs: pytest.fail("single-Issue constraint must not POST"),
    )

    result = call(paths, apply=True)
    linked = json.loads(paths["linked"].read_text())["linked_prs"][0]

    assert result["status"] == "verified"
    assert result["association_status"] == "verified_cross_reference"
    assert linked["public_cross_reference_verified"] is True
    assert linked["native_linked_issue"] == "41"


def test_other_native_issue_without_verified_duplicate_blocks(
    tmp_path: Path, monkeypatch
) -> None:
    paths = write_inputs(tmp_path)
    monkeypatch.setenv("GITCODE_TOKEN", "token")

    responses = [issue_payload(), pr_payload(), [{"number": "41"}]]
    monkeypatch.setattr(ASSOCIATE, "api_get", lambda *_args, **_kwargs: responses.pop(0))
    monkeypatch.setattr(
        ASSOCIATE, "api_post_json",
        lambda *_args, **_kwargs: pytest.fail("conflicting association must not POST"),
    )

    result = call(paths, apply=True)

    assert result["status"] == "blocked_existing_association"
    assert not paths["linked"].exists()


def test_changed_live_pr_is_blocked_before_post(tmp_path: Path, monkeypatch) -> None:
    paths = write_inputs(tmp_path)
    monkeypatch.setenv("GITCODE_TOKEN", "token")

    responses = [issue_payload(), pr_payload(head={"sha": "changed"})]
    monkeypatch.setattr(ASSOCIATE, "api_get", lambda *_args, **_kwargs: responses.pop(0))
    monkeypatch.setattr(
        ASSOCIATE, "api_post_json",
        lambda *_args, **_kwargs: pytest.fail("changed PR must not POST"),
    )

    with pytest.raises(ASSOCIATE.AssociationError, match="head changed"):
        call(paths, apply=True)


def test_unknown_post_is_read_back_and_never_retried(tmp_path: Path, monkeypatch) -> None:
    paths = write_inputs(tmp_path)
    monkeypatch.setenv("GITCODE_TOKEN", "token")
    post_calls = []

    def get(_session, endpoint, _token, **_kwargs):
        if endpoint.endswith("/issues/42"):
            return issue_payload()
        if endpoint.endswith("/pulls/7"):
            return pr_payload()
        return []

    def post(*_args, **_kwargs):
        post_calls.append(True)
        raise OSError("connection lost")

    monkeypatch.setattr(ASSOCIATE, "api_get", get)
    monkeypatch.setattr(ASSOCIATE, "api_post_json", post)

    first = call(paths, apply=True)
    second = call(paths, apply=True)

    assert first["status"] == "unknown"
    assert second["status"] == "unknown"
    assert len(post_calls) == 1


def test_apply_requires_auto_response_or_explicit_approval(tmp_path: Path, monkeypatch) -> None:
    paths = write_inputs(tmp_path, auto_response=False)
    monkeypatch.setenv("GITCODE_TOKEN", "token")

    with pytest.raises(ASSOCIATE.AssociationError, match="association is disabled"):
        call(paths, apply=True)


@pytest.mark.parametrize("state", ["closed", "close", "unknown", None])
def test_current_non_open_issue_blocks_association_despite_verified_reply(tmp_path, monkeypatch, state):
    paths = write_inputs(tmp_path)
    monkeypatch.setenv("GITCODE_TOKEN", "token")

    def get(_session, endpoint, _token, **kwargs):
        if endpoint.endswith("/issues/42"):
            return {**issue_payload(), "state": state}
        if endpoint.endswith("/pulls/7"):
            return pr_payload()
        return []

    monkeypatch.setattr(ASSOCIATE, "api_get", get)
    monkeypatch.setattr(ASSOCIATE, "api_post_json", lambda *a, **kw: pytest.fail("non-open Issue must not be linked"))
    with pytest.raises(ASSOCIATE.AssociationError, match="must still be open"):
        call(paths, apply=True)
    assert not paths["result"].exists()
    assert not paths["linked"].exists()
