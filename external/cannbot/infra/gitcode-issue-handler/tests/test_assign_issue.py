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
SCRIPT = ROOT / "scripts" / "assign_issue.py"
SPEC = importlib.util.spec_from_file_location("assign_issue_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ASSIGN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ASSIGN)

URL = "https://gitcode.com/xujiachen8/ops-math/issues/42"


def write_inputs(tmp_path: Path, *, auto_response: bool = False,
                 auto_assign: bool = False, candidate: dict | None = None,
                 response_status: str = "verified") -> dict[str, Path]:
    config = tmp_path / "classify_config.yaml"
    config.write_text(
        f"auto-response: {'true' if auto_response else 'false'}\n"
        f"auto-assign: {'true' if auto_assign else 'false'}\n"
        f"cache_dir: {tmp_path / 'cache'}\n",
        encoding="utf-8",
    )
    response = tmp_path / "response.json"
    response.write_text(json.dumps({
        "status": response_status,
        "target": {"owner": "xujiachen8", "repo": "ops-math",
                    "issue_number": "42", "url": URL},
    }), encoding="utf-8")
    candidate_path = tmp_path / "candidates.json"
    candidate_path.write_text(json.dumps(candidate or {
        "issue_iid": 42,
        "candidates": [{
            "login": "test-owner",
            "operators": ["Add"],
            "summary_description": "核心 host tiling implementation",
            "evidence": ["commit abc", "PR !123"],
            "login_verified": True,
            "identity_evidence": ["GET PR !123 user.login=test-owner"],
            "actual_core_contribution": True,
        }],
    }), encoding="utf-8")
    owners = config.with_name("operator_owners.yaml")
    owners.write_text("operators: {}\n", encoding="utf-8")
    return {"config": config, "response": response, "candidate": candidate_path,
            "result": tmp_path / "result.json", "owners": owners}


def call(paths: dict[str, Path], *, apply: bool = False,
         assignment_approved: bool = False, owner: str = "test-owner",
         linked: Path | None = None, session=None):
    return ASSIGN.execute(
        config_path=paths["config"], issue_url=URL, owner=owner,
        candidate_file=None if linked else paths["candidate"],
        linked_pr_file=linked,
        response_result=paths["response"], result_file=paths["result"],
        apply=apply, assignment_approved=assignment_approved, session=session,
    )


def mock_successful_assignment(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ASSIGN,
        "api_get",
        lambda *_: calls.append("get")
        or (
            {"state": "open", "assignee": None}
            if len(calls) == 1
            else {"state": "open", "assignee": {"login": "test-owner"}}
        ),
    )
    monkeypatch.setattr(
        ASSIGN, "api_patch",
        lambda *_args, **_kwargs: calls.append("patch") or SimpleNamespace(status_code=200),
    )
    return calls


def write_linked_pr(tmp_path: Path, **record_overrides) -> Path:
    record = {
        "pr_url": "https://gitcode.com/xujiachen8/ops-math/merge_requests/7",
        "pr_author": "test-owner", "state": "open", "merged": False,
        "preexisting_association": True, "covers_issue": True,
    }
    record.update(record_overrides)
    linked = tmp_path / "linked.json"
    linked.write_text(json.dumps({
        "target_issue_url": URL, "issue_author": "reporter",
        "repository": "xujiachen8/ops-math", "linked_prs": [record],
    }), encoding="utf-8")
    return linked


def test_preview_persists_provisional_candidate_without_http(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path)
    result = call(paths)

    assert result["status"] == "preview"
    assert result["provisional"] is True
    assert result["owner_confirmation"] == "pending"
    assert result["assignment_source"] == "core_candidate"
    assert json.loads(paths["result"].read_text())["target"]["issue_number"] == "42"


def test_apply_requires_policy_or_explicit_approval(tmp_path: Path, monkeypatch) -> None:
    paths = write_inputs(tmp_path)
    monkeypatch.setenv("GITCODE_TOKEN", "token")
    with pytest.raises(ASSIGN.AssignmentError, match="assignment is disabled"):
        call(paths, apply=True)


def test_known_owner_requires_auto_response_when_both_flags_are_false(
    tmp_path: Path, monkeypatch
) -> None:
    paths = write_inputs(tmp_path)
    paths["owners"].write_text("operators:\n  Add: test-owner\n", encoding="utf-8")
    monkeypatch.setenv("GITCODE_TOKEN", "token")

    with pytest.raises(ASSIGN.AssignmentError, match="assignment is disabled"):
        call(paths, apply=True, session=object())


def test_known_owner_auto_response_allows_assignment_without_auto_assign(
    tmp_path: Path, monkeypatch
) -> None:
    paths = write_inputs(tmp_path, auto_response=True)
    paths["owners"].write_text("operators:\n  Add: test-owner\n", encoding="utf-8")
    monkeypatch.setenv("GITCODE_TOKEN", "token")
    calls = mock_successful_assignment(monkeypatch)

    result = call(paths, apply=True, session=object())
    assert result["assignment_source"] == "operator_owner"
    assert calls == ["get", "patch", "get"]


def test_candidate_manual_assignment_override_allows_both_flags_false(
    tmp_path: Path, monkeypatch
) -> None:
    paths = write_inputs(tmp_path)
    monkeypatch.setenv("GITCODE_TOKEN", "token")
    mock_successful_assignment(monkeypatch)

    result = call(paths, apply=True, assignment_approved=True, session=object())
    assert result["status"] == "verified"
    assert result["provisional"] is True



def test_apply_patches_only_after_get_and_verifies_assignee(tmp_path: Path, monkeypatch) -> None:
    paths = write_inputs(tmp_path, auto_assign=True, auto_response=True)
    monkeypatch.setenv("GITCODE_TOKEN", "token")
    calls = []
    session = object()

    def get(_session, endpoint, token):
        calls.append(("get", endpoint, token))
        return (
            {"state": "open", "assignee": None}
            if len(calls) == 1
            else {"state": "open", "assignee": {"login": "test-owner"}}
        )

    def patch(_session, endpoint, token, *, json_data):
        calls.append(("patch", endpoint, token, json_data))
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(ASSIGN, "api_get", get)
    monkeypatch.setattr(ASSIGN, "api_patch", patch)
    result = call(paths, apply=True, session=session)

    assert result["status"] == "verified"
    assert [item[0] for item in calls] == ["get", "patch", "get"]
    assert calls[1][3] == {"assignee": "test-owner"}


def test_existing_same_assignee_is_reused_without_patch(tmp_path: Path, monkeypatch) -> None:
    paths = write_inputs(tmp_path, auto_assign=True, auto_response=True)
    monkeypatch.setenv("GITCODE_TOKEN", "token")

    def fail_patch(*args, **kwargs):
        pytest.fail("existing assignee must not be patched")

    monkeypatch.setattr(ASSIGN, "api_get", lambda *_args: {"state": "open", "assignee": {"login": "test-owner"}})
    monkeypatch.setattr(ASSIGN, "api_patch", fail_patch)

    assert call(paths, apply=True, session=object())["status"] == "reused"


def test_existing_different_assignee_is_not_overwritten(tmp_path: Path, monkeypatch) -> None:
    paths = write_inputs(tmp_path, auto_assign=True, auto_response=True)
    monkeypatch.setenv("GITCODE_TOKEN", "token")
    monkeypatch.setattr(ASSIGN, "api_get", lambda *_args: {"state": "open", "assignee": {"login": "other"}})

    def fail_patch(*args, **kwargs):
        pytest.fail("different assignee must not be overwritten")

    monkeypatch.setattr(ASSIGN, "api_patch", fail_patch)

    result = call(paths, apply=True, session=object())
    assert result["status"] == "blocked_existing_assignee"


def test_response_target_mismatch_is_rejected(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path)
    paths["response"].write_text(json.dumps({
        "status": "verified",
        "target": {"owner": "xujiachen8", "repo": "ops-math", "issue_number": "43"},
    }), encoding="utf-8")
    with pytest.raises(ASSIGN.AssignmentError, match="response result target"):
        call(paths)


def test_linked_pr_uses_auto_response_without_candidate_core_evidence(tmp_path: Path, monkeypatch) -> None:
    paths = write_inputs(tmp_path, auto_response=True, candidate={
        "candidates": [{"login": "unused"}],
    })
    linked = tmp_path / "linked.json"
    linked.write_text(json.dumps({
        "target_issue_url": URL,
        "issue_author": "reporter",
        "repository": "xujiachen8/ops-math",
        "linked_prs": [{
            "pr_url": "https://gitcode.com/xujiachen8/ops-math/merge_requests/7",
            "pr_author": "test-owner", "state": "open", "merged": False,
            "preexisting_association": True, "covers_issue": True,
        }],
    }), encoding="utf-8")
    monkeypatch.setenv("GITCODE_TOKEN", "token")
    monkeypatch.setattr(ASSIGN, "api_get", lambda *_args: {"state": "open", "assignee": {"login": "test-owner"}})

    result = call(paths, apply=True, linked=linked, session=object())
    assert result["status"] == "reused"
    assert result["assignment_source"] == "linked_pr"
    assert result["provisional"] is True


@pytest.mark.parametrize("change", [
    {"state": "closed", "merged": False},
    {"state": "open", "merged": False, "preexisting_association": False, "covers_issue": True},
    {"state": "open", "merged": False, "preexisting_association": True, "covers_issue": False},
])
def test_linked_pr_requires_merged_or_open_preexisting_covering_evidence(
    tmp_path: Path, change: dict
) -> None:
    paths = write_inputs(tmp_path, auto_response=True)
    linked = tmp_path / "linked.json"
    record = {
        "pr_url": "https://gitcode.com/xujiachen8/ops-math/merge_requests/7",
        "pr_author": "test-owner",
        "preexisting_association": True,
        "covers_issue": True,
        **change,
    }
    linked.write_text(json.dumps({
        "target_issue_url": URL, "issue_author": "reporter",
        "repository": "xujiachen8/ops-math", "linked_prs": [record],
    }), encoding="utf-8")
    with pytest.raises(ASSIGN.AssignmentError, match="linked PR"):
        call(paths, linked=linked)


def test_linked_pr_rejects_cross_repository_and_missing_coverage_review(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path, auto_response=True)

    def linked_document(**overrides):
        record = {
            "pr_url": "https://gitcode.com/xujiachen8/ops-math/merge_requests/7",
            "pr_author": "test-owner", "state": "open", "merged": False,
            "preexisting_association": True, "covers_issue": True,
        }
        record.update(overrides.pop("record", {}))
        document = {
            "target_issue_url": URL, "issue_author": "reporter",
            "repository": "xujiachen8/ops-math", "linked_prs": [record],
        }
        document.update(overrides)
        return document

    for name, document, message in [
        ("cross", linked_document(repository="other/ops-math"), "linked PR"),
    ]:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ASSIGN.AssignmentError, match=message):
            call(paths, linked=path)

    conflict = linked_document(linked_prs=[
        {"pr_url": "https://gitcode.com/xujiachen8/ops-math/merge_requests/7",
         "pr_author": "test-owner", "state": "open", "merged": False,
         "preexisting_association": True, "covers_issue": True},
        {"pr_url": "https://gitcode.com/xujiachen8/ops-math/merge_requests/8",
         "pr_author": "other-owner", "state": "open", "merged": False,
         "preexisting_association": True, "covers_issue": True},
    ])
    path = tmp_path / "conflict.json"
    path.write_text(json.dumps(conflict), encoding="utf-8")
    with pytest.raises(ASSIGN.AssignmentError, match="coverage_review"):
        call(paths, linked=path)


def test_linked_pr_live_author_and_state_are_verified_before_patch(tmp_path: Path, monkeypatch) -> None:
    paths = write_inputs(tmp_path, auto_response=True)
    linked = write_linked_pr(tmp_path)
    monkeypatch.setenv("GITCODE_TOKEN", "token")
    calls = []

    def get(_session, endpoint, _token, **_kwargs):
        calls.append(endpoint)
        if len(calls) == 1:
            return {"state": "open", "user": {"login": "reporter"}, "assignee": None}
        if len(calls) == 2:
            return {"user": {"login": "test-owner"}, "state": "open", "merged": False}
        if len(calls) == 3:
            return [{"number": "42"}]
        return {"state": "open", "assignee": {"login": "test-owner"}}

    monkeypatch.setattr(ASSIGN, "api_get", get)
    monkeypatch.setattr(ASSIGN, "api_patch", lambda *_args, **_kwargs: SimpleNamespace(status_code=200))
    result = call(paths, apply=True, linked=linked, session=object())
    assert result["status"] == "verified"
    assert "/pulls/7" in calls[1]
    assert calls[2].endswith("/pulls/7/issues")


def test_linked_pr_current_run_association_is_accepted_and_rechecked_live(
    tmp_path: Path, monkeypatch
) -> None:
    paths = write_inputs(tmp_path, auto_response=True)
    linked = tmp_path / "linked.json"
    linked.write_text(json.dumps({
        "target_issue_url": URL, "issue_author": "reporter",
        "repository": "xujiachen8/ops-math", "linked_prs": [{
            "pr_url": "https://gitcode.com/xujiachen8/ops-math/merge_requests/7",
            "pr_author": "test-owner", "state": "open", "merged": False,
            "association_verified": True, "association_source": "current_run",
            "covers_issue": True,
        }],
    }), encoding="utf-8")
    monkeypatch.setenv("GITCODE_TOKEN", "token")
    responses = [
        {"state": "open", "user": {"login": "reporter"}, "assignee": None},
        {"user": {"login": "test-owner"}, "state": "open", "merged": False},
        [{"number": "42"}],
        {"state": "open", "assignee": {"login": "test-owner"}},
    ]
    monkeypatch.setattr(ASSIGN, "api_get", lambda *_args, **_kwargs: responses.pop(0))
    monkeypatch.setattr(
        ASSIGN, "api_patch",
        lambda *_args, **_kwargs: SimpleNamespace(status_code=200),
    )

    result = call(paths, apply=True, linked=linked, session=object())

    assert result["status"] == "verified"
    assert result["assignment_source"] == "linked_pr"
    assert result["assignment_provisional"] is True


def test_linked_pr_assignment_requires_live_target_association(
    tmp_path: Path, monkeypatch
) -> None:
    paths = write_inputs(tmp_path, auto_response=True)
    linked = tmp_path / "linked.json"
    linked.write_text(json.dumps({
        "target_issue_url": URL, "issue_author": "reporter",
        "repository": "xujiachen8/ops-math", "linked_prs": [{
            "pr_url": "https://gitcode.com/xujiachen8/ops-math/merge_requests/7",
            "pr_author": "test-owner", "state": "open", "merged": False,
            "association_verified": True, "covers_issue": True,
        }],
    }), encoding="utf-8")
    monkeypatch.setenv("GITCODE_TOKEN", "token")
    responses = [
        {"state": "open", "user": {"login": "reporter"}, "assignee": None},
        {"user": {"login": "test-owner"}, "state": "open", "merged": False},
        [{"number": "41"}],
    ]
    monkeypatch.setattr(ASSIGN, "api_get", lambda *_args, **_kwargs: responses.pop(0))
    monkeypatch.setattr(
        ASSIGN, "api_patch",
        lambda *_args, **_kwargs: pytest.fail("unlinked PR must not assign"),
    )

    with pytest.raises(ASSIGN.AssignmentError, match="not associated"):
        call(paths, apply=True, linked=linked, session=object())


def test_duplicate_cross_reference_accepts_live_native_duplicate_link(
    tmp_path: Path, monkeypatch
) -> None:
    paths = write_inputs(tmp_path, auto_response=True)
    linked = tmp_path / "linked.json"
    linked.write_text(json.dumps({
        "target_issue_url": URL, "issue_author": "reporter",
        "repository": "xujiachen8/ops-math", "linked_prs": [{
            "pr_url": "https://gitcode.com/xujiachen8/ops-math/merge_requests/7",
            "pr_author": "test-owner", "state": "open", "merged": False,
            "association_verified": False,
            "association_mode": "duplicate_cross_reference",
            "public_cross_reference_verified": True,
            "native_linked_issue": "41",
            "covers_issue": True,
        }],
    }), encoding="utf-8")
    monkeypatch.setenv("GITCODE_TOKEN", "token")
    responses = [
        {"state": "open", "user": {"login": "reporter"}, "assignee": None},
        {"user": {"login": "test-owner"}, "state": "open", "merged": False},
        [{"number": "41"}],
        {"state": "open", "assignee": {"login": "test-owner"}},
    ]
    monkeypatch.setattr(ASSIGN, "api_get", lambda *_args, **_kwargs: responses.pop(0))
    monkeypatch.setattr(
        ASSIGN, "api_patch",
        lambda *_args, **_kwargs: SimpleNamespace(status_code=200),
    )

    result = call(paths, apply=True, linked=linked, session=object())

    assert result["status"] == "verified"
    assert result["linked_pr_live_readback"]["association_mode"] == "duplicate_cross_reference"


@pytest.mark.parametrize("issue_author, live_pr", [
    ("test-owner", {"user": {"login": "test-owner"}, "state": "open"}),
    ("reporter", {"user": {"login": "other-owner"}, "state": "open"}),
    ("reporter", {"user": {"login": "test-owner"}, "state": "closed", "merged": False}),
])
def test_linked_pr_live_verification_blocks_self_author_mismatch_and_closed(
    tmp_path: Path, monkeypatch, issue_author: str, live_pr: dict
) -> None:
    paths = write_inputs(tmp_path, auto_response=True)
    linked = write_linked_pr(tmp_path)
    monkeypatch.setenv("GITCODE_TOKEN", "token")
    # Use an explicit sequence so the PR GET cannot be confused with the
    # Issue preflight GET.
    responses = [{"state": "open", "user": {"login": issue_author}, "assignee": None}, live_pr]
    monkeypatch.setattr(ASSIGN, "api_get", lambda *_args: responses.pop(0))

    def fail_patch(*_args, **_kwargs):
        pytest.fail("invalid PR evidence must block PATCH")

    monkeypatch.setattr(ASSIGN, "api_patch", fail_patch)
    with pytest.raises(ASSIGN.AssignmentError):
        call(paths, apply=True, linked=linked, session=object())


def test_unknown_patch_is_read_back_and_never_retried(tmp_path: Path, monkeypatch) -> None:
    paths = write_inputs(tmp_path, auto_assign=True, auto_response=True)
    monkeypatch.setenv("GITCODE_TOKEN", "token")
    patch_calls = []
    monkeypatch.setattr(ASSIGN, "api_get", lambda *_args: {"state": "open", "assignee": None})

    def unknown(*_args, **_kwargs):
        patch_calls.append(True)
        raise OSError("connection lost")

    monkeypatch.setattr(ASSIGN, "api_patch", unknown)
    first = call(paths, apply=True, session=object())
    second = call(paths, apply=True, session=object())
    assert first["status"] == "unknown"
    assert second["status"] == "unknown"
    assert len(patch_calls) == 1


def test_preview_preserves_unknown_patch_recovery_marker(tmp_path: Path, monkeypatch) -> None:
    paths = write_inputs(tmp_path, auto_assign=True, auto_response=True)
    monkeypatch.setenv("GITCODE_TOKEN", "token")
    monkeypatch.setattr(ASSIGN, "api_get", lambda *_args: {"state": "open", "assignee": None})
    patch_calls = []

    def unknown(*_args, **_kwargs):
        patch_calls.append(True)
        raise OSError("connection lost")

    monkeypatch.setattr(ASSIGN, "api_patch", unknown)
    assert call(paths, apply=True, session=object())["status"] == "unknown"
    preserved = call(paths, apply=False, session=object())
    assert preserved["status"] == "unknown"
    assert preserved["patch_started"] is True
    assert call(paths, apply=True, session=object())["status"] == "unknown"
    assert len(patch_calls) == 1


def test_interrupted_preview_patch_started_only_recovers_by_get(tmp_path: Path, monkeypatch) -> None:
    paths = write_inputs(tmp_path, auto_assign=True, auto_response=True)
    paths["result"].write_text(json.dumps({
        "target": {"owner": "xujiachen8", "repo": "ops-math",
                    "issue_number": "42", "url": URL},
        "owner": "test-owner", "assignment_source": "core_candidate",
        "status": "preview", "patch_started": True, "patch_attempted": True,
    }), encoding="utf-8")
    monkeypatch.setenv("GITCODE_TOKEN", "token")
    monkeypatch.setattr(ASSIGN, "api_get", lambda *_args: {"state": "open", "assignee": None})
    monkeypatch.setattr(ASSIGN, "api_patch", lambda *_args, **_kwargs: pytest.fail("must not retry"))

    result = call(paths, apply=True, session=object())
    assert result["status"] == "unknown"


def test_unknown_then_readback_get_failure_preserves_no_retry_marker(
    tmp_path: Path, monkeypatch
) -> None:
    paths = write_inputs(tmp_path, auto_assign=True, auto_response=True)
    monkeypatch.setenv("GITCODE_TOKEN", "token")
    get_calls = []
    patch_calls = []

    def get(_session, _endpoint, _token):
        get_calls.append(True)
        if len(get_calls) == 1:
            return {"state": "open", "assignee": None}
        raise OSError("readback unavailable")

    def patch(_session, _endpoint, _token, *, json_data):
        patch_calls.append(json_data)
        raise OSError("patch outcome unknown")

    monkeypatch.setattr(ASSIGN, "api_get", get)
    monkeypatch.setattr(ASSIGN, "api_patch", patch)
    first = call(paths, apply=True, session=object())
    second = call(paths, apply=True, session=object())

    assert first["status"] == "unknown"
    assert first["patch_started"] is True
    assert second["status"] == "unknown"
    assert second["patch_started"] is True
    assert len(get_calls) == 3  # preflight GET, failed readback, recovery GET
    assert len(patch_calls) == 1


def fallback_inputs(tmp_path, *, fallback="test-owner", enabled=True):
    paths = write_inputs(tmp_path, auto_response=True, auto_assign=enabled, candidate={
        "target_issue_url": URL, "status": "insufficient_evidence",
        "operators": ["Add"], "candidates": [],
    })
    with paths["config"].open("a") as stream:
        stream.write(f'auto-assign-fallback-user: "{fallback}"\n')
    return paths


@pytest.mark.parametrize("fallback, enabled", [("", True), ("test-owner", False)])
def test_empty_or_disabled_fallback_skips_without_http(tmp_path, monkeypatch, fallback, enabled):
    paths = fallback_inputs(tmp_path, fallback=fallback, enabled=enabled)
    monkeypatch.setattr(ASSIGN, "resolve_token", lambda: pytest.fail("skip must be local"))
    result = call(paths, apply=True, assignment_approved=True, owner=None)
    assert result["status"] == "skipped"
    assert result["patch_attempted"] is False


def test_fallback_validates_user_then_assigns_and_reuses(tmp_path, monkeypatch):
    paths = fallback_inputs(tmp_path)
    monkeypatch.setenv("GITCODE_TOKEN", "token")
    calls = []
    current = None

    def get(_session, endpoint, _token):
        calls.append(endpoint)
        if "/users/" in endpoint:
            return {"login": "test-owner", "id": 123}
        return {"state": "open", "assignee": {"login": current}} if current else {"state": "open", "assignee": None}

    def patch(*_args, json_data):
        nonlocal current
        calls.append("PATCH")
        current = json_data["assignee"]
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(ASSIGN, "api_get", get)
    monkeypatch.setattr(ASSIGN, "api_patch", patch)
    result = call(paths, apply=True, owner=None, session=object())
    assert result["status"] == "verified"
    assert result["assignment_source"] == "fallback_user"
    assert result["assignment_provisional"] is True
    assert result["owner_confirmation"] == "pending"
    assert calls[1].endswith("/users/test-owner")
    assert calls[2] == "PATCH"
    assert call(paths, apply=True, owner=None, session=object())["status"] == "reused"
    assert calls.count("PATCH") == 1


@pytest.mark.parametrize("user", [None, {}, {"login": "wrong-user"}, OSError("unavailable")])
def test_invalid_or_unavailable_fallback_user_never_patches(tmp_path, monkeypatch, user):
    paths = fallback_inputs(tmp_path)
    monkeypatch.setenv("GITCODE_TOKEN", "token")

    def get(_session, endpoint, _token):
        if "/users/" not in endpoint:
            return {"state": "open", "assignee": None}
        if isinstance(user, Exception):
            raise user
        return user

    monkeypatch.setattr(ASSIGN, "api_get", get)
    monkeypatch.setattr(ASSIGN, "api_patch", lambda *a, **kw: pytest.fail("invalid user must not assign"))
    with pytest.raises(ASSIGN.AssignmentError, match="fallback user"):
        call(paths, apply=True, owner=None, session=object())


@pytest.mark.parametrize("change", [
    {"status": "pending"}, {"candidates": [None]},
    {"candidates": [{"login": "test-owner"}]},
    {"operator_candidates": [{"login": "other"}]},
    {"target_issue_url": URL.replace("42", "43")},
])
def test_fallback_cannot_replace_unfinished_invalid_or_existing_candidates(tmp_path, change):
    paths = fallback_inputs(tmp_path)
    document = json.loads(paths["candidate"].read_text())
    document.update(change)
    paths["candidate"].write_text(json.dumps(document))
    with pytest.raises(ASSIGN.AssignmentError):
        call(paths, owner=None)


def test_fallback_does_not_replace_configured_owner(tmp_path):
    paths = fallback_inputs(tmp_path)
    paths["owners"].write_text("operators:\n  Add: actual-owner\n")
    with pytest.raises(ASSIGN.AssignmentError, match="takes precedence"):
        call(paths, owner=None)


def test_fallback_does_not_replace_existing_assignee(tmp_path, monkeypatch):
    paths = fallback_inputs(tmp_path)
    monkeypatch.setenv("GITCODE_TOKEN", "token")
    monkeypatch.setattr(ASSIGN, "api_get", lambda *a: {"state": "open", "assignee": {"login": "existing-owner"}})
    monkeypatch.setattr(ASSIGN, "api_patch", lambda *a, **kw: pytest.fail("no overwrite"))
    assert call(paths, owner=None, apply=True, session=object())["status"] == "blocked_existing_assignee"


def test_configured_fallback_does_not_override_real_candidate(tmp_path):
    paths = write_inputs(tmp_path, auto_response=True, auto_assign=True)
    with paths["config"].open("a") as stream:
        stream.write('auto-assign-fallback-user: "someone-else"\n')
    result = call(paths)
    assert result["owner"] == "test-owner"
    assert result["assignment_source"] == "core_candidate"


@pytest.mark.parametrize("state", ["closed", "close", "unknown", None])
def test_current_non_open_issue_blocks_assignment_despite_verified_reply(tmp_path, monkeypatch, state):
    paths = write_inputs(tmp_path, auto_response=True, auto_assign=True)
    monkeypatch.setenv("GITCODE_TOKEN", "token")
    monkeypatch.setattr(ASSIGN, "api_get", lambda *_: {"state": state, "assignee": None})
    monkeypatch.setattr(ASSIGN, "api_patch", lambda *a, **kw: pytest.fail("non-open Issue must not be assigned"))
    with pytest.raises(ASSIGN.AssignmentError, match="must still be open"):
        call(paths, apply=True, session=object())
    assert not paths["result"].exists()


def test_closed_issue_preserves_assignment_recovery_without_retry(tmp_path, monkeypatch):
    paths = write_inputs(tmp_path, auto_response=True, auto_assign=True)
    preview = call(paths)
    preview.update(status="unknown", patch_started=True, patch_attempted=True)
    paths["result"].write_text(json.dumps(preview))
    monkeypatch.setenv("GITCODE_TOKEN", "token")
    monkeypatch.setattr(ASSIGN, "api_get", lambda *_: {"state": "closed", "assignee": {"login": "test-owner"}})
    monkeypatch.setattr(ASSIGN, "api_patch", lambda *a, **kw: pytest.fail("must not retry"))
    result = call(paths, apply=True, session=object())
    assert result["status"] == "reused"
    assert result["patch_started"] is True
