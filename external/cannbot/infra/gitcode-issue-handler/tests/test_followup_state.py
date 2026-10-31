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
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

HANDLER_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = HANDLER_ROOT / "scripts" / "followup_state.py"
SPEC = importlib.util.spec_from_file_location("followup_state_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
FOLLOWUP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FOLLOWUP)

ISSUE_URL = "https://gitcode.com/cann/ops-math/issues/2535"


def _issue_payload(state: str = "closed") -> dict:
    return {"state": state, "repository": {"id": 321}}


def _catalog() -> dict:
    return {
        "content": [
            {"id": 285, "name": "进行中", "enabled": True},
            {"id": 917, "name": "挂起", "enabled": True},
        ]
    }


def test_watch_cursor_and_resolve_share_one_atomic_state(tmp_path: Path) -> None:
    state_file = tmp_path / "followup-watch.json"
    watched = FOLLOWUP.watch_issue(
        state_file,
        "cann/ops-math",
        "2535",
        reporter="issue-author",
        issue_url=ISSUE_URL,
        maintainer_comment_id=1001,
        maintainer_comment_at="2026-08-13T08:00:00Z",
    )
    FOLLOWUP.advance_updated_cursor(state_file, "cann/ops-math", "2026-08-13T09:00:00Z")

    state = FOLLOWUP.load_followup_state(state_file, "cann/ops-math")

    assert watched["conversation_state"] == "awaiting_reporter"
    assert watched["last_maintainer_comment_id"] == 1001
    assert state["updated_cursor"] == "2026-08-13T09:00:00Z"
    assert state["issues"]["2535"]["reporter"] == "issue-author"
    assert FOLLOWUP.resolve_issue(state_file, "cann/ops-math", "2535") is True
    assert FOLLOWUP.load_followup_state(state_file, "cann/ops-math")["issues"] == {}


def test_watch_can_track_an_issue_waiting_for_its_assignee(tmp_path: Path) -> None:
    state_file = tmp_path / "followup-watch.json"

    watched = FOLLOWUP.watch_issue(
        state_file,
        "cann/ops-math",
        "2617",
        reporter="issue-author",
        assignee="@operator-owner",
        waiting_on="assignee",
        issue_url="https://gitcode.com/cann/ops-math/issues/2617",
        maintainer_comment_id=184760205,
        maintainer_comment_at="2026-08-14T02:11:02Z",
    )

    assert watched["conversation_state"] == "awaiting_assignee"
    assert watched["waiting_on"] == "assignee"
    assert watched["assignee"] == "operator-owner"
    assert watched["expires_at"] is None


def test_assignee_watch_requires_an_assignee(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="assignee must not be empty"):
        FOLLOWUP.watch_issue(
            tmp_path / "followup-watch.json",
            "cann/ops-math",
            "2617",
            reporter="issue-author",
            waiting_on="assignee",
            maintainer_comment_at="2026-08-14T02:11:02Z",
        )


def test_state_file_cannot_be_reused_for_another_repository(tmp_path: Path) -> None:
    state_file = tmp_path / "followup-watch.json"
    FOLLOWUP.advance_updated_cursor(state_file, "cann/ops-math", "2026-08-13T09:00:00Z")

    with pytest.raises(ValueError, match="different repository"):
        FOLLOWUP.load_followup_state(state_file, "cann/ops-nn")


def test_transition_dry_run_resolves_status_name_dynamically() -> None:
    with (
        patch.object(FOLLOWUP, "make_session", return_value=object()),
        patch.object(
            FOLLOWUP,
            "api_get",
            side_effect=[_issue_payload(), {"status": "已完成"}, _catalog()],
        ),
        patch.object(FOLLOWUP, "api_put") as api_put,
        patch.object(FOLLOWUP, "api_patch") as api_patch,
    ):
        result = FOLLOWUP.transition_issue_status(
            ISSUE_URL,
            "挂起",
            token="token",
        )

    assert result["status"] == "would_transition"
    assert result["target_status"] == "挂起"
    assert result["target_status_id"] == 917
    assert result["reopen"] is False
    api_put.assert_not_called()
    api_patch.assert_not_called()


def test_transition_apply_verifies_custom_state_and_core_reopen() -> None:
    response = SimpleNamespace(status_code=200)
    with (
        patch.object(FOLLOWUP, "make_session", return_value=object()),
        patch.object(
            FOLLOWUP,
            "api_get",
            side_effect=[
                _issue_payload(),
                {"status": "已完成"},
                _catalog(),
                {"status": "进行中"},
                _issue_payload(state="open"),
            ],
        ),
        patch.object(FOLLOWUP, "api_put", return_value=response) as api_put,
        patch.object(FOLLOWUP, "api_patch", return_value=response) as api_patch,
    ):
        result = FOLLOWUP.transition_issue_status(
            ISSUE_URL,
            "进行中",
            token="token",
            reopen=True,
            apply=True,
            authorization_evidence="preview-confirmed:run-42",
        )

    assert result["status"] == "transitioned"
    assert result["reopen"] is True
    assert api_put.call_args.kwargs["json_data"] == {
        "status_before": "已完成",
        "status_current": "进行中",
    }
    assert api_patch.call_args.kwargs["json_data"] == {"state": "open"}


def test_transition_apply_requires_authorization_evidence() -> None:
    with (
        patch.object(FOLLOWUP, "make_session", return_value=object()),
        patch.object(
            FOLLOWUP,
            "api_get",
            side_effect=[_issue_payload("open"), {"status": "进行中"}, _catalog()],
        ),
        pytest.raises(ValueError, match="authorization-evidence"),
    ):
        FOLLOWUP.transition_issue_status(
            ISSUE_URL,
            "挂起",
            token="token",
            apply=True,
        )
