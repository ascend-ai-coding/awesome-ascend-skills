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

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("post_issue_comment", ROOT / "scripts" / "post_issue_comment.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
NOW = getattr(mod, "_now")
DIGEST = getattr(mod, "_digest")
READBACK = getattr(mod, "_readback")


@pytest.fixture(autouse=True)
def gitcode_token(monkeypatch):
    """Supply the credential required by apply-mode tests without using a real token."""
    monkeypatch.setenv("GITCODE_TOKEN", "test-token")


class Response:
    def __init__(self, status=200, payload=None):
        self.status_code, self.payload = status, payload

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class Session:
    def __init__(self, pages, post=None, post_comment="auto"):
        self.pages, self.post_response, self.post_comment = pages, post, post_comment
        self.get_calls, self.post_calls = [], []

    def get(self, url, *, params=None, timeout=30):
        self.get_calls.append((url, params))
        if url.endswith("/user"):
            return Response(payload={"login": "me"})
        page = (params or {}).get("page", 1)
        return Response(payload=self.pages[page - 1] if page <= len(self.pages) else [])

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        if isinstance(self.post_response, Exception):
            if self.post_comment == "auto":
                self.post_comment = {"id": 22, "body": kwargs["data"]["body"], "user": {"login": "me"}}
            if self.post_comment:
                item = dict(self.post_comment)
                item.setdefault("created_at", NOW())
                self.pages[0].append(item)
            raise self.post_response
        if (
            self.post_comment == "auto"
            and isinstance(self.post_response.payload, dict)
            and self.post_response.payload.get("id")
        ):
            self.post_comment = {
                "id": self.post_response.payload["id"],
                "body": kwargs["data"]["body"],
                "user": {"login": "me"},
            }
        if self.post_comment:
            item = dict(self.post_comment)
            item.setdefault("created_at", NOW())
            self.pages[0].append(item)
        return self.post_response


def issue():
    return "https://gitcode.com/me/repo/issues/9"


def test_preview_does_not_need_api_or_write(tmp_path):
    body = "hello\n"
    out = mod.execute(issue(), body, tmp_path / "result.json")
    assert out["body"] == body
    assert out["body_digest"] == DIGEST(body)
    assert not (tmp_path / "result.json").exists()


def test_pagination_and_same_actor_dedup(tmp_path):
    pages = [
        [{"id": 3, "body": "same", "user": {"login": "me"}}]
        + [{"id": 4, "body": "x", "user": {"login": "other"}}]
    ]
    s = Session(pages)
    result = mod.execute(issue(), "same", tmp_path / "r.json", apply=True, session=s)
    assert result["status"] == "reused"
    assert result["comment_id"] == 3
    assert not s.post_calls


def test_different_author_same_body_is_posted_once(tmp_path):
    s = Session([[{"id": 3, "body": "same", "user": {"login": "other"}}]],
                Response(payload={"id": 8}))
    result = mod.execute(issue(), "same", tmp_path / "r.json", apply=True, session=s)
    assert result["status"] == "verified"
    assert len(s.post_calls) == 1
    assert s.post_calls[0][1]["allow_redirects"] is False


def test_full_page_limit_blocks_post(tmp_path):
    s = Session([[{"id": i, "body": "x", "user": {"login": "other"}} for i in range(100)]] * 100,
                Response(payload={"id": 1}))
    with pytest.raises(RuntimeError, match="pagination limit"):
        mod.execute(issue(), "new", tmp_path / "r.json", apply=True, session=s)
    assert not s.post_calls


def test_timeout_readback_verifies_without_retry(tmp_path):
    s = Session([[]], requests.Timeout("timed out"), post_comment={"id": 22, "body": "new", "user": {"login": "me"}})
    result = mod.execute(issue(), "new", tmp_path / "r.json", apply=True, session=s)
    assert result["status"] == "verified"
    assert len(s.post_calls) == 1


def test_unknown_reinvoke_only_readback_no_repost(tmp_path):
    path = tmp_path / "r.json"
    first = Session([[]], requests.Timeout("timed out"), post_comment=None)
    result = mod.execute(issue(), "body", path, apply=True, session=first)
    assert result["status"] == "unknown"
    second = Session([[]], Response(payload={"id": 99}))
    again = mod.execute(issue(), "body", path, apply=True, session=second)
    assert again["status"] == "unknown"
    assert not second.post_calls


def test_post_json_missing_id_then_readback(tmp_path):
    s = Session([[]], Response(payload={"ok": True}), post_comment={"id": 2, "body": "b", "user": {"login": "me"}})
    result = mod.execute(issue(), "b", tmp_path / "r.json", apply=True, session=s)
    assert result["status"] == "verified"


def test_content_mismatch_is_failed(tmp_path):
    s = Session([[]], Response(payload={"id": 2}), post_comment={"id": 2, "body": "other", "user": {"login": "me"}})
    result = mod.execute(issue(), "b", tmp_path / "r.json", apply=True, session=s)
    assert result["status"] == "failed"


def test_body_credential_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("GITCODE_TOKEN", "secret-token")
    with pytest.raises(ValueError):
        mod.execute(issue(), "contains secret-token", tmp_path / "r.json")


def test_duplicate_on_second_page_prevents_post(tmp_path):
    first = [{"id": i, "body": "other", "user": {"login": "someone"}} for i in range(100)]
    s = Session([first, [{"id": 101, "body": "same", "user": {"login": "me"}}]])
    out = mod.execute(issue(), "same", tmp_path / "r.json", apply=True, session=s)
    assert out["status"] == "reused"
    assert out["comment_id"] == 101
    assert not s.post_calls


def test_known_id_cannot_be_replaced_by_another_same_body(tmp_path):
    s = Session([[]], Response(payload={"id": 8}), post_comment={"id": 9, "body": "same", "user": {"login": "me"}})
    out = mod.execute(issue(), "same", tmp_path / "r.json", apply=True, session=s)
    assert out["status"] == "unknown"
    assert out["comment_id"] == 8
    assert len(s.post_calls) == 1


@pytest.mark.parametrize("timestamp", [None, "invalid", "2099-01-01T00:00:00Z"])
def test_unknown_post_requires_valid_time(timestamp):
    c = {"id": 9, "body": "same", "user": {"login": "me"}, "created_at": timestamp}
    session = Session([[c]])
    found, status = READBACK(
        session, "https://api.test/comments", "test", "me", "same", since=NOW()
    )
    assert found is None and status == "unknown"


def test_terminal_result_never_reposts_deleted_comment(tmp_path):
    path = tmp_path / "r.json"
    first = Session([[]], Response(payload={"id": 8}))
    assert mod.execute(issue(), "same", path, apply=True, session=first)["status"] == "verified"
    second = Session([[]])
    assert mod.execute(issue(), "same", path, apply=True, session=second)["status"] == "unknown"
    assert not second.post_calls


def test_readback_failure_does_not_reuse_old_verified_status(tmp_path, monkeypatch):
    path = tmp_path / "r.json"
    first = Session([[]], Response(payload={"id": 8}))
    mod.execute(issue(), "same", path, apply=True, session=first)

    def fail(*args, **kwargs):
        raise requests.Timeout("sensitive access_token=never-print")

    monkeypatch.setattr(mod, "api_get", fail)
    assert mod.execute(issue(), "same", path, apply=True, session=Session([[]]))["status"] == "unknown"


def test_post_pending_record_exists_before_network_write(tmp_path):
    path = tmp_path / "r.json"

    class CheckingSession(Session):
        def post(self, url, **kwargs):
            saved = json.loads(path.read_text())
            assert saved["post_started"] is True
            assert saved["status"] == "pending"
            assert "body" not in saved and "access_token" not in path.read_text()
            return super().post(url, **kwargs)
    mod.execute(issue(), "same", path, apply=True, session=CheckingSession([[]], Response(payload={"id": 8})))


def test_changed_body_rejected_before_network(tmp_path):
    path = tmp_path / "r.json"
    mod.execute(issue(), "same", path, apply=True, session=Session([[]], Response(payload={"id": 8})))
    second = Session([[]])
    with pytest.raises(RuntimeError, match="conflicts"):
        mod.execute(issue(), "changed", path, apply=True, session=second)
    assert not second.get_calls and not second.post_calls


@pytest.mark.parametrize("server_date", ["Tue, 15 Sep 2026 10:00:00 GMT", "Tue, 15 Sep 2037 10:00:00 GMT"])
def test_unknown_post_uses_server_clock_despite_local_clock_skew(tmp_path, server_date):
    server_time = mod.parsedate_to_datetime(server_date).isoformat()

    class ClockSession(Session):
        def get(self, *args, **kwargs):
            response = super().get(*args, **kwargs)
            response.headers = {"Date": server_date}
            return response
    c = {"id": 22, "body": "clock-test", "user": {"login": "me"}, "created_at": server_time}
    session = ClockSession([[]], requests.Timeout("sent but response lost"), post_comment=c)
    out = mod.execute(issue(), "clock-test", tmp_path / "clock.json", apply=True, session=session)
    assert out["status"] == "verified"
    assert out["attempt_started_at"] == server_time
    assert out["clock_source"] == "gitcode_http_date"
    assert len(session.post_calls) == 1
