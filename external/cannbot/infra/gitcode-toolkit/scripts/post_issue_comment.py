#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Preview or idempotently post one GitCode Issue comment."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import sys
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _logging import init_error_logger, init_output_logger
from gitcode_client import api_get, make_session, parse_issue_url, resolve_api_base

MAX_PAGES = 100
PAGE_SIZE = 100
_PEM = re.compile(r"-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----")


class _ClockedSession:
    """Keep GitCode's response clock without changing the shared API client."""

    def __init__(self, session):
        self.session = session
        self.server_time = None

    def __getattr__(self, name):
        return getattr(self.session, name)

    def get(self, *args, **kwargs):
        response = self.session.get(*args, **kwargs)
        date = (getattr(response, "headers", None) or {}).get("Date")
        if date:
            try:
                self.server_time = parsedate_to_datetime(date).astimezone(timezone.utc)
            except (ValueError, TypeError, OverflowError):
                self.server_time = None
        return response


def _target(issue_url):
    owner, repo, number = parse_issue_url(issue_url)
    return {"owner": owner, "repo": repo, "issue_number": str(number), "url": issue_url}


def _digest(body):
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_body(body):
    if not body.strip():
        raise ValueError("comment body must not be empty")
    token = os.environ.get("GITCODE_TOKEN")
    if token and token in body:
        raise ValueError("comment body contains the configured token")
    if _PEM.search(body):
        raise ValueError("comment body contains a credential-like value")


def _comment_author(comment):
    user = comment.get("user") or comment.get("author") or {}
    if isinstance(user, dict):
        return user.get("login")
    return user


def _comment_id(comment):
    return comment.get("id") or comment.get("comment_id")


def _comment_url(comment):
    return comment.get("html_url") or comment.get("url")


def _comments(session, endpoint, token):
    """Return all comments, refusing to assume page 100 was complete."""
    found = []
    for page in range(1, MAX_PAGES + 1):
        payload = api_get(session, endpoint, token, params={"per_page": PAGE_SIZE, "page": page})
        if not isinstance(payload, list):
            if isinstance(payload, dict) and isinstance(payload.get("data"), list):
                payload = payload["data"]
            else:
                raise RuntimeError("comments response was not a JSON list")
        found.extend(payload)
        if len(payload) < PAGE_SIZE:
            return found
    raise RuntimeError("comment pagination limit reached before completion")


def _readback(session, endpoint, token, actor, body, *, wanted_id=None, since=None):
    comments = _comments(session, endpoint, token)
    exact = []
    for comment in comments:
        if wanted_id is not None:
            if str(_comment_id(comment)) == str(wanted_id):
                matches = _comment_author(comment) == actor and comment.get("body") == body
                return (comment, "verified") if matches else (None, "failed")
            continue
        if _comment_author(comment) != actor or comment.get("body") != body:
            continue
        created = comment.get("created_at") or comment.get("createdAt")
        if since and (not created or not _after(created, since, now=getattr(session, "server_time", None))):
            continue
        exact.append(comment)
    return (exact[-1], "verified") if exact else (None, "unknown")


def _after(value, since, now=None):
    try:
        text = str(value).replace("Z", "+00:00")
        created = datetime.fromisoformat(text)
        started = datetime.fromisoformat(since).replace(microsecond=0)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return started <= created <= (now or datetime.now(timezone.utc)) + timedelta(seconds=5)
    except (TypeError, ValueError):
        return False


@contextmanager
def _result_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _load_result(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise RuntimeError("result file is unreadable") from exc
    if not isinstance(data, dict):
        raise RuntimeError("result file is invalid")
    return data


def _save_result(path, result):
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _base_result(target, digest, actor=None):
    return {"target": target, "body_digest": digest, "actor": actor,
            "attempt_started_at": _now(), "status": "pending",
            "post_started": False, "comment_id": None, "url": None}


def _current_actor(session, issue_url, token):
    current = api_get(session, f"{resolve_api_base(repo_url=issue_url)}/user", token)
    actor = current.get("login") if isinstance(current, dict) else None
    if not actor:
        raise RuntimeError("current user response has no login")
    return actor


class CommentOperation(NamedTuple):
    session: object
    endpoint: str
    token: str
    body: str
    result: dict
    path: Path
    target: dict
    actor: str | None = None


def _save_readback(operation, comment, status):
    result = operation.result
    if comment:
        result.update(
            status="verified",
            comment_id=_comment_id(comment),
            url=_comment_url(comment) or operation.target["url"],
        )
        if operation.actor is not None:
            result["actor"] = operation.actor
    else:
        result["status"] = status
    _save_result(operation.path, result)
    return result


def _resume_started(operation, old, digest, issue_url):
    if not old:
        return None
    if old.get("target") != operation.target or old.get("body_digest") != digest:
        raise RuntimeError("result file conflicts with target or body")
    started = old.get("post_started") or old.get("status") == "unknown" or old.get("comment_id")
    if not started:
        return None
    actor = old.get("actor")
    try:
        current_actor = _current_actor(operation.session, issue_url, operation.token)
    except Exception:
        old["status"] = "unknown"
        _save_result(operation.path, old)
        return old
    if actor and actor != current_actor:
        raise RuntimeError("result file actor conflicts with current account")
    actor = actor or current_actor
    try:
        comment, status = _readback(
            operation.session, operation.endpoint, operation.token, actor, operation.body,
            wanted_id=old.get("comment_id"), since=old.get("attempt_started_at"),
        )
    except Exception:
        old["status"] = "unknown"
        _save_result(operation.path, old)
        return old
    return _save_readback(operation._replace(actor=actor), comment, status)


def _reuse_existing(operation, comments):
    for comment in comments:
        if _comment_author(comment) == operation.actor and comment.get("body") == operation.body:
            operation.result.update(
                status="reused",
                comment_id=_comment_id(comment),
                url=_comment_url(comment) or operation.target["url"],
            )
            _save_result(operation.path, operation.result)
            return operation.result
    return None


def _post_comment(operation):
    response = operation.session.post(
        operation.endpoint,
        data={"access_token": operation.token, "body": operation.body},
        timeout=30, allow_redirects=False,
    )
    operation.result["post_http_status"] = response.status_code
    payload = response.json() if response.status_code < 300 else None
    post_id = payload.get("id") if isinstance(payload, dict) else None
    if post_id:
        operation.result["comment_id"] = post_id
        _save_result(operation.path, operation.result)
    if not (200 <= response.status_code < 300 and post_id):
        raise RuntimeError("comment POST did not return a usable ID")
    return post_id


def _readback_after_post(operation, post_id=None):
    try:
        comment, status = _readback(
            operation.session, operation.endpoint, operation.token,
            operation.actor, operation.body, wanted_id=post_id,
            since=None if post_id else operation.result.get("attempt_started_at"),
        )
    except Exception:
        comment, status = None, "unknown"
    return _save_readback(operation, comment, status)


def execute(issue_url, body, result_file, *, apply=False, session=None):
    target = _target(issue_url)
    digest = _digest(body)
    _safe_body(body)
    preview = {"status": "preview", "target": target, "body_digest": digest, "body": body}
    if not apply:
        return preview
    token = os.environ.get("GITCODE_TOKEN")
    if not token:
        raise RuntimeError("GITCODE_TOKEN is required with --apply")
    path = Path(result_file)
    endpoint = (
        f"{resolve_api_base(repo_url=issue_url)}/repos/{target['owner']}/"
        f"{target['repo']}/issues/{target['issue_number']}/comments"
    )
    session = _ClockedSession(session or make_session())
    with _result_lock(path):
        old = _load_result(path)
        operation = CommentOperation(session, endpoint, token, body, old, path, target)
        resumed = _resume_started(operation, old, digest, issue_url)
        if resumed is not None:
            return resumed
        actor = _current_actor(session, issue_url, token)
        old = old or _base_result(target, digest, actor)
        old["actor"] = actor
        old["status"] = "pending"
        _save_result(path, old)
        operation = operation._replace(result=old, actor=actor)
        comments = _comments(session, endpoint, token)
        reused = _reuse_existing(operation, comments)
        if reused is not None:
            return reused
        old["attempt_started_at"] = session.server_time.isoformat() if session.server_time else _now()
        old["clock_source"] = "gitcode_http_date" if session.server_time else "local"
        old["post_started"] = True
        _save_result(path, old)
        try:
            post_id = _post_comment(operation)
        except Exception:
            return _readback_after_post(operation)
        return _readback_after_post(operation, post_id)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue", required=True)
    parser.add_argument("--body-file", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    try:
        body = Path(args.body_file).read_text(encoding="utf-8")
        result = execute(args.issue, body, args.result_file, apply=args.apply)
        if not args.apply:
            result["status"] = "preview"
        init_output_logger().info("%s", json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") in {"preview", "verified", "reused"} else 1
    except Exception:
        init_error_logger().error(
            "Error: operation failed; inspect the result file before retrying"
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
