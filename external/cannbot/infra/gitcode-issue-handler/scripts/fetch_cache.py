#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Small, durable caches for Issue comments and native PR linkage."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile

CACHE_VERSION = 1


def _repo_key(repo):
    return repo.replace("/", "__")


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)
    except (OSError, json.JSONDecodeError):
        return None


def _atomic_write_json(path, payload):
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=parent,
            prefix=".cache-",
            suffix=".tmp",
            delete=False,
        ) as file_obj:
            temp_path = file_obj.name
            json.dump(payload, file_obj, ensure_ascii=False, indent=2)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def _issue_number(issue):
    return issue.get("number") or issue.get("iid")


def _comment_cache_path(cache_dir, repo, issue):
    number = _issue_number(issue)
    return os.path.join(
        cache_dir,
        "comments",
        _repo_key(repo),
        f"issue-{number}.json",
    )


def load_comments(cache_dir, repo, issue):
    """Return cached comments when the Issue snapshot is unchanged."""
    if not cache_dir or _issue_number(issue) is None:
        return None
    payload = _read_json(_comment_cache_path(cache_dir, repo, issue))
    if not isinstance(payload, dict):
        return None
    expected = {
        "version": CACHE_VERSION,
        "repo": repo,
        "issue_number": str(_issue_number(issue)),
        "updated_at": issue.get("updated_at", ""),
        "comments_count": int(issue.get("comments_count", 0) or 0),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        return None
    comments = payload.get("comments")
    return comments if isinstance(comments, list) else None


def save_comments(cache_dir, repo, issue, comments):
    if not cache_dir or _issue_number(issue) is None:
        return
    payload = {
        "version": CACHE_VERSION,
        "repo": repo,
        "issue_number": str(_issue_number(issue)),
        "updated_at": issue.get("updated_at", ""),
        "comments_count": int(issue.get("comments_count", 0) or 0),
        "comments": comments,
    }
    _atomic_write_json(_comment_cache_path(cache_dir, repo, issue), payload)


def _pr_signature(pr):
    stable = {
        "updated_at": pr.get("updated_at", ""),
        "head_sha": (pr.get("head") or {}).get("sha", ""),
        "title": pr.get("title", ""),
        "body": pr.get("body", ""),
    }
    encoded = json.dumps(stable, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pr_cache_path(cache_dir, repo, pr_number):
    return os.path.join(
        cache_dir,
        "pr-links",
        _repo_key(repo),
        f"pr-{pr_number}.json",
    )


def load_pr_links(cache_dir, repo, pr):
    pr_number = pr.get("number")
    if not cache_dir or pr_number is None:
        return None
    payload = _read_json(_pr_cache_path(cache_dir, repo, pr_number))
    if not isinstance(payload, dict):
        return None
    expected = {
        "version": CACHE_VERSION,
        "repo": repo,
        "pr_number": str(pr_number),
        "signature": _pr_signature(pr),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        return None
    linked_issues = payload.get("linked_issues")
    return linked_issues if isinstance(linked_issues, list) else None


def save_pr_links(cache_dir, repo, pr, linked_issues):
    pr_number = pr.get("number")
    if not cache_dir or pr_number is None:
        return
    payload = {
        "version": CACHE_VERSION,
        "repo": repo,
        "pr_number": str(pr_number),
        "signature": _pr_signature(pr),
        "linked_issues": [str(number) for number in linked_issues],
    }
    _atomic_write_json(_pr_cache_path(cache_dir, repo, pr_number), payload)
