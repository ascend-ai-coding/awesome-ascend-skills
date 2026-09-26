#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Versioned, agent-reviewed scope evidence, separate from response state."""

from __future__ import annotations

import hashlib
import json

from responsibility import policy_digest, review_responsibility


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def content_digest(issue):
    fields = ("title", "description", "body", "labels", "author")
    values = {}
    for key in fields:
        values[key] = issue.get(key)
    return digest(values)


def activity_digest(issue):
    # Metadata changes are conservative delta-review triggers: a new comment
    # can correct the reported version/path without editing the description.
    fields = ("updated_at", "comments_count", "assignee", "issue_state", "state")
    values = {}
    for key in fields:
        values[key] = issue.get(key)
    return digest(values)


def task_digest(issue, policy, ignored):
    return digest([content_digest(issue), activity_digest(issue), policy_digest(policy), str(issue["iid"]) in ignored])


def restore(entry, issue, policy):
    """Return (review or None, reason); never infer scope from keywords."""
    if not isinstance(entry, dict) or entry.get("version") != 1:
        return None, "missing_review"
    if entry.get("policy_digest") != policy_digest(policy):
        return None, "policy_changed"
    if entry.get("content_digest") != content_digest(issue):
        return None, "content_changed"
    if entry.get("activity_digest") != activity_digest(issue):
        return None, "activity_changed"
    if entry.get("source_mode") != "fixed":
        return None, "moving_source_requires_review"
    review = entry.get("review")
    if review_responsibility({"responsibility_review": review}, policy)["level"] == "pending":
        return None, "invalid_review"
    return review, "reused"


def make_entry(issue, policy, review, source_mode="fixed", provenance="coordinator"):
    if source_mode not in {"fixed", "moving"}:
        raise ValueError("invalid_source_mode")
    if review_responsibility({"responsibility_review": review}, policy)["level"] == "pending":
        raise ValueError("invalid_scope_review")
    return {
        "version": 1,
        "policy_digest": policy_digest(policy),
        "content_digest": content_digest(issue),
        "activity_digest": activity_digest(issue),
        "review": review,
        "source_mode": source_mode,
        "provenance": provenance,
    }
