# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reuse settled classifications only for unchanged, assigned, answered Issues.

This is a read-only routing shortcut, never authorization for a mutation.
Explicit single-Issue handling and forced comment refresh bypass it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import NamedTuple

from fetch_cache import _atomic_write_json, _read_json

_SETTLED = {"our_team_replied", "our_team_done_with_pr", "awaiting_assignee", "awaiting_reporter"}
_ISSUE_SIGNATURE_FIELDS = (
    "number", "iid", "url", "created_at", "title", "description", "body", "author", "assignee",
    "state", "issue_state", "updated_at", "comments_count", "followup_watch",
    "labels", "issue_type", "type", "category",
    "responsibility_review", "operator_routing_required",
)
_POLICY_SIGNATURE_FIELDS = (
    "repo", "gitcode_api", "responsibility", "auto-response", "auto-assign", "auto-assign-fallback-user",
)


class ClassificationCacheEntry(NamedTuple):
    cache_dir: str
    repo: str
    issue: dict
    config: dict
    revision: str
    result: dict


def _path(cache_dir, repo, issue):
    number = issue.get("number") or issue.get("iid")
    return str(Path(cache_dir) / "classifications" / repo.replace("/", "__") / f"issue-{number}.json")


def _signature(issue, config, revision):
    # Include watch state and responsibility evidence: unchanged remote metadata
    # does not mean a changed local routing decision can be reused.
    values = {key: issue.get(key) for key in _ISSUE_SIGNATURE_FIELDS}
    values["policy"] = {key: config.get(key) for key in _POLICY_SIGNATURE_FIELDS}
    values["explicitly_ignored"] = str(issue.get("iid") or issue.get("number")) in {
        str(i) for i in config.get("ignored_issue_ids", [])
    }
    values["revision"] = revision
    return hashlib.sha256(json.dumps(values, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_settled(cache_dir, repo, issue, config, revision):
    missing_identity = not cache_dir or not issue.get("assignee") or not issue.get("updated_at")
    missing_comment_count = "comments_count" not in issue
    issue_not_open = issue.get("state") not in {"open", "opened"}
    if missing_identity or missing_comment_count or issue_not_open:
        return None
    payload = _read_json(_path(cache_dir, repo, issue))
    if not isinstance(payload, dict) or payload.get("signature") != _signature(issue, config, revision):
        return None
    result = payload.get("result")
    if not isinstance(result, dict) or result.get("bucket") != "no_attention":
        return None
    if result.get("category") not in _SETTLED or not result.get("latest_maintainer_comment_id"):
        return None
    # PR state may change without changing Issue metadata. Refresh PR-dependent
    # classifications instead of treating an old link as completion evidence.
    if result.get("linked_prs"):
        return None
    # Supplied comments are newer evidence than an old routing result, even if
    # a caller forgot to update the Issue timestamp/count. Recompute locally.
    if "comments" in issue:
        return None
    return dict(result)


def save_classification(entry: ClassificationCacheEntry):
    if not entry.cache_dir or not entry.issue.get("updated_at"):
        return
    _atomic_write_json(_path(entry.cache_dir, entry.repo, entry.issue), {
        "signature": _signature(entry.issue, entry.config, entry.revision),
        "result": entry.result,
    })
