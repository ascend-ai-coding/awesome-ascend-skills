#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Consume Agent-reviewed responsibility decisions; never infer chips from text."""
from __future__ import annotations

import hashlib
import json

LEVELS = ("handle", "list-only", "ignore")


def policy_digest(policy: dict) -> str:
    return hashlib.sha256(
        json.dumps(policy, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def review_responsibility(issue: dict, policy: dict) -> dict:
    """Only a complete review of the current policy can enable handling.

    The review is local workflow data supplied by the Agent, not a field copied
    from the Issue body or a user's configurable matching expression.
    """
    pending = {"level": "pending", "summary": "待核查责任范围", "evidence": []}
    review = issue.get("responsibility_review")
    if not isinstance(review, dict):
        return pending
    level = review.get("level")
    summary = review.get("summary")
    evidence = review.get("evidence")
    invalid_level = level not in LEVELS
    stale_policy = review.get("policy_digest") != policy_digest(policy)
    invalid_summary = not isinstance(summary, str) or not summary.strip()
    invalid_evidence = (
        not isinstance(evidence, list)
        or not evidence
        or any(not isinstance(value, str) or not value.strip() for value in evidence)
    )
    invalid_review = any((invalid_level, stale_policy, invalid_summary, invalid_evidence))
    if invalid_review:
        return pending
    # Empty categories are deliberately disabled by the user.
    if not policy.get(level):
        return pending
    return {"level": level, "summary": " ".join(summary.split()), "evidence": evidence}
