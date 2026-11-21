# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared PR eligibility and evidence-based response assignment policy."""
from __future__ import annotations


def pr_author(pr):
    """Accept classifier records and live API records."""
    value = pr.get("pr_author") or pr.get("user") or pr.get("author") or ""
    return str(value.get("login") or "").strip() if isinstance(value, dict) else str(value).strip()


def pr_merged(pr):
    return bool(pr.get("merged") is True or pr.get("pr_merged") is True
                or pr.get("merged_at") or str(pr.get("state") or pr.get("pr_state") or "").casefold() == "merged")


def pr_expired(pr):
    """Expiration requires explicit evidence, never an inactivity threshold."""
    return bool((pr.get("expired") is True or pr.get("pr_expired") is True)
                and pr.get("expiration_evidence"))


def pr_active(pr):
    state = str(pr.get("state") or pr.get("pr_state") or "").casefold()
    return not pr_expired(pr) and (pr_merged(pr) or state in {"open", "opened"})


def pr_inactive(pr):
    state = str(pr.get("state") or pr.get("pr_state") or "").casefold()
    return pr_expired(pr) or (state in {"closed", "close"} and not pr_merged(pr))


def self_authored(author, prs):
    normalized_author = str(author or "").strip().casefold()
    return bool(normalized_author and any(pr_author(pr).casefold() == normalized_author for pr in prs))


def _pr_order(pr):
    url = str(pr.get("pr_url") or pr.get("html_url") or "")
    number = str(pr.get("pr_number") or pr.get("number") or url.rstrip("/").split("/")[-1])
    return int(number) if number.isdigit() else float("inf"), url


def select_response_pr(prs, issue_author):
    """Select from verified linked records; the Agent supplies semantic coverage.

    Multiple-PR comparisons use the same Issue-point identifiers for every PR,
    backed by actual description/diff evidence. File/line counts are irrelevant.
    """
    active = sorted((pr for pr in prs if pr_active(pr)), key=_pr_order)
    own = [pr for pr in active if self_authored(issue_author, [pr])]
    if own:
        return own[0], "self_authored"
    if not active or any(not pr_author(pr) for pr in active):
        raise ValueError("linked PR author evidence is incomplete")
    if len(active) == 1:
        return active[0], "single_author"
    ranked = []
    for pr in active:
        review = pr.get("coverage_review") or {}
        if not isinstance(review, dict):
            raise ValueError("coverage_review must be an object")
        points = review.get("covered_issue_points")
        evidence = review.get("evidence")
        for values in (points, evidence):
            if (not isinstance(values, list) or not values
                    or not all(isinstance(item, str) and item.strip() for item in values)):
                raise ValueError("multiple linked PRs require coverage_review for every valid PR")
        rank = (-len(set(points)), -int(review.get("target_version_match") is True), _pr_order(pr))
        ranked.append((rank, pr))
    return min(ranked, key=lambda entry: entry[0])[1], "issue_coverage"
