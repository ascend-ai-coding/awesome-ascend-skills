# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Read each target Issue's PR evidence independently of the recent-PR window."""
import re

import requests

from gitcode_client import api_get, make_session
from pr_response_policy import pr_active, pr_author, pr_inactive


def _pr_number(record):
    if not isinstance(record, dict):
        raise ValueError("PR evidence must be an object")
    number = str(record.get("number") or record.get("pr_number") or "")
    if not number.isdigit():
        raise ValueError("PR evidence lacks a numeric PR number")
    return number


class IssuePrReader:
    """Share one HTTP session and verified PR details across target Issues."""

    def __init__(self, options, known_prs):
        self.options = options
        self.session = make_session(rate_limiter=options.rate_limiter)
        self.endpoint = f"{options.api_base}/repos/{options.repo}"
        self.details = {_pr_number(pr): pr for pr in known_prs}
        self.api_calls = 0
        self.link_pattern = re.compile(
            r"https://(?:www\.)?gitcode\.com/" + re.escape(options.repo)
            + r"/(?:pull|pulls|merge_requests)/(\d+)(?=$|[/#?\s)>])", re.IGNORECASE,
        )

    @staticmethod
    def _complete_pr(record):
        if not isinstance(record, dict) or not pr_author(record):
            return False
        return pr_active(record) or pr_inactive(record)

    def read(self, issue):
        number = issue.get("number") or issue.get("iid")
        native = self._native_prs(number)
        numbers = set(native) | self._explicit_numbers(issue)
        return [self._detail(pr_number, native.get(pr_number)) for pr_number in sorted(numbers, key=int)]

    def _get(self, path, params=None):
        self.api_calls += 1
        return api_get(self.session, f"{self.endpoint}/{path}", self.options.token, params=params)

    def _native_prs(self, number):
        found = {}
        for page in range(1, self.options.max_pages + 1):
            batch = self._get(f"issues/{number}/pull_requests", {
                "mode": 0, "page": page, "per_page": 100,
            })
            if not isinstance(batch, list):
                raise ValueError("Issue PR response must be a list")
            for record in batch:
                found[_pr_number(record)] = record
            if len(batch) < 100:
                return found
        raise ValueError("Issue PR pagination limit reached")

    def _explicit_numbers(self, issue):
        texts = [str(issue.get(key) or "") for key in ("title", "body", "description")]
        texts.extend(str(comment.get("body") or "") for comment in issue.get("comments") or [])
        numbers = set(self.link_pattern.findall("\n".join(texts)))
        return numbers

    def _detail(self, number, native):
        record = native or self.details.get(number)
        closed_without_merge_evidence = bool(
            record and str(record.get("state") or "").casefold() in {"closed", "close"}
            and not any(key in record for key in ("merged", "pr_merged", "merged_at"))
        )
        if not self._complete_pr(record) or closed_without_merge_evidence:
            record = self._get(f"pulls/{number}")
        if _pr_number(record) != number or not self._complete_pr(record):
            raise ValueError("PR detail is incomplete or does not match its requested number")
        self.details[number] = record
        return dict(record, number=int(number))


def collect_issue_prs(issues, options, known_prs):
    """Return raw PRs by Issue; failed reads block only the affected Issue."""
    reader = IssuePrReader(options, known_prs)
    mapping = {}
    errors = []
    for issue in issues:
        number = str(issue.get("number") or issue.get("iid"))
        try:
            mapping[number] = reader.read(issue)
        except (requests.RequestException, ValueError, TypeError) as exc:
            errors.append({"issue_number": number, "error": type(exc).__name__})
    return mapping, {
        "complete": not errors,
        "api_calls": reader.api_calls,
        "errors": errors,
        "incomplete_issue_numbers": [error["issue_number"] for error in errors],
    }
