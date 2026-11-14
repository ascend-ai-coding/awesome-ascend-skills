# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise direct association reads without accessing any live repository."""
from unittest.mock import patch

import pytest

from test_classify_issues import CLASSIFIER, comment, issue, pull_request
import issue_pr_evidence as evidence


@pytest.fixture
def options():
    return CLASSIFIER.PRFetchOptions(
        api_base="https://api.example.test", repo="cann/ops-math", token="test-token", max_pages=2,
    )


def test_explicit_body_and_comment_links_fetch_old_pr_once(options):
    url = "https://gitcode.com/cann/ops-math/merge_requests/77"
    source = issue(description=f"对应修复：{url}", comments=[comment(body=f"补充：{url}")])
    source["comments"].append(comment(body="https://gitcode.com/other/repo/merge_requests/88"))
    with patch.object(evidence, "api_get", side_effect=[[], pull_request(77)]) as get:
        mapping, diagnostics = evidence.collect_issue_prs([source], options, [])
    assert [pr["number"] for pr in mapping["1"]] == [77]
    assert diagnostics["complete"] is True
    assert get.call_count == 2
    assert get.call_args.args[1].endswith("/pulls/77")
    assert get.call_args.args[2] == "test-token"


def test_native_pr_state_overrides_recent_snapshot(options):
    source = issue()
    known = pull_request(77, state="open")
    closed = pull_request(77, state="closed")
    with patch.object(evidence, "api_get", side_effect=[[{"number": 77, "state": "closed"}], closed]):
        mapping, diagnostics = evidence.collect_issue_prs([source], options, [known])
    assert diagnostics["complete"] is True
    assert mapping["1"][0]["state"] == "closed"


def test_closed_native_summary_does_not_hide_merged_pr(options):
    closed = pull_request(77, state="closed")
    with patch.object(evidence, "api_get", side_effect=[[closed], {**closed, "merged": True}]) as get:
        mapping, diagnostics = evidence.collect_issue_prs([issue()], options, [])
    assert diagnostics["complete"] is True
    assert mapping["1"][0]["merged"] is True
    assert get.call_count == 2


@pytest.mark.parametrize("payload", [{}, [{"title": "missing number"}], [{"number": "invalid"}]])
def test_malformed_association_is_not_treated_as_no_pr(options, payload):
    with patch.object(evidence, "api_get", return_value=payload):
        mapping, diagnostics = evidence.collect_issue_prs([issue()], options, [])
    assert mapping == {}
    assert diagnostics["complete"] is False
    assert diagnostics["incomplete_issue_numbers"] == ["1"]


def test_native_pagination_does_not_hide_same_author_pr(options):
    first_page = [pull_request(number) for number in range(100, 200)]
    own = pull_request(77, user={"login": "reporter"})
    with patch.object(evidence, "api_get", side_effect=[first_page, [own]]) as get:
        mapping, diagnostics = evidence.collect_issue_prs([issue()], options, [])
    assert diagnostics["complete"] is True
    assert len(mapping["1"]) == 101
    assert any(pr["user"]["login"] == "reporter" for pr in mapping["1"])
    assert get.call_args.kwargs["params"]["page"] == 2


def test_full_page_at_limit_keeps_evidence_incomplete(options):
    limited = options._replace(max_pages=1)
    batch = [pull_request(number) for number in range(100, 200)]
    with patch.object(evidence, "api_get", return_value=batch) as get:
        mapping, diagnostics = evidence.collect_issue_prs([issue()], limited, [])
    assert not mapping
    assert diagnostics["incomplete_issue_numbers"] == ["1"]
    assert get.call_count == 1
