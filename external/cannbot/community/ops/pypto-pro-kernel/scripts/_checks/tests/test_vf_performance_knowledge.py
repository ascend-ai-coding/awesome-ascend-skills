# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""VF cost questions resolve conditional decisions without replacing API facts."""

import pytest

from _checks import query


@pytest.mark.parametrize(
    "text,owner,source",
    [
        (
            "VF metadata reused getval scalar broadcast performance",
            "vf-scalar-consumer",
            "patterns/vf-outer-reduce.md",
        ),
        (
            "VF 归约结果 getval 反复读取 广播 性能",
            "vf-scalar-consumer",
            "patterns/vf-outer-reduce.md",
        ),
        (
            "VF loop repeated runtime masks update_mask",
            "vf-loop-predicates",
            "api/vf.md",
        ),
        ("VF 循环重复构造掩码", "vf-loop-predicates", "api/vf.md"),
    ],
)
def test_cost_question_reaches_original_conditional_body(text, owner, source):
    result = query.report(text, limit=3)["knowledge"]
    row = next(item for item in result["items"] if item["id"] == "contract:" + owner)
    assert row["reference"]["path"].endswith(source)
    assert row["conditions"] and row["missing_facts"]
    assert row["evidence"]["performance"] == "not_established"
    resolved = query.report(**row["resolve"])["knowledge"]["items"][0]
    assert (
        resolved["text"]
        and resolved["reference"]["sha256"] == row["reference"]["sha256"]
    )


@pytest.mark.parametrize("symbol", ["pl.getval", "vf.load_align", "vf.update_mask"])
def test_transport_cost_topics_preserve_exact_api_queries(symbol):
    row = query.report(symbol, limit=1)["knowledge"]["items"][0]
    assert row["id"] == "symbol:" + symbol
    assert row["evidence"]["performance"] == "not_established"


@pytest.mark.parametrize(
    "text,excluded",
    [
        ("VF getval scalar loop-bound control", "vf-scalar-consumer"),
        ("vf.update_mask prefix count saturation", "vf-loop-predicates"),
    ],
)
def test_control_or_prefix_contract_is_not_a_repeated_cost_question(text, excluded):
    rows = query.report(text, kinds=["contract"])["knowledge"]["items"]
    assert "contract:" + excluded not in {row["id"] for row in rows}


@pytest.mark.parametrize(
    "text,owner",
    [
        ("VF repeated getval broadcast", "vf-scalar-consumer"),
        ("VF loop repeated mask construction", "vf-loop-predicates"),
    ],
)
def test_unresolved_physical_representation_remains_unknown(text, owner):
    rows = query.report(text, kinds=["contract"], filters={"source_memory": "L1"})[
        "knowledge"
    ]["items"]
    row = next(item for item in rows if item["id"] == "contract:" + owner)
    assert row["applicability"]["status"] == "unknown"
    assert row["applicability"]["unknown_filters"] == ["source_memory"]
    assert row["missing_facts"]
