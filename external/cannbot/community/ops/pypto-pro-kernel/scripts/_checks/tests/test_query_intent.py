# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Combined symbol/intent discovery without changing exact or conditional queries."""

import pytest

from _checks import query


@pytest.mark.parametrize(
    "text",
    [
        "pl.jit postponed Tensor annotations dtype alias recognition __wrapped__ capture selected callable",
        "Investigate native entry classification with pl.jit before changing buffers",
        "postponed Tensor annotations with dtype aliases under @pl.jit() capture",
        "pl.jit does not recognize postponed Tensor annotations containing a dtype name",
        "pl.Vf.mul_add_dst inspect FMA destination before copying registers",
    ],
)
def test_qualified_symbol_and_owner_phrase_precede_generic_mentions(text):
    owner = "vf-destination-state" if "mul_add_dst" in text else "jit-entry-kind"
    result = query.report(text)
    row = result["knowledge"]["items"][0]
    assert row["id"] == "contract:" + owner
    assert row["match"]["reason"] == "symbol_intent"
    assert row["conditions"] and row["missing_facts"]
    assert "text" not in row and not result["context"]
    assert any(route["id"] == owner for route in result["related_contracts"])
    assert (
        row["evidence"]["native"]
        == row["evidence"]["numerical"]
        == row["evidence"]["performance"]
        == "not_established"
    )


@pytest.mark.parametrize(
    "text",
    [
        "pl.jitter postponed Tensor annotations dtype alias",
        "vf.jit postponed Tensor annotations dtype alias",
        "pl.store postponed Tensor annotations dtype alias",
        "pl.jit cache options; native entry classification is tracked separately",
        "pl.jit scalar parameter annotation, not postponed Tensor annotations",
        "pl.jit without postponed Tensor annotations inspect cache behavior",
        "pl.jit has no postponed Tensor annotations and cache behavior is the question",
        "pl.jit 没有 postponed Tensor annotations，需要检查缓存",
        "pl.jit postponed Tensor annotations are not the issue; inspect launch cache",
        "postponed Tensor annotations caused by an unrelated decorator",
    ],
)
def test_background_negated_or_foreign_symbols_do_not_promote_entry_owner(text):
    result = query.report(text)
    assert "contract:jit-entry-kind" not in {
        row["id"] for row in result["knowledge"]["items"]
    }
    assert not result["context"]


def test_combined_intent_keeps_unknown_filters_and_exact_symbol_resolution():
    result = query.report(
        "pl.jit native entry classification for an imported wrapper",
        filters={"source_memory": "L1"},
    )
    row = result["knowledge"]["items"][0]
    assert row["id"] == "contract:jit-entry-kind"
    assert row["applicability"]["status"] == "unknown"
    assert row["applicability"]["unknown_filters"] == ["source_memory"]
    assert not result["context"]
    exact = query.report("symbol:pl.jit", limit=1)["knowledge"]["items"][0]
    assert exact["id"] == "symbol:pl.jit" and exact["match"]["reason"] == "id"
