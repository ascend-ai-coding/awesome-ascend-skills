# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""SIMT discovery must preserve conditional evidence and exact API resolution."""

import pytest

from _checks import query


@pytest.mark.parametrize(
    "text,owner",
    [
        ("SIMT 寄存器 spill", "simt-register-pressure"),
        ("SIMT launch bounds", "simt-register-pressure"),
        ("SIMT DCache", "tile-capacity"),
        ("SIMT cache", "simt-memory-diagnosis"),
        ("SIMT 访存合并", "simt-memory-layout"),
        ("SIMT 原子返回值", "simt-atomic-aggregation"),
        ("SIMT ticket reservation", "simt-ticket-ranges"),
        ("SIMT atomic sector", "simt-atomic-target-layout"),
        ("SIMT 索引溢出", "simt-index-arithmetic"),
        ("SIMT unsigned divmod", "simt-index-arithmetic"),
        ("SIMT loop unrolling", "simt-control-flow"),
        ("SIMT 谓词化", "simt-control-flow"),
        ("SIMT 查表驻留", "simt-table-residency"),
        ("SIMT 整行间接搬运", "simt-mte-payload"),
        ("SIMT 数学接口成本", "simt-math-lowering"),
    ],
)
def test_conditional_simt_topics_retain_unknown_applicability(text, owner):
    result = query.report(text, kinds=["contract"], filters={"source_memory": "L1"})
    row = next(
        item
        for item in result["knowledge"]["items"]
        if item["id"] == "contract:" + owner
    )
    assert row["missing_facts"]
    if owner != "tile-capacity":
        assert row["conditions"]
    assert row["applicability"]["status"] == "unknown"
    assert row["applicability"]["unknown_filters"] == ["source_memory"]
    assert (
        row["evidence"]["native"]
        == row["evidence"]["numerical"]
        == row["evidence"]["performance"]
        == "not_established"
    )
    resolved = query.report(**row["resolve"])["knowledge"]["items"][0]
    assert resolved["id"] == row["id"] and resolved["text"]
    assert resolved["reference"]["sha256"] == row["reference"]["sha256"]


@pytest.mark.parametrize(
    "symbol",
    [
        "pl.simt.atomic_add",
        "pl.simt.mul_hi",
        "pl.simt.warp_size",
        "pl.simt.log1p",
        "pl.simt.tanh",
    ],
)
def test_performance_topics_do_not_replace_exact_public_api(symbol):
    row = query.report(symbol)["knowledge"]["items"][0]
    assert row["id"] == "symbol:" + symbol
    assert row["evidence"]["performance"] == "not_established"


def test_upstream_cache_api_name_does_not_become_a_pro_declaration():
    result = query.report("symbol:pl.simt.ldca")
    assert result["knowledge"]["lookup"]["knowledge_absence"] == "not_established"
    assert all(
        row["id"] != "symbol:pl.simt.ldca" for row in result["knowledge"]["items"]
    )


@pytest.mark.parametrize(
    "text,owner,program",
    [
        ("SIMT tensor rotation layout", "simt-memory-layout", "simt_tiled_transpose"),
        (
            "SIMT categorical local counters",
            "simt-atomic-aggregation",
            "simt_histogram",
        ),
        ("SIMT ticket reservation", "simt-ticket-ranges", "simt_ticket_reservation"),
    ],
)
def test_workload_discovery_connects_owner_to_complete_native_program(
    text, owner, program
):
    result = query.report(text, kinds=["contract", "example"], limit=5)
    contract = next(
        row for row in result["knowledge"]["items"] if row["id"] == "contract:" + owner
    )
    identity = f"example:algorithms/{program}/kernel.py"
    relation = next(row for row in contract["relations"] if row["target"] == identity)
    assert relation["availability"] == "available" and relation["sha256"]
    resolved = query.report(identity, kinds=["example"])["knowledge"]["items"][0]
    assert set(resolved["program"]["entries"]) == {"kernel", "reference", "test"}
    assert resolved["program"]["entries"]["kernel"]["sha256"] == relation["sha256"]
    assert (
        resolved["evidence"]["numerical"]
        == resolved["evidence"]["performance"]
        == "not_established"
    )


def test_contiguous_permutation_recipe_is_excluded_for_explicit_arbitrary_input_strides():
    result = query.report(
        "SIMT transpose with arbitrary positive row and column strides",
        kinds=["example"],
    )
    identity = "example:algorithms/simt_tiled_transpose/kernel.py"
    assert all(row["id"] != identity for row in result["knowledge"]["items"])
    excluded = next(
        row
        for row in result["knowledge"]["rejected_conditions"]
        if row["id"] == identity
    )
    assert excluded["source_conditions"] and excluded["reference"]["sha256"]
    assert "example:algorithms/strided_transpose/kernel.py" in str(
        excluded["source_conditions"]
    )


def test_strided_lane_access_does_not_exclude_a_contiguous_input_recipe():
    result = query.report(
        "SIMT transpose contiguous input with strided lane stores", kinds=["example"]
    )
    assert any(
        row["id"] == "example:algorithms/simt_tiled_transpose/kernel.py"
        for row in result["knowledge"]["items"]
    )


def test_indirect_row_owner_resolves_native_mte_program():
    result = query.report("SIMT 整行间接搬运", kinds=["contract", "example"], limit=5)
    owner = next(
        row
        for row in result["knowledge"]["items"]
        if row["id"] == "contract:simt-mte-payload"
    )
    identity = "example:basic_api/datamove/indexed_rows/kernel.py"
    relation = next(row for row in owner["relations"] if row["target"] == identity)
    program = query.report(identity, kinds=["example"])["knowledge"]["items"][0]
    assert relation["availability"] == "available"
    assert set(program["program"]["entries"]) == {"kernel", "reference", "test"}
    assert relation["sha256"] == program["program"]["entries"]["kernel"]["sha256"]
    assert (
        program["evidence"]["numerical"]
        == program["evidence"]["performance"]
        == "not_established"
    )


@pytest.mark.parametrize(
    "text",
    [
        "indirect row copy with noncontiguous input",
        "indirect row copy with arbitrary input strides",
        "indirect row copy with noncontiguous output",
    ],
)
def test_contiguous_row_copy_is_excluded_for_explicit_noncontiguous_layouts(text):
    result = query.report(text, kinds=["example"])
    identity = "example:basic_api/datamove/indexed_rows/kernel.py"
    assert all(row["id"] != identity for row in result["knowledge"]["items"])
    excluded = next(
        row
        for row in result["knowledge"]["rejected_conditions"]
        if row["id"] == identity
    )
    assert excluded["source_conditions"] and excluded["reference"]["sha256"]


def test_strided_lane_access_does_not_exclude_contiguous_row_copy():
    result = query.report(
        "indirect row copy contiguous input with strided lane stores", kinds=["example"]
    )
    assert any(
        row["id"] == "example:basic_api/datamove/indexed_rows/kernel.py"
        for row in result["knowledge"]["items"]
    )
