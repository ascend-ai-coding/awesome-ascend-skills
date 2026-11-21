# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Knowledge identity, routing, applicability and stateless source-bound browsing."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from _checks import feedback, knowledge, query
from _checks.paths import SKILL_ROOT

CASES = json.loads((Path(__file__).parent / "data/knowledge_queries.json").read_text())


def test_mutex_warning_routes_to_lifetime_review_and_example():
    result = query.report("MutexMismatch", kinds=["contract"], limit=1)
    (item,) = result["knowledge"]["items"]
    assert item["id"] == "contract:manual-mutex-phase-reuse"
    assert "PIPE_ALL" in item["text"] and "attn_epilogue/kernel.py" in item["text"]
    assert "declaration" in item["text"] and "does not prove" in item["text"]
    assert {r["target"] for r in item["relations"]} >= {
        "diagnostic:overlapping-groups-need-shared-mutex",
        "example:algorithms/attn_epilogue/kernel.py",
    }


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_retrieval_contract(case):
    result = query.report(**case["query"])
    view = result["knowledge"]
    assert view["schema"] == knowledge.SCHEMA
    assert {row["id"] for row in view["items"]} == set(case["ids"])
    assert view["lookup"]["status"] == case.get("lookup", "matched")
    assert view["lookup"]["knowledge_absence"] == "not_established"
    assert set(result["validation"].values()) <= {"not_established", "not_run"}
    for row in view["items"]:
        assert (
            row["reference"]["sha256"]
            == feedback.file_identity(Path(row["reference"]["path"]))["sha256"]
        )
        assert row["match"]["reason"] and row["missing_facts"]
        assert (
            row["evidence"]["native"]
            == row["evidence"]["numerical"]
            == row["evidence"]["performance"]
            == "not_established"
        )
        if row["kind"] in {"document", "example"}:
            assert row["resolve"]["text"].split("#", 1)[0] == row["id"]
            assert row["resolve"]["reference_sha256"] == row["reference"]["sha256"]
        else:
            assert row["resolve"] == {"text": row["id"]}
        if "kind" in case:
            assert row["kind"] == case["kind"]
        if "unknown_filters" in case:
            assert row["applicability"]["status"] == "unknown"
            assert row["applicability"]["unknown_filters"] == case["unknown_filters"]
    if case.get("no_recipe"):
        assert not result["context"] and all("text" not in row for row in view["items"])
    if case.get("kind") == "constant":
        assert not result["signatures"] and view["items"][0]["callable"] is False


def test_stable_ids_resolve_to_source_and_contract_without_rewriting_facts():
    first = query.report("从UB向L1搬运", kinds=["contract"])["knowledge"]["items"][0]
    resolved = query.report(**first["resolve"])["knowledge"]["items"][0]
    assert resolved["id"] == first["id"] and "pl.insert" in resolved["text"]
    assert resolved["reference"] == first["reference"] and resolved["conditions"]
    assert {r["target"] for r in resolved["relations"]} >= {
        "symbol:pl.move",
        "symbol:pl.insert",
        "example:basic_api/datamove/ub_l1_handover/kernel.py",
    }
    assert all(
        r.get("discovery", "available") == "available"
        for r in resolved["relations"]
        if r["kind"] == "example"
    )
    symbol = query.report("symbol:pl.DT_FP16")["knowledge"]["items"][0]
    assert symbol["kind"] == "constant" and "DT_FP16" in symbol["declaration"]


@pytest.mark.parametrize(
    "text",
    [
        "VF add with carry UINT32 addc output carry target declaration",
        "pl.Vf addc carry destination",
        "VF的addc输出进位如何声明",
    ],
)
def test_bare_vf_name_in_a_question_delivers_separate_public_sources(text):
    view = query.report(text, limit=3)["knowledge"]
    assert [row["id"] for row in view["items"]] == [
        "symbol:vf.addc",
        "document:pro-api/SIMD-API/Reg-API/basic_arithmetic/addc.md",
        "docstring:vf.addc",
    ]
    assert all(row["match"]["reason"] == "public_name_hint" for row in view["items"])
    assert all(
        row["evidence"]["numerical"] == "not_established" for row in view["items"]
    )


@pytest.mark.parametrize(
    "text",
    [
        "请比较vf.add和vf.sub的输出声明和调用",
        "Compare pl.Vf.add and pl.Vf.sub call operands",
    ],
)
def test_multiple_qualified_api_clues_keep_original_declarations_and_official_roles(
    text,
):
    view = query.report(text, kinds=["api", "document"], limit=10)["knowledge"]
    rows = {row["id"]: row for row in view["items"]}
    assert {"symbol:vf.add", "symbol:vf.sub"} <= rows.keys()
    assert {
        row["document_role"] for row in rows.values() if row["kind"] == "document"
    } >= {"official_pro_api", "source_docstring"}
    assert all(row["evidence"]["native"] == "not_established" for row in rows.values())


@pytest.mark.parametrize(
    "text",
    ["UB Vec TileType capacity streaming tile", "查询UB容量和槽位地址的容量检查"],
)
def test_capacity_scenarios_select_the_original_contract_without_an_algorithm(text):
    view = query.report(text, kinds=["contract", "document", "example"], limit=3)[
        "knowledge"
    ]
    row = view["items"][0]
    assert (
        row["id"] == "contract:tile-capacity"
        and row["evidence"]["native"] == "not_established"
    )
    assert row["reference"]["path"].endswith("api/memory-limits.md")
    assert (
        "_get_memory_capacity" in row["text"]
        and "_infer_dynamic_ub_size" in row["text"]
    )
    assert not any(item["kind"] == "example" for item in view["items"])
    assert "recovery" not in view
    filtered = query.report(text, kinds=["contract"], filters={"source_memory": "L1"})[
        "knowledge"
    ]
    row = next(
        item for item in filtered["items"] if item["id"] == "contract:tile-capacity"
    )
    assert row["applicability"]["unknown_filters"] == ["source_memory"]


def test_filters_find_related_symbols_without_claiming_they_implement_the_route():
    view = query.report(filters={"source_memory": "UB", "destination_memory": "L1"})[
        "knowledge"
    ]
    records = {row["id"]: row for row in view["items"]}
    assert {"contract:ub-l1", "symbol:pl.move", "symbol:pl.insert"} <= set(records)
    assert all(
        row["applicability"]["status"] == "unknown"
        for row in records.values()
        if row["kind"] == "document"
    )
    assert records["contract:ub-l1"]["applicability"]["matched_filters"] == {
        "source_memory": "Vec",
        "destination_memory": "Mat",
    }
    for identity in ("symbol:pl.move", "symbol:pl.insert"):
        row = records[identity]
        assert row["applicability"]["status"] == "unknown"
        assert row["match"]["related_contracts"] == ["contract:ub-l1"]


def test_declared_filter_owner_stays_visible_as_linked_documents_expand(
    tmp_path, monkeypatch
):
    references = tmp_path / "references"
    references.mkdir()
    marker = {
        "id": "transfer",
        "aliases": ["forward transfer"],
        "symbols": ["pl.move", "pl.insert"],
        "facets": {"source_memory": ["Vec"], "destination_memory": ["Mat"]},
    }
    links = []
    for index in range(knowledge.DEFAULT_LIMIT + 3):
        name = f"guide-{index}.md"
        (references / name).write_text(
            f"# Related guide {index}\nA linked source does not declare transfer conditions.\n"
        )
        links.append(f"[Guide {index}]({name})")
    (references / "owner.md").write_text(
        "<!-- kernel-query: "
        + json.dumps(marker)
        + " -->\n## Transfer\n"
        + "\n".join(links)
    )
    monkeypatch.setattr(query, "REFERENCES_ROOT", references)
    view = query.report(
        filters={"source_memory": "UB", "destination_memory": "L1"}, limit=3
    )["knowledge"]
    assert [row["id"] for row in view["items"]] == [
        "contract:transfer",
        "symbol:pl.insert",
        "symbol:pl.move",
    ]
    assert view["page"]["total"] > knowledge.DEFAULT_LIMIT
    assert view["items"][0]["applicability"]["matched_filters"] == {
        "source_memory": "Vec",
        "destination_memory": "Mat",
    }
    assert all(row["applicability"]["status"] == "unknown" for row in view["items"][1:])


def test_browsing_is_complete_deterministic_and_limit_can_change():
    expected = query.report(kinds=["contract"], limit=50)["knowledge"]
    page = query.report(kinds=["contract"], limit=2)["knowledge"]
    observed = list(page["items"])
    while page["page"]["next_cursor"]:
        page = query.report(
            kinds=["contract"], limit=3, cursor=page["page"]["next_cursor"]
        )["knowledge"]
        observed.extend(page["items"])
    assert [row["id"] for row in observed] == [row["id"] for row in expected["items"]]
    assert (
        len(observed)
        == len({row["id"] for row in observed})
        == expected["page"]["total"]
    )


def test_moving_reference_files_changes_locations_but_not_identity_or_rank(
    tmp_path, monkeypatch
):
    original = tmp_path / "a.md"
    original.write_text(
        '<!-- kernel-query: {"id":"b","aliases":["shared"]} -->\n## Beta\nBody B\n\n<!-- kernel-query: {"id":"a","aliases":["shared"]} -->\n## Alpha\nBody A\n'
    )
    monkeypatch.setattr(query, "REFERENCES_ROOT", tmp_path)
    before = query.report("shared", kinds=["contract"])["knowledge"]
    original.rename(tmp_path / "z.md")
    after = query.report("shared", kinds=["contract"])["knowledge"]
    assert (
        [r["id"] for r in before["items"]]
        == [r["id"] for r in after["items"]]
        == ["contract:a", "contract:b"]
    )
    assert (
        before["items"][0]["reference"]["path"]
        != after["items"][0]["reference"]["path"]
    )


def test_cursor_rejects_changed_query_conditions_and_malformed_tokens():
    cursor = query.report(kinds=["contract"], limit=1)["knowledge"]["page"][
        "next_cursor"
    ]
    for kwargs in (
        {"kinds": ["api"]},
        {"text": "move", "kinds": ["contract"]},
        {"kinds": ["contract"], "filters": {"dtype": "FP16"}},
    ):
        with pytest.raises(ValueError, match="cursor"):
            query.report(**kwargs, cursor=cursor)
    for value in ("bad", "{}", "%%%%", "W10", "bnVsbA"):
        with pytest.raises(ValueError, match="cursor"):
            query.report(kinds=["contract"], cursor=value)


@pytest.mark.parametrize("change", ["body", "metadata", "delete", "rename"])
def test_cursor_rejects_reference_drift_even_with_the_same_timestamp(
    tmp_path, monkeypatch, change
):
    owner = tmp_path / "owner.md"
    owner.write_text(
        '<!-- kernel-query: {"id":"a"} -->\n## A\nBody A\n\n<!-- kernel-query: {"id":"b"} -->\n## B\nBody B\n'
    )
    monkeypatch.setattr(query, "REFERENCES_ROOT", tmp_path)
    cursor = query.report(kinds=["contract"], limit=1)["knowledge"]["page"][
        "next_cursor"
    ]
    stamp = owner.stat()
    if change in {"body", "metadata"}:
        owner.write_text(
            owner.read_text().replace("Body A", "Body Z")
            if change == "body"
            else owner.read_text().replace('"id":"b"', '"id":"z"')
        )
        os.utime(owner, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    elif change == "delete":
        owner.unlink()
    else:
        owner.rename(tmp_path / "renamed.md")
    with pytest.raises(ValueError, match="cursor"):
        query.report(kinds=["contract"], cursor=cursor)


def test_provider_unavailable_is_distinct_from_a_retrieval_miss(tmp_path):
    result = query.report("pl.system.bar_all", pypto_root=tmp_path, kinds=["api"])
    assert result["knowledge"]["lookup"]["status"] == "source_unavailable"
    assert not result["knowledge"]["coverage"]["api"]["complete"]
    assert result["knowledge"]["coverage"]["api"]["errors"]
    assert result["knowledge"]["lookup"]["knowledge_absence"] == "not_established"
    # A source provider failure does not hide a separately maintained contract.
    assert (
        query.report("ub-l1", pypto_root=tmp_path)["knowledge"]["lookup"]["status"]
        == "matched"
    )


def test_contract_cursor_keeps_root_and_interpreter_identity_when_api_sources_are_unavailable(
    tmp_path, monkeypatch
):
    page = query.report(kinds=["contract"], limit=1, pypto_root=tmp_path)["knowledge"]
    cursor = page["page"]["next_cursor"]
    with pytest.raises(ValueError, match="cursor"):
        query.report(kinds=["contract"], cursor=cursor, pypto_root=tmp_path / "other")
    monkeypatch.setattr(
        feedback.platform, "python_version", lambda: "different-interpreter"
    )
    with pytest.raises(ValueError, match="cursor"):
        query.report(kinds=["contract"], cursor=cursor, pypto_root=tmp_path)


def test_missing_declared_example_is_not_an_api_absence_claim(tmp_path):
    view = query.report("ub-l1", examples_root=tmp_path)["knowledge"]
    assert view["lookup"]["status"] == "matched"
    assert view["missing_items"][0]["reason"] == "declared_knowledge_missing"
    assert view["missing_items"][0]["relation"]["availability"] == "missing"


def test_unknown_filters_are_not_silently_made_into_support():
    view = query.report(
        "ub-l1", kinds=["contract"], filters={"dtype": "pl.DT_FP16", "layout": "NZ"}
    )["knowledge"]
    item = view["items"][0]
    assert item["applicability"]["status"] == "unknown"
    assert set(item["applicability"]["unknown_filters"]) == {"dtype", "layout"}
    assert not item["applicability"]["matched_filters"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"text": ""},
        {"text": 12},
        {"text": "x" * 4097},
        {"kinds": []},
        {"kinds": ["function"]},
        {"kinds": "api"},
        {"text": "x", "filters": {"dtype": True}},
        {"text": "x", "filters": {"dtype": ""}},
        {"text": "x", "filters": {"source": "UB"}},
        {"text": "x", "filters": []},
        {"text": "x", "limit": True},
        {"text": "x", "limit": 0},
        {"text": "x", "limit": 51},
        {"text": "x", "cursor": ""},
        {"cursor": "previous"},
    ],
)
def test_query_contract_rejects_invalid_requests(kwargs):
    with pytest.raises(ValueError):
        query.report(**kwargs)


def test_reference_snapshot_cannot_stamp_old_body_with_a_new_hash(
    tmp_path, monkeypatch
):
    owner = tmp_path / "owner.md"
    owner.write_text('<!-- kernel-query: {"id":"a"} -->\n## A\nBody A\n')
    monkeypatch.setattr(query, "REFERENCES_ROOT", tmp_path)
    original = knowledge.discover

    def changed(*args, **kwargs):
        owner.write_text(owner.read_text().replace("Body A", "Body B"))
        return original(*args, **kwargs)

    monkeypatch.setattr(knowledge, "discover", changed)
    with pytest.raises(ValueError, match="changed during discovery"):
        query.report("a")


def test_object_ids_are_unique_and_facets_validate_at_the_owner(tmp_path):
    entries = list(query.sections())[:1]
    with pytest.raises(ValueError, match="Duplicate"):
        knowledge.objects(entries * 2, [], tmp_path)
    metadata = copy.deepcopy(entries[0][0])
    metadata["facets"] = {"not_a_filter": ["value"]}
    with pytest.raises(ValueError, match="facets"):
        knowledge.objects([(metadata, entries[0][1], entries[0][2])], [], tmp_path)


@pytest.mark.parametrize(
    "args,code,expected",
    [
        (["pl.DT_FP16"], 0, "[constant;"),
        (
            ["--kind", "contract", "--limit", "1", "--format", "json"],
            0,
            '"next_cursor"',
        ),
        (
            [
                "--filters",
                '{"source_memory":"UB","destination_memory":"L1"}',
                "--format",
                "json",
            ],
            0,
            '"contract:ub-l1"',
        ),
        (["pl.move", "--filters", "null"], 2, "JSON object"),
        (["pl.move", "--filters", "{"], 2, "invalid"),
        (["--task", "unused.json", "--kind", "api"], 2, "separate operations"),
        (["pl.move", "--limit", "0"], 2, "1..50"),
    ],
)
def test_cli_query_modes(args, code, expected):
    result = subprocess.run(
        [sys.executable, str(SKILL_ROOT / "scripts/check.py"), "query", *args],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == code and expected in result.stdout + result.stderr
