# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Knowledge admission, located source failures and refresh through shared query semantics."""

import copy
import json
import os

import pytest

from _checks import api_stub, knowledge, knowledge_admission, query
from _checks.tests.knowledge_fixtures import make_corpus, write


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    value = make_corpus(tmp_path)
    monkeypatch.setattr(query, "REFERENCES_ROOT", value.references)
    return value


def admission(corpus):
    return knowledge_admission.report(
        pypto_root=corpus.framework, examples_root=corpus.examples
    )


def discover(corpus, text=None, **kwargs):
    return query.report(
        text, pypto_root=corpus.framework, examples_root=corpus.examples, **kwargs
    )


def test_complete_scope_passes_existing_gates_without_importing_sources(corpus):
    result = admission(corpus)
    assert result["status"] == "passed"
    assert set(result["validation"].values()) <= {"not_run", "not_established"}
    assert {row["name"] for row in result["admission"]["maintenance"]} == {
        "links",
        "catalog",
        "coverage",
        "api_freshness",
    }
    assert all(row["status"] == "passed" for row in result["admission"]["maintenance"])
    assert (
        result["admission"]["queryability"]["objects_checked"]
        == result["admission"]["objects"]
    )
    assert result["admission"]["queryability"]["declared_routes_checked"] > 0
    assert not result["diagnostics"]


def test_declared_route_overlap_preserves_conditional_alternatives(corpus):
    write(
        corpus.references / "other.md",
        '<!-- kernel-query: {"id":"other","aliases":["copy route"],"facets":{"source_memory":["Mat"]}} -->\n## Other\nBody\n',
    )
    result = admission(corpus)
    assert result["status"] == "passed"
    (overlap,) = result["admission"]["queryability"]["routing_overlaps"]
    assert overlap["owners"] == ["contract:copy", "contract:other"]
    assert overlap["status"] == "conditional_alternatives"
    assert result["admission"]["source_reconciliation"] == "not_assessed"
    filtered = discover(
        corpus, "copy route", kinds=["contract"], filters={"source_memory": "UB"}
    )["knowledge"]
    assert [row["id"] for row in filtered["items"]] == ["contract:copy"]


def test_alias_cannot_hide_another_exact_identity(corpus):
    write(
        corpus.references / "other.md",
        '<!-- kernel-query: {"id":"other","aliases":["contract:copy"]} -->\n## Other\nBody\n',
    )
    result = admission(corpus)
    assert result["status"] == "failed"
    assert {row["rule"] for row in result["diagnostics"]} >= {
        "knowledge-shadowed-route",
        "knowledge-unreachable-route",
    }


@pytest.mark.parametrize("kind", ["symbol", "document", "example"])
def test_ambiguous_provider_ids_never_resolve_the_first_source(
    corpus, monkeypatch, kind
):
    original = query.collect
    identity = {
        "symbol": "pl.move",
        "document": "document:pro-api/move.md",
        "example": "example:algorithms/composition/kernel.py",
    }[kind]

    def duplicate(*args, **kwargs):
        sources = original(*args, **kwargs)
        field = {
            "symbol": "declarations",
            "document": "document_records",
            "example": "example_records",
        }[kind]
        selected = next(
            row
            for row in sources[field]
            if row["symbol" if kind == "symbol" else "id"] == identity
        )
        sources[field].append(copy.deepcopy(selected))
        return sources

    monkeypatch.setattr(query, "collect", duplicate)
    result = discover(
        corpus, identity, **({"kinds": ["api"]} if kind == "symbol" else {})
    )
    assert (
        not result["knowledge"]["items"]
        and result["knowledge"]["lookup"]["status"] == "source_unavailable"
    )
    assert not result["signatures"] and "content" not in result["knowledge"]
    gate = admission(corpus)
    assert gate["status"] == "failed" and any(
        row["rule"] == "knowledge-duplicate-id" for row in gate["diagnostics"]
    )


def test_missing_companion_points_to_its_owner_and_blocks_admission(corpus):
    corpus.owner.write_text(
        corpus.owner.read_text().replace(
            "chapter/copy/kernel.py", "chapter/missing/kernel.py"
        )
    )
    result = admission(corpus)
    assert result["status"] == "failed"
    failure = next(
        row
        for row in result["diagnostics"]
        if row["rule"] == "knowledge-missing-companion"
    )
    assert failure["location"]["file"] == str(corpus.owner)
    assert (
        failure["evidence"]["target"] == "example:basic_api/chapter/missing/kernel.py"
    )
    assert (
        discover(corpus, "copy route", kinds=["contract"])["knowledge"]["lookup"][
            "status"
        ]
        == "matched"
    )


def test_missing_program_role_does_not_become_execution_acceptance(corpus):
    (corpus.examples / "algorithms/composition/reference.py").unlink()
    result = admission(corpus)
    assert result["status"] == "failed"
    assert any(
        row["rule"] == "knowledge-missing-program-role" for row in result["diagnostics"]
    )
    item = discover(corpus, "example:algorithms/composition")["knowledge"]["items"][0]
    assert (
        item["program"]["missing_roles"] == ["reference"]
        and item["validation"]["status"] == "not_established"
    )


def test_corrupt_source_leaves_valid_query_results_but_not_a_complete_miss(corpus):
    bad = corpus.references / "damaged.md"
    bad.write_bytes(b"\xff\xfe")
    result = admission(corpus)
    assert (
        result["status"] == "failed"
    )  # The existing links gate rejects malformed Markdown bytes.
    assert not result["coverage"]["complete"]
    known = discover(corpus, "contract:copy", kinds=["contract"])["knowledge"]
    assert (
        known["lookup"]["status"] == "matched"
        and known["lookup"]["coverage"] == "partial"
    )
    absent = discover(corpus, "contract:unknown", kinds=["contract"])["knowledge"]
    assert (
        absent["lookup"]["status"] == "source_unavailable"
        and absent["lookup"]["knowledge_absence"] == "not_established"
    )


def test_program_rename_requires_the_selection_update(corpus):
    folder = corpus.examples / "algorithms/composition"
    folder.rename(folder.with_name("renamed"))
    broken = admission(corpus)
    assert broken["status"] == "failed"
    assert any(row["rule"] == "example-catalog" for row in broken["diagnostics"])
    corpus.catalog.write_text(
        corpus.catalog.read_text().replace(
            "algorithms/composition/", "algorithms/renamed/"
        )
    )
    assert admission(corpus)["status"] == "passed"
    assert discover(corpus, "example:algorithms/renamed")["knowledge"]["items"]
    assert (
        discover(corpus, "example:algorithms/composition")["knowledge"]["lookup"][
            "status"
        ]
        == "source_unavailable"
    )


def test_document_move_keeps_owner_identity_and_rechecks_links(corpus):
    write(
        corpus.references / "consumer.md",
        "# Consumer\n\n[contract](owner.md#copy-contract)\n",
    )
    before = discover(corpus, "contract:copy", kinds=["contract"])["knowledge"][
        "items"
    ][0]
    corpus.owner.rename(corpus.references / "renamed.md")
    assert admission(corpus)["status"] == "failed"
    (corpus.references / "consumer.md").write_text(
        "# Consumer\n\n[contract](renamed.md#copy-contract)\n"
    )
    assert admission(corpus)["status"] == "passed"
    after = discover(corpus, "contract:copy", kinds=["contract"])["knowledge"]["items"][
        0
    ]
    assert (
        before["id"] == after["id"]
        and before["reference"]["path"] != after["reference"]["path"]
    )
    assert (
        discover(corpus, "document:skill/owner.md")["knowledge"]["lookup"]["status"]
        == "source_unavailable"
    )


def test_same_timestamp_source_edit_invalidates_the_generated_reference(corpus):
    stamp = corpus.api_source.stat()
    corpus.api_source.write_text(
        corpus.api_source.read_text().replace("Move source data", "Move edited data")
    )
    os.utime(corpus.api_source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    stale = admission(corpus)
    assert stale["status"] == "failed"
    assert any(
        row["rule"] == "knowledge-api-reference-stale" for row in stale["diagnostics"]
    )
    write(
        corpus.references / "api/pypto_pro_language.pyi",
        api_stub.generate(corpus.framework),
    )
    assert admission(corpus)["status"] == "passed"


def test_unknown_source_usage_is_not_a_claim_of_api_absence(corpus):
    kernel = corpus.examples / "algorithms/composition/kernel.py"
    kernel.write_text(
        kernel.read_text().replace("pl.move(x, x)", "pl.not_in_selected_exports(x)")
    )
    result = admission(corpus)
    assert result["status"] == "passed"
    assert (
        result["admission"]["relations"]["unresolved"][
            "inferred_symbol_outside_selected_exports"
        ]
        == 1
    )
    (unknown,) = result["admission"]["relations"]["unresolved_items"]
    assert (
        unknown["target"] == "symbol:pl.not_in_selected_exports"
        and unknown["references"][0]["line"] == 4
    )


@pytest.mark.parametrize("target", ["kernel", "api_reference"])
def test_mid_check_edits_cannot_be_stamped_with_fresh_identity(
    corpus, monkeypatch, target
):
    if target == "kernel":
        original = knowledge_admission._maintenance

        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            path = corpus.examples / "algorithms/composition/kernel.py"
            path.write_text(path.read_text() + "\n# changed\n")
            return result

        monkeypatch.setattr(knowledge_admission, "_maintenance", changed)
    else:
        original = api_stub.inspect_reference

        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            path = corpus.references / "api/pypto_pro_language.pyi"
            path.write_text(path.read_text() + "\n# changed\n")
            return result

        monkeypatch.setattr(api_stub, "inspect_reference", changed)
    with pytest.raises(knowledge.QueryError) as raised:
        admission(corpus)
    assert raised.value.status == "source_changed"


def test_stale_cursor_has_a_structured_failure_class(corpus):
    page = discover(corpus, kinds=["document"], limit=1)["knowledge"]
    corpus.owner.write_text(corpus.owner.read_text() + "\nChanged source text.\n")
    with pytest.raises(knowledge.QueryError) as raised:
        discover(
            corpus, kinds=["document"], limit=1, cursor=page["page"]["next_cursor"]
        )
    result = query.failure(raised.value, pypto_root=corpus.framework)
    assert result["knowledge"]["lookup"]["status"] == "cursor_invalidated"
    assert result["status"] == "unverified" and not result["knowledge"]["items"]


def test_many_shared_aliases_are_checked_through_all_result_pages(corpus):
    for index in range(55):
        write(
            corpus.references / f"owner-{index}.md",
            "<!-- kernel-query: "
            + json.dumps({"id": f"extra-{index}", "aliases": ["shared alias"]})
            + " -->\n## Owner\nBody\n",
        )
    result = admission(corpus)
    assert result["status"] == "passed"
    shared = next(
        row
        for row in result["admission"]["queryability"]["routing_overlaps"]
        if row["alias"] == "shared alias"
    )
    assert len(shared["owners"]) == 55
