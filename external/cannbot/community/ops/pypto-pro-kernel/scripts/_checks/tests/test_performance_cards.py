# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Optional card sources preserve registry, scope, provenance and candidate boundaries."""

import builtins
import copy
import json
from pathlib import Path

import pytest

from _checks import (
    documents,
    feedback,
    knowledge,
    knowledge_admission,
    performance_cards,
    query,
    suggest,
)
from _checks.paths import EXAMPLES_ROOT, SKILL_ROOT
from _checks.tests.knowledge_fixtures import make_corpus, write

yaml = pytest.importorskip("yaml", reason="Card acceptance uses requirements-cards.txt")


def relation(kind="uses_contract", target="contract:copy"):
    return {
        "kind": kind,
        "target": {"skill": "pypto-pro-kernel", "id": target},
        "scope": "A fabricated overlap for a source-only test.",
        "boundary": "No applicability or execution evidence is inherited.",
    }


def library(root, *, links=None, item_id="cube-01"):
    data = {
        "type": "PyPTO Performance Optimization Card",
        "title": "Fixture optimization",
        "description": "Source-only teaching data",
        "status": "stable",
        "tags": ["fixture"],
        "item_id": item_id,
        "bound_hint": "MAC",
        "applicability": "Only the declared fixture",
        "target_api_gate": "No runtime support is established",
    }
    if links is not None:
        data["knowledge_links"] = {
            "schema": performance_cards.SCHEMA,
            "relations": links,
        }
    path = write(
        root / f"cube/{item_id}-fixture.md",
        "---\n"
        + yaml.safe_dump(data, sort_keys=False)
        + "---\n\n# Fixture optimization\n\n## Mechanism\nOriginal mechanism text.\n",
    )
    write(
        root / "index.md",
        "# Cards\n\n## Active items\n\n| item_id | Card | status |\n|---|---|---|\n|"
        f" `{item_id}` | [Fixture](cube/{item_id}-fixture.md) | `stable` |\n\n## Draft"
        " items\n",
    )
    return path


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    corpus = make_corpus(tmp_path)
    monkeypatch.setattr(query, "REFERENCES_ROOT", corpus.references)
    root = tmp_path / "card library with spaces"
    path = library(root, links=[relation()])
    return corpus, root, path


def collect(bundle):
    corpus, root, _ = bundle
    return performance_cards.collect(
        root, references_root=corpus.references, entries=list(query.sections())
    )


def discover(bundle, text, **kwargs):
    corpus, root, _ = bundle
    return query.report(
        text,
        pypto_root=corpus.framework,
        examples_root=corpus.examples,
        cards_root=root,
        **kwargs,
    )


def test_registry_only_and_legacy_cards_without_links(bundle):
    corpus, root, path = bundle
    front = performance_cards.frontmatter(path.read_text())
    del front["knowledge_links"]
    path.write_text("---\n" + yaml.safe_dump(front) + "---\n# Fixture\n")
    write(root / "cube/cube-02-unregistered.md", "broken YAML must not be read")
    records, coverage = collect(bundle)
    assert (
        coverage["complete"] and coverage["active_items"] == coverage["cards_read"] == 1
    )
    assert str(root / "cube/cube-02-unregistered.md") not in coverage["_files"]
    assert not records[0]["relations"]
    assert coverage["linked_cards"] == coverage["relation_count"] == 0
    assert coverage["unlinked_cards"] == 1
    assert records[0]["evidence"]["performance"] == "not_established"
    assert records[0]["card"]["status"] == "stable"
    assert "knowledge_links" not in front


def test_card_identity_original_read_and_pinned_source(bundle):
    _, _, path = bundle
    result = discover(bundle, "cube-01", kinds=["document"])
    item = result["knowledge"]["items"][0]
    assert (
        item["document_role"] == "performance_card"
        and item["card"]["item_id"] == "cube-01"
    )
    assert item["facets"] == {} and item["conditions"]
    resolved = discover(
        bundle,
        item["resolve"]["text"],
        reference_sha256=item["resolve"]["reference_sha256"],
    )
    assert resolved["status"] == "passed"
    assert (
        "".join(row["text"] for row in resolved["knowledge"]["content"]["items"])
        == path.read_text()
    )
    path.write_text(path.read_text() + "\nChanged source bytes.\n")
    stale = discover(
        bundle,
        item["resolve"]["text"],
        reference_sha256=item["resolve"]["reference_sha256"],
    )
    assert (
        stale["status"] == "unverified"
        and stale["knowledge"]["lookup"]["status"] == "source_changed"
    )


def test_exact_card_read_does_not_require_framework_or_runtime(bundle):
    corpus, root, _ = bundle
    value = query.report(
        "document:perf-cards/cube/cube-01-fixture.md#mechanism",
        kinds=["document"],
        pypto_root=corpus.framework / "missing",
        examples_root=corpus.examples,
        cards_root=root,
    )
    assert value["status"] == "passed" and value["knowledge"]["coverage"]["complete"]
    assert value["knowledge"]["content"]["items"][0]["text"].startswith("## Mechanism")
    assert value["validation"]["numerical"] == "not_established"


def test_unconfigured_source_does_not_import_yaml_or_change_base_query(
    bundle, monkeypatch
):
    corpus, root, _ = bundle
    original_import = builtins.__import__

    def without_yaml(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_yaml)
    base = query.report(
        "copy route", pypto_root=corpus.framework, examples_root=corpus.examples
    )
    assert (
        base["status"] == "passed"
        and "cards" not in base["knowledge"]["coverage"]["documents"]
    )
    configured = discover(bundle, "cube-01", kinds=["document"])
    assert configured["status"] == "unverified"
    assert "requirements-cards.txt" in json.dumps(configured)


@pytest.mark.parametrize(
    "change",
    [
        "schema",
        "kind",
        "empty_scope",
        "empty_boundary",
        "candidate_id",
        "duplicate",
        "other_skill",
        "extra_field",
    ],
)
def test_invalid_associations_are_quarantined_with_explicit_coverage(bundle, change):
    _, _, path = bundle
    data = performance_cards.frontmatter(path.read_text())
    link = data["knowledge_links"]["relations"][0]
    if change == "schema":
        data["knowledge_links"]["schema"] = "unknown/9"
    elif change == "kind":
        link["kind"] = "verified"
    elif change == "empty_scope":
        link["scope"] = ""
    elif change == "empty_boundary":
        link["boundary"] = []
    elif change == "candidate_id":
        link["target"]["id"] = "candidate:copy"
    elif change == "duplicate":
        data["knowledge_links"]["relations"].append(copy.deepcopy(link))
    elif change == "other_skill":
        link["target"]["skill"] = "unselected-skill"
    else:
        link["supported"] = True
    path.write_text("---\n" + yaml.safe_dump(data) + "---\n# Fixture\n")
    rows, coverage = collect(bundle)
    assert (
        not rows
        and not coverage["complete"]
        and coverage["errors"][0]["reason"] == "metadata_invalid"
    )
    assert discover(bundle, "cube-01", kinds=["document"])["status"] == "unverified"


@pytest.mark.parametrize(
    "target",
    [
        "contract:absent",
        "document:skill/owner.md#missing",
        "document:skill/../../outside.md",
        "document:skill/%2e%2e/outside.md",
    ],
)
def test_invalid_target_or_anchor_does_not_become_a_valid_relation(bundle, target):
    _, root, _ = bundle
    library(root, links=[relation(target=target)])
    rows, coverage = collect(bundle)
    assert not rows and coverage["error_count"] == 1
    assert coverage["errors"][0]["detail"]


@pytest.mark.parametrize(
    "body",
    [
        "---\nitem_id: cube-01\nitem_id: cube-02\n---\n",
        "---\na: &a [1]\nb: *a\n---\n",
        "---\na: !!python/object/apply:os.system ['false']\n---\n",
    ],
)
def test_yaml_is_data_with_no_duplicate_keys_aliases_or_constructors(body):
    with pytest.raises(ValueError):
        performance_cards.frontmatter(body)


@pytest.mark.parametrize("change", ["index", "card", "target"])
def test_cursor_is_invalidated_by_every_part_of_card_identity(bundle, change):
    corpus, root, path = bundle
    first = discover(bundle, None, kinds=["document"], limit=1)
    cursor = first["knowledge"]["page"]["next_cursor"]
    assert cursor
    changed = (
        root / "index.md"
        if change == "index"
        else path
        if change == "card"
        else corpus.owner
    )
    changed.write_text(changed.read_text() + "\nChanged original bytes.\n")
    with pytest.raises(knowledge.QueryError, match="stale knowledge cursor"):
        discover(bundle, None, kinds=["document"], limit=1, cursor=cursor)


def test_direct_reverse_links_do_not_propagate_facets_or_evidence(bundle):
    _, root, _ = bundle
    library(root, links=[relation("same_mechanism")])
    result = discover(bundle, "contract:copy")
    owner = result["knowledge"]["items"][0]
    (reverse,) = [row for row in owner["relations"] if row["kind"] == "knowledge_card"]
    assert reverse["relation"] == "same_mechanism" and reverse["boundary"]
    card = discover(
        bundle, "cube-01", kinds=["document"], filters={"source_memory": "Vec"}
    )["knowledge"]["items"][0]
    assert card["applicability"]["unknown_filters"] == ["source_memory"]
    assert (
        card["evidence"]["numerical"]
        == card["evidence"]["performance"]
        == "not_established"
    )


def test_admission_uses_the_same_registry_and_target_validation(bundle):
    corpus, root, _ = bundle
    result = knowledge_admission.report(
        pypto_root=corpus.framework, examples_root=corpus.examples, cards_root=root
    )
    assert result["status"] == "passed" and not result["diagnostics"]
    assert result["coverage"]["documents"]["cards"]["cards_read"] == 1
    library(root, links=[relation(target="contract:absent")])
    failed = knowledge_admission.report(
        pypto_root=corpus.framework, examples_root=corpus.examples, cards_root=root
    )
    assert failed["status"] == "failed" and any(
        row["rule"] == "knowledge-metadata-invalid" for row in failed["diagnostics"]
    )


def test_duplicate_registration_and_outside_paths_are_not_silently_adopted(bundle):
    _, root, _ = bundle
    index = root / "index.md"
    text = index.read_text()
    row = next(line for line in text.splitlines() if line.startswith("| `cube-01`"))
    index.write_text(text.replace("\n## Draft items", "\n" + row + "\n## Draft items"))
    records, coverage = collect(bundle)
    assert not records and not coverage["complete"]
    index.write_text(text.replace("cube/cube-01-fixture.md", "../cube-01-outside.md"))
    records, coverage = collect(bundle)
    assert not records and not coverage["complete"]


@pytest.mark.parametrize("wrapper", ["```markdown\n{}\n```", "<!--\n{}\n-->"])
def test_active_section_examples_and_comments_do_not_register_cards(bundle, wrapper):
    _, root, _ = bundle
    original = (root / "index.md").read_text()
    library(root, item_id="cube-02")
    example = "| `cube-02` | [Example](cube/cube-02-fixture.md) | `stable` |"
    (root / "index.md").write_text(
        original.replace(
            "## Draft items", wrapper.format(example) + "\n\n## Draft items"
        )
    )
    records, coverage = collect(bundle)
    assert coverage["complete"] and coverage["active_items"] == 1
    assert [row["card"]["item_id"] for row in records] == ["cube-01"]


def test_indented_markdown_table_is_not_silently_dropped(bundle):
    _, root, _ = bundle
    index = root / "index.md"
    index.write_text(
        "\n".join(
            "  " + line if line.startswith("|") else line
            for line in index.read_text().splitlines()
        )
    )
    records, coverage = collect(bundle)
    assert coverage["complete"] and len(records) == coverage["active_items"] == 1


def test_card_document_id_normalizes_relative_path_before_resolving(bundle):
    _, root, _ = bundle
    index = root / "index.md"
    index.write_text(
        index.read_text().replace(
            "cube/cube-01-fixture.md", "./cube//cube-01-fixture.md"
        )
    )
    result = discover(bundle, "cube-01", kinds=["document"])
    item = result["knowledge"]["items"][0]
    assert item["id"] == "document:perf-cards/cube/cube-01-fixture.md"
    assert discover(bundle, item["resolve"]["text"])["knowledge"]["content"]["items"]


def test_document_relation_targets_are_canonical_and_alias_duplicates_are_quarantined(
    bundle,
):
    _, root, _ = bundle
    alias = "document:skill/./owner.md#copy-contract"
    canonical = "document:skill/owner.md#copy-contract"
    library(root, links=[relation(target=alias)])
    records, coverage = collect(bundle)
    assert coverage["complete"] and records[0]["relations"][0]["target"] == canonical
    library(root, links=[relation(target=alias), relation(target=canonical)])
    records, coverage = collect(bundle)
    assert not records and not coverage["complete"]


def test_parent_document_coverage_counts_card_errors(bundle):
    _, _, path = bundle
    path.write_text("---\ninvalid: yaml\n---\n# Damaged card\n")
    result = discover(bundle, "copy route", kinds=["document"])
    coverage = result["knowledge"]["coverage"]["documents"]
    assert not coverage["complete"] and coverage["error_count"] == 1
    assert (
        len(coverage["errors"]) + coverage["omitted_errors"] == coverage["error_count"]
    )


def test_explicit_missing_library_is_unverified_not_an_empty_success(bundle):
    corpus, root, _ = bundle
    result = query.report(
        "copy route",
        pypto_root=corpus.framework,
        examples_root=corpus.examples,
        cards_root=root / "missing",
    )
    assert result["status"] == "unverified" and result["knowledge"]["items"]
    assert (
        result["knowledge"]["coverage"]["documents"]["cards"]["errors"][0]["reason"]
        == "source_unavailable"
    )


def test_looping_card_symlink_is_reported_without_reading_or_crashing(bundle):
    _, _, path = bundle
    path.unlink()
    path.symlink_to(path.name)
    rows, coverage = collect(bundle)
    assert not rows and not coverage["complete"]
    assert coverage["errors"][0]["reason"] == "source_unavailable"
    assert discover(bundle, "cube-01", kinds=["document"])["status"] == "unverified"


def test_same_mechanism_attaches_once_without_new_detectors_or_acceptance(tmp_path):
    root = tmp_path / "cards"
    library(
        root,
        item_id="cube-06",
        links=[relation("same_mechanism", "contract:grouped-ownership")],
    )
    path = write(
        tmp_path / "candidate.py",
        "import pypto_pro.language as pl\n@pl.jit\ndef kernel(groups, count):\n    for"
        " group in pl.range(groups):\n        for row in pl.range(pl.get_block_idx(),"
        " count, pl.get_block_num()):\n            consume(row)\n",
    )
    result = suggest.report([path], cards_root=root)
    assert len(result["candidates"]) == result["coverage"]["matched_sites"] == 1
    candidate = result["candidates"][0]
    (attached,) = candidate["related_cards"]
    assert (
        candidate["pattern"] == "nested-worker-restart"
        and attached["grouping"] == "same_mechanism_only"
    )
    assert (
        attached["candidate_equivalence"]
        == attached["applicability"]
        == "not_established"
    )
    assert candidate["prediction"]["latency_delta_us"] is None
    clean = write(tmp_path / "clean.py", "value = 1\n")
    assert not suggest.report([clean], cards_root=root)["candidates"]


def test_contract_use_does_not_deduplicate_different_transport_interventions(tmp_path):
    root = tmp_path / "cards"
    library(
        root,
        item_id="pipe-07",
        links=[relation("uses_contract", "contract:ub-l1")],
    )
    result = suggest.report(
        [EXAMPLES_ROOT / "basic_api/datamove/ub_l1_handover/kernel.py"], cards_root=root
    )
    (candidate,) = result["candidates"]
    (attached,) = candidate["related_cards"]
    assert candidate["pattern"] == "ub-l1-extra-layout-move"
    assert (
        attached["grouping"] == "reference_only"
        and attached["candidate_equivalence"] == "not_established"
    )
    assert result["coverage"]["cards"]["complete"]


def test_bundled_cards_only_enter_through_the_active_table(bundle):
    corpus, _, _ = bundle
    root = corpus.references / "knowledge-cards"
    library(root, links=[relation()])
    write(root / "draft.md", "# Quarantined example\nNot an Active card.\n")
    kwargs = {"pypto_root": corpus.framework, "examples_root": corpus.examples}
    ordinary = query.report("Quarantined example", kinds=["document"], **kwargs)
    assert all("knowledge-cards/" not in item["id"] for item in ordinary["knowledge"]["items"])
    admitted = query.report("cube-01", kinds=["document"], cards_root=root, **kwargs)
    assert any(item.get("card", {}).get("item_id") == "cube-01" for item in admitted["knowledge"]["items"])
    with pytest.raises(ValueError, match="Active cards"):
        documents.locator("document:skill/knowledge-cards/draft.md", {"skill": corpus.references})


REAL_CARDS = SKILL_ROOT / "references/knowledge-cards"


@pytest.mark.parametrize(
    "chain",
    [
        (
            "pipe-07",
            "contract:ub-l1",
            "example:basic_api/datamove/ub_l1_handover/kernel.py",
        ),
        (
            "mem-06",
            "document:skill/api/memory-allocation.md#aliasing-and-mutual-exclusion",
            "contract:manual-mutex-phase-reuse",
            "example:algorithms/attn_epilogue/kernel.py",
        ),
    ],
)
def test_card_contract_example_chain_retains_source_roles_and_evidence_boundaries(
    chain,
):
    result = query.report(chain[0], kinds=["document"], cards_root=REAL_CARDS, limit=1)
    (current,) = result["knowledge"]["items"]
    assert current["card"]["item_id"] == chain[0]
    for target in chain[1:]:
        assert target in {row["target"] for row in current["relations"]}
        assert current["evidence"]["performance"] == "not_established"
        result = query.report(target, cards_root=REAL_CARDS, limit=1)
        (current,) = result["knowledge"]["items"]
    assert current["id"] == chain[-1]
    assert set(current["program"]["entries"]) == {"kernel", "reference", "test"}
    assert (
        current["conditions"] and current["validation"]["status"] == "not_established"
    )
    for entry in current["program"]["entries"].values():
        assert entry["sha256"] == feedback.file_identity(Path(entry["path"]))["sha256"]


def test_owner_library_resolves_all_current_cards_and_preserves_relation_boundaries():
    rows, coverage = performance_cards.collect(
        REAL_CARDS,
        references_root=query.REFERENCES_ROOT,
        entries=list(query.sections()),
    )
    assert coverage["complete"] and coverage["active_items"] == coverage[
        "cards_read"
    ] == len(rows)
    cards = {row["card"]["item_id"]: row for row in rows}
    expected = {
        f"{category}-{number:02d}"
        for category, count in {
            "vec": 19,
            "cube": 13,
            "mem": 6,
            "pipe": 7,
            "scalar": 3,
            "sync": 1,
            "algo": 1,
        }.items()
        for number in range(1, count + 1)
    }
    assert expected <= cards.keys()
    for identity in expected:
        card = cards[identity]
        assert card["relations"], identity
        for card_link in card["relations"]:
            assert card_link["scope"] and card_link["boundary"]
            assert card_link["reference"]["sha256"]
            assert card_link["reference"]["line"] > 0
        assert card["evidence"]["performance"] == "not_established"
    assert coverage["linked_cards"] + coverage["unlinked_cards"] == len(cards)
    assert coverage["relation_count"] == sum(len(row["relations"]) for row in rows)
    assert "cross-engine-01" not in cards
    assert cards["cube-06"]["relations"][0]["kind"] == "same_mechanism"
    assert cards["vec-09"]["relations"][0]["kind"] == "overlaps_with"
    assert {row["kind"] for row in cards["pipe-07"]["relations"]} == {
        "overlaps_with",
        "uses_contract",
    }
    assert cards["pipe-05"]["relations"][0]["kind"] == "alternative_to"
