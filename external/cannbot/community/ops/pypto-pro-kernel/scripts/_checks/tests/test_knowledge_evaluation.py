# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent retrieval grading, source continuation and honest incomplete metrics."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from _checks import knowledge, query
from _checks import knowledge_evaluation as evaluator
from _checks.paths import SKILL_ROOT


def case(**changes):
    return {
        "id": "sample",
        "split": "development",
        "category": "source",
        "query": {"text": "pl.move", "kinds": ["api"]},
        "relevant_ids": ["symbol:pl.move"],
        **changes,
    }


def report(ids, status="matched", **changes):
    return {
        "identity": {"checker_sha256": "a"},
        "knowledge": {
            "lookup": {"status": status, "coverage": "complete"},
            "items": [
                {"id": value, "kind": "api", "applicability": {"unknown_filters": []}}
                for value in ids
            ],
        },
        **changes,
    }


def write_cases(path, cases):
    path.write_text(json.dumps({"schema": evaluator.CASE_SCHEMA, "cases": cases}))
    return path


def test_top5_relevance_is_not_arbitrary_success_or_full_framework_absence():
    ranked = report(["symbol:pl.other", "symbol:pl.move"])
    row = evaluator.grade(case(), ranked, {"exit_code": 0})
    assert row["status"] == "passed" and row["first_relevant_rank"] == 2
    assert row["reciprocal_rank_at_5"] == 0.5
    missed = evaluator.grade(
        case(),
        report([f"symbol:pl.other{index}" for index in range(5)] + ["symbol:pl.move"]),
        {"exit_code": 0},
    )
    assert missed["status"] == "failed" and not missed["hit_at_5"]
    unknown = evaluator.grade(
        case(relevant_ids=[], expected_lookup="source_unavailable"),
        report([], "source_unavailable"),
        {"exit_code": 2},
    )
    assert unknown["status"] == "passed" and unknown["hit_at_5"] is None


def test_conditions_and_source_requirements_are_graded_separately():
    item = case(
        required_ids=["docstring:pl.move"],
        forbidden_ids=["symbol:pl.move"],
        unknown_filters=["layout"],
        no_recipe=True,
    )
    row = evaluator.grade(
        item,
        report(["symbol:pl.move"], context=[{"text": "unconditional recipe"}]),
        {"exit_code": 0},
    )
    assert set(row["failures"]) == {
        "forbidden_condition_route",
        "required_source_missing",
        "unknown_conditions_lost",
        "weak_discovery_promoted_to_recipe",
    }


def test_incomplete_corpus_cannot_be_counted_as_a_complete_retrieval_score():
    value = report(["symbol:pl.move"])
    value["knowledge"]["coverage"] = {"complete": False}
    row = evaluator.grade(case(), value, {"exit_code": 0})
    assert row["status"] == "unverified" and row["hit_at_5"] is None
    assert (
        row["reciprocal_rank_at_5"] is None
        and "source_coverage_incomplete" in row["failures"]
    )


@pytest.mark.parametrize(
    "change",
    [
        {"relevant_ids": []},
        {"split": "train"},
        {"resolve": "yes"},
        {"query": {"source": "outside.py"}},
    ],
)
def test_invalid_grading_contracts_are_not_silently_accepted(tmp_path, change):
    with pytest.raises(ValueError):
        evaluator.load(write_cases(tmp_path / "cases.json", [case(**change)]))


def test_duplicate_questions_cannot_inflate_denominators(tmp_path):
    with pytest.raises(ValueError, match="Duplicate"):
        evaluator.load(write_cases(tmp_path / "cases.json", [case(), case()]))


def test_missing_output_keeps_aggregate_retrieval_unknown(tmp_path, monkeypatch):
    path = write_cases(
        tmp_path / "cases.json", [case(), case(id="missing", resolve=True)]
    )
    calls = 0

    def invoke(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("missing JSON")
        return report(["symbol:pl.move"]), {"exit_code": 0, "wall_seconds": 1}

    monkeypatch.setattr(evaluator, "invoke", invoke)
    result = evaluator.run(path)
    assert result["status"] == "unverified" and result["metrics"]["positive_cases"] == 2
    assert result["metrics"]["scored_positive_cases"] == 1
    assert (
        result["metrics"]["hit_at_5"] is None and result["metrics"]["mrr_at_5"] is None
    )
    assert result["metrics"]["hit_at_5_on_scored"] == 1
    assert result["metrics"]["source_read_completion"] == 0


def test_mixed_checker_versions_refuse_a_comparison(tmp_path, monkeypatch):
    path = write_cases(tmp_path / "cases.json", [case(), case(id="second")])
    calls = 0

    def invoke(*args):
        nonlocal calls
        calls += 1
        result = report(["symbol:pl.move"])
        result["identity"]["checker_sha256"] = str(calls)
        return result, {"exit_code": 0, "wall_seconds": 1}

    monkeypatch.setattr(evaluator, "invoke", invoke)
    with pytest.raises(ValueError, match="Checker changed"):
        evaluator.run(path)


def test_fragment_continuation_preserves_source_hash_and_section(tmp_path, monkeypatch):
    path = str(tmp_path / "topic.md")
    content = {
        "status": "partial",
        "items": [{"text": "first"}],
        "coverage": {"complete": False},
        "next": {
            "path": path,
            "offset_bytes": 5,
            "source_sha256": "a" * 64,
            "fragment": "section",
            "request_index": 0,
        },
    }
    observed = []

    def invoke(command, argv):
        observed.append(json.loads(argv[2]))
        return {
            "status": "complete",
            "coverage": {"complete": True},
            "items": [{"text": "second"}],
        }, {"exit_code": 0}

    monkeypatch.setattr(evaluator, "invoke", invoke)
    result = evaluator.drain(content, ["checker"])
    assert result == {"status": "complete", "pages": 2, "text_bytes": 11}
    assert observed == [
        {
            "path": path,
            "offset_bytes": 5,
            "source_sha256": "a" * 64,
            "fragment": "section",
        }
    ]


def test_cli_grades_real_sources_and_keeps_expected_negative_status(tmp_path):
    path = write_cases(
        tmp_path / "cases.json",
        [
            case(
                query={"text": "pl.system.bar_all", "kinds": ["api"]},
                relevant_ids=["symbol:pl.system.bar_all"],
            ),
            case(
                id="unavailable",
                query={"text": "document:pro-guide/absent.md"},
                relevant_ids=[],
                expected_lookup="source_unavailable",
            ),
        ],
    )
    child = subprocess.run(
        [
            sys.executable,
            str(SKILL_ROOT / "scripts/check.py"),
            "evaluate",
            "--knowledge-cases",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    result = json.loads(child.stdout)
    assert child.returncode == 0 and result["status"] == "complete"
    assert result["metrics"]["cases"] == 2 and result["metrics"]["hit_at_5"] == 1


def test_benchmark_declares_independent_splits_and_resolvable_object_owners():
    cases, _ = evaluator.load(Path(__file__).parent / "data/knowledge_benchmark.json")
    assert sum(row["split"] == "development" for row in cases) == 90
    assert sum(row["split"] == "heldout" for row in cases) == 20
    sources = query.collect(knowledge.KINDS)
    records, _, _, issues = knowledge.assemble(
        sources["entries"],
        sources["declarations"],
        query.EXAMPLES_ROOT,
        sources["document_records"],
        sources["example_records"],
    )
    assert not issues
    known = {row["id"]: row for row in records}
    for row in cases:
        assert (
            set(
                row.get("relevant_ids", [])
                + row.get("required_ids", [])
                + row.get("forbidden_ids", [])
            )
            <= known.keys()
        )
        if row["query"].get("kinds"):
            assert all(
                known[identity]["kind"] in row["query"]["kinds"]
                for identity in row["relevant_ids"]
            )
        if row.get("resolve"):
            assert all(
                known[identity]["kind"] in {"document", "example"}
                for identity in row["relevant_ids"]
            )
