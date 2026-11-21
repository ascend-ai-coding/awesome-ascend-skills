# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Validate golden reference file structure for anchor_construction TDD baseline."""

import json
from pathlib import Path

import pytest

GOLDEN_REF_DIR = Path(__file__).parent / "golden_ref"

DOC_PART_FILES = {
    "constraint": "constraint.json",
    "support": "support.json",
    "summary": "summary.json",
    "function_proto": "function_proto.json",
    "sub_function_hash": "sub_function_hash.json",
}


def test_all_golden_files_exist():
    """All 5 golden reference JSON files must exist."""
    for doc_part, filename in DOC_PART_FILES.items():
        filepath = GOLDEN_REF_DIR / filename
        assert filepath.exists(), f"Missing golden file: {filename}"


def test_golden_has_doc_and_anchor_tuple(golden_data):
    """Each golden file must have 'doc' and 'anchor_tuple' top-level keys."""
    for doc_part, data in golden_data.items():
        assert "doc" in data, f"{doc_part}: missing 'doc' key"
        assert "anchor_tuple" in data, f"{doc_part}: missing 'anchor_tuple' key"
        assert isinstance(data["anchor_tuple"], list), (
            f"{doc_part}: anchor_tuple must be a list"
        )


def test_constraint_anchor_types(golden_data):
    """constraint.json must have only file anchors."""
    types = _get_anchor_types(golden_data["constraint"])
    assert "file" in types, "constraint: must have file anchors"
    assert types == {"file"}, f"constraint: expected only file anchors, got {types}"


def test_support_anchor_types(golden_data):
    """support.json must have file and section anchors."""
    types = _get_anchor_types(golden_data["support"])
    assert "file" in types, "support: must have file anchors"
    assert "section" in types, "support: must have section anchors"
    assert types == {"file", "section"}, (
        f"support: expected file+section anchors, got {types}"
    )


def test_summary_anchor_types(golden_data):
    """summary.json must have file and note anchors."""
    types = _get_anchor_types(golden_data["summary"])
    assert "file" in types, "summary: must have file anchors"
    assert "note" in types, "summary: must have note anchors"
    assert types == {"file", "note"}, (
        f"summary: expected file+note anchors, got {types}"
    )


def test_function_proto_anchor_types(golden_data):
    """function_proto.json must have file and function anchors."""
    types = _get_anchor_types(golden_data["function_proto"])
    assert "file" in types, "function_proto: must have file anchors"
    assert "function" in types, "function_proto: must have function anchors"
    assert types == {"file", "function"}, (
        f"function_proto: expected file+function anchors, got {types}"
    )


def test_sub_function_anchor_types(golden_data):
    """sub_function_hash.json must have ALL 4 anchor types."""
    types = _get_anchor_types(golden_data["sub_function_hash"])
    expected = {"file", "function", "note", "section"}
    assert types == expected, f"sub_function_hash: expected all 4 types, got {types}"


def test_anchor_information_structure(golden_data):
    """Each anchor must have anchor_information with type, level1, level2. level3 is optional."""
    for doc_part, data in golden_data.items():
        for anchor in data["anchor_tuple"]:
            assert "anchor_information" in anchor, (
                f"{doc_part}: anchor missing anchor_information"
            )
            info = anchor["anchor_information"]
            assert "type" in info, f"{doc_part}: anchor_information missing type"
            assert "level1" in info, f"{doc_part}: anchor_information missing level1"
            assert "level2" in info, f"{doc_part}: anchor_information missing level2"
            # level3 is optional — some anchors (e.g. support section) don't have it


def test_evidence_structure(golden_data):
    """Each anchor must have evidence list with evidence_id."""
    for doc_part, data in golden_data.items():
        for anchor in data["anchor_tuple"]:
            assert "evidence" in anchor, f"{doc_part}: anchor missing evidence"
            assert isinstance(anchor["evidence"], list), (
                f"{doc_part}: evidence must be a list"
            )
            for ev in anchor["evidence"]:
                assert "evidence_id" in ev, (
                    f"{doc_part}: evidence item missing evidence_id"
                )


def test_doc_part_field(golden_data):
    """Each golden file's doc must have doc_part matching its filename."""
    expected_parts = {
        "constraint": "constraint",
        "support": "support",
        "summary": "summary",
        "function_proto": "function_proto",
        "sub_function_hash": "sub_function_hash",
    }
    for doc_part, data in golden_data.items():
        doc = data["doc"]
        assert "doc_part" in doc, f"{doc_part}: doc missing doc_part"
        assert doc["doc_part"] == expected_parts[doc_part], (
            f"{doc_part}: doc_part={doc['doc_part']}, expected={expected_parts[doc_part]}"
        )


def _get_anchor_types(data: dict) -> set[str]:
    """Extract set of anchor types from anchor_tuple."""
    return {a["anchor_information"]["type"] for a in data["anchor_tuple"]}
