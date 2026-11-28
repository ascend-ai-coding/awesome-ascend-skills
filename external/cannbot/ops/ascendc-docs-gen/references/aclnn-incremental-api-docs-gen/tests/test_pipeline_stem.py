# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
TDD tests for pipeline.py B1/B2/B3 — md_file/md_stem priority over glob[0].

Verifies that AnchorPipeline uses path_config.md_file / path_config.md_stem
when set, falling back to sorted(glob)[0] only when None (backward compat).

B1: _build_doc_structure doc_name uses md_file (line 517-526)
B2: _get_sub_function_count uses md_stem (line 586-590)
B3: _build_support_summary uses md_file (line 759-761)
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.snapshot_construction.anchor_construction.common.path_config import PathConfig
from src.snapshot_construction.anchor_construction.gene_rule_registry import (
    get_registry,
)
from src.snapshot_construction.anchor_construction.pipeline import AnchorPipeline


# =============================================================================
# Helpers
# =============================================================================


def _make_operator_path(tmp_path: Path) -> Path:
    """Create operator_path with aclnnFakeOpV2.md and aclnnFakeOpV3.md.

    Sorted glob returns V2 before V3 (alphabetical), so glob[0] == V2.
    """
    op_path = tmp_path / "operator"
    op_path.mkdir()
    (op_path / "aclnnFakeOpV2.md").write_text("# aclnnFakeOpV2")
    (op_path / "aclnnFakeOpV3.md").write_text("# aclnnFakeOpV3")
    return op_path


def _make_pipeline(tmp_path: Path, md_file=None) -> AnchorPipeline:
    """Construct AnchorPipeline with tmp_path-based PathConfig.

    operator_path contains both V2.md and V3.md (glob[0] == V2).
    """
    op_path = _make_operator_path(tmp_path)
    out_path = tmp_path / "out"
    out_path.mkdir()
    pc = PathConfig(op_path, out_path, md_file=md_file)
    return AnchorPipeline(pc)


# =============================================================================
# B1: _build_doc_structure doc_name uses md_file
# =============================================================================


class TestB1DocNameUsesMdFile:
    """B1: _build_doc_structure should use path_config.md_file for doc_name."""

    def test_b1_doc_name_uses_md_file(self, tmp_path):
        """md_file='aclnnFakeOpV3.md' -> doc_name == 'aclnnFakeOpV3.md' (NOT V2).

        Without fix: glob[0] == aclnnFakeOpV2.md -> doc_name == 'aclnnFakeOpV2.md' (FAIL).
        With fix: md_file set -> doc_name == 'aclnnFakeOpV3.md' (PASS).
        """
        pipeline = _make_pipeline(tmp_path, md_file="aclnnFakeOpV3.md")
        config = get_registry().get_config("constraint")

        doc = pipeline._build_doc_structure("constraint", config, [])

        assert doc["doc_name"] == "aclnnFakeOpV3.md", (
            f"Expected aclnnFakeOpV3.md (from md_file), got {doc['doc_name']}"
        )

    def test_b1_fallback_when_md_file_none(self, tmp_path):
        """md_file=None -> doc_name == 'aclnnFakeOpV2.md' (alphabetically first, backward compat).

        Both before and after fix: glob[0] == aclnnFakeOpV2.md -> doc_name == 'aclnnFakeOpV2.md'.
        This test ensures the fix doesn't break existing behavior when md_file is None.
        """
        pipeline = _make_pipeline(tmp_path, md_file=None)
        config = get_registry().get_config("constraint")

        doc = pipeline._build_doc_structure("constraint", config, [])

        assert doc["doc_name"] == "aclnnFakeOpV2.md", (
            f"Expected aclnnFakeOpV2.md (glob fallback), got {doc['doc_name']}"
        )


# =============================================================================
# B2: _get_sub_function_count uses md_stem
# =============================================================================


class TestB2SubFunctionCountUsesMdStem:
    """B2: _get_sub_function_count should use path_config.md_stem for doc_json."""

    def test_b2_sub_function_count_uses_md_stem(self, tmp_path):
        """md_stem='aclnnFakeOpV3' -> reads aclnnFakeOpV3.json (2 sub_funcs), NOT V2 (1).

        Without fix: glob[0].stem == aclnnFakeOpV2 -> reads V2.json (1 key) -> returns 1 (FAIL).
        With fix: md_stem == aclnnFakeOpV3 -> reads V3.json (2 keys) -> returns 2 (PASS).
        """
        pipeline = _make_pipeline(tmp_path, md_file="aclnnFakeOpV3.md")

        # Create doc_json with V2 (1 sub_function_hash key) and V3 (2 keys)
        doc_json_dir = pipeline.path_config.doc_json
        doc_json_dir.mkdir(parents=True, exist_ok=True)
        (doc_json_dir / "aclnnFakeOpV2.json").write_text(
            json.dumps({"sub_function_hash_1": {"data": "v2"}})
        )
        (doc_json_dir / "aclnnFakeOpV3.json").write_text(
            json.dumps(
                {
                    "sub_function_hash_1": {"data": "v3a"},
                    "sub_function_hash_2": {"data": "v3b"},
                }
            )
        )

        count = pipeline._get_sub_function_count()

        assert count == 2, f"Expected 2 (from V3 json via md_stem), got {count}"


# =============================================================================
# B3: _build_support_summary uses md_file
# =============================================================================


class TestB3SupportSummaryUsesMdFile:
    """B3: _build_support_summary should use path_config.md_file for support_map key."""

    def test_b3_support_summary_uses_md_file(self, tmp_path):
        """md_file='aclnnFakeOpV3.md' -> support_map key 'aclnnFakeOpV3.md' used.

        operator_path has ONLY aclnnFakeOpV2.md (no V3.md on disk), so glob[0]
        is deterministically V2. Without fix: support_map[V2] == "" -> len 0 (FAIL).
        With fix: md_file set -> md_name == aclnnFakeOpV3.md -> support_map[V3] has content (PASS).

        Note: B3 site uses list(glob) (not sorted), so we ensure only one .md file
        exists to make glob[0] deterministic.
        """
        # Custom operator_path with ONLY V2.md (not the helper's V2+V3)
        op_path = tmp_path / "operator"
        op_path.mkdir()
        (op_path / "aclnnFakeOpV2.md").write_text("# aclnnFakeOpV2")
        out_path = tmp_path / "out"
        out_path.mkdir()
        pc = PathConfig(op_path, out_path, md_file="aclnnFakeOpV3.md")
        pipeline = AnchorPipeline(pc)

        # Monkeypatch load_support_matrix to return a controlled support_map
        v3_content = "V3 support content here"
        fake_support_map = {
            "aclnnFakeOpV3.md": v3_content,
            "aclnnFakeOpV2.md": "",
        }
        with patch.object(
            pipeline.data_loader, "load_support_matrix", return_value=fake_support_map
        ):
            result = pipeline._build_support_summary({"section": [], "file": []})

        expected_len = len(v3_content)
        assert result["support_matrix_content_len"] == expected_len, (
            f"Expected {expected_len} (V3 content via md_file), "
            f"got {result['support_matrix_content_len']}"
        )
