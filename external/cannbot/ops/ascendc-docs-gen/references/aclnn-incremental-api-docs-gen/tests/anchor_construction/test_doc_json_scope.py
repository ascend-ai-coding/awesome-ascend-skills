# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Tests for _build_doc_structure edge cases related to doc_json file scope.

Tests cover:
- Test 1: first_md_file=None (no .md files found) + doc_part="sub_function_hash" → no NameError,
  returns dict with template sub_part
- Test 2: first_md_file exists but doc_json_file doesn't exist → graceful fallback, returns dict with template sub_part
"""

from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest

from src.snapshot_construction.anchor_construction.gene_rule_registry import (
    get_registry,
)
from src.snapshot_construction.anchor_construction.pipeline import AnchorPipeline


class TestBuildDocStructureEdgeCases:
    """Test _build_doc_structure edge cases for doc_json file scope."""

    def test_no_md_files_returns_template_sub_part(self):
        """Test that when no .md files are found, _build_doc_structure returns dict with template sub_part only.

        This tests the fix for NameError when first_md_file=None and doc_part="sub_function_hash".
        The method should gracefully handle missing .md files and return a doc with the template sub_part.
        """
        # Create a mock pipeline with mocked path_config
        pipeline = MagicMock(spec=AnchorPipeline)
        pipeline.path_config = MagicMock()
        pipeline.path_config.md_file = None

        # Mock operator_path.glob to return empty list (no .md files found)
        pipeline.path_config.operator_path.glob.return_value = []

        # Get sub_function_hash config from registry
        registry = get_registry()
        config = registry.get_config("sub_function_hash")

        # Create empty anchor_tuple (required parameter)
        anchor_tuple = []

        # Call _build_doc_structure directly (bind to AnchorPipeline class)
        doc = AnchorPipeline._build_doc_structure(pipeline, "sub_function_hash", config, anchor_tuple)

        # Verify no NameError occurred and doc has expected structure
        assert isinstance(doc, dict)
        assert "doc_path" in doc
        assert "doc_name" in doc
        assert "doc_part" in doc
        assert "md_format" in doc
        assert "sub_part" in doc

        # Verify doc_part is correct
        assert doc["doc_part"] == "sub_function_hash"

        # Verify doc_path and doc_name are empty strings (no .md files found)
        assert doc["doc_path"] == ""
        assert doc["doc_name"] == ""

        # Verify md_format is empty list
        assert doc["md_format"] == []

        # Verify sub_part has template keys from config
        expected_sub_part_keys = {
            "param_matrix",
            "fixconfig_1",
            "fixconfig_2",
            "changeconfig",
            "errorcode_matrix",
        }
        assert set(doc["sub_part"].keys()) == expected_sub_part_keys

        # Verify sub_part values are empty dicts (template structure)
        for key in expected_sub_part_keys:
            assert doc["sub_part"][key] == {}

    def test_md_file_exists_but_doc_json_missing_returns_template_sub_part(self):
        """Test that when first_md_file exists but doc_json_file doesn't exist,
        _build_doc_structure returns dict with template sub_part.

        This tests the graceful fallback when doc_json file is missing.
        The method should return a doc with the template sub_part (no doc_json data loaded).
        """
        # Create a mock pipeline with mocked path_config
        pipeline = MagicMock(spec=AnchorPipeline)
        pipeline.path_config = MagicMock()
        pipeline.path_config.md_file = None

        # Create a fake .md file path
        fake_md_file = MagicMock(spec=Path)
        fake_md_file.stem = "aclnn_abs"
        fake_md_file.name = "aclnn_abs.md"
        fake_md_file.parent = Path("/fake/operator/path")

        # Mock operator_path.glob to return list with fake .md file
        pipeline.path_config.operator_path.glob.return_value = [fake_md_file]

        # Mock doc_json path to return a Path that doesn't exist
        fake_doc_json_path = MagicMock(spec=Path)
        fake_doc_json_path.exists.return_value = False
        pipeline.path_config.doc_json.__truediv__ = MagicMock(return_value=fake_doc_json_path)

        # Mock _populate_sub_function_hash_sub_part to return the doc unchanged
        # (This simulates the actual behavior when doc_json_file doesn't exist)
        def mock_populate(doc, doc_json_file):
            # The actual method checks doc_json_file.exists() and returns doc unchanged if False
            return doc

        pipeline._populate_sub_function_hash_sub_part = MagicMock(side_effect=mock_populate)

        # Get sub_function_hash config from registry
        registry = get_registry()
        config = registry.get_config("sub_function_hash")

        # Create empty anchor_tuple (required parameter)
        anchor_tuple = []

        # Call _build_doc_structure directly (bind to AnchorPipeline class)
        doc = AnchorPipeline._build_doc_structure(pipeline, "sub_function_hash", config, anchor_tuple)

        # Verify doc has expected structure
        assert isinstance(doc, dict)
        assert "doc_path" in doc
        assert "doc_name" in doc
        assert "doc_part" in doc
        assert "md_format" in doc
        assert "sub_part" in doc

        # Verify doc_part is correct
        assert doc["doc_part"] == "sub_function_hash"

        # Verify doc_path and doc_name are populated from fake .md file
        assert doc["doc_path"] == "/fake/operator/path"
        assert doc["doc_name"] == "aclnn_abs.md"

        # Verify md_format is empty list (no doc_json data loaded)
        assert doc["md_format"] == []

        # Verify sub_part has template keys from config
        expected_sub_part_keys = {
            "param_matrix",
            "fixconfig_1",
            "fixconfig_2",
            "changeconfig",
            "errorcode_matrix",
        }
        assert set(doc["sub_part"].keys()) == expected_sub_part_keys

        # Verify sub_part values are empty dicts (template structure, no doc_json data loaded)
        for key in expected_sub_part_keys:
            assert doc["sub_part"][key] == {}
