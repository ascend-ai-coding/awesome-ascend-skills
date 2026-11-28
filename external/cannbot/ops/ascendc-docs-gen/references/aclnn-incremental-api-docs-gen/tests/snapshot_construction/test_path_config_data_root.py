# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Tests for PathConfig data_root redirection (Task 2: snapshot-data-prep-dedup).

Verifies that operator-level data (doc_json, code_map, tree_id,
original_docs_dir, operator_note_dir) can be redirected to a shared
`data_root` directory, while per-md artifacts (anchor_dir, temp_dir)
remain anchored to `out_path`. Also verifies backward compatibility:
when `data_root` is omitted (None), all attributes derive from out_path
exactly as before the refactor.
"""

from pathlib import Path

import pytest

from src.snapshot_construction.anchor_construction.common.path_config import PathConfig

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def op_path(tmp_path) -> Path:
    """Operator source directory (not redirected; any path works)."""
    p = tmp_path / "operator"
    p.mkdir()
    return p


@pytest.fixture
def out_path(tmp_path) -> Path:
    """Per-md output directory (anchor_dir / temp_dir stay here)."""
    p = tmp_path / "out"
    p.mkdir()
    return p


@pytest.fixture
def shared_path(tmp_path) -> Path:
    """Shared data_root directory for operator-level data redirection."""
    p = tmp_path / "shared"
    p.mkdir()
    return p


# =============================================================================
# Tests
# =============================================================================


class TestDataRootRedirect:
    """data_root redirects operator-level data to a shared directory."""

    def test_data_root_redirects_operator_data(self, op_path, out_path, shared_path):
        """With data_root=shared, the 5 operator-level attributes resolve
        under shared; anchor_dir/temp_dir stay under out_path.
        """
        cfg = PathConfig(op_path, out_path, data_root=shared_path)

        # Operator-level data -> shared
        assert cfg.doc_json == shared_path / PathConfig.DOC_JSON_DIR
        assert cfg.code_map == shared_path / PathConfig.CODE_MAP_DIR
        assert cfg.tree_id == shared_path / PathConfig.TREE_ID_DIR
        assert cfg.original_docs_dir == (shared_path / PathConfig.TREE_ID_DIR / PathConfig.ORIGINAL_DOCS_DIR)
        assert cfg.operator_note_dir == (shared_path / PathConfig.TREE_ID_DIR / PathConfig.OPERATOR_NOTE_DIR)

        # Per-md artifacts -> out_path (unchanged)
        assert cfg.anchor_dir == out_path / PathConfig.ANCHOR_DIR
        assert cfg.temp_dir == out_path / PathConfig.TEMP_DIR

    def test_data_root_accepts_str(self, op_path, out_path, shared_path):
        """data_root accepts a str path (coerced to Path internally)."""
        cfg = PathConfig(op_path, out_path, data_root=str(shared_path))

        assert cfg.doc_json == shared_path / PathConfig.DOC_JSON_DIR
        assert cfg.code_map == shared_path / PathConfig.CODE_MAP_DIR


class TestDataRootBackwardCompat:
    """data_root=None preserves pre-refactor behavior (all under out_path)."""

    def test_data_root_none_backward_compat(self, op_path, out_path):
        """Without data_root, every attribute derives from out_path."""
        cfg = PathConfig(op_path, out_path)

        # Operator-level data -> out_path (pre-refactor behavior)
        assert cfg.doc_json == out_path / PathConfig.DOC_JSON_DIR
        assert cfg.code_map == out_path / PathConfig.CODE_MAP_DIR
        assert cfg.tree_id == out_path / PathConfig.TREE_ID_DIR
        assert cfg.original_docs_dir == (out_path / PathConfig.TREE_ID_DIR / PathConfig.ORIGINAL_DOCS_DIR)
        assert cfg.operator_note_dir == (out_path / PathConfig.TREE_ID_DIR / PathConfig.OPERATOR_NOTE_DIR)

        # Per-md artifacts -> out_path (unchanged)
        assert cfg.anchor_dir == out_path / PathConfig.ANCHOR_DIR
        assert cfg.temp_dir == out_path / PathConfig.TEMP_DIR

    def test_data_root_explicit_none_matches_omitted(self, op_path, out_path):
        """Passing data_root=None explicitly must equal omitting it."""
        cfg_omitted = PathConfig(op_path, out_path)
        cfg_explicit_none = PathConfig(op_path, out_path, data_root=None)

        for attr in (
            "doc_json",
            "code_map",
            "tree_id",
            "original_docs_dir",
            "operator_note_dir",
            "anchor_dir",
            "temp_dir",
        ):
            assert getattr(cfg_omitted, attr) == getattr(cfg_explicit_none, attr)


class TestUnchangedAttributes:
    """Attributes that must NOT be affected by data_root."""

    def test_operator_path_out_path_md_file_unchanged(self, op_path, out_path, shared_path):
        """operator_path / out_path / md_file / md_stem ignore data_root."""
        md_file = "aclnnAbs.md"
        cfg = PathConfig(op_path, out_path, md_file=md_file, data_root=shared_path)

        assert cfg.operator_path == op_path
        assert cfg.out_path == out_path
        assert cfg.md_file == md_file
        assert cfg.md_stem == "aclnnAbs"
