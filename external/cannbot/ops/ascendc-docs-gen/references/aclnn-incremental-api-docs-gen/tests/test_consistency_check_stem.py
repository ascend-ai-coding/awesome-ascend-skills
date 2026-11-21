# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""TDD tests for run_consistency_check_pipeline.py D1/D2/D3 — selected_md_stem over selected_file.stem.

Verifies that the 3 V2-bias sites in run_consistency_check_pipeline.py are fixed:

D1: _auto_select_doc_file (line 197) — replaced with select_doc_file(doc_files, selected_md_stem=...)
D2: lines 253, 268 — f"{selected_md_stem}.json" instead of f"{selected_file.stem}.json"
D3: line 185 — doc_json_dir = version_dir.parent / "doc_json" (operator level, not version level)

Test strategy:
  D1: Set up docs=[V2.md, V3.md], selected_md_stem="aclnnV3" (from single dir).
      Without fix: _auto_select_doc_file returns V2.md (first) -> path has aclnnV2.json.
      With fix: select_doc_file returns V3.md -> path has aclnnV3.json.
  D2: Mock select_doc_file to return V2.md (V2-bias), selected_md_stem="aclnnV3".
      Without fix: path uses selected_file.stem -> aclnnV2.json.
      With fix: path uses selected_md_stem -> aclnnV3.json.
  D3: Check doc_json_dir is at operator level (version_dir.parent / "doc_json"),
      NOT at version level (version_dir / "doc_json").
"""

import os

# Ensure env vars set before importing run_consistency_check_pipeline (module-level
# check at L36-37 raises Exception if INCREMENTAL_LLM_KEY/URL/MODEL missing).
os.environ.setdefault("INCREMENTAL_LLM_KEY", "test-key")
os.environ.setdefault("INCREMENTAL_LLM_URL", "test-url")
os.environ.setdefault("INCREMENTAL_LLM_MODEL", "test-model")

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import src.run_consistency_check_pipeline as rcc
from src.generation_validation.ui.cli import PipelineResult


# =============================================================================
# Helpers
# =============================================================================


def _setup_workspace(tmp_path: Path, stem: str = "aclnnV3") -> tuple:
    """Create a workspace with the directory structure needed for run_pipeline.

    Structure:
      workspace/
        op/                          <- original_op_dir (operator source, just needs name)
        data/
          op/                        <- md_stems_dir (operator-level snapshots)
            doc_json/                <- operator-level doc_json (D3 correct location)
              {stem}.json
            {stem}/                  <- version_dir (per-stem)
              anchor/                <- evidence_dir (must exist)
              docs/                  <- source_docs
                aclnnV2.md
                aclnnV3.md
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Operator source dir (just needs to exist for the name)
    (workspace / "op").mkdir()

    # Data dir structure
    op_data = workspace / "data" / "op"  # md_stems_dir
    op_data.mkdir(parents=True)

    # Operator-level doc_json (D3 correct location — sibling of version dirs)
    doc_json = op_data / "doc_json"
    doc_json.mkdir()
    (doc_json / f"{stem}.json").write_text("{}")

    # Per-stem version dir
    version_dir = op_data / stem
    version_dir.mkdir()

    # Evidence dir (must exist — line 187 checks)
    (version_dir / "anchor").mkdir()

    # Docs dir with V2.md and V3.md
    docs_dir = version_dir / "docs"
    docs_dir.mkdir()
    v2_path = docs_dir / "aclnnV2.md"
    v2_path.write_text("# V2")
    v3_path = docs_dir / "aclnnV3.md"
    v3_path.write_text("# V3")

    return workspace, v2_path, v3_path


def _run_pipeline(workspace: Path, mock_select_return=None) -> MagicMock:
    """Run run_pipeline with all external deps mocked. Returns mock_validate.

    If mock_select_return is provided, patches select_doc_file to return it
    (used by D2 test to simulate V2-bias: selected_file=V2.md but
    selected_md_stem="aclnnV3").
    """
    mock_validate = MagicMock(return_value=PipelineResult())
    mock_full = MagicMock(return_value=PipelineResult())
    mock_resolve = MagicMock(return_value=MagicMock())  # non-None evidence file
    mock_extractor_inst = MagicMock()
    mock_extractor_inst.extract_fine_grained_mappings.return_value = {
        "sub_function_hash": [{"path": "test.json"}]
    }

    patches = [
        patch(
            "src.change_detection.doc_part_change_detection.docpart_file_extractor.DocpartFileExtractor",
            return_value=mock_extractor_inst,
        ),
        patch("src.generation_validation.ui.cli.run_validate_only", mock_validate),
        patch("src.generation_validation.ui.cli.run_full_pipeline", mock_full),
        patch("src.generation_validation.ui.cli.resolve_evidence_file", mock_resolve),
        patch("src.logger.setup_logger"),
    ]

    if mock_select_return is not None:
        # create=True allows patching select_doc_file even before D1 fix
        # (when it doesn't exist yet in the module namespace)
        patches.append(
            patch.object(
                rcc, "select_doc_file", create=True, return_value=mock_select_return
            )
        )

    for p in patches:
        p.start()

    try:
        rcc.run_pipeline(
            snapshot_id="test",
            workspace_dir=str(workspace),
            original_op_dir="op",
            data_path="data",
            operator_version="aclnnV3",
        )
    finally:
        for p in patches:
            p.stop()

    return mock_validate


# =============================================================================
# D1: select_doc_file uses selected_md_stem (not _auto_select_doc_file)
# =============================================================================


class TestD1SelectsV3WithStem:
    """D1: select_doc_file should be called with selected_md_stem to select V3.md."""

    def test_d1_selects_v3_with_stem(self, tmp_path):
        """selected_md_stem='aclnnV3', docs=[V2.md, V3.md] -> V3.md selected.

        Without fix: _auto_select_doc_file returns default_files[0] = V2.md (FAIL).
        With fix: select_doc_file(doc_files, selected_md_stem='aclnnV3') returns V3.md (PASS).

        Verify: doc_json_dir kwarg passed to run_validate_only contains "aclnnV3.json"
        (because V3 was selected, selected_file.stem = "aclnnV3").
        """
        workspace, _, _ = _setup_workspace(tmp_path, stem="aclnnV3")
        mock_validate = _run_pipeline(workspace)

        assert mock_validate.called, "run_validate_only should have been called"
        doc_json_arg = mock_validate.call_args.kwargs.get("doc_json_dir", "")
        assert "aclnnV3.json" in doc_json_arg, (
            f"Expected aclnnV3.json (V3 selected via stem), got: {doc_json_arg}"
        )
        assert "aclnnV2.json" not in doc_json_arg, (
            f"V2 selected instead of V3 (V2-bias): {doc_json_arg}"
        )


# =============================================================================
# D2: doc_json filename uses selected_md_stem, not selected_file.stem
# =============================================================================


class TestD2FilenameUsesStem:
    """D2: doc_json filename should use selected_md_stem, not selected_file.stem."""

    def test_d2_filename_uses_stem(self, tmp_path):
        """Mock select_doc_file to return V2.md, selected_md_stem='aclnnV3'.

        Without fix: path uses selected_file.stem -> aclnnV2.json (FAIL).
        With fix: path uses selected_md_stem -> aclnnV3.json (PASS).
        """
        workspace, v2_path, _ = _setup_workspace(tmp_path, stem="aclnnV3")
        # Mock select_doc_file to return V2.md — simulates V2-bias where
        # selected_file is V2 but selected_md_stem is V3
        mock_validate = _run_pipeline(workspace, mock_select_return=v2_path)

        assert mock_validate.called, "run_validate_only should have been called"
        doc_json_arg = mock_validate.call_args.kwargs.get("doc_json_dir", "")
        assert "aclnnV3.json" in doc_json_arg, (
            f"Expected aclnnV3.json (from selected_md_stem), got: {doc_json_arg}"
        )
        assert "aclnnV2.json" not in doc_json_arg, (
            f"V2 stem leaked into filename (selected_file.stem used instead of "
            f"selected_md_stem): {doc_json_arg}"
        )


# =============================================================================
# D3: doc_json_dir at operator level (version_dir.parent), not version level
# =============================================================================


class TestD3DocJsonDirAtOperatorLevel:
    """D3: doc_json_dir should be at operator level (version_dir.parent / "doc_json")."""

    def test_d3_doc_json_dir_at_operator_level(self, tmp_path):
        """version_dir = md_stems_dir / 'aclnnV3' -> doc_json_dir = md_stems_dir / 'doc_json'.

        Without fix: doc_json_dir = version_dir / "doc_json" (version level, WRONG).
        With fix: doc_json_dir = version_dir.parent / "doc_json" (operator level, CORRECT).

        Verify: the parent directory of the doc_json file path (kwarg passed to
        run_validate_only) is at operator level, not inside version_dir.
        """
        workspace, _, _ = _setup_workspace(tmp_path, stem="aclnnV3")
        mock_validate = _run_pipeline(workspace)

        assert mock_validate.called, "run_validate_only should have been called"
        doc_json_arg = mock_validate.call_args.kwargs.get("doc_json_dir", "")
        assert doc_json_arg, "doc_json_dir kwarg should be non-empty"

        # The kwarg is str(doc_json_dir / "filename.json"), so parent is doc_json_dir
        doc_json_dir_path = Path(doc_json_arg).parent

        # After D3 fix: doc_json_dir = version_dir.parent / "doc_json"
        #             = md_stems_dir / "doc_json" = workspace/data/op/doc_json
        expected_operator_level = workspace / "data" / "op" / "doc_json"
        # Before D3 fix (WRONG): doc_json_dir = version_dir / "doc_json"
        wrong_version_level = workspace / "data" / "op" / "aclnnV3" / "doc_json"

        assert doc_json_dir_path == expected_operator_level, (
            f"doc_json_dir should be at OPERATOR level ({expected_operator_level}), "
            f"got: {doc_json_dir_path}"
        )
        assert doc_json_dir_path != wrong_version_level, (
            f"doc_json_dir is at VERSION level (inside version_dir, WRONG): "
            f"{doc_json_dir_path}"
        )
