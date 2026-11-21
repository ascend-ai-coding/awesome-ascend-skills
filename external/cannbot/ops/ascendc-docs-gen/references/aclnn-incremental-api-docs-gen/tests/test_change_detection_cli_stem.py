# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""TDD tests for run_change_detection_pipeline.py E1/E2 — selected_md_stem over selected_file.stem.

Verifies that the V2-bias sites in run_change_detection_pipeline.py are fixed:

E1: _auto_select_doc_file (line 155) — replaced with select_doc_file(doc_files, selected_md_stem=...)
E2: line 162 — f"{selected_md_stem}-summary.json" instead of f"{selected_file.stem}-summary.json"
E2: lines 259, 267 — f"[new]{selected_md_stem}.md" instead of f"[new]{selected_file.stem}.md"

Test strategy:
  E1: Set up docs=[V2.md, V3.md], selected_md_stem="aclnnV3" (from operator_version).
      Without fix: _auto_select_doc_file returns V2.md (first) -> doc_content="# V2".
      With fix: select_doc_file returns V3.md -> doc_content="# V3".
  E2 summary: Mock select_doc_file to return V2.md (V2-bias), selected_md_stem="aclnnV3".
      Without fix: summary filename uses selected_file.stem -> aclnnV2-summary.json.
      With fix: summary filename uses selected_md_stem -> aclnnV3-summary.json.
  E2 new_md: Mock select_doc_file to return V2.md, selected_md_stem="aclnnV3".
      Without fix: [new] filename uses selected_file.stem -> [new]aclnnV2.md.
      With fix: [new] filename uses selected_md_stem -> [new]aclnnV3.md.
"""

import os

# Ensure env vars set before importing run_change_detection_pipeline (module-level
# check at L36-37 raises Exception if INCREMENTAL_LLM_KEY/URL/MODEL missing).
os.environ.setdefault("INCREMENTAL_LLM_KEY", "test-key")
os.environ.setdefault("INCREMENTAL_LLM_URL", "test-url")
os.environ.setdefault("INCREMENTAL_LLM_MODEL", "test-model")

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import src.run_change_detection_pipeline as rcd


# =============================================================================
# Helpers
# =============================================================================


def _setup_workspace(tmp_path: Path, stem: str = "aclnnV3") -> tuple:
    """Create a workspace with the directory structure needed for run_pipeline.

    Structure:
      workspace/
        op/                           <- original_op_dir (operator source)
        op_mod/                       <- modified_op_dir (just needs to exist)
        data/
          op/                         <- md_stems_dir (operator-level snapshots)
            aclnnV3/                  <- version_dir (per-stem)
              anchor/                 <- evidence_dir
              docs/                   <- source_docs
                aclnnV2.md
                aclnnV3.md
              style/                  <- summary_dir (created by run_pipeline)
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Operator source dir (just needs to exist for the name)
    (workspace / "op").mkdir()
    (workspace / "op_mod").mkdir()

    # Data dir structure
    op_data = workspace / "data" / "op"  # md_stems_dir
    op_data.mkdir(parents=True)

    # Per-stem version dir
    version_dir = op_data / stem
    version_dir.mkdir(parents=True)

    # Evidence dir (must exist — line 139 uses it)
    (version_dir / "anchor").mkdir()

    # Docs dir with V2.md and V3.md
    docs_dir = version_dir / "docs"
    docs_dir.mkdir()
    v2_path = docs_dir / "aclnnV2.md"
    v2_path.write_text("# V2")
    v3_path = docs_dir / "aclnnV3.md"
    v3_path.write_text("# V3")

    return workspace, v2_path, v3_path


def _run_pipeline(workspace: Path, mock_select_return=None) -> tuple:
    """Run run_pipeline with all external deps mocked.

    Returns (mock_summarize, mock_update_md, results) for assertion.

    If mock_select_return is provided, patches select_doc_file to return it
    (used by E2 tests to simulate V2-bias: selected_file=V2.md but
    selected_md_stem="aclnnV3").
    """
    mock_summarize = MagicMock()
    mock_update_md = MagicMock()
    # run_evaluation 返回 4 元组: (todo_list, reasons, modified_file_dir_list, repo_other_changed_files)
    mock_evaluation = MagicMock(return_value=(["sub_function_hash"], [], [], []))
    mock_change_cli = MagicMock()
    mock_merge = MagicMock()

    # run_full_pipeline mock: 满足 _process_doc_part 决策矩阵的成功结果
    # (success=True, 无证据冲突/无需修改, 非空 generated, 验证通过) ——
    # MagicMock 默认属性为真值 Mock, 不显式设置会被新契约误判为 evidence_conflict
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.evidence_conflict = False
    mock_result.no_change_needed = False
    mock_result.generated = [{"generated_text": "generated content"}]
    mock_result.validation = [{"consistency_passed": True, "metric_passed": "PASS"}]
    mock_full = MagicMock(return_value=mock_result)

    mock_resolve = MagicMock(return_value=MagicMock())  # non-None evidence file

    patches = [
        patch(
            "src.generation_validation.style.summarizer.gen_doc_summarize",
            mock_summarize,
        ),
        patch(
            "src.change_detection.doc_part_change_detection.change_evaluator.run_evaluation",
            mock_evaluation,
        ),
        patch(
            "src.change_detection.run_change_detection_cli.change_deteciton_cli",
            mock_change_cli,
        ),
        patch(
            "src.change_detection.merge_evidence.merge_redirect_to_anchor", mock_merge
        ),
        patch("src.generation_validation.ui.cli.run_full_pipeline", mock_full),
        patch("src.generation_validation.ui.cli.resolve_evidence_file", mock_resolve),
        patch("src.logger.setup_logger"),
        patch("src.md_fill.main_fill.update_md_with_dict", mock_update_md),
    ]

    if mock_select_return is not None:
        # create=True allows patching select_doc_file even before E1 fix
        # (when it doesn't exist yet in the module namespace)
        patches.append(
            patch.object(
                rcd, "select_doc_file", create=True, return_value=mock_select_return
            )
        )

    for p in patches:
        p.start()

    try:
        results = rcd.run_pipeline(
            snapshot_id="test",
            workspace_dir=str(workspace),
            original_op_dir="op",
            modified_op_dir="op_mod",
            data_path="data",
            operator_version="aclnnV3",
        )
    finally:
        for p in patches:
            p.stop()

    return mock_summarize, mock_update_md, results


# =============================================================================
# E1: select_doc_file uses selected_md_stem (not _auto_select_doc_file)
# =============================================================================


class TestE1SelectsV3WithStem:
    """E1: select_doc_file should be called with selected_md_stem to select V3.md."""

    def test_e1_selects_v3_with_stem(self, tmp_path):
        """selected_md_stem='aclnnV3', docs=[V2.md, V3.md] -> V3.md selected.

        Without fix: _auto_select_doc_file returns default_files[0] = V2.md (FAIL).
        With fix: select_doc_file(doc_files, selected_md_stem='aclnnV3') returns V3.md (PASS).

        Verify: gen_doc_summarize called with doc_content="# V3" (content of V3.md,
        not V2.md). doc_content comes from Path(selected_file).read_text() at line 158.
        """
        workspace, _, _ = _setup_workspace(tmp_path, stem="aclnnV3")
        mock_summarize, _, _ = _run_pipeline(workspace)

        assert mock_summarize.called, "gen_doc_summarize should have been called"
        doc_content = mock_summarize.call_args.args[0]  # 1st positional arg
        assert doc_content == "# V3", (
            f"Expected '# V3' (V3 selected via stem), got: {doc_content!r}"
        )


# =============================================================================
# E2: summary filename uses selected_md_stem, not selected_file.stem
# =============================================================================


class TestE2SummaryFilenameUsesStem:
    """E2 (line 162): summary filename should use selected_md_stem, not selected_file.stem."""

    def test_e2_summary_filename_uses_stem(self, tmp_path):
        """Mock select_doc_file to return V2.md, selected_md_stem='aclnnV3'.

        Without fix: filename = f"{selected_file.stem}-summary.json" = aclnnV2-summary.json (FAIL).
        With fix: filename = f"{selected_md_stem}-summary.json" = aclnnV3-summary.json (PASS).
        """
        workspace, v2_path, _ = _setup_workspace(tmp_path, stem="aclnnV3")
        mock_summarize, _, _ = _run_pipeline(workspace, mock_select_return=v2_path)

        assert mock_summarize.called, "gen_doc_summarize should have been called"
        summary_file_dir = mock_summarize.call_args.args[1]  # 2nd positional arg
        assert "aclnnV3-summary.json" in summary_file_dir, (
            f"Expected aclnnV3-summary.json (from selected_md_stem), got: {summary_file_dir}"
        )
        assert "aclnnV2-summary.json" not in summary_file_dir, (
            f"V2 stem leaked into summary filename (selected_file.stem used instead of "
            f"selected_md_stem): {summary_file_dir}"
        )


# =============================================================================
# E2: [new] markdown filename uses selected_md_stem, not selected_file.stem
# =============================================================================


class TestE2NewMdFilenameUsesStem:
    """E2 (lines 259, 267): [new] markdown filename should use selected_md_stem."""

    def test_e2_new_md_filename_uses_stem(self, tmp_path):
        """Mock select_doc_file to return V2.md, selected_md_stem='aclnnV3'.

        Without fix: [new] filename = f"[new]{selected_file.stem}.md" = [new]aclnnV2.md (FAIL).
        With fix: [new] filename = f"[new]{selected_md_stem}.md" = [new]aclnnV3.md (PASS).

        Verify: update_md_with_dict 3rd arg (output_path) and results["markdownFiles"][0]["fileName"].
        """
        workspace, v2_path, _ = _setup_workspace(tmp_path, stem="aclnnV3")
        mock_summarize, mock_update_md, results = _run_pipeline(
            workspace, mock_select_return=v2_path
        )

        assert mock_update_md.called, "update_md_with_dict should have been called"
        output_path = mock_update_md.call_args.args[2]  # 3rd positional arg
        assert "[new]aclnnV3.md" in output_path, (
            f"Expected [new]aclnnV3.md (from selected_md_stem), got: {output_path}"
        )
        assert "[new]aclnnV2.md" not in output_path, (
            f"V2 stem leaked into [new] filename: {output_path}"
        )

        # Also check results["markdownFiles"][0]["fileName"] (line 267)
        assert results["markdownFiles"], "markdownFiles should not be empty"
        file_name = results["markdownFiles"][0]["fileName"]
        assert file_name == "[new]aclnnV3.md", (
            f"Expected [new]aclnnV3.md (from selected_md_stem), got: {file_name}"
        )
