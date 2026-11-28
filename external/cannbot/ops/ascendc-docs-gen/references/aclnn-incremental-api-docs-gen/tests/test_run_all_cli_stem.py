# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""TDD tests for run_all_cli.py A1/A2/A3/A4 — selected_md_stem over selected_file.stem.

Verifies that the 4 V2-bias sites in run_all_cli.py use ctx.selected_md_stem
(the normalized stem) instead of ctx.selected_file.stem (which may be V2 when
V3 is explicitly requested).

A1: _step_config (line 386) — select_doc_file with selected_md_stem, not _auto_select_doc_file
A2: _step_summary (line 408) — summary filename uses selected_md_stem
A3: _step_fill_back (line 831) — fillback filename uses selected_md_stem
A4: _step_gen_val (line 701) — doc_json path uses selected_md_stem

Test strategy:
  Each test sets up a PipelineContext where selected_file points to V2.md but
  selected_md_stem is "aclnnMoeDistributeDispatchV3". If any site uses
  selected_file.stem, it gets V2 (FAIL); if it uses selected_md_stem, it gets V3 (PASS).
"""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from rich.console import Console

from src.run_all_cli import (
    PipelineContext,
    _step_config,
    _step_summary,
    _step_fill_back,
    _step_gen_val,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_ctx(
    tmp_path: Path, selected_md_stem: str = "aclnnMoeDistributeDispatchV3"
) -> PipelineContext:
    """Create a PipelineContext with V2 as selected_file but V3 as selected_md_stem.

    This mismatch is the key test invariant: if any site uses selected_file.stem,
    it gets V2; if it uses selected_md_stem, it gets V3.
    """
    return PipelineContext(
        workspace_dir=tmp_path,
        output_data_path=tmp_path / "output",
        base_output_data_path=tmp_path / "base",
        original_op_dir=tmp_path / "op",
        modified_op_dir=tmp_path / "op_mod",
        selected_op="moe_distribute_dispatch_v2",
        selected_md_stem=selected_md_stem,
        selected_file=Path("aclnnMoeDistributeDispatchV2.md"),  # V2 file!
        version_dir=tmp_path / "base" / "moe_distribute_dispatch_v2" / selected_md_stem,
        interactive=False,
        dashboard=False,
    )


# =============================================================================
# A1: _step_config selects V3 with stem
# =============================================================================


class TestA1SelectsV3WithStem:
    """A1: _step_config should use select_doc_file with selected_md_stem."""

    def test_a1_selects_v3_with_stem(self, tmp_path):
        """ctx.selected_md_stem='aclnnMoeDistributeDispatchV3', docs=[V2.md, V3.md]
        -> ctx.selected_file is V3.md (not V2).

        Without fix: _auto_select_doc_file returns default_files[0] = V2.md (FAIL).
        With fix: select_doc_file(doc_files, selected_md_stem=...) returns V3.md (PASS).
        """
        # Set up docs directory with V2.md and V3.md
        op_dir = tmp_path / "op"
        docs_dir = op_dir / "docs"
        docs_dir.mkdir(parents=True)
        (docs_dir / "aclnnMoeDistributeDispatchV2.md").write_text("# V2")
        (docs_dir / "aclnnMoeDistributeDispatchV3.md").write_text("# V3")

        ctx = PipelineContext(
            workspace_dir=tmp_path,
            output_data_path=tmp_path / "output",
            base_output_data_path=tmp_path / "base",
            original_op_dir=op_dir,
            selected_op="moe_distribute_dispatch_v2",
            selected_md_stem="aclnnMoeDistributeDispatchV3",
            interactive=False,
        )

        console = Console(quiet=True)
        _step_config(console, ctx)

        assert ctx.selected_file is not None, "selected_file should not be None"
        assert ctx.selected_file.name == "aclnnMoeDistributeDispatchV3.md", (
            f"Expected aclnnMoeDistributeDispatchV3.md (from selected_md_stem), "
            f"got {ctx.selected_file.name}"
        )


# =============================================================================
# A2: _step_summary filename uses selected_md_stem
# =============================================================================


class TestA2SummaryFilenameUsesStem:
    """A2: _step_summary should use ctx.selected_md_stem for summary filename."""

    @patch("src.generation_validation.style.summarizer.gen_doc_summarize")
    def test_a2_summary_filename_uses_stem(self, mock_summarize, tmp_path):
        """ctx.selected_md_stem='aclnnMoeDistributeDispatchV3' -> summary filename is
        'aclnnMoeDistributeDispatchV3-summary.json' (not V2).

        Without fix: filename = f"{ctx.selected_file.stem}-summary.json" = V2-summary.json (FAIL).
        With fix: filename = f"{ctx.selected_md_stem}-summary.json" = V3-summary.json (PASS).
        """
        ctx = _make_ctx(tmp_path)
        ctx.summary_dir = tmp_path / "style"
        ctx.doc_content = "# V2 content"

        console = Console(quiet=True)
        _step_summary(console, ctx)

        assert (
            ctx.summary_file_dir.name == "aclnnMoeDistributeDispatchV3-summary.json"
        ), (
            f"Expected aclnnMoeDistributeDispatchV3-summary.json (from selected_md_stem), "
            f"got {ctx.summary_file_dir.name}"
        )


# =============================================================================
# A3: _step_fill_back filename uses selected_md_stem
# =============================================================================


class TestA3FillbackFilenameUsesStem:
    """A3: _step_fill_back should use ctx.selected_md_stem for fillback filename."""

    @patch("src.md_fill.main_fill.update_md_with_dict")
    def test_a3_fillback_filename_uses_stem(self, mock_fill, tmp_path):
        """ctx.selected_md_stem='aclnnMoeDistributeDispatchV3' -> fillback output is
        '[new]aclnnMoeDistributeDispatchV3_{ts}.md' (not V2).

        Without fix: output_path = f"[new]{ctx.selected_file.stem}_{ts}.md" = [new]V2_{ts}.md (FAIL).
        With fix: output_path = f"[new]{ctx.selected_md_stem}_{ts}.md" = [new]V3_{ts}.md (PASS).
        """
        ctx = _make_ctx(tmp_path)
        ctx.backfill_map = {"sub_function_hash": "generated content"}
        ctx.version_dir = tmp_path / "version"
        ctx.version_dir.mkdir(parents=True, exist_ok=True)

        console = Console(quiet=True)
        _step_fill_back(console, ctx)

        # update_md_with_dict(ctx.backfill_map, ctx.selected_file, output_path)
        assert mock_fill.called, "update_md_with_dict should have been called"
        output_path = mock_fill.call_args.args[2]  # 3rd positional arg
        assert "[new]aclnnMoeDistributeDispatchV3_" in output_path, (
            f"Expected [new]aclnnMoeDistributeDispatchV3_ in output_path, got {output_path}"
        )
        assert "[new]aclnnMoeDistributeDispatchV2_" not in output_path, (
            f"V2 stem leaked into output_path: {output_path}"
        )


# =============================================================================
# A4: _step_gen_val doc_json path uses selected_md_stem
# =============================================================================


class TestA4DocJsonPathUsesStem:
    """A4: _step_gen_val should use ctx.selected_md_stem for doc_json path."""

    @patch("src.run_all_cli._display_results")
    @patch("src.run_all_cli.run_validate_only")
    @patch("src.run_all_cli.resolve_evidence_file")
    def test_a4_doc_json_path_uses_stem(
        self, mock_resolve, mock_validate, mock_display, tmp_path
    ):
        """ctx.selected_md_stem='aclnnMoeDistributeDispatchV3' -> doc_json path is
        'doc_json/aclnnMoeDistributeDispatchV3.json' (not V2).

        Without fix: doc_json_dir = .../doc_json/f"{ctx.selected_file.stem}.json" = V2.json (FAIL).
        With fix: doc_json_dir = .../doc_json/f"{ctx.selected_md_stem}.json" = V3.json (PASS).
        """
        # Mock evidence file
        mock_evidence = MagicMock()
        mock_evidence.path = tmp_path / "evidence.json"
        mock_evidence.relative_path = "evidence.json"
        mock_resolve.return_value = mock_evidence

        # Mock run_validate_only return value
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.validation = []
        mock_result.generated = []
        mock_validate.return_value = mock_result

        ctx = _make_ctx(tmp_path)
        ctx.doc_parts = ["sub_function_hash"]
        ctx.evidence_dir = tmp_path / "evidence"
        ctx.evidence_dir.mkdir(parents=True, exist_ok=True)
        ctx.re_detection_output_path = tmp_path / "re_detection"

        console = Console(quiet=True)
        _step_gen_val(console, ctx, validate_only=True)

        assert mock_validate.called, "run_validate_only should have been called"
        doc_json_dir = mock_validate.call_args.kwargs.get("doc_json_dir")
        assert doc_json_dir is not None, "doc_json_dir kwarg should be passed"
        assert doc_json_dir.name == "aclnnMoeDistributeDispatchV3.json", (
            f"Expected aclnnMoeDistributeDispatchV3.json (from selected_md_stem), "
            f"got {doc_json_dir.name}"
        )
