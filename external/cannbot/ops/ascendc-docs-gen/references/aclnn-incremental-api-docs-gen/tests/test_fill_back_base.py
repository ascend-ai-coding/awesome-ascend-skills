# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Tests for _step_fill_back backfill base selection.

B1: head (modified repo) docs/ 下存在同名 md -> 回填基底 = head md。
    端到端断言: 输出包含 LLM 重写章节 + head 特有章节内容, 不含 original 特有内容。
B2: head docs/ 缺同名 md -> 回退 original selected_file。
B3: modified_op_dir=None (交互 validate_only 等场景) -> 回退 original selected_file。
"""

from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from src.run_all_cli import PipelineContext, _step_fill_back

ORIGINAL_MD = """# aclnnTest

## 功能说明
original summary

## 约束说明
original constraint
"""

HEAD_MD = """# aclnnTest

## 功能说明
head summary (manual edit)

## 约束说明
head constraint (manual edit)
"""


def _make_ctx(tmp_path: Path, with_head_md: bool) -> PipelineContext:
    original_dir = tmp_path / "orig_op"
    original_docs = original_dir / "docs"
    original_docs.mkdir(parents=True)
    original_md = original_docs / "aclnnTest.md"
    original_md.write_text(ORIGINAL_MD, encoding="utf-8")

    modified_dir = tmp_path / "head_op"
    if with_head_md:
        modified_docs = modified_dir / "docs"
        modified_docs.mkdir(parents=True)
        (modified_docs / "aclnnTest.md").write_text(HEAD_MD, encoding="utf-8")

    version_dir = tmp_path / "output" / "test_op" / "aclnnTest"
    return PipelineContext(
        workspace_dir=tmp_path,
        output_data_path=tmp_path / "output",
        base_output_data_path=tmp_path / "output",
        original_op_dir=original_dir,
        modified_op_dir=modified_dir,
        selected_op="test_op",
        selected_md_stem="aclnnTest",
        selected_file=original_md,
        version_dir=version_dir,
        interactive=False,
    )


def _read_generated_md(version_dir: Path) -> str:
    results_dir = version_dir / "results"
    # glob 中 "[new]" 是字符类语法, 用 startswith 规避
    new_files = [p for p in results_dir.iterdir() if p.name.startswith("[new]aclnnTest_")]
    assert len(new_files) == 1, f"expected exactly one [new] md, got {new_files}"
    return new_files[0].read_text(encoding="utf-8")


class TestB1HeadBasePreferred:
    """B1: head docs/ 有同名 md -> 基底 = head md, 保留 PR 人工修改章节。"""

    def test_b1_head_md_as_base(self, tmp_path):
        ctx = _make_ctx(tmp_path, with_head_md=True)
        ctx.backfill_map = {"summary": "## 功能说明\nllm generated summary"}

        _step_fill_back(Console(quiet=True), ctx)

        content = _read_generated_md(ctx.version_dir)
        assert "llm generated summary" in content, "backfill 章节应被重写"
        assert "head constraint (manual edit)" in content, "head 基底: 未重写覆盖的章节应保留 head 人工修改内容"
        assert "original constraint" not in content, "不应残留 original 基底内容"
        assert "original summary" not in content

    @patch("src.md_fill.main_fill.update_md_with_dict")
    def test_b1_base_arg_is_head_md(self, mock_fill, tmp_path):
        ctx = _make_ctx(tmp_path, with_head_md=True)
        ctx.backfill_map = {"summary": "## 功能说明\nx"}

        _step_fill_back(Console(quiet=True), ctx)

        base_arg = Path(mock_fill.call_args.args[1])
        assert base_arg == tmp_path / "head_op" / "docs" / "aclnnTest.md"


class TestB2FallbackWhenHeadMissing:
    """B2: head docs/ 缺同名 md -> 回退 original selected_file (旧行为)。"""

    def test_b2_fallback_when_head_md_missing(self, tmp_path):
        ctx = _make_ctx(tmp_path, with_head_md=False)
        ctx.backfill_map = {"summary": "## 功能说明\nllm generated summary"}

        _step_fill_back(Console(quiet=True), ctx)

        content = _read_generated_md(ctx.version_dir)
        assert "llm generated summary" in content
        assert "original constraint" in content, "回退 original 基底: 其章节应保留"

    @patch("src.md_fill.main_fill.update_md_with_dict")
    def test_b2_base_arg_is_original(self, mock_fill, tmp_path):
        ctx = _make_ctx(tmp_path, with_head_md=False)
        ctx.backfill_map = {"summary": "## 功能说明\nx"}

        _step_fill_back(Console(quiet=True), ctx)

        base_arg = Path(mock_fill.call_args.args[1])
        assert base_arg == tmp_path / "orig_op" / "docs" / "aclnnTest.md"


class TestB3FallbackWhenNoModifiedDir:
    """B3: modified_op_dir=None -> 回退 original selected_file。"""

    def test_b3_fallback_when_no_modified_dir(self, tmp_path):
        ctx = _make_ctx(tmp_path, with_head_md=True)
        ctx.modified_op_dir = None
        ctx.backfill_map = {"summary": "## 功能说明\nllm generated summary"}

        _step_fill_back(Console(quiet=True), ctx)

        content = _read_generated_md(ctx.version_dir)
        assert "llm generated summary" in content
        assert "original constraint" in content, "无 modified_op_dir: 应回退 original 基底"
