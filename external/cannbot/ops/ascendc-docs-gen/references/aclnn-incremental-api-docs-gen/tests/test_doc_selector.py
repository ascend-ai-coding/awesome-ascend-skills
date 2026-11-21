# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Tests for stem-aware document file selector (Task 2 — TDD RED/GREEN)."""

from pathlib import Path

from src.snapshot_construction.anchor_construction.common.doc_selector import (
    select_doc_file,
)


def test_exact_stem_match():
    """selected_md_stem="aclnnV3" with [aclnnV2.md, aclnnV3.md] -> returns aclnnV3.md."""
    files = [Path("aclnnV2.md"), Path("aclnnV3.md")]
    result = select_doc_file(files, selected_md_stem="aclnnV3")
    assert result == Path("aclnnV3.md")


def test_fallback_none():
    """selected_md_stem=None with [aclnnV2.md, aclnnV3.md] -> returns aclnnV2.md (alphabetically first)."""
    files = [Path("aclnnV3.md"), Path("aclnnV2.md")]
    result = select_doc_file(files, selected_md_stem=None)
    assert result == Path("aclnnV2.md")


def test_empty_list():
    """Empty doc_files list -> returns None."""
    result = select_doc_file([], selected_md_stem=None)
    assert result is None


def test_non_aclnn_filtered():
    """Non-aclnn prefix files filtered out; returns None if no aclnn files remain."""
    files = [Path("README.md"), Path("notes.md")]
    result = select_doc_file(files, selected_md_stem=None)
    assert result is None


def test_no_match_returns_none():
    """selected_md_stem="aclnnNonExist" with [aclnnV2.md, aclnnV3.md] -> returns None (no fallback to first)."""
    files = [Path("aclnnV2.md"), Path("aclnnV3.md")]
    result = select_doc_file(files, selected_md_stem="aclnnNonExist")
    assert result is None


def test_empty_string_stem_falls_back():
    """Empty-string selected_md_stem behaves like None (backward-compat fallback to first aclnn*.md)."""
    files = [Path("aclnnV2.md"), Path("aclnnV3.md")]
    result = select_doc_file(files, selected_md_stem="")
    assert result == Path("aclnnV2.md")


def test_non_aclnn_ignored_when_exact_stem_requested():
    """When exact stem requested, non-aclnn files are not matched even if stem equals."""
    # A non-aclnn file whose stem happens to equal the requested stem must NOT match,
    # because the selector only considers aclnn* files.
    files = [Path("aclnnV2.md"), Path("randomstem.md")]
    result = select_doc_file(files, selected_md_stem="randomstem")
    assert result is None
