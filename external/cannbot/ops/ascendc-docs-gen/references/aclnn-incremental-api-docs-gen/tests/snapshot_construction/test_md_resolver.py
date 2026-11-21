# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Tests for md_resolver.resolve_md_to_source_files.

Tests cover the VALIDATED naming convention rules (Task 3 dry-run, 49 md files,
16 operators, 100% match rate with fallback):
- Standard camelCase: aclnnAbs.md -> aclnn_abs.cpp, aclnn_abs.h
- Multi-API (& split): aclnnAddV3&aclnnInplaceAddV3.md -> both API source files
- Prefix marker ([new], [deprecated]) stripping
- No-match fallback to all aclnn_*.cpp/h in op_api
- Shared files (proto/def/tiling/kernel) inclusion for all mds
- V2/V3 version suffix in API name (part of name, not stripped)
- Nested op_host/op_api directory
- Deduplication of results
- Empty operator directory
- Return type is list[Path]
"""

from pathlib import Path

import pytest

from src.snapshot_construction.anchor_construction.common.md_resolver import (
    resolve_md_to_source_files,
)


# =============================================================================
# Helper functions
# =============================================================================


def make_operator_dir(
    tmp_path,
    op_name="test_op",
    op_api_files=None,
    shared_files=None,
    op_api_nested=False,
):
    """Create a fake operator directory with source files.

    Args:
        tmp_path: pytest tmp_path fixture
        op_name: operator directory name
        op_api_files: list of filenames to create in op_api/ (e.g. ['aclnn_abs.cpp'])
        shared_files: list of relative paths for shared files (e.g. ['op_graph/abs_proto.h'])
        op_api_nested: if True, create op_api under op_host/ instead of top-level

    Returns:
        Path to the operator directory
    """
    op_dir = tmp_path / op_name
    if op_api_nested:
        op_api_dir = op_dir / "op_host" / "op_api"
    else:
        op_api_dir = op_dir / "op_api"
    op_api_dir.mkdir(parents=True, exist_ok=True)

    for fname in op_api_files or []:
        (op_api_dir / fname).touch()

    for rel_path in shared_files or []:
        full_path = op_dir / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.touch()

    return op_dir


def names_of(paths):
    """Extract sorted filenames from a list of Path objects."""
    return sorted(p.name for p in paths)


# =============================================================================
# Tests for resolve_md_to_source_files
# =============================================================================


class TestResolveMdToSourceFiles:
    """Tests for resolve_md_to_source_files function."""

    def test_standard_camelcase(self, tmp_path):
        """aclnnAbs.md -> matches aclnn_abs.cpp, aclnn_abs.h"""
        op_dir = make_operator_dir(
            tmp_path,
            op_api_files=["aclnn_abs.cpp", "aclnn_abs.h"],
        )
        md_path = tmp_path / "aclnnAbs.md"

        result = resolve_md_to_source_files(md_path, op_dir)
        result_names = names_of(result)

        assert "aclnn_abs.cpp" in result_names
        assert "aclnn_abs.h" in result_names

    def test_amp_multi_api(self, tmp_path):
        """aclnnAddV3&aclnnInplaceAddV3.md -> matches aclnn_add_v3.* AND aclnn_inplace_add_v3.*"""
        op_dir = make_operator_dir(
            tmp_path,
            op_api_files=[
                "aclnn_add_v3.cpp",
                "aclnn_add_v3.h",
                "aclnn_inplace_add_v3.cpp",
                "aclnn_inplace_add_v3.h",
            ],
        )
        md_path = tmp_path / "aclnnAddV3&aclnnInplaceAddV3.md"

        result = resolve_md_to_source_files(md_path, op_dir)
        result_names = names_of(result)

        assert "aclnn_add_v3.cpp" in result_names
        assert "aclnn_add_v3.h" in result_names
        assert "aclnn_inplace_add_v3.cpp" in result_names
        assert "aclnn_inplace_add_v3.h" in result_names

    def test_new_prefix_stripped(self, tmp_path):
        """[new]aclnnAbs.md -> strip [new] -> matches aclnn_abs.*"""
        op_dir = make_operator_dir(
            tmp_path,
            op_api_files=["aclnn_abs.cpp", "aclnn_abs.h"],
        )
        md_path = tmp_path / "[new]aclnnAbs.md"

        result = resolve_md_to_source_files(md_path, op_dir)
        result_names = names_of(result)

        assert "aclnn_abs.cpp" in result_names
        assert "aclnn_abs.h" in result_names

    def test_deprecated_prefix_stripped(self, tmp_path):
        """[deprecated]aclnnAbs.md -> strip [deprecated] -> matches aclnn_abs.*"""
        op_dir = make_operator_dir(
            tmp_path,
            op_api_files=["aclnn_abs.cpp", "aclnn_abs.h"],
        )
        md_path = tmp_path / "[deprecated]aclnnAbs.md"

        result = resolve_md_to_source_files(md_path, op_dir)
        result_names = names_of(result)

        assert "aclnn_abs.cpp" in result_names
        assert "aclnn_abs.h" in result_names

    def test_no_match_fallback(self, tmp_path):
        """md with no naming-matched source -> returns all aclnn_*.cpp/h in op_api"""
        op_dir = make_operator_dir(
            tmp_path,
            op_api_files=[
                "aclnn_abs.cpp",
                "aclnn_abs.h",
                "aclnn_other.cpp",
                "aclnn_other.h",
            ],
        )
        md_path = tmp_path / "aclnnNonexistent.md"

        result = resolve_md_to_source_files(md_path, op_dir)
        result_names = names_of(result)

        # Fallback: ALL aclnn_*.cpp/h in op_api
        assert "aclnn_abs.cpp" in result_names
        assert "aclnn_abs.h" in result_names
        assert "aclnn_other.cpp" in result_names
        assert "aclnn_other.h" in result_names

    def test_shared_files_included(self, tmp_path):
        """Every result includes proto/def/tiling/kernel files"""
        shared = [
            "op_graph/abs_proto.h",
            "op_host/abs_def.cpp",
            "op_host/abs_def.h",
            "op_host/abs_tiling.cpp",
            "op_host/abs_tiling.h",
            "op_kernel/abs_kernel.cpp",
            "op_kernel/abs_kernel.h",
        ]
        op_dir = make_operator_dir(
            tmp_path,
            op_api_files=["aclnn_abs.cpp", "aclnn_abs.h"],
            shared_files=shared,
        )
        md_path = tmp_path / "aclnnAbs.md"

        result = resolve_md_to_source_files(md_path, op_dir)
        result_names = names_of(result)

        # Shared files must be included
        assert "abs_proto.h" in result_names
        assert "abs_def.cpp" in result_names
        assert "abs_def.h" in result_names
        assert "abs_tiling.cpp" in result_names
        assert "abs_tiling.h" in result_names
        assert "abs_kernel.cpp" in result_names
        assert "abs_kernel.h" in result_names
        # Plus the naming-matched files
        assert "aclnn_abs.cpp" in result_names
        assert "aclnn_abs.h" in result_names

    def test_shared_files_included_with_fallback(self, tmp_path):
        """Shared files included even when fallback is used"""
        shared = [
            "op_graph/op_proto.h",
            "op_host/op_def.cpp",
            "op_kernel/kernel.cpp",
        ]
        op_dir = make_operator_dir(
            tmp_path,
            op_api_files=["aclnn_base.cpp", "aclnn_base.h"],
            shared_files=shared,
        )
        md_path = tmp_path / "aclnnNonexistent.md"

        result = resolve_md_to_source_files(md_path, op_dir)
        result_names = names_of(result)

        # Fallback aclnn files
        assert "aclnn_base.cpp" in result_names
        assert "aclnn_base.h" in result_names
        # Shared files
        assert "op_proto.h" in result_names
        assert "op_def.cpp" in result_names
        assert "kernel.cpp" in result_names

    def test_v2_in_api_name(self, tmp_path):
        """aclnnCumsumV2.md -> aclnn_cumsum_v2.* (V2 is API name, not stripped)"""
        op_dir = make_operator_dir(
            tmp_path,
            op_api_files=[
                "aclnn_cumsum_v2.cpp",
                "aclnn_cumsum_v2.h",
                "aclnn_cumsum.cpp",
                "aclnn_cumsum.h",
            ],
        )
        md_path = tmp_path / "aclnnCumsumV2.md"

        result = resolve_md_to_source_files(md_path, op_dir)
        result_names = names_of(result)

        # Should match v2 files (naming match succeeds)
        assert "aclnn_cumsum_v2.cpp" in result_names
        assert "aclnn_cumsum_v2.h" in result_names
        # Should NOT include non-v2 files (naming match succeeded, no fallback)
        assert "aclnn_cumsum.cpp" not in result_names
        assert "aclnn_cumsum.h" not in result_names

    def test_op_host_op_api_nested(self, tmp_path):
        """When op_api doesn't exist at top level, use op_host/op_api"""
        op_dir = make_operator_dir(
            tmp_path,
            op_api_files=["aclnn_abs.cpp", "aclnn_abs.h"],
            op_api_nested=True,
        )
        md_path = tmp_path / "aclnnAbs.md"

        result = resolve_md_to_source_files(md_path, op_dir)
        result_names = names_of(result)

        assert "aclnn_abs.cpp" in result_names
        assert "aclnn_abs.h" in result_names

    def test_deduplication(self, tmp_path):
        """No duplicate paths in result"""
        op_dir = make_operator_dir(
            tmp_path,
            op_api_files=["aclnn_abs.cpp", "aclnn_abs.h"],
            shared_files=["op_graph/abs_proto.h"],
        )
        md_path = tmp_path / "aclnnAbs.md"

        result = resolve_md_to_source_files(md_path, op_dir)

        paths_as_str = [str(p) for p in result]
        assert len(paths_as_str) == len(set(paths_as_str))

    def test_empty_operator(self, tmp_path):
        """No source files at all -> returns empty list"""
        op_dir = tmp_path / "empty_op"
        op_dir.mkdir()
        md_path = tmp_path / "aclnnAbs.md"

        result = resolve_md_to_source_files(md_path, op_dir)

        assert result == []

    def test_multiple_bracket_prefixes(self, tmp_path):
        """[new][deprecated]aclnnAbs.md -> strip both -> matches aclnn_abs.*"""
        op_dir = make_operator_dir(
            tmp_path,
            op_api_files=["aclnn_abs.cpp", "aclnn_abs.h"],
        )
        md_path = tmp_path / "[new][deprecated]aclnnAbs.md"

        result = resolve_md_to_source_files(md_path, op_dir)
        result_names = names_of(result)

        assert "aclnn_abs.cpp" in result_names
        assert "aclnn_abs.h" in result_names

    def test_returns_list_of_paths(self, tmp_path):
        """Result is a list of Path objects"""
        op_dir = make_operator_dir(
            tmp_path,
            op_api_files=["aclnn_abs.cpp", "aclnn_abs.h"],
        )
        md_path = tmp_path / "aclnnAbs.md"

        result = resolve_md_to_source_files(md_path, op_dir)

        assert isinstance(result, list)
        assert all(isinstance(p, Path) for p in result)

    def test_amp_multi_api_partial_match(self, tmp_path):
        """Multi-API where only one API has matching files -> uses naming match (not fallback)"""
        op_dir = make_operator_dir(
            tmp_path,
            op_api_files=[
                "aclnn_add_v3.cpp",
                "aclnn_add_v3.h",
                # aclnn_inplace_add_v3.* does NOT exist
            ],
        )
        md_path = tmp_path / "aclnnAddV3&aclnnInplaceAddV3.md"

        result = resolve_md_to_source_files(md_path, op_dir)
        result_names = names_of(result)

        # Should match the one that exists
        assert "aclnn_add_v3.cpp" in result_names
        assert "aclnn_add_v3.h" in result_names
        # Should NOT fallback (partial match is still a match)
        assert "aclnn_inplace_add_v3.cpp" not in result_names

    def test_op_tiling_shared_files(self, tmp_path):
        """op_tiling/ directory files are collected as shared"""
        shared = [
            "op_tiling/tiling.cpp",
            "op_tiling/tiling.h",
        ]
        op_dir = make_operator_dir(
            tmp_path,
            op_api_files=["aclnn_abs.cpp", "aclnn_abs.h"],
            shared_files=shared,
        )
        md_path = tmp_path / "aclnnAbs.md"

        result = resolve_md_to_source_files(md_path, op_dir)
        result_names = names_of(result)

        assert "tiling.cpp" in result_names
        assert "tiling.h" in result_names
