# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Tests for unified file discovery functions.
"""

from pathlib import Path

import pytest

from src.snapshot_construction.anchor_construction.common.file_discovery import (
    find_aclnn_cpp_files,
    find_all_aclnn_docs,
    find_def_h_file,
    find_header_files,
    find_op_api_files,
    find_operator_files,
)


class TestFindOpApiFiles:
    """Tests for find_op_api_files function."""

    def test_find_op_api_files_with_op_api_dir(self, tmp_path: Path):
        """Test finding files in op_api/ directory."""
        # Setup
        op_api = tmp_path / "op_api"
        op_api.mkdir()
        (op_api / "aclnn_test.h").touch()
        (op_api / "aclnn_test.cpp").touch()
        (op_api / "other.h").touch()  # Should not match

        # Execute
        result = find_op_api_files(tmp_path)

        # Verify
        assert len(result) == 2
        paths = [p for p, _ in result]
        sources = [s for _, s in result]
        assert any("aclnn_test.h" in str(p) for p in paths)
        assert any("aclnn_test.cpp" in str(p) for p in paths)
        assert "aclnn_*.h" in sources
        assert "aclnn_*.cpp" in sources

    def test_find_op_api_files_with_op_host_op_api_dir(self, tmp_path: Path):
        """Test finding files in op_host/op_api/ directory."""
        # Setup
        op_host = tmp_path / "op_host"
        op_host.mkdir()
        op_api = op_host / "op_api"
        op_api.mkdir()
        (op_api / "aclnn_custom.h").touch()
        (op_api / "aclnn_custom.cpp").touch()

        # Execute
        result = find_op_api_files(tmp_path)

        # Verify
        assert len(result) == 2
        paths = [p for p, _ in result]
        assert any("aclnn_custom.h" in str(p) for p in paths)
        assert any("aclnn_custom.cpp" in str(p) for p in paths)

    def test_find_op_api_files_no_op_api_dir(self, tmp_path: Path):
        """Test when no op_api directory exists."""
        # Execute
        result = find_op_api_files(tmp_path)

        # Verify
        assert result == []

    def test_find_op_api_files_empty_op_api_dir(self, tmp_path: Path):
        """Test when op_api directory is empty."""
        # Setup
        op_api = tmp_path / "op_api"
        op_api.mkdir()

        # Execute
        result = find_op_api_files(tmp_path)

        # Verify
        assert result == []


class TestFindOperatorFiles:
    """Tests for find_operator_files function."""

    def test_find_operator_files_all_patterns(self, tmp_path: Path):
        """Test finding all operator file patterns."""
        # Setup
        op_api = tmp_path / "op_api"
        op_api.mkdir()
        (op_api / "aclnn_op.h").touch()
        (op_api / "aclnn_op.cpp").touch()

        op_host = tmp_path / "op_host"
        op_host.mkdir()
        (op_host / "def.h").touch()
        (op_host / "tiling.h").touch()

        op_graph = tmp_path / "op_graph"
        op_graph.mkdir()
        (op_graph / "custom_proto.h").touch()

        # Execute
        result = find_operator_files(tmp_path)

        # Verify
        assert len(result) == 5
        sources = [s for _, s in result]
        assert "aclnn_*.h" in sources
        assert "aclnn_*.cpp" in sources
        assert "*def*.h" in sources
        assert "*tiling*.h" in sources
        assert "*_proto.h" in sources

    def test_find_operator_files_deduplication(self, tmp_path: Path):
        """Test that duplicate file paths are removed."""
        op_host = tmp_path / "op_host"
        op_host.mkdir()
        op_tiling = op_host / "op_tiling"
        op_tiling.mkdir()
        # tiling_def.h matches *tiling*.h, *def*.h, and op_tiling/*.h patterns
        (op_tiling / "tiling_def.h").touch()

        result = find_operator_files(tmp_path)

        # File should appear only once despite matching multiple patterns
        matching = [p for p, _ in result if "tiling_def.h" in p.name]
        assert len(matching) == 1

    def test_find_operator_files_partial_patterns(self, tmp_path: Path):
        """Test finding only some patterns when others don't exist."""
        # Setup
        op_api = tmp_path / "op_api"
        op_api.mkdir()
        (op_api / "aclnn_op.h").touch()
        # No op_graph, no def.h, no tilling.h

        # Execute
        result = find_operator_files(tmp_path)

        # Verify
        assert len(result) == 1
        assert result[0][1] == "aclnn_*.h"

    def test_find_operator_files_empty_directory(self, tmp_path: Path):
        """Test when operator directory is empty."""
        # Execute
        result = find_operator_files(tmp_path)

        # Verify
        assert result == []


class TestFindAclnnCppFiles:
    """Tests for find_aclnn_cpp_files function."""

    def test_find_aclnn_cpp_files_basic(self, tmp_path: Path):
        """Test finding aclnn_*.cpp files."""
        # Setup
        (tmp_path / "aclnn_test.cpp").touch()
        (tmp_path / "aclnn_other.cpp").touch()
        (tmp_path / "regular.cpp").touch()  # Should not match

        # Execute
        result = find_aclnn_cpp_files(tmp_path)

        # Verify
        assert len(result) == 2
        assert all("aclnn_" in f.name for f in result)

    def test_find_aclnn_cpp_files_recursive(self, tmp_path: Path):
        """Test finding files recursively in subdirectories."""
        # Setup
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "aclnn_root.cpp").touch()
        (subdir / "aclnn_sub.cpp").touch()

        # Execute
        result = find_aclnn_cpp_files(tmp_path)

        # Verify
        assert len(result) == 2
        names = [f.name for f in result]
        assert "aclnn_root.cpp" in names
        assert "aclnn_sub.cpp" in names

    def test_find_aclnn_cpp_files_no_match(self, tmp_path: Path):
        """Test when no aclnn_*.cpp files exist."""
        # Setup
        (tmp_path / "test.cpp").touch()
        (tmp_path / "other.h").touch()

        # Execute
        result = find_aclnn_cpp_files(tmp_path)

        # Verify
        assert result == []

    def test_find_aclnn_cpp_files_nonexistent_path(self, tmp_path: Path):
        """Test when path does not exist."""
        # Setup
        nonexistent = tmp_path / "nonexistent"

        # Execute
        result = find_aclnn_cpp_files(nonexistent)

        # Verify
        assert result == []


class TestFindHeaderFiles:
    """Tests for find_header_files function."""

    def test_find_header_files_op_api(self, tmp_path: Path):
        """Test finding header files in op_api directory."""
        # Setup
        op_api = tmp_path / "op_api"
        op_api.mkdir()
        (op_api / "test.h").touch()
        (op_api / "other.h").touch()

        # Execute
        result = find_header_files(tmp_path)

        # Verify
        assert len(result) == 2
        assert all(f.suffix == ".h" for f in result)

    def test_find_header_files_aclnn_prefix(self, tmp_path: Path):
        """Test finding header files with aclnn_ prefix."""
        # Setup
        (tmp_path / "aclnn_test.h").touch()
        (tmp_path / "regular.h").touch()  # Should not match

        # Execute
        result = find_header_files(tmp_path)

        # Verify
        assert len(result) == 1
        assert result[0].name == "aclnn_test.h"

    def test_find_header_files_combined(self, tmp_path: Path):
        """Test finding both op_api and aclnn_ prefixed files."""
        # Setup
        op_api = tmp_path / "op_api"
        op_api.mkdir()
        (op_api / "api_header.h").touch()
        (tmp_path / "aclnn_header.h").touch()
        (tmp_path / "regular.h").touch()  # Should not match

        # Execute
        result = find_header_files(tmp_path)

        # Verify
        assert len(result) == 2
        names = [f.name for f in result]
        assert "api_header.h" in names
        assert "aclnn_header.h" in names

    def test_find_header_files_no_match(self, tmp_path: Path):
        """Test when no matching header files exist."""
        # Setup
        (tmp_path / "regular.h").touch()  # Not in op_api, not aclnn_ prefix

        # Execute
        result = find_header_files(tmp_path)

        # Verify
        assert result == []

    def test_find_header_files_nonexistent_path(self, tmp_path: Path):
        """Test when path does not exist."""
        # Setup
        nonexistent = tmp_path / "nonexistent"

        # Execute
        result = find_header_files(nonexistent)

        # Verify
        assert result == []


class TestFindDefHFile:
    """Tests for find_def_h_file function."""

    def test_find_def_h_file_root(self, tmp_path: Path):
        """Test finding def.h in root directory."""
        # Setup
        (tmp_path / "def.h").touch()

        # Execute
        result = find_def_h_file(tmp_path)

        # Verify
        assert result is not None
        assert result.name == "def.h"

    def test_find_def_h_file_subdirectory(self, tmp_path: Path):
        """Test finding def.h in subdirectory."""
        # Setup
        subdir = tmp_path / "op_api"
        subdir.mkdir()
        (subdir / "def.h").touch()

        # Execute
        result = find_def_h_file(tmp_path)

        # Verify
        assert result is not None
        assert result.name == "def.h"
        assert "op_api" in str(result)

    def test_find_def_h_file_not_found(self, tmp_path: Path):
        """Test when def.h does not exist."""
        # Execute
        result = find_def_h_file(tmp_path)

        # Verify
        assert result is None

    def test_find_def_h_file_nonexistent_path(self, tmp_path: Path):
        """Test when path does not exist."""
        # Setup
        nonexistent = tmp_path / "nonexistent"

        # Execute
        result = find_def_h_file(nonexistent)

        # Verify
        assert result is None


class TestFindAllAclnnDocs:
    """Tests for find_all_aclnn_docs function."""

    def test_find_all_aclnn_docs_basic(self, tmp_path: Path):
        """Test finding aclnn*.md files."""
        # Setup
        (tmp_path / "aclnn_test.md").touch()
        (tmp_path / "aclnn_other.md").touch()
        (tmp_path / "regular.md").touch()  # Should not match

        # Execute
        result = find_all_aclnn_docs(tmp_path)

        # Verify
        assert len(result) == 2
        assert all(f.name.startswith("aclnn") for f in result)

    def test_find_all_aclnn_docs_recursive(self, tmp_path: Path):
        """Test finding files recursively."""
        # Setup
        subdir = tmp_path / "docs"
        subdir.mkdir()
        (tmp_path / "aclnn_root.md").touch()
        (subdir / "aclnn_sub.md").touch()

        # Execute
        result = find_all_aclnn_docs(tmp_path)

        # Verify
        assert len(result) == 2
        names = [f.name for f in result]
        assert "aclnn_root.md" in names
        assert "aclnn_sub.md" in names

    def test_find_all_aclnn_docs_no_match(self, tmp_path: Path):
        """Test when no aclnn*.md files exist."""
        # Setup
        (tmp_path / "test.md").touch()
        (tmp_path / "other.txt").touch()

        # Execute
        result = find_all_aclnn_docs(tmp_path)

        # Verify
        assert result == []

    def test_find_all_aclnn_docs_nonexistent_path(self, tmp_path: Path):
        """Test when path does not exist."""
        # Setup
        nonexistent = tmp_path / "nonexistent"

        # Execute
        result = find_all_aclnn_docs(nonexistent)

        # Verify
        assert result == []


class TestIntegration:
    """Integration tests combining multiple discovery functions."""

    def test_full_operator_structure(self, tmp_path: Path):
        """Test finding all files in a complete operator structure."""
        # Setup - create a realistic operator directory structure
        op_api = tmp_path / "op_api"
        op_api.mkdir()
        (op_api / "aclnn_add.h").touch()
        (op_api / "aclnn_add.cpp").touch()
        (op_api / "def.h").touch()
        (op_api / "tilling.h").touch()

        op_graph = tmp_path / "op_graph"
        op_graph.mkdir()
        (op_graph / "add_proto.h").touch()

        op_kernel = tmp_path / "op_kernel"
        op_kernel.mkdir()
        (op_kernel / "add_kernel.cpp").touch()
        (op_kernel / "add_kernel.h").touch()

        (tmp_path / "aclnn_add_doc.md").touch()

        # Execute all discovery functions
        op_api_files = find_op_api_files(tmp_path)
        operator_files = find_operator_files(tmp_path)
        aclnn_cpp = find_aclnn_cpp_files(tmp_path)
        headers = find_header_files(tmp_path)
        def_h = find_def_h_file(tmp_path)
        docs = find_all_aclnn_docs(tmp_path)

        # Verify
        assert len(op_api_files) == 2  # aclnn_add.h, aclnn_add.cpp
        assert len(operator_files) == 5  # 2 op_api + 1 proto + 1 def.h + 1 tilling.h
        assert len(aclnn_cpp) == 1  # aclnn_add.cpp
        assert len(headers) == 3  # aclnn_add.h, def.h, tilling.h (all in op_api)
        assert def_h is not None
        assert def_h.name == "def.h"
        assert len(docs) == 1  # aclnn_add_doc.md
