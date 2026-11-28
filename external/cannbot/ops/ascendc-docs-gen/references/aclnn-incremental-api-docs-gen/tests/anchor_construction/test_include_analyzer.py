# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
RED-phase tests for build_file_index and _search_candidates in include_analyzer.

These tests lock in the EXPECTED post-fix behavior so they FAIL against the
current (unfixed) code:

  - build_file_index accepts an optional ``op_dir`` to scope the index to the
    operator directory plus the shared ``common/inc`` tree (excluding sibling
    operators).
  - _search_candidates normalizes ``..`` in step 2 (normpath) so relative
    includes resolve through the index instead of missing and falling through
    to fuzzy matching.
  - A scoped index prevents cross-operator fuzzy matches.
  - build_transitive(expand_dir_cpp=True) pulls same-directory .cpp companion
    files (kernel entry / tiling template registrations that no file
    #includes) into the closure and expands their includes.
"""

from pathlib import Path

from src.snapshot_construction.anchor_construction.common.include_analyzer import (
    build_file_index,
    build_transitive,
    _search_candidates,
)


class TestBuildFileIndexScoped:
    """Tests for build_file_index scoped indexing via the op_dir parameter."""

    def test_scoped_index_excludes_sibling_dirs(self, tmp_path: Path):
        """Scoped index (op_dir) includes op_dir files + common/inc, excludes sibling operators."""
        # Setup
        (tmp_path / "math" / "add" / "op_kernel").mkdir(parents=True)
        (tmp_path / "math" / "add" / "op_kernel" / "a.h").touch()
        (tmp_path / "common" / "inc" / "external").mkdir(parents=True)
        (tmp_path / "common" / "inc" / "external" / "b.h").touch()
        (tmp_path / "math" / "sub" / "op_kernel").mkdir(parents=True)
        (tmp_path / "math" / "sub" / "op_kernel" / "c.h").touch()

        # Execute
        result = build_file_index(tmp_path, op_dir=tmp_path / "math" / "add")

        # Verify
        assert "a.h" in result["name"]
        assert "b.h" in result["name"]
        assert "c.h" not in result["name"]

    def test_backward_compat_no_op_dir_scans_full_repo(self, tmp_path: Path):
        """Without op_dir, build_file_index scans the whole repo (backward compat)."""
        # Setup
        (tmp_path / "math" / "add" / "op_kernel").mkdir(parents=True)
        (tmp_path / "math" / "add" / "op_kernel" / "a.h").touch()
        (tmp_path / "common" / "inc" / "external").mkdir(parents=True)
        (tmp_path / "common" / "inc" / "external" / "b.h").touch()
        (tmp_path / "math" / "sub" / "op_kernel").mkdir(parents=True)
        (tmp_path / "math" / "sub" / "op_kernel" / "c.h").touch()

        # Execute
        result = build_file_index(tmp_path)

        # Verify — old behavior preserved: all three files indexed
        assert "a.h" in result["name"]
        assert "b.h" in result["name"]
        assert "c.h" in result["name"]

    def test_common_inc_missing_guard(self, tmp_path: Path):
        """Scoped index tolerates a missing common/inc directory (no OSError)."""
        # Setup — note: no common/inc directory exists
        (tmp_path / "math" / "add" / "op_kernel").mkdir(parents=True)
        (tmp_path / "math" / "add" / "op_kernel" / "a.h").touch()

        # Execute — must not raise
        result = build_file_index(tmp_path, op_dir=tmp_path / "math" / "add")

        # Verify
        assert "a.h" in result["name"]

    def test_non_op_dirs_included_in_name_index(self, tmp_path: Path):
        """NON_OP_DIRS (tests/, common/, etc.) are scanned for name_index, siblings are not."""
        (tmp_path / "math" / "add" / "op_host").mkdir(parents=True)
        (tmp_path / "math" / "add" / "op_host" / "foo.cpp").touch()
        (tmp_path / "tests" / "ut" / "op_api" / "stub" / "opdev").mkdir(parents=True)
        (
            tmp_path
            / "tests"
            / "ut"
            / "op_api"
            / "stub"
            / "opdev"
            / "make_op_executor.h"
        ).touch()
        (tmp_path / "math" / "sub" / "op_kernel").mkdir(parents=True)
        (tmp_path / "math" / "sub" / "op_kernel" / "make_op_executor.h").touch()

        idx = build_file_index(tmp_path, op_dir=tmp_path / "math" / "add")

        assert "make_op_executor.h" in idx["name"]
        name_hits = idx["name"]["make_op_executor.h"]
        hit_paths = [str(p) for p in name_hits]
        assert any("tests" in p for p in hit_paths)
        assert not any("math/sub" in p for p in hit_paths)

        candidates = _search_candidates(
            "opdev/make_op_executor.h",
            file_dir=tmp_path / "math" / "add" / "op_host",
            ops_root=tmp_path,
            file_index=idx,
        )
        expected = (
            tmp_path
            / "tests"
            / "ut"
            / "op_api"
            / "stub"
            / "opdev"
            / "make_op_executor.h"
        )
        assert any(p == expected for p in candidates)


class TestSearchCandidatesNormpath:
    """Tests for _search_candidates normpath resolution and scoped fuzzy match."""

    def test_step2_normpath_resolves_double_dot(self, tmp_path: Path):
        """Step 2 normalizes ``..`` so relative includes resolve via the index."""
        # Setup
        (tmp_path / "math" / "add" / "op_host").mkdir(parents=True)
        (tmp_path / "math" / "add" / "op_host" / "foo.cpp").touch()
        (tmp_path / "math" / "add" / "op_kernel").mkdir(parents=True)
        (tmp_path / "math" / "add" / "op_kernel" / "abs_struct.h").touch()

        # Execute
        idx = build_file_index(tmp_path, op_dir=tmp_path / "math" / "add")
        candidates = _search_candidates(
            "../../op_kernel/abs_struct.h",
            file_dir=tmp_path / "math" / "add" / "op_host",
            ops_root=tmp_path,
            file_index=idx,
        )

        # Verify — resolved via normpath to the op_kernel header
        expected = tmp_path / "math" / "add" / "op_kernel" / "abs_struct.h"
        assert any(p == expected for p in candidates)

    def test_fuzzy_scoped_no_cross_operator(self, tmp_path: Path):
        """Scoped fuzzy match does not return sibling operator files."""
        # Setup
        (tmp_path / "math" / "add" / "op_kernel").mkdir(parents=True)
        (tmp_path / "math" / "add" / "op_kernel" / "abs_struct.h").touch()
        (tmp_path / "math" / "sub" / "op_kernel").mkdir(parents=True)
        (tmp_path / "math" / "sub" / "op_kernel" / "abs_struct.h").touch()

        # Execute — bare filename triggers fuzzy fallback
        idx = build_file_index(tmp_path, op_dir=tmp_path / "math" / "add")
        candidates = _search_candidates(
            "abs_struct.h",
            file_dir=tmp_path / "math" / "add" / "op_host",
            ops_root=tmp_path,
            file_index=idx,
        )

        # Verify — own op_dir file is a candidate, sibling is not
        expected = tmp_path / "math" / "add" / "op_kernel" / "abs_struct.h"
        sibling = tmp_path / "math" / "sub" / "op_kernel" / "abs_struct.h"
        assert any(p == expected for p in candidates)
        assert not any(p == sibling for p in candidates)

    def test_cross_operator_relative_include_resolved(self, tmp_path: Path):
        """Cross-operator ../ include resolved via step 2 normpath + full-repo rel_index."""
        (tmp_path / "math" / "add" / "op_host").mkdir(parents=True)
        (tmp_path / "math" / "add" / "op_host" / "foo.cpp").touch()
        (tmp_path / "math" / "complex_abs" / "op_api").mkdir(parents=True)
        (tmp_path / "math" / "complex_abs" / "op_api" / "complex_abs.h").touch()

        idx = build_file_index(tmp_path, op_dir=tmp_path / "math" / "add")

        rel_key = str(
            (
                tmp_path / "math" / "complex_abs" / "op_api" / "complex_abs.h"
            ).relative_to(tmp_path)
        )
        assert rel_key in idx["rel"]
        assert "complex_abs.h" not in idx["name"]

        candidates = _search_candidates(
            "../../complex_abs/op_api/complex_abs.h",
            file_dir=tmp_path / "math" / "add" / "op_host",
            ops_root=tmp_path,
            file_index=idx,
        )
        expected = tmp_path / "math" / "complex_abs" / "op_api" / "complex_abs.h"
        assert any(p == expected for p in candidates)


class TestBuildTransitiveDirCompanions:
    """Directory-cpp companion expansion in build_transitive.

    Mirrors the ops-nn kernel layout: a shared op_kernel dir whose .cpp entry
    file (kernel_entry.cpp) is #included by nobody — only the directory
    companion rule can reach it and expand its includes (mix_online.h).
    """

    def _make_tree(self, tmp_path: Path) -> Path:
        shared = tmp_path / "math" / "sub" / "op_kernel"
        op_host = tmp_path / "math" / "add" / "op_host"
        op_host.mkdir(parents=True)
        shared.mkdir(parents=True)
        (op_host / "entry.cpp").touch()
        (shared / "impl_unit.h").touch()
        (shared / "kernel_entry.cpp").write_text(
            '#include "mix_online.h"\n', encoding="utf-8"
        )
        (shared / "mix_online.h").touch()
        (shared / "mix_online.cpp").touch()
        return shared

    def _run(self, tmp_path: Path, **kwargs):
        op_dir = tmp_path / "math" / "add"
        idx = build_file_index(tmp_path, op_dir=op_dir)
        direct_repo_other = [
            {
                "原始引用": "../../sub/op_kernel/impl_unit.h",
                "解析路径": f"{tmp_path.name}/math/sub/op_kernel/impl_unit.h",
            }
        ]
        return build_transitive(
            [],
            direct_repo_other,
            [],
            tmp_path,
            op_dir,
            file_index=idx,
            **kwargs,
        )

    def _repo_paths(self, result) -> set[str]:
        return {item["解析路径"] for item in result["②_仓内其他"]}

    def test_companion_cpp_enters_closure_and_expands_includes(self, tmp_path: Path):
        shared = self._make_tree(tmp_path)
        result = self._run(tmp_path, expand_dir_cpp=True)

        paths = self._repo_paths(result)
        root = tmp_path.name
        assert f"{root}/math/sub/op_kernel/impl_unit.h" in paths
        assert f"{root}/math/sub/op_kernel/kernel_entry.cpp" in paths
        assert f"{root}/math/sub/op_kernel/mix_online.h" in paths
        assert f"{root}/math/sub/op_kernel/mix_online.cpp" in paths
        assert shared.is_dir()

    def test_disabled_keeps_include_only_closure(self, tmp_path: Path):
        self._make_tree(tmp_path)
        result = self._run(tmp_path, expand_dir_cpp=False)

        paths = self._repo_paths(result)
        root = tmp_path.name
        assert f"{root}/math/sub/op_kernel/impl_unit.h" in paths
        assert f"{root}/math/sub/op_kernel/kernel_entry.cpp" not in paths
        assert f"{root}/math/sub/op_kernel/mix_online.h" not in paths

    def test_max_companions_cap(self, tmp_path: Path):
        self._make_tree(tmp_path)
        result = self._run(tmp_path, expand_dir_cpp=True, max_companions=1)

        paths = self._repo_paths(result)
        root = tmp_path.name
        assert f"{root}/math/sub/op_kernel/kernel_entry.cpp" in paths
        assert f"{root}/math/sub/op_kernel/mix_online.cpp" not in paths
        assert f"{root}/math/sub/op_kernel/mix_online.h" in paths

    def test_companion_in_op_dir_classified_as_self(self, tmp_path: Path):
        op_dir = tmp_path / "math" / "add"
        op_host = op_dir / "op_host"
        op_host.mkdir(parents=True)
        (op_host / "entry.h").touch()
        (op_host / "entry.cpp").touch()

        idx = build_file_index(tmp_path, op_dir=op_dir)
        result = build_transitive(
            [f"{tmp_path.name}/math/add/op_host/entry.h"],
            [],
            [],
            tmp_path,
            op_dir,
            file_index=idx,
            expand_dir_cpp=True,
        )

        assert f"{tmp_path.name}/math/add/op_host/entry.cpp" in result["①_自身目录"]
