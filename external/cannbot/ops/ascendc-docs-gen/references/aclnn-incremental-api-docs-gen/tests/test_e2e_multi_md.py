# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""E2e multi-md regression test — verifies V2/V3/V4 stem isolation.

End-to-end test that constructs a mini operator directory with multiple
aclnn*.md documentation files (V2/V3/V4), runs the AnchorPipeline for the
"constraint" doc_part, and asserts that the V3 round's output does NOT
cross-contaminate with V2.

Regression target: the V2-bias bug where pipeline.py sites B1/B2/B3 used
``sorted(glob)[0]`` (which is alphabetically V2) instead of the explicit
``path_config.md_file`` / ``path_config.md_stem``. With the fixes in place,
setting ``md_file="aclnnFakeOpV3.md"`` must produce constraint.json with
``doc.doc_name == "aclnnFakeOpV3.md"`` (NOT V2).

Test matrix:
  1. test_v3_doc_name_not_v2          — core regression (B1 site)
  2. test_v2_v3_v4_doc_blocks_differ  — isolation across all 3 versions
  3. test_select_doc_file_v3_not_v2   — selector integration
  4. test_single_md_backward_compat   — backward compat with md_file=None

No INCREMENTAL_LLM_* env dependency: the constraint doc_part reads pre-built doc_json
(no LLM). Env vars are set via setdefault as a safety net for any module-level
checks in the import chain, and LLMClient is patched as a belt-and-suspenders
guard.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Safety net: set INCREMENTAL_LLM_* env vars so any module-level check in the import
# chain doesn't raise. The constraint doc_part itself does NOT call LLM.
os.environ.setdefault("INCREMENTAL_LLM_KEY", "test-key-e2e")
os.environ.setdefault("INCREMENTAL_LLM_URL", "http://localhost:9999")
os.environ.setdefault("INCREMENTAL_LLM_MODEL", "test-model")

from src.snapshot_construction.anchor_construction.common.path_config import PathConfig
from src.snapshot_construction.anchor_construction.pipeline import AnchorPipeline
from src.snapshot_construction.anchor_construction.common.doc_selector import (
    select_doc_file,
)


# =============================================================================
# Test data
# =============================================================================

_MD_CONTENT = {
    "aclnnFakeOpV2": "# aclnnFakeOpV2\n\nV2 content here.",
    "aclnnFakeOpV3": "# aclnnFakeOpV3\n\nV3 content here.",
    "aclnnFakeOpV4": "# aclnnFakeOpV4\n\nV4 content here.",
}

# Each version's doc_json has a DISTINCT constraint.origin_txt so we can
# detect cross-contamination by inspecting md_format[0].
_CONSTRAINT_ORIGIN_TXT = {
    "aclnnFakeOpV2": "V2_ONLY constraint origin text",
    "aclnnFakeOpV3": "V3_ONLY constraint origin text",
    "aclnnFakeOpV4": "V4_ONLY constraint origin text",
}

_CONSTRAINT_CLARIFY = {
    "aclnnFakeOpV2": "V2 clarify text",
    "aclnnFakeOpV3": "V3 clarify text",
    "aclnnFakeOpV4": "V4 clarify text",
}


# =============================================================================
# Helpers
# =============================================================================


def _make_multi_md_operator(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create a mini operator dir with V2/V3/V4 md files + source + doc_json.

    Layout::
        tmp_path/operator/
            docs/aclnnFakeOpV2.md   (H1: # aclnnFakeOpV2)
            docs/aclnnFakeOpV3.md   (H1: # aclnnFakeOpV3)
            docs/aclnnFakeOpV4.md   (H1: # aclnnFakeOpV4)
            op_api/aclnn_fake_op_v2.h   (minimal header)
            op_api/aclnn_fake_op_v2.cpp (minimal source)
        tmp_path/data/
            doc_json/aclnnFakeOpV2.json  (constraint.origin_txt = V2_ONLY...)
            doc_json/aclnnFakeOpV3.json  (constraint.origin_txt = V3_ONLY...)
            doc_json/aclnnFakeOpV4.json  (constraint.origin_txt = V4_ONLY...)
        tmp_path/out/   (anchor output — created by pipeline)

    Returns:
        (operator_path, data_root, out_path)
    """
    operator_path = tmp_path / "operator"
    data_root = tmp_path / "data"
    out_path = tmp_path / "out"

    # docs/ — 3 md files
    docs_dir = operator_path / "docs"
    docs_dir.mkdir(parents=True)
    for stem, content in _MD_CONTENT.items():
        (docs_dir / f"{stem}.md").write_text(content, encoding="utf-8")

    # op_api/ — minimal source files (discovered by find_operator_files)
    op_api_dir = operator_path / "op_api"
    op_api_dir.mkdir()
    (op_api_dir / "aclnn_fake_op_v2.h").write_text(
        "#pragma once\n// minimal header for aclnnFakeOpV2\n",
        encoding="utf-8",
    )
    (op_api_dir / "aclnn_fake_op_v2.cpp").write_text(
        '// minimal source\n#include "aclnn_fake_op_v2.h"\n',
        encoding="utf-8",
    )

    # doc_json/ — 3 json files with DISTINCT constraint content
    doc_json_dir = data_root / "doc_json"
    doc_json_dir.mkdir(parents=True)
    for stem in _MD_CONTENT:
        (doc_json_dir / f"{stem}.json").write_text(
            json.dumps(
                {
                    "constraint": {
                        "origin_txt": _CONSTRAINT_ORIGIN_TXT[stem],
                        "constrain_clarify": _CONSTRAINT_CLARIFY[stem],
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    return operator_path, data_root, out_path


def _run_constraint_pipeline(
    operator_path: Path,
    data_root: Path,
    out_path: Path,
    md_file: str | None,
) -> dict:
    """Run AnchorPipeline for the constraint doc_part, return parsed constraint.json.

    Args:
        operator_path: Operator source directory (contains docs/ and op_api/).
        data_root: Shared data directory (contains doc_json/).
        out_path: Output directory (anchor/ subdir created here by pipeline).
        md_file: md filename to select (e.g. "aclnnFakeOpV3.md"), or None for
            backward-compat glob fallback.

    Returns:
        Parsed constraint.json dict with keys: "doc", "anchor_tuple".
    """
    pc = PathConfig(
        operator_path=operator_path,
        out_path=out_path,
        md_file=md_file,
        data_root=data_root,
    )
    pipeline = AnchorPipeline(pc, doc_part_control=["constraint"])
    pipeline.run()

    constraint_file = pc.anchor_dir / "constraint.json"
    assert constraint_file.exists(), f"constraint.json not written: {constraint_file}"
    with open(constraint_file, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Test 1: V3 doc_name not V2 (core regression)
# =============================================================================


class TestV3DocNameNotV2:
    """Core regression: V3 pipeline must not bleed into V2.

    Verifies B1 site (pipeline.py:517-522): _build_doc_structure uses
    path_config.md_file when set, NOT sorted(glob)[0] (which is V2).
    """

    def test_v3_doc_name_not_v2(self, tmp_path):
        """md_file='aclnnFakeOpV3.md' -> constraint.json doc_name == V3 (NOT V2).

        Without fix (B1): sorted(glob)[0] == aclnnFakeOpV2.md -> doc_name == V2.
        With fix: md_file set -> doc_name == aclnnFakeOpV3.md.
        """
        op_path, data_root, out_path = _make_multi_md_operator(tmp_path)

        result = _run_constraint_pipeline(
            op_path, data_root, out_path, md_file="aclnnFakeOpV3.md"
        )

        doc = result["doc"]

        # --- Assertion 1: doc_name == aclnnFakeOpV3.md (NOT V2) ---
        assert doc["doc_name"] == "aclnnFakeOpV3.md", (
            f"V2-bias leak: expected doc_name='aclnnFakeOpV3.md', "
            f"got '{doc['doc_name']}'"
        )

        # --- Assertion 2: md_format contains V3 content (NOT V2) ---
        md_format = doc.get("md_format", [])
        assert len(md_format) > 0, "md_format should not be empty"
        assert "V3_ONLY" in md_format[0], (
            f"Expected V3_ONLY in md_format, got: {md_format[0]}"
        )
        assert "V2_ONLY" not in md_format[0], (
            f"V2 content leaked into V3 md_format: {md_format[0]}"
        )

        # --- Assertion 4: V3 md file H1 == "# aclnnFakeOpV3" ---
        v3_md = op_path / "docs" / "aclnnFakeOpV3.md"
        v3_content = v3_md.read_text(encoding="utf-8")
        first_line = v3_content.splitlines()[0] if v3_content.splitlines() else ""
        assert first_line == "# aclnnFakeOpV3", (
            f"V3 md H1 mismatch: expected '# aclnnFakeOpV3', got '{first_line}'"
        )

        # --- Assertion 2 (fill_back stem): md_stem derives to aclnnFakeOpV3 ---
        # The fill_back filename pattern is "[new]{md_stem}_{timestamp}.md"
        # (run_all_cli.py A3 site). Verify md_stem is V3, NOT V2.
        pc = PathConfig(
            operator_path=op_path,
            out_path=out_path,
            md_file="aclnnFakeOpV3.md",
            data_root=data_root,
        )
        assert pc.md_stem == "aclnnFakeOpV3", (
            f"md_stem mismatch: expected 'aclnnFakeOpV3', got '{pc.md_stem}' "
            f"(fill_back would produce [new]aclnnFakeOpV2_*.md — V2-bias)"
        )


# =============================================================================
# Test 2: V2/V3/V4 doc blocks differ (isolation)
# =============================================================================


class TestV2V3V4DocBlocksDiffer:
    """Isolation: V2/V3/V4 runs must produce different doc blocks.

    Runs AnchorPipeline 3 times with md_file=V2, V3, V4 respectively, into
    separate out_dirs. Asserts the 3 constraint.json doc blocks have:
      - Different doc_name values
      - Different md_format values
    """

    def test_v2_v3_v4_doc_blocks_differ(self, tmp_path):
        """Run AnchorPipeline 3 times (V2, V3, V4) -> doc blocks all differ."""
        op_path, data_root, _ = _make_multi_md_operator(tmp_path)

        docs = {}
        for version in ("V2", "V3", "V4"):
            md_file = f"aclnnFakeOp{version}.md"
            out_path = tmp_path / f"out_{version}"
            result = _run_constraint_pipeline(
                op_path, data_root, out_path, md_file=md_file
            )
            docs[version] = result["doc"]

        # --- Assertion 3a: doc_name values all different ---
        doc_names = {v: docs[v]["doc_name"] for v in ("V2", "V3", "V4")}
        assert doc_names["V2"] == "aclnnFakeOpV2.md"
        assert doc_names["V3"] == "aclnnFakeOpV3.md"
        assert doc_names["V4"] == "aclnnFakeOpV4.md"
        assert len(set(doc_names.values())) == 3, (
            f"doc_names not all different: {doc_names}"
        )

        # --- Assertion 3b: md_format values all different ---
        md_formats = {v: docs[v].get("md_format", [""])[0] for v in ("V2", "V3", "V4")}
        assert "V2_ONLY" in md_formats["V2"], f"V2 md_format wrong: {md_formats['V2']}"
        assert "V3_ONLY" in md_formats["V3"], f"V3 md_format wrong: {md_formats['V3']}"
        assert "V4_ONLY" in md_formats["V4"], f"V4 md_format wrong: {md_formats['V4']}"
        assert len(set(md_formats.values())) == 3, (
            f"md_format values not all different: {md_formats}"
        )


# =============================================================================
# Test 3: select_doc_file V3 not V2 (selector integration)
# =============================================================================


class TestSelectDocFileV3NotV2:
    """Selector integration: select_doc_file must return V3 for stem=V3.

    Verifies doc_selector.select_doc_file exact-match logic: when
    selected_md_stem="aclnnFakeOpV3", returns aclnnFakeOpV3.md (NOT V2).
    """

    def test_select_doc_file_v3_not_v2(self, tmp_path):
        """select_doc_file(doc_files, selected_md_stem='aclnnFakeOpV3') -> V3.md.

        Without exact-match: sorted[0] == aclnnFakeOpV2.md (FAIL).
        With exact-match: returns aclnnFakeOpV3.md (PASS).
        """
        op_path, _, _ = _make_multi_md_operator(tmp_path)
        docs_dir = op_path / "docs"
        doc_files = sorted(docs_dir.glob("aclnn*.md"), key=lambda f: f.name)

        # Sanity: sorted order is V2, V3, V4 (alphabetical)
        assert doc_files[0].name == "aclnnFakeOpV2.md", (
            f"Test setup error: expected V2 first, got {doc_files[0].name}"
        )

        selected = select_doc_file(doc_files, selected_md_stem="aclnnFakeOpV3")

        assert selected is not None, "select_doc_file returned None for V3"
        assert selected.name == "aclnnFakeOpV3.md", (
            f"Expected aclnnFakeOpV3.md, got {selected.name}"
        )
        assert selected.name != "aclnnFakeOpV2.md", (
            "V2-bias: selector returned V2 instead of V3"
        )


# =============================================================================
# Test 4: single-md backward compat
# =============================================================================


class TestSingleMdBackwardCompat:
    """Backward compat: single-md operator with md_file=None still works.

    Verifies the fix doesn't break existing single-md behavior: when
    md_file=None, the pipeline falls back to sorted(glob)[0] (the only md
    file) and produces correct output.
    """

    def test_single_md_backward_compat(self, tmp_path):
        """Single md file + md_file=None -> doc_name == that file (no error).

        With only 1 aclnn*.md file, glob[0] is deterministic (the only file).
        The pipeline should produce doc_name == aclnnFakeOpV2.md without error.
        """
        operator_path = tmp_path / "operator"
        data_root = tmp_path / "data"
        out_path = tmp_path / "out"

        # Create operator with only 1 md file
        docs_dir = operator_path / "docs"
        docs_dir.mkdir(parents=True)
        (docs_dir / "aclnnFakeOpV2.md").write_text(
            "# aclnnFakeOpV2\n\nSingle md content.", encoding="utf-8"
        )

        # Create op_api/ with minimal source
        op_api_dir = operator_path / "op_api"
        op_api_dir.mkdir()
        (op_api_dir / "aclnn_fake_op_v2.h").write_text(
            "#pragma once\n", encoding="utf-8"
        )
        (op_api_dir / "aclnn_fake_op_v2.cpp").write_text(
            "// minimal\n", encoding="utf-8"
        )

        # Create doc_json with 1 file
        doc_json_dir = data_root / "doc_json"
        doc_json_dir.mkdir(parents=True)
        (doc_json_dir / "aclnnFakeOpV2.json").write_text(
            json.dumps(
                {
                    "constraint": {
                        "origin_txt": "single md constraint text",
                        "constrain_clarify": "single clarify",
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        # Run with md_file=None (backward compat — glob fallback)
        result = _run_constraint_pipeline(
            operator_path, data_root, out_path, md_file=None
        )

        doc = result["doc"]
        assert doc["doc_name"] == "aclnnFakeOpV2.md", (
            f"Expected aclnnFakeOpV2.md (single-md fallback), got '{doc['doc_name']}'"
        )
        assert "single md constraint text" in doc.get("md_format", [""])[0], (
            f"md_format mismatch: {doc.get('md_format')}"
        )
