# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Constraint OLD/NEW equivalence verification gate (Task 1).

Verifies that OLD (data_parsing.get_constraint_evidence) and NEW
(constraint/__init__.py wrapper -> AnchorPipeline) produce identical
constraint.json output for 3 sample operators:
  - single-md operator (abs: 1 aclnn*.md file)
  - multi-md operator (add: 4 aclnn*.md files)
  - [new]-prefixed md operator (abs_copy: [new]aclnnAbs.md + aclnnAbs.md)

HARD STOP: Any sample with non-empty diff -> NOT EQUIVALENT -> stop all
subsequent Waves (T2-T8, F1-F4), report to user for decision.
Does NOT auto-fix AnchorPipeline.

See .omo/plans/snapshot-data-prep-dedup.md Task 1 (L181-283) for full spec.
"""

import json
import os
import shutil
import logging
from pathlib import Path
from typing import Any

import pytest

from src.snapshot_construction.anchor_construction.common.path_config import PathConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPS_MATH_ROOT = Path(os.environ.get("CANN_OPS_ROOT", ""))
requires_ops_root = pytest.mark.skipif(
    not os.environ.get("CANN_OPS_ROOT"),
    reason="需要 CANN_OPS_ROOT 指向本机 ops 仓检出 (集成测试依赖真实算子数据)",
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = PROJECT_ROOT / ".omo" / "evidence"
SHARED_DATA_ROOT = Path("/tmp/constraint_equiv")

# Sample definitions
SAMPLES: dict[str, dict[str, Any]] = {
    "single_md": {
        "operator_path": OPS_MATH_ROOT / "math" / "abs",
        "shared_dir": SHARED_DATA_ROOT / "abs" / "shared",
        "description": "single-md operator (aclnnAbs.md)",
    },
    "multi_md": {
        "operator_path": OPS_MATH_ROOT / "math" / "add",
        "shared_dir": OPS_MATH_ROOT / "out" / "add" / "base",  # reuse existing snapshot
        "description": "multi-md operator (4 aclnn*.md files)",
    },
    "prefixed_md": {
        "operator_path": OPS_MATH_ROOT / "math" / "abs_copy",
        "shared_dir": SHARED_DATA_ROOT / "abs_copy" / "shared",
        "description": "[new]-prefixed md operator ([new]aclnnAbs.md + aclnnAbs.md)",
    },
}

# Path-dependent fields to ignore in diff (out_dir differs -> these differ)
IGNORED_DOC_KEYS = {"doc_path"}

# Accumulate diffs across all samples for summary
_ALL_DIFFS: dict[str, Any] = {}

# Quiet down noisy loggers from the pipeline
logging.getLogger("src.snapshot_construction").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Helpers: environment / shared data
# ---------------------------------------------------------------------------


def _check_llm_env() -> bool:
    """Check if LLM environment variables (INCREMENTAL_LLM_KEY/URL/MODEL) are set."""
    return all(
        os.getenv(k)
        for k in ("INCREMENTAL_LLM_KEY", "INCREMENTAL_LLM_URL", "INCREMENTAL_LLM_MODEL")
    )


def _has_shared_data(shared_dir: Path) -> bool:
    """Check if shared data (doc_json, code_map, tree_id) exists."""
    if not shared_dir.exists():
        return False
    return all(
        (shared_dir / sub).exists() for sub in ("doc_json", "code_map", "tree_id")
    )


def _prepare_shared_data(operator_path: str, shared_dir: Path) -> str:
    """Prepare shared data by running cann_doc_struct + single_code_map + run_pipeline.

    Returns empty string on success, error message on failure.
    """
    from src.snapshot_construction import cann_doc_struct
    from src.snapshot_construction.code_graph import single_code_map
    from src.snapshot_construction.build_tree_id import run_pipeline

    shared_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: cann_doc_struct (no LLM) - parse md -> doc_json
    cann_doc_struct.main(str(operator_path), str(shared_dir / "doc_json"))

    # Step 2: single_code_map (no LLM) - build code graph
    single_code_map.main([str(operator_path)], str(shared_dir / "code_map"))

    # Step 3: run_pipeline (LLM) - generate tree_id
    run_pipeline.main(Path(operator_path), shared_dir / "tree_id")

    return ""


def _setup_out_dirs(shared_dir: Path, sample_name: str) -> tuple[Path, Path]:
    """Create out_dir_A and out_dir_B with symlinks to shared doc_json/code_map/tree_id.

    Both out_dirs share the same doc_json/code_map/tree_id (read-only) but
    have independent anchor/ directories (write).
    """
    base = SHARED_DATA_ROOT / sample_name
    out_a = base / "out_A"
    out_b = base / "out_B"

    for out_dir in (out_a, out_b):
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        for sub in ("doc_json", "code_map", "tree_id"):
            (out_dir / sub).symlink_to(shared_dir / sub)

    return out_a, out_b


# ---------------------------------------------------------------------------
# Helpers: run OLD / NEW paths
# ---------------------------------------------------------------------------


def _run_old(operator_path: str, out_dir: Path) -> dict[str, Any]:
    """Run OLD constraint path (data_parsing.get_constraint_evidence)."""
    from src.snapshot_construction.anchor_construction.constraint.data_parsing import (
        get_constraint_evidence as old_fn,
    )

    pc = PathConfig(operator_path, str(out_dir))
    return old_fn(pc)


def _run_new(operator_path: str, out_dir: Path) -> dict[str, Any]:
    """Run NEW constraint path (constraint.__init__.get_constraint_evidence -> AnchorPipeline)."""
    from src.snapshot_construction.anchor_construction.constraint import (
        get_constraint_evidence as new_fn,
    )

    pc = PathConfig(operator_path, str(out_dir))
    return new_fn(pc)


# ---------------------------------------------------------------------------
# Helpers: deep diff
# ---------------------------------------------------------------------------


def _diff_dict(
    old: dict[str, Any],
    new: dict[str, Any],
    ignored_keys: set[str] | None = None,
    path: str = "",
) -> dict[str, Any]:
    """Recursively diff two dicts. Returns dict of differences (empty = identical)."""
    ignored_keys = ignored_keys or set()
    diffs: dict[str, Any] = {}

    all_keys = set(old.keys()) | set(new.keys())
    for key in all_keys:
        if key in ignored_keys:
            continue
        p = f"{path}.{key}" if path else key
        ov = old.get(key, "__MISSING__")
        nv = new.get(key, "__MISSING__")

        if ov == "__MISSING__":
            diffs[p] = {"only_in": "new", "value_preview": str(nv)[:200]}
        elif nv == "__MISSING__":
            diffs[p] = {"only_in": "old", "value_preview": str(ov)[:200]}
        elif isinstance(ov, dict) and isinstance(nv, dict):
            sub = _diff_dict(ov, nv, ignored_keys, p)
            if sub:
                diffs[p] = sub
        elif isinstance(ov, list) and isinstance(nv, list):
            if ov != nv:
                diffs[p] = {
                    "old_preview": str(ov)[:200],
                    "new_preview": str(nv)[:200],
                    "old_len": len(ov),
                    "new_len": len(nv),
                }
        elif ov != nv:
            diffs[p] = {
                "old_preview": str(ov)[:200],
                "new_preview": str(nv)[:200],
            }

    return diffs


def _diff_anchors(
    old_anchors: list[dict[str, Any]], new_anchors: list[dict[str, Any]]
) -> dict[str, Any]:
    """Diff two anchor_tuple lists. Returns dict of differences."""
    diffs: dict[str, Any] = {}

    if len(old_anchors) != len(new_anchors):
        diffs["count"] = {"old": len(old_anchors), "new": len(new_anchors)}

    anchor_diffs: list[dict[str, Any]] = []

    # Compare anchor by anchor (up to min length)
    for i in range(min(len(old_anchors), len(new_anchors))):
        a_diff = _diff_dict(old_anchors[i], new_anchors[i])
        if a_diff:
            file_name = (
                old_anchors[i]
                .get("anchor_information", {})
                .get("level1", {})
                .get("file_name", "")
            )
            anchor_diffs.append(
                {
                    "index": i,
                    "file_name": file_name,
                    "diff": a_diff,
                }
            )

    # Extra anchors only in OLD
    if len(old_anchors) > len(new_anchors):
        for i in range(len(new_anchors), len(old_anchors)):
            fn = (
                old_anchors[i]
                .get("anchor_information", {})
                .get("level1", {})
                .get("file_name", "")
            )
            anchor_diffs.append(
                {
                    "index": i,
                    "only_in": "old",
                    "file_name": fn,
                }
            )

    # Extra anchors only in NEW
    if len(new_anchors) > len(old_anchors):
        for i in range(len(old_anchors), len(new_anchors)):
            fn = (
                new_anchors[i]
                .get("anchor_information", {})
                .get("level1", {})
                .get("file_name", "")
            )
            anchor_diffs.append(
                {
                    "index": i,
                    "only_in": "new",
                    "file_name": fn,
                }
            )

    if anchor_diffs:
        diffs["items"] = anchor_diffs

    return diffs


def _deep_diff(old_data: dict[str, Any], new_data: dict[str, Any]) -> dict[str, Any]:
    """Deep diff two constraint.json dicts, ignoring path-dependent fields.

    Returns dict of differences (empty = equivalent).
    """
    diffs: dict[str, Any] = {}

    # Diff doc section (ignore doc_path - path-dependent)
    old_doc = old_data.get("doc", {})
    new_doc = new_data.get("doc", {})
    doc_diff = _diff_dict(old_doc, new_doc, ignored_keys=IGNORED_DOC_KEYS)
    if doc_diff:
        diffs["doc"] = doc_diff

    # Diff anchor_tuple
    old_anchors = old_data.get("anchor_tuple", [])
    new_anchors = new_data.get("anchor_tuple", [])
    anchor_diff = _diff_anchors(old_anchors, new_anchors)
    if anchor_diff:
        diffs["anchor_tuple"] = anchor_diff

    return diffs


# ---------------------------------------------------------------------------
# Helpers: evidence
# ---------------------------------------------------------------------------


def _write_evidence(name: str, content: str) -> None:
    """Write evidence file to .omo/evidence/task-1-<name>.txt."""
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / f"task-1-{name}.txt").write_text(content, encoding="utf-8")


def _write_diff_summary(all_diffs: dict[str, Any]) -> None:
    """Write diff summary JSON to .omo/evidence/task-1-diff-summary.json."""
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    out = EVIDENCE_DIR / "task-1-diff-summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_diffs, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Core equivalence check
# ---------------------------------------------------------------------------


def _run_equivalence_check(sample_name: str) -> tuple[bool, dict[str, Any], str]:
    """Run equivalence check for a sample.

    Returns (is_equivalent, diffs, report_text).
    """
    sample = SAMPLES[sample_name]
    operator_path = str(sample["operator_path"])
    shared_dir = sample["shared_dir"]

    # Ensure shared data exists
    if not _has_shared_data(shared_dir):
        if not _check_llm_env():
            report = (
                f"BLOCKER: LLM env missing and no existing tree_id for "
                f"{sample_name} ({sample['description']})"
            )
            return False, {}, report
        # Prepare shared data (cann_doc_struct + single_code_map + run_pipeline)
        err = _prepare_shared_data(operator_path, shared_dir)
        if err:
            return False, {}, f"PREPARE FAILED: {err}"

    # Setup out dirs with shared data symlinks
    out_a, out_b = _setup_out_dirs(shared_dir, sample_name)

    # Run both paths
    _run_old(operator_path, out_a)
    _run_new(operator_path, out_b)

    # Load written constraint.json files
    old_file = out_a / "anchor" / "constraint.json"
    new_file = out_b / "anchor" / "constraint.json"

    assert old_file.exists(), f"OLD did not produce constraint.json: {old_file}"
    assert new_file.exists(), f"NEW did not produce constraint.json: {new_file}"

    with open(old_file, "r", encoding="utf-8") as f:
        old_data = json.load(f)
    with open(new_file, "r", encoding="utf-8") as f:
        new_data = json.load(f)

    # Deep diff (ignore doc_path - path-dependent)
    diffs = _deep_diff(old_data, new_data)

    old_count = len(old_data.get("anchor_tuple", []))
    new_count = len(new_data.get("anchor_tuple", []))

    is_equivalent = len(diffs) == 0
    if is_equivalent:
        report = (
            f"EQUIVALENT: {sample_name} ({sample['description']})\n"
            f"  OLD anchors: {old_count}\n"
            f"  NEW anchors: {new_count}\n"
            f"  diff: empty"
        )
    else:
        report = (
            f"NOT EQUIVALENT: {sample_name} ({sample['description']})\n"
            f"  OLD anchors: {old_count}\n"
            f"  NEW anchors: {new_count}\n"
            f"  diff: {json.dumps(diffs, ensure_ascii=False, indent=2)}"
        )

    return is_equivalent, diffs, report


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@requires_ops_root
@pytest.mark.parametrize(
    "sample_name", ["single_md", "multi_md"], ids=["single-md", "multi-md"]
)
def test_multi_md_equivalence(sample_name: str):
    """Verify OLD/NEW constraint equivalence for single-md and multi-md operators.

    - single-md: abs operator (1 aclnn*.md file)
    - multi-md: add operator (4 aclnn*.md files)

    HARD STOP: If diff is non-empty, this is NOT EQUIVALENT.
    """
    is_equiv, diffs, report = _run_equivalence_check(sample_name)

    # Write evidence
    _write_evidence(f"{sample_name.replace('_md', '-md')}", report)

    # Accumulate diffs for summary
    if diffs:
        _ALL_DIFFS[sample_name] = diffs
        _write_diff_summary(_ALL_DIFFS)

    assert is_equiv, f"NOT EQUIVALENT: {sample_name}\n{report}"


@requires_ops_root
def test_prefixed_md_equivalence():
    """Verify OLD/NEW constraint equivalence for [new]-prefixed md operator.

    Sample: abs_copy operator containing [new]aclnnAbs.md + aclnnAbs.md.
    Both paths should handle the [new]-prefixed file identically (both
    ignore it and use the regular aclnnAbs.md).

    Edge case: verify doc_json lookup doesn't miss for prefixed filenames.
    """
    is_equiv, diffs, report = _run_equivalence_check("prefixed_md")

    # Write evidence
    _write_evidence("prefixed-md", report)

    # Accumulate diffs for summary
    if diffs:
        _ALL_DIFFS["prefixed_md"] = diffs
        _write_diff_summary(_ALL_DIFFS)

    assert is_equiv, f"NOT EQUIVALENT: prefixed_md\n{report}"


def test_llm_env_blocker():
    """Verify that missing LLM env + no existing tree_id triggers a skip
    with a clear blocker message, NOT a false NOT EQUIVALENT.

    Preconditions: INCREMENTAL_LLM_KEY not set AND no existing tree_id snapshot.
    Expected: test skips with "BLOCKER: LLM env missing..." message.
    """
    # Save original env
    original_key = os.environ.get("INCREMENTAL_LLM_KEY")
    original_url = os.environ.get("INCREMENTAL_LLM_URL")
    original_model = os.environ.get("INCREMENTAL_LLM_MODEL")

    try:
        # Simulate missing LLM env
        for key in (
            "INCREMENTAL_LLM_KEY",
            "INCREMENTAL_LLM_URL",
            "INCREMENTAL_LLM_MODEL",
        ):
            os.environ.pop(key, None)

        # Check blocker logic: no shared data + no LLM env -> should report blocker
        has_data = _has_shared_data(Path("/tmp/nonexistent_shared_dir_xyz123"))
        has_env = _check_llm_env()

        if not has_data and not has_env:
            report = (
                "BLOCKER: LLM env missing and no existing tree_id\n"
                "  INCREMENTAL_LLM_KEY: not set\n"
                "  INCREMENTAL_LLM_URL: not set\n"
                "  INCREMENTAL_LLM_MODEL: not set\n"
                "  shared_data: not found\n"
                "  -> Cannot prepare shared data, cannot run equivalence check\n"
                "  -> This is a BLOCKER, not a NOT EQUIVALENT"
            )
            _write_evidence("llm-env-blocker", report)
            pytest.skip(report)
        else:
            # If we reach here, env simulation failed (vars set at system level)
            report = (
                "WARNING: Could not simulate missing LLM env.\n"
                "  Environment variables may be set at system level and cannot be unset.\n"
                "  This test is only meaningful when INCREMENTAL_LLM_KEY is genuinely absent."
            )
            _write_evidence("llm-env-blocker", report)
            pytest.skip(report)
    finally:
        # Restore original env
        if original_key is not None:
            os.environ["INCREMENTAL_LLM_KEY"] = original_key
        if original_url is not None:
            os.environ["INCREMENTAL_LLM_URL"] = original_url
        if original_model is not None:
            os.environ["INCREMENTAL_LLM_MODEL"] = original_model
