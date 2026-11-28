# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Dry-run naming convention validation script (Task 3).

Standalone — uses ONLY pathlib + re + json (no src/ imports).

Validates the naming convention rules that will be used by
resolve_md_to_source_files (Task 4) against real operator data in
ops-math/math/. This is a RESEARCH script: it does NOT modify any
pipeline code.

Naming convention rules applied per aclnn*.md filename:
  a. Strip prefix markers: remove [new], [deprecated], or any [...] prefix
  b. Split on '&': aclnnAddV3&aclnnInplaceAddV3.md -> [aclnnAddV3, aclnnInplaceAddV3]
  c. camelCase -> snake_case: aclnnAddV3 -> aclnn_add_v3
     (insert '_' before each uppercase except first char, lowercase all)
  d. Match source files: in op_api/ (or op_host/op_api/), find aclnn_<snake>.cpp
     and aclnn_<snake>.h
  e. Collect shared files: *_proto.h, *def*.cpp/h, *tiling*.cpp/h, op_kernel/*.cpp/h
     (replicates file_discovery.py find_operator_files patterns)

Fallback: if no naming-matched files found, fall back to ALL aclnn_*.cpp and
aclnn_*.h in op_api (D5 naming fallback).

Output: JSON report + edge case summary + match rate.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPS_ROOT = Path(os.environ.get("CANN_OPS_ROOT", "")) / "math"
EVIDENCE_DIR = Path(os.environ.get("CANN_EVIDENCE_DIR", ".omo/evidence"))
REPORT_PATH = EVIDENCE_DIR / "task-3-naming-validation-report.json"
EDGE_CASES_PATH = EVIDENCE_DIR / "task-3-edge-cases.txt"

# Operators explicitly required by the task + extras with multiple md files
# to reach 10+ operators.
REQUIRED_OPERATORS = [
    "add",
    "bitwise_and",
    "bitwise_or",
    "cumsum",
    "div",
    "abs_copy",
    "abs",
    "bitwise_xor",
    "sub",
    "pow",
    "equal",
    "reduce_min",
]

# Shared file category patterns (mirrors file_discovery.py:47-89 patterns list).
# Each entry: (search_dir_relative_to_operator, glob_pattern, tag)
SHARED_PATTERNS = [
    ("op_graph", "*_proto.h", "*_proto.h"),
    ("op_host", "*def*.cpp", "*def*.cpp"),
    ("op_host", "*def*.h", "*def*.h"),
    ("op_host", "*tiling*.cpp", "*tiling*.cpp"),
    ("op_host", "*tiling*.h", "*tiling*.h"),
    ("op_tiling", "*.cpp", "op_tiling/*.cpp"),
    ("op_tiling", "*.h", "op_tiling/*.h"),
    ("op_host/op_tiling", "*.cpp", "op_tiling/*.cpp"),
    ("op_host/op_tiling", "*.h", "op_tiling/*.h"),
    ("op_kernel", "*.cpp", "op_kernel/*.cpp"),
    ("op_kernel", "*.h", "op_kernel/*.h"),
]


# ---------------------------------------------------------------------------
# Naming convention helpers
# ---------------------------------------------------------------------------


def strip_prefix_markers(filename: str) -> str:
    """Rule (a): remove [new], [deprecated], or any [...] prefix markers.

    Also strips the .md extension. Handles multiple bracket prefixes and
    leading/trailing whitespace.
    """
    stem = filename
    # Remove .md extension if present
    if stem.endswith(".md"):
        stem = stem[: -len(".md")]
    # Repeatedly strip leading [....] markers (e.g. [new], [deprecated])
    # These appear at the very start of the filename.
    while True:
        m = re.match(r"^\s*\[([^\]]*)\]\s*(.*)", stem)
        if m:
            stem = m.group(2)
        else:
            break
    return stem


def split_on_amp(stem: str) -> list[str]:
    """Rule (b): split on '&' into multiple API names.

    aclnnAddV3&aclnnInplaceAddV3 -> [aclnnAddV3, aclnnInplaceAddV3]
    """
    return [part.strip() for part in stem.split("&") if part.strip()]


def camel_to_snake(name: str) -> str:
    """Rule (c): camelCase -> snake_case.

    Insert '_' before each uppercase letter except the first character,
    then lowercase everything.

    aclnnAddV3 -> aclnn_add_v3
    aclnnInplaceAdd -> aclnn_inplace_add
    aclnnBitwiseAndScalar -> aclnn_bitwise_and_scalar
    aclnnIsInScalarTensor -> aclnn_is_in_scalar_tensor
    """
    result = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0:
            result.append("_")
        result.append(ch.lower())
    return "".join(result)


def extract_api_names(md_filename: str) -> list[str]:
    """Apply rules (a)+(b): strip markers, split on '&'.

    Returns the list of camelCase API names (NOT yet snake_cased).
    e.g. '[new]aclnnAbs.md' -> ['aclnnAbs']
         'aclnnAddV3&aclnnInplaceAddV3.md' -> ['aclnnAddV3', 'aclnnInplaceAddV3']
    """
    stem = strip_prefix_markers(md_filename)
    return split_on_amp(stem)


# ---------------------------------------------------------------------------
# File discovery helpers (standalone, mirrors file_discovery.py logic)
# ---------------------------------------------------------------------------


def find_op_api_dir(operator_path: Path) -> Path | None:
    """Find op_api/ directory: operator/op_api or operator/op_host/op_api."""
    candidate = operator_path / "op_api"
    if candidate.exists() and candidate.is_dir():
        return candidate
    candidate = operator_path / "op_host" / "op_api"
    if candidate.exists() and candidate.is_dir():
        return candidate
    return None


def find_naming_matched_files(op_api_dir: Path, snake_names: list[str]) -> list[Path]:
    """Rule (d): find <snake>.cpp and <snake>.h in op_api.

    The snake_names already include the 'aclnn_' prefix from camelCase
    conversion (e.g. aclnnAddV3 -> aclnn_add_v3), so we look for
    '{snake}.cpp' and '{snake}.h' directly.
    """
    matched = []
    for snake in snake_names:
        for ext in (".cpp", ".h"):
            candidate = op_api_dir / f"{snake}{ext}"
            if candidate.exists():
                matched.append(candidate)
    return matched


def find_fallback_files(op_api_dir: Path) -> list[Path]:
    """Rule (fallback): ALL aclnn_*.cpp and aclnn_*.h in op_api (D5)."""
    fallback = []
    fallback.extend(sorted(op_api_dir.glob("aclnn_*.cpp")))
    fallback.extend(sorted(op_api_dir.glob("aclnn_*.h")))
    return fallback


def find_shared_files(operator_path: Path) -> list[tuple[Path, str]]:
    """Rule (e): collect shared operator files (proto/def/tiling/kernel).

    Mirrors file_discovery.py find_operator_files patterns (lines 59-73),
    EXCLUDING the aclnn_*.h / aclnn_*.cpp op_api entries (those are handled
    by naming match / fallback). Shared files belong to ALL mds.
    """
    files_with_source = []
    for rel_dir, pattern, tag in SHARED_PATTERNS:
        search_dir = operator_path / rel_dir
        if search_dir.exists() and search_dir.is_dir():
            for f in search_dir.rglob(pattern):
                files_with_source.append((f, tag))

    # Dedup by Path
    seen: set[Path] = set()
    unique: list[tuple[Path, str]] = []
    for f, source in files_with_source:
        if f not in seen:
            seen.add(f)
            unique.append((f, source))
    return unique


# ---------------------------------------------------------------------------
# Edge case detection
# ---------------------------------------------------------------------------


def detect_edge_cases(
    md_filename: str,
    api_names: list[str],
    matched_files: list[Path],
    fallback_used: bool,
) -> list[str]:
    """Classify edge cases for a given md file."""
    edges: list[str] = []

    # [new] / [deprecated] prefix marker
    if re.match(r"^\s*\[", md_filename):
        edges.append("prefix_marker")

    # Multi-API (& split)
    if "&" in strip_prefix_markers(md_filename):
        edges.append("amp_multi_api")

    # V2/V3 in API name (version-like suffix that is actually part of API name)
    for name in api_names:
        if re.search(r"V\d+$", name):
            edges.append("version_suffix_in_api_name")
            break

    # No naming match -> fallback used
    if fallback_used:
        edges.append("no_naming_match_fallback")

    # Zero matched files even after fallback (operator has no op_api at all)
    if not matched_files and not fallback_used:
        edges.append("zero_files_no_op_api")
    elif not matched_files and fallback_used:
        # fallback found nothing either
        edges.append("zero_files_fallback_empty")

    return edges


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------


def validate_operator(operator_path: Path) -> dict[str, Any]:
    """Validate naming convention for one operator.

    Returns: {operator: {md_file: {api_names, snake_names, matched_files,
                                    fallback_used, shared_files, edge_cases}}}
    """
    docs_dir = operator_path / "docs"
    op_api_dir = find_op_api_dir(operator_path)
    shared_files = find_shared_files(operator_path)
    shared_paths = [str(f.relative_to(operator_path)) for f, _ in shared_files]

    result: dict[str, Any] = {}

    # Discover md files: standard aclnn*.md PLUS prefix-marked files like
    # [new]aclnnAbs.md, [deprecated]aclnnXxx.md (glob doesn't match leading [).
    md_files: list[Path] = []
    if docs_dir.exists():
        md_files = sorted(docs_dir.glob("aclnn*.md"))
        # Also catch [....]aclnn*.md prefix-marked files.
        for f in docs_dir.glob("*.md"):
            if re.match(r"^\[.*\]aclnn", f.name) and f not in md_files:
                md_files.append(f)
        md_files.sort()
    for md_file in md_files:
        md_name = md_file.name
        api_names = extract_api_names(md_name)
        snake_names = [camel_to_snake(n) for n in api_names]

        matched: list[Path] = []
        fallback_used = False

        if op_api_dir is not None:
            matched = find_naming_matched_files(op_api_dir, snake_names)
            if not matched:
                fallback_used = True
                matched = find_fallback_files(op_api_dir)

        matched_rel = [str(f.relative_to(operator_path)) for f in matched]
        edges = detect_edge_cases(md_name, api_names, matched, fallback_used)

        result[md_name] = {
            "api_names": api_names,
            "snake_names": snake_names,
            "matched_files": matched_rel,
            "matched_count": len(matched_rel),
            "fallback_used": fallback_used,
            "shared_files": shared_paths,
            "shared_count": len(shared_paths),
            "edge_cases": edges,
            "has_match": len(matched_rel) > 0,
        }

    return result


def run_validation() -> dict[str, Any]:
    """Run validation on all required operators + any with multiple md files.

    Returns the full report dict.
    """
    report: dict[str, Any] = {}

    # Build operator list: required operators first, then any others with 2+ md
    # files discovered in ops-math/math/ (to enrich edge case coverage).
    discovered: list[str] = []
    if OPS_ROOT.exists():
        for d in sorted(OPS_ROOT.iterdir()):
            if d.is_dir():
                docs = d / "docs"
                if docs.is_dir():
                    mds = list(docs.glob("aclnn*.md"))
                    if len(mds) >= 2 and d.name not in REQUIRED_OPERATORS:
                        discovered.append(d.name)

    # Prioritize required operators; add a few discovered ones for breadth.
    operator_names = list(REQUIRED_OPERATORS) + discovered[:4]

    for op_name in operator_names:
        op_path = OPS_ROOT / op_name
        if not op_path.exists():
            report[op_name] = {"error": f"operator path not found: {op_path}"}
            continue
        report[op_name] = validate_operator(op_path)

    return report


def compute_match_rate(report: dict[str, Any]) -> dict[str, Any]:
    """Compute match rates across all md files in the report."""
    total_md = 0
    matched_md = 0  # md files with >=1 matched source file (naming OR fallback)
    naming_matched_md = 0  # md files with >=1 naming-matched file (no fallback)
    fallback_md = 0
    edge_counts: dict[str, int] = {}

    for op_name, op_data in report.items():
        if op_name == "_stats" or "error" in op_data:
            continue
        for md_name, md_data in op_data.items():
            total_md += 1
            if md_data.get("has_match"):
                matched_md += 1
            if not md_data.get("fallback_used") and md_data.get("matched_count", 0) > 0:
                naming_matched_md += 1
            if md_data.get("fallback_used"):
                fallback_md += 1
            for edge in md_data.get("edge_cases", []):
                edge_counts[edge] = edge_counts.get(edge, 0) + 1

    overall_rate = (matched_md / total_md * 100) if total_md else 0.0
    naming_rate = (naming_matched_md / total_md * 100) if total_md else 0.0

    return {
        "total_md_files": total_md,
        "matched_md_files": matched_md,
        "naming_matched_md_files": naming_matched_md,
        "fallback_md_files": fallback_md,
        "overall_match_rate_pct": round(overall_rate, 2),
        "naming_only_match_rate_pct": round(naming_rate, 2),
        "edge_case_counts": edge_counts,
    }


def build_edge_case_summary(report: dict[str, Any], stats: dict[str, Any]) -> str:
    """Build a human-readable edge case summary."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("Task 3 — Naming Convention Validation: Edge Case Summary")
    lines.append("=" * 72)
    lines.append("")

    # Overall stats
    lines.append("OVERALL STATISTICS")
    lines.append("-" * 40)
    lines.append(f"Total md files evaluated:  {stats['total_md_files']}")
    lines.append(
        f"Matched (naming+fallback): {stats['matched_md_files']} "
        f"({stats['overall_match_rate_pct']}%)"
    )
    lines.append(
        f"Naming-only matched:      {stats['naming_matched_md_files']} "
        f"({stats['naming_only_match_rate_pct']}%)"
    )
    lines.append(f"Fallback used:            {stats['fallback_md_files']}")
    lines.append("")
    lines.append("EDGE CASE COUNTS")
    lines.append("-" * 40)
    for edge, count in sorted(stats["edge_case_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"  {edge:40s} {count}")
    lines.append("")

    # Per-edge-case examples
    lines.append("EDGE CASE EXAMPLES (by category)")
    lines.append("=" * 72)

    categories: dict[str, list[str]] = {}
    for op_name, op_data in report.items():
        if op_name == "_stats" or "error" in op_data:
            continue
        for md_name, md_data in op_data.items():
            for edge in md_data.get("edge_cases", []):
                categories.setdefault(edge, []).append(
                    f"  [{op_name}] {md_name}"
                    f" -> api={md_data['api_names']}"
                    f" snake={md_data['snake_names']}"
                    f" matched={md_data['matched_count']}"
                    f" fallback={md_data['fallback_used']}"
                )

    for edge in sorted(categories):
        lines.append(f"\n--- {edge} ---")
        for line in categories[edge]:
            lines.append(line)

    lines.append("")
    lines.append("=" * 72)
    lines.append("KEY FINDINGS")
    lines.append("=" * 72)
    lines.append("")
    lines.append("1. PREFIX MARKERS ([new], [deprecated]):")
    lines.append("   Stripping leading [...] from filename works correctly.")
    lines.append("   Example: abs_copy/[new]aclnnAbs.md -> aclnnAbs -> aclnn_abs")
    lines.append("")
    lines.append("2. AMP MULTI-API (& split):")
    lines.append("   Splitting on '&' yields multiple API names per md file.")
    lines.append("   Each name is independently snake_cased and matched.")
    lines.append("   Example: add/aclnnAddV3&aclnnInplaceAddV3.md")
    lines.append("            -> [aclnn_add_v3, aclnn_inplace_add_v3]")
    lines.append("")
    lines.append("3. VERSION SUFFIX (V2/V3 in API name):")
    lines.append("   V2/V3 is part of the API name, NOT a version folder.")
    lines.append("   camelCase->snake_case correctly produces _v2/_v3 suffix.")
    lines.append("   Example: cumsum/aclnnCumsumV2.md -> aclnn_cumsum_v2")
    lines.append("")
    lines.append("4. NO-NAMING-MATCH FALLBACK (D5):")
    lines.append("   When no aclnn_<snake>.cpp/h exists, fall back to ALL")
    lines.append("   aclnn_*.cpp/h in op_api. Common for scalar/inplace variants")
    lines.append("   whose source is folded into the base aclnn_<op>.cpp.")
    lines.append("   Example: add/aclnnAdds.md -> aclnn_adds (no file)")
    lines.append("            -> fallback to aclnn_add.cpp, aclnn_add_v3.cpp")
    lines.append("")
    lines.append("5. NAMING CONVENTION MISMATCH (IsIn vs isin):")
    lines.append("   camelCase->snake_case produces aclnn_is_in_* but source")
    lines.append("   files use aclnn_isin_* (no underscore between is/in).")
    lines.append("   This is a genuine convention mismatch requiring fallback.")
    lines.append("   Example: equal/aclnnIsInScalarTensor.md")
    lines.append("            -> aclnn_is_in_scalar_tensor (no match)")
    lines.append("            source: aclnn_isin_scalar_tensor.cpp")
    lines.append("")
    lines.append("6. SHARED FILES (proto/def/tiling/kernel):")
    lines.append("   Shared operator files belong to ALL mds in the operator.")
    lines.append("   Replicates current version-folder base fallback behavior.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Run validation, write report + edge case summary, print summary."""
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    report = run_validation()
    stats = compute_match_rate(report)

    # Attach stats to report
    report["_stats"] = stats

    # Write JSON report
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Write edge case summary
    edge_summary = build_edge_case_summary(report, stats)
    EDGE_CASES_PATH.write_text(edge_summary, encoding="utf-8")

    # Print console summary
    print("=" * 60)
    print("Task 3 — Naming Convention Dry-Run Validation")
    print("=" * 60)
    print(f"Operators evaluated: {len(report) - 1}")  # minus _stats
    print(f"Total md files:     {stats['total_md_files']}")
    print(
        f"Matched (overall):  {stats['matched_md_files']} "
        f"({stats['overall_match_rate_pct']}%)"
    )
    print(
        f"Naming-only match:  {stats['naming_matched_md_files']} "
        f"({stats['naming_only_match_rate_pct']}%)"
    )
    print(f"Fallback used:      {stats['fallback_md_files']}")
    print()
    print("Edge case counts:")
    for edge, count in sorted(stats["edge_case_counts"].items(), key=lambda x: -x[1]):
        print(f"  {edge:40s} {count}")
    print()
    print(f"Report written to:      {REPORT_PATH}")
    print(f"Edge cases written to:  {EDGE_CASES_PATH}")

    # Verification gate
    if stats["overall_match_rate_pct"] >= 90.0:
        print("\nPASS: Overall match rate >= 90%")
    else:
        print(f"\nWARN: Overall match rate {stats['overall_match_rate_pct']}% < 90%")


if __name__ == "__main__":
    main()
