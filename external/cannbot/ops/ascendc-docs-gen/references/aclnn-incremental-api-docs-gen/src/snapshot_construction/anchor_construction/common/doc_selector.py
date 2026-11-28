# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Stem-aware document file selector.

Shared utility for selecting an aclnn*.md documentation file from a list of
candidate paths. Replaces the 3 duplicated ``_auto_select_doc_file`` copies in
the CLI orchestrators (Task 6/7/8 will wire them in).

Selection rules (HARDCODED — no pluggable heuristics):
  1. Filter to ``aclnn*`` files (non-aclnn files are ignored entirely).
  2. If ``selected_md_stem`` is non-empty: return the file whose ``f.stem``
     matches it EXACTLY. No fuzzy/prefix/case folding. No fallback to first
     if the explicit stem is absent (avoids reintroducing V2-bias).
  3. If ``selected_md_stem`` is None or empty: return the alphabetically-first
     ``aclnn*.md`` file (backward-compat with the old "take first" behavior).
  4. Empty input or no aclnn files -> None.
"""

from pathlib import Path
from typing import Optional


def select_doc_file(
    doc_files: list[Path], selected_md_stem: str | None = None
) -> Optional[Path]:
    """Select a doc file by explicit stem, falling back to first aclnn*.md.

    Args:
        doc_files: List of .md file paths to select from.
        selected_md_stem: Optional stem to match exactly
            (e.g., "aclnnMoeDistributeDispatchV3"). When None or empty,
            returns the first aclnn*.md file (backward compat).

    Returns:
        Matching Path, or None if no match found.
    """
    # Rule 1: only consider aclnn* files (same filter as _auto_select_doc_file).
    default_files = [f for f in doc_files if f.name.startswith("aclnn")]
    if not default_files:
        return None

    # Rule 3: None/empty stem -> backward-compat "take first" (sorted for determinism).
    if not selected_md_stem:
        return sorted(default_files, key=lambda f: f.name)[0]

    # Rule 2: exact stem match within aclnn* files. No fallback to first.
    for f in default_files:
        if f.stem == selected_md_stem:
            return f

    return None
