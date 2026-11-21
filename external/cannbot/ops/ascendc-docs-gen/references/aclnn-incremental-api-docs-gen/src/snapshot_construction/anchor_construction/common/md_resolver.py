# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
MD-to-source-file resolver.

Resolves an aclnn*.md documentation filename to the set of source files
that belong to it, using the VALIDATED naming convention (Task 3 dry-run:
49 md files, 16 operators, 100% match rate with fallback).

Naming convention rules (HARDCODED — not pluggable/configurable):
  1. Strip prefix markers: remove [new], [deprecated], or any [...] prefix
  2. Split on &: aclnnAddV3&aclnnInplaceAddV3 -> [aclnnAddV3, aclnnInplaceAddV3]
  3. Normalize: lowercase + remove underscores (no camelCase splitting)
  4. Two-layer match: exact normalized equality first, then substring
     containment (file_norm in api_norm) excluding _vN version files
  5. Match in op_api/ (or op_host/op_api/ if op_api doesn't exist)
  6. Collect shared files: *_proto.h, *def*.cpp/h, *tiling*.cpp/h, op_kernel/*.cpp/h
     (from file_discovery.py find_operator_files patterns, excluding aclnn_ entries)
  7. Fallback: if no naming-matched files found, return ALL aclnn_*.cpp/h in op_api
  8. Return list[Path] (deduplicated)
"""

import re
from pathlib import Path

from src.snapshot_construction.anchor_construction.common.file_discovery import (
    find_operator_files,
)


def _strip_prefix_markers(stem: str) -> str:
    """Rule 1: remove [new], [deprecated], or any [...] prefix markers.

    Repeatedly strips leading [....] markers. Also handles multiple
    consecutive brackets and leading/trailing whitespace.
    """
    while True:
        m = re.match(r"^\s*\[([^\]]*)\]\s*(.*)", stem)
        if m:
            stem = m.group(2)
        else:
            break
    return stem


def _split_on_amp(stem: str) -> list[str]:
    """Rule 2: split on & into multiple API names.

    aclnnAddV3&aclnnInplaceAddV3 -> [aclnnAddV3, aclnnInplaceAddV3]
    """
    return [part.strip() for part in stem.split("&") if part.strip()]


_VERSION_RE = re.compile(r"_v\d+$", re.IGNORECASE)


def _normalize(name: str) -> str:
    """Normalize name: lowercase + remove underscores.

    Replaces the old camelCase->snake_case conversion. Instead of splitting
    on uppercase boundaries (which breaks on acronyms like FFN->f_f_n and
    merged words like ArgMax->arg_max), we simply lowercase and strip
    underscores. Matching is then done by normalized equality (Layer 1)
    or substring containment (Layer 2), which naturally handles:
      - merged words: aclnnArgMax -> aclnnargmax == aclnn_argmax
      - acronyms: aclnnFFN -> aclnnffn == aclnn_ffn
      - scalar variants: aclnnAdds -> aclnnadds contains aclnnadd (aclnn_add)
      - inplace variants: aclnnInplaceAdd -> aclnninplaceadd contains aclnnadd
      - version sharing: aclnnCumsumV2 -> aclnncumsumv2 contains aclnncumsum
    """
    return name.lower().replace("_", "")


def _find_op_api_dir(operator_path: Path) -> Path | None:
    """Rule 5: find op_api directory.

    Checks operator/op_api first, then operator/op_host/op_api.
    Returns None if neither exists.
    """
    candidate = operator_path / "op_api"
    if candidate.exists():
        return candidate
    candidate = operator_path / "op_host" / "op_api"
    if candidate.exists():
        return candidate
    return None


def resolve_md_to_source_files(md_path: Path, operator_path: Path) -> list[Path]:
    """Resolve md filename to source files.

    Uses the VALIDATED naming convention (Task 3 dry-run) to map an aclnn*.md
    documentation file to its corresponding source files.

    Args:
        md_path: Path to the .md file (only .name is used for resolution)
        operator_path: Path to the operator directory containing op_api/,
            op_graph/, op_host/, op_kernel/, etc.

    Returns:
        Deduplicated list of source file paths. Includes:
        - Naming-matched aclnn_*.cpp/h from op_api/ (or fallback to ALL
          aclnn_*.cpp/h if no naming match)
        - Shared operator files (proto/def/tiling/kernel) from find_operator_files
    """
    # Rule 1: strip .md extension, then strip prefix markers
    stem = md_path.name
    if stem.endswith(".md"):
        stem = stem[: -len(".md")]
    stem = _strip_prefix_markers(stem)

    # Rule 2: split on &
    api_names = _split_on_amp(stem)

    # Rule 3: normalize API names (lowercase + strip underscores, no camelCase split)
    # Rule 4-5: match source files in op_api via two-layer normalized matching
    op_api_dir = _find_op_api_dir(operator_path)

    matched: list[Path] = []
    if op_api_dir is not None:
        # Collect all aclnn_*.h and aclnn_*.cpp in op_api once
        aclnn_files = sorted(op_api_dir.glob("aclnn_*.h")) + sorted(
            op_api_dir.glob("aclnn_*.cpp")
        )

        for api_name in api_names:
            api_norm = _normalize(api_name)
            # Layer 1: exact normalized match (handles merged words / acronyms)
            hits = [f for f in aclnn_files if _normalize(f.stem) == api_norm]
            if not hits:
                # Layer 2: substring match (file_norm contained in api_norm)
                #   + exclude other-version files (_vN suffix) to avoid version pollution
                hits = [
                    f
                    for f in aclnn_files
                    if _normalize(f.stem) in api_norm and not _VERSION_RE.search(f.stem)
                ]
            if hits:
                matched.extend(hits)

        # Rule 7: fallback if no naming-matched files found across all APIs
        if not matched:
            matched = sorted(op_api_dir.glob("aclnn_*.cpp"))
            matched.extend(sorted(op_api_dir.glob("aclnn_*.h")))

    # Rule 6: collect shared files (proto/def/tiling/kernel)
    # find_operator_files returns list[(Path, source_tag)].
    # Shared files = everything EXCEPT aclnn_*.h and aclnn_*.cpp (those are
    # handled by naming match / fallback above).
    all_files = find_operator_files(operator_path)
    shared = [f for f, tag in all_files if tag not in ("aclnn_*.h", "aclnn_*.cpp")]

    # Rule 8: combine and deduplicate
    result: list[Path] = []
    seen: set[Path] = set()
    for f in matched + shared:
        if f not in seen:
            seen.add(f)
            result.append(f)

    return result
