# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Version-mismatch anchor filter (post-expansion cleanup).

Removes file anchors whose filename version suffix (_vN before extension)
does not match the md_stem's version (VN suffix). Applied after
expand_file_deps to serve as a final safety net against version pollution
introduced by either source_files fallback or transitive include expansion.

Rules:
  - md_stem has version (e.g. aclnnAddV3 -> "3"): keep same-version + no-suffix
    files; remove other versions.
  - md_stem has no version (e.g. aclnnAdds): keep only no-suffix files;
    remove all versioned files.

Cleans up dangling references left by removed anchors:
  - evidence_id / evidence_ids in doc.sub_part
  - dep_to_anchor_ids / dep_from_anchor_ids in surviving anchors
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MD_STEM_VER = re.compile(r"V(\d+)$")
_FILE_VER = re.compile(r"_v(\d+)(?=\.[^.]+$)")


def _get_md_version(md_stem: str) -> str | None:
    m = _MD_STEM_VER.search(md_stem)
    return m.group(1) if m else None


def _get_file_version(file_name: str) -> str | None:
    m = _FILE_VER.search(file_name)
    return m.group(1) if m else None


def _should_keep(md_ver: str | None, file_ver: str | None) -> bool:
    if file_ver is None:
        return True
    if md_ver is None:
        return False
    return file_ver == md_ver


def filter_version_mismatch_anchors(anchor_dir: str | Path, md_stem: str) -> int:
    """Filter version-mismatched file anchors across all anchor JSONs.

    Args:
        anchor_dir: Directory containing the 5 anchor JSON files.
        md_stem: The md stem string (e.g. "aclnnAddV3", "aclnnAdds").

    Returns:
        Total number of file anchors removed.
    """
    anchor_dir = Path(anchor_dir)
    md_ver = _get_md_version(md_stem)
    total_removed = 0

    for json_path in sorted(anchor_dir.glob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        anchor_tuple = data.get("anchor_tuple", [])
        if not anchor_tuple:
            continue

        removed_aids: set[str] = set()
        removed_eids: set[str] = set()
        kept: list[dict[str, Any]] = []

        for anchor in anchor_tuple:
            info = anchor.get("anchor_information", {})
            if info.get("type") != "file":
                kept.append(anchor)
                continue
            file_name = info.get("level1", {}).get("file_name", "")
            file_ver = _get_file_version(file_name)
            if _should_keep(md_ver, file_ver):
                kept.append(anchor)
            else:
                removed_aids.add(anchor.get("anchor_id", ""))
                for ev in anchor.get("evidence", []):
                    removed_eids.add(ev.get("evidence_id", ""))

        if not removed_aids:
            continue

        total_removed += len(removed_aids)

        for anchor in kept:
            info = anchor.get("anchor_information", {})
            dto = info.get("dep_to_anchor_ids")
            if isinstance(dto, list):
                info["dep_to_anchor_ids"] = [x for x in dto if x not in removed_aids]
            dfrom = info.get("dep_from_anchor_ids")
            if isinstance(dfrom, list):
                info["dep_from_anchor_ids"] = [x for x in dfrom if x not in removed_aids]

        sub_part = data.get("doc", {}).get("sub_part", {})
        for _key, items in sub_part.items():
            for item in items:
                if not isinstance(item, dict):
                    continue
                for eid_key in ("evidence_id", "evidence_ids"):
                    eids = item.get(eid_key)
                    if isinstance(eids, list):
                        item[eid_key] = [e for e in eids if e not in removed_eids]

        data["anchor_tuple"] = kept
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")
        logger.info(
            f"version_filter: {json_path.name} removed {len(removed_aids)} "
            f"anchors (md_stem={md_stem}, md_ver={md_ver})"
        )

    return total_removed
