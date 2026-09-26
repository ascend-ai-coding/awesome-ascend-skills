#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# ----------------------------------------------------------------------------------------------------------
"""
generate_knowledge_index.py — 生成 evolution-knowledge skill 的 INDEX.json

不像 evolution-strategies 有 frontmatter，evolution-knowledge 是纯文档库，
INDEX.json 仅记录路径、kind、所属 category，方便 source_key 反查和 LLM 导航。
"""

import json
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)

SKILL_ROOT = Path("plugins-community/ops-perf-evolution/skills/evolution-knowledge/references")
INDEX_PATH = SKILL_ROOT / "INDEX.json"

SKILL = "evolution-knowledge"

KIND_DIRS = {
    "a3": SKILL_ROOT / "a3",
}


def discover():
    entries = []
    for kind, root in KIND_DIRS.items():
        if not root.exists():
            continue
        for md in sorted(root.rglob("*.md")):
            if md.name == "SOURCE_KEY.md":
                continue
            rel = md.relative_to(SKILL_ROOT).as_posix()
            # source_key 用不带后缀的相对路径作为标识
            key_path = rel[:-3]  # strip ".md"
            # 跳过 a3 顶层 INDEX.md 自身
            entry = {
                "source_key": f"{SKILL}#{key_path}",
                "kind": kind,
                "category": md.parent.name if md.parent != root else None,
                "is_index": md.name == "INDEX.md",
                "is_guide": md.name == "guide.md",
                "doc_path": rel,
            }
            entries.append(entry)
    return entries


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    entries = discover()
    by_kind = {}
    for e in entries:
        by_kind.setdefault(e["kind"], 0)
        by_kind[e["kind"]] += 1

    index = {
        "skill": SKILL,
        "version": "1.0",
        "architectures": ["a3"],
        "categories": {
            "a3": ["hardware", "algorithm_insights", "ascendc_api",
                   "optimization_patterns", "proven_solutions", "profiling_reference"],
        },
        "stats": {**by_kind, "total": len(entries)},
        "entries": entries,
    }
    INDEX_PATH.write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Wrote %s: total=%d (%s)", INDEX_PATH, len(entries), by_kind)


if __name__ == "__main__":
    main()
