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
generate_index_json.py — 从策略卡 frontmatter 程序化生成 INDEX.json

INDEX.json 是 machine-parseable 的 source_key 权威清单，用途：
- bottleneck_to_sources.py 按 triggers.bottleneck_labels 反查策略
- 防偷懒 audit 检查 source_key 合法性
- partial-prompt 注入时快速定位资源路径

输出位置：plugins-community/ops-perf-evolution/skills/evolution-strategies/references/INDEX.json
"""

from __future__ import annotations
import json
import logging
import re
from pathlib import Path

LOGGER = logging.getLogger(__name__)

SKILL_ROOT = Path("plugins-community/ops-perf-evolution/skills/evolution-strategies/references")
CARDS_DIR = SKILL_ROOT / "cards"
PRECOND_DIR = SKILL_ROOT / "preconditions"
PLAYBOOK_DIR = SKILL_ROOT / "playbooks"
FAMILIES_DIR = SKILL_ROOT / "families"
DISCOVERED_DIR = SKILL_ROOT / "discovered"
INDEX_PATH = SKILL_ROOT / "INDEX.json"

SKILL = "evolution-strategies"


def split_frontmatter(text: str):
    if not text.startswith("---\n"):
        return None, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return None, text
    return text[4:end], text[end + 5:]


def parse_yaml_frontmatter(fm: str) -> dict:
    """轻量 YAML 解析器，仅支持本仓 frontmatter 用到的子集。"""
    result = {}
    current_key = None
    current_list = None
    i = 0
    lines = fm.splitlines()
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        # 嵌套列表项（以 "- " 或 "  - " 开头的行）
        if line.startswith(("- ", "  - ")) and current_list is not None:
            current_list.append(stripped[2:].strip())
            i += 1
            continue

        # 匹配 key 与 value 的分隔行
        m = re.match(r"^([a-zA-Z_]+):\s*(.*)$", line)
        if not m:
            i += 1
            continue
        key, val = m.group(1), m.group(2).strip()

        if val == "":
            # 可能是后续缩进列表
            current_key = key
            current_list = []
            result[key] = current_list
        elif val.startswith("[") and val.endswith("]"):
            # 行内列表
            inner = val[1:-1].strip()
            if not inner:
                result[key] = []
            else:
                result[key] = [item.strip() for item in inner.split(",")]
            current_key = None
            current_list = None
        elif val.lower() in ("true", "false"):
            result[key] = val.lower() == "true"
            current_key = None
            current_list = None
        else:
            result[key] = val.strip('"').strip("'")
            current_key = None
            current_list = None
        i += 1
    return result


def discover_cards() -> list[dict]:
    entries = []
    for card_path in sorted(CARDS_DIR.glob("*.md")):
        if card_path.name in {"CONTRIBUTING.md", "SCHEMA.md"}:
            continue
        text = card_path.read_text(encoding="utf-8")
        fm, _ = split_frontmatter(text)
        if not fm:
            continue
        meta = parse_yaml_frontmatter(fm)
        sid = meta.get("id", "")
        slug = card_path.stem  # e.g. "P1_double_buffer"
        # 提取标题（# 之后的第一行）
        body_first_h1 = re.search(r"^#\s+(.+?)$", text, re.MULTILINE)
        title = body_first_h1.group(1).strip() if body_first_h1 else slug

        entry = {
            "source_key": f"{SKILL}#card/{slug}",
            "id": sid,
            "title": title,
            "doc_path": f"cards/{card_path.name}",
            "complexity": meta.get("complexity", ""),
            "op_families": meta.get("op_families", []),
            "triggers": {
                "bottleneck_labels": meta.get("bottlenecks", []),
            },
            "conflicts_with": meta.get("conflicts_with", []),
            "synergizes_with": meta.get("synergizes_with", []),
            "requires": meta.get("requires", []),
            "has_preconditions": meta.get("has_preconditions", False),
            "has_playbook": meta.get("has_playbook", False),
            "quantified_gain": meta.get("quantified_gain", []),
        }
        if meta.get("deprecated", False):
            entry["deprecated"] = True
            entry["deprecated_reason"] = meta.get("deprecated_reason", "")
        entries.append(entry)
    return entries


def discover_preconditions() -> list[dict]:
    entries = []
    for p in sorted(PRECOND_DIR.glob("*.yaml")):
        sid = p.stem
        entries.append({
            "source_key": f"{SKILL}#preconditions/{sid}",
            "id": sid,
            "doc_path": f"preconditions/{p.name}",
        })
    return entries


def discover_playbooks() -> list[dict]:
    entries = []
    for p in sorted(PLAYBOOK_DIR.glob("*.md")):
        if p.name == "SCHEMA.md":
            continue
        slug = p.stem
        sid = slug.split("_", 1)[0] if "_" in slug else slug
        entries.append({
            "source_key": f"{SKILL}#playbook/{slug}",
            "id": sid,
            "doc_path": f"playbooks/{p.name}",
        })
    return entries


def discover_families() -> list[dict]:
    entries = []
    if not FAMILIES_DIR.exists():
        return entries
    for p in sorted(FAMILIES_DIR.glob("*.md")):
        slug = p.stem  # e.g. "matmul_guide"
        entries.append({
            "source_key": f"{SKILL}#family/{slug}",
            "title": slug.replace("_", " ").title(),
            "doc_path": f"families/{p.name}",
        })
    return entries


def discover_discovered() -> list[dict]:
    entries = []
    if not DISCOVERED_DIR.exists():
        return entries
    for p in sorted(DISCOVERED_DIR.glob("disc_*.md")):
        slug = p.stem
        sid = slug.split("_", 1)[1] if "_" in slug else slug  # disc_X1 → X1
        entries.append({
            "source_key": f"{SKILL}#discovered/{slug}",
            "id": sid,
            "doc_path": f"discovered/{p.name}",
        })
    return entries


# 标签词表（与 evolution-world-model/scripts/profiling_evidence.py 的
# BOTTLENECK_STRATEGY_MAP 对齐）
BOTTLENECK_LABELS = [
    "mte2_stall", "mte3_stall", "tiling_imbalance",
    "scalar_loading", "scalar_compute", "compute_bound",
    "near_optimal", "no_overlap", "partial_overlap",
    "undersize_transfer", "icache_miss", "bus_contention",
    "l2_cache_thrash", "ub_memory_pressure",
]

OP_FAMILIES = [
    "elementwise", "normalization", "reduction", "softmax", "attention",
    "flash_attention", "cv_fusion", "matmul", "moe", "quantization",
    "pooling_gather", "optimizer", "index_scatter", "broadcast_mask",
    "special", "datapath",
]


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cards = discover_cards()
    preconds = discover_preconditions()
    playbooks = discover_playbooks()
    families = discover_families()
    discovered = discover_discovered()

    deprecated_count = sum(1 for c in cards if c.get("deprecated", False))
    active_cards = len(cards) - deprecated_count

    index = {
        "skill": SKILL,
        "version": "1.0",
        "vocabulary": {
            "bottleneck_labels": BOTTLENECK_LABELS,
            "op_families": OP_FAMILIES,
            "complexity_levels": ["L0", "L1", "L2"],
        },
        "stats": {
            "cards": len(cards),
            "active_cards": active_cards,
            "deprecated": deprecated_count,
            "preconditions": len(preconds),
            "playbooks": len(playbooks),
            "families": len(families),
            "discovered": len(discovered),
        },
        "cards": cards,
        "preconditions": preconds,
        "playbooks": playbooks,
        "families": families,
        "discovered": discovered,
    }

    INDEX_PATH.write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Wrote %s", INDEX_PATH)
    LOGGER.info("  cards=%d (active=%d, deprecated=%d), "
                "preconditions=%d, playbooks=%d, "
                "families=%d, discovered=%d",
                len(cards), active_cards, deprecated_count,
                len(preconds), len(playbooks),
                len(families), len(discovered))


if __name__ == "__main__":
    main()
