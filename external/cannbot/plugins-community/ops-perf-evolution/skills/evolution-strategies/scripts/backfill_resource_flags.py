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
backfill_resource_flags.py — 一次性脚本：给所有策略卡补全 has_preconditions / has_playbook 字段

扫描 preconditions/ 和 playbooks/ 目录，根据策略 ID 判断对应资源是否存在，
然后在每张 card 的 frontmatter 的 synergizes_with 后插入这两个字段。

幂等：如果字段已存在则跳过。
"""

from __future__ import annotations
import logging
import re
from pathlib import Path

LOGGER = logging.getLogger(__name__)

SKILL_ROOT = Path("plugins-community/ops-perf-evolution/skills/evolution-strategies/references")
CARDS_DIR = SKILL_ROOT / "cards"
PRECOND_DIR = SKILL_ROOT / "preconditions"
PLAYBOOK_DIR = SKILL_ROOT / "playbooks"

ID_RE = re.compile(r"^id:\s*(\S+)\s*$", re.MULTILINE)
SYNERGIZES_LINE_RE = re.compile(r"^synergizes_with:.*$", re.MULTILINE)
HAS_PRECOND_RE = re.compile(r"^has_preconditions:", re.MULTILINE)
HAS_PB_RE = re.compile(r"^has_playbook:", re.MULTILINE)


def discover_resources():
    precond_ids = {p.stem for p in PRECOND_DIR.glob("*.yaml")}
    playbook_ids = set()
    for p in PLAYBOOK_DIR.glob("*.md"):
        # 文件名形如 P1_double_buffering.md → 提取 P1
        stem = p.stem
        if "_" in stem:
            playbook_ids.add(stem.split("_", 1)[0])
        else:
            playbook_ids.add(stem)
    return precond_ids, playbook_ids


def split_frontmatter(text: str):
    """返回 (frontmatter, body)。无 frontmatter 返回 (None, text)。"""
    if not text.startswith("---\n"):
        return None, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return None, text
    fm = text[4:end]
    body = text[end + 5:]
    return fm, body


def _insert_after_synergizes(fm: str, block: str) -> str:
    """在 synergizes_with 行后插入内容；无该行时追加到末尾。"""
    synergizes_match = SYNERGIZES_LINE_RE.search(fm)
    if synergizes_match:
        return fm[:synergizes_match.end()] + "\n" + block + fm[synergizes_match.end():]
    return fm.rstrip("\n") + "\n" + block + "\n"


def _upsert_flag(fm: str, field_name: str, value: bool) -> str:
    """字段已存在则替换值，否则插入到 synergizes_with 行后。"""
    line = f"{field_name}: {str(value).lower()}"
    if re.search(rf"^{field_name}:", fm, flags=re.MULTILINE):
        return re.sub(rf"^{field_name}:.*$", line, fm, flags=re.MULTILINE)
    return _insert_after_synergizes(fm, line)


def patch_card(card_path: Path, precond_ids: set[str], playbook_ids: set[str],
               force: bool = False) -> bool:
    text = card_path.read_text(encoding="utf-8")
    fm, body = split_frontmatter(text)
    if fm is None:
        LOGGER.info("  SKIP %s: no frontmatter", card_path.name)
        return False

    # 跳过元数据文件（CONTRIBUTING/SCHEMA）
    id_match = ID_RE.search(fm)
    if not id_match:
        return False
    sid = id_match.group(1).strip()

    has_precond_existed = bool(HAS_PRECOND_RE.search(fm))
    has_pb_existed = bool(HAS_PB_RE.search(fm))

    if has_precond_existed and has_pb_existed and not force:
        return False  # already done

    if force and (has_precond_existed or has_pb_existed):
        # Force mode: 替换现有字段值（不存在的字段落回插入逻辑）
        fm = _upsert_flag(fm, "has_preconditions", sid in precond_ids)
        fm = _upsert_flag(fm, "has_playbook", sid in playbook_ids)
    else:
        # Non-force original logic: 只插入缺失字段，已存在的保留原值
        inserts = []
        if not has_precond_existed:
            inserts.append(f"has_preconditions: {str(sid in precond_ids).lower()}")
        if not has_pb_existed:
            inserts.append(f"has_playbook: {str(sid in playbook_ids).lower()}")
        fm = _insert_after_synergizes(fm, "\n".join(inserts))

    new_text = f"---\n{fm}\n---\n{body}"
    card_path.write_text(new_text, encoding="utf-8")
    return True


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Backfill has_preconditions / has_playbook")
    parser.add_argument("--force", action="store_true",
                        help="强制覆盖现有 has_* 字段（基于当前文件系统）")
    args = parser.parse_args()

    precond_ids, playbook_ids = discover_resources()
    LOGGER.info("Preconditions IDs: %s", sorted(precond_ids))
    LOGGER.info("Playbook IDs:      %s", sorted(playbook_ids))
    LOGGER.info("")

    cards = sorted(CARDS_DIR.glob("*.md"))
    # 排除非策略卡
    cards = [c for c in cards if c.name not in {"CONTRIBUTING.md", "SCHEMA.md"}]

    patched = 0
    for card in cards:
        if patch_card(card, precond_ids, playbook_ids, force=args.force):
            patched += 1

    LOGGER.info("Patched %d/%d cards (force=%s)", patched, len(cards), args.force)


if __name__ == "__main__":
    main()
