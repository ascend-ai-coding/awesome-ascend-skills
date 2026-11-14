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
backfill_resource_flags.py — 维护脚本：校准 strategy_meta 的 has_preconditions / has_playbook 字段

策略卡正文已迁至 cannbot-knowledge 仓，机器元数据吸收在 preconditions/{sid}.yaml
的 strategy_meta 块中。本脚本按当前文件系统事实校准两个标志：
- has_preconditions: 该 yaml 是否含 checks: 段（可执行检查）
- has_playbook:      strategy_meta 是否带 playbook_path（知识仓 Runbook 存在）

幂等：值一致则跳过；--force 无实际区别（总是以文件系统为准），保留仅为兼容旧调用。
"""

from __future__ import annotations
import logging
import re
from pathlib import Path

LOGGER = logging.getLogger(__name__)

SKILL_ROOT = Path(__file__).resolve().parent.parent / "references"
PRECOND_DIR = SKILL_ROOT / "preconditions"

CHECKS_RE = re.compile(r"^checks:\s*$", re.MULTILINE)
META_START_RE = re.compile(r"^strategy_meta:\s*$", re.MULTILINE)
PLAYBOOK_PATH_RE = re.compile(r"^\s+playbook_path:", re.MULTILINE)


def _upsert_meta_flag(text: str, field_name: str, value: bool) -> str:
    line = f"  {field_name}: {str(value).lower()}"
    if re.search(rf"^\s+{field_name}:.*$", text, flags=re.MULTILINE):
        return re.sub(rf"^\s+{field_name}:.*$", line, text, flags=re.MULTILINE)
    # 插入到 strategy_meta: 行后
    m = META_START_RE.search(text)
    if m:
        insert_at = m.end()
        return text[:insert_at] + "\n" + line + text[insert_at:]
    return text


def patch_yaml(yaml_path: Path) -> bool:
    text = yaml_path.read_text(encoding="utf-8")
    if not META_START_RE.search(text):
        LOGGER.info("  SKIP %s: no strategy_meta", yaml_path.name)
        return False
    has_checks = bool(CHECKS_RE.search(text))
    has_playbook = bool(PLAYBOOK_PATH_RE.search(text))
    new_text = _upsert_meta_flag(text, "has_preconditions", has_checks)
    new_text = _upsert_meta_flag(new_text, "has_playbook", has_playbook)
    if new_text == text:
        return False
    yaml_path.write_text(new_text, encoding="utf-8")
    return True


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="校准 strategy_meta has_preconditions / has_playbook"
    )
    parser.add_argument(
        "--force", action="store_true", help="兼容旧调用（校准总是以文件系统为准）"
    )
    parser.parse_args()  # 仅用于校验参数合法性（--force 兼容旧调用）

    yamls = sorted(PRECOND_DIR.glob("*.yaml"))
    patched = 0
    for y in yamls:
        if patch_yaml(y):
            patched += 1

    LOGGER.info("Patched %d/%d preconditions yamls", patched, len(yamls))


if __name__ == "__main__":
    main()
