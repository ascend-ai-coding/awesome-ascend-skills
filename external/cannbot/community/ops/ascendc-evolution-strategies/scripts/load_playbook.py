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
"""load_playbook.py — 按 ID 加载 Playbook markdown，拼接为纯文本

用途:
    主 agent 填充 ops-partial prompt 前，按 node.strategy_combination 加载
    命中的 Playbook，注入到子 agent prompt 的 [STRATEGY PLAYBOOKS] 段落。

用法:
    python3 load_playbook.py \\
        --strategy-ids P1,P14,P99 \\
        --output /tmp/playbooks.txt

    # P99 无 Playbook 时不报错（只加载已存在的）
    # 输出格式：各 Playbook 之间用 '\n\n---\n\n' 分隔

输出:
    如果至少加载了一个 Playbook：写入 output，stdout 报告加载数
    如果全部都无 Playbook：output 为空文件，stdout 报告 "no_playbooks_loaded"
    知识仓根完全无法解析（配置错误）：stderr 报错 + exit 3（fail-loud，
    与 ascendc-port-orchestrator external_kb 语义对齐，不静默裸奔）
"""

import argparse
import glob
import logging
import os
import re
import sys

_SCRIPT_DIR_FOR_IMPORT = os.path.dirname(os.path.realpath(__file__))
if _SCRIPT_DIR_FOR_IMPORT not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR_FOR_IMPORT)

from knowledge_root import (  # noqa: E402
    KnowledgeRootError,
    playbook_dir,
    resolve_knowledge_root,
)


LOGGER = logging.getLogger(__name__)

# 数据输出专用 logger：CLI 结果（JSON/表格/报告）走 stdout（agent 调用协议通道），
# 与 LOGGER（stderr 进度/警告）分离，避免 lint G.LOG.02 误报 print。
DATA_LOGGER = logging.getLogger(f"{__name__}.data")
_DATA_HANDLER = logging.StreamHandler(sys.stdout)
_DATA_HANDLER.setFormatter(logging.Formatter("%(message)s"))
DATA_LOGGER.addHandler(_DATA_HANDLER)
DATA_LOGGER.propagate = False
DATA_LOGGER.setLevel(logging.INFO)

# Playbook 正文已迁至 cannbot-knowledge 仓（knowledge/contrib/ascendc/runbooks/performance/）。
# 知识仓根解析与 ascendc-port-orchestrator 的 external_kb.py 对齐，三级来源：
#   1. CANNBOT_KNOWLEDGE_ROOT 环境变量（显式配置；配了但路径无效视为配置错误，fail-loud）
#   2. 自 cwd 向上最近的 .cannbot/knowledge.env（cannbot-knowledge install.sh 写入项目）
#   3. skills 仓真实路径的兄弟检出 ../cannbot-knowledge（开发布局便利回退）
#   4. 旧版仓内 references/playbooks（迁移过渡期回退）
# 不依赖调用方 cwd 下的相对 skills 路径——主 agent 的工作目录可能是仓库根或其它
# 插件目录；脚本经 .claude/skills 符号链接调用时 __file__ 也未解析到真实仓，
# 故一律用 realpath 定位 skills 仓根。
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
LEGACY_PLAYBOOK_DIR = os.path.normpath(
    os.path.join(SCRIPT_DIR, os.pardir, "references", "playbooks")
)
KNOWLEDGE_PLAYBOOK_REL = os.path.join(
    "knowledge", "contrib", "ascendc", "runbooks", "performance"
)


def _parse_knowledge_env(path: str) -> str:
    """从 .cannbot/knowledge.env 提取 CANNBOT_KNOWLEDGE_ROOT（去 export 前缀与引号）。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                m = re.match(
                    r"\s*(?:export\s+)?CANNBOT_KNOWLEDGE_ROOT\s*=\s*(.*?)\s*$", line
                )
                if m:
                    val = m.group(1)
                    if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                        val = val[1:-1]
                    return val
    except OSError:
        pass
    return ""


def _find_project_knowledge_root() -> str:
    """自 cwd 向上找最近的 .cannbot/knowledge.env，返回其中配置的知识仓根。"""
    d = os.getcwd()
    while True:
        cand = os.path.join(d, ".cannbot", "knowledge.env")
        if os.path.isfile(cand):
            return _parse_knowledge_env(cand)
        parent = os.path.dirname(d)
        if parent == d:
            return ""
        d = parent


def _playbook_dir_of(root: str) -> str:
    return os.path.join(os.path.expanduser(root), KNOWLEDGE_PLAYBOOK_REL)


def _resolve_playbook_dir(cli_root: str | None = None) -> str:
    """按统一解析链定位 Playbook 目录；显式来源无效时 fail-loud（SystemExit(3)）。

    解析优先级的唯一实现见 ``knowledge_root.resolve_knowledge_root``：
    CLI → env → ``.cannbot/knowledge.env`` → 兄弟检出。
    只有「完全未配置知识仓」时才允许旧版仓内 Playbook 兜底；显式配置了无效
    路径属于配置错误，必须报错退出，不得静默换仓。
    """
    try:
        root, source = resolve_knowledge_root(cli_root)
        LOGGER.info("knowledge root: %s (source=%s)", root, source)
        return str(playbook_dir(root))
    except KnowledgeRootError as exc:
        if not exc.configured and os.path.isdir(LEGACY_PLAYBOOK_DIR):
            LOGGER.warning("%s；回退旧版仓内 Playbook：%s", exc, LEGACY_PLAYBOOK_DIR)
            return LEGACY_PLAYBOOK_DIR
        LOGGER.error("%s", exc)
        raise SystemExit(3)


def find_playbook(strategy_id: str, playbook_dir: str) -> str | None:
    """查找策略对应的 Playbook 文件。

    支持：
    1. 知识仓命名: {id小写}_*.md（cannbot-knowledge runbooks/performance，原文件名小写化）
    2. 精确前缀匹配: {ID}_*.md（旧版仓内布局）
    3. 兼容: {ID}.md
    """
    for pattern in (
        f"{strategy_id.lower()}_*.md",
        f"{strategy_id}_*.md",
        f"{strategy_id}.md",
    ):
        matches = glob.glob(os.path.join(playbook_dir, pattern))
        if matches:
            return matches[0]
    return None


def load_playbook_text(strategy_id: str, playbook_dir: str) -> tuple[str, str]:
    """加载单个 Playbook 的内容。

    返回 (content, source_path)，若无 Playbook 则返回 ("", "")。
    """
    path = find_playbook(strategy_id, playbook_dir)
    if not path:
        return "", ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return "", ""
    # 知识仓 Playbook 带 OKF frontmatter（元数据噪声），注入子 agent prompt 前剥离
    content = re.sub(r"\A---\n.*?\n---\n+", "", content, count=1, flags=re.DOTALL)
    return content, path


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="按 ID 加载 Playbook")
    parser.add_argument("--strategy-ids", required=True, help="逗号分隔的策略 ID 列表")
    parser.add_argument(
        "--playbook-dir",
        default=None,
        help="Playbook 目录（默认按三级来源解析知识仓，见模块注释）",
    )
    parser.add_argument(
        "--knowledge-root",
        default=None,
        help="cannbot-knowledge 仓根（显式覆盖；配置无效时 fail-loud）",
    )
    parser.add_argument("--output", default="-", help="输出文件路径（- 为 stdout）")
    parser.add_argument(
        "--list-only", action="store_true", help="只列出命中的 Playbook ID，不输出内容"
    )
    args = parser.parse_args()

    playbook_dir = args.playbook_dir or _resolve_playbook_dir(args.knowledge_root)

    strategy_ids = [x.strip() for x in args.strategy_ids.split(",") if x.strip()]

    loaded = []
    missing = []
    chunks = []
    for sid in strategy_ids:
        content, path = load_playbook_text(sid, playbook_dir)
        if content:
            loaded.append((sid, path))
            # 用显式 BEGIN/END 标记包裹（Playbook 正文含 '---' 用作分隔线，避免歧义）
            chunks.append(
                f"===== PLAYBOOK {sid} BEGIN =====\n\n"
                f"{content.strip()}\n\n"
                f"===== PLAYBOOK {sid} END ====="
            )
        else:
            missing.append(sid)

    if args.list_only:
        DATA_LOGGER.info("loaded: %s", [sid for sid, _ in loaded])
        DATA_LOGGER.info("missing: %s", missing)
        return

    if not chunks:
        # 无任何 Playbook 命中：输出空文件 + stderr 说明
        if args.output != "-":
            with open(args.output, "w", encoding="utf-8") as f:
                f.write("")
        LOGGER.warning("no_playbooks_loaded (requested: %s)", strategy_ids)
        return

    combined = "\n\n".join(chunks)

    if args.output == "-":
        DATA_LOGGER.info("%s", combined)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(combined)

    loaded_ids = [sid for sid, _ in loaded]
    LOGGER.info(
        "Loaded %d playbooks: %s. Missing: %s", len(loaded), loaded_ids, missing
    )


if __name__ == "__main__":
    main()
