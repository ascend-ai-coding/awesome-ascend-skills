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
generate_index_json.py — 从 preconditions strategy_meta 程序化生成 INDEX.json

策略卡/Playbook 正文已迁至 cannbot-knowledge 仓（条目带 doc_repo 标记，
doc_path 为 knowledge 仓相对路径）；卡片的机器元数据（bottlenecks/
op_families/complexity/conflicts/synergizes/requires/quantified_gain）
吸收在 preconditions/{sid}.yaml 的 strategy_meta 块中，本脚本以其为权威来源。

INDEX.json 是 machine-parseable 的 source_key 权威清单，用途：
- bottleneck_to_sources.py 按 triggers.bottleneck_labels 反查策略
- 防偷懒 audit 检查 source_key 合法性
- partial-prompt 注入时快速定位资源路径

输出位置：community/ops/ascendc-evolution-strategies/references/INDEX.json
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

_SCRIPT_DIR_FOR_IMPORT = os.path.dirname(os.path.realpath(__file__))
if _SCRIPT_DIR_FOR_IMPORT not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR_FOR_IMPORT)

from knowledge_root import (  # noqa: E402
    KnowledgeRootError,
    discovered_dir,
    resolve_knowledge_root,
)

LOGGER = logging.getLogger(__name__)

# 以脚本自身位置锚定路径（不依赖 cwd），兼容从插件根或 workspace 根运行
SKILL_ROOT = Path(__file__).resolve().parent.parent / "references"
PRECOND_DIR = SKILL_ROOT / "preconditions"
FAMILIES_DIR = SKILL_ROOT / "families"
INDEX_PATH = SKILL_ROOT / "INDEX.json"

# X 系列（进化发现）正文已迁至 cannbot-knowledge 仓
# （knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/，
# OKF frontmatter，文件名小写 disc_x{n}.md）。
# 知识仓根解析与 load_playbook.py 对齐，三级来源：
#   1. CANNBOT_KNOWLEDGE_ROOT 环境变量
#   2. 自 cwd 向上最近的 .cannbot/knowledge.env（cannbot-knowledge install.sh 写入项目）
#   3. skills 仓真实路径的兄弟检出 ../cannbot-knowledge（开发布局便利回退）
KNOWLEDGE_DISCOVERED_REL = (
    "knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered"
)


def _parse_knowledge_env(path: Path) -> str:
    """从 .cannbot/knowledge.env 提取 CANNBOT_KNOWLEDGE_ROOT（去 export 前缀与引号）。"""
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
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
    d = Path.cwd()
    while True:
        cand = d / ".cannbot" / "knowledge.env"
        if cand.is_file():
            return _parse_knowledge_env(cand)
        if d.parent == d:
            return ""
        d = d.parent


def resolve_knowledge_discovered_dir(cli_root: str | None = None) -> Path | None:
    """按统一解析链解析知识仓 discovered 目录。

    显式来源（CLI / env / .cannbot/knowledge.env）无效时 fail-loud（SystemExit(3)）；
    完全未配置时返回 None，由调用方降级复用现有 INDEX.json 的 discovered 数组。
    """
    try:
        root, source = resolve_knowledge_root(cli_root)
        LOGGER.info("knowledge root: %s (source=%s)", root, source)
        return discovered_dir(root)
    except KnowledgeRootError as exc:
        if not exc.configured:
            return None
        LOGGER.error("%s", exc)
        raise SystemExit(3)


# OKF 知识库根目录（兄弟 skill ascendc-evolution-knowledge 下）
# 脚本在 <skills根>/ascendc-evolution-strategies/scripts/ → 上三级到 skills 根
# （仓内 community/ops 或安装后 .claude/skills，两种布局下兄弟 skill 均并列）
OKF_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "ascendc-evolution-knowledge"
    / "references"
    / "okf"
)

SKILL = "ascendc-evolution-strategies"
OKF_SKILL = "ascendc-evolution-knowledge"


def split_frontmatter(text: str):
    if not text.startswith("---\n"):
        return None, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return None, text
    tail_start = end + 5
    return text[4:end], text[tail_start:]


def parse_yaml_frontmatter(fm: str) -> dict:
    """轻量 YAML 解析器，仅支持本仓 frontmatter 用到的子集。"""
    result = {}
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
            current_list = []
            result[key] = current_list
        elif val.startswith("[") and val.endswith("]"):
            # 行内列表（兼容带引号项：op_families: ['matmul']）
            inner = val[1:-1].strip()
            if not inner:
                result[key] = []
            else:
                items = [s.strip().strip('"').strip("'") for s in inner.split(",")]
                result[key] = items
            current_list = None
        elif val.lower() in ("true", "false"):
            result[key] = val.lower() == "true"
            current_list = None
        else:
            result[key] = val.strip('"').strip("'")
            current_list = None
        i += 1
    return result


def parse_strategy_meta(text: str) -> dict | None:
    """解析 preconditions YAML 中的 strategy_meta 块（卡片机器元数据的权威来源）。

    策略卡正文已迁至 cannbot-knowledge 仓（knowledge/contrib/ascendc/
    optimizations/evolution_strategies/），其 frontmatter 受 OKF schema
    additionalProperties:false 约束不能携带 bottlenecks/op_families 等机器字段，
    故元数据吸收进 preconditions/{sid}.yaml 的 strategy_meta 块（位于 checks: 前）。
    quantified_gain 为单行 JSON flow（含嵌套引号），先摘出单独 json.loads。
    """
    m = re.search(r"^strategy_meta:\s*$", text, re.M)
    if not m:
        return None
    meta_end = m.end()
    lines = []
    for line in text[meta_end:].splitlines():
        if re.match(r"^[a-zA-Z_]+:", line):
            break
        lines.append(line)
    block = "\n".join(lines)
    qg = []
    qg_m = re.search(r"^\s+quantified_gain: (.*)$", block, re.M)
    if qg_m:
        try:
            qg = json.loads(qg_m.group(1))
        except json.JSONDecodeError:
            qg = []
        block = re.sub(r"^\s+quantified_gain: .*$", "", block, flags=re.M)
    # 通用解析要求 key 顶格，strategy_meta 子键统一 2 空格缩进，先去缩进
    block = "\n".join(
        raw_line[2:] if raw_line.startswith("  ") else raw_line
        for raw_line in block.splitlines()
    )
    meta = parse_yaml_frontmatter(block)
    meta["quantified_gain"] = qg
    return meta


def discover_cards() -> list[dict]:
    entries = []
    for p in sorted(PRECOND_DIR.glob("*.yaml")):
        meta = parse_strategy_meta(p.read_text(encoding="utf-8"))
        if not meta:
            continue
        sid = meta.get("id", p.stem)
        slug = meta.get("slug", p.stem)  # e.g. "P1_double_buffer"

        entry = {
            "source_key": f"{SKILL}#card/{slug}",
            "id": sid,
            "title": meta.get("title", slug),
            "doc_repo": meta.get("doc_repo", "cannbot-knowledge"),
            "doc_path": meta.get("doc_path", ""),
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
        entries.append(
            {
                "source_key": f"{SKILL}#preconditions/{sid}",
                "id": sid,
                "doc_path": f"preconditions/{p.name}",
            }
        )
    return entries


def discover_playbooks() -> list[dict]:
    entries = []
    for p in sorted(PRECOND_DIR.glob("*.yaml")):
        meta = parse_strategy_meta(p.read_text(encoding="utf-8"))
        if not meta or not meta.get("playbook_path"):
            continue
        slug = meta.get("playbook_slug", meta.get("slug", p.stem))
        sid = meta.get("id", p.stem)
        entries.append(
            {
                "source_key": f"{SKILL}#playbook/{slug}",
                "id": sid,
                "doc_repo": meta.get("doc_repo", "cannbot-knowledge"),
                "doc_path": meta["playbook_path"],
            }
        )
    return entries


def discover_families() -> list[dict]:
    entries = []
    if not FAMILIES_DIR.exists():
        return entries
    for p in sorted(FAMILIES_DIR.glob("*.md")):
        slug = p.stem  # e.g. "matmul_guide"
        entries.append(
            {
                "source_key": f"{SKILL}#family/{slug}",
                "title": slug.replace("_", " ").title(),
                "doc_path": f"families/{p.name}",
            }
        )
    return entries


def discover_discovered(cli_root: str | None = None) -> list[dict]:
    """扫描 cannbot-knowledge 仓 discovered/ 生成 X 系列条目（doc_repo 标记）。

    知识仓根完全未配置时降级复用现有 INDEX.json 的 discovered 数组（告警，
    不静默清空——避免在无知识仓检出的环境重建索引时丢光 X 条目）；显式配置
    无效时上层已 fail-loud，不会走到这里。
    """
    disc_dir = resolve_knowledge_discovered_dir(cli_root)
    if disc_dir is None:
        LOGGER.warning(
            "WARN: 无法定位 cannbot-knowledge 仓 discovered 目录，"
            "复用现有 INDEX.json 的 discovered 数组（已尝试 CANNBOT_KNOWLEDGE_ROOT / "
            ".cannbot/knowledge.env / 兄弟检出 ../cannbot-knowledge）"
        )
        try:
            existing = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
            return list(existing.get("discovered", []))
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("WARN: 现有 INDEX.json 不可读，discovered 数组为空")
            return []
    entries = []
    for p in sorted(disc_dir.glob("disc_x*.md")):
        slug = p.stem  # disc_x17
        sid = slug.split("_", 1)[1].upper() if "_" in slug else slug  # X17
        entries.append(
            {
                "source_key": f"{SKILL}#discovered/{slug}",
                "id": sid,
                "doc_repo": "cannbot-knowledge",
                "doc_path": f"{KNOWLEDGE_DISCOVERED_REL}/{p.name}",
            }
        )
    return entries


def discover_okf() -> list[dict]:
    """扫描 OKF 知识库（ascendc-evolution-knowledge/references/okf/）生成卡片条目。

    三类卡片合并进 INDEX.json 的 cards 数组，source_key 前缀
    ascendc-evolution-knowledge#okf/ 供 ascendc-evolution-knowledge 检索（按 kind /
    op_families / bottlenecks / phenomenon / failed_combination 过滤）：
      - discovered_strategy（X 系列，OKF 增强版，与 discovered/ 双轨保留）
      - proven_solution（跨算子已验证方案）
      - failure_pattern（已知失败模式，用于 anti 提示）

    每张卡都必须带 id 字段（profiling_evidence/wm_ledger 按 id 反查会硬取）。
    """
    entries = []
    if not OKF_ROOT.exists():
        return entries

    # 1) discovered_strategy（X 系列 OKF 增强版）
    disc_dir = OKF_ROOT / "strategies" / "discovered"
    for p in sorted(disc_dir.glob("disc_*.md")):
        text = p.read_text(encoding="utf-8")
        fm, _ = split_frontmatter(text)
        if not fm:
            continue
        meta = parse_yaml_frontmatter(fm)
        slug = p.stem  # disc_X5
        sid = slug.split("_", 1)[1] if "_" in slug else slug  # X5
        entries.append(
            {
                "source_key": f"{OKF_SKILL}#okf/strategies/discovered/{sid}",
                "id": sid,
                "title": meta.get("title", sid),
                "doc_path": f"skills/ascendc-evolution-knowledge/references/okf/strategies/discovered/{p.name}",
                "kind": "discovered_strategy",
                "confidence": meta.get("confidence", "single_op"),
                "status": meta.get("status", "active"),
                "op_families": meta.get("op_families", []),
                "bottlenecks": meta.get("bottlenecks", []),
                "tags": meta.get("tags", []),
                "description": meta.get("description", ""),
            }
        )

    # 2) proven_solution（跨算子已验证方案）
    proven_dir = OKF_ROOT / "runbooks" / "proven-solutions"
    for p in sorted(proven_dir.glob("*.md")):
        if p.name == "index.md":
            continue
        text = p.read_text(encoding="utf-8")
        fm, _ = split_frontmatter(text)
        if not fm:
            continue
        meta = parse_yaml_frontmatter(fm)
        slug = p.stem
        entries.append(
            {
                "source_key": f"{OKF_SKILL}#okf/runbooks/proven-solutions/{slug}",
                "id": slug,
                "title": meta.get("title", slug),
                "doc_path": f"skills/ascendc-evolution-knowledge/references/okf/runbooks/proven-solutions/{p.name}",
                "kind": "proven_solution",
                "confidence": meta.get("confidence", "single_op"),
                "status": meta.get("status", "active"),
                "op_families": meta.get("op_families", []),
                "strategy_ids": meta.get("strategy_ids", []),
                "tags": [],
            }
        )

    # 3) failure_pattern（已知失败模式，用于 anti 提示）
    failure_dir = OKF_ROOT / "runbooks" / "failure-patterns"
    for p in sorted(failure_dir.glob("*.md")):
        if p.name == "index.md":
            continue
        text = p.read_text(encoding="utf-8")
        fm, _ = split_frontmatter(text)
        if not fm:
            continue
        meta = parse_yaml_frontmatter(fm)
        slug = p.stem
        entries.append(
            {
                "source_key": f"{OKF_SKILL}#okf/runbooks/failure-patterns/{slug}",
                "id": meta.get("key", slug),
                "title": f"{meta.get('phenomenon', slug)}: {' + '.join(meta.get('failed_combination', []))}",
                "doc_path": f"skills/ascendc-evolution-knowledge/references/okf/runbooks/failure-patterns/{p.name}",
                "kind": "failure_pattern",
                "phenomenon": meta.get("phenomenon", ""),
                "failed_combination": meta.get("failed_combination", []),
                "status": meta.get("status", "stub"),
                "severity": meta.get("severity", ""),
                "operator_type": meta.get("operator_type", []),
                "op_families": meta.get("op_families", []),
                "bottlenecks": [],
                "tags": meta.get("tags", []),
            }
        )
    return entries


# 标签词表（与 ascendc-evolution-world-model/scripts/profiling_evidence.py 的
# BOTTLENECK_STRATEGY_MAP 对齐）
BOTTLENECK_LABELS = [
    "mte2_stall",
    "mte3_stall",
    "tiling_imbalance",
    "scalar_loading",
    "scalar_compute",
    "compute_bound",
    "near_optimal",
    "no_overlap",
    "partial_overlap",
    "undersize_transfer",
    "icache_miss",
    "bus_contention",
    "l2_cache_thrash",
    "ub_memory_pressure",
    "cube_bound",
    "db_not_effective",
]

OP_FAMILIES = [
    "elementwise",
    "normalization",
    "reduction",
    "softmax",
    "attention",
    "flash_attention",
    "cv_fusion",
    "matmul",
    "moe",
    "quantization",
    "pooling_gather",
    "optimizer",
    "index_scatter",
    "broadcast_mask",
    "special",
    "datapath",
    "ssm",
    "linear_attention",
    "recurrent",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从 strategy_meta 生成 INDEX.json")
    parser.add_argument(
        "--knowledge-root",
        default=None,
        help="cannbot-knowledge 仓根（显式覆盖；配置无效时 fail-loud）",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cards = discover_cards()
    okf = discover_okf()
    cards = cards + okf
    preconds = discover_preconditions()
    playbooks = discover_playbooks()
    families = discover_families()
    discovered = discover_discovered(args.knowledge_root)

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
            "okf": len(okf),
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
    LOGGER.info(
        "  cards=%d (active=%d, deprecated=%d), "
        "preconditions=%d, playbooks=%d, "
        "families=%d, discovered=%d, okf=%d",
        len(cards),
        active_cards,
        deprecated_count,
        len(preconds),
        len(playbooks),
        len(families),
        len(discovered),
        len(okf),
    )


if __name__ == "__main__":
    raise SystemExit(main())
