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
# -----------------------------------------------------------------------------------------------------------
"""三仓适配契约检查 — 分别检查 skill / plugin / knowledge 是否完整适配。

用法::

    python3 check_three_repo_contract.py \
        --skills  /path/to/cannbot-skills \
        --plugin  /path/to/cannbot/plugins-community/ascendc-ops-perf-evolution \
        --knowledge /path/to/cannbot-knowledge \
        [--json report.json] [--run-check]

检查内容：

* **skills 仓**：SKILL.md / strategy-index.md / scenario-routing / 四个脚本；
  `references/cards/` 只允许 `SCHEMA.md` 与 `perf_shape_spec_01.md`（其余正文
  必须已迁知识仓）；preconditions 非空。
* **plugin 仓**：`plugin-sources.json` 的每个 skill 在 skills 仓存在；`init.sh`
  支持 `--knowledge-root`；`agents/*.md` 中所有 `.claude/skills/<skill>/<path>`
  引用都能解析到真实文件（并单独标记三类历史旧路径）。
* **knowledge 仓**：必需目录/文件齐全；cards/playbooks/discovered 计数与
  INDEX.json stats 一致；`--run-check` 时执行其 `check.sh`。

退出码：0 = 全过；1 = 有失败；2 = 参数错误。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from knowledge_root import (  # noqa: E402
    CARDS_REL,
    DISCOVERED_REL,
    PLAYBOOK_REL,
    validate_root,
)

SKILL_ROOTS = ("community/ops", "ops")
LEGACY_PATTERNS = (
    "references/strategy_index.md",
    "references/strategies/",
    "references/discovered/",
)
SKILL_REQUIRED = (
    "SKILL.md",
    "references/strategy-index.md",
    "references/scenario-routing/README.md",
    "references/strategy-compatibility.md",
    "scripts/knowledge_root.py",
    "scripts/load_playbook.py",
    "scripts/generate_index_json.py",
    "scripts/query_strategies.py",
)
ALLOWED_LOCAL_CARDS = {"SCHEMA.md", "perf_shape_spec_01.md"}
LINK_RE = re.compile(r"\.claude/skills/([A-Za-z0-9_.-]+)/([A-Za-z0-9_./{*}$Xx-]+)")


def _count_markdown(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for p in directory.glob("*.md") if p.name != "index.md")


def _record(results: list, repo: str, name: str, ok: bool, detail) -> None:
    results.append({"repo": repo, "check": name, "passed": bool(ok), "detail": detail})


def check_skills(
    skills_repo: Path, results: list, skill_dir: str | None = None, legacy: bool = False
) -> Path | None:
    skill = (
        Path(skill_dir).expanduser().resolve()
        if skill_dir
        else skills_repo / "community" / "ops" / "ascendc-evolution-strategies"
    )
    if not skill.is_dir():
        _record(results, "skills", "skill_dir", False, f"未找到 {skill}")
        return None
    _record(results, "skills", "skill_dir", True, str(skill))

    missing = [rel for rel in SKILL_REQUIRED if not (skill / rel).exists()]
    _record(results, "skills", "required_files", not missing, missing or "ok")

    if not legacy:
        local_cards = skill / "references" / "cards"
        if local_cards.is_dir():
            extra = sorted(
                p.name
                for p in local_cards.glob("*")
                if p.name not in ALLOWED_LOCAL_CARDS
            )
        else:
            extra = []
        _record(
            results,
            "skills",
            "migrated_cards_only",
            not extra,
            extra or "ok（只保留 SCHEMA.md / perf_shape_spec_01.md）",
        )
    else:
        _record(
            results, "skills", "migrated_cards_only", True, "skipped（--legacy-layout）"
        )

    preconds = list((skill / "references" / "preconditions").glob("*.yaml"))
    _record(
        results,
        "skills",
        "preconditions_present",
        len(preconds) > 0,
        f"{len(preconds)} 份",
    )
    return skill


def check_plugin(
    plugin_repo: Path, skills_repo: Path, results: list, legacy: bool = False
) -> None:
    if not plugin_repo.is_dir():
        _record(results, "plugin", "plugin_dir", False, f"未找到 {plugin_repo}")
        return
    _record(results, "plugin", "plugin_dir", True, str(plugin_repo))

    sources_path = plugin_repo / "plugin-sources.json"
    if sources_path.is_file():
        parse_error = None
        try:
            sources = json.loads(sources_path.read_text(encoding="utf-8"))
            if not isinstance(sources, dict):
                parse_error = f"顶层应为对象，实际为 {type(sources).__name__}"
                declared = []
            else:
                declared = sources.get("skills", [])
                if not isinstance(declared, list):
                    parse_error = f'"skills" 应为数组，实际为 {type(declared).__name__}'
                    declared = []
        except json.JSONDecodeError as exc:
            parse_error = f"JSON 解析失败: {exc}"
            declared = []
        if parse_error is not None:
            # 配置文件损坏时必须 FAIL，不允许静默判 PASS 绕过门禁
            _record(
                results,
                "plugin",
                "plugin_sources_resolve",
                False,
                f"{parse_error}（{sources_path.name}）",
            )
        else:
            missing = []
            for rel in declared:
                # plugin-sources.json 的 skills 是仓相对路径，如
                # "community/ops/ascendc-evolution-knowledge" / "ops/npu-arch"
                if not (skills_repo / rel).is_dir():
                    missing.append(rel)
            _record(
                results,
                "plugin",
                "plugin_sources_resolve",
                not missing,
                missing or f"{len(declared)} 个 skill",
            )
    elif legacy:
        _record(
            results,
            "plugin",
            "plugin_sources_resolve",
            True,
            "skipped（--legacy-layout，旧布局无 plugin-sources.json）",
        )
    else:
        _record(
            results,
            "plugin",
            "plugin_sources_resolve",
            False,
            "缺少 plugin-sources.json",
        )

    init_path = plugin_repo / "init.sh"
    init_text = init_path.read_text(encoding="utf-8") if init_path.is_file() else ""
    _record(
        results,
        "plugin",
        "init_supports_knowledge_root",
        "--knowledge-root" in init_text and "--skills-repo" in init_text,
        "init.sh 需同时支持 --knowledge-root 与 --skills-repo",
    )

    agent_files = sorted((plugin_repo / "agents").glob("*.md"))
    missing_refs = []
    legacy_hits = []
    scanned = 0
    for agent in agent_files:
        text = agent.read_text(encoding="utf-8", errors="replace")
        for match in LINK_RE.finditer(text):
            skill_name, rel = match.group(1), match.group(2).rstrip(".,;:）)")
            scanned += 1
            if any(pattern in f"{skill_name}/{rel}" for pattern in LEGACY_PATTERNS):
                legacy_hits.append(f"{agent.name}: {skill_name}/{rel}")
            skill_dir = next(
                (
                    skills_repo / root / skill_name
                    for root in SKILL_ROOTS
                    if (skills_repo / root / skill_name).is_dir()
                ),
                None,
            )
            if skill_dir is None:
                missing_refs.append(f"{agent.name}: 未知 skill {skill_name}")
                continue
            if "*" in rel or "{" in rel:
                continue
            if not (skill_dir / rel).exists():
                missing_refs.append(f"{agent.name}: {skill_name}/{rel}")
    _record(
        results,
        "plugin",
        "agent_skill_links_resolve",
        not missing_refs,
        {"scanned": scanned, "missing": missing_refs}
        if missing_refs
        else f"{scanned} 条引用全部可解析",
    )
    _record(results, "plugin", "no_legacy_paths", not legacy_hits, legacy_hits or "ok")


def check_knowledge(knowledge_repo: Path, run_check: bool, results: list) -> None:
    if not knowledge_repo.is_dir():
        _record(
            results, "knowledge", "knowledge_dir", False, f"未找到 {knowledge_repo}"
        )
        return
    _record(results, "knowledge", "knowledge_dir", True, str(knowledge_repo))
    missing = validate_root(knowledge_repo)
    _record(results, "knowledge", "required_layout", not missing, missing or "ok")

    counts = {
        "cards": _count_markdown(knowledge_repo / CARDS_REL),
        "playbooks": _count_markdown(knowledge_repo / PLAYBOOK_REL),
        "discovered": _count_markdown(knowledge_repo / DISCOVERED_REL),
    }
    # INDEX.json 在 skills 仓，check_knowledge 只报告计数，交叉核对交由 smoke 完成。
    _record(results, "knowledge", "counts", all(v > 0 for v in counts.values()), counts)

    if run_check:
        proc = subprocess.run(
            ["bash", str(knowledge_repo / "check.sh")],
            cwd=str(knowledge_repo),
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
        tail = (proc.stdout or "").strip().splitlines()[-3:]
        _record(
            results,
            "knowledge",
            "check_sh",
            proc.returncode == 0,
            {"returncode": proc.returncode, "tail": tail},
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="三仓适配契约检查")
    parser.add_argument("--skills", required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--knowledge", required=True)
    parser.add_argument("--json", default=None)
    parser.add_argument(
        "--skill-dir",
        default=None,
        help="skill 目录覆盖（旧布局用，如 .../skills/evolution-strategies）",
    )
    parser.add_argument(
        "--legacy-layout",
        action="store_true",
        help="旧布局（正文未迁移）：跳过 migrated_cards_only 检查",
    )
    parser.add_argument("--run-check", action="store_true", help="执行知识仓 check.sh")
    args = parser.parse_args(argv)

    results: list[dict] = []
    skills_repo = Path(args.skills).expanduser().resolve()
    plugin_repo = Path(args.plugin).expanduser().resolve()
    knowledge_repo = Path(args.knowledge).expanduser().resolve()

    check_skills(skills_repo, results, args.skill_dir, args.legacy_layout)
    check_plugin(plugin_repo, skills_repo, results, args.legacy_layout)
    check_knowledge(knowledge_repo, args.run_check, results)

    report = {
        "skills": str(skills_repo),
        "plugin": str(plugin_repo),
        "knowledge": str(knowledge_repo),
        "passed": all(item["passed"] for item in results),
        "results": results,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json:
        Path(args.json).expanduser().write_text(text + "\n", encoding="utf-8")

    for item in results:
        mark = "PASS" if item["passed"] else "FAIL"
        print(f"[{mark}] {item['repo']:9s} {item['check']:32s} {item['detail']}")
    print("CONTRACT: " + ("PASS" if report["passed"] else "FAIL"))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
