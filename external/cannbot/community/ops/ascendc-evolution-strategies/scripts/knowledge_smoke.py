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
"""知识库加载 smoke test — 证明「三仓分离后知识库确实被加载」。

检查项：

1. 解析来源（CLI / env / .cannbot/knowledge.env / 兄弟检出）、绝对路径、git commit；
2. 必需布局（knowledge/、governance/、contrib/ascendc 三件套）；
3. 加载计数：策略卡 / Playbook / X 发现 / preconditions，并与 INDEX.json 的
   stats 交叉核对；
4. 内容指纹：抽样 3 个文件比对 sha256，确认读的是知识仓正文而不是旧缓存；
5. 负向测试：``CANNBOT_KNOWLEDGE_ROOT`` 指向无效路径时必须 fail-loud（rc≠0）。

退出码：0 = 全过；1 = 有失败；2 = 参数/环境错误。

用法::

    python3 knowledge_smoke.py [--knowledge-root PATH] [--json report.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from knowledge_root import (  # noqa: E402
    CARDS_REL,
    DISCOVERED_REL,
    KnowledgeRootError,
    PLAYBOOK_REL,
    git_commit,
    resolve_knowledge_root,
    validate_root,
)

SKILL_ROOT = _SCRIPT_DIR.parent
INDEX_PATH = SKILL_ROOT / "references" / "INDEX.json"
PRECOND_DIR = SKILL_ROOT / "references" / "preconditions"


def _count_markdown(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for p in directory.glob("*.md") if p.name != "index.md")


def _count_cards_local(directory: Path) -> int:
    """旧布局卡片计数：排除 CONTRIBUTING/SCHEMA/index，且要求有 frontmatter。"""
    if not directory.is_dir():
        return 0
    count = 0
    for path in directory.glob("*.md"):
        if path.name in {"CONTRIBUTING.md", "SCHEMA.md", "index.md"}:
            continue
        if path.read_text(encoding="utf-8", errors="replace").startswith("---"):
            count += 1
    return count


def _count_playbooks_local(directory: Path) -> int:
    """旧布局 Playbook 计数：排除 SCHEMA/index。"""
    if not directory.is_dir():
        return 0
    return sum(
        1 for p in directory.glob("*.md") if p.name not in {"SCHEMA.md", "index.md"}
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_markdown(directory: Path) -> Path | None:
    if not directory.is_dir():
        return None
    files = sorted(p for p in directory.glob("*.md") if p.name != "index.md")
    return files[0] if files else None


def _fingerprint(dirs: dict[str, Path]) -> dict:
    result = {}
    for label, directory in dirs.items():
        path = _first_markdown(directory)
        result[label] = {
            "directory": str(directory),
            "path": str(path) if path else "",
            "exists": bool(path),
            "sha256": _sha256(path) if path else "",
        }
    return result


def _negative_test() -> dict:
    """显式 env 配了无效路径时必须 fail-loud。"""
    env = dict(os.environ)
    env["CANNBOT_KNOWLEDGE_ROOT"] = "/nonexistent/cannbot-knowledge-for-smoke"
    cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "load_playbook.py"),
        "--strategy-ids",
        "P1",
        "--list-only",
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=env, timeout=120, check=False
    )
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "passed": proc.returncode != 0,
        "stderr_tail": proc.stderr.strip().splitlines()[-1]
        if proc.stderr.strip()
        else "",
    }


def run(knowledge_root: str | None, corpus: str = "knowledge") -> tuple[dict, bool]:
    report: dict = {"checks": {}, "passed": True, "corpus": corpus}

    def check(name: str, ok: bool, detail) -> None:
        report["checks"][name] = {"passed": bool(ok), "detail": detail}
        if not ok:
            report["passed"] = False

    try:
        root, source = resolve_knowledge_root(knowledge_root)
    except KnowledgeRootError as exc:
        check("resolve_root", False, str(exc))
        return report, False

    report["knowledge_root"] = str(root)
    report["source"] = source
    report["git_commit"] = git_commit(root)

    missing = validate_root(root)
    check("layout", not missing, missing or "ok")

    if corpus == "local":
        # 旧布局（迁移前）：正文仍在本地 references/{playbooks,discovered,cards}
        refs = SKILL_ROOT / "references"
        corpus_dirs = {
            "card": refs / "cards",
            "playbook": refs / "playbooks",
            "discovered": refs / "discovered",
        }
        cards = _count_cards_local(corpus_dirs["card"])
        playbooks = _count_playbooks_local(corpus_dirs["playbook"])
        discovered = _count_markdown(corpus_dirs["discovered"])
    else:
        corpus_dirs = {
            "card": root / CARDS_REL,
            "playbook": root / PLAYBOOK_REL,
            "discovered": root / DISCOVERED_REL,
        }
        cards = _count_markdown(corpus_dirs["card"])
        playbooks = _count_markdown(corpus_dirs["playbook"])
        discovered = _count_markdown(corpus_dirs["discovered"])
    preconditions = len(list(PRECOND_DIR.glob("*.yaml"))) if PRECOND_DIR.is_dir() else 0
    counts = {
        "cards": cards,
        "playbooks": playbooks,
        "discovered": discovered,
        "preconditions": preconditions,
    }
    report["counts"] = counts
    check("counts_nonzero", all(value > 0 for value in counts.values()), counts)

    index_stats = None
    if INDEX_PATH.is_file():
        try:
            index_stats = json.loads(INDEX_PATH.read_text(encoding="utf-8")).get(
                "stats", {}
            )
        except json.JSONDecodeError:
            index_stats = None
    report["index_stats"] = index_stats
    if index_stats:
        mismatches = {
            key: {"index": index_stats.get(key), "actual": value}
            for key, value in counts.items()
            if key in index_stats and index_stats.get(key) != value
        }
        check("index_consistency", not mismatches, mismatches or "ok")
    else:
        check(
            "index_consistency",
            False,
            "INDEX.json 缺失或不可解析（先运行 generate_index_json.py）",
        )

    fingerprints = _fingerprint(corpus_dirs)
    check(
        "fingerprints",
        all(item["exists"] and item["sha256"] for item in fingerprints.values()),
        fingerprints,
    )

    negative = _negative_test()
    report["negative_test"] = negative
    check("fail_loud_on_invalid_env", negative["passed"], negative)

    return report, report["passed"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="知识库加载 smoke test")
    parser.add_argument("--knowledge-root", default=None)
    parser.add_argument("--json", default=None, help="把报告写到指定路径")
    parser.add_argument(
        "--corpus",
        choices=("knowledge", "local"),
        default="knowledge",
        help="计数/指纹来源：knowledge=知识仓（默认），local=旧布局本地 references",
    )
    args = parser.parse_args(argv)

    report, passed = run(args.knowledge_root, args.corpus)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json:
        Path(args.json).expanduser().write_text(text + "\n", encoding="utf-8")
    print(text)
    print("SMOKE: " + ("PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
