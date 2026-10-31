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
"""Canonical CANNBOT knowledge-root resolution (fail-loud).

本模块是进化 skill 侧访问 cannbot-knowledge 仓的唯一适配点，与
``ascendc-port-orchestrator`` 的 ``briefs/external_kb.py`` 和知识仓
``knowledge-query/scripts/retrieval/root.py`` 的解析顺序保持一致。

解析优先级：

1. 显式 ``--knowledge-root`` / 调用方传入路径
2. ``CANNBOT_KNOWLEDGE_ROOT`` 环境变量
3. 自 cwd（或 ``start``）向上最近的 ``.cannbot/knowledge.env``
4. 兄弟检出 ``<skills-repo>/../cannbot-knowledge``（仅开发布局兜底，打印 WARN）

**只要显式来源（1/2/3）被配置，路径无效就直接抛 ``KnowledgeRootError``，不再静默回退。**
只有完全未配置时才允许第 4 级兜底。

校验布局（与 cannbot-knowledge master 安装器一致）：:

    <root>/knowledge/index.md
    <root>/governance/schemas/registries.yaml
    <root>/knowledge/contrib/ascendc/index.md
    <root>/knowledge/contrib/ascendc/runbooks/performance/
    <root>/knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/
    <root>/knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/

CLI::

    python3 knowledge_root.py --validate --knowledge-root /path/to/cannbot-knowledge
    python3 knowledge_root.py --print
    python3 knowledge_root.py --write-env --project /path/to/project \
        [--knowledge-root /path/to/cannbot-knowledge] [--mode consumer]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ENV_KEY = "CANNBOT_KNOWLEDGE_ROOT"
MODE_KEY = "CANNBOT_KNOWLEDGE_MODE"
COMMIT_KEY = "CANNBOT_KNOWLEDGE_COMMIT"

KNOWLEDGE_REL = Path("knowledge")
CONTRIB_REL = KNOWLEDGE_REL / "contrib" / "ascendc"
PLAYBOOK_REL = CONTRIB_REL / "runbooks" / "performance"
CARDS_REL = CONTRIB_REL / "optimizations" / "evolution_strategies" / "cards"
DISCOVERED_REL = CONTRIB_REL / "optimizations" / "evolution_strategies" / "discovered"

REQUIRED_REL_PATHS = (
    KNOWLEDGE_REL / "index.md",
    Path("governance") / "schemas" / "registries.yaml",
    CONTRIB_REL / "index.md",
    PLAYBOOK_REL,
    CARDS_REL,
    DISCOVERED_REL,
)

_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


class KnowledgeRootError(RuntimeError):
    """知识仓根未配置、配置无效或布局不完整。

    ``configured=True`` 表示用户显式配置过来源（CLI/env/.cannbot），
    调用方不得再做任何静默回退。
    """

    def __init__(self, message: str, *, configured: bool = True) -> None:
        super().__init__(message)
        self.configured = configured


def parse_knowledge_env(path: str | Path) -> dict[str, str]:
    """解析 ``.cannbot/knowledge.env``（去 export 前缀与引号）。"""
    values: dict[str, str] = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _ENV_LINE.match(line)
        if not match:
            continue
        key, value = match.group(1), match.group(2)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value
    return values


def find_project_env(
    start: str | Path | None = None,
) -> tuple[Path, dict[str, str]] | None:
    """自 start（默认 cwd）向上找最近的 ``.cannbot/knowledge.env``。"""
    current = Path(start or os.getcwd()).expanduser().resolve()
    for parent in (current, *current.parents):
        candidate = parent / ".cannbot" / "knowledge.env"
        if candidate.is_file():
            return candidate, parse_knowledge_env(candidate)
    return None


def validate_root(root: str | Path) -> list[str]:
    """返回缺失项（相对路径字符串）；空列表表示布局完整。"""
    base = Path(root).expanduser()
    missing: list[str] = []
    for rel in REQUIRED_REL_PATHS:
        target = base / rel
        if rel in (PLAYBOOK_REL, CARDS_REL, DISCOVERED_REL):
            if not target.is_dir():
                missing.append(str(rel))
        elif not target.is_file():
            missing.append(str(rel))
    return missing


def _validated(source: str, root: str | Path) -> Path:
    candidate = Path(root).expanduser()
    try:
        candidate = candidate.resolve()
    except OSError as exc:  # pragma: no cover - 极端文件系统错误
        raise KnowledgeRootError(
            f"无法解析知识仓路径（source={source}）：{root}：{exc}"
        ) from exc
    missing = validate_root(candidate)
    if missing:
        detail = "；缺失：" + ", ".join(missing)
        raise KnowledgeRootError(
            f"知识仓布局不完整（source={source}）：{candidate}{detail}。"
            "请指向 cannbot-knowledge 完整 checkout（含 knowledge/、governance/、"
            ".agents/skills/knowledge-query/）。"
        )
    return candidate


def _knowledge_candidates(script_file: str | Path | None = None) -> list[Path]:
    """收集可能的兄弟检出路径。

    兼容新布局 community/ops 与旧布局 plugins-community/<plugin>/skills。
    """
    here = Path(script_file or __file__).resolve()
    candidates: list[Path] = []

    # 1) git 仓根：<repo>/../cannbot-knowledge 与 <repo>/cannbot-knowledge
    try:
        proc = subprocess.run(
            ["git", "-C", str(here.parent), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            repo = Path(proc.stdout.strip()).resolve()
            candidates.extend(
                [repo.parent / "cannbot-knowledge", repo / "cannbot-knowledge"]
            )
    except (OSError, subprocess.SubprocessError):
        pass

    # 2) 逐级祖先目录（两种常见相对位置都试）
    for parent in here.parents:
        candidates.append(parent / "cannbot-knowledge")
        candidates.append(parent.parent / "cannbot-knowledge")

    seen: set[str] = set()
    unique: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def sibling_knowledge_root(script_file: str | Path | None = None) -> Path:
    """开发布局兜底：在祖先目录中寻找布局完整的 cannbot-knowledge。"""
    candidates = _knowledge_candidates(script_file)
    for candidate in candidates:
        if not validate_root(candidate):
            return candidate
    return (
        candidates[0]
        if candidates
        else Path(__file__).resolve().parent / "cannbot-knowledge"
    )


def resolve_knowledge_root(
    explicit: str | Path | None = None,
    *,
    start: str | Path | None = None,
    allow_sibling: bool = True,
    script_file: str | Path | None = None,
) -> tuple[Path, str]:
    """按优先级解析知识仓根，返回 ``(root, source)``。

    ``source`` ∈ ``{"cli", "env", "project_env", "sibling"}``。
    显式来源无效时抛 :class:`KnowledgeRootError`（fail-loud）。
    """
    if explicit:
        return _validated("cli", explicit), "cli"

    env_root = os.environ.get(ENV_KEY, "").strip()
    if env_root:
        # 显式环境变量配了就必须有效，禁止静默回退。
        return _validated("env", env_root), "env"

    project_env = find_project_env(start)
    if project_env is not None:
        env_path, values = project_env
        configured = values.get(ENV_KEY, "").strip()
        if configured:
            candidate = Path(configured)
            if not candidate.is_absolute():
                candidate = env_path.parent / candidate
            return _validated(f"project_env:{env_path}", candidate), "project_env"

    if allow_sibling:
        candidate = sibling_knowledge_root(script_file)
        if not validate_root(candidate):
            print(
                f"[WARN] CANNBOT_KNOWLEDGE_ROOT 未配置，回退到兄弟检出：{candidate}。"
                "生产环境请通过 init.sh --knowledge-root 或 .cannbot/knowledge.env 显式绑定。",
                file=sys.stderr,
            )
            return candidate.resolve(), "sibling"

    searched = [
        "CLI --knowledge-root（未提供）",
        f"{ENV_KEY}（未设置）",
        ".cannbot/knowledge.env（未找到）",
        f"兄弟检出 {sibling_knowledge_root(script_file)}（不存在或布局不完整）"
        if allow_sibling
        else "兄弟检出（已禁用）",
    ]
    raise KnowledgeRootError(
        "无法定位 cannbot-knowledge 仓。请显式指定 --knowledge-root、设置 "
        f"{ENV_KEY}，或在项目根运行 cannbot-knowledge/install.sh（会写入 "
        ".cannbot/knowledge.env）。已尝试：" + "；".join(searched),
        configured=False,
    )


def playbook_dir(root: str | Path) -> Path:
    return Path(root) / PLAYBOOK_REL


def cards_dir(root: str | Path) -> Path:
    return Path(root) / CARDS_REL


def discovered_dir(root: str | Path) -> Path:
    return Path(root) / DISCOVERED_REL


def git_commit(root: str | Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def write_project_env(
    project: str | Path,
    root: str | Path,
    *,
    mode: str = "consumer",
    extra: dict[str, str] | None = None,
) -> Path:
    """原子写入 ``<project>/.cannbot/knowledge.env``。"""
    project_path = Path(project).expanduser().resolve()
    root_path = _validated("write_project_env", root)
    env_dir = project_path / ".cannbot"
    env_dir.mkdir(parents=True, exist_ok=True)
    target = env_dir / "knowledge.env"

    lines = [
        "# Generated by cannbot knowledge bootstrap. Do not edit by hand.",
        f"export {ENV_KEY}='{root_path}'",
        f"{MODE_KEY}='{mode}'",
    ]
    commit = git_commit(root_path)
    if commit:
        lines.append(f"{COMMIT_KEY}='{commit}'")
    for key, value in (extra or {}).items():
        lines.append(f"{key}='{value}'")
    payload = "\n".join(lines) + "\n"

    fd, tmp = tempfile.mkstemp(dir=str(env_dir), prefix=".knowledge.env.", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return target


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="解析/校验 CANNBOT 知识仓根")
    parser.add_argument("--knowledge-root", default=None, help="显式知识仓根")
    parser.add_argument(
        "--project", default=None, help="写入 .cannbot/knowledge.env 的项目根"
    )
    parser.add_argument(
        "--mode", default="consumer", choices=("consumer", "contributor")
    )
    parser.add_argument("--validate", action="store_true", help="只校验")
    parser.add_argument(
        "--print", dest="print_root", action="store_true", help="打印解析结果"
    )
    parser.add_argument("--write-env", action="store_true", help="写入项目配置")
    args = parser.parse_args(argv)

    try:
        root, source = resolve_knowledge_root(args.knowledge_root)
    except KnowledgeRootError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if args.print_root or args.validate or not args.write_env:
        print(f"root={root}")
        print(f"source={source}")
        commit = git_commit(root)
        if commit:
            print(f"commit={commit}")
        print(f"playbooks={playbook_dir(root)}")
        print(f"cards={cards_dir(root)}")
        print(f"discovered={discovered_dir(root)}")

    if args.write_env:
        project = args.project or os.getcwd()
        target = write_project_env(project, root, mode=args.mode)
        print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
