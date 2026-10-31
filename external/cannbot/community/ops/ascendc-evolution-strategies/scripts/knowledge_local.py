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
"""本地知识库（local-first）— 自进化知识先落本地，再决定是否提 PR。

目录布局（默认 ``<project>/.cannbot/knowledge-local/``）::

    staging/       本次进化新发现，未审核
    promoted/      审核通过、待提 PR
    rejected/      审核未通过（留档）
    manifest.jsonl 每条知识的来源/状态/指标

设计约束：

* 进化运行期**只写 staging/**，不直接改 cannbot-knowledge 仓，也不改 skill 的
  ``references/strategy-index.md``；
* 检索侧（skill 的 load_playbook / query_strategies）不读本目录；需要让新知识
  参与检索时，先 ``promote``，再由 ``pr --execute`` 同步到知识仓并重建索引；
* 所有落盘内容必须满足知识仓 OKF frontmatter 契约（type/title/description/
  tags/status/sources/created_at/updated_at/platforms）。

CLI::

    knowledge_local.py init   --project <dir>
    knowledge_local.py add    --project <dir> --id X90 --file new.md [--op ...]
    knowledge_local.py list   --project <dir> [--status staging]
    knowledge_local.py promote --project <dir> --id X90
    knowledge_local.py reject  --project <dir> --id X90 [--reason ...]
    knowledge_local.py pr     --project <dir> [--id X90] [--knowledge-root <dir>] [--execute]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from knowledge_root import (  # noqa: E402
    KnowledgeRootError,
    DISCOVERED_REL,
    git_commit,
    resolve_knowledge_root,
)

LOCAL_DIRNAME = Path(".cannbot") / "knowledge-local"
STATUSES = ("staging", "promoted", "rejected")
REQUIRED_FRONTMATTER = (
    "type",
    "title",
    "description",
    "tags",
    "status",
    "sources",
    "created_at",
    "updated_at",
    "platforms",
)
_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*)\s*:")


def local_root(project: str | Path) -> Path:
    return Path(project).expanduser().resolve() / LOCAL_DIRNAME


def init_local(project: str | Path) -> Path:
    root = local_root(project)
    for sub in STATUSES:
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "manifest.jsonl").touch(exist_ok=True)
    (root / "README.md").write_text(
        "# 本地知识库\n\n"
        "- `staging/`：进化运行期写入，未审核。\n"
        "- `promoted/`：审核通过，待提 PR。\n"
        "- `rejected/`：未通过，留档。\n"
        "- `manifest.jsonl`：逐条状态与来源。\n\n"
        "运行期只写 staging；`knowledge_local.py promote` 后由 `pr` 同步到 "
        "cannbot-knowledge 并发起 PR。\n",
        encoding="utf-8",
    )
    return root


def _frontmatter_keys(text: str) -> tuple[dict[str, str], str]:
    """返回 (顶层 key→原始值, 去掉 frontmatter 的正文)。

    只做 key 级校验，不引入 PyYAML 依赖。
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    keys: dict[str, str] = {}
    end = None
    for index in range(1, len(lines)):
        line = lines[index]
        if line.strip() == "---":
            end = index
            break
        match = _KEY_RE.match(line)
        if match:
            keys[match.group(1)] = line.split(":", 1)[1].strip()
    if end is None:
        return keys, text
    body_start = end + 1
    return keys, "\n".join(lines[body_start:]).lstrip("\n")


def validate_okf(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    keys, _ = _frontmatter_keys(text)
    if not keys:
        return ["缺少 YAML frontmatter（文件须以 --- 开头）"]
    return [
        f"缺少 frontmatter 字段: {key}"
        for key in REQUIRED_FRONTMATTER
        if key not in keys
    ]


def _manifest_path(root: Path) -> Path:
    return root / "manifest.jsonl"


def _read_manifest(root: Path) -> list[dict]:
    path = _manifest_path(root)
    if not path.is_file():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _write_manifest(root: Path, entries: list[dict]) -> None:
    path = _manifest_path(root)
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False, sort_keys=True) for e in entries)
        + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _find_entry(entries: list[dict], entry_id: str) -> dict | None:
    for entry in entries:
        if str(entry.get("id", "")).lower() == entry_id.lower():
            return entry
    return None


def add_entry(
    project: str | Path,
    entry_id: str,
    source_file: str | Path,
    *,
    kind: str = "discovered",
    op: str = "",
    run_id: str = "",
    metrics: dict[str, str] | None = None,
) -> dict:
    root = init_local(project)
    src = Path(source_file).expanduser().resolve()
    if not src.is_file():
        raise SystemExit(f"源文件不存在: {src}")
    problems = validate_okf(src)
    if problems:
        raise SystemExit("OKF 校验失败：\n  - " + "\n  - ".join(problems))

    target = root / "staging" / f"{entry_id.lower()}.md"
    if target.exists():
        raise SystemExit(f"staging 已存在同 ID 文件: {target}")
    shutil.copy2(src, target)

    entries = _read_manifest(root)
    if _find_entry(entries, entry_id):
        raise SystemExit(f"manifest 已存在同 ID 记录: {entry_id}")
    entry = {
        "id": entry_id,
        "kind": kind,
        "title": _frontmatter_keys(src.read_text(encoding="utf-8"))[0].get("title", ""),
        "status": "staging",
        "op": op,
        "run_id": run_id,
        "metrics": metrics or {},
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "path": str(target.relative_to(root)),
    }
    entries.append(entry)
    _write_manifest(root, entries)
    print(f"staged: {target}")
    return entry


def promote(project: str | Path, entry_id: str) -> dict:
    root = local_root(project)
    entries = _read_manifest(root)
    entry = _find_entry(entries, entry_id)
    if entry is None:
        raise SystemExit(f"未找到条目: {entry_id}")
    if entry["status"] != "staging":
        raise SystemExit(f"只有 staging 可 promote（当前 status={entry['status']}）")
    src = root / "staging" / f"{entry_id.lower()}.md"
    dst = root / "promoted" / src.name
    if not src.is_file():
        raise SystemExit(f"staging 文件缺失: {src}")
    shutil.move(str(src), str(dst))
    entry.update(
        status="promoted",
        path=str(dst.relative_to(root)),
        promoted_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    )
    _write_manifest(root, entries)
    print(f"promoted: {dst}")
    return entry


def reject(project: str | Path, entry_id: str, reason: str = "") -> dict:
    root = local_root(project)
    entries = _read_manifest(root)
    entry = _find_entry(entries, entry_id)
    if entry is None:
        raise SystemExit(f"未找到条目: {entry_id}")
    current = root / entry["status"] / f"{entry_id.lower()}.md"
    dst = root / "rejected" / current.name
    if current.is_file():
        shutil.move(str(current), str(dst))
    entry.update(
        status="rejected",
        path=str(dst.relative_to(root)),
        reason=reason,
        rejected_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    )
    _write_manifest(root, entries)
    print(f"rejected: {dst}")
    return entry


def list_entries(project: str | Path, status: str | None = None) -> list[dict]:
    root = local_root(project)
    entries = _read_manifest(root)
    if status:
        entries = [e for e in entries if e.get("status") == status]
    for entry in entries:
        print(
            f"{entry.get('status', '?'):9s} {entry.get('id', '?'):8s} "
            f"{entry.get('title', '')[:60]}  ({entry.get('path', '')})"
        )
    if not entries:
        print("(empty)")
    return entries


def _pr_commands(
    root: Path, entries: list[dict], knowledge_root: Path
) -> tuple[list[list[str]], list[list[str]]]:
    """返回 (自动执行命令, 人工执行命令)。

    均为 argv 列表，用 shell=False 执行，防注入/空格断链。
    """
    branch = "knowledge/evolution-local"
    index_py = (
        knowledge_root
        / ".agents"
        / "skills"
        / "knowledge-query"
        / "scripts"
        / "knowledge_index.py"
    )
    # 自动执行：建分支 + 复制 promoted 条目 + 重建索引 + 知识仓校验
    auto: list[list[str]] = [
        ["git", "-C", str(knowledge_root), "checkout", "-b", branch],
    ]
    for entry in entries:
        src = root / entry["path"]
        dst = knowledge_root / DISCOVERED_REL / src.name
        auto.append(["cp", str(src), str(dst)])
    auto.append(["python3", str(index_py), "build"])
    auto.append(["bash", str(knowledge_root / "check.sh")])
    # 人工执行：commit/push 由人确认后自行运行
    manual: list[list[str]] = [
        ["git", "-C", str(knowledge_root), "add"]
        + [
            str(
                (knowledge_root / DISCOVERED_REL / (root / e["path"]).name).relative_to(
                    knowledge_root
                )
            )
            for e in entries
        ],
        [
            "git",
            "-C",
            str(knowledge_root),
            "commit",
            "-m",
            "knowledge(ascendc): add local evolution discoveries",
        ],
        ["git", "-C", str(knowledge_root), "push", "-u", "origin", branch],
    ]
    return auto, manual


def _format_argv(argv: list[str]) -> str:
    return " ".join(shlex.quote(a) for a in argv)


def submit_pr(
    project: str | Path, entry_id: str | None, knowledge_root: str | None, execute: bool
) -> int:
    project_path = Path(project).expanduser().resolve()
    root = local_root(project_path)
    entries = [e for e in _read_manifest(root) if e.get("status") == "promoted"]
    if entry_id:
        wanted = entry_id.lower()
        entries = [e for e in entries if str(e.get("id", "")).lower() == wanted]
    if not entries:
        print("没有 promoted 条目可提 PR（先运行 promote）")
        return 1

    try:
        kroot, source = resolve_knowledge_root(knowledge_root)
    except KnowledgeRootError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    print(f"knowledge root: {kroot} (source={source}, commit={git_commit(kroot)})")

    auto_cmds, manual_cmds = _pr_commands(root, entries, kroot)
    if not execute:
        print(
            "\n# 待执行命令（加 --execute 仅执行复制/索引/校验，不执行 commit/push）："
        )
        for argv in auto_cmds + manual_cmds:
            print(_format_argv(argv))
        return 0

    # 自动执行全部分组一（建分支/复制/重建索引/check.sh）；commit/push 交由人工确认。
    for argv in auto_cmds:
        print(f"$ {_format_argv(argv)}")
        rc = subprocess.run(argv, shell=False, check=False).returncode
        if rc != 0:
            print(f"[ERROR] 命令失败 rc={rc}: {_format_argv(argv)}", file=sys.stderr)
            return rc
    print("\n# 复制、索引重建与 check.sh 校验均已通过，请人工执行：")
    for argv in manual_cmds:
        print(_format_argv(argv))
    return 0


def _parse_metrics(values: list[str]) -> dict[str, str]:
    metrics = {}
    for item in values:
        if "=" in item:
            key, value = item.split("=", 1)
            metrics[key] = value
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="本地知识库（local-first）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="初始化本地库")
    p_init.add_argument("--project", default=os.getcwd())

    p_add = sub.add_parser("add", help="把一条新知识写入 staging")
    p_add.add_argument("--project", default=os.getcwd())
    p_add.add_argument("--id", required=True)
    p_add.add_argument("--file", required=True)
    p_add.add_argument("--kind", default="discovered")
    p_add.add_argument("--op", default="")
    p_add.add_argument("--run-id", default="")
    p_add.add_argument("--metric", action="append", default=[], help="k=v，可重复")

    p_list = sub.add_parser("list", help="列出本地条目")
    p_list.add_argument("--project", default=os.getcwd())
    p_list.add_argument("--status", choices=STATUSES, default=None)

    p_promote = sub.add_parser("promote", help="staging → promoted")
    p_promote.add_argument("--project", default=os.getcwd())
    p_promote.add_argument("--id", required=True)

    p_reject = sub.add_parser("reject", help="→ rejected")
    p_reject.add_argument("--project", default=os.getcwd())
    p_reject.add_argument("--id", required=True)
    p_reject.add_argument("--reason", default="")

    p_pr = sub.add_parser("pr", help="生成/执行提 PR 命令")
    p_pr.add_argument("--project", default=os.getcwd())
    p_pr.add_argument("--id", default=None)
    p_pr.add_argument("--knowledge-root", default=None)
    p_pr.add_argument("--execute", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            print(init_local(args.project))
        elif args.command == "add":
            add_entry(
                args.project,
                args.id,
                args.file,
                kind=args.kind,
                op=args.op,
                run_id=args.run_id,
                metrics=_parse_metrics(args.metric),
            )
        elif args.command == "list":
            list_entries(args.project, args.status)
        elif args.command == "promote":
            promote(args.project, args.id)
        elif args.command == "reject":
            reject(args.project, args.id, args.reason)
        elif args.command == "pr":
            return submit_pr(args.project, args.id, args.knowledge_root, args.execute)
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - CLI 兜底
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
