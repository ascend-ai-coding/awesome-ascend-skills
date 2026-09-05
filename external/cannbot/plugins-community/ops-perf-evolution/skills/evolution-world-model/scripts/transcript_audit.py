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
"""transcript_audit.py — Audit Claude Code subagent transcript JSONL files.

Used by the SubagentStop hook to verify that a subagent actually executed
the work it claims, rather than lying in its final assistant message.

Transcript JSONL schema (observed via SubagentStop hook stdin sample):
  Each line is a JSON object with `type` in {user, assistant, tool_result}.
  `type == "assistant"` rows carry `message.content`, a list of blocks where
  `block.type == "tool_use"` records an actual tool invocation:
    {"type":"tool_use", "name":"Bash", "input": {"command": "..."}}
    {"type":"tool_use", "name":"Write", "input": {"file_path": "..."}}
    {"type":"tool_use", "name":"Read", "input": {"file_path": "..."}}
    {"type":"tool_use", "name":"Task", "input": {"subagent_type":..., "prompt":...}}

CLI:
  list-calls   --transcript PATH [--tool NAME]
    Print "<TOOL>\\t<summary>" lines for each tool call in chronological order.
    summary = command (Bash) | file_path (Read/Write/Edit) | json(input) (other).
    Used by hooks to grep for evidence of specific actions.

  audit-partial --transcript PATH --partial-type {lingxi-partial,ops-partial}
                [--last-msg STR] [--cwd DIR]
    Run the v0.4 anti-cheat audit on an evolution partial subagent.
    Prints violations (one per line, "Sx: ...") to stderr.
    Exit 0 if no violations, 1 otherwise.

Audit rules (v0.4 first cut, extensible):
  S1: lingxi-partial must have invoked `bash <path>/evaluate_ascendc.sh ...`
      via Bash. Passive `cat`/`grep` references don't count.
      ops-partial must have invoked `python <path>/build_ops.py` or
      `evaluate_ops_direct.py`.

Note on dropped S6: an earlier draft also checked that, if last_msg
contained success markers (通过/ok/done/...), the transcript must include a
Write on `evaluation_results.json`. Real-world testing showed this NLP-style
matching has 25% false-positive rate — substrings like `ok` inside `tokens`
or 通过 inside "通过 ... 传递" (verb form) trigger false alarms. S1 alone
catches the core lie (fake completion without running evaluation); main
Stop hook's R2/R7/R8 catch filesystem-level artifact gaps. S6 was overlap +
unreliable, so it was removed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from typing import Iterator

LOGGER = logging.getLogger(__name__)

# 数据输出专用 logger：CLI 结果（JSON/表格/报告）走 stdout（agent 调用协议通道），
# 与 LOGGER（stderr 进度/警告）分离，避免 lint G.LOG.02 误报 print。
DATA_LOGGER = logging.getLogger(f"{__name__}.data")
_DATA_HANDLER = logging.StreamHandler(sys.stdout)
_DATA_HANDLER.setFormatter(logging.Formatter("%(message)s"))
DATA_LOGGER.addHandler(_DATA_HANDLER)
DATA_LOGGER.propagate = False
DATA_LOGGER.setLevel(logging.INFO)


def _iter_assistant_lines(transcript_path: str) -> Iterator[tuple[int, dict]]:
    """逐行解析 transcript，产出 (line_idx, assistant 行的 JSON 对象)。"""
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "assistant":
                yield line_idx, obj


def iter_tool_calls(transcript_path: str) -> Iterator[dict]:
    """Yield each tool_use block from the transcript in chronological order.

    Each yielded dict has keys: name, input, raw_block, line_idx.
    """
    for line_idx, obj in _iter_assistant_lines(transcript_path):
        content = obj.get("message", {}).get("content", []) or []
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                yield {
                    "name": block.get("name", ""),
                    "input": block.get("input", {}) or {},
                    "line_idx": line_idx,
                }


def summarize_call(call: dict) -> str:
    """Build a one-line searchable summary of a tool call."""
    name = call["name"]
    inp = call["input"]
    if name == "Bash":
        return inp.get("command", "")
    if name in ("Read", "Write", "Edit"):
        return inp.get("file_path", "")
    if name == "Task":
        st = inp.get("subagent_type", "")
        desc = inp.get("description", "")
        return f"subagent_type={st} desc={desc}"
    # fallback
    try:
        return json.dumps(inp, ensure_ascii=False)[:200]
    except (TypeError, ValueError):
        return str(inp)[:200]


def cmd_list_calls(args: argparse.Namespace) -> int:
    if not os.path.isfile(args.transcript):
        LOGGER.error("transcript_audit: transcript not found: %s", args.transcript)
        return 2
    try:
        for call in iter_tool_calls(args.transcript):
            if args.tool and call["name"] != args.tool:
                continue
            summary = summarize_call(call)
            DATA_LOGGER.info("%s\t%s", call['name'], summary)
    except BrokenPipeError:
        # Downstream pipe closed (e.g. `| head`) — graceful exit
        try:
            sys.stdout.close()
        except Exception as e:
            # 关闭失败不影响退出结果，仅记录
            LOGGER.debug("stdout close failed: %s", e)
        return 0
    return 0


def _has_call_matching(transcript_path: str, predicate) -> bool:
    """Return True if any tool call satisfies the predicate."""
    for call in iter_tool_calls(transcript_path):
        if predicate(call):
            return True
    return False


def _bash_contains(transcript_path: str, needle: str) -> bool:
    """True if any Bash command in the transcript contains `needle` (substring).

    NOTE: matches passive references too (cat/grep on the script). Use
    `_bash_invokes` for "did it actually execute this?".
    """
    return _has_call_matching(
        transcript_path,
        lambda c: c["name"] == "Bash" and needle in c["input"].get("command", ""),
    )


def _bash_invokes(transcript_path: str, script_basename: str,
                  interpreter_pattern: str) -> bool:
    """True if any Bash command actually *executes* the given script.

    Args:
        script_basename: literal filename like "evaluate_ascendc.sh"
        interpreter_pattern: regex matching the interpreter before the script:
          r'\\b(?:bash|sh)\\s+'    for .sh scripts
          r'\\bpython3?\\s+'        for .py scripts

    Matches:
        bash ops/.../evaluate_ascendc.sh output/...
        python3 -u plugins-community/ops-perf-evolution/skills/evolution-world-model/scripts/build_ops.py
    Does NOT match:
        cat .../evaluate_ascendc.sh
        grep foo build_ops.py
        ls evaluate_ascendc.sh
    """
    pattern = re.compile(
        interpreter_pattern + r"\S*" + re.escape(script_basename) + r"\b"
    )
    for call in iter_tool_calls(transcript_path):
        if call["name"] != "Bash":
            continue
        cmd = call["input"].get("command", "")
        if pattern.search(cmd):
            return True
    return False


def _wrote_path_matching(transcript_path: str, regex: re.Pattern) -> bool:
    """True if any Write/Edit targets a path matching the regex."""
    return _has_call_matching(
        transcript_path,
        lambda c: c["name"] in ("Write", "Edit")
        and regex.search(c["input"].get("file_path", "")),
    )


def _find_implementation_notes(transcript_path: str) -> list[str]:
    """Return all file paths that the subagent Write/Edit'd named implementation_note.txt."""
    notes: list[str] = []
    for call in iter_tool_calls(transcript_path):
        if call["name"] not in ("Write", "Edit"):
            continue
        fp = call["input"].get("file_path", "")
        if fp.endswith("implementation_note.txt"):
            if fp not in notes:
                notes.append(fp)
    return notes


# R11 v3.2: Playbook 完成度 audit
# 期望 implementation_note.txt 格式：
#   ## Strategy: P1 - Double Buffer
#   ### Playbook Step 1: ...
#   ### Playbook Step 2: ...
#   ### Playbook Step 3: ...
#   ### Playbook Step 4: ...
#   ### Playbook Step 5: ...

_STRATEGY_HEADER_RE = re.compile(
    r"^##\s+Strategy:\s*([A-Z]+\d+(?:_\w+)?)\b",
    re.MULTILINE | re.IGNORECASE,
)
_STEP_HEADER_RE = re.compile(
    r"^###\s+Playbook\s+Step\s+(\d+)",
    re.MULTILINE | re.IGNORECASE,
)
REQUIRED_STEPS = {1, 2, 3, 4, 5}


def audit_implementation_note(note_path: str) -> list[str]:
    """Check implementation_note.txt against R11 Playbook completeness rule.

    Returns list of violation strings. Empty list = pass.

    Lenient mode (v3.2 v0.1):
    - If no "## Strategy: PX" header found → skip (老形态 note，不强制升级)
    - If headers found → strict check Steps 1-5 for each strategy
    """
    if not os.path.isfile(note_path):
        return [f"R11: implementation_note.txt not found at {note_path}"]

    try:
        with open(note_path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError as e:
        return [f"R11: cannot read {note_path}: {e}"]

    # 切分 strategy sections
    strategy_matches = list(_STRATEGY_HEADER_RE.finditer(content))
    if not strategy_matches:
        # 老形态，无 Strategy header → 跳过（lenient mode）
        return []

    violations: list[str] = []
    for i, m in enumerate(strategy_matches):
        sid = m.group(1)
        start = m.end()
        end = strategy_matches[i + 1].start() if i + 1 < len(strategy_matches) else len(content)
        section = content[start:end]

        # 找出本 section 内的 Playbook Step header
        step_nums = {int(sm.group(1)) for sm in _STEP_HEADER_RE.finditer(section)}
        missing = REQUIRED_STEPS - step_nums
        if missing:
            sorted_missing = sorted(missing)
            violations.append(
                f"R11: strategy {sid} missing Playbook Step(s) "
                f"{sorted_missing} in implementation_note.txt"
            )

    return violations


def cmd_audit_playbook(args: argparse.Namespace) -> int:
    """Standalone CLI: audit a single implementation_note.txt against R11."""
    violations = audit_implementation_note(args.note_path)
    for v in violations:
        LOGGER.error("%s", v)
    return 1 if violations else 0


def _check_s1_invocation(args: argparse.Namespace) -> list[str]:
    """S1: partial 必须实际执行评估脚本（被动 cat/grep 引用不算）。"""
    if args.partial_type == "lingxi-partial":
        if not _bash_invokes(args.transcript, "evaluate_ascendc.sh",
                             r"\b(?:bash|sh)\s+"):
            return [
                "S1: lingxi-partial did not execute evaluate_ascendc.sh "
                "(no `bash <path>/evaluate_ascendc.sh ...` invocation found; "
                "passive cat/grep references don't count)"
            ]
        return []
    if args.partial_type == "ops-partial":
        ran_build = _bash_invokes(args.transcript, "build_ops.py",
                                  r"\bpython3?\s+")
        ran_eval = _bash_invokes(args.transcript, "evaluate_ops_direct.py",
                                 r"\bpython3?\s+")
        if not (ran_build or ran_eval):
            return [
                "S1: ops-partial did not invoke build_ops.py or "
                "evaluate_ops_direct.py via python (passive refs don't count)"
            ]
    return []


def _check_r12_note_exists(note_paths: list[str], cwd: str) -> list[str]:
    """R12: implementation_note.txt 必写（hard block）。

    只检查"文件存在 + ≥100 字符"，不检查 Playbook 格式（那是 R11 的事）。
    """
    r12_min_len = 100
    if not note_paths:
        return [
            "R12: partial did not Write any implementation_note.txt "
            "(R12 hard block: partial must produce per-variant metadata; "
            "file is required regardless of content format)"
        ]
    issues: list[str] = []
    for note_path in note_paths:
        abs_note = note_path if os.path.isabs(note_path) else os.path.join(cwd, note_path)
        if not os.path.isfile(abs_note):
            issues.append(
                f"R12: claimed Write of {note_path} but file not on disk"
            )
            continue
        try:
            with open(abs_note, encoding="utf-8") as f:
                content_len = len(f.read().strip())
        except OSError:
            issues.append(f"R12: cannot read {abs_note}")
            continue
        if content_len < r12_min_len:
            issues.append(
                f"R12: implementation_note.txt at {note_path} is too short "
                f"(length={content_len}, need ≥{r12_min_len} chars)"
            )
    return issues


def _check_r11_format(note_paths: list[str], cwd: str) -> list[str]:
    """R11 内容格式 audit（仅当有 note 时）。"""
    issues: list[str] = []
    for note_path in note_paths:
        # 路径可能是相对路径，转绝对路径以便 read
        abs_note = note_path if os.path.isabs(note_path) else os.path.join(cwd, note_path)
        if not os.path.isfile(abs_note):
            # transcript 引用但实际文件未持久化 → 跳过（fail open）
            continue
        issues.extend(audit_implementation_note(abs_note))
    return issues


def cmd_audit_partial(args: argparse.Namespace) -> int:
    if not os.path.isfile(args.transcript):
        # Missing transcript — can't audit, fail open (let main Stop hook catch)
        LOGGER.warning("transcript_audit: transcript not found, skipping audit: %s",
                       args.transcript)
        return 0

    if args.partial_type not in ("lingxi-partial", "ops-partial"):
        # Unknown partial type — fail open
        return 0

    issues: list[str] = _check_s1_invocation(args)

    # R11 v3.2: audit Playbook completeness in implementation_note.txt
    # 仅当本次有 Write implementation_note.txt 时才触发；老形态无 note 文件 → skip
    note_paths = _find_implementation_notes(args.transcript)

    # R12 v3.2 C8: implementation_note.txt 必写 (hard block)
    issues.extend(_check_r12_note_exists(note_paths, args.cwd))

    # R11 内容格式 audit (仅当有 note 时)
    issues.extend(_check_r11_format(note_paths, args.cwd))

    for i in issues:
        LOGGER.error("%s", i)
    return 1 if issues else 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Audit Claude Code subagent transcripts for v0.4 anti-cheat checks."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list-calls", help="List all tool calls (name<TAB>summary)")
    p.add_argument("--transcript", required=True)
    p.add_argument("--tool", default=None,
                   help="Filter by tool name (e.g. Bash, Write)")
    p.set_defaults(func=cmd_list_calls)

    p = sub.add_parser("audit-partial", help="Run v0.4 audit on a partial subagent")
    p.add_argument("--transcript", required=True)
    p.add_argument("--partial-type", required=True,
                   choices=["lingxi-partial", "ops-partial"])
    p.add_argument("--last-msg", default="")
    p.add_argument("--cwd", default=os.getcwd())
    p.set_defaults(func=cmd_audit_partial)

    p = sub.add_parser(
        "audit-playbook",
        help="v3.2 R11: audit implementation_note.txt Playbook completeness",
    )
    p.add_argument("--note-path", required=True,
                   help="Path to implementation_note.txt")
    p.set_defaults(func=cmd_audit_playbook)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
