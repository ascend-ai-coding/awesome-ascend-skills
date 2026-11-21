# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Codex PreToolUse adapter for recognizable source-read commands.

Unrelated execution keeps its existing permissions and semantics. This adapter
does not claim to sandbox arbitrary programs, redirections or hosted tools.
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

from . import host_io
from .paths import SKILL_ROOT

READERS = {"rg", "cat", "head", "sed", "grep", "find", "ls", "awk", "tail"}
OPERATORS = {"&&", "||", ";", "|", ">", ">>", "<", "<<", "&"}


def read_requests(words):
    """Recognize only literal source readers; preserve each requested range."""
    import re

    name, arguments = Path(words[0]).name, words[1:]
    options = {"start_line": 1}
    if name == "cat":
        paths = arguments[1:] if arguments and arguments[0] == "--" else arguments
        if any(item.startswith("-") for item in paths):
            raise ValueError("Use bounded read for cat options")
    elif (
        name == "head"
        and len(arguments) >= 3
        and arguments[0] == "-n"
        and arguments[1].isdigit()
    ):
        options["max_lines"], paths = int(arguments[1]), arguments[2:]
    elif name == "sed" and len(arguments) == 3 and arguments[0] == "-n":
        match = re.fullmatch(r"(\d+)(?:,(\d+))?p", arguments[1])
        if not match:
            raise ValueError(
                "Use read --start-line/--max-lines for this sed expression"
            )
        options["start_line"] = int(match[1])
        options["max_lines"] = int(match[2] or match[1]) - options["start_line"] + 1
        paths = arguments[2:]
    else:
        raise ValueError("Use bounded read with explicit files for this reader syntax")
    if not paths or options.get("max_lines", 1) <= 0 or options["start_line"] <= 0:
        raise ValueError("Read requires explicit files and a positive line range")
    return [{"path": path, **options} for path in paths]


def pre_tool(payload, *, policy_path, python=None):
    if payload.get("hook_event_name") != "PreToolUse" or payload.get(
        "tool_name"
    ) not in {"Bash", "exec_command", "shell_command"}:
        return {}
    parameters = payload.get("tool_input", {})
    command = parameters.get("command", parameters.get("cmd"))
    if not isinstance(command, str):
        return {}
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
        lexer.whitespace_split = True
        words = list(lexer)
    except ValueError:
        return {}
    if not words:
        return {}
    # Some hosts expose the already-selected shell as a command prefix.
    if (
        len(words) == 3
        and Path(words[0]).name in {"sh", "bash", "zsh"}
        and words[1] in {"-c", "-lc"}
    ):
        nested = {**payload, "tool_input": {**parameters, "command": words[2]}}
        return pre_tool(nested, policy_path=policy_path, python=python)
    cwd = Path(
        parameters.get("workdir") or parameters.get("cwd") or payload["cwd"]
    ).resolve()
    entry = SKILL_ROOT / "scripts/check.py"
    context_cli = None
    if (
        len(words) > 2
        and Path(words[0]).name.startswith("python")
        and (cwd / words[1]).resolve() == entry
    ):
        index = 2
        while index < len(words) and words[index] in {
            "--pypto-root",
            "--examples-root",
        }:
            index += 2
        if index < len(words) and (
            words[index] in {"search", "read", "status", "api", "surface"}
            or words[index] == "query"
            and any(
                word == "--task" or word.startswith("--task=")
                for word in words[index + 1:]
            )
        ):
            context_cli = words[index]
    direct = Path(words[0]).name in READERS
    chained_read = any(
        index and words[index - 1] in OPERATORS and Path(word).name in READERS
        for index, word in enumerate(words)
    )
    if not direct and not chained_read and not context_cli:
        return {}
    config = host_io.policy(policy_path, workspace=payload.get("cwd"))

    def deny(reason):
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    reason
                    + f" Syntax: {sys.executable} {entry} search --path DIR --pattern"
                    " TEXT; or read --path FILE --start-line N --max-lines N. Query"
                    " a specific unresolved API first. Partial search does not prove absence."
                )[:700],
            }
        }

    if "&&" in words and not any(word in OPERATORS - {"&&"} for word in words):
        segments, current = [], []
        for word in words + ["&&"]:
            if word == "&&":
                segments.append(current)
                current = []
            else:
                current.append(word)
        if all(segment and Path(segment[0]).name == "cat" for segment in segments):
            words = ["cat", *(word for segment in segments for word in segment[1:])]
            direct, chained_read = True, False
        elif all(
            segment and Path(segment[0]).name in {"cat", "head", "sed"}
            for segment in segments
        ):
            if any(
                "$" in word or "`" in word for segment in segments for word in segment
            ):
                return deny("Shell expansion is outside the literal read adapter.")
            try:
                requests = [
                    request
                    for segment in segments
                    for request in read_requests(segment)
                ]
                if len(requests) > 16:
                    raise ValueError("Batch read needs at most 16 requests")
                for request in requests:
                    target = host_io.checked_path(cwd / request["path"], config)
                    if str(target) in config["receipts"]:
                        raise ValueError(
                            "Read a pinned receipt separately through status"
                        )
                    request["path"] = str(target)
            except (ValueError, OSError) as exc:
                return deny(str(exc))
            rewritten = [
                str(Path(python or sys.executable).absolute()),
                str(entry),
                "read",
                "--policy",
                str(Path(policy_path).resolve()),
            ]
            for request in requests:
                rewritten += ["--request", json.dumps(request, separators=(",", ":"))]
            updated = {**parameters, "command": shlex.join(rewritten)}
            updated.pop("cmd", None)
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "allow",
                    "updatedInput": updated,
                }
            }
    if chained_read or any(
        word in OPERATORS or "$" in word or "`" in word for word in words
    ):
        return deny(
            "Split compound source reads into explicit scoped operations; shell"
            " expansion is not interpreted by the context adapter."
        )
    if context_cli:
        if context_cli in {"api", "surface"}:
            if "--help" in words or "-h" in words:
                return {}
            return deny(
                "Bulk API inventories are outside this context profile; use query with"
                " the exact API or conditional route already returned."
            )
        fixed, index = [], 0
        while index < len(words):
            if words[index] == "--policy":
                index += 2
            elif words[index].startswith("--policy="):
                index += 1
            else:
                fixed.append(words[index])
                index += 1
        fixed += ["--policy", str(Path(policy_path).resolve())]
        updated = {**parameters, "command": shlex.join(fixed)}
        updated.pop("cmd", None)
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "updatedInput": updated,
            }
        }
    name, arguments = Path(words[0]).name, words[1:]
    rewritten = [str(Path(python or sys.executable).absolute()), str(entry)]
    if name == "rg":
        options, paths, pattern = [], [], None
        files = False
        index = 0
        while index < len(arguments):
            item = arguments[index]
            if item == "--":
                arguments = arguments[:index] + arguments[index + 1:]
                break
            if item == "--files":
                files = True
            elif item in {
                "-n",
                "--line-number",
                "--no-heading",
                "-H",
                "--with-filename",
            }:
                pass
            elif item in {"-i", "--ignore-case"}:
                options.append("--ignore-case")
            elif item in {"-F", "--fixed-strings"}:
                options.append("--fixed")
            elif item in {"-g", "--glob"} and index + 1 < len(arguments):
                options += ["--glob", arguments[index + 1]]
                index += 1
            elif item.startswith("-"):
                return deny(
                    "This rg option is outside the bounded adapter's supported syntax."
                )
            elif pattern is None and not files:
                pattern = item
            else:
                paths.append(item)
            index += 1
        if index < len(arguments):
            remainder = arguments[index:]
            if not files and pattern is None and remainder:
                pattern, remainder = remainder[0], remainder[1:]
            paths += remainder
        if not paths:
            return deny(
                "Source search requires an explicit directory or file; implicit"
                " whole-workspace enumeration is disabled."
            )
        rewritten += ["search", "--policy", str(Path(policy_path).resolve()), *options]
        if files:
            rewritten.append("--files")
        elif pattern:
            rewritten += ["--pattern=" + pattern]
        else:
            return deny("Missing search pattern.")
    else:
        start, line_count = 1, config["limits"]["max_lines"]
        if name == "cat":
            paths = arguments[1:] if arguments and arguments[0] == "--" else arguments
            if any(item.startswith("-") for item in paths):
                return deny("Use bounded read for cat options.")
        elif (
            name == "head"
            and len(arguments) >= 3
            and arguments[0] == "-n"
            and arguments[1].isdigit()
        ):
            line_count, paths = int(arguments[1]), arguments[2:]
        elif name == "sed" and len(arguments) == 3 and arguments[0] == "-n":
            import re

            match = re.fullmatch(r"(\d+)(?:,(\d+))?p", arguments[1])
            if not match:
                return deny(
                    "Use read --start-line/--max-lines for this sed expression."
                )
            start = int(match[1])
            line_count = int(match[2] or match[1]) - start + 1
            paths = arguments[2:]
        else:
            return deny("Use bounded read with explicit files for this reader syntax.")
        if not paths or line_count <= 0:
            return deny("Read requires explicit files and a positive line range.")
        rewritten += [
            "read",
            "--policy",
            str(Path(policy_path).resolve()),
            "--start-line",
            str(start),
        ]
        if name != "cat":
            rewritten += ["--max-lines", str(line_count)]
    try:
        paths = [host_io.checked_path(cwd / path, config) for path in paths]
    except (ValueError, OSError) as exc:
        return deny(str(exc))
    pinned = [
        config["receipts"][str(path)]
        for path in paths
        if str(path) in config["receipts"]
    ]
    if pinned:
        if len(paths) != 1:
            return deny(
                "Read the pinned receipt separately; its bounded status query waits on the existing request."
            )
        item = pinned[0]
        rewritten = [
            str(Path(python or sys.executable).absolute()),
            str(entry),
            "status",
            "--receipt",
            str(paths[0]),
            "--source",
            item["source"],
            "--request-id",
            item["request_id"],
            "--policy",
            str(Path(policy_path).resolve()),
            "--wait-seconds",
            "20",
        ]
        paths = []
    for path in paths:
        rewritten += ["--path", str(path)]
    updated = {**parameters, "command": shlex.join(rewritten)}
    updated.pop("cmd", None)
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "updatedInput": updated,
        }
    }


def count(path, payload, result):
    """Overwrite bounded per-session counters; keep no command or output history."""
    import fcntl
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(path.read_text()) if path.exists() else {}
        if data.get("session_id") != payload.get("session_id"):
            data = {
                "schema": "pypto-host-counters/1",
                "session_id": payload.get("session_id"),
                "calls": 0,
                "denied": 0,
                "rewritten": 0,
            }
        data["calls"] += 1
        specific = result.get("hookSpecificOutput", {})
        data["denied"] += specific.get("permissionDecision") == "deny"
        data["rewritten"] += "updatedInput" in specific
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(data))
        temporary.replace(path)


def configuration(policy_path, *, python=None, counters=None):
    command = shlex.join(
        [
            str(Path(python or sys.executable).absolute()),
            str(SKILL_ROOT / "scripts/check.py"),
            "host-hook",
            "--policy",
            str(Path(policy_path).resolve()),
        ]
    )
    if counters:
        command += " --counters " + shlex.quote(str(Path(counters).resolve()))
    return {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "^Bash$",
                    "hooks": [{"type": "command", "command": command, "timeout": 5}],
                }
            ]
        }
    }
