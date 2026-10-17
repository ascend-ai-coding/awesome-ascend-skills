#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Stateless interface probe for npu-objdump-toolkit.

纯函数式检查：(当前安装的 -h 输出, references/interface.json) -> 结论。
无缓存、无 accept 流程：任何会话、任何机器，同样输入得到同样答案。
skill 文档与安装的双向漂移都能发现：
- missing-options  安装比 skill 记载的必备面更旧/不兼容
- newer-install    安装出现了 skill 未记载的选项（skill 文档落后）
- ok               记载的必备面齐备；可选能力缺失只提示不算故障
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

EXIT_OK = 0
EXIT_ATTENTION = 3  # 不用 2：与 argparse 参数错误码区分
EXIT_ERROR = 1

BUILTIN_OPTIONS = {"-h", "--help"}

OPTION_TOKEN = re.compile(r"(?<![\w<])-{1,2}[A-Za-z][A-Za-z-]*")
LOGGER = logging.getLogger(__name__)


def skill_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def load_interface(path: Path) -> Dict[str, Any]:
    interface = json.loads(path.read_text(encoding="utf-8"))
    if interface.get("schema_version") != 1:
        raise ValueError("unsupported interface schema in %s" % path)
    return interface


def run_tool(command: str, timeout: int = 10) -> Dict[str, Any]:
    try:
        process = subprocess.run(
            [command, "-h"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"returncode": None, "stdout": "", "stderr": str(error)}
    return {
        "returncode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }


def module_source() -> Optional[str]:
    """用当前解释器探测模块来源，避免通过 PATH 启动非预期程序。"""
    try:
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, msobjdump; sys.stdout.write(msobjdump.__file__)",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if process.returncode != 0:
        return None
    return process.stdout.strip() or None


def options_in_help(help_text: str) -> List[str]:
    """从帮助文本提取选项名；优先只看 options 段，缺段时回退全文。"""
    section = ""
    in_options = False
    for line in help_text.splitlines():
        stripped = line.strip()
        if stripped in ("options:", "optional arguments:"):
            in_options = True
            continue
        if in_options:
            if stripped and not line.startswith((" ", "\t")):
                break
            section += line + "\n"
    tokens = OPTION_TOKEN.findall(section or help_text)
    return sorted(set(tokens) - BUILTIN_OPTIONS)


def documented_options(interface: Dict[str, Any]) -> List[Dict[str, Any]]:
    return interface.get("cli", {}).get("options", [])


def resolve_command(command: str) -> Optional[str]:
    """命令名走 which；显式路径直接校验可执行。"""
    if "/" in command:
        path = Path(command).expanduser()
        if path.is_file() and path.stat().st_mode & 0o111:
            return str(path.resolve())
        return None
    return shutil.which(command)


def option_name_sets(documented: List[Dict[str, Any]]) -> tuple:
    names = set()
    required_names = set()
    optional_names = set()
    for option in documented:
        current_names = set(option.get("names", []))
        names.update(current_names)
        if option.get("optional_capability"):
            optional_names.update(current_names)
        else:
            required_names.update(current_names)
    return names, required_names, optional_names


def tool_context(interface: Dict[str, Any], tool: Optional[str]) -> tuple:
    commands = interface.get("commands", {})
    primary = commands.get("primary", "npu-objdump")
    compat = commands.get("compat", "msobjdump")
    tool_presence = {
        primary: shutil.which(primary),
        compat: shutil.which(compat),
    }
    resolved_tool = tool or (primary if tool_presence.get(primary) else compat)
    resolved_path = resolve_command(resolved_tool) if resolved_tool else None
    return primary, compat, tool_presence, resolved_tool, resolved_path


def record_option_drift(
    result: Dict[str, Any], interface: Dict[str, Any], present: List[str]
) -> None:
    names, required_names, optional_names = option_name_sets(
        documented_options(interface)
    )
    present_names = set(present)
    missing = sorted(required_names - present_names)
    optional_missing = sorted(optional_names - present_names)
    undocumented = sorted(present_names - names)

    result["missing_options"] = missing
    result["optional_missing_options"] = optional_missing
    result["undocumented_options"] = undocumented
    result["limitations"] = []
    if optional_missing:
        result["limitations"].append(
            "optional capability not present in this install: %s (documented for "
            "newer msobjdump; report the limitation instead of using it)"
            % ", ".join(optional_missing)
        )


def record_probe_status(result: Dict[str, Any]) -> None:
    missing = result["missing_options"]
    undocumented = result["undocumented_options"]
    if missing:
        result["status"] = "missing-options"
        result["message"] = (
            "documented required options absent from this install: %s" % missing
        )
    elif undocumented:
        result["status"] = "newer-install"
        result["message"] = (
            "install exposes options the skill does not document: %s; the skill "
            "snapshot is stale — check asc-tools for interface changes before "
            "using them" % undocumented
        )
    else:
        result["status"] = "ok"


def probe(interface: Dict[str, Any], tool: Optional[str]) -> Dict[str, Any]:
    primary, compat, tool_presence, resolved_tool, resolved_path = tool_context(
        interface, tool
    )

    result: Dict[str, Any] = {
        "tool": resolved_tool,
        "tool_resolved": resolved_path,
        "tool_presence": tool_presence,
    }
    if resolved_tool is None or resolved_path is None:
        result["status"] = "unavailable"
        result["message"] = (
            "no usable msobjdump command (checked: %s, %s%s); "
            "source the matching CANN environment first"
            % (primary, compat, ", %s" % tool if tool else "")
        )
        return result

    LOGGER.debug("probing executable: %s", resolved_path)
    help_result = run_tool(resolved_tool)
    result["returncode"] = help_result["returncode"]
    result["module_file"] = module_source()
    if help_result["returncode"] != 0:
        result["status"] = "failed"
        result["message"] = help_result["stderr"].strip() or "msobjdump -h failed"
        return result

    help_text = help_result["stdout"] + help_result["stderr"]
    present = options_in_help(help_text)
    result["present_options"] = present
    record_option_drift(result, interface, present)
    record_probe_status(result)
    return result


def human_summary(result: Dict[str, Any]) -> str:
    lines = ["npu-objdump-toolkit interface check: %s" % result["status"]]
    presence = result.get("tool_presence", {})
    lines.append(
        "tool: %s (%s)"
        % (result.get("tool"), result.get("tool_resolved") or "not on PATH")
    )
    lines.append(
        "commands: %s"
        % ", ".join(
            "%s=%s" % (name, path or "absent") for name, path in presence.items()
        )
    )
    for option in result.get("missing_options", []):
        lines.append("  missing option: %s" % option)
    for option in result.get("undocumented_options", []):
        lines.append("  undocumented option present: %s" % option)
    for limitation in result.get("limitations", []):
        lines.append("  %s" % limitation)
    if result.get("module_file"):
        lines.append("python module: %s" % result["module_file"])
    if result.get("message"):
        lines.append("note: %s" % result["message"])
    return "\n".join(lines)


def configure_logging() -> None:
    """Keep CLI output on stdout with no logging prefix."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.handlers.clear()
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def write_stdout(message: str) -> None:
    LOGGER.info("%s", message)


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(
        description=(
            "Stateless probe: compare the installed msobjdump CLI surface with "
            "the skill's interface snapshot."
        )
    )
    parser.add_argument(
        "--tool",
        help="msobjdump/npu-objdump command to probe; defaults to auto-discovery",
    )
    parser.add_argument(
        "--interface",
        type=Path,
        default=skill_dir() / "references/interface.json",
        help="interface snapshot path (default: %(default)s)",
    )
    parser.add_argument(
        "--json", action="store_true", help="print machine-readable JSON"
    )
    args = parser.parse_args()

    try:
        interface = load_interface(args.interface)
    except (OSError, ValueError) as error:
        result = {
            "status": "error",
            "message": "cannot load interface snapshot %s: %s"
            % (args.interface, error),
        }
        write_stdout(json.dumps(result, ensure_ascii=False, indent=2))
        return EXIT_ERROR

    result = probe(interface, args.tool)
    if args.json:
        write_stdout(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        write_stdout(human_summary(result))
    if result["status"] == "ok":
        return EXIT_OK
    if result["status"] in ("missing-options", "newer-install"):
        return EXIT_ATTENTION
    return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
