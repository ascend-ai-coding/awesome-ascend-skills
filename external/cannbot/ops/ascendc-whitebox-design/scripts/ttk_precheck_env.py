#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Precheck whether local CANN + TTK kernel environment can run."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Optional

_logger = logging.getLogger(__name__)


REPORT_FILE = "ttk_precheck_report.json"
ENV_VARS = ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH", "ASCEND_AICPU_PATH")
DEFAULT_CANN_ROOTS = (
    "/usr/local/Ascend/ascend-toolkit/latest",
    "/usr/local/Ascend/ascend-toolkit",
)
SETENV_NAMES = ("set_env.sh", "setenv.sh")


def command_tail(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def run_command(command: list[str], cwd: Path) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        return {
            "cmd": " ".join(shlex.quote(part) for part in command),
            "cwd": str(cwd),
            "returncode": proc.returncode,
            "stdout_tail": command_tail(proc.stdout),
            "stderr_tail": command_tail(proc.stderr),
        }
    except Exception as exc:  # noqa: BLE001 - report precheck failure details
        return {
            "cmd": " ".join(shlex.quote(part) for part in command),
            "cwd": str(cwd),
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": str(exc),
        }


def run_shell(command: str, cwd: Path) -> dict[str, Any]:
    return run_command(["bash", "-lc", command], cwd)


def passed(result: dict[str, Any]) -> bool:
    return result.get("returncode") == 0


def collect_setenv_candidates(explicit_path: Optional[str]) -> list[Path]:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())

    roots: list[str] = []
    for var in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        value = os.environ.get(var)
        if value:
            roots.append(value)
    roots.extend(DEFAULT_CANN_ROOTS)

    for root in roots:
        for name in SETENV_NAMES:
            candidates.append(Path(root).expanduser() / name)

    seen: set[str] = set()
    unique: list[Path] = []
    for candidate in candidates:
        resolved = str(candidate)
        if resolved not in seen:
            seen.add(resolved)
            unique.append(candidate)
    return unique


def env_snapshot() -> dict[str, str]:
    return {var: os.environ[var] for var in ENV_VARS if os.environ.get(var)}


def check_ops_test_kit(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "failed", "detail": f"path does not exist: {path}"}
    if not path.is_dir():
        return {"status": "failed", "detail": f"path is not a directory: {path}"}
    ttk_main = path / "ttk" / "__main__.py"
    if not ttk_main.is_file():
        return {"status": "failed", "detail": f"missing ttk module entry: {ttk_main}"}
    return {"status": "passed", "detail": str(path)}


def try_ttk_current_env(ops_test_kit_path: Path) -> dict[str, Any]:
    result = run_command(["python3", "-m", "ttk", "kernel", "--help"], ops_test_kit_path)
    if passed(result):
        result["stdout_tail"] = ""
        result["stderr_tail"] = ""
    return {
        "status": "passed" if passed(result) else "failed",
        "mode": "current_env",
        "command": result,
    }


def try_ttk_with_setenv(ops_test_kit_path: Path, setenv_candidates: list[Path]) -> dict[str, Any]:
    attempts = []
    for candidate in setenv_candidates:
        if not candidate.is_file():
            attempts.append({"setenv_path": str(candidate), "status": "missing"})
            continue
        shell_cmd = f"source {shlex.quote(str(candidate))} && python3 -m ttk kernel --help"
        result = run_shell(shell_cmd, ops_test_kit_path)
        attempt = {
            "setenv_path": str(candidate),
            "status": "passed" if passed(result) else "failed",
            "command": result,
        }
        if passed(result):
            result["stdout_tail"] = ""
            result["stderr_tail"] = ""
        attempts.append(attempt)
        if passed(result):
            return {
                "status": "passed_after_source",
                "mode": "source_setenv",
                "setenv_path": str(candidate),
                "attempts": attempts,
            }
    return {"status": "failed", "mode": "source_setenv", "attempts": attempts}


def build_kernel_gate(report: dict[str, Any]) -> dict[str, str]:
    ops_check = report.get("checks", {}).get("ops_test_kit_path", {})
    if ops_check.get("status") != "passed":
        return {"status": "skipped", "reason": "env_unavailable"}

    cann_env = report.get("checks", {}).get("cann_env", {})
    if cann_env.get("status") not in ("passed", "passed_after_source"):
        return {"status": "skipped", "reason": "env_unavailable"}

    return {"status": "passed", "reason": "passed"}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    ops_test_kit_path = Path(args.ops_test_kit_path).resolve()
    report: dict[str, Any] = {
        "status": "failed",
        "checks": {},
        "env": env_snapshot(),
    }

    ops_check = check_ops_test_kit(ops_test_kit_path)
    report["checks"]["ops_test_kit_path"] = ops_check
    if ops_check["status"] != "passed":
        report["kernel_gate"] = build_kernel_gate(report)
        return report

    current = try_ttk_current_env(ops_test_kit_path)
    report["checks"]["ttk_kernel_current_env"] = current
    if current["status"] == "passed":
        report["checks"]["cann_env"] = {
            "status": "passed",
            "mode": "current_env",
            "detail": "python3 -m ttk kernel --help succeeded in current environment",
        }
        report["kernel_gate"] = build_kernel_gate(report)
        if report["kernel_gate"]["status"] == "passed":
            report["status"] = "passed"
        return report

    setenv_candidates = collect_setenv_candidates(args.setenv_path)
    source_result = try_ttk_with_setenv(ops_test_kit_path, setenv_candidates)
    report["checks"]["ttk_kernel_after_setenv"] = source_result
    if source_result["status"] == "passed_after_source":
        report["checks"]["cann_env"] = {
            "status": "passed_after_source",
            "mode": "source_setenv",
            "setenv_path": source_result["setenv_path"],
            "detail": "python3 -m ttk kernel --help succeeded after sourcing CANN setenv",
        }
    else:
        report["checks"]["cann_env"] = {
            "status": "failed",
            "mode": "current_env_or_source_setenv",
            "detail": "TTK kernel command failed in current environment and after trying available CANN setenv scripts",
        }
    report["kernel_gate"] = build_kernel_gate(report)
    if report["kernel_gate"]["status"] == "passed":
        report["status"] = "passed"
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precheck local CANN + TTK kernel environment.")
    parser.add_argument(
        "--whitebox-dir", required=True,
        help="Directory where ttk_precheck_report.json will be written.",
    )
    parser.add_argument("--ops-test-kit-path", required=True, help="Path to ops-test-kit repository.")
    parser.add_argument("--setenv-path", default=None, help="Optional explicit CANN set_env.sh or setenv.sh path.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    whitebox_dir = Path(args.whitebox_dir).resolve()
    whitebox_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args)
    report_path = whitebox_dir / REPORT_FILE
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    if report["status"] != "passed":
        _logger.info(f"FAIL: TTK environment precheck failed, see {report_path}")
        return
    _logger.info(f"PASS: TTK environment precheck passed, see {report_path}")


if __name__ == "__main__":
    main()
