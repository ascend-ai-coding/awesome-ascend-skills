#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""收集 pytest nodeid，可选择执行测试，并写入 ST 执行证据 JSON。

该工具只记录运行时事实，不判断测试判定基准、覆盖矩阵或算子契约是否可信。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import time
from typing import Any


RECORDED_ENV = (
    "TILELANG_DEFAULT_TARGET",
    "OPS_TILELANG_TEST_LEVEL",
    "CUDA_VISIBLE_DEVICES",
    "ASCEND_RT_VISIBLE_DEVICES",
    "CPLUS_INCLUDE_PATH",
)
RESULT_WORDS = (
    "passed",
    "failed",
    "skipped",
    "deselected",
    "xfailed",
    "xpassed",
    "error",
    "errors",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_value(repo_root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 else None


def run_command(
    command: list[str], repo_root: Path, timeout_seconds: int
) -> dict[str, Any]:
    child_env = os.environ.copy()
    child_env["PY_COLORS"] = "0"
    child_env["NO_COLOR"] = "1"
    started_at = utc_now()
    start = time.monotonic()
    print(f"正在运行：{shlex.join(command)}", flush=True)
    try:
        result = subprocess.run(
            command,
            cwd=repo_root,
            env=child_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
        output = result.stdout
        returncode: int | None = result.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        captured = exc.stdout or ""
        output = (
            captured.decode(errors="replace")
            if isinstance(captured, bytes)
            else captured
        )
        returncode = None
        timed_out = True
    duration = time.monotonic() - start
    if output:
        print(output, end="" if output.endswith("\n") else "\n")
    return {
        "command": command,
        "started_at": started_at,
        "duration_seconds": round(duration, 3),
        "returncode": returncode,
        "timed_out": timed_out,
        "output": output,
    }


def extract_nodeids(output: str) -> list[str]:
    nodeids: list[str] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if "::" not in line or line.startswith(("=", "<", "WARNING", "ERROR")):
            continue
        if line not in nodeids:
            nodeids.append(line)
    return nodeids


def extract_counts(output: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    pattern = re.compile(r"(?<!\w)(\d+)\s+(" + "|".join(RESULT_WORDS) + r")\b")
    for number, label in pattern.findall(output.lower()):
        normalized = "error" if label == "errors" else label
        counts[normalized] = max(counts.get(normalized, 0), int(number))
    return counts


def collection_args(pytest_args: list[str]) -> list[str]:
    """保留测试选择参数，同时统一 nodeid 输出的详细程度。"""
    return [
        item
        for item in pytest_args
        if item not in {"--quiet", "--verbose"}
        and not re.fullmatch(r"-(?:q+|v+)", item)
    ]


def classify_execution(run: dict[str, Any]) -> str:
    if run["timed_out"]:
        return "timeout"
    returncode = run["returncode"]
    output = str(run["output"]).lower()
    counts = extract_counts(str(run["output"]))
    if returncode == 0:
        if counts.get("passed", 0) == 0 and counts.get("skipped", 0) > 0:
            return "all_skipped"
        return "pytest_passed"
    if returncode == 5:
        return "no_tests"
    if any(
        marker in output
        for marker in ("modulenotfounderror", "no module named", "importerror:")
    ):
        return "dependency_error"
    if any(
        marker in output
        for marker in (
            "device compilation failed",
            "compilation failed",
            "compile error",
            "bisheng",
        )
    ):
        return "compile_error"
    if any(
        marker in output
        for marker in (
            "out of memory",
            "device not found",
            "no npu",
            "device is not available",
            "device lost",
            "failed to start the device",
            "tsdopen failed",
            "device retain error",
            "hdc link",
            "507033",
        )
    ):
        return "device_or_resource_error"
    # 仓库的 xdist 快速失败插件可能在真实测试失败后返回退出码 2。
    # 此时应优先依据 pytest 报告的结果数量判断，不能直接按退出码 2
    # 通常表示的“中断”进行分类。
    if counts.get("failed", 0) > 0 or counts.get("error", 0) > 0:
        return "test_failure_or_error"
    if returncode == 2:
        return "interrupted"
    if returncode == 3:
        return "pytest_internal_error"
    if returncode == 4:
        return "pytest_usage_error"
    return "test_failure_or_error"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="执行证据 JSON 路径；相对路径以仓库根目录为基准",
    )
    parser.add_argument("--label", default="tilelang-op-test-design")
    parser.add_argument(
        "--device", default="未指定", help="便于阅读的目标设备说明，例如 Ascend NPU"
    )
    parser.add_argument(
        "--backend", default="未指定", help="便于阅读的目标编译器或后端说明"
    )
    parser.add_argument("--seed-note", default="使用仓库默认设置")
    parser.add_argument(
        "--collect-only", action="store_true", help="只记录测试收集结果，不执行测试"
    )
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "pytest_args", nargs=argparse.REMAINDER, help="`--` 之后的参数会原样传给 pytest"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    pytest_args = list(args.pytest_args)
    if pytest_args and pytest_args[0] == "--":
        pytest_args.pop(0)
    if not pytest_args:
        print(
            "错误：请在 `--` 后提供 pytest 文件、nodeid 或测试选择条件", file=sys.stderr
        )
        return 2
    if args.timeout_seconds <= 0:
        print("错误：--timeout-seconds 必须为正数", file=sys.stderr)
        return 2
    if any(item == "--collect-only" for item in pytest_args):
        print(
            "错误：请使用本工具的 --collect-only，不要将该参数直接传给 pytest",
            file=sys.stderr,
        )
        return 2

    output_path = args.output if args.output.is_absolute() else repo_root / args.output
    status = git_value(repo_root, "status", "--short")
    report: dict[str, Any] = {
        "schema_version": 1,
        "label": args.label,
        "created_at": utc_now(),
        "repository": {
            "root": str(repo_root),
            "commit": git_value(repo_root, "rev-parse", "HEAD"),
            "worktree_dirty": bool(status),
            "worktree_status": status.splitlines() if status else [],
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: distribution_version(name)
                for name in ("pytest", "torch", "torch-npu", "tilelang")
            },
            "selected_variables": {name: os.environ.get(name) for name in RECORDED_ENV},
            "intended_device": args.device,
            "intended_backend": args.backend,
            "seed_note": args.seed_note,
        },
        "pytest_args": pytest_args,
        "limitations": [
            "pytest 命令通过不能证明其契约、判定基准、断言或覆盖可信。",
            "自动失败分类采用保守策略，必须结合首次失败信息进行复核。",
        ],
    }

    collect_command = [
        sys.executable,
        "-m",
        "pytest",
        *collection_args(pytest_args),
        "--collect-only",
        "-q",
    ]
    collection = run_command(collect_command, repo_root, args.timeout_seconds)
    collection["nodeids"] = extract_nodeids(str(collection["output"]))
    collection["collected_count"] = len(collection["nodeids"])
    report["collection"] = collection

    if (
        collection["timed_out"]
        or collection["returncode"] != 0
        or not collection["nodeids"]
    ):
        report["automated_conclusion"] = "NOT_VERIFIED"
        report["execution"] = None
        exit_code = 1
    elif args.collect_only:
        report["automated_conclusion"] = "COLLECTED_NOT_EXECUTED"
        report["execution"] = None
        exit_code = 0
    else:
        run_command_line = [sys.executable, "-m", "pytest", *pytest_args]
        execution = run_command(run_command_line, repo_root, args.timeout_seconds)
        execution["counts"] = extract_counts(str(execution["output"]))
        execution["classification"] = classify_execution(execution)
        report["execution"] = execution
        if (
            execution["classification"] == "pytest_passed"
            and execution["counts"].get("passed", 0) > 0
        ):
            report["automated_conclusion"] = "EXECUTED_PASS_REQUIRES_CREDIBILITY_REVIEW"
            exit_code = 0
        elif execution["classification"] in {"all_skipped", "no_tests"}:
            report["automated_conclusion"] = "NOT_VERIFIED"
            exit_code = 1
        else:
            report["automated_conclusion"] = "EXECUTION_FAILED_OR_BLOCKED"
            exit_code = 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"执行证据已写入 {output_path}")
    print(f"结论：{report['automated_conclusion']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
