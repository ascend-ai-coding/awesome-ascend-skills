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
"""Run the white-box TTK kernel CSV module without LLM orchestration.

This wrapper intentionally contains only workflow orchestration. CSV field
mapping, validation, and environment precheck remain delegated to the existing
fixed scripts so the normal path can run as a single command.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)


REPORT_FILE = "ttk_module_report.json"
PRECHECK_FILE = "ttk_precheck_report.json"


@dataclass
class KernelRunCase:
    csv_path: Path
    result_path: Path
    case_name: str


@dataclass
class StepConfig:
    label: str
    fail_reason: str
    fail_msg: str


@dataclass
class KernelExecEnv:
    ops_test_kit_path: Path
    setenv_path: str | None


@dataclass
class CsvGenPaths:
    generate_script: Path
    low_cases: Path
    high_cases: Path
    low_csv: Path
    high_csv: Path


@dataclass
class PrecheckPaths:
    precheck_script: Path
    whitebox_dir: Path
    ops_test_kit_path: Path
    precheck_report: Path


def now_s() -> float:
    return time.perf_counter()


def command_string(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def tail(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def run_command(command: list[str], cwd: Path | None = None) -> dict[str, Any]:
    start = now_s()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return {
            "cmd": command_string(command),
            "cwd": str(cwd) if cwd else os.getcwd(),
            "returncode": proc.returncode,
            "duration_s": round(now_s() - start, 3),
            "stdout_tail": tail(proc.stdout),
            "stderr_tail": tail(proc.stderr),
        }
    except Exception as exc:  # noqa: BLE001 - surfaced in report
        return {
            "cmd": command_string(command),
            "cwd": str(cwd) if cwd else os.getcwd(),
            "returncode": None,
            "duration_s": round(now_s() - start, 3),
            "stdout_tail": "",
            "stderr_tail": str(exc),
        }


def run_shell(command: str, cwd: Path | None = None) -> dict[str, Any]:
    result = run_command(
        ["bash", "-lc", command],  # noqa: G.EDV.04 - shell required for source builtins
        cwd=cwd,
    )
    result["cmd"] = command
    return result


def fail_step(step: dict[str, Any], reason: str) -> dict[str, Any]:
    step["status"] = "failed"
    step["reason"] = reason
    return step


def pass_step(step: dict[str, Any]) -> dict[str, Any]:
    step["status"] = "passed"
    return step


def skip_step(step: dict[str, Any], reason: str) -> dict[str, Any]:
    step["status"] = "skipped"
    step["reason"] = reason
    return step


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_cases(path: Path) -> int:
    data = load_json(path)
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        return len(data["cases"])
    raise RuntimeError(f"{path} is not a case list")


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        return max(sum(1 for _ in csv.reader(f)) - 1, 0)


def pick_case_name(csv_path: Path) -> str:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"{csv_path} contains no testcase rows")
    names = [row.get("testcase_name", "") for row in rows]
    if "case00001" in names:
        return "case00001"
    for name in names:
        if name and "empty" not in name.lower():
            return name
    return names[len(names) // 2]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _select_result_row(rows: list[dict[str, str]], testcase_name: str) -> dict[str, str] | None:
    for row in rows:
        if row.get("testcase_name") == testcase_name:
            return row
    return None


def _dyn_precision_suppressed(value: str) -> bool:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return bool(parts) and all(part == "SUPPRESSED" for part in parts)


def inspect_result(result_csv: Path, testcase_name: str, command_result: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    checks: dict[str, Any] = {
        "command_returncode": command_result.get("returncode"),
        "command_returncode_zero": command_result.get("returncode") == 0,
        "result_csv_exists": result_csv.is_file(),
        "row_count": 0,
        "row_found": False,
        "required_fields_present": False,
        "dyn_precision": "",
        "memory_oob_status": "",
    }

    if not checks["command_returncode_zero"]:
        failures.append("TTK kernel command returncode must be 0")
    if not result_csv.is_file():
        failures.append("result CSV does not exist")
        return _build_inspection(result_csv, testcase_name, None, checks, failures)

    try:
        rows = _read_csv_rows(result_csv)
    except Exception as exc:  # noqa: BLE001 - surfaced in module report
        failures.append(f"failed to read result CSV: {exc}")
        return _build_inspection(result_csv, testcase_name, None, checks, failures)

    checks["row_count"] = len(rows)
    if not rows:
        failures.append("result CSV has no data rows")
        return _build_inspection(result_csv, testcase_name, None, checks, failures)

    row = _select_result_row(rows, testcase_name)
    checks["row_found"] = row is not None
    if row is None:
        failures.append("expected testcase row not found")
        return _build_inspection(result_csv, testcase_name, None, checks, failures)

    missing_fields = [field for field in ("testcase_name", "dyn_precision", "memory_oob_status") if field not in row]
    checks["required_fields_present"] = not missing_fields
    if missing_fields:
        failures.append("missing required fields: " + ", ".join(missing_fields))

    dyn_precision = row.get("dyn_precision", "").strip()
    checks["dyn_precision"] = dyn_precision
    if not missing_fields and not _dyn_precision_suppressed(dyn_precision):
        failures.append("dyn_precision must be SUPPRESSED under --golden-mode Disable")

    memory_oob_status = row.get("memory_oob_status", "").strip()
    checks["memory_oob_status"] = memory_oob_status
    if not missing_fields and memory_oob_status not in ("", "PASS"):
        failures.append("memory_oob_status must be PASS or empty")

    return _build_inspection(result_csv, testcase_name, row, checks, failures)


def _build_inspection(
    result_csv: Path,
    testcase_name: str,
    row: dict[str, str] | None,
    checks: dict[str, Any],
    failures: list[str],
) -> dict[str, Any]:
    observed_fields = {}
    if row is not None:
        observed_fields = {
            field: row.get(field, "")
            for field in ("dyn_precision", "perf_status", "memory_oob_status")
            if field in row
        }
    return {
        "status": "failed" if failures else "passed",
        "execution_pass": not failures,
        "result_csv": str(result_csv),
        "expected_testcase": testcase_name,
        "selected_testcase": row.get("testcase_name", "") if row else "",
        "checks": checks,
        "observed_fields": observed_fields,
        "pass_conditions": [
            "TTK kernel command returncode is 0",
            "result CSV exists",
            "target testcase row exists",
            "required fields exist: testcase_name, dyn_precision, memory_oob_status",
            "dyn_precision is SUPPRESSED under --golden-mode Disable",
            "memory_oob_status is PASS or empty",
        ],
        "failures": failures,
    }


def acceptance_criteria() -> dict[str, Any]:
    return {
        "source": "result_csv_inline_analysis",
        "required_result_fields": ["testcase_name", "dyn_precision", "memory_oob_status"],
        "accepted_dyn_precision": ["SUPPRESSED"],
        "accepted_memory_oob_status": ["", "PASS"],
        "perf_status_required": False,
    }


def _step(report: dict[str, Any], name: str) -> dict[str, Any]:
    step = report.get("steps", {}).get(name, {})
    return step if isinstance(step, dict) else {}


def _inspection(report: dict[str, Any], name: str) -> dict[str, Any]:
    inspection = _step(report, name).get("inspection", {})
    return inspection if isinstance(inspection, dict) else {}


def _first_issue_reason(report: dict[str, Any]) -> str:
    issues = report.get("issues", [])
    if issues and isinstance(issues[0], dict):
        return str(issues[0].get("reason") or "unknown")
    return "unknown"


def _acceptance_result(report: dict[str, Any], step_name: str) -> dict[str, Any]:
    inspection = _inspection(report, step_name)
    checks = inspection.get("checks", {}) if isinstance(inspection.get("checks", {}), dict) else {}
    return {
        "accepted": inspection.get("status") == "passed",
        "testcase_name": inspection.get("selected_testcase", ""),
        "dyn_precision": checks.get("dyn_precision", ""),
        "memory_oob_status": checks.get("memory_oob_status", ""),
        "result_csv": inspection.get("result_csv", ""),
    }


def _acceptance_failures(report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for step_name in ("run_low_one", "run_high_one"):
        inspection = _inspection(report, step_name)
        for failure in inspection.get("failures", []) or []:
            failures.append(f"{step_name}: {failure}")
    if failures:
        return failures

    for issue in report.get("issues", []) or []:
        if isinstance(issue, dict):
            reason = issue.get("reason")
            step = issue.get("step", "module")
            if reason:
                failures.append(f"{step}: {reason}")
    return failures


def build_acceptance(report: dict[str, Any]) -> dict[str, Any]:
    low = _acceptance_result(report, "run_low_one")
    high = _acceptance_result(report, "run_high_one")
    accepted = low["accepted"] and high["accepted"]

    if accepted:
        status = "passed"
        reason = "result_csv_accepted"
        summary = "TTK module acceptance passed: low/high result CSV inline analysis passed."
        failures: list[str] = []
        next_action = "none"
    elif report.get("status") == "skipped":
        status = "skipped"
        reason = _first_issue_reason(report)
        if reason == "unknown":
            reason = "result_csv_analysis_not_run"
        summary = "TTK module acceptance was skipped because low/high result CSV inline analysis was not completed."
        failures = _acceptance_failures(report)
        next_action = "rerun without --skip-kernel-run in a valid TTK/CANN environment"
    else:
        status = "failed"
        failures = _acceptance_failures(report)
        reason = "result_csv_rejected"
        if failures:
            reason = failures[0].split(": ", 1)[-1]
        summary = "TTK module acceptance failed: result CSV inline analysis rejected at least one case."
        next_action = "inspect failed step and its inspection.failures in ttk_module_report.json"

    return {
        "accepted": accepted,
        "status": status,
        "reason": reason,
        "summary": summary,
        "criteria": acceptance_criteria(),
        "results": {
            "low": low,
            "high": high,
        },
        "failures": failures,
        "next_action": next_action,
    }


def build_ttk_command(args: argparse.Namespace, csv_path: Path, result_path: Path, case_name: str) -> list[str]:
    return [
        "python3",
        "-m",
        "ttk",
        "kernel",
        "-i",
        str(csv_path),
        "-o",
        str(result_path),
        "-t",
        case_name,
        "--pc",
        "1",
        "--seed",
        "42",
        "--golden-mode",
        "Disable",
    ]


def shell_ttk_command(command: list[str], ops_test_kit_path: Path, setenv_path: str) -> str:
    return (
        f"source {shlex.quote(setenv_path)} && "
        f"cd {shlex.quote(str(ops_test_kit_path))} && "
        f"{command_string(command)}"
    )


def run_ttk_kernel(
    args: argparse.Namespace,
    case: KernelRunCase,
    ops_test_kit_path: Path,
    setenv_path: str | None,
) -> dict[str, Any]:
    command = build_ttk_command(args, case.csv_path, case.result_path, case.case_name)
    if setenv_path:
        return run_shell(shell_ttk_command(command, ops_test_kit_path, setenv_path))
    return run_command(command, cwd=ops_test_kit_path)


def build_input_check(
    required_for_csv: list[Path],
    optional_for_kernel: list[Path],
) -> dict[str, Any]:
    missing_required = [str(path) for path in required_for_csv if not path.exists()]
    missing_optional = [str(path) for path in optional_for_kernel if not path.exists()]
    return {
        "status": "passed" if not missing_required else "failed",
        "required_for_csv": [str(path) for path in required_for_csv],
        "optional_for_kernel": [str(path) for path in optional_for_kernel],
        "missing_required_for_csv": missing_required,
        "missing_optional_for_kernel": missing_optional,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run white-box TTK kernel module as a fixed script workflow.")
    parser.add_argument("--op-name", required=True)
    parser.add_argument("--whitebox-dir", required=True)
    parser.add_argument("--op-path", required=True)
    parser.add_argument(
        "--op-def-cpp",
        default=None,
        help="Deprecated compatibility option; CSV generation no longer uses it.",
    )
    parser.add_argument("--ops-test-kit-path", required=True)
    parser.add_argument("--skill-base", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--setenv-path", default=None)
    parser.add_argument(
        "--skip-kernel-run",
        action="store_true",
        help="Debug/smoke only. Forbidden for normal acceptance.",
    )
    return parser.parse_args()


def _run_single_kernel(
    report: dict[str, Any],
    args: argparse.Namespace,
    case: KernelRunCase,
    env: KernelExecEnv,
    config: StepConfig,
) -> None:
    step = run_ttk_kernel(args, case, env.ops_test_kit_path, env.setenv_path)
    step["testcase_name"] = case.case_name
    step["inspection"] = inspect_result(case.result_path, case.case_name, step)
    if step["inspection"]["status"] == "failed":
        report["steps"][config.label] = fail_step(step, "kernel_execution_failed")
    else:
        report["steps"][config.label] = pass_step(step)
    if report["steps"][config.label]["status"] != "passed":
        report["issues"].append({"step": config.label, "reason": config.fail_reason})
        raise RuntimeError(config.fail_msg)


def _run_csv_generation(
    report: dict[str, Any],
    args: argparse.Namespace,
    paths: CsvGenPaths,
) -> None:
    generate_cmd = [
        "python3", str(paths.generate_script),
        "--op-name", args.op_name,
        "--low-cases", str(paths.low_cases),
        "--high-cases", str(paths.high_cases),
        "--low-csv", str(paths.low_csv),
        "--high-csv", str(paths.high_csv),
    ]
    step = run_command(generate_cmd)
    if step["returncode"] != 0:
        report["steps"]["generate_csv"] = fail_step(step, "command_failed")
        report["issues"].append({"step": "generate_csv", "reason": "command_failed"})
        raise RuntimeError("generate_csv failed")
    low_rows = count_csv_rows(paths.low_csv)
    high_rows = count_csv_rows(paths.high_csv)
    step["low_csv_rows"] = low_rows
    step["high_csv_rows"] = high_rows
    report["counts"]["low_csv_rows"] = low_rows
    report["counts"]["high_csv_rows"] = high_rows
    if low_rows != report["counts"]["low_cases"] or high_rows != report["counts"]["high_cases"]:
        report["steps"]["generate_csv"] = fail_step(step, "csv_row_count_mismatch")
        report["issues"].append(
            {
                "step": "generate_csv",
                "reason": "csv_row_count_mismatch",
                "low_cases": report["counts"]["low_cases"],
                "low_csv_rows": low_rows,
                "high_cases": report["counts"]["high_cases"],
                "high_csv_rows": high_rows,
            }
        )
        raise RuntimeError("csv row count mismatch")
    report["steps"]["generate_csv"] = pass_step(step)


def _run_csv_validation(report, validate_script, low_csv, high_csv):
    for label, csv_path in (("validate_low_csv", low_csv), ("validate_high_csv", high_csv)):
        validate_cmd = ["python3", str(validate_script), str(csv_path)]
        step = run_command(validate_cmd)
        report["steps"][label] = pass_step(step) if step["returncode"] == 0 else fail_step(step, "command_failed")
        if step["returncode"] != 0:
            report["issues"].append({"step": label, "reason": "csv_validation_failed"})
            raise RuntimeError(f"{label} failed")


def _run_precheck(
    report: dict[str, Any],
    args: argparse.Namespace,
    paths: PrecheckPaths,
) -> dict[str, Any]:
    precheck_cmd = [
        "python3", str(paths.precheck_script),
        "--whitebox-dir", str(paths.whitebox_dir),
        "--ops-test-kit-path", str(paths.ops_test_kit_path),
    ]
    if args.setenv_path:
        precheck_cmd.extend(["--setenv-path", args.setenv_path])
    step = run_command(precheck_cmd)
    if not paths.precheck_report.is_file():
        report["steps"]["precheck"] = fail_step(step, "missing_precheck_report")
        report["issues"].append({"step": "precheck", "reason": "precheck_report_missing"})
        raise RuntimeError("precheck report missing")
    precheck = load_json(paths.precheck_report)
    report["kernel_gate"] = precheck.get("kernel_gate", {"status": "unknown", "reason": "missing_kernel_gate"})
    step["precheck_status"] = precheck.get("status")
    step["kernel_gate"] = report["kernel_gate"]
    if step["returncode"] != 0 and precheck.get("status") is None:
        report["steps"]["precheck"] = fail_step(step, "command_failed")
        report["issues"].append({"step": "precheck", "reason": "command_failed"})
        raise RuntimeError("precheck command failed")
    if report["kernel_gate"].get("status") == "passed":
        report["steps"]["precheck"] = pass_step(step)
    else:
        report["steps"]["precheck"] = skip_step(
            step, report["kernel_gate"].get("reason", "kernel_gate_not_passed")
        )
    return precheck


def _build_paths(args: argparse.Namespace, skill_base: Path) -> dict[str, Path]:
    scripts_dir = skill_base / "scripts"
    whitebox_dir = Path(args.whitebox_dir).resolve()
    op_path = Path(args.op_path).resolve()
    ops_test_kit_path = Path(args.ops_test_kit_path).resolve()
    return {
        "whitebox_dir": whitebox_dir,
        "op_path": op_path,
        "ops_test_kit_path": ops_test_kit_path,
        "scripts_dir": scripts_dir,
        "low_cases": whitebox_dir / "S5_cases_low.json",
        "high_cases": whitebox_dir / "S5_cases_high.json",
        "generate_script": scripts_dir / "ttk_generate_kernel_csv.py",
        "validate_script": scripts_dir / "ttk_validate_csv.py",
        "precheck_script": scripts_dir / "ttk_precheck_env.py",
        "low_csv": whitebox_dir / f"ttk_{args.op_name}_cases_low.csv",
        "high_csv": whitebox_dir / f"ttk_{args.op_name}_cases_high.csv",
        "precheck_report": whitebox_dir / PRECHECK_FILE,
        "low_result": whitebox_dir / f"ttk_{args.op_name}_cases_low_one_result.csv",
        "high_result": whitebox_dir / f"ttk_{args.op_name}_cases_high_one_result.csv",
        "report_path": whitebox_dir / REPORT_FILE,
    }


def _run_kernels(
    report: dict[str, Any],
    args: argparse.Namespace,
    paths: dict[str, Path],
    precheck: dict[str, Any],
) -> None:
    cann_env = precheck.get("checks", {}).get("cann_env", {})
    setenv_path = cann_env.get("setenv_path") if cann_env.get("status") == "passed_after_source" else None
    exec_env = KernelExecEnv(ops_test_kit_path=paths["ops_test_kit_path"], setenv_path=setenv_path)

    low_case_name = pick_case_name(paths["low_csv"])
    low_case = KernelRunCase(
        csv_path=paths["low_csv"], result_path=paths["low_result"],
        case_name=low_case_name,
    )
    _run_single_kernel(
        report, args, low_case, exec_env,
        StepConfig("run_low_one", "low_kernel_execution_failed", "low kernel execution failed"),
    )

    high_case_name = pick_case_name(paths["high_csv"])
    high_case = KernelRunCase(
        csv_path=paths["high_csv"], result_path=paths["high_result"],
        case_name=high_case_name,
    )
    _run_single_kernel(
        report, args, high_case, exec_env,
        StepConfig("run_high_one", "high_kernel_execution_failed", "high kernel execution failed"),
    )


def _run_input_check(
    report: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    required_for_csv = [
        paths["whitebox_dir"], paths["low_cases"], paths["high_cases"],
        paths["generate_script"], paths["validate_script"], paths["precheck_script"],
    ]
    optional_for_kernel = [paths["ops_test_kit_path"]]
    input_step = build_input_check(required_for_csv, optional_for_kernel)
    report["steps"]["input_check"] = input_step
    if input_step["status"] != "passed":
        report["issues"].append(
            {
                "step": "input_check",
                "reason": "missing_required_for_csv",
                "paths": input_step["missing_required_for_csv"],
            }
        )
        raise RuntimeError("missing required inputs for CSV generation")
    report["counts"]["low_cases"] = count_cases(paths["low_cases"])
    report["counts"]["high_cases"] = count_cases(paths["high_cases"])


def _run_workflow(
    report: dict[str, Any],
    args: argparse.Namespace,
    paths: dict[str, Path],
) -> None:
    _run_input_check(report, paths)

    _run_csv_generation(
        report, args,
        CsvGenPaths(
            generate_script=paths["generate_script"],
            low_cases=paths["low_cases"],
            high_cases=paths["high_cases"],
            low_csv=paths["low_csv"],
            high_csv=paths["high_csv"],
        ),
    )
    _run_csv_validation(report, paths["validate_script"], paths["low_csv"], paths["high_csv"])
    precheck = _run_precheck(
        report, args,
        PrecheckPaths(
            precheck_script=paths["precheck_script"],
            whitebox_dir=paths["whitebox_dir"],
            ops_test_kit_path=paths["ops_test_kit_path"],
            precheck_report=paths["precheck_report"],
        ),
    )

    if args.skip_kernel_run:
        report["status"] = "skipped"
        report["issues"].append({"step": "kernel_run", "reason": "skip_kernel_run_requested"})
        return

    if report["kernel_gate"].get("status") != "passed":
        report["status"] = "skipped"
        report["issues"].append(
            {
                "step": "kernel_run",
                "status": "skipped",
                "reason": report["kernel_gate"].get("reason", "kernel_gate_not_passed"),
                "detail": "CSV files were generated and validated, but TTK kernel validation was skipped.",
                "preserved_outputs": ["low_csv", "high_csv", "precheck_report"],
            }
        )
        return

    _run_kernels(report, args, paths, precheck)
    report["status"] = "passed"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    start = now_s()
    skill_base = Path(args.skill_base).resolve()
    paths = _build_paths(args, skill_base)

    report: dict[str, Any] = {
        "status": "failed",
        "op_name": args.op_name,
        "duration_s": 0.0,
        "counts": {},
        "steps": {},
        "outputs": {
            "low_csv": str(paths["low_csv"]),
            "high_csv": str(paths["high_csv"]),
            "precheck_report": str(paths["precheck_report"]),
            "low_result_csv": str(paths["low_result"]),
            "high_result_csv": str(paths["high_result"]),
            "module_report": str(paths["report_path"]),
        },
        "kernel_gate": {"status": "unknown", "reason": "not_run"},
        "acceptance": {
            "accepted": False,
            "status": "unknown",
            "reason": "not_run",
            "summary": "",
            "criteria": acceptance_criteria(),
            "results": {},
            "failures": [],
            "next_action": "",
        },
        "issues": [],
        "warnings": [],
    }

    try:
        _run_workflow(report, args, paths)
    except Exception as exc:  # noqa: BLE001 - keep a machine-readable report for diagnosis
        report["issues"].append({"step": "module", "reason": str(exc)})
    finally:
        report["duration_s"] = round(now_s() - start, 3)
        report["acceptance"] = build_acceptance(report)
        write_report(paths["report_path"], report)
        _logger.info(f"TTK module status: {report['status']}")
        _logger.info(f"TTK module report: {paths['report_path']}")
        if report["status"] != "passed":
            raise SystemExit(1 if report["status"] == "failed" else 0)


if __name__ == "__main__":
    main()
