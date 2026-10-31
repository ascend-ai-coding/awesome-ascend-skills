#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Coarse file-level acceptance for Step 2 Phase 1 outputs.

Covers existence + JSON parse + top-level structure, and owner-task failure routing.
"""

import argparse
import json
import sys
from pathlib import Path


REPORT_FILE = "S2P1_postcheck_report.json"

TASK_FILES = {
    "Task A": [
        "S2P1_route_dispatch_model.md",
        "S2P1_tiling_glossary.md",
        "S2P1_route_extract.py",
        "S2P1_case_generator.py",
        "S2P1_route_cases.json",
        "S2P1_path_list.json",
        "S2P1_cases.json",
    ],
    "Task B": ["S2P1_operator_model.json"],
    "Task C": ["S2P1_low_configs.json"],
}

JSON_FILES = {
    "S2P1_route_cases.json": "Task A",
    "S2P1_path_list.json": "Task A",
    "S2P1_cases.json": "Task A",
    "S2P1_operator_model.json": "Task B",
    "S2P1_low_configs.json": "Task C",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check Step 2 Phase 1 output files (coarse acceptance + owner routing)"
    )
    parser.add_argument("--whitebox-dir", required=True)
    return parser.parse_args()


def add_failure(failures, owner_task, file_name, check, message):
    failures.append(
        {
            "owner_task": owner_task,
            "file": file_name,
            "check": check,
            "severity": "fail",
            "message": message,
        }
    )


def add_warning(warnings, owner_task, file_name, check, message):
    warnings.append(
        {
            "owner_task": owner_task,
            "file": file_name,
            "check": check,
            "severity": "warn",
            "message": message,
        }
    )


def load_json(path, owner_task, failures):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as err:  # noqa: BLE001 - report exact parse failure.
        add_failure(failures, owner_task, path.name, "json_parse", str(err))
        return None


def check_exists(whitebox_dir, failures):
    for owner_task, file_names in TASK_FILES.items():
        for file_name in file_names:
            if not (whitebox_dir / file_name).exists():
                add_failure(failures, owner_task, file_name, "exists", "required output is missing")


def check_route_cases(data, failures, warnings):
    if data is None:
        return
    if not isinstance(data, (list, dict)):
        add_failure(failures, "Task A", "S2P1_route_cases.json", "structure", "root must be a list or object")
    elif isinstance(data, list) and not data:
        add_warning(warnings, "Task A", "S2P1_route_cases.json", "structure", "route cases list is empty")
    elif isinstance(data, dict) and not data:
        add_warning(warnings, "Task A", "S2P1_route_cases.json", "structure", "route cases object is empty")


def check_path_list(data, failures, warnings):
    if data is None:
        return
    if not isinstance(data, dict):
        add_failure(failures, "Task A", "S2P1_path_list.json", "structure", "root must be an object")
        return
    for key in ("tiling_paths", "kernel_branches", "paths"):
        if key not in data:
            add_failure(failures, "Task A", "S2P1_path_list.json", "structure", f"missing key: {key}")
    paths = data.get("paths")
    if "paths" in data and not isinstance(paths, list):
        add_failure(failures, "Task A", "S2P1_path_list.json", "structure", "paths must be a list")
    elif isinstance(paths, list) and not paths:
        add_warning(warnings, "Task A", "S2P1_path_list.json", "structure", "paths list is empty")


def check_cases(data, failures):
    if data is None:
        return
    if not isinstance(data, list):
        add_failure(failures, "Task A", "S2P1_cases.json", "structure", "root must be a list")
        return
    if not data:
        add_failure(failures, "Task A", "S2P1_cases.json", "structure", "cases list is empty")
        return
    if any(not isinstance(item, dict) for item in data):
        add_failure(failures, "Task A", "S2P1_cases.json", "structure", "each case must be an object")


def check_operator_model(data, failures):
    if data is None:
        return
    if not isinstance(data, dict):
        add_failure(failures, "Task B", "S2P1_operator_model.json", "structure", "root must be an object")
        return
    for key in ("inputs", "outputs", "attributes"):
        value = data.get(key)
        if value is None:
            add_failure(failures, "Task B", "S2P1_operator_model.json", "structure", f"missing key: {key}")
        elif not isinstance(value, list):
            add_failure(failures, "Task B", "S2P1_operator_model.json", "structure", f"{key} must be a list")


def check_low_configs(data, failures, warnings):
    if data is None:
        return
    if not isinstance(data, list):
        add_failure(failures, "Task C", "S2P1_low_configs.json", "structure", "root must be a list")
        return
    if not data:
        add_warning(warnings, "Task C", "S2P1_low_configs.json", "structure", "low configs list is empty")
    if any(not isinstance(item, dict) for item in data):
        add_failure(failures, "Task C", "S2P1_low_configs.json", "structure", "each config must be an object")


def owner_from_failures(failures):
    owners = sorted({failure["owner_task"] for failure in failures})
    if not owners:
        return None
    if len(owners) == 1:
        return owners[0]
    return "multiple"


def write_report(whitebox_dir, failures, warnings):
    status = "fail" if failures else "pass"
    owner_task = owner_from_failures(failures)
    report = {
        "status": status,
        "owner_task": owner_task,
        "failures": failures,
        "warnings": warnings,
        "checked_files": TASK_FILES,
    }
    report_path = whitebox_dir / REPORT_FILE
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report_path, report


def print_result(report_path, report):
    if report["status"] == "pass":
        print("PASS S2P1")
        for name in (
            "S2P1_path_list.json",
            "S2P1_cases.json",
            "S2P1_operator_model.json",
            "S2P1_low_configs.json",
            REPORT_FILE,
        ):
            print(report_path.parent / name)
        return 0

    print("FAIL S2P1", file=sys.stderr)
    print(f"owner_task={report['owner_task']}", file=sys.stderr)
    owners = sorted({failure["owner_task"] for failure in report["failures"]})
    if len(owners) > 1:
        print("owners=" + ",".join(owners), file=sys.stderr)
    print(f"failures={len(report['failures'])}", file=sys.stderr)
    first = report["failures"][0]
    print(f"reason={first['check']}", file=sys.stderr)
    print(f"file={first['file']}", file=sys.stderr)
    print(f"detail={first['message']}", file=sys.stderr)
    print(f"report={report_path}", file=sys.stderr)
    return 1


def main():
    args = parse_args()
    whitebox_dir = Path(args.whitebox_dir).resolve()
    failures = []
    warnings = []

    if not whitebox_dir.exists():
        add_failure(failures, "unknown", str(whitebox_dir), "exists", "whitebox directory is missing")
        fallback_dir = whitebox_dir.parent if whitebox_dir.parent.exists() else Path.cwd()
        report_path, report = write_report(fallback_dir, failures, warnings)
        return print_result(report_path, report)

    check_exists(whitebox_dir, failures)

    loaded = {}
    for file_name, owner_task in JSON_FILES.items():
        path = whitebox_dir / file_name
        if path.exists():
            loaded[file_name] = load_json(path, owner_task, failures)

    check_route_cases(loaded.get("S2P1_route_cases.json"), failures, warnings)
    check_path_list(loaded.get("S2P1_path_list.json"), failures, warnings)
    check_cases(loaded.get("S2P1_cases.json"), failures)
    check_operator_model(loaded.get("S2P1_operator_model.json"), failures)
    check_low_configs(loaded.get("S2P1_low_configs.json"), failures, warnings)

    report_path, report = write_report(whitebox_dir, failures, warnings)
    return print_result(report_path, report)


if __name__ == "__main__":
    raise SystemExit(main())
