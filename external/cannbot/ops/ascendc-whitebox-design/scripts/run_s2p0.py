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
"""Run Phase 0 source scouts and verification as one script gate."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


OUTPUT_FILES = [
    "S2P0_scout_t.md",
    "S2P0_scout_t.json",
    "S2P0_scout_k.md",
    "S2P0_scout_k.json",
    "S2P0_file_manifest.json",
    "S2P0_source_scope.md",
]

JSON_FILES = [
    "S2P0_scout_t.json",
    "S2P0_scout_k.json",
    "S2P0_file_manifest.json",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run S2 Phase 0 scout gate")
    parser.add_argument("--op-name", required=True)
    parser.add_argument("--op-path", required=True)
    parser.add_argument("--npu-arch", required=True)
    parser.add_argument("--soc-version", required=True)
    return parser.parse_args()


def script_path(name):
    return Path(__file__).resolve().parent / name


def make_cmd(script, args):
    return [
        sys.executable,
        str(script_path(script)),
        "--op-name",
        args.op_name,
        "--op-path",
        args.op_path,
        "--npu-arch",
        args.npu_arch,
        "--soc-version",
        args.soc_version,
    ]


def tail(text, limit=2000):
    if not text:
        return ""
    return text[-limit:]


def fail(stage, message, result=None):
    print("FAIL S2P0", file=sys.stderr)
    print(f"stage: {stage}", file=sys.stderr)
    print(message, file=sys.stderr)
    if result is not None:
        print(f"returncode: {result.returncode}", file=sys.stderr)
        if result.stdout:
            print("stdout_tail:", file=sys.stderr)
            print(tail(result.stdout), file=sys.stderr)
        if result.stderr:
            print("stderr_tail:", file=sys.stderr)
            print(tail(result.stderr), file=sys.stderr)
    return 1


def run_scouts(args):
    procs = []
    for stage, script in (("scout_t", "scout_t.py"), ("scout_k", "scout_k.py")):
        procs.append(
            (
                stage,
                subprocess.Popen(
                    make_cmd(script, args),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                ),
            )
        )

    results = []
    for stage, proc in procs:
        stdout, stderr = proc.communicate()
        result = subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)
        results.append((stage, result))

    for stage, result in results:
        if result.returncode != 0:
            return fail(stage, "scout script failed", result)
    return 0


def run_verify(args):
    result = subprocess.run(make_cmd("s2p0_verify.py", args), capture_output=True, text=True)
    if result.returncode != 0:
        return fail("s2p0_verify", "verify script failed", result)
    return 0


def check_outputs(op_path):
    whitebox_dir = Path(op_path) / "tests" / "whitebox"
    missing = [name for name in OUTPUT_FILES if not (whitebox_dir / name).exists()]
    if missing:
        return fail("output_check", "missing files: " + ", ".join(missing))

    for name in JSON_FILES:
        path = whitebox_dir / name
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as err:  # noqa: BLE001 - surface parse failure directly.
            return fail("json_check", f"failed to parse {path}: {err}")

        if name == "S2P0_file_manifest.json":
            status = data.get("verification", {}).get("status")
            if status != "pass":
                missing_files = data.get("verification", {}).get("missing_files", [])
                return fail(
                    "verify_status",
                    f"verification.status={status!r}, missing_files={missing_files}",
                )

    print("PASS S2P0")
    for name in OUTPUT_FILES:
        print(whitebox_dir / name)
    return 0


def main():
    args = parse_args()
    rc = run_scouts(args)
    if rc != 0:
        return rc
    rc = run_verify(args)
    if rc != 0:
        return rc
    return check_outputs(args.op_path)


if __name__ == "__main__":
    raise SystemExit(main())
