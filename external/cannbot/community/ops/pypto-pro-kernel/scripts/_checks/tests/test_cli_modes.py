# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CLI operations preserve structured reports and source-gated execution."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from _checks import cli, feedback
from _checks.paths import SKILL_ROOT

ENTRY = SKILL_ROOT / "scripts/check.py"


def call_cli(tmp_path, *args):
    return subprocess.run(
        [sys.executable, str(ENTRY), *map(str, args)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize(
    "mode",
    [
        "query",
        "symbol",
        "discovery",
        "document",
        "docstring",
        "example",
        "admission",
        "rules",
        "native",
        "run",
        "suggest",
        "explain",
        "experiment",
    ],
)
def test_cli_operations(tmp_path, mode):
    source = tmp_path / "candidate.py"
    source.write_text("def run(): return 1\n")
    spec = tmp_path / "native.json"
    spec.write_text('{"entry":"candidate.py:run","args":[]}')
    identity = tmp_path / "toolchain.json"
    identity.write_text('{"scope":"no-device CLI test"}')
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema": "pypto-experiment-plan/1",
                "toolchain": feedback.file_identity(identity),
                "max_trials": 1,
                "baseline": {
                    "root": str(tmp_path),
                    "files": [source.name],
                    "entry": "candidate.py:run",
                    "configuration": {},
                },
                "scope": {
                    "cases": {"one": {"rows": 1}},
                    "timing_cases": ["one"],
                    "metric": "native_task_sum_us",
                    "numerical_contract": {
                        "atol": 0,
                        "rtol": 0,
                        "oracle_sha256": "f" * 64,
                    },
                },
            }
        )
    )
    commands = {
        "query": ["query", "vf.full", "--format", "json"],
        "symbol": ["query", "pl.DT_FP16", "--kind", "constant", "--format", "json"],
        "discovery": [
            "query",
            "--filters",
            '{"source_memory":"UB","destination_memory":"L1"}',
            "--format",
            "json",
        ],
        "document": [
            "query",
            "document:pro-api/Utils-API/debugging/pto_assert.md#功能说明",
            "--format",
            "json",
        ],
        "docstring": ["query", "docstring:pl.pto_assert", "--format", "json"],
        "example": [
            "query",
            "example:algorithms/batched_transposed_gemm",
            "--format",
            "json",
        ],
        "admission": ["query", "--validate", "--format", "json"],
        "rules": ["rules", source, "--format", "json"],
        "native": ["native", "--spec", spec, "--format", "json"],
        "run": [
            "run",
            "--source",
            source,
            "--format",
            "json",
            "--",
            sys.executable,
            "-c",
            "print('CLI_ONLY_OK')",
        ],
        "suggest": ["suggest", "--source", source],
        "explain": ["explain", "--source", source],
        "experiment": [
            "experiment",
            "--state",
            tmp_path / "current.json",
            "--init",
            plan,
        ],
    }
    result = call_cli(tmp_path, *commands[mode])
    assert result.returncode == (2 if mode == "native" else 0), result.stderr
    report = json.loads(result.stdout)
    assert report and isinstance(report, dict)
    if mode == "native":
        assert report["status"] == "unverified"
        assert report["validation"]["numerical"] == "not_established"
    if mode == "run":
        assert report["command"]["started"] and report["command"]["exit_code"] == 0
        assert "CLI_ONLY_OK" in result.stderr
    if mode == "experiment":
        shown = call_cli(
            tmp_path, "experiment", "--state", tmp_path / "current.json"
        )
        assert shown.returncode == 0
        assert json.loads(shown.stdout) == report


def test_cli_run_retains_source_gate(tmp_path):
    source = tmp_path / "candidate.py"
    source.write_text("missing_name()\n")
    marker = tmp_path / "started"
    result = call_cli(
        tmp_path,
        "run",
        "--source",
        source,
        "--format",
        "json",
        "--",
        sys.executable,
        "-c",
        f"open({str(marker)!r}, 'w').write('ran')",
    )
    assert result.returncode == 1
    assert not json.loads(result.stdout)["command"]["started"]
    assert not marker.exists()


@pytest.mark.parametrize("mode", ["success", "child_failure", "source_change"])
def test_cli_run_feedback_file_preserves_execution_evidence(tmp_path, mode):
    source = tmp_path / "candidate.py"
    source.write_text("def run(): return 1\n")
    report_path = tmp_path / "reports/current feedback.json"
    child = "import sys; print('ORIGINAL_CHILD_DIAGNOSTIC'); "
    child += (
        f"open({str(source)!r}, 'w').write('def run(): return 2\\n')"
        if mode == "source_change"
        else f"sys.exit({7 if mode == 'child_failure' else 0})"
    )
    result = call_cli(
        tmp_path, "run", "--source", source, "--feedback-file", report_path,
        "--", sys.executable, "-c", child,
    )
    expected = {"success": 0, "child_failure": 7, "source_change": 1}[mode]
    assert result.returncode == expected, result.stderr
    assert "ORIGINAL_CHILD_DIAGNOSTIC" in result.stdout
    receipt = json.loads(next(
        line.removeprefix("kernel_feedback_file ")
        for line in result.stdout.splitlines() if line.startswith("kernel_feedback_file ")
    ))
    payload = report_path.read_bytes()
    assert report_path.stat().st_mode & 0o777 == 0o600
    assert not list(report_path.parent.glob(".kernel-feedback-*"))
    report = json.loads(payload)
    assert receipt["path"] == str(report_path)
    assert receipt["sha256"] == hashlib.sha256(payload).hexdigest()
    assert receipt["status"] == report["status"]
    assert report["command"]["started"]
    assert report["command"]["exit_code"] == (7 if mode == "child_failure" else 0)
    assert "kernel_feedback {" not in result.stdout
    if mode == "source_change":
        assert set(report["validation"].values()) == {"invalidated"}
        assert any(item["rule"] == "source-changed" for item in report["diagnostics"])


@pytest.mark.parametrize("operation", ["write_bytes", "replace"])
def test_cli_run_feedback_file_cleans_up_after_failure(tmp_path, monkeypatch, operation):
    source = tmp_path / "candidate.py"
    source.write_text("missing_name()\n")
    report_path = tmp_path / "current.json"

    def fail(*args, **kwargs):
        raise OSError("feedback output failed")

    target = cli.os if operation == "replace" else Path
    monkeypatch.setattr(target, operation, fail)
    with pytest.raises(OSError, match="feedback output failed"):
        cli.run_command([source], [], pypto_root=tmp_path, feedback_file=report_path)
    assert not report_path.exists()
    assert not list(tmp_path.glob(".kernel-feedback-*"))


def test_cli_run_feedback_file_keeps_static_rejection(tmp_path):
    source = tmp_path / "candidate.py"
    source.write_text("missing_name()\n")
    report_path, marker = tmp_path / "current.json", tmp_path / "started"
    result = call_cli(
        tmp_path, "run", "--source", source, "--feedback-file", report_path,
        "--", sys.executable, "-c", f"open({str(marker)!r}, 'w').write('ran')",
    )
    assert result.returncode == 1
    report = json.loads(report_path.read_text())
    assert not report["command"]["started"]
    assert report["status"] == "failed"
    assert not marker.exists()


def test_cli_run_feedback_file_cannot_overwrite_source(tmp_path):
    source = tmp_path / "candidate.py"
    original = "def run(): return 1\n"
    source.write_text(original)
    marker = tmp_path / "started"
    result = call_cli(
        tmp_path, "run", "--source", source, "--feedback-file", source,
        "--", sys.executable, "-c", f"open({str(marker)!r}, 'w').write('ran')",
    )
    assert result.returncode == 2
    assert "cannot overwrite" in result.stderr
    assert source.read_text() == original
    assert not marker.exists()


@pytest.mark.parametrize("scope,expected", [("business", 0), ("public", 1), ("private", 0)])
def test_writing_style_is_scoped_to_the_selected_example_library(tmp_path, scope, expected):
    examples = tmp_path / "examples"
    relative = {"business": "submission/kernel.py", "public": "examples/algorithms/demo/kernel.py",
                "private": "examples/algorithms/_helper/kernel.py"}[scope]
    source = tmp_path / relative
    source.parent.mkdir(parents=True)
    source.write_text("def run(): return 1\n")
    result = call_cli(
        tmp_path, "--examples-root", examples, "rules", source, "--format", "json"
    )
    assert result.returncode == expected, result.stderr
    report = json.loads(result.stdout)
    assert any(item["rule"] == "writing-style-declaration" for item in report["diagnostics"]) == bool(expected)


def test_business_kernel_keeps_native_checks_without_example_style(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text("import pypto_pro.language as pl\ndef run():\n    pl.TileType(shape=[64])\n")
    result = call_cli(tmp_path, "rules", source, "--format", "json")
    report = json.loads(result.stdout)
    assert result.returncode == 1
    rules = {item["rule"] for item in report["diagnostics"]}
    assert "tile-shape-must-be-2d" in rules
    assert "writing-style-declaration" not in rules


def test_example_symlink_does_not_apply_style_to_external_business_source(tmp_path):
    examples = tmp_path / "examples"
    business = tmp_path / "kernel.py"
    business.write_text("def run(): return 1\n")
    public = examples / "algorithms/demo/kernel.py"
    public.parent.mkdir(parents=True)
    public.write_text('\"\"\"写法：原始。Example.\"\"\"\ndef run(): return 1\n')
    escaped = examples / "algorithms/escaped/kernel.py"
    escaped.parent.mkdir(parents=True)
    escaped.symlink_to(business)
    result = call_cli(
        tmp_path, "--examples-root", examples, "rules", business, public, "--format", "json"
    )
    assert result.returncode == 0, result.stderr
    assert not any(item["rule"] == "writing-style-declaration"
                   for item in json.loads(result.stdout)["diagnostics"])
