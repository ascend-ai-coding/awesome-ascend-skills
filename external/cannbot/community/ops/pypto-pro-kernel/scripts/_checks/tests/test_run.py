# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""The public runner gates execution and preserves failure/source identity."""

import hashlib
import subprocess
import sys

import pytest

from _checks.paths import SKILL_ROOT

ENTRY = SKILL_ROOT / "scripts/check.py"


def run(tmp_path, code, command):
    source = tmp_path / "candidate.py"
    source.write_text(code)
    result = subprocess.run(
        [
            sys.executable,
            str(ENTRY),
            "run",
            "--source",
            str(source),
            "--",
            sys.executable,
            "-c",
            command,
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=30,
    )
    return source, result


def test_violation_prevents_command_side_effects(tmp_path):
    _, result = run(
        tmp_path,
        "import pypto_pro.language as pl\npl.row_expand_div(a,b,c)\n",
        "from pathlib import Path; Path('started').touch()",
    )
    assert result.returncode == 1
    assert not (tmp_path / "started").exists()


def test_undefined_name_prevents_command_side_effects(tmp_path):
    from _checks import name_resolution

    if name_resolution.backend()[0] is None:
        pytest.skip("the pinned optional pyflakes backend is not installed")
    _, result = run(
        tmp_path,
        "def kernel():\n    return cur_acc\n",
        "from pathlib import Path; Path('started').touch()",
    )
    assert result.returncode == 1
    assert "undefined-name" in result.stdout and "cur_acc" in result.stdout
    assert not (tmp_path / "started").exists()


@pytest.mark.parametrize("exit_code", [0, 7])
def test_exit_code_output_and_source_hash_are_preserved(tmp_path, exit_code):
    code = "raise RuntimeError('source must not be imported by preflight')\n"
    source, result = run(
        tmp_path, code, f"print('child-output'); raise SystemExit({exit_code})"
    )
    assert result.returncode == exit_code
    assert "child-output" in result.stdout
    assert f"command_exit={exit_code}" in result.stdout
    assert (
        f"source_sha256 {hashlib.sha256(code.encode()).hexdigest()}  {source}"
        in result.stdout
    )


def test_numerical_assertion_stops_before_timing(tmp_path):
    _, result = run(
        tmp_path,
        "value = 1\n",
        "from pathlib import Path\n"
        "actual, expected = 8.856, 1.0\n"
        "assert abs(actual - expected) < 1e-3\n"
        "Path('timed').touch()\n",
    )
    assert result.returncode == 1
    assert "AssertionError" in result.stderr
    assert not (tmp_path / "timed").exists()


def test_changed_source_invalidates_successful_command(tmp_path):
    _, result = run(
        tmp_path,
        "value = 1\n",
        "from pathlib import Path; Path('candidate.py').write_text('value = 2\\n')",
    )
    assert result.returncode == 1
    assert "command_exit=0" in result.stdout
    assert "sources changed" in result.stderr


@pytest.mark.parametrize("missing_command", [False, True])
def test_empty_source_or_command_is_rejected(tmp_path, missing_command):
    command = (
        [] if missing_command else ["--", sys.executable, "-c", "raise SystemExit(19)"]
    )
    result = subprocess.run(
        [sys.executable, str(ENTRY), "run", "--source", str(tmp_path), *command],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 2


def test_directory_source_detects_new_python_files(tmp_path):
    (tmp_path / "candidate.py").write_text("value = 1\n")
    result = subprocess.run(
        [
            sys.executable,
            str(ENTRY),
            "run",
            "--source",
            str(tmp_path),
            "--",
            sys.executable,
            "-c",
            "from pathlib import Path; Path('new.py').write_text('value=2')",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert "sources changed" in result.stderr


def test_metadata_launch_parameter_is_not_rejected_for_nonliteral_spelling(tmp_path):
    code = (
        "WORKERS = 2 * 4\n"
        "def launch(kernel, x, out, workers=WORKERS):\n"
        "    return kernel[None, workers](x, out)\n"
    )
    _, result = run(tmp_path, code, "from pathlib import Path; Path('started').touch()")
    assert result.returncode == 0
    assert (tmp_path / "started").exists()


def test_launch_geometry_accepts_evaluated_values():
    from _checks.evaluation import native_launch_geometry

    assert (
        native_launch_geometry([(None, 2 * 4)], workers=8, tasks=1)["status"]
        == "matched"
    )


def test_launch_value_validation_distinguishes_mismatch_and_unavailable():
    from _checks.evaluation import native_launch_geometry

    assert native_launch_geometry(None, workers=8, tasks=1)["status"] == "unverified"
    assert native_launch_geometry([], workers=8, tasks=1)["status"] == "mismatch"
    assert (
        native_launch_geometry([(None, 9)], workers=8, tasks=1)["status"] == "mismatch"
    )
    assert (
        native_launch_geometry([(object(), 8)], workers=8, tasks=1)["status"]
        == "mismatch"
    )
    assert (
        native_launch_geometry([(None, True)], workers=8, tasks=1)["status"]
        == "mismatch"
    )
    with pytest.raises(ValueError):
        native_launch_geometry([], workers=0, tasks=1)


def test_native_integer_subclass_is_accepted_as_a_launch_value():
    from enum import IntEnum
    from _checks.evaluation import native_launch_geometry

    class Workers(IntEnum):
        EIGHT = 8

    assert (
        native_launch_geometry([(None, Workers.EIGHT)], workers=8, tasks=1)["status"]
        == "matched"
    )
