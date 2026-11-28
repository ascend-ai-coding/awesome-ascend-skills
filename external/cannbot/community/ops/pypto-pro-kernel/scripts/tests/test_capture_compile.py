# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "capture_compile.py"
spec = importlib.util.spec_from_file_location("capture_compile", SCRIPT)
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def fake_compiler(tmp_path):
    compiler = tmp_path / "real compiler"
    compiler.write_text(
        f"#!{sys.executable}\n"
        "import pathlib,sys\n"
        "if '--help' in sys.argv: print('--cce-res-usage');sys.exit(0)\n"
        "pathlib.Path(sys.argv[sys.argv.index('-o')+1]).write_bytes(b'compiled image')\n"
        "print('[BISHENG] SIMT VF Function properties for body.vector_simt_entry: "
        "Stack size: 0 bytes, Used register number: 18',file=sys.stderr)\n"
        "sys.exit(17 if '--fail' in sys.argv else 0)\n"
    )
    compiler.chmod(0o755)
    return compiler


@pytest.mark.parametrize("fail", [False, True])
def test_capture_matches_actual_output_and_preserves_arguments_and_exit(tmp_path, fail):
    compiler = fake_compiler(tmp_path)
    source = tmp_path / "kernel source.cpp"
    source.write_text("native source")
    header = tmp_path / "kernel.h"
    header.write_text("native header")
    output = tmp_path / "evidence"
    library = tmp_path / "image.so"
    runner = tmp_path / "runner.py"
    arguments = ["bisheng", str(source), "-O3", "-o", str(library)] + (
        ["--fail"] if fail else []
    )
    runner.write_text(
        f"import subprocess\nraise SystemExit(subprocess.run({arguments!r}).returncode)\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--compiler",
            str(compiler),
            "--output",
            str(output),
            "--",
            sys.executable,
            str(runner),
        ],
        capture_output=True,
    )
    assert result.returncode == (17 if fail else 0)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["execution_binding"] == "not_established"
    assert manifest["status"] == "captured" and len(manifest["captures"]) == 1
    path = output / manifest["captures"][0]["path"]
    assert capture._sha(path) == manifest["captures"][0]["sha256"]
    record = json.loads(path.read_text())
    assert record["original_arguments"] == arguments[1:]
    assert record["effective_arguments"] == [*arguments[1:], "--cce-res-usage"]
    assert record["compiler_unchanged"] and record["inputs_unchanged"]
    assert record["inputs"][str(header)] == capture._sha(header)
    assert record["output_sha256"] == (
        None if fail else hashlib.sha256(b"compiled image").hexdigest()
    )
    assert record["resources"] == [
        {"symbol": "body.vector_simt_entry", "stack_bytes": 0, "used_registers": 18}
    ]
    assert b"Used register number: 18" in result.stderr


def test_runner_success_without_compile_does_not_fabricate_resource_evidence(tmp_path):
    output = tmp_path / "evidence"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--compiler",
            str(fake_compiler(tmp_path)),
            "--output",
            str(output),
            "--",
            sys.executable,
            "-c",
            "pass",
        ],
        capture_output=True,
    )
    assert result.returncode == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "unavailable" and manifest["captures"] == []


def test_changed_report_format_stays_unknown_instead_of_zero():
    assert (
        capture._resources("Used register number: unknown; Stack size: unavailable")
        == []
    )
    assert (
        capture._resources(
            "[BISHENG] scalar Function properties for foo: Stack size: 0 bytes, Used register number: 3"
        )
        == []
    )


def test_existing_output_is_not_overwritten(tmp_path):
    output = tmp_path / "evidence"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("untouched")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--compiler",
            str(fake_compiler(tmp_path)),
            "--output",
            str(output),
            "--",
            sys.executable,
            "-c",
            "pass",
        ],
        capture_output=True,
    )
    assert result.returncode != 0 and marker.read_text() == "untouched"


def test_preexisting_compiler_output_keeps_artifact_binding_unverified(tmp_path):
    library = tmp_path / "image.so"
    library.write_bytes(b"old image")
    output = tmp_path / "evidence"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--compiler",
            str(fake_compiler(tmp_path)),
            "--output",
            str(output),
            "--",
            "bisheng",
            "-o",
            str(library),
        ],
        capture_output=True,
    )
    assert result.returncode == 0
    record = json.loads(next((output / "records").glob("*.json")).read_text())
    assert record["output_was_absent"] is False
    assert record["output_sha256"] is None
