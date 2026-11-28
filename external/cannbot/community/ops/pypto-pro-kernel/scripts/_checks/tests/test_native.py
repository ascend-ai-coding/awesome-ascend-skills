# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Metadata specifications and native feedback without pretending to run hardware."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from _checks import native, native_capture, runtime_diagnostics
from _checks.paths import SKILL_ROOT


def test_entry_loading_does_not_reuse_stale_bytecode(tmp_path):
    path = tmp_path / "entry.py"
    path.write_text("def kernel():\n    return 1\n")
    first = path.stat()
    assert native._load(path, "kernel")() == 1
    path.write_text("def kernel():\n    return 2\n")
    os.utime(path, ns=(first.st_atime_ns, first.st_mtime_ns))
    assert native._load(path, "kernel")() == 2


def test_entry_loading_preserves_its_own_annotation_evaluation(tmp_path):
    path = tmp_path / "candidate.py"
    path.write_text("Alias = int\ndef kernel(value: Alias):\n    return value\n")
    assert native._load(path, "kernel").__annotations__["value"] is int
    path.write_text(
        "from __future__ import annotations\nAlias = int\ndef kernel(value: Alias):\n    return value\n"
    )
    assert native._load(path, "kernel").__annotations__["value"] == "Alias"


def test_rejected_entry_metadata_describes_strings_without_banning_them(tmp_path):
    path = tmp_path / "candidate.py"
    path.write_text('def kernel(value: "pl.Tensor[[64, 64], F32]"):\n    pass\n')
    kernel = native._load(path, "kernel")
    observed = native.entry_metadata(kernel, path)
    assert observed["callable_type"] == "builtins.function"
    assert observed["string_annotations"] == {"value": "pl.Tensor[[64, 64], F32]"}
    assert observed["declaration"] == {
        "file": str(path),
        "line": 1,
        "sha256": native.feedback.file_identity(path)["sha256"],
    }
    wrapped = SimpleNamespace(_func=kernel, to_kernel_def=lambda: None)
    assert native.entry_metadata(wrapped, path)["tile_interface"] is True


def test_entry_metadata_keeps_selected_wrapper_distinct_from_wrapped_function(tmp_path):
    path = tmp_path / "candidate.py"
    path.write_text(
        'def inner(value: "TensorAlias"):\n    pass\n\n'
        "def wrapper(value):\n    return inner(value)\nwrapper.__wrapped__ = inner\n"
    )
    observed = native.entry_metadata(native._load(path, "wrapper"), path)
    assert observed["declaration"]["line"] == 4
    assert observed["wrapped_declaration"]["line"] == 1
    assert (
        observed["declaration"]["sha256"] == observed["wrapped_declaration"]["sha256"]
    )
    assert observed["string_annotations"] == {} and not observed["tile_interface"]


def test_optional_wrapping_or_source_failures_preserve_entry_metadata(tmp_path):
    path = tmp_path / "entry.py"
    path.write_text("def wrapper():\n    pass\nwrapper.__wrapped__ = wrapper\n")
    kernel = native._load(path, "wrapper")
    assert native.entry_metadata(kernel, path)["declaration"]["line"] == 1
    path.unlink()
    observed = native.entry_metadata(kernel, path)
    assert (
        observed["callable_type"] == "builtins.function"
        and not observed["tile_interface"]
    )


def test_capture_entry_rejection_does_not_claim_a_parser_attempt(tmp_path, monkeypatch):
    path = tmp_path / "entry.py"
    path.write_text("def kernel():\n    pass\n")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"entry": "entry.py:kernel", "args": []}))
    producer = dict.fromkeys(
        ("package", "version", "python", "attributes_api_version", "observer"),
        "test-double",
    )
    producer.update(native_libraries={}, python_sources={})

    def reject(*args, **kwargs):
        raise ValueError("Expected a native @pl.jit Pro kernel")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setattr(native_capture, "capture_ir", reject)
    monkeypatch.setattr(native_capture, "_runtime", lambda: (None, None, producer))
    monkeypatch.setattr(
        native.feedback,
        "identity",
        lambda *a, **kw: {
            "selected_source": {"files": {}, "complete": True},
            "installed_runtime": None,
        },
    )
    result = native.report(spec)
    assert result["status"] == "failed" and result["stage"] == "entry_classification"
    assert result["validation"]["native_parse"] == "not_established"
    assert result["diagnostics"][0]["evidence"]["stage"] == "entry_classification"
    assert result["diagnostics"][0]["evidence"]["entry"]["declaration"]["line"] == 1


def test_missing_native_runtime_is_explicit_and_blocks_the_child(tmp_path):
    source = tmp_path / "candidate.py"
    source.write_text(
        "raise RuntimeError('entry must not be imported without capabilities')\n"
    )
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"entry": "candidate.py:kernel", "args": []}))
    entry = SKILL_ROOT / "scripts/check.py"
    wrapper = (
        "import sys,runpy\nsys.modules['torch']=None\nsys.path.insert(0,sys.argv[1])\nsys.argv=sys.argv[2:]\n"
        "runpy.run_path(sys.argv[0],run_name='__main__')"
    )
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            wrapper,
            str(entry.parent),
            str(entry),
            "run",
            "--source",
            str(source),
            "--native-spec",
            str(spec),
            "--format",
            "json",
            "--",
            sys.executable,
            "-c",
            "raise RuntimeError('child must not start')",
        ],
        text=True,
        capture_output=True,
        timeout=30,
    )
    report = json.loads(process.stdout)
    assert process.returncode == 2
    assert report["status"] == "unverified" and not report["command"]["started"]
    assert report["diagnostics"][0]["rule"] == "native-capability-unavailable"


def test_metadata_spec_resolves_entry_relative_to_spec(tmp_path):
    entry = tmp_path / "entry.py"
    entry.write_text("raise RuntimeError('spec parsing must not import')")
    path = tmp_path / "input.json"
    path.write_text(
        json.dumps(
            {
                "entry": "entry.py:kernel",
                "args": [{"shape": [32, 128], "dtype": "float16"}],
                "block_dim": 1,
                "tiling_key": {"ROWS": 32},
            }
        )
    )
    spec, source, name = native.read_spec(path)
    assert source == entry and name == "kernel"
    assert spec["args"][0]["shape"] == [32, 128]


@pytest.mark.parametrize(
    "changes",
    [
        {"entry": "entry.py:kernel()"},
        {"args": [{"shape": [-1], "dtype": "float16"}]},
        {"args": [{"shape": [True], "dtype": "float16"}]},
        {"block_dim": 0},
        {"unexpected": True},
        {"args": ["arbitrary"]},
    ],
)
def test_invalid_specs_fail_before_import(tmp_path, changes):
    (tmp_path / "entry.py").write_text("raise RuntimeError('must not import')")
    path = tmp_path / "input.json"
    path.write_text(json.dumps({"entry": "entry.py:kernel", "args": [], **changes}))
    with pytest.raises(ValueError):
        native.read_spec(path)


def test_native_exception_fields_survive_context_enrichment():
    class Error(ValueError):
        message = "insert: offset must contain exactly 2 elements"
        module = "PRO_IR"
        error_code = 0xF00004
        span = {"filename": "candidate.py", "line": 7, "column": 5}
        hint = "Use two coordinates"

    sites = [
        {
            "file": "candidate.py",
            "line": 7,
            "source": {"space": "Vec"},
            "destination": {"space": "Mat"},
        }
    ]
    found, context = runtime_diagnostics.from_exception(
        Error(), stage="native_parse", transfers=sites
    )
    assert found[0]["location"]["line"] == 7
    assert found[0]["evidence"]["error_code"] == f"{Error.error_code:06X}"
    assert found[0]["evidence"]["hint"] == Error.hint
    assert found[0]["evidence"]["stage"] == "native_parse"
    assert context[0]["id"] == "ub-l1"
