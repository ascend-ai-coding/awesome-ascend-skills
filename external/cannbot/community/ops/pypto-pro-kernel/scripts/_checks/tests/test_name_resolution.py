# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Lexical source failures must not require candidate imports or device work."""

import pytest

from _checks import name_resolution


def source(tmp_path, code):
    path = tmp_path / "candidate.py"
    path.write_text(code)
    return path


available = pytest.mark.skipif(
    name_resolution.backend()[0] is None,
    reason="the pinned optional pyflakes backend is not installed",
)


@available
def test_undefined_name_in_nested_kernel_is_reported_at_use(tmp_path):
    path = source(
        tmp_path,
        """import pypto_pro.language as pl
def factory():
    @pl.jit
    def kernel(x):
        pl.store(x, cur_acc, [0, 0])
    return kernel
""",
    )
    findings, scope = name_resolution.inspect(path)
    assert scope["checked"] == 1 and scope["unverified"] == 0
    assert [(f.lineno, f.rule) for f in findings] == [(5, "undefined-name")]
    assert "cur_acc" in findings[0].message


@available
def test_binding_in_another_function_does_not_cover_stale_store(tmp_path):
    path = source(
        tmp_path,
        """def first():
    cur_acc = 1
    return cur_acc
def second():
    return cur_acc + n0
""",
    )
    findings, _ = name_resolution.inspect(path)
    assert len(findings) == 2
    assert all(f.lineno == 5 for f in findings)


@available
def test_imports_closures_parameters_and_forward_globals_are_legal(tmp_path):
    path = source(
        tmp_path,
        """import math
import missing_runtime_package as runtime
raise RuntimeError('candidate must never be imported')
def factory(width):
    def kernel(rows):
        return runtime.call(rows, width, K)
    return kernel
K = 128
""",
    )
    findings, scope = name_resolution.inspect(path)
    assert not findings and scope["checked"] == 1


@pytest.mark.parametrize(
    "code,reason",
    [
        (
            "@pl.jit(tiling_key=Key)\ndef kernel():\n    return TileBlock\n",
            "tiling-key",
        ),
        (
            "@pl.jit(**options)\ndef kernel():\n    return TileBlock\n",
            "keyword expansion",
        ),
        ("from arbitrary_module import *\nvalue = injected\n", "star-import"),
        ("globals()['injected'] = 1\nvalue = injected\n", "dynamic namespace"),
    ],
)
def test_injected_names_are_unverified_not_rejected(tmp_path, code, reason):
    findings, scope = name_resolution.inspect(source(tmp_path, code))
    assert not findings
    assert scope["checked"] == 0 and scope["unverified"] == 1
    assert reason in scope["reason"]


def test_missing_backend_cannot_claim_name_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(name_resolution, "backend", lambda: (None, "unavailable"))
    findings, scope = name_resolution.inspect(source(tmp_path, "value = missing\n"))
    assert not findings and scope["unverified"] == 1
    assert "requirements-checks.txt" in scope["reason"]


@available
def test_unverified_backend_version_is_guarded(monkeypatch):
    monkeypatch.setattr(name_resolution, "version", lambda _: "9999.0.0")
    api, installed = name_resolution.backend()
    assert api is None and installed == "unsupported 9999.0.0"
