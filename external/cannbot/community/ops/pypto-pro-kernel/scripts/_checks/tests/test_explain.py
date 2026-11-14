# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Source-bound failure interpretation and changed-work inspection."""

import json
import subprocess
import sys

from _checks.explain import report
from _checks.paths import SKILL_ROOT


SOURCE = """import pypto_pro.language as pl
import torch
SPLITS = 4
def down(partial: pl.Tensor[[SPLITS, pl.DYNAMIC, 64], pl.DT_FP32]):
    pass
def run(m):
    partial = torch.zeros((m, 64), dtype=torch.float32)
"""
ERROR = "Tensor parameter 'partial' rank mismatch: expected 3, got 2"


def test_rank_error_points_to_declaration_and_allocation_without_importing(tmp_path):
    path = tmp_path / "candidate.py"
    path.write_text(SOURCE + "raise RuntimeError('must not import')\n")
    log = tmp_path / "error.txt"
    log.write_text(ERROR)
    result = report(path, error_log=log)
    symptom = result["runtime_symptoms"][0]
    assert symptom["declarations"][0]["rank"] == 3
    assert symptom["declarations"][0]["line"] == 4
    assert symptom["same_name_allocations"][0]["rank"] == 2
    assert result["error_log"]["sha256"]
    path.write_text(SOURCE.replace("[SPLITS, pl.DYNAMIC, 64]", "[pl.DYNAMIC, 64]"))
    assert (
        report(path, error_log=log)["runtime_symptoms"][0]["declarations"][0]["rank"]
        == 2
    )


def test_shared_load_and_new_compute_are_distinguished(tmp_path):
    baseline = tmp_path / "baseline.py"
    current = tmp_path / "current.py"
    code = """import pypto_pro.language as pl
K = 64
def kernel(a, b, c, x):
    for k in pl.range(K):
        pl.load(a, x, [0, k])
        pl.gemm(c, a, b)
"""
    baseline.write_text(code)
    current.write_text(code + "        pl.gemm(d, a, b)\n")
    result = report(current, baseline=baseline)
    assert result["comparison"]["unchanged_static_sites"] == 2
    assert [s["api"] for s in result["comparison"]["added_or_changed_work_sites"]] == [
        "pl.gemm"
    ]
    load = result["current"]["work_sites"][0]
    assert load["loops"][0]["iterator"] == "pl.range(K)"
    assert load["transferred_bytes"] is None
    assert load["dynamic_invocations"] is None
    current.write_text(code.replace("K = 64", "K = 128"))
    result = report(current, baseline=baseline)
    assert (
        result["comparison"]["changed_module_bindings"]["K"]["after"][0]["value"]
        == "128"
    )


def test_initialization_cost_is_visible_and_unknown_errors_stay_unknown(tmp_path):
    source = tmp_path / "source.py"
    source.write_text(SOURCE)
    log = tmp_path / "error.txt"
    log.write_text("opaque device error")
    result = report(source, error_log=log, function="run")
    assert result["current"]["work_sites"][0]["api"] == "torch.zeros"
    assert result["runtime_symptoms"][0]["status"] == "unrecognized"
    assert not result["current"]["parameters"]


def test_changed_execution_section_is_not_an_unchanged_load(tmp_path):
    baseline, candidate = tmp_path / "before.py", tmp_path / "after.py"
    code = (
        "import pypto_pro.language as pl\ndef kernel(a, x):\n    with pl.section_cube():\n"
        "        pl.load(a, x, [0, 0])\n"
    )
    baseline.write_text(code)
    candidate.write_text(code.replace("section_cube", "section_vector"))
    result = report(candidate, baseline=baseline)
    assert result["comparison"]["unchanged_static_sites"] == 0
    assert result["comparison"]["added_or_changed_work_sites"][0]["sections"] == [
        "pl.section_vector()"
    ]


def test_source_gated_runner_automatically_explains_rank_failure(tmp_path):
    source = tmp_path / "candidate.py"
    source.write_text(SOURCE)
    result = subprocess.run(
        [
            sys.executable,
            str(SKILL_ROOT / "scripts/check.py"),
            "run",
            "--source",
            str(source),
            "--",
            sys.executable,
            "-c",
            f"import sys; print({ERROR!r}, file=sys.stderr); sys.exit(7)",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 7
    assert ERROR in result.stderr
    line = next(
        s
        for s in result.stdout.splitlines()
        if s.startswith("source_contract_diagnostic ")
    )
    diagnostic = json.loads(line.split(" ", 1)[1])
    assert diagnostic["symptoms"][0]["declarations"][0]["rank"] == 3
    assert diagnostic["symptoms"][0]["same_name_allocations"][0]["rank"] == 2
