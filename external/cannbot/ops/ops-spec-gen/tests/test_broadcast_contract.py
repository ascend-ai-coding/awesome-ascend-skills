# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Broadcast declarations must preserve independent shape validation."""

from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys

import pytest
import yaml

import validate_spec
from evaluators.stages import stage_3
from generate_spec import GenInput, TensorSpec, render


ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    "legacy",
    [
        None,
        {"kind": "none"},
        {"kind": "numpy"},
        {"kind": "explicit", "rules": []},
        {"kind": "heterogeneous"},
        {"kind": "unknown"},
        "old",
    ],
)
def test_broadcast_schema_and_independent_stages(legacy):
    spec = yaml.safe_load(
        (ROOT / "examples/softmax/spec.yaml").read_text(encoding="utf-8")
    )
    spec.pop("broadcast", None)
    registries = validate_spec.load_registries()
    expected = [fn(spec, registries) for fn in (validate_spec.stage_2, validate_spec.stage_6)]
    if legacy is not None:
        spec["broadcast"] = legacy
    valid_kind = isinstance(legacy, dict) and legacy.get("kind") in {"none", "numpy", "explicit"}
    assert validate_spec.stage_1(spec).status == ("PASS" if valid_kind else "FAIL")
    for fn, before in zip((validate_spec.stage_2, validate_spec.stage_6), expected):
        assert fn(spec, registries) == before


def test_broadcast_boundaries_cannot_be_disabled_by_legacy_kind():
    spec = yaml.safe_load((ROOT / "examples/add/spec.yaml").read_text(encoding="utf-8"))
    spec["boundary_conditions"] = []
    registries = validate_spec.load_registries()
    for legacy in (None, {"kind": "none"}, {"kind": "numpy"}):
        candidate = deepcopy(spec)
        candidate.pop("broadcast", None)
        if legacy is not None:
            candidate["broadcast"] = legacy
        result = validate_spec.stage_6(candidate, registries)
        assert result.status == "PASS"
        assert any(f.rule_id == "boundary_min_set.candidate_not_documented" for f in result.findings)


def test_wrong_output_shape_still_fails_without_broadcast_field():
    spec = yaml.safe_load((ROOT / "examples/add/spec.yaml").read_text(encoding="utf-8"))
    spec.pop("broadcast", None)
    spec["outputs"][0]["shape_rule"] = "z.shape = x.shape"
    status, findings = stage_3(spec)
    assert status == "FAIL"
    assert any("numpy_expr_shape_mismatch" in f["rule_id"] for f in findings)


@pytest.mark.parametrize("kind", ["none", "numpy", "explicit"])
def test_generator_emits_broadcast(kind):
    spec = yaml.safe_load(
        render(
            GenInput(
                op_name="reduce_sum",
                broadcast_kind=kind,
                description="Reduce input",
                category="Reduction",
                paradigms=["Reduction"],
                inputs=[TensorSpec(name="x", dtype_set=["float32"])],
                outputs=["y"],
                supported_chips=["Ascend910B"],
            )
        )
    )
    assert spec["broadcast"] == {"kind": kind}
    assert validate_spec.stage_1(spec).status == "PASS"


def test_pipeline_has_contiguous_stage_ids():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_spec.py"),
            str(ROOT / "examples/add/spec.yaml"),
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    stages = json.loads(result.stdout)["stages"]
    assert [s["stage_id"] for s in stages] == list(range(1, 13))


@pytest.mark.parametrize(
    "script,args",
    [
        ("validate_spec.py", ["unused.yaml", "--stage", "13"]),
    ],
)
def test_removed_cli_options_are_rejected(script, args):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert (
        "invalid choice" in result.stderr or "unrecognized arguments" in result.stderr
    )


@pytest.mark.parametrize("stage_id", range(6, 13))
def test_each_renumbered_stage_can_be_selected(stage_id):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_spec.py"),
            str(ROOT / "examples/add/spec.yaml"),
            "--json",
            "--stage",
            str(stage_id),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    stages = json.loads(result.stdout)["stages"]
    assert [stage["stage_id"] for stage in stages] == [stage_id]
