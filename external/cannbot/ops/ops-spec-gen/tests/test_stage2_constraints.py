# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""stage 2 字段单测：supported_chips / op.error_codes。

走子进程 validate_spec.py，验证 stage 2 校验逻辑：
  G1: dtype_chip_mismatch（fp8/fp4/hf8/int4 在不支持的芯片上）
  G2: unknown_chip
  J:  error_code_not_declared（machine_check.error_type 不在 op.error_codes 内）
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from validate_spec import (
    Finding,
    _check_interface_contracts,
    _check_list_length_cycles,
    _validate_list_length_expression,
    _validate_semantic_guard,
)


SKILL_ROOT = Path(__file__).resolve().parent.parent
VALIDATOR = SKILL_ROOT / "scripts" / "validate_spec.py"


def _run(spec_yaml: str, tmp_path: Path) -> tuple[int, dict]:
    p = tmp_path / "spec.yaml"
    p.write_text(spec_yaml, encoding="utf-8")
    r = subprocess.run(
        [sys.executable, str(VALIDATOR), str(p), "--json", "--stage", "2"],
        capture_output=True, text=True, timeout=30,
    )
    return r.returncode, json.loads(r.stdout) if r.stdout else {}


# 最小可校验的 add example，便于通过 string.replace 制造负向 fixture
_BASE_ADD = (SKILL_ROOT / "examples" / "add" / "spec.yaml").read_text(encoding="utf-8")


class TestSupportedChips:
    def test_baseline_passes(self, tmp_path):
        # add example 已声明 supported_chips=[Ascend910B]，本身 PASS
        rc, out = _run(_BASE_ADD, tmp_path)
        assert rc == 0
        s2 = next(s for s in out["stages"] if s["stage_id"] == 2)
        assert s2["status"] == "PASS"

    def test_unknown_chip(self, tmp_path):
        bad = _BASE_ADD.replace(
            "supported_chips: [Ascend910B]",
            "supported_chips: [atlas_unicorn, Ascend910B]",
        )
        rc, out = _run(bad, tmp_path)
        # stage 1 schema enum 也会报，但 stage 2 也独立报 unknown_chip
        s2 = next(s for s in out["stages"] if s["stage_id"] == 2)
        assert s2["status"] == "FAIL"
        # schema 拦下 unicorn 时 stage 2 不一定再跑（依赖实现）；只断言整体不通过
        assert rc != 0

    def test_dtype_chip_mismatch(self, tmp_path):
        # 把 supported_chips 收窄到不支持 bfloat16 的 Ascend310，应该报 dtype_chip_mismatch
        bad = _BASE_ADD.replace(
            "supported_chips: [Ascend910B]",
            "supported_chips: [Ascend310]",
        )
        rc, out = _run(bad, tmp_path)
        s2 = next(s for s in out["stages"] if s["stage_id"] == 2)
        assert s2["status"] == "FAIL"
        assert any("dtype_chip_mismatch" in f["rule_id"] for f in s2["findings"])


class TestErrorCodes:
    def test_error_code_not_declared(self, tmp_path):
        # add example 的 op.error_codes 声明了 shape_mismatch；删掉它后，引用该类别的
        # boundary case 会触发 paradigm_constraint.error_code_not_declared
        bad = _BASE_ADD.replace(
            "    - shape_mismatch\n",
            "",
            1,
        )
        rc, out = _run(bad, tmp_path)
        s2 = next(s for s in out["stages"] if s["stage_id"] == 2)
        assert s2["status"] == "FAIL"
        assert any("error_code_not_declared" in f["rule_id"] for f in s2["findings"])


def test_guard_rejects_bare_input_reference():
    errors = _validate_semantic_guard("input.x", set(), {"x"})
    assert errors
    assert any("裸 input.x" in error for error in errors)


def test_list_length_expression_subtraction_requires_rhs_upper_bound():
    attrs = {
        "a": {"name": "a", "type": "int64", "machine_constraint": {"lower_inclusive": 0}},
        "b": {"name": "b", "type": "int64", "machine_constraint": {"lower_inclusive": 0}},
    }
    assert _validate_list_length_expression("attr.a - attr.b", attrs, set())

    attrs["a"]["machine_constraint"]["lower_inclusive"] = 3
    attrs["b"]["machine_constraint"]["upper_inclusive"] = 3
    assert not _validate_list_length_expression("attr.a - attr.b", attrs, set())


def test_list_length_expression_multiplication_requires_non_negative_operands():
    attrs = {
        "a": {"name": "a", "type": "int64", "machine_constraint": {"lower_inclusive": -1}},
        "b": {"name": "b", "type": "int64", "machine_constraint": {"lower_inclusive": -1}},
    }
    assert _validate_list_length_expression("attr.a * attr.b", attrs, set())

    attrs["a"]["machine_constraint"]["lower_inclusive"] = 0
    attrs["b"]["machine_constraint"]["lower_inclusive"] = 0
    assert not _validate_list_length_expression("attr.a * attr.b", attrs, set())


def test_list_length_cycle_is_reported_once():
    inputs = [
        {"name": "a", "role": "tensor_list", "list_length": {"kind": "same_as", "ref": "input.b"}},
        {"name": "b", "role": "tensor_list", "list_length": {"kind": "same_as", "ref": "input.c"}},
        {"name": "c", "role": "tensor_list", "list_length": {"kind": "same_as", "ref": "input.a"}},
    ]
    findings: list[Finding] = []
    _check_list_length_cycles(inputs, [], findings)
    assert len(findings) == 1
    assert findings[0].rule_id == "interface_contract.list_length_cycle"


def test_semantic_case_and_layout_contract_valid_and_invalid_references():
    base = {
        "inputs": [
            {"name": "x", "role": "tensor", "optional": True},
            {"name": "mask", "role": "tensor", "optional": True},
        ],
        "outputs": [{"name": "y", "role": "tensor", "optional": True}],
        "attributes": [{"name": "mode", "type": "string"}],
    }
    valid = {
        **base,
        "semantic_cases": [{
            "id": "with_mask",
            "when": "input.mask.is_present and attr.mode == 'fast'",
            "inputs": {"required": ["x"], "forbidden": ["mask"]},
            "outputs": {"present": ["y"], "absent": []},
        }],
        "layout_contract": {
            "variants": [{
                "id": "default",
                "tensors": [
                    {"ref": "input.x", "logical_axes": ["B", "S"]},
                    {"ref": "output.y", "logical_axes": ["B", "S"]},
                ],
            }],
        },
    }
    findings: list[Finding] = []
    _check_interface_contracts(valid, findings)
    assert not findings

    invalid = {
        **base,
        "semantic_cases": [{
            "id": "bad",
            "when": "input.unknown.is_present",
            "inputs": {"required": ["unknown"], "forbidden": []},
            "outputs": {"present": [], "absent": []},
        }],
        "layout_contract": {
            "variants": [{
                "id": "bad",
                "tensors": [{"ref": "input.missing", "logical_axes": ["B"]}],
            }],
        },
    }
    findings = []
    _check_interface_contracts(invalid, findings)
    rule_ids = {finding.rule_id for finding in findings}
    assert "interface_contract.semantic_case_invalid_guard" in rule_ids
    assert "interface_contract.semantic_case_unknown_member" in rule_ids
    assert "interface_contract.layout_contract_unknown_ref" in rule_ids


class TestOutputCountDeterminedBy:
    """VariableOutput + output_count_determined_by=attribute：豁免 nonzero 类 data_dependent_shape。"""

    _SPLIT = (SKILL_ROOT / "examples" / "split" / "spec.yaml").read_text(encoding="utf-8")

    def test_with_flag_no_variable_output_flag_error(self, tmp_path):
        # split example 声明了 output_count_determined_by: attribute → 豁免 data_dependent_shape
        rc, out = _run(self._SPLIT, tmp_path)
        s2 = next(s for s in out["stages"] if s["stage_id"] == 2)
        assert s2["status"] == "PASS"
        assert not any("variable_output_flag" in f["rule_id"] for f in s2["findings"])

    def test_without_flag_reports_variable_output_flag(self, tmp_path):
        # 去掉 flag → 回到默认 data-determined，stage 2 应报 variable_output_flag
        import re
        bad = re.sub(r"(?m)^  output_count_determined_by:.*\n", "", self._SPLIT)
        assert not re.search(r"(?m)^  output_count_determined_by:", bad)  # op 级声明已移除（注释提及不算）
        rc, out = _run(bad, tmp_path)
        s2 = next(s for s in out["stages"] if s["stage_id"] == 2)
        assert s2["status"] == "FAIL"
        assert any("variable_output_flag" in f["rule_id"] for f in s2["findings"])
