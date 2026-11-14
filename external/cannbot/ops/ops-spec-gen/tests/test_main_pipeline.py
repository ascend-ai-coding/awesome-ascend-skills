# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Main pipeline 反向集成单测——端到端跑 scripts/validate_spec.py 子进程。

* TestStage5Tags：双轨匹配（tag 优先 + 子串兜底）
* TestNegativeIntegration：故意构造错误 spec，验证错误码完整传到 exit 1
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


SKILL_ROOT = Path(__file__).resolve().parent.parent
VALIDATOR = SKILL_ROOT / "scripts" / "validate_spec.py"


class TestStage5Tags:
    """stage 5 双轨匹配：tag 命中 / 子串兜底 / 都不命中提示适用性。"""

    def _run(self, spec_yaml: str, tmp_path: Path) -> subprocess.CompletedProcess:
        path = tmp_path / "spec.yaml"
        path.write_text(spec_yaml, encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(VALIDATOR), str(path), "--json", "--stage", "6"],
            capture_output=True, text=True, timeout=30,
        )

    def test_explicit_tag_covers_requirement_with_unrelated_description(self, tmp_path):
        # 改 softmax example 的"reduce 轴长度为 1" case 描述为完全不含子串关键词的措辞，
        # 但保留 tags: [reduce_axis_size_1] —— stage 5 应通过 tag 匹配认可。
        good = (SKILL_ROOT / "examples" / "softmax" / "spec.yaml").read_text(encoding="utf-8")
        modified = good.replace(
            '- case: "reduce 轴长度为 1"',
            '- case: "退化为常数情形"',
        )
        result = self._run(modified, tmp_path)
        out = json.loads(result.stdout)
        stage_5 = next(s for s in out["stages"] if s["stage_id"] == 6)
        # tags 仍带着 reduce_axis_size_1，要求被覆盖
        assert stage_5["status"] == "PASS", stage_5["findings"]

    def test_no_tag_no_keyword_is_advisory(self, tmp_path):
        # 同时移除 tag 和关键词，只提示候选适用性，不强制扩充契约
        good = (SKILL_ROOT / "examples" / "softmax" / "spec.yaml").read_text(encoding="utf-8")
        modified = good.replace(
            '- case: "reduce 轴长度为 1"\n    tags: [reduce_axis_size_1]\n',
            '- case: "退化为常数情形"\n',
        )
        result = self._run(modified, tmp_path)
        out = json.loads(result.stdout)
        stage_5 = next(s for s in out["stages"] if s["stage_id"] == 6)
        assert result.returncode == 0, result.stderr
        assert stage_5["status"] == "PASS"
        assert all(f["severity"] == "info" for f in stage_5["findings"])
        assert any("reduce_axis_size_1" in (f.get("message") or "")
                   for f in stage_5["findings"])


class TestNegativeIntegration:
    """构造故意错的 spec 跑 main pipeline，验证错误码完整传到 exit 1。"""

    def _run(self, spec_yaml: str, tmp_path: Path) -> subprocess.CompletedProcess:
        bad = tmp_path / "spec.yaml"
        bad.write_text(spec_yaml, encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(VALIDATOR), str(bad), "--json"],
            capture_output=True, text=True, timeout=30,
        )

    def test_dtype_closure_mismatch_exits_1(self, tmp_path):
        # 复制 add example 但故意把 dtype 输出弄错
        good = (SKILL_ROOT / "examples" / "add" / "spec.yaml").read_text(encoding="utf-8")
        bad = good.replace(
            "{inputs: {x: float16, y: float32}, outputs: {z: float32}}",
            "{inputs: {x: float16, y: float32}, outputs: {z: float16}}",  # 错：promote 应是 fp32
        )
        result = self._run(bad, tmp_path)
        assert result.returncode == 1, result.stderr
        out = json.loads(result.stdout)
        stage_4 = next(s for s in out["stages"] if s["stage_id"] == 4)
        assert stage_4["status"] == "FAIL"
        assert any("combination_mismatch" in f["rule_id"] for f in stage_4["findings"])

    def test_shape_rule_kind_missing_exits_1(self, tmp_path):
        good = (SKILL_ROOT / "examples" / "softmax" / "spec.yaml").read_text(encoding="utf-8")
        # 删 shape_rule_kind 这一行 → stage 3 报 shape_rule_kind_missing
        bad = good.replace("    shape_rule_kind: numpy_expr\n", "")
        result = self._run(bad, tmp_path)
        assert result.returncode == 1, result.stderr
        out = json.loads(result.stdout)
        stage_3 = next(s for s in out["stages"] if s["stage_id"] == 3)
        assert stage_3["status"] == "FAIL"
        assert any("shape_rule_kind_missing" in f["rule_id"] for f in stage_3["findings"])

    def test_data_dependent_requires_variable_output_paradigm(self, tmp_path):
        # LayoutTransform 输出 shape 只依赖 input shape / attribute，应写 numpy_expr。
        # 若误写 data_dependent，下游无法求解输出 shape，会生成不合法 ST shape。
        good = (SKILL_ROOT / "examples" / "complex" / "spec.yaml").read_text(encoding="utf-8")
        bad = good.replace(
            """    shape_rule_kind: numpy_expr
    shape_rule: |
      out.shape = np.broadcast_shapes(real.shape, imag.shape)
""",
            """    shape_rule_kind: data_dependent
    data_dependent_shape: true
    shape_rule_description: |
      out.shape depends on runtime input values.
    shape_bounds:
      max_elements: "prod(real.shape)"
""",
        )

        result = self._run(bad, tmp_path)

        assert result.returncode == 1, result.stderr
        out = json.loads(result.stdout)
        stage_3 = next(s for s in out["stages"] if s["stage_id"] == 3)
        assert stage_3["status"] == "FAIL"
        assert any("data_dependent_requires_variable_output" in f["rule_id"]
                   for f in stage_3["findings"])

    def test_data_dependent_allows_dynamic_shape_paradigm(self, tmp_path):
        # DynamicShape paradigm 的 value-driven 子场景（如 splitv）允许 data_dependent。
        good = (SKILL_ROOT / "examples" / "complex" / "spec.yaml").read_text(encoding="utf-8")
        modified = good.replace(
            "paradigms: [Broadcast, LayoutTransform]",
            "paradigms: [Broadcast, LayoutTransform, DynamicShape]",
        ).replace(
            """    shape_rule_kind: numpy_expr
    shape_rule: |
      out.shape = np.broadcast_shapes(real.shape, imag.shape)
""",
            """    shape_rule_kind: data_dependent
    data_dependent_shape: true
    shape_rule_description: |
      out.shape depends on runtime input values.
    shape_bounds:
      max_elements: "prod(real.shape)"
""",
        )

        result = self._run(modified, tmp_path)

        assert result.returncode == 0, result.stderr
        out = json.loads(result.stdout)
        stage_3 = next(s for s in out["stages"] if s["stage_id"] == 3)
        assert not any("data_dependent_requires_variable_output" in f["rule_id"]
                       for f in stage_3["findings"]), \
            "DynamicShape paradigm 应允许 data_dependent"

    def test_numpy_expr_shape_rule_must_match_boundary_oracle(self, tmp_path):
        # shape_rule 不能只写默认/简化形状，再把完整广播语义留给 notes / formula。
        # 这里故意把 add 的输出 shape 写成 x.shape；标量 + tensor 的边界样例应触发 stage 3。
        good = (SKILL_ROOT / "examples" / "add" / "spec.yaml").read_text(encoding="utf-8")
        bad = good.replace(
            "      z.shape = np.broadcast_shapes(x.shape, y.shape)",
            "      z.shape = x.shape",
        )

        result = self._run(bad, tmp_path)

        assert result.returncode == 1, result.stderr
        out = json.loads(result.stdout)
        stage_3 = next(s for s in out["stages"] if s["stage_id"] == 3)
        assert stage_3["status"] == "FAIL"
        assert any("numpy_expr_shape_mismatch" in f["rule_id"]
                   for f in stage_3["findings"])

    def test_shape_rule_placeholder_exits_1(self, tmp_path):
        good = (SKILL_ROOT / "examples" / "add" / "spec.yaml").read_text(encoding="utf-8")
        bad = good.replace(
            "      z.shape = np.broadcast_shapes(x.shape, y.shape)",
            "      z.shape = np.broadcast_shapes(x.shape, y.shape)  # TODO: replace placeholder",
        )

        result = self._run(bad, tmp_path)

        assert result.returncode == 1, result.stderr
        out = json.loads(result.stdout)
        stage_3 = next(s for s in out["stages"] if s["stage_id"] == 3)
        assert stage_3["status"] == "FAIL"
        assert any("shape_rule_placeholder" in f["rule_id"] for f in stage_3["findings"])

    def test_shape_rule_reference_comment_is_not_placeholder(self, tmp_path):
        good = (SKILL_ROOT / "examples" / "add" / "spec.yaml").read_text(encoding="utf-8")
        modified = good.replace(
            "      z.shape = np.broadcast_shapes(x.shape, y.shape)",
            "      z.shape = np.broadcast_shapes(x.shape, y.shape)  # 完整 shape 说明见 notes",
        )

        result = self._run(modified, tmp_path)

        out = json.loads(result.stdout)
        stage_3 = next(s for s in out["stages"] if s["stage_id"] == 3)
        assert stage_3["status"] == "PASS", stage_3["findings"]
        assert not any("shape_rule_placeholder" in f["rule_id"] for f in stage_3["findings"])

    def test_invalid_synthesize_shape_reports_warning(self, tmp_path):
        good = (SKILL_ROOT / "examples" / "add" / "spec.yaml").read_text(encoding="utf-8")
        modified = good.replace(
            '    synthesize: {x.shape: "[2, 3]", y.shape: "[2, 3]"}',
            '    synthesize: {x.shape: "[2, bad]", y.shape: "[2, 3]"}',
        )

        result = self._run(modified, tmp_path)

        out = json.loads(result.stdout)
        stage_3 = next(s for s in out["stages"] if s["stage_id"] == 3)
        assert stage_3["status"] == "PASS", stage_3["findings"]
        assert any(
            f["severity"] == "warning"
            and f["rule_id"] == "shape_closure.synthesize_parse_error"
            for f in stage_3["findings"]
        )

    def test_pure_reduction_identity_shape_rule_exits_1(self, tmp_path):
        spec = """
schema_version: 1
op:
  name: reduce_sum
  category: Reduction
  paradigms: [Reduction]

inputs:
  - name: x
    role: tensor
    dtype_set: [float32]
    rank_range: [1, 8]
    layout: ND
    shape: {symbolic: [...x, R]}

attributes:
  - name: dim
    type: int64
    default: -1
  - name: keep_dims
    type: bool
    default: false

outputs:
  - name: y
    shape_rule_kind: numpy_expr
    shape_rule: |
      y.shape = x.shape
    dtype_rule_kind: numpy_expr
    dtype_rule: |
      y.dtype = x.dtype

shape_constraints:
  symbols:
    R: {kind: dim, range: [0, 9223372036854775807]}

dtype_policy:
  supported_combinations:
    - {inputs: {x: float32}, outputs: {y: float32}}


math_semantics:
  formula_kind: numpy_expr
  formula: |
    y = np.sum(x, axis=dim, keepdims=keep_dims)

boundary_conditions:
  - case: "reduce last axis"
    synthesize: {x.shape: "[2, 3]", attr.dim: -1, attr.keep_dims: false}
    machine_check: {kind: matches_oracle}
"""
        result = self._run(spec, tmp_path)

        assert result.returncode == 1, result.stderr
        out = json.loads(result.stdout)
        stage_3 = next(s for s in out["stages"] if s["stage_id"] == 3)
        assert stage_3["status"] == "FAIL"
        assert any(
            f["rule_id"] == "shape_closure.reduction_shape_identity_suspicious"
            for f in stage_3["findings"]
        )

    def test_reduction_composite_identity_shape_rule_is_allowed(self, tmp_path):
        good = (SKILL_ROOT / "examples" / "softmax" / "spec.yaml").read_text(encoding="utf-8")

        result = self._run(good, tmp_path)

        out = json.loads(result.stdout)
        stage_3 = next(s for s in out["stages"] if s["stage_id"] == 3)
        assert stage_3["status"] == "PASS", stage_3["findings"]
        assert not any(
            f["rule_id"] == "shape_closure.reduction_shape_identity_suspicious"
            for f in stage_3["findings"]
        )

    def test_numpy_expr_shape_rule_can_use_attr_dependent_arithmetic(self, tmp_path):
        # 输出 shape 由 input rank + offset attribute 共同决定。该表达式在具体样例上
        # 可执行且完整，即使当前 SymbolicShape evaluator 不能表达 min/max 维运算，也应通过。
        spec = """
schema_version: 1
op:
  name: rank_offset_shape
  category: LayoutTransform
  paradigms: [LayoutTransform]

inputs:
  - name: x
    role: tensor
    dtype_set: [float32]
    rank_range: [1, 2]
    layout: ND
    shape: {symbolic: [N, M]}

attributes:
  - name: offset
    type: int64
    default: 0

outputs:
  - name: y
    shape_rule_kind: numpy_expr
    shape_rule: |
      y.shape = (
          (x.shape[0] + abs(offset), x.shape[0] + abs(offset))
          if len(x.shape) == 1
          else (
              (max(0, min(x.shape[0], x.shape[1] - offset)),)
              if offset >= 0
              else (max(0, min(x.shape[0] + offset, x.shape[1])),)
          )
      )
    dtype_rule_kind: numpy_expr
    dtype_rule: |
      y.dtype = x.dtype

shape_constraints:
  symbols:
    N: {kind: dim, range: [0, 9223372036854775807]}
    M: {kind: dim, range: [0, 9223372036854775807]}

dtype_policy:
  supported_combinations:
    - {inputs: {x: float32}, outputs: {y: float32}}


math_semantics:
  formula_kind: numpy_expr
  formula: |
    y = np.zeros((
        (x.shape[0] + abs(offset), x.shape[0] + abs(offset))
        if len(x.shape) == 1
        else (
            (max(0, min(x.shape[0], x.shape[1] - offset)),)
            if offset >= 0
            else (max(0, min(x.shape[0] + offset, x.shape[1])),)
        )
    ), dtype=x.dtype)

boundary_conditions:
  - case: "rank1 positive offset"
    synthesize: {x.shape: "[3]", attr.offset: 2}
    machine_check: {kind: matches_oracle}
  - case: "rank2 negative offset"
    synthesize: {x.shape: "[3, 5]", attr.offset: -1}
    machine_check: {kind: matches_oracle}
"""
        result = self._run(spec, tmp_path)

        out = json.loads(result.stdout)
        stage_3 = next(s for s in out["stages"] if s["stage_id"] == 3)
        assert stage_3["status"] == "PASS", stage_3["findings"]

    def test_unknown_anti_pattern_exits_1(self, tmp_path):
        # P0 #3：anti_pattern_id 必须在 registry 内；改成 AP-999 应该 stage 2 FAIL
        good = (SKILL_ROOT / "examples" / "matmul" / "spec.yaml").read_text(encoding="utf-8")
        bad = good.replace("anti_pattern_id: AP-007", "anti_pattern_id: AP-999")
        result = self._run(bad, tmp_path)
        assert result.returncode == 1, result.stderr
        out = json.loads(result.stdout)
        stage_2 = next(s for s in out["stages"] if s["stage_id"] == 2)
        assert stage_2["status"] == "FAIL"
        assert any("unknown_anti_pattern" in f["rule_id"] for f in stage_2["findings"])


class TestOutputCountDeterminedBySchema:
    """stage 1 schema 校验 output_count_determined_by 取值（enum: data/attribute）。
    锁定"非法值被 stage 1 拦截"，避免拼错静默失效。"""

    _SPLIT = (SKILL_ROOT / "examples" / "split" / "spec.yaml").read_text(encoding="utf-8")

    def _run(self, spec_yaml: str, tmp_path: Path) -> tuple[int, dict]:
        p = tmp_path / "spec.yaml"
        p.write_text(spec_yaml, encoding="utf-8")
        r = subprocess.run(
            [sys.executable, str(VALIDATOR), str(p), "--json", "--stage", "1"],
            capture_output=True, text=True, timeout=30,
        )
        return r.returncode, json.loads(r.stdout) if r.stdout else {}

    def test_valid_attribute_passes_stage1(self, tmp_path):
        rc, out = self._run(self._SPLIT, tmp_path)
        s1 = next(s for s in out["stages"] if s["stage_id"] == 1)
        assert s1["status"] == "PASS"

    def test_invalid_value_rejected_stage1(self, tmp_path):
        import re
        bad = re.sub(r"(?m)^  output_count_determined_by:.*",
                     "  output_count_determined_by: attr", self._SPLIT)
        rc, out = self._run(bad, tmp_path)
        s1 = next(s for s in out["stages"] if s["stage_id"] == 1)
        assert s1["status"] == "FAIL"
        assert any("output_count_determined_by" in f.get("field_path", "") for f in s1["findings"])


class TestToleranceSchema:
    @pytest.mark.parametrize("policy", [
        {"metric": "staged_relative", "rtol": .001, "max_relative_rtol": .01},
        {"metric": "staged_relative", "rtol": .001, "max_relative_rtol": .01,
         "atol": 0, "epsilon": 1e-7, "fallback": {"reference_mode": "native", "error_ratio_limit": 2}},
        {"metric": "staged_relative", "backend": "native", "rtol": .001,
         "max_relative_rtol": .01, "atol": 0, "epsilon": 1e-7,
         "fallback": {"small_value_threshold": .0001, "small_value_error": 1e-5,
                      "cancel_boundary": .01, "cancel_zero_threshold": .01,
                      "error_ratio_limit": 2, "reference_mode": "native_or_truncated"}},
        {"metric": "max_relative", "rtol": .001, "atol": .001, "legacy_metadata": "kept"},
        {"metric": "bitwise_equal"},
        {"metric": "bitwise_equal", "rtol": 0, "atol": 0},
        {"metric": "max_relative", "rtol": .001, "atol": .001},
    ])
    def test_supported_precision_policies_pass_without_irrelevant_fields(self, policy):
        import yaml
        from validate_spec import stage_1
        spec = yaml.safe_load((SKILL_ROOT / "examples/add/spec.yaml").read_text())
        spec["numerical_tolerance"]["per_dtype"]["float16"] = policy
        result = stage_1(spec)
        assert result.status == "PASS", result.findings

    @pytest.mark.parametrize("policy", [
        {"metric": "staged_relative", "rtol": .001},
        {"metric": "staged_relative", "rtol": 0, "max_relative_rtol": .01},
        {"metric": "staged_relative", "rtol": .001, "max_relative_rtol": .01, "atol": .01},
        {"metric": "staged_relative", "rtol": .001, "max_relative_rtol": .01, "fallback": {"unknown": 1}},
        {"metric": "staged_relative", "rtol": .001, "max_relative_rtol": .01, "fallback": {"reference_mode": "other"}},
        {"metric": "staged_relative", "rtol": .001, "max_relative_rtol": .01, "fallback": {"error_ratio_limit": 0}},
        {"metric": "staged_relative", "rtol": .001, "max_relative_rtol": .01, "reference_mode": "native"},
        {"metric": "staged_relative", "rtol": .001, "max_relative_rtol": .01, "error_ratio_limit": .5},
        {"metric": "staged_relative", "rtol": .001, "max_relative_rtol": .01, "unknown": True},
        {"metric": "max_relative", "rtol": .001},
    ])
    def test_incomplete_and_invalid_policies_still_fail(self, policy):
        import yaml
        from validate_spec import stage_1
        spec = yaml.safe_load((SKILL_ROOT / "examples/add/spec.yaml").read_text())
        spec["numerical_tolerance"]["per_dtype"]["float16"] = policy
        assert stage_1(spec).status == "FAIL"


class TestArithmeticAndBroadcastIntegration:
    def _run(self, spec, tmp_path):
        import yaml
        path = tmp_path / "spec.yaml"
        path.write_text(yaml.safe_dump(spec, sort_keys=False))
        result = subprocess.run([sys.executable, str(VALIDATOR), str(path), "--json"],
                                capture_output=True, text=True, timeout=30)
        return result.returncode, json.loads(result.stdout)

    def test_packed_output_with_heterogeneous_inputs_uses_full_pipeline(self, tmp_path):
        import yaml
        spec = yaml.safe_load((SKILL_ROOT / "examples/fused_quant_matmul/spec.yaml").read_text())
        spec["inputs"][0]["shape"]["symbolic"] = ["M", "K"]
        spec["inputs"][1]["shape"]["symbolic"] = ["G", "K", "N"]
        spec["shape_constraints"]["symbols"]["G"] = {"kind": "dim", "range": [1, 8]}
        spec["outputs"][0]["shape_rule"] = "y.shape = (x1.shape[0], x2.shape[-1] // 2)"
        spec["math_semantics"]["formula_kind"] = "python_block"
        code, out = self._run(spec, tmp_path)
        assert code == 0, out
        assert all(s["status"] != "FAIL" for s in out["stages"])

    def test_arithmetic_failure_still_exits_nonzero(self, tmp_path):
        import yaml
        spec = yaml.safe_load((SKILL_ROOT / "examples/add/spec.yaml").read_text())
        spec["outputs"][0]["shape_rule"] = "z.shape = (x.shape[-1] // 0,)"
        code, out = self._run(spec, tmp_path)
        assert code == 1
        assert next(s for s in out["stages"] if s["stage_id"] == 3)["status"] == "FAIL"


@pytest.mark.parametrize("section", ["boundary_conditions", "extreme_inputs"])
@pytest.mark.parametrize("formula_kind", ["numpy_expr", "python_block"])
def test_concrete_sample_outside_declared_domain_fails_cli(tmp_path, section, formula_kind):
    import yaml
    spec = yaml.safe_load((SKILL_ROOT / "examples/add/spec.yaml").read_text())
    for inp in spec["inputs"]:
        inp["shape"]["symbolic"] = ["N"]
    spec["shape_constraints"]["symbols"] = {"N": {"kind": "dim", "range": [32, 128]}}
    spec["math_semantics"]["formula_kind"] = formula_kind
    spec["boundary_conditions"] = []
    spec["extreme_inputs"] = []
    spec[section] = [{"case": "normal sample", "synthesize": {"x.shape": "[8]", "y.shape": "[8]"},
                      "machine_check": {"kind": "matches_oracle"}}]
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(spec))
    result = subprocess.run([sys.executable, str(VALIDATOR), str(path), "--json", "--stage", "3"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 1, result.stderr
    stage = json.loads(result.stdout)["stages"][0]
    assert stage["status"] == "FAIL"
    assert any(f["rule_id"] == "shape_closure.sample_dimension" and section in f["field_path"]
               for f in stage["findings"])


@pytest.mark.parametrize("paradigm,case_id", [
    ("Contraction", "k_zero_empty_inner_product"),
    ("Reduction", "empty_tensor"),
])
def test_unconfirmed_paradigm_boundary_does_not_force_contract(tmp_path, paradigm, case_id):
    # 只验证候选门禁；K=0 不在已声明支持域内，不能据范式强制补造报错。
    import yaml
    spec = {
        "op": {"paradigms": [paradigm]},
        "shape_constraints": {"symbols": {"K": {"range": [64, 8192]}}},
        "boundary_conditions": [],
        "extreme_inputs": [],
    }
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(spec))
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), str(path), "--json", "--stage", "6", "--strict"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    stage = json.loads(result.stdout)["stages"][0]
    assert stage["status"] == "PASS"
    assert any(case_id in f["message"] for f in stage["findings"])
    assert all(f["severity"] == "info" for f in stage["findings"])


def test_declared_negative_behavior_still_validates_error_type(tmp_path):
    import yaml
    spec = yaml.safe_load((SKILL_ROOT / "examples/softmax/spec.yaml").read_text())
    spec["boundary_conditions"].append({
        "case": "来源明确要求的拒绝行为",
        "synthesize": {"shapes": {"x": "[2, 3]"}},
        "machine_check": {"kind": "raises_error", "error_type": "not_a_real_error"},
    })
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(spec))
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), str(path), "--json", "--stage", "2"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 1
    findings = json.loads(result.stdout)["stages"][0]["findings"]
    assert any(f["rule_id"] == "error_type_unknown" and f["severity"] == "error" for f in findings)


@pytest.mark.parametrize("lower,kind,expect_error", [
    (0, "matches_oracle", False),       # 来源允许空内积，合法边界仍可声明。
    (64, "matches_oracle", True),       # 已声明普通样例超出支持域，仍须拒绝。
    (64, "raises_error", False),        # 来源明确要求拒绝时，域外负向条目仍合法。
])
def test_declared_boundary_domain_checks_remain_strict(lower, kind, expect_error):
    import yaml
    from evaluators.stages import check_concrete_sample_domain
    spec = yaml.safe_load((SKILL_ROOT / "examples/matmul/spec.yaml").read_text())
    spec["shape_constraints"]["symbols"]["K"]["range"][0] = lower
    case = {
        "case": "已确认的空内积契约",
        "synthesize": {"shapes": {"a": "[2, 0]", "b": "[0, 3]"}},
        "machine_check": {"kind": kind},
    }
    if kind == "raises_error":
        case["machine_check"]["error_type"] = "shape_mismatch"
    findings = check_concrete_sample_domain(spec, case, "boundary_conditions[0]")
    assert any(f["severity"] == "error" for f in findings) == expect_error, findings
