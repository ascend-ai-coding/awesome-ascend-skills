# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Main pipeline 反向集成单测——端到端跑 scripts/validate_spec.py 子进程。

* TestStage6Tags：双轨匹配（tag 优先 + 子串兜底）
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


class TestStage6Tags:
    """stage 6 双轨匹配：tag 命中 / 子串兜底 / 都不命中报缺失。"""

    def _run(self, spec_yaml: str, tmp_path: Path) -> subprocess.CompletedProcess:
        path = tmp_path / "spec.yaml"
        path.write_text(spec_yaml, encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(VALIDATOR), str(path), "--json", "--stage", "6"],
            capture_output=True, text=True, timeout=30,
        )

    def test_explicit_tag_covers_requirement_with_unrelated_description(self, tmp_path):
        # 改 softmax example 的"reduce 轴长度为 1" case 描述为完全不含子串关键词的措辞，
        # 但保留 tags: [reduce_axis_size_1] —— stage 6 应通过 tag 匹配认可。
        good = (SKILL_ROOT / "examples" / "softmax" / "spec.yaml").read_text(encoding="utf-8")
        modified = good.replace(
            '- case: "reduce 轴长度为 1"',
            '- case: "退化为常数情形"',
        )
        result = self._run(modified, tmp_path)
        out = json.loads(result.stdout)
        stage_6 = next(s for s in out["stages"] if s["stage_id"] == 6)
        # tags 仍带着 reduce_axis_size_1，要求被覆盖
        assert stage_6["status"] == "PASS", stage_6["findings"]

    def test_no_tag_no_keyword_fails(self, tmp_path):
        # 同时移除 tag 和关键词，stage 6 必须 FAIL
        good = (SKILL_ROOT / "examples" / "softmax" / "spec.yaml").read_text(encoding="utf-8")
        modified = good.replace(
            '- case: "reduce 轴长度为 1"\n    tags: [reduce_axis_size_1]\n',
            '- case: "退化为常数情形"\n',
        )
        result = self._run(modified, tmp_path)
        out = json.loads(result.stdout)
        stage_6 = next(s for s in out["stages"] if s["stage_id"] == 6)
        assert stage_6["status"] == "FAIL"
        assert any("reduce_axis_size_1" in (f.get("message") or "")
                   for f in stage_6["findings"])


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

broadcast: {kind: none}

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

broadcast: {kind: none}

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
