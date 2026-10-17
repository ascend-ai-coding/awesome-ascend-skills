# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Stage 11 invariant_exec 单测。

覆盖：
  * SKIP 路径：formula_kind 非 numpy_expr / 无 invariants
  * PASS 路径：value 组（elementwise_ge / reduce_equals / range_in）+ algebraic 组
              （equals_under_swap / equals_input_when_other_is_zero / equals_when_input_is_zero）
  * FAIL 路径：formula 违反声明的 invariant 被抓（形式错误在此暴露）
  * 边界：索引输入保护 / 空输出保护 / 不可执行 kind info 跳过
"""
from __future__ import annotations

import pytest

from evaluators.invariant_exec import stage_11


def _spec(
    formula="y = x + 1",
    formula_kind="numpy_expr",
    invariants=None,
    inputs=None,
    outputs=None,
    attributes=None,
) -> dict:
    return {
        "math_semantics": {
            "formula_kind": formula_kind,
            "formula": formula,
            "invariants": invariants or [],
        },
        "attributes": attributes or [],
        "inputs": inputs or [{"name": "x", "shape": {"symbolic": [2, 3]}, "dtype_set": ["float32"]}],
        "outputs": outputs or [{"name": "y"}],
        "dtype_policy": {
            "supported_combinations": [
                {"inputs": {"x": "float32"}, "outputs": {"y": "float32"}},
            ],
        },
        "test_matrix": {"random": {"seed": 42}},
        "numerical_tolerance": {"per_dtype": {"float32": {"atol": 1.0e-5, "rtol": 1.0e-5}}},
    }


_DUAL_INPUTS = [
    {"name": "x", "shape": {"symbolic": [2, 3]}, "dtype_set": ["float32"]},
    {"name": "y", "shape": {"symbolic": [2, 3]}, "dtype_set": ["float32"]},
]


class TestStage11Skip:
    def test_skip_non_numpy_formula_kind(self):
        s = _spec(formula_kind="textual_only", invariants=[{"kind": "elementwise_ge", "value": 0.0}])
        status, findings = stage_11(s)
        assert status == "SKIP"
        assert any("skipped_non_numpy" in f["rule_id"] for f in findings)

    def test_skip_no_invariants(self):
        s = _spec(invariants=[])
        status, findings = stage_11(s)
        assert status == "SKIP"
        assert any("no_invariants" in f["rule_id"] for f in findings)


class TestStage11ValuePass:
    def test_elementwise_ge_pass(self):
        """y = abs(x) satisfies out >= 0."""
        s = _spec(
            formula="y = np.abs(x)",
            invariants=[{"name": "nonneg", "kind": "elementwise_ge", "value": 0.0}],
        )
        status, findings = stage_11(s)
        assert status == "PASS", findings

    def test_range_in_pass(self):
        """y = sigmoid(x) satisfies out in [0, 1]."""
        s = _spec(
            formula="y = 1.0 / (1.0 + np.exp(-x))",
            invariants=[{"name": "unit_interval", "kind": "range_in", "range": [0.0, 1.0]}],
        )
        status, findings = stage_11(s)
        assert status == "PASS", findings

    def test_reduce_equals_pass(self):
        """row-normalized exp satisfies sum(axis=1) == 1."""
        s = _spec(
            formula="y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)",
            invariants=[{"name": "sum_to_one", "kind": "reduce_equals",
                         "reducer": "sum", "axis": 1, "value": 1.0, "tolerance_inherit": True}],
        )
        status, findings = stage_11(s)
        assert status == "PASS", findings

    def test_reduce_equals_axis_from_attr(self):
        """reduce_equals axis=${attr.dim} resolves via attribute default."""
        s = _spec(
            formula="y = np.exp(x) / np.sum(np.exp(x), axis=dim, keepdims=True)",
            invariants=[{"name": "sum_to_one", "kind": "reduce_equals",
                         "reducer": "sum", "axis": "${attr.dim}", "value": 1.0, "tolerance_inherit": True}],
            attributes=[{"name": "dim", "default": 1}],
        )
        status, findings = stage_11(s)
        assert status == "PASS", findings


class TestStage11AlgebraicPass:
    def test_equals_under_swap_pass(self):
        """z = x + y is commutative."""
        s = _spec(
            formula="z = x + y",
            inputs=_DUAL_INPUTS,
            outputs=[{"name": "z"}],
            invariants=[{"name": "commutative", "kind": "equals_under_swap", "swap": ["x", "y"]}],
        )
        status, findings = stage_11(s)
        assert status == "PASS", findings

    def test_identity_when_other_zero_pass(self):
        """z = x + y; zeroing y leaves z == x."""
        s = _spec(
            formula="z = x + y",
            inputs=_DUAL_INPUTS,
            outputs=[{"name": "z"}],
            invariants=[{"name": "id_zero", "kind": "equals_input_when_other_is_zero",
                         "identity_input": "x", "zero_input": "y"}],
        )
        status, findings = stage_11(s)
        assert status == "PASS", findings

    def test_equals_when_input_is_zero_pass(self):
        """y = x * 2; zeroing x yields y == 0."""
        s = _spec(
            formula="y = x * 2",
            invariants=[{"name": "zero_absorb", "kind": "equals_when_input_is_zero", "value": 0.0}],
        )
        status, findings = stage_11(s)
        assert status == "PASS", findings


class TestStage11Fail:
    def test_elementwise_ge_violated(self):
        """y = x with x drawn from normal has negative values → violates out >= 0."""
        s = _spec(
            formula="y = x",
            invariants=[{"name": "nonneg", "kind": "elementwise_ge", "value": 0.0}],
        )
        status, findings = stage_11(s)
        assert status == "FAIL"
        assert any("elementwise_ge_violated" in f["rule_id"] for f in findings)

    def test_reduce_equals_violated(self):
        """y = exp(x) is NOT normalized → sum(axis=1) != 1."""
        s = _spec(
            formula="y = np.exp(x)",
            invariants=[{"name": "sum_to_one", "kind": "reduce_equals",
                         "reducer": "sum", "axis": 1, "value": 1.0, "tolerance_inherit": True}],
        )
        status, findings = stage_11(s)
        assert status == "FAIL"
        assert any("reduce_equals_violated" in f["rule_id"] for f in findings)

    def test_equals_under_swap_violated(self):
        """z = x - y is NOT commutative → swap invariant fails."""
        s = _spec(
            formula="z = x - y",
            inputs=_DUAL_INPUTS,
            outputs=[{"name": "z"}],
            invariants=[{"name": "commutative", "kind": "equals_under_swap", "swap": ["x", "y"]}],
        )
        status, findings = stage_11(s)
        assert status == "FAIL"
        assert any("equals_under_swap_violated" in f["rule_id"] for f in findings)

    def test_identity_when_other_zero_violated(self):
        """z = x - y; zeroing y yields z == x, but z = x * y + x violates it."""
        s = _spec(
            formula="z = x * y + 1",
            inputs=_DUAL_INPUTS,
            outputs=[{"name": "z"}],
            invariants=[{"name": "id_zero", "kind": "equals_input_when_other_is_zero",
                         "identity_input": "x", "zero_input": "y"}],
        )
        status, findings = stage_11(s)
        assert status == "FAIL"
        assert any("equals_input_when_other_is_zero_violated" in f["rule_id"] for f in findings)


class TestStage11Edge:
    def test_kind_not_executable_emits_info(self):
        """Structural kinds (no_leak_intermediates) are info-skipped, not errors."""
        s = _spec(
            formula="y = x",
            invariants=[{"name": "no_leak", "kind": "no_leak_intermediates"}],
        )
        status, findings = stage_11(s)
        assert status == "PASS"
        assert any("kind_not_executable" in f["rule_id"] for f in findings)

    def test_index_input_not_zeroed(self):
        """equals_when_input_is_zero must NOT zero the integer `axis` input.

        Mirrors reduce_sum: x is data (zero → output 0), axis is an index
        tensor that must be left alone.
        """
        s = _spec(
            formula="y = np.sum(x, axis=0)",
            inputs=[
                {"name": "x", "shape": {"symbolic": [2, 3]}, "dtype_set": ["float32"]},
                {"name": "axis", "shape": {"symbolic": []}, "dtype_set": ["int32"]},
            ],
            invariants=[{"name": "zero_in_zero_out", "kind": "equals_when_input_is_zero", "value": 0.0}],
        )
        status, findings = stage_11(s)
        # No false violation from zeroing the axis index tensor.
        assert status == "PASS", findings

    def test_empty_output_does_not_crash(self):
        """IndexGather-style empty output (nonzero on all-zero input) must not crash.

        equals_when_input_is_zero with empty produced array → skip cleanly.
        """
        s = _spec(
            formula="y = x[x != 0]",  # all-zero input → empty output
            invariants=[{"name": "zero_empty", "kind": "equals_when_input_is_zero", "value": 0.0}],
        )
        status, findings = stage_11(s)
        assert status in ("PASS", "SKIP")
        assert not any(f["severity"] == "error" for f in findings)
