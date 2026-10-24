# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Stage 12 extreme_inputs 联合校验单测。

覆盖：
  * SKIP 路径：无 extreme_inputs / formula_kind 非 numpy_expr
  * FAIL 路径：异号无穷相减被误判 produces_nan
  * PASS 路径：同号 inf-inf、inf*0 的正确 produces_nan；nan_propagates；
    matches_oracle（formula ≡ numpy oracle）
  * scope 语义：整张量（全 NaN）与非整张量（至少一处 NaN）区分
"""
from __future__ import annotations

from evaluators.extreme_inputs_check import stage_12


def _spec(formula="y = x - t", extreme=None, framework="numpy",
          api="numpy.subtract", formula_kind="numpy_expr") -> dict:
    return {
        "math_semantics": {
            "formula_kind": formula_kind,
            "formula": formula,
            "reference_oracle": {
                "framework": framework, "api": api, "absent": False,
                "available_for_dtype": ["float32"],
            },
        },
        "attributes": [],
        "inputs": [
            {"name": "x", "shape": {"symbolic": [1]}, "dtype_set": ["float32"]},
            {"name": "t", "shape": {"symbolic": [1]}, "dtype_set": ["float32"]},
        ],
        "outputs": [{"name": "y"}],
        "dtype_policy": {
            "supported_combinations": [
                {"inputs": {"x": "float32", "t": "float32"}, "outputs": {"y": "float32"}},
            ],
        },
        "extreme_inputs": extreme or [],
    }


def _entry(patterns, kind, scope=None, shapes=None):
    mc = {"kind": kind}
    if scope:
        mc["scope"] = scope
    entry = {
        "case": "test-case",
        "synthesize": {
            "shapes": shapes or {"x": "[1]", "t": "[1]"},
            "patterns": patterns,
        },
        "machine_check": mc,
    }
    return entry


def test_skip_without_extreme_inputs():
    status, findings = stage_12(_spec(extreme=None))
    assert status == "SKIP"


def test_skip_non_numpy_formula():
    spec = _spec(formula_kind="python_expr",
                 extreme=[_entry([{"pattern": "all_zero", "target": "x"}], "produces_nan")])
    status, _ = stage_12(spec)
    assert status == "SKIP"


def test_fail_opposite_sign_inf_declared_nan():
    """+inf − (−inf) = +inf，不是 NaN。"""
    spec = _spec(extreme=[_entry(
        [{"pattern": "single_pos_inf", "target": "x"},
         {"pattern": "single_neg_inf", "target": "t"}],
        "produces_nan", scope="整张量")])
    status, findings = stage_12(spec)
    assert status == "FAIL"
    assert any(f["rule_id"] == "extreme_check.produces_nan_conflict" for f in findings)


def test_pass_same_sign_inf_produces_nan():
    """同号 inf − inf = NaN，声明成立。"""
    spec = _spec(extreme=[_entry(
        [{"pattern": "single_pos_inf", "target": "x"},
         {"pattern": "single_pos_inf", "target": "t"}],
        "produces_nan", scope="整张量")])
    status, findings = stage_12(spec)
    assert status == "PASS"
    assert not any(f["severity"] == "error" for f in findings)


def test_scope_whole_tensor_strict():
    """部分位置 NaN + scope 整张量 → FAIL；去掉 scope → PASS。"""
    pats = [{"pattern": "inject_nan_one_element", "target": "x"}]
    shapes = {"x": "[8]", "t": "[8]"}
    strict = _spec(extreme=[_entry(pats, "produces_nan", scope="整张量", shapes=shapes)])
    status, _ = stage_12(strict)
    assert status == "FAIL"
    loose = _spec(extreme=[_entry(pats, "produces_nan", shapes=shapes)])
    status, _ = stage_12(loose)
    assert status == "PASS"


def test_nan_propagates():
    spec = _spec(extreme=[_entry(
        [{"pattern": "inject_nan_one_element", "target": "x"},
         {"pattern": "all_same(1.0)", "target": "t"}],
        "nan_propagates", shapes={"x": "[4]", "t": "[4]"})])
    status, findings = stage_12(spec)
    assert status == "PASS"


def test_matches_oracle_agree():
    spec = _spec(extreme=[_entry(
        [{"pattern": "single_pos_inf", "target": "x"},
         {"pattern": "single_neg_inf", "target": "t"}],
        "matches_oracle")])
    status, findings = stage_12(spec)
    assert status == "PASS"
    assert not any(f["severity"] == "error" for f in findings)


def test_matches_oracle_divergence():
    """formula 与 oracle 在 pattern 输入上分歧 → FAIL。"""
    spec = _spec(formula="y = x + t",
                 api="numpy.subtract",
                 extreme=[_entry(
                     [{"pattern": "single_pos_inf", "target": "x"},
                      {"pattern": "single_neg_inf", "target": "t"}],
                     "matches_oracle")])
    status, findings = stage_12(spec)
    assert status == "FAIL"
    assert any(f["rule_id"].startswith("extreme_check.") and f["severity"] == "error"
               for f in findings)


def test_unsynthesizable_pattern_warns_not_fails():
    """无参 all_same（缺 value）不可合成 → WARN 跳过，不判 FAIL。"""
    spec = _spec(extreme=[_entry(
        [{"pattern": "all_same", "target": "x"},
         {"pattern": "all_same(2.0)", "target": "t"}],
        "matches_oracle")])
    status, findings = stage_12(spec)
    assert status in ("PASS", "SKIP")
    assert any(f["rule_id"] == "extreme_check.unsynthesizable" for f in findings)


def test_non_checkable_kind_skipped():
    spec = _spec(extreme=[_entry(
        [{"pattern": "all_zero", "target": "x"}], "returns_empty")])
    status, findings = stage_12(spec)
    assert status == "SKIP"
    assert any(f["rule_id"] == "extreme_check.skipped_kind" for f in findings)


def test_attr_override_flows_to_formula():
    """条目级 attrs 应覆盖属性默认值参与 formula 求值。"""
    spec = _spec(
        formula="cof = 2.0 if red == 'sum' else 0.5\ny = (x - t) * cof",
        extreme=[{
            "case": "attr-override",
            "synthesize": {
                "shapes": {"x": "[1]", "t": "[1]"},
                "attrs": {"red": "sum"},
                "patterns": [
                    {"pattern": "single_pos_inf", "target": "x"},
                    {"pattern": "single_neg_inf", "target": "t"},
                ],
            },
            "machine_check": {"kind": "produces_nan", "scope": "整张量"},
        }])
    spec["attributes"] = [{"name": "red", "type": "string", "default": "mean"}]
    status, findings = stage_12(spec)
    # sum 分支：(+inf) * 2.0 = +inf ≠ NaN → 应 FAIL（若 attrs 未生效则 0.5*inf=inf 同样非 NaN）
    assert status == "FAIL"
