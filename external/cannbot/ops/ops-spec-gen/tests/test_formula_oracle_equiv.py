# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Stage 10 formula_oracle_equiv 单测。

覆盖：
  * SKIP 路径：formula_kind 非 numpy_expr / oracle absent / composition 模式 / framework 未装
  * PASS 路径：formula 与 oracle 在特殊值输入上等价（numpy.add / numpy.exp）
  * FAIL 路径：formula 使用了与 oracle 不同的计算形式，NaN/inf 模式发散
"""
from __future__ import annotations

import pytest

from evaluators.formula_oracle_equiv import stage_10


def _spec(
    formula="y = x + 1",
    formula_kind="numpy_expr",
    framework="numpy",
    api="numpy.add",
    absent=False,
    composition=None,
    output_id=None,
    kwargs=None,
    inputs=None,
    outputs=None,
    attributes=None,
) -> dict:
    oracle = {
        "framework": framework,
        "api": api,
        "absent": absent,
        "available_for_dtype": ["float32"],
    }
    if kwargs is not None:
        oracle["kwargs"] = kwargs
    if composition is not None:
        oracle["composition"] = composition
        oracle["api"] = None
        oracle["output"] = output_id
    return {
        "math_semantics": {
            "formula_kind": formula_kind,
            "formula": formula,
            "reference_oracle": oracle,
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
    }


class TestStage10Skip:
    def test_skip_non_numpy_formula_kind(self):
        s = _spec(formula_kind="textual_only")
        status, findings = stage_10(s)
        assert status == "SKIP"
        assert any("skipped_non_numpy" in f["rule_id"] for f in findings)

    def test_skip_oracle_absent(self):
        s = _spec(absent=True)
        status, findings = stage_10(s)
        assert status == "SKIP"
        assert any("absent" in f["rule_id"] for f in findings)

    def test_skip_composition_mode(self):
        s = _spec(
            composition=[
                {"id": "n", "api": "numpy.add", "args": ["x", "x"]},
            ],
            output_id="n",
        )
        status, findings = stage_10(s)
        assert status == "SKIP"
        assert any("composition_not_supported" in f["rule_id"] for f in findings)

    def test_skip_framework_not_installed(self):
        s = _spec(framework="nonexistent_framework", api="nonexistent_framework.add")
        status, findings = stage_10(s)
        assert status == "SKIP"
        assert any("framework_not_installed" in f["rule_id"] or "api_unreachable" in f["rule_id"] for f in findings)


class TestStage10Pass:
    def test_numpy_exp_equivalent(self):
        """formula y = np.exp(x) should match numpy.exp(x)."""
        s = _spec(
            formula="y = np.exp(x)",
            framework="numpy",
            api="numpy.exp",
        )
        status, findings = stage_10(s)
        assert status == "PASS", findings

    def test_numpy_square_equivalent(self):
        """formula y = x * x should match numpy.square(x) on finite values.
        NaN/inf patterns will also match since x*x and square(x) propagate identically."""
        s = _spec(
            formula="y = x * x",
            framework="numpy",
            api="numpy.square",
        )
        status, findings = stage_10(s)
        assert status == "PASS", findings

    def test_numpy_negative_equivalent(self):
        """formula y = -x should match numpy.negative(x)."""
        s = _spec(
            formula="y = -x",
            framework="numpy",
            api="numpy.negative",
        )
        status, findings = stage_10(s)
        assert status == "PASS", findings

    def test_zero_sign_no_false_positive_on_neg(self):
        """formula y = -x vs numpy.negative must not mis-flag zero sign.

        On +0 input both produce -0; on -0 input both produce +0. The new
        zero-sign comparison must not raise a false divergence here.
        """
        s = _spec(
            formula="y = -x",
            framework="numpy",
            api="numpy.negative",
        )
        status, findings = stage_10(s)
        assert status == "PASS", findings
        assert not any("zero_sign_divergence" in f["rule_id"] for f in findings)


class TestStage10Fail:
    def test_value_divergence_detected(self):
        """Formula computes exp(x) but oracle computes sin(x) → value divergence on normal inputs."""
        s = _spec(
            formula="y = np.exp(x)",
            framework="numpy",
            api="numpy.sin",
        )
        status, findings = stage_10(s)
        assert status == "FAIL"
        assert any(
            "value_divergence" in f["rule_id"]
            or "nan_pattern_divergence" in f["rule_id"]
            or "inf_pattern_divergence" in f["rule_id"]
            for f in findings
        )

    def test_nan_pattern_divergence_textbook_vs_incremental(self):
        """Simulate the Adam optimizer bug: textbook form vs incremental form.

        Textbook: m_new = beta * m + (1 - beta) * grad
        Incremental: m = m + (1 - beta) * (grad - m)

        When m = inf:
          textbook: beta*inf + (1-beta)*grad = inf  (no NaN)
          incremental: inf + (1-beta)*(grad - inf) = inf + (1-beta)*(-inf) = inf + (-inf) = NaN

        This test verifies stage 10 catches the NaN pattern difference.
        """
        # We need a 2-input operator: m and grad
        spec = {
            "math_semantics": {
                "formula_kind": "numpy_expr",
                "formula": "y = 0.9 * m + 0.1 * grad",
                "reference_oracle": {
                    "framework": "numpy",
                    "api": None,
                    "composition": [
                        {"id": "diff", "api": "numpy.subtract", "args": ["grad", "m"]},
                        {"id": "scaled", "api": "numpy.multiply", "args": ["diff"], "kwargs": {"x2": 0.1}},
                        {"id": "y", "api": "numpy.add", "args": ["m", "scaled"]},
                    ],
                    "output": "y",
                    "absent": False,
                    "available_for_dtype": ["float32"],
                },
            },
            "attributes": [],
            "inputs": [
                {"name": "m", "shape": {"symbolic": [2, 3]}, "dtype_set": ["float32"]},
                {"name": "grad", "shape": {"symbolic": [2, 3]}, "dtype_set": ["float32"]},
            ],
            "outputs": [{"name": "y"}],
            "dtype_policy": {
                "supported_combinations": [
                    {"inputs": {"m": "float32", "grad": "float32"}, "outputs": {"y": "float32"}},
                ],
            },
            "test_matrix": {"random": {"seed": 42}},
        }
        # composition mode → SKIP currently
        status, findings = stage_10(spec)
        assert status == "SKIP"
        assert any("composition_not_supported" in f["rule_id"] for f in findings)

    def test_nan_pattern_divergence_single_api(self):
        """Simpler version: formula produces NaN on inf input, oracle doesn't.

        formula: y = x - x  (produces NaN when x=inf, since inf - inf = NaN)
        oracle:  numpy.negative(x)  (produces -inf when x=inf, no NaN)
        """
        s = _spec(
            formula="y = x - x",
            framework="numpy",
            api="numpy.negative",
        )
        status, findings = stage_10(s)
        # On inf input: formula → NaN, oracle → -inf → NaN pattern divergence
        # On normal input: formula → 0, oracle → -x → value divergence
        assert status == "FAIL"
        assert any(
            "nan_pattern_divergence" in f["rule_id"] or "value_divergence" in f["rule_id"]
            for f in findings
        )

    def test_zero_sign_divergence_detected(self):
        """+0/-0 sign divergence is caught independently of NaN/inf/value.

        formula: y = 0 - x  (0 - (+0) = +0; the subtraction clears the sign)
        oracle:  numpy.negative(x)  (neg(+0) = -0)

        These agree on every finite non-zero value, NaN and inf patterns, but
        diverge on the sign of zero. Stage 10 must catch this via the
        zero_sign_divergence rule when fed the pos_zero / neg_zero inputs.
        """
        s = _spec(
            formula="y = 0 - x",
            framework="numpy",
            api="numpy.negative",
        )
        status, findings = stage_10(s)
        assert status == "FAIL"
        assert any("zero_sign_divergence" in f["rule_id"] for f in findings), findings


def test_gen_special_tensors_empty_shape_returns_labelled_sets():
    """空输入（first_arr.size==0）早退路径须返回 _label-keyed 结构（PR review #5 回归锁）。

    此前返回 [{"normal": dict(base)}]——结构与正常路径不一致，stage10 formula 沙箱拿到的是名为
    "normal" 的变量（值=dict）而非输入张量 x，导致 formula 因缺输入而失败。
    """
    import numpy as np
    from evaluators.formula_oracle_equiv import _gen_special_tensors
    spec = {"inputs": [{"name": "x", "shape": {"symbolic": [0, 4]}, "dtype_set": ["float32"]}]}
    sets = _gen_special_tensors(np, spec, {"x": "float32"}, 0)
    assert sets, "空 shape 应至少返回 normal 集"
    for s in sets:
        assert "_label" in s, f"测试集须含 _label（review #5），得到 {list(s.keys())}"
        assert "x" in s, "输入张量须在顶层（不应嵌套在 'normal' 下）"


def test_gen_special_tensors_preserves_integer_dtype():
    """整数 dtype 输入不得被强制 astype('float32')（PR review Huang-Peng 回归锁）。

    此前 _gen_special_tensors 对所有输入无条件 .astype('float32')，导致：
      1) int32 > 2^24 丢精度；2) 整数语义被改变；3) NaN/inf 注入处的
      dtype.kind=='f' 恒为真，向整数输入注入 NaN/inf 误报。
    修复后仅对 dtype.kind=='f' 做 astype，整数/bool 保持原类型。
    """
    import numpy as np
    from evaluators.formula_oracle_equiv import _gen_special_tensors
    spec = {"inputs": [{"name": "x", "shape": {"symbolic": [2, 3]}, "dtype_set": ["int32"]}]}
    sets = _gen_special_tensors(np, spec, {"x": "int32"}, 0)
    assert sets, "int32 输入应返回测试集"
    for s in sets:
        assert s["x"].dtype == np.int32, f"int32 输入须保持 int32，得到 {s['x'].dtype}"
        assert not np.isnan(s["x"]).any(), "int32 输入不得被注入 NaN"
