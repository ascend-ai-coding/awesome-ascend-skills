# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Stage 8 formula_smoke_eval 沙箱单测。

覆盖：成功路径（加法 / softmax 多行）/ AST 白名单（import 拒绝 / dunder 属性逃逸拒绝）/
banned name / numpy API 拼错 / 输出未赋值 / SKIP 路径（textual_only）/
dtype stand-in（int4/uint4/bf16/fp4/fp8 系列）/ complex 输出。
"""
from __future__ import annotations

import pytest

from evaluators.formula_eval import stage_8  # noqa: E402


def _basic_spec(formula: str, dtype: str = "float32") -> dict:
    """Minimal spec dict that stage 8 can execute on."""
    return {
        "math_semantics": {"formula_kind": "numpy_expr", "formula": formula},
        "inputs": [
            {"name": "x", "shape": {"symbolic": ["...d"]}, "dtype_set": [dtype]},
        ],
        "attributes": [{"name": "dim", "default": -1}],
        "outputs": [{"name": "y"}],
        "dtype_policy": {
            "supported_combinations": [
                {"inputs": {"x": dtype}, "outputs": {"y": dtype}},
            ],
        },
    }


class TestStage8:
    def test_simple_addition_passes(self):
        s = _basic_spec("y = x + 1")
        status, findings = stage_8(s)
        assert status == "PASS", findings

    def test_softmax_multiline_passes(self):
        s = _basic_spec(
            "m = x.max(axis=dim, keepdims=True)\n"
            "e = np.exp(x - m)\n"
            "y = e / e.sum(axis=dim, keepdims=True)\n"
        )
        status, _ = stage_8(s)
        assert status == "PASS"

    def test_numpy_api_typo_fails(self):
        # x.maks instead of x.max
        s = _basic_spec("y = x.maks(axis=dim, keepdims=True)")
        status, findings = stage_8(s)
        assert status == "FAIL"
        assert any("numpy_eval_error" in f["rule_id"] for f in findings)

    def test_undefined_output_fails(self):
        # formula assigns to z, not y
        s = _basic_spec("z = x + 1")
        status, findings = stage_8(s)
        assert status == "FAIL"
        assert any("missing_output" in f["rule_id"] for f in findings)

    def test_textual_only_skips(self):
        s = _basic_spec("anything")
        s["math_semantics"]["formula_kind"] = "textual_only"
        status, findings = stage_8(s)
        assert status == "SKIP"

    def test_disallowed_ast_fails(self):
        s = _basic_spec("import os\ny = x")
        status, findings = stage_8(s)
        assert status == "FAIL"
        assert any("ast_disallowed" in f["rule_id"] for f in findings)

    def test_banned_name_fails(self):
        s = _basic_spec("y = __import__('os').getcwd()")
        status, findings = stage_8(s)
        assert status == "FAIL"
        # Either banned_name or syntax/numpy error — accept any
        assert any(
            f["rule_id"].startswith("formula_smoke_eval.")
            for f in findings
        )

    def test_dunder_attribute_escape_blocked(self):
        # 经典沙箱逃逸：().__class__.__bases__[0].__subclasses__() 不依赖任何 banned name。
        # 沙箱必须在 AST 层面拒绝 dunder 属性。
        s = _basic_spec("y = ().__class__.__bases__[0]")
        status, findings = stage_8(s)
        assert status == "FAIL"
        assert any("banned_name" in f["rule_id"] for f in findings)

    def test_int4_standin_runs(self):
        # int4 在 spec 层声明，stage 8 用 int8 容器（值域 [-8, 7]）跑通；
        # dtype 比对自动放过 int4↔int8 这一对（同 bf16↔fp32）
        s = _basic_spec("y = x + 1", dtype="int4")
        status, findings = stage_8(s)
        assert status == "PASS", findings

    def test_uint4_standin_runs(self):
        s = _basic_spec("y = x + 1", dtype="uint4")
        status, findings = stage_8(s)
        assert status == "PASS", findings

    @pytest.mark.parametrize("dtype", [
        "float8_e4m3fn", "float8_e5m2", "float8_e8m0", "hifloat8",
        "float4_e2m1", "float4_e1m2",
    ])
    def test_narrow_float_standin_runs(self, dtype):
        # 窄浮点 stand-in（fp16 容器）；formula 简单加法；spec 输出 dtype 与输入相同
        s = _basic_spec("y = x + 1.0", dtype=dtype)
        status, findings = stage_8(s)
        assert status == "PASS", findings

    def test_uint1_standin_runs(self):
        s = _basic_spec("y = x", dtype="uint1")
        status, findings = stage_8(s)
        assert status == "PASS", findings


# ---------- stage 8 复数支持（#14） ---------------------------------------


class TestStage8Complex:
    def test_complex_formula(self):
        # complex64 input → complex output (basic identity)
        spec = {
            "math_semantics": {
                "formula_kind": "numpy_expr",
                "formula": "out = real + 1j * imag",
            },
            "inputs": [
                {"name": "real", "shape": {"symbolic": ["...a"]}, "dtype_set": ["float32"]},
                {"name": "imag", "shape": {"symbolic": ["...b"]}, "dtype_set": ["float32"]},
            ],
            "attributes": [],
            "outputs": [{"name": "out"}],
            "dtype_policy": {
                "supported_combinations": [
                    {"inputs": {"real": "float32", "imag": "float32"},
                     "outputs": {"out": "complex64"}},
                ],
            },
        }
        from evaluators.formula_eval import stage_8 as fs
        status, findings = fs(spec)
        # 输出 dtype 是 complex64，runtime 跑出来也是 complex64 — PASS
        assert status == "PASS", findings

