# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""DtypeEvaluator 单测 —— 在 dtype 字符串上执行 numpy 子集表达式。"""
from __future__ import annotations

import pytest

from evaluators.dtype_eval import evaluate_dtype_rule
from evaluators.types import DslError


class TestSameAs:
    def test_propagation(self):
        out = evaluate_dtype_rule(
            "y.dtype = x.dtype",
            output_name="y",
            input_dtypes={"x": "float16"},
        )
        assert out == "float16"

    def test_other_dtypes(self):
        for dt in ("bfloat16", "int32", "complex64"):
            out = evaluate_dtype_rule(
                "y.dtype = x.dtype",
                output_name="y",
                input_dtypes={"x": dt},
            )
            assert out == dt


class TestPromoteTypes:
    def test_basic(self):
        out = evaluate_dtype_rule(
            "c.dtype = np.promote_types(a.dtype, b.dtype)",
            output_name="c",
            input_dtypes={"a": "float16", "b": "float32"},
        )
        assert out == "float32"

    def test_bf16_fp32(self):
        out = evaluate_dtype_rule(
            "c.dtype = np.promote_types(a.dtype, b.dtype)",
            output_name="c",
            input_dtypes={"a": "bfloat16", "b": "float32"},
        )
        assert out == "float32"


class TestFixed:
    def test_np_int32(self):
        out = evaluate_dtype_rule(
            "y.dtype = np.int32",
            output_name="y",
            input_dtypes={},
        )
        assert out == "int32"

    def test_np_float16(self):
        out = evaluate_dtype_rule(
            "y.dtype = np.float16",
            output_name="y",
            input_dtypes={},
        )
        assert out == "float16"

    def test_np_int64(self):
        out = evaluate_dtype_rule(
            "y.dtype = np.int64",
            output_name="y",
            input_dtypes={},
        )
        assert out == "int64"


class TestConditional:
    def test_ifexp_complex_of(self):
        # 模拟 complex 例的 dtype_rule
        rule = """
out.dtype = np.complex64 if real.dtype == np.float32 else (
    np.complex128 if real.dtype == np.float64 else np.complex64
)
"""
        out = evaluate_dtype_rule(
            rule,
            output_name="out",
            input_dtypes={"real": "float32"},
        )
        assert out == "complex64"

        out = evaluate_dtype_rule(
            rule,
            output_name="out",
            input_dtypes={"real": "float64"},
        )
        assert out == "complex128"


class TestUnresolved:
    def test_unknown_input(self):
        with pytest.raises(DslError, match="unresolved_symbol|未声明"):
            evaluate_dtype_rule(
                "y.dtype = z.dtype",                  # z 未声明
                output_name="y",
                input_dtypes={"x": "float16"},
            )

    def test_no_assignment(self):
        with pytest.raises(DslError, match="未给.*赋值|unresolved_symbol"):
            evaluate_dtype_rule(
                "tmp = x.dtype",
                output_name="y",
                input_dtypes={"x": "float16"},
            )


class TestSandbox:
    def test_import_blocked(self):
        with pytest.raises(DslError):
            evaluate_dtype_rule(
                "y.dtype = __import__('os').name",
                output_name="y",
                input_dtypes={},
            )


class TestNamespaceSync:
    """np.<dtype> 字面写法须覆盖 schema enum 中全部 dtype。

    回归项：dtype_eval._NpDtypeNamespace 历史上硬编码 17 个 numpy-native
    名称，缺 ascend 私有窄浮点（hifloat8 / float8_e4m3fn / float8_e5m2 /
    float4_e2m1）与 int4 / uint4。已改为从 promote._ALL_DTYPES 动态注入；
    本测试锁死 schema enum / promote 表 / dtype_eval 三处单一真值。
    """

    def test_all_dtypes_addressable_via_np(self):
        from evaluators import promote

        for dt in sorted(promote._ALL_DTYPES):
            out = evaluate_dtype_rule(
                f"y.dtype = np.{dt}",
                output_name="y",
                input_dtypes={},
            )
            assert out == dt, f"np.{dt} → {out!r}, expected {dt!r}"

    def test_ascend_private_dtypes_evaluable(self):
        """硬编码若干 ascend 私有 dtype 与 complex32，确保新增 dtype 时这条测试也跟着扩。"""
        for dt in ("hifloat8",
                   "float8_e4m3fn", "float8_e5m2", "float8_e8m0",
                   "float4_e2m1", "float4_e1m2",
                   "int4", "uint4", "uint1", "complex32"):
            out = evaluate_dtype_rule(
                f"y.dtype = np.{dt}",
                output_name="y",
                input_dtypes={},
            )
            assert out == dt
