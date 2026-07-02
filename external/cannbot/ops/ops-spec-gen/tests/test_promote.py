# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""dtype promote 表单测。

覆盖 promote_pair 所有边界（同 dtype / 同 rank signed-unsigned / 整数与浮点混合 /
窄浮点 fp4/fp8/hf8 / complex 吸收）。其他 dtype_rule 表达式行为在 test_dtype_eval.py。
"""
from __future__ import annotations

import pytest

from evaluators.promote import promote_pair, promote_many, complex_of, real_of
from evaluators.types import DslError


class TestDtypePromote:
    @pytest.mark.parametrize("a,b,expected", [
        ("float16", "float16", "float16"),
        ("float32", "float32", "float32"),
        ("float16", "float32", "float32"),
        ("bfloat16", "float32", "float32"),
        ("bfloat16", "float16", "float32"),  # mixed narrow → fp32
        ("int8", "int32", "int32"),
        ("int32", "float32", "float64"),  # int32 (32-bit) 不能在 fp32 mantissa 装下 → fp64
        ("int32", "float16", "float64"),  # 静默截 fp16 是 bug，必须升 fp64
        ("int16", "float16", "float32"),  # int16 mantissa 超 fp16 → fp32
        ("int8", "float16", "float16"),  # int8 在 fp16 内 → 保持 fp16
        ("int8", "bfloat16", "bfloat16"),  # bf16 也能装 int8 → 保持 bf16（不能漂到 fp16）
        ("int64", "float32", "float64"),  # int64 forces fp64
        ("complex64", "float32", "complex64"),
        ("complex64", "float64", "complex128"),
        ("int32", "uint32", "int64"),  # signed/unsigned 同宽 → 升一档 signed
        ("int8", "uint8", "int16"),
        ("int16", "uint16", "int32"),
        # ---- int4 / uint4 ----
        ("int4", "int4", "int4"),  # 同 dtype 不变
        ("uint4", "uint4", "uint4"),
        ("int4", "uint4", "int8"),  # 同 rank signed/unsigned → 升一档
        ("int4", "int8", "int8"),  # rank 0 vs rank 1 → int8 胜
        ("int4", "int16", "int16"),
        ("int4", "float16", "float16"),
        ("int4", "bfloat16", "bfloat16"),
        ("int4", "float32", "float32"),
        ("uint4", "float16", "float16"),
        ("bool", "int4", "int4"),
        # ---- 窄浮点 fp8 / fp4 / hf8 ----
        ("float8_e4m3fn", "float8_e4m3fn", "float8_e4m3fn"),
        ("float8_e5m2", "float8_e5m2", "float8_e5m2"),
        ("hifloat8", "hifloat8", "hifloat8"),
        ("float4_e2m1", "float4_e2m1", "float4_e2m1"),
        ("float8_e4m3fn", "float8_e5m2", "float16"),  # 异型窄浮点 → fp16
        ("float8_e4m3fn", "hifloat8", "float16"),
        ("float4_e2m1", "float8_e4m3fn", "float16"),
        ("float8_e4m3fn", "float16", "float16"),
        ("float8_e5m2", "bfloat16", "bfloat16"),
        ("float8_e4m3fn", "float32", "float32"),
        ("float4_e2m1", "float64", "float64"),
        ("float8_e4m3fn", "int8", "float16"),
        ("float8_e4m3fn", "int32", "float64"),
        ("float8_e4m3fn", "complex64", "complex64"),
        # ---- complex32 ----
        ("complex32", "complex32", "complex32"),  # 同型保持
        ("complex32", "complex64", "complex64"),  # 升宽
        ("complex32", "complex128", "complex128"),
        ("complex32", "float16", "complex64"),  # complex+fp 保守升 c64
        ("complex32", "float32", "complex64"),
        ("complex32", "int32", "complex64"),
        # ---- uint1 (1-bit unsigned int, 0/1, distinct from bool) ----
        ("uint1", "uint1", "uint1"),
        ("bool", "uint1", "uint1"),
        ("uint1", "int4", "int4"),
        ("uint1", "uint4", "uint4"),
        ("uint1", "int8", "int8"),
        ("uint1", "uint8", "uint8"),
        ("uint1", "float16", "float16"),
        # ---- float4_e1m2 / float8_e8m0 (new narrow floats) ----
        ("float4_e1m2", "float4_e1m2", "float4_e1m2"),
        ("float4_e1m2", "float4_e2m1", "float16"),  # 异型 narrow → fp16
        ("float8_e8m0", "float8_e8m0", "float8_e8m0"),
        ("float8_e8m0", "float8_e4m3fn", "float16"),  # 异型 narrow → fp16
        ("float8_e8m0", "float16", "float16"),
        ("float4_e1m2", "float32", "float32"),
    ])
    def test_pairs(self, a, b, expected):
        assert promote_pair(a, b) == expected

    def test_many(self):
        assert promote_many(["float16", "float32", "bfloat16"]) == "float32"

    def test_unknown_dtype(self):
        with pytest.raises(DslError):
            promote_pair("foo", "float32")


class TestComplexOfRealOf:
    """complex_of / real_of 走位宽对应（fp16 ↔ complex32 / fp32 ↔ complex64 / fp64 ↔ complex128）。

    锁住 fp16 → complex32 这条不再退化到 complex64。语义动机：complex32 由两个 fp16
    虚实部组成；与 real_of(complex32)=float16 互逆。bfloat16 没有 complex 变体（complex
    的实部只有 fp16/fp32/fp64 三档），显式抛 DslError，避免静默落到 complex32 丢精度。
    """

    @pytest.mark.parametrize("real,cmplx", [
        ("float16", "complex32"),
        ("float32", "complex64"),
        ("float64", "complex128"),
    ])
    def test_complex_of(self, real, cmplx):
        assert complex_of(real) == cmplx

    def test_complex_of_bfloat16_rejected(self):
        with pytest.raises(DslError, match="bfloat16"):
            complex_of("bfloat16")

    @pytest.mark.parametrize("cmplx,real", [
        ("complex32", "float16"),
        ("complex64", "float32"),
        ("complex128", "float64"),
    ])
    def test_real_of(self, cmplx, real):
        assert real_of(cmplx) == real
