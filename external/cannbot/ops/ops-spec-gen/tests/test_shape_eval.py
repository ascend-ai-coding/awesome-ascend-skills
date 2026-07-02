# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""SymbolicShapeEvaluator 单测 —— 在 SymbolicShape 上执行 numpy 子集表达式。"""
from __future__ import annotations

import pytest

from evaluators.parser import parse_shape_literal
from evaluators.shape_eval import evaluate_shape_rule
from evaluators.types import DslError


def _sh(lst):
    return parse_shape_literal(lst)


class TestSameShape:
    def test_same_as(self):
        out = evaluate_shape_rule(
            "y.shape = x.shape",
            output_name="y",
            inputs={"x": _sh(["...d"])},
            attr_values={},
        )
        assert str(out) == "[...d]"

    def test_explicit_passthrough(self):
        out = evaluate_shape_rule(
            "y.shape = x.shape",
            output_name="y",
            inputs={"x": _sh(["M", "K"])},
            attr_values={},
        )
        assert str(out) == "[M, K]"


class TestBroadcastShape:
    def test_broadcast_two_folded(self):
        out = evaluate_shape_rule(
            "z.shape = np.broadcast_shapes(x.shape, y.shape)",
            output_name="z",
            inputs={"x": _sh(["...d"]), "y": _sh(["...d"])},
            attr_values={},
        )
        # 两个同名折叠维 → 直接保留命名（folded_name 沿用）
        assert out.folded_name is not None
        assert out.explicit == []

    def test_broadcast_explicit(self):
        out = evaluate_shape_rule(
            "z.shape = np.broadcast_shapes(x.shape, y.shape)",
            output_name="z",
            inputs={"x": _sh(["M", "K"]), "y": _sh(["M", "K"])},
            attr_values={},
        )
        assert [d.name for d in out.explicit] == ["M", "K"]


class TestMatmulShape:
    def test_matmul_no_transpose(self):
        rule = """
c.shape = (
    np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    + ((a.shape[-1] if transpose_a else a.shape[-2]),)
    + ((b.shape[-2] if transpose_b else b.shape[-1]),)
)
"""
        out = evaluate_shape_rule(
            rule,
            output_name="c",
            inputs={
                "a": _sh(["...batch_a", "M", "K"]),
                "b": _sh(["...batch_b", "K", "N"]),
            },
            attr_values={"transpose_a": False, "transpose_b": False},
        )
        # batch 维广播 + (M, N)
        assert [d.name for d in out.explicit] == ["M", "N"]
        assert out.folded_name is not None

    def test_matmul_with_transpose_a(self):
        rule = """
c.shape = (
    a.shape[:-2]
    + ((a.shape[-1] if transpose_a else a.shape[-2]),)
    + (b.shape[-1],)
)
"""
        out = evaluate_shape_rule(
            rule,
            output_name="c",
            inputs={
                "a": _sh(["K", "M"]),         # transpose_a=True 时实际 K 在 axis -2
                "b": _sh(["K", "N"]),
            },
            attr_values={"transpose_a": True, "transpose_b": False},
        )
        # transpose_a=True → 取 a.shape[-1] (M)
        assert [d.name for d in out.explicit] == ["M", "N"]


class TestUnresolvedSymbol:
    def test_unknown_input(self):
        with pytest.raises(DslError, match="unresolved_symbol|未声明"):
            evaluate_shape_rule(
                "y.shape = z.shape",                # z 未声明
                output_name="y",
                inputs={"x": _sh(["M"])},
                attr_values={},
            )

    def test_unknown_attribute(self):
        with pytest.raises(DslError, match="unresolved_symbol|未声明"):
            evaluate_shape_rule(
                "y.shape = x.shape if some_attr else x.shape",
                output_name="y",
                inputs={"x": _sh(["M"])},
                attr_values={},
            )


class TestSandbox:
    def test_import_blocked(self):
        with pytest.raises(DslError):
            evaluate_shape_rule(
                "y.shape = __import__('os').name",
                output_name="y",
                inputs={"x": _sh(["M"])},
                attr_values={},
            )

    def test_dunder_attr_blocked(self):
        with pytest.raises(DslError):
            evaluate_shape_rule(
                "y.shape = ().__class__",
                output_name="y",
                inputs={"x": _sh(["M"])},
                attr_values={},
            )

    def test_for_loop_blocked(self):
        rule = "for i in range(3):\n    pass\ny.shape = x.shape"
        with pytest.raises(DslError):
            evaluate_shape_rule(
                rule,
                output_name="y",
                inputs={"x": _sh(["M"])},
                attr_values={},
            )


class TestMissingOutput:
    def test_no_assignment(self):
        with pytest.raises(DslError, match="未给.*赋值|unresolved_symbol"):
            evaluate_shape_rule(
                "tmp = x.shape",                     # 没赋给 y.shape
                output_name="y",
                inputs={"x": _sh(["M"])},
                attr_values={},
            )


class TestSlicing:
    def test_negative_index(self):
        # x.shape[-1] 返回最后一维 Dim
        out = evaluate_shape_rule(
            "y.shape = (x.shape[-1],)",
            output_name="y",
            inputs={"x": _sh(["M", "K", "N"])},
            attr_values={},
        )
        assert [d.name for d in out.explicit] == ["N"]

    def test_slice_with_folded(self):
        # x.shape[:-2] 保留折叠维
        out = evaluate_shape_rule(
            "y.shape = x.shape[:-2]",
            output_name="y",
            inputs={"x": _sh(["...batch", "M", "K"])},
            attr_values={},
        )
        # 切片含起点（key.start is None）→ 保留 folded
        assert out.folded_name == "batch"
        assert out.explicit == []
