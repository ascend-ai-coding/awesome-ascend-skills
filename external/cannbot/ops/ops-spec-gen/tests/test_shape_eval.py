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


class TestReduceShape:
    def test_reduce_folded_last_axis_without_keepdims(self):
        out = evaluate_shape_rule(
            "y.shape = np.reduce_shape(x.shape, axis=dim, keepdims=keep_dims)",
            output_name="y",
            inputs={"x": _sh(["...batch", "R"])},
            attr_values={"dim": -1, "keep_dims": False},
        )
        assert out.folded_name == "batch"
        assert out.explicit == []

    def test_reduce_folded_last_axis_with_keepdims(self):
        out = evaluate_shape_rule(
            "y.shape = np.reduce_shape(x.shape, axis=dim, keepdims=keep_dims)",
            output_name="y",
            inputs={"x": _sh(["...batch", "R"])},
            attr_values={"dim": -1, "keep_dims": True},
        )
        assert str(out) == "[...batch, 1]"

    def test_reduce_explicit_middle_axis_without_keepdims(self):
        out = evaluate_shape_rule(
            "y.shape = np.reduce_shape(x.shape, axis=dim, keepdims=keep_dims)",
            output_name="y",
            inputs={"x": _sh(["N", "R", "C"])},
            attr_values={"dim": 1, "keep_dims": False},
        )
        assert [d.name for d in out.explicit] == ["N", "C"]


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


class TestIntegerShapeExpressions:
    def evaluate(self, rule, shape=None, attrs=None):
        return evaluate_shape_rule(rule, output_name="y",
                                   inputs={"x": _sh(shape or ["M", "N"])},
                                   attr_values=attrs or {})

    @pytest.mark.parametrize("factor", [2, 4, 16, 32])
    def test_split_and_pack_preserve_symbols(self, factor):
        out = self.evaluate(f"y.shape = (x.shape[0], x.shape[1] // {factor}, {factor} * x.shape[1])")
        half, packed = out.explicit[1:]
        assert half.kind == packed.kind == "expr"
        assert half.symbols == packed.symbols == frozenset(["N"])
        for n in [0, 1, factor, factor * 3, factor * 3 + 1, 2048]:
            assert half.resolve({"N": n}) == n // factor
            assert packed.resolve({"N": n}) == factor * n
        assert str(half) != str(packed)

    def test_expressions_are_distinct_and_can_be_composed(self):
        out = self.evaluate("y.shape = (x.shape[1] // 2, x.shape[1] // 4, (x.shape[1] * 3) // 2)")
        assert out.explicit[0] != out.explicit[1]
        assert out.explicit[2].resolve({"N": 7}) == 10
        assert out.explicit[0].value is None

    def test_constant_and_identity_arithmetic(self):
        out = self.evaluate("y.shape = (x.shape[1] // 2, 2 * x.shape[1])", [3, 128])
        assert [d.value for d in out.explicit] == [64, 256]
        out = self.evaluate("y.shape = (x.shape[1] // 1, x.shape[1] * 1, x.shape[1] * 0)")
        assert out.explicit[:2] == [_sh(["N"]).explicit[0]] * 2
        assert out.explicit[2].value == 0

    @pytest.mark.parametrize("operation", ["// 0", "// -1", "// 2.0", "// True", "* -1", "* True"])
    def test_illegal_integer_arithmetic_is_rejected(self, operation):
        with pytest.raises(DslError):
            self.evaluate(f"y.shape = (x.shape[1] {operation},)")

    def test_missing_attributes_do_not_get_invented_defaults(self):
        with pytest.raises(DslError, match="unresolved_symbol"):
            self.evaluate("y.shape = (x.shape[1] // factor,)")
        assert self.evaluate("y.shape = (x.shape[1] // factor,)",
                             attrs={"factor": 4}).explicit[0].resolve({"N": 128}) == 32

    def test_same_expression_can_be_broadcast_without_losing_it(self):
        out = self.evaluate("y.shape = np.broadcast_shapes((x.shape[1] // 2,), (x.shape[1] // 2,))")
        assert out.explicit[0].resolve({"N": 128}) == 64

    def test_folded_dimension_arithmetic_is_not_guessed(self):
        from evaluators.types import Dim
        for op in [lambda d: d // 2, lambda d: d * 2]:
            with pytest.raises(DslError, match="folded_dim_misuse"):
                op(Dim(kind="folded", name="batch"))


class TestConcreteSampleDomain:
    @staticmethod
    def _spec():
        return {
            "inputs": [{"name": "x", "role": "tensor", "rank_range": [2, 2],
                        "shape": {"symbolic": ["M", "N"]}}],
            "shape_constraints": {"symbols": {"M": {"kind": "dim", "range": [2, 8]},
                                              "N": {"kind": "dim", "range": [16, 128]}}},
        }

    @staticmethod
    def _check(spec, shape, kind="matches_oracle", **synth):
        from evaluators.stages import check_concrete_sample_domain
        return check_concrete_sample_domain(spec, {
            "case": "negative-looking name must not bypass normal validation",
            "synthesize": {"shapes": {"x": shape}, **synth},
            "machine_check": {"kind": kind},
        }, "boundary_conditions[0]")

    @pytest.mark.parametrize("shape", [[2, 16], [8, 128], "[4, 32]"])
    def test_inclusive_bounds(self, shape):
        assert not self._check(self._spec(), shape)

    @pytest.mark.parametrize("shape", [[1, 16], [9, 16], [2, 8], [2, 129]])
    def test_outside_range(self, shape):
        findings = self._check(self._spec(), shape)
        assert any(f["rule_id"] == "shape_closure.sample_dimension" for f in findings)
        assert all(f["field_path"] == "boundary_conditions[0].synthesize" for f in findings)

    @pytest.mark.parametrize("shape", [[2], [2, 16, 1]])
    def test_rank(self, shape):
        assert any(f["rule_id"] == "shape_closure.sample_rank"
                   for f in self._check(self._spec(), shape))

    @pytest.mark.parametrize("shape", [[True, 16], [-1, 16], "[M, 16]", [2.0, 16]])
    def test_malformed_explicit_shape(self, shape):
        findings = self._check(self._spec(), shape)
        assert findings[0]["rule_id"] == "shape_closure.sample_shape_unverified"
        assert findings[0]["severity"] == "info"

    def test_explicit_error_case_not_forced_into_positive_domain(self):
        assert not self._check(self._spec(), [-1, 16], "raises_error")

    def test_empty_is_legal_only_when_declared(self):
        spec = self._spec()
        assert self._check(spec, [0, 16], "returns_empty")
        spec["shape_constraints"]["symbols"]["M"]["range"][0] = 0
        assert not self._check(spec, [0, 16], "returns_empty")

    def test_scalar_and_constant_dimensions(self):
        spec = self._spec()
        spec["inputs"][0].update(rank_range=[0, 0], shape={"symbolic": []})
        assert not self._check(spec, [])
        spec["inputs"][0].update(rank_range=[2, 2], shape={"symbolic": [2, "N"]})
        assert self._check(spec, [3, 32])

    def test_folded_prefix_and_suffix(self):
        spec = self._spec()
        spec["inputs"][0].update(rank_range=[1, 8], shape={"symbolic": ["...batch", "N"]})
        assert not self._check(spec, [999, 999, 16])
        assert self._check(spec, [999, 999, 8])
        assert self._check(spec, [])

    def test_no_guessed_missing_shapes_or_text_constraint_interpretation(self):
        from evaluators.stages import check_concrete_sample_domain
        spec = self._spec()
        spec["shape_constraints"]["global_constraints"] = ["M must be a prime number"]
        assert not check_concrete_sample_domain(spec, {"synthesize": {}, "machine_check": {}}, "case")
        assert not self._check(spec, [4, 32])

    def test_layout_variant_and_missing_required_attr(self):
        spec = self._spec()
        spec["attributes"] = [{"name": "layout", "default": "MN"}]
        spec["layout_contract"] = {"variants": [
            {"when": "attr.layout == 'MN'", "tensors": [{"ref": "input.x", "logical_axes": ["M", "N"]}]},
            {"when": "attr.layout in ['NM']", "tensors": [{"ref": "input.x", "logical_axes": ["N", "M"]}]},
        ]}
        assert not self._check(spec, [2, 16])
        assert not self._check(spec, [16, 2], attrs={"layout": "NM"})
        assert self._check(spec, [8, 2], attrs={"layout": "NM"})
        del spec["attributes"][0]["default"]
        findings = self._check(spec, [16, 2])
        assert findings and all(f["severity"] == "info" for f in findings)
        assert any(f["severity"] == "error" for f in self._check(spec, [2]))

    @pytest.mark.parametrize("guard", ["input.x.dtype == 'float32'", "__import__('os').system('false')",
                                       "attr.missing == 1", "True and (attr.missing == 1)"])
    def test_unknown_guards_do_not_select_default_axes(self, guard):
        spec = self._spec()
        spec["layout_contract"] = {"variants": [{"when": guard, "tensors": [
            {"ref": "input.x", "logical_axes": ["N", "M"]}]}]}
        findings = self._check(spec, [16, 2])
        assert findings and all(f["severity"] == "info" for f in findings)

    def test_tensor_list_and_missing_optional_inputs_not_ndarrays(self):
        spec = self._spec()
        spec["inputs"][0]["role"] = "tensor_list"
        assert not self._check(spec, [99, 99])
        spec["inputs"][0].update(role="tensor", optional=True)
        from evaluators.stages import check_concrete_sample_domain
        assert not check_concrete_sample_domain(spec, {"synthesize": {}}, "case")

    def test_large_shape_is_metadata_only(self):
        spec = self._spec()
        spec["shape_constraints"]["symbols"]["M"]["range"][1] = 10**18
        assert not self._check(spec, [10**18, 32])

    def test_rejected_boundary_not_run_as_formula_shape_oracle(self):
        from evaluators.stages import _iter_concrete_shape_cases
        spec = self._spec()
        spec["boundary_conditions"] = [{"synthesize": {"x.shape": "[2, 8]"},
                                        "machine_check": {"kind": "matches_oracle"}}]
        assert list(_iter_concrete_shape_cases(spec)) == []

    def test_rejected_extreme_not_allocated(self, monkeypatch):
        from evaluators import extreme_inputs_check as ext

        def no_allocation(*args):
            raise AssertionError("invalid sample must not allocate")
        monkeypatch.setattr(ext, "_synthesize_inputs", no_allocation)
        checked, findings = ext._run_entry_check({"spec": self._spec()}, {
            "synthesize": {"shapes": {"x": [2, 8]}}, "machine_check": {"kind": "produces_nan"},
        }, 0, 0)
        assert not checked
        assert any(f["rule_id"] == "shape_closure.sample_dimension" for f in findings)

    def test_variable_rank_does_not_invent_axis_mapping(self):
        spec = self._spec()
        spec["inputs"][0]["rank_range"] = [1, 4]
        findings = self._check(spec, [3])
        assert findings and all(f["severity"] == "info" for f in findings)
        assert self._check(spec, [1, 16])[0]["severity"] == "error"
