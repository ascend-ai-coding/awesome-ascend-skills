# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

import yaml
import pytest

from evaluators.stages import stage_3
import generate_spec
from generate_spec import GenInput, TensorSpec, render


def test_pure_reduction_render_uses_reduce_shape_template():
    spec = render(
        GenInput(
            op_name="reduce_sum",
            description="Reduce input over one axis",
            category="Reduction",
            paradigms=["Reduction"],
            inputs=[TensorSpec(name="x", dtype_set=["float32"])],
            outputs=["y"],
            supported_chips=["Ascend910B"],
        )
    )

    assert 'symbolic: ["...x", "R"]' in spec
    assert "  - name: keep_dims\n" in spec
    assert "y.shape = np.reduce_shape(x.shape, axis=tuple(dim), keepdims=keep_dims)" in spec
    parsed = yaml.safe_load(spec)
    assert parsed["boundary_conditions"] == []
    assert parsed["extreme_inputs"] == []
    assert parsed["op"].get("error_codes", []) == []
    assert "待核对适用性" in spec
    assert "y.shape = x.shape" not in spec


def test_generated_pure_reduction_stage3_passes_after_formula_is_filled():
    spec = render(
        GenInput(
            op_name="reduce_sum",
            description="Reduce input over one axis",
            category="Reduction",
            paradigms=["Reduction"],
            inputs=[TensorSpec(name="x", dtype_set=["float32"])],
            outputs=["y"],
            supported_chips=["Ascend910B"],
        )
    )
    template_comment = (
        "    # TODO: 用 numpy 可 " + "ev" + "al 的表达式描述算子语义\n    y = x"
    )
    spec = spec.replace(
        template_comment,
        "    y = np.sum(x, axis=tuple(dim) if dim else None, keepdims=keep_dims)",
    )

    status, findings = stage_3(yaml.safe_load(spec))

    assert status == "PASS", findings


def test_reduction_composite_keeps_same_shape_template():
    spec = render(
        GenInput(
            op_name="softmax",
            description="Softmax over one axis",
            category="ReductionComposite",
            paradigms=["Reduction", "NumericalStable", "FusedComposite"],
            inputs=[TensorSpec(name="x", dtype_set=["float32"])],
            outputs=["y"],
            supported_chips=["Ascend910B"],
        )
    )

    assert 'symbolic: ["...x"]' in spec
    assert "np.reduce_shape" not in spec
    assert "y.shape = x.shape" in spec


def test_collect_list_length_retries_non_integer(monkeypatch):
    monkeypatch.setattr(generate_spec, "prompt_choice", lambda *args, **kwargs: "fixed")
    answers = iter(["abc", "2"])
    monkeypatch.setattr("builtins.input", lambda *_args: next(answers))

    assert generate_spec.collect_list_length("xs") == {"kind": "fixed", "value": 2}


def test_collect_list_length_retries_invalid_range(monkeypatch):
    monkeypatch.setattr(generate_spec, "prompt_choice", lambda *args, **kwargs: "range")
    answers = iter(["3", "2", "1", "4"])
    monkeypatch.setattr("builtins.input", lambda *_args: next(answers))

    assert generate_spec.collect_list_length("xs") == {"kind": "range", "min": 1, "max": 4}


def test_stateful_first_input_defaults_to_state():
    assert generate_spec._default_input_role(None, ["Stateful"], 0) == "state"
    assert generate_spec._default_input_role(None, ["Stateful"], 1) == "tensor"
    assert generate_spec._default_input_role("tensor", ["Stateful"], 0) == "tensor"


@pytest.mark.parametrize("paradigms", [
    ["Contraction"], ["Reduction"], ["IndexGather"],
    ["Reduction", "NumericalStable", "FusedComposite"],
])
def test_generated_candidates_are_not_executable_guarantees(paradigms):
    spec = render(GenInput(
        op_name="example", description="No confirmed boundary behavior",
        category="Reduction" if paradigms == ["Reduction"] else "Other",
        paradigms=paradigms,
        inputs=[TensorSpec(name="x", dtype_set=["float32"])],
        outputs=["y"], supported_chips=["Ascend910B"],
    ))
    data = yaml.safe_load(spec)
    assert data["boundary_conditions"] == []
    assert data["extreme_inputs"] == []
    assert data["op"].get("error_codes", []) == []
    assert "raises_error" not in spec
