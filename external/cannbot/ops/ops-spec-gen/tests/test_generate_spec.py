# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

import yaml

from evaluators.stages import stage_3
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
    assert 'case: "rank=0 标量输入 → 归约轴越界"' in spec
    assert "machine_check: {kind: raises_error, error_type: attribute_value_out_of_range}" in spec
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
    spec = spec.replace(
        "    # TODO: 用 numpy 可 eval 的表达式描述算子语义\n    y = x",
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
