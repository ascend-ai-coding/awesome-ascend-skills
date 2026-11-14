# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Generation selection must not silently borrow an A5 platform configuration."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "references/tiling/broadcast/fallback/script/broadcast_tiling.py"
spec = importlib.util.spec_from_file_location("broadcast_generation_model", SCRIPT)
model = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = model
spec.loader.exec_module(model)


def params(**overrides):
    values = dict(
        input_shapes=[[4, 1, 8], [1, 3, 1]], output_shape=[4, 3, 8], is_dav3510=False, ub_size=192 * 1024, core_num=8
    )
    values.update(overrides)
    return model.BroadcastTilingInput(**values)


@pytest.mark.parametrize("generation,expected", [(False, "ub_broadcast"), (True, "nddma_broadcast")])
def test_same_shape_routes_by_explicit_generation_and_preserves_identity(generation, expected):
    result = model.compute_broadcast_tiling(params(is_dav3510=generation))
    assert result.branch == expected
    assert result.is_dav3510 is generation
    assert result.to_dict()["is_dav3510"] is generation


@pytest.mark.parametrize("generation", [False, True])
def test_onedim_keeps_generation_and_target_budget(generation):
    request = params(
        input_shapes=[[100000], [1]], output_shape=[100000], is_dav3510=generation, ub_size=32 * 1024, core_num=4
    )
    result = model.compute_broadcast_tiling(request)
    assert result.branch == "onedim"
    assert result.to_dict()["is_dav3510"] is generation
    assert result.ub_former * 2 * result.buffer_num <= request.ub_size
    assert result.core_num <= request.core_num


@pytest.mark.parametrize(
    "field,value",
    [
        ("is_dav3510", None),
        ("is_dav3510", "false"),
        ("ub_size", None),
        ("core_num", None),
        ("ub_size", 0),
        ("core_num", 0),
    ],
)
def test_missing_or_invalid_platform_does_not_produce_a_tiling(field, value):
    with pytest.raises(ValueError, match=field):
        model.compute_broadcast_tiling(params(**{field: value}))


def test_old_shape_only_call_cannot_silently_choose_a5():
    with pytest.raises(ValueError, match="is_dav3510"):
        model.compute_broadcast_tiling(model.BroadcastTilingInput([[4, 1], [1, 8]], [4, 8]))


@pytest.mark.parametrize("shapes,output", [([[4, 1, 8], [1, 3, 1]], [4, 3, 8]), ([[32], [1]], [32])])
def test_forced_nddma_is_rejected_on_a2a3_even_for_degenerate_input(shapes, output):
    with pytest.raises(ValueError, match="NDDMA"):
        model.compute_broadcast_tiling(params(input_shapes=shapes, output_shape=output, force_nddma=True))


def test_a5_dynamic_and_forced_nddma_routes_remain_available():
    ordinary = model.compute_broadcast_tiling(
        params(input_shapes=[[4, 1, 64], [1, 3, 1]], output_shape=[4, 3, 64], is_dav3510=True)
    )
    forced = model.compute_broadcast_tiling(
        params(input_shapes=[[4, 1, 64], [1, 3, 1]], output_shape=[4, 3, 64], is_dav3510=True, force_nddma=True)
    )
    assert ordinary.branch == "dynamic_ub"
    assert forced.branch == "nddma_broadcast"
