# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Transfer diagnostics preserve legal layouts, uncertainty and version scope."""

import pytest

from _checks import ub_l1
from _checks.paths import PYPTO_ROOT


def source(src_layout="ND", dst_layout="NZ", *, dst_space="Mat", rows=32, api="move"):
    call = f"pl.{api}(dst, src" + (", [0, 0])" if api == "insert" else ")")
    return f"""import pypto_pro.language as pl
@pl.jit(auto_mutex=False)
def candidate():
    src = pl.make_tile(pl.TileType(shape=[{rows}, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, layout=pl.{src_layout}), addr=0)
    dst = pl.make_tile(pl.TileType(shape=[{rows}, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.{dst_space}, layout=pl.{dst_layout}), addr=16384)
    {call}
"""


def inspect(tmp_path, code, **kwargs):
    path = tmp_path / "candidate.py"
    path.write_text(code)
    return ub_l1.inspect(path, **kwargs)


@pytest.mark.parametrize("api", ["move", "insert"])
@pytest.mark.parametrize("layout", ["NZ", "ZN"])
def test_nd_to_fractal_is_an_address_map_advisory(tmp_path, api, layout):
    found, facts, counts = inspect(tmp_path, source(dst_layout=layout, api=api))
    assert [(f.rule, f.advisory) for f in found] == [(ub_l1.LAYOUT, True)]
    assert facts[0]["source"]["layout"] == "ND"
    assert counts == {"resolved": 1, "unverified": 0}


@pytest.mark.parametrize("rows", [8, 16, 32, 33])
@pytest.mark.parametrize("api", ["move", "insert"])
def test_nz_handover_has_no_universal_odd_pitch_rule(tmp_path, rows, api):
    found, _, _ = inspect(tmp_path, source(src_layout="NZ", rows=rows, api=api))
    assert not found


def test_odd_nz_conversion_is_rejected_only_with_recognized_frontend(tmp_path):
    assert ub_l1.pitch_guard(PYPTO_ROOT)
    found, _, _ = inspect(tmp_path, source(dst_space="Vec", rows=33))
    assert [(f.rule, f.advisory) for f in found] == [(ub_l1.PITCH, False)]
    found, _, _ = inspect(
        tmp_path, source(dst_space="Vec", rows=33), pypto_root=tmp_path / "unavailable"
    )
    assert [(f.rule, f.advisory) for f in found] == [(ub_l1.PITCH, True)]


def test_changed_guard_is_not_assumed_equivalent(tmp_path):
    path = tmp_path / "python/pypto_pro/ir/op/block_ops.py"
    path.parent.mkdir(parents=True)
    original = (PYPTO_ROOT / "python/pypto_pro/ir/op/block_ops.py").read_text()
    path.write_text(original.replace("_nz_dst_shape[0] % 16", "_nz_dst_shape[0] % 32"))
    assert not ub_l1.pitch_guard(tmp_path)


def test_group_slots_and_single_assignment_constants(tmp_path):
    code = source(dst_space="Vec", rows="R + 1").replace("@pl.jit", "R = 32\n@pl.jit")
    code = code.replace("src = pl.make_tile", "src_group = pl.make_tile_group").replace(
        "addr=0", "addrs=0"
    )
    code = code.replace("    pl.move", "    src = src_group.current()\n    pl.move")
    found, facts, _ = inspect(tmp_path, code)
    assert found[0].rule == ub_l1.PITCH
    assert facts[0]["source"]["shape"] == [33, 128]


def test_direct_tuple_bindings_retain_each_tile_descriptor(tmp_path):
    code = source().replace(
        "    pl.move(dst, src)",
        "    alias_src, alias_dst = src, dst\n    pl.insert(alias_dst, alias_src, [0, 0])",
    )
    found, facts, _ = inspect(tmp_path, code)
    assert found[0].rule == ub_l1.LAYOUT
    assert (
        facts[0]["source"]["space"] == "Vec"
        and facts[0]["destination"]["space"] == "Mat"
    )


@pytest.mark.parametrize(
    "replacement", ["src = helper()", "src = unknown", "src = (lambda: None)()"]
)
def test_unknown_operands_stay_unverified(tmp_path, replacement):
    code = source().replace("    pl.move", "    " + replacement + "\n    pl.move")
    found, _, counts = inspect(tmp_path, code)
    assert not found
    assert counts["unverified"] == 1


def test_join_rebinding_does_not_inherit_a_previous_descriptor(tmp_path):
    code = source().replace(
        "    pl.move", "    if flag:\n        src = helper()\n    pl.move"
    )
    found, _, counts = inspect(tmp_path, code)
    assert not found
    assert counts["unverified"] == 1


def test_shadowed_language_namespace_cannot_produce_a_hard_rejection(tmp_path):
    found, _, counts = inspect(
        tmp_path,
        source(dst_space="Vec", rows=33).replace(
            "def candidate():", "def candidate(pl):"
        ),
    )
    assert not found
    assert counts["unverified"] == 1


def test_module_is_not_executed(tmp_path):
    found, _, _ = inspect(tmp_path, source() + "raise RuntimeError('must never run')\n")
    assert found[0].rule == ub_l1.LAYOUT


@pytest.mark.parametrize("rebind", ["import unrelated as pl", "pl.move = object()"])
def test_rebound_import_or_operation_is_unknown(tmp_path, rebind):
    code = source(dst_space="Vec", rows=33).replace(
        "    pl.move", "    " + rebind + "\n    pl.move"
    )
    found, _, counts = inspect(tmp_path, code)
    assert not found
    assert counts["unverified"] == 1


def test_statically_dead_branch_does_not_reject_legal_kernel(tmp_path):
    code = source(dst_space="Vec", rows=33).replace(
        "    pl.move(dst, src)", "    if False:\n        pl.move(dst, src)"
    )
    found, facts, _ = inspect(tmp_path, code)
    assert not found and not facts


def test_unresolved_specialization_branch_requires_native_confirmation(tmp_path):
    code = source(dst_space="Vec", rows=33).replace(
        "    pl.move(dst, src)", "    if MODE:\n        pl.move(dst, src)"
    )
    found, _, _ = inspect(tmp_path, code)
    assert found[0].rule == ub_l1.PITCH and found[0].advisory


def test_unresolved_shape_is_counted_as_unknown(tmp_path):
    found, _, counts = inspect(
        tmp_path,
        source(src_layout="NZ", rows="ROWS").replace(
            "def candidate():", "def candidate(ROWS):"
        ),
    )
    assert not found
    assert counts["unverified"] == 1
