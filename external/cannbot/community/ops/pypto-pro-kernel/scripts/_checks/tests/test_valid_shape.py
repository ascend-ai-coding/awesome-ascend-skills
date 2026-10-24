# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest

from _checks import valid_shape
from _checks.cli import run_rules
from _checks.paths import EXAMPLES_ROOT


def probe(tmp_path, body, alias="pl"):
    path = tmp_path / "probe.py"
    path.write_text(
        f"import pypto_pro.language as {alias}\n@{alias}.jit()\ndef kernel(n, incoming):\n"
        + "\n".join("    " + line for line in body.splitlines())
        + "\n"
    )
    return path


@pytest.mark.parametrize("shape", ["[1, 65]", "[1, n]", "[-1, 32]", "[0, -1]"])
@pytest.mark.parametrize(
    "slot", ["group.next()", "group.current()", "group[0]", "group"]
)
def test_fixed_shapes_are_rejected_through_group_aliases(tmp_path, shape, slot):
    path = probe(
        tmp_path,
        f"kind = pl.TileType(shape=[1, 128], valid_shape={shape})\n"
        "group = pl.make_tile_group(type=kind, addrs=[0], mutex_ids=[0])\n"
        f"tile = {slot}\npl.set_validshape(tile, [1, n])",
    )
    findings = valid_shape.check(path)
    assert len(findings) == 1 and findings[0].lineno == 7


@pytest.mark.parametrize(
    "option", ["", ", valid_shape=[-1, -1]", ", valid_shape=unknown"]
)
def test_dynamic_default_and_unknown_shapes_are_not_rejected(tmp_path, option):
    path = probe(
        tmp_path,
        f"tile = pl.make_tile(pl.TileType(shape=[1, 64]{option}), addr=0)\n"
        "pl.set_validshape(tile, [1, n])",
    )
    assert valid_shape.check(path) == []


def test_constructor_inside_loop_is_checked(tmp_path):
    path = probe(
        tmp_path,
        "kind = pl.TileType(shape=[1, 64], valid_shape=[1, 64])\n"
        "group = pl.make_tile_group(kind, addrs=[0], mutex_ids=[0])\n"
        "for row in pl.range(n):\n"
        "    tile = group.next()\n"
        "    pl.set_validshape(tile, [1, 64])",
    )
    assert len(valid_shape.check(path)) == 1


@pytest.mark.parametrize(
    "binding",
    [
        "tile = incoming",
        "if n:\n    tile = incoming",
        "for i in pl.range(n):\n    tile = incoming",
    ],
)
def test_uncertain_rebindings_do_not_reuse_old_type_facts(tmp_path, binding):
    path = probe(
        tmp_path,
        "tile = pl.make_tile(pl.TileType(shape=[1, 64], valid_shape=[1, 64]), addr=0)\n"
        + binding
        + "\npl.set_validshape(tile, [1, n])",
    )
    assert valid_shape.check(path) == []


def test_alias_cli_and_waiver(tmp_path):
    path = probe(
        tmp_path,
        '"""反例：fixed-validshape-mutation —— SDK 拒绝的编译负例。"""\n'
        "tile = pto.make_tile(pto.TileType(shape=[1,64], valid_shape=[1,64]), addr=0)\n"
        "pto.set_validshape(tile, [1,64])",
        alias="pto",
    )
    findings = [
        finding for finding in run_rules([path]) if finding.rule == valid_shape.RULE
    ]
    assert len(findings) == 1 and findings[0].waived


def test_parameters_and_helper_returns_are_unknown(tmp_path):
    path = probe(
        tmp_path,
        "tile = helper()\npl.set_validshape(tile, [1,n])\n"
        "pl.set_validshape(incoming, [1,n])",
    )
    assert valid_shape.check(path) == []


def test_checked_in_examples_have_no_fixed_shape_setter():
    assert [
        finding
        for path in EXAMPLES_ROOT.rglob("*.py")
        for finding in valid_shape.check(path)
    ] == []
