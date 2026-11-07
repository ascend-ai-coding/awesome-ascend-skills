# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import pytest

from _checks import tile_shape
from _checks.cli import run_rules
from _checks.paths import EXAMPLES_ROOT


def source(tmp_path, shape, *, alias="pl"):
    path = tmp_path / "probe.py"
    path.write_text(
        f"import pypto_pro.language as {alias}\n"
        f"def factory(n):\n"
        f"    def kernel():\n"
        f"        t = {alias}.TileType(shape={shape})\n"
    )
    return path


@pytest.mark.parametrize("shape", ["[]", "[64]", "(n,)", "[1, 2, n]"])
def test_rejects_known_wrong_rank(tmp_path, shape):
    findings = tile_shape.check(source(tmp_path, shape))
    assert [(f.rule, f.lineno) for f in findings] == [(tile_shape.RULE, 4)]


@pytest.mark.parametrize("shape", ["[1, n]", "(1, 64)", "shape_variable", "[*dims]"])
def test_valid_or_unknown_rank_is_not_rejected(tmp_path, shape):
    assert tile_shape.check(source(tmp_path, shape)) == []


def test_reports_unchecked_dynamic_shapes(tmp_path):
    assert tile_shape.scope_counts(source(tmp_path, "[1, n]")) == (1, 0)
    assert tile_shape.scope_counts(source(tmp_path, "shape_variable")) == (0, 1)
    assert tile_shape.scope_counts(source(tmp_path, "[*dims]")) == (0, 1)


def test_native_alias_and_cli_gate(tmp_path):
    path = source(tmp_path, "[64]", alias="pto")
    assert any(f.rule == tile_shape.RULE for f in run_rules([path]))


def test_unrelated_constructor_is_not_checked(tmp_path):
    path = tmp_path / "other.py"
    path.write_text("import unrelated as pl\nx = pl.TileType(shape=[64])\n")
    assert tile_shape.check(path) == []


def test_positional_shape_and_function_waiver(tmp_path):
    path = tmp_path / "probe.py"
    path.write_text(
        "import pypto_pro.language as pl\n"
        "def kernel():\n"
        '    """反例：tile-shape-must-be-2d —— 解析拒绝的负对照。"""\n'
        "    t = pl.TileType([64])\n"
    )
    findings = tile_shape.check(path)
    assert len(findings) == 1 and findings[0].waived


def test_checked_in_examples_have_no_rank_violation():
    assert [
        f for path in EXAMPLES_ROOT.rglob("*.py") for f in tile_shape.check(path)
    ] == []
