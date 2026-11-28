# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest

from _checks.allocation_scope import RULE, check, scope_counts
from _checks.cli import run_rules


def source(tmp_path, body, *, alias="pl", args="", claim=""):
    path = tmp_path / "probe.py"
    path.write_text(
        f"import pypto_pro.language as {alias}\n"
        f"@{alias}.jit()\ndef kernel({args}):\n"
        f'    """{claim}"""\n' + body + "\n"
    )
    return path


def test_direct_outside_allocation_has_location_and_cli_integration(tmp_path):
    path = source(tmp_path, "    a = pl.allocate_tile_group([16,16], dtype, space)")
    found = check(path)
    assert len(found) == 1 and found[0].rule == RULE and found[0].lineno == 5
    assert any(f.rule == RULE for f in run_rules([path]))
    assert scope_counts(path) == (1, 0)


def test_removed_apis_are_rejected_inside_and_outside_with(tmp_path):
    path = source(
        tmp_path,
        "    with pl.mem_space():\n"
        "        a = pl.allocate_tile_group([16,16], dtype, space)\n"
        "    b = pl.allocate_tile_group([16,16], dtype, space)",
    )
    assert [f.lineno for f in check(path)] == [5, 6, 7]
    assert scope_counts(path) == (3, 0)


def test_native_alias_and_nested_control_flow(tmp_path):
    path = source(
        tmp_path,
        "    with native.section_cube():\n"
        "        with native.mem_space():\n"
        "            a = native.allocate_tile_group([16,16], dtype, space)\n"
        "        if tail:\n"
        "            b = native.allocate_tile_group([16,16], dtype, space)",
        alias="native",
    )
    assert [f.lineno for f in check(path)] == [6, 7, 9]


@pytest.mark.parametrize("args", ["pl", "*pl", "**pl"])
def test_parameter_shadowing_is_unknown(tmp_path, args):
    path = source(
        tmp_path, "    a = pl.allocate_tile_group([16,16], dtype, space)", args=args
    )
    assert check(path) == [] and scope_counts(path) == (0, 1)


def test_rebound_namespace_and_helper_context_are_unknown(tmp_path):
    path = source(
        tmp_path,
        "    pl = other\n    a = pl.allocate_tile_group([16,16], dtype, space)",
    )
    assert check(path) == [] and scope_counts(path) == (0, 1)
    path = source(
        tmp_path,
        "    with pl.mem_space():\n"
        "        def helper():\n"
        "            a = pl.allocate_tile_group([16,16], dtype, space)",
    )
    assert [f.lineno for f in check(path)] == [5]
    assert scope_counts(path) == (1, 1)


def test_scope_alias_is_not_assumed_native(tmp_path):
    path = source(
        tmp_path,
        "    with unknown_scope():\n"
        "        a = pl.allocate_tile_group([16,16], dtype, space)",
    )
    assert len(check(path)) == 1


def test_negative_probe_waiver_and_stale_claim(tmp_path):
    claim = f"反例：{RULE} —— 验证解析期拒绝，不上板。"
    path = source(
        tmp_path, "    a = pl.allocate_tile_group([16,16], dtype, space)", claim=claim
    )
    assert len(check(path)) == 1 and check(path)[0].waived
    path = source(
        tmp_path,
        '    a = pl.make_tile_group(type=tt, addrs=0, mutex_ids="auto", depth=1)',
        claim=claim,
    )
    assert len(check(path)) == 1 and not check(path)[0].waived
