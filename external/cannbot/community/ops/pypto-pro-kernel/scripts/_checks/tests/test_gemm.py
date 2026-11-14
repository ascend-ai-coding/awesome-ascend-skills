# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest

from _checks.gemm import RULE, check


def scan(tmp_path, body, *, alias="pl", args="", claim=""):
    path = tmp_path / "probe.py"
    path.write_text(
        f"import pypto_pro.language as {alias}\n"
        f"@{alias}.jit()\ndef kernel({args}):\n"
        f'    """{claim}"""\n' + body + "\n"
    )
    return check(path)


@pytest.mark.parametrize("op", ["gemm", "gemm_mx"])
@pytest.mark.parametrize("compact", ["1", "0", "LAYOUT"])
def test_explicit_layout_keyword_is_rejected_for_either_sugar(tmp_path, op, compact):
    found = scan(tmp_path, f"    pl.{op}(acc,a,b,compact={compact})")
    assert len(found) == 1 and found[0].rule == RULE
    assert found[0].lineno == 5 and "Tile/TileGroup" in found[0].message


def test_valid_layout_declaration_and_extent_call_are_allowed(tmp_path):
    assert not scan(
        tmp_path,
        "    acc = pl.allocate_tile_group([M,N],FP32,space,compact=1)\n"
        "    pl.gemm(acc,a,b,l0a=left,l0b=right,m=rows,n=cols)",
    )


def test_module_alias_and_nested_control_flow_are_checked(tmp_path):
    found = scan(
        tmp_path,
        "    with native.section_cube():\n"
        "        if tail:\n"
        "            native.gemm(acc,a,b,compact=1)",
        alias="native",
    )
    assert len(found) == 1 and found[0].lineno == 7


@pytest.mark.parametrize(
    "body,args",
    [
        ("    pl.gemm(acc,a,b,compact=1)", "pl"),
        ("    pl = other\n    pl.gemm(acc,a,b,compact=1)", ""),
    ],
)
def test_shadowed_namespace_stays_unknown(tmp_path, body, args):
    assert not scan(tmp_path, body, args=args)


def test_unresolved_kwargs_and_helper_context_stay_unknown(tmp_path):
    assert not scan(
        tmp_path,
        "    def helper():\n"
        "        pl.gemm(acc,a,b,compact=1)\n"
        "    pl.gemm(acc,a,b,**options)",
    )


def test_explicit_parse_negative_can_be_waived(tmp_path):
    found = scan(
        tmp_path,
        "    pl.gemm(acc,a,b,compact=1)",
        claim=f"反例：{RULE} —— 验证解析器拒绝，不上板。",
    )
    assert len(found) == 1 and found[0].waived
