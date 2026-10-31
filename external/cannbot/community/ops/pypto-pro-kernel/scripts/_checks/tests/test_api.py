# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Minimal regression coverage for the `api` subcommand's stub signature table."""

from _checks import api


def test_pl_stub_declares_known_names():
    rows = api.declarations(api.STUBS["pl"])
    names = {signature.split("(")[0] for signature, _ in rows}
    assert {"load", "store", "mul", "muls", "make_tile_group"} <= names


def test_vf_methods_drop_self():
    rows = api.declarations(api.STUBS["vf"])
    assert rows, "vf 存根不应为空"
    for signature, _ in rows:
        assert "self" not in signature.split("(", 1)[1].split(",")[0], signature


def test_unknown_namespace_is_refused(capsys):
    assert api.report(["nope"]) is False
    assert "未知命名空间" in capsys.readouterr().out
