# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest

from _checks.scalar_annotation import RULE, check


def scan(tmp_path, code):
    path = tmp_path / "probe.py"
    path.write_text(
        "import pypto_pro.language as pl\nFP32 = pl.DT_FP32\n" + code + "\n"
    )
    return check(path)


@pytest.mark.parametrize("wrapped", [False, True])
def test_known_alias_at_top_level_or_in_factory_is_reported(tmp_path, wrapped):
    code = "@pl.jit()\ndef kernel(scale: FP32):\n    pass"
    if wrapped:
        code = "def make(cols):\n" + "\n".join(
            "    " + line for line in code.splitlines()
        )
    found = scan(tmp_path, code)
    assert len(found) == 1 and found[0].rule == RULE
    assert "pl.DT_FP32" in found[0].message


def test_tensor_dtype_alias_and_qualified_scalar_are_allowed(tmp_path):
    assert not scan(
        tmp_path,
        "@pl.jit()\ndef kernel(x: pl.Tensor[[1,64],FP32], scale: pl.DT_FP32):\n    pass",
    )


def test_named_tiling_class_is_not_treated_as_scalar_alias(tmp_path):
    assert not scan(
        tmp_path,
        "from dataclasses import dataclass\n@dataclass\nclass Config:\n    rows: int\n"
        "@pl.jit()\ndef kernel(config: Config):\n    pass",
    )


@pytest.mark.parametrize(
    "prefix",
    [
        "FP32 = custom_type\n",
        "def mutate():\n    global FP32\n    FP32 = custom_type\n",
        "def unrelated(FP32):\n    pass\n",
    ],
)
def test_additional_bindings_keep_alias_unknown(tmp_path, prefix):
    assert not scan(tmp_path, prefix + "@pl.jit()\ndef kernel(scale: FP32):\n    pass")


def test_factory_parameter_shadowing_is_unknown(tmp_path):
    assert not scan(
        tmp_path,
        "def make(FP32):\n    @pl.jit()\n    def kernel(scale: FP32):\n        pass",
    )


def test_unknown_names_host_annotations_and_nested_dtype_positions_are_allowed(
    tmp_path,
):
    assert not scan(
        tmp_path,
        "def host(scale: FP32):\n    pass\n"
        "@pl.jit()\ndef kernel(x: pl.Ptr[FP32], config: Config):\n    pass",
    )


def test_explicit_parse_negative_can_be_waived(tmp_path):
    found = scan(
        tmp_path,
        "@pl.jit()\ndef kernel(scale: FP32):\n"
        f'    """反例：{RULE} —— 仅解析验证拒绝，不上板。"""\n    pass',
    )
    assert len(found) == 1 and found[0].waived
