# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Criteria for `writing-style-declaration`."""

from pathlib import Path

from _checks import declaration
from _checks.paths import EXAMPLES_ROOT

HEAD = 'r"""写法：原始。一句话说这一例干什么。\n\n正文。\n"""\n'


def _kernel(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "kernel.py"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_compliant_first_line_passes(tmp_path):
    assert declaration.check(_kernel(tmp_path, HEAD)) == []


def test_all_three_styles_are_accepted(tmp_path):
    for style in declaration.STYLES:
        path = _kernel(tmp_path, f'"""写法：{style}。随便一句。"""\n')
        assert declaration.check(path) == [], style


def test_a_missing_module_docstring_is_reported(tmp_path):
    findings = declaration.check(_kernel(tmp_path, "X = 1\n"))
    assert [finding.rule for finding in findings] == [declaration.RULE]
    assert "没有模块 docstring" in findings[0].message


def test_a_first_line_that_is_not_the_declaration_is_reported(tmp_path):
    findings = declaration.check(
        _kernel(tmp_path, '"""四个归约算子。\n\n写法：原始。\n"""\n')
    )
    assert len(findings) == 1
    assert "第一行" in findings[0].message


def test_a_style_outside_the_vocabulary_is_reported(tmp_path):
    """This is why the vocabulary is controlled: `写法：naive` looks compliant, and in two months there could be five
    different spellings.
    """
    findings = declaration.check(_kernel(tmp_path, '"""写法：naive。"""\n'))
    assert len(findings) == 1
    assert all(style in findings[0].message for style in declaration.STYLES)


def test_every_example_in_the_repo_complies(tmp_path):
    """Pin the current convention. All 21 examples complied when this check was written; from example 22 onward,
    compliance no longer depends on memory.
    """
    kernels = sorted((EXAMPLES_ROOT / "basic_api").glob("*/*/kernel.py"))
    assert len(kernels) >= 21
    assert [
        str(finding) for path in kernels for finding in declaration.check(path)
    ] == []
