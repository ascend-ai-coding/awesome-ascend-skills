# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Rule `writing-style-declaration`: the first line of an example's `kernel.py` docstring must contain a “写法：”
declaration.

Based on §7.3 of `references/tooling/example-coverage.md`: sugar/raw forms are not separate directory levels; each
example gets one declaration line—“the declaration travels with the example, so moving a directory needs no edits; **a
tool can assert that every example declares it**.” This check implements that statement.

All 21 examples already complied when this check was written, so it pins the current convention rather than fixing an
existing violation. A check that passes from day one still has value: from example 22 onward, the convention does not
depend on memory.
"""

from __future__ import annotations

from pathlib import Path

from .source import Finding, first_docstring_line, read

RULE = "writing-style-declaration"
PREFIX = "写法："
STYLES = ("语法糖", "原始", "混合")
"""Controlled vocabulary. This follows the `TOPOLOGIES` pattern in ascriptor `library/tools/api_examples.py`: a
filterable field needs a finite set of values, or the “writing style” column will have five spellings in two months."""


def check(path: Path) -> list[Finding]:
    source = read(path)
    first = first_docstring_line(source.tree)
    if first is None:
        return [Finding(source.path, 1, RULE, "没有模块 docstring，第一行无处声明写法")]
    if not first.startswith(PREFIX):
        return [
            Finding(
                source.path,
                1,
                RULE,
                f"docstring 第一行不是「{PREFIX}」声明，实际是 {first[:50]!r}",
            )
        ]
    rest = first[slice(len(PREFIX), None)]
    if not any(rest.startswith(style) for style in STYLES):
        return [
            Finding(
                source.path,
                1,
                RULE,
                f"「{PREFIX}」后面要紧跟 {' / '.join(STYLES)} 之一，实际是 {rest[:40]!r}",
            )
        ]
    return []
