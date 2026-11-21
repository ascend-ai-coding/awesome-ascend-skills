# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check the writing-style declaration on public example kernels.

The caller selects examples from the configured library. Business kernels do not
need this example-catalog convention; native checks still apply to their source.
"""

from __future__ import annotations

from pathlib import Path

from .source import Finding, first_docstring_line, read

RULE = "writing-style-declaration"
PREFIX = "写法："
STYLES = ("语法糖", "原始", "混合")
"""Controlled vocabulary for the filterable writing-style field; use the declared finite set consistently."""


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
