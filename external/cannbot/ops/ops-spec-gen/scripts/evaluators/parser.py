# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""shape_literal parser — 解析 `inputs[].shape.symbolic` 列表为 SymbolicShape。

历史上这里曾经有一个项目自定义的小 DSL（parse_expr / Ident / IntLit / Call /
ShapeSolver / DtypeSolver），用来解析 `shape_rule: MATMUL_SHAPE(...)` 等表达式。
那套 DSL 已被弃用，shape_rule 与 dtype_rule 现在直接写 numpy 子集表达式，由
shape_eval.py / dtype_eval.py 在受限 AST 沙箱里求值。

本文件只剩 `parse_shape_literal` —— 解析 yaml 的字符串列表（与 DSL 求值不相关）。
"""

from __future__ import annotations

import re

from .types import Dim, SymbolicShape, DslError


_FOLDED_RE = re.compile(r"^\.\.\.(?P<name>[a-z][a-zA-Z0-9_]*)$")
_SYMBOL_RE = re.compile(r"^[A-Z][a-zA-Z0-9_]*$")


def parse_shape_literal(symbolic: list, *, field_path: str = "") -> SymbolicShape:
    """Convert spec.yaml's `inputs[].shape.symbolic` list to a SymbolicShape."""
    if symbolic is None:
        symbolic = []
    if not isinstance(symbolic, list):
        raise DslError(
            code="dsl_parse_error",
            message=f"symbolic 必须是列表，得到 {type(symbolic).__name__}",
            field_path=field_path,
        )

    dims: list[Dim] = []
    for i, e in enumerate(symbolic):
        if isinstance(e, bool):  # bool is subclass of int — guard against True/False
            raise DslError("dsl_parse_error",
                           f"shape 元素不能是 bool: {e!r}", f"{field_path}[{i}]")
        if isinstance(e, int):
            if e < 0:
                raise DslError("dsl_parse_error",
                               f"const 维必须非负，得到 {e}", f"{field_path}[{i}]")
            dims.append(Dim(kind="const", value=e))
        elif isinstance(e, str):
            if m := _FOLDED_RE.match(e):
                dims.append(Dim(kind="folded", name=m.group("name")))
            elif _SYMBOL_RE.match(e):
                dims.append(Dim(kind="symbol", name=e))
            else:
                raise DslError(
                    code="dsl_parse_error",
                    message=(
                        f"shape 元素 {e!r} 不合法：显式维必须 ^[A-Z]…$（大写起始），"
                        f"折叠维必须 '...lower_name'"
                    ),
                    field_path=f"{field_path}[{i}]",
                )
        else:
            raise DslError(
                code="dsl_parse_error",
                message=f"shape 元素类型不支持：{type(e).__name__}",
                field_path=f"{field_path}[{i}]",
            )
    return SymbolicShape.from_dims(dims)
