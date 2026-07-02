# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""dtype_eval — 在 dtype 字符串上执行 numpy 子集表达式，求出 outputs 的 dtype。

spec.yaml 里 dtype_rule 形如：

    dtype_rule: |
      c.dtype = np.promote_types(a.dtype, b.dtype)

或：

    dtype_rule: |
      y.dtype = np.int32

或：

    dtype_rule: |
      y.dtype = x.dtype

stage 4 对 `dtype_policy.supported_combinations` 的每一行跑这个 evaluator，
得到的 dtype 与显式表交叉比对。语义替换了旧 DSL（promote / same_as / fixed 等）。
"""

from __future__ import annotations

import ast
import types as pytypes
from typing import Any

from . import _ast_sandbox
from ._ast_sandbox import SandboxError
from .types import DslError
from . import promote as promote_mod


_TIMEOUT_S = 5


# ---------- dtype proxy ----------------------------------------------------


class _DtypeProxy:
    """input 在 dtype_rule 中的代理；.dtype 返回 dtype 字符串。"""

    __slots__ = ("_name", "_dtype")

    def __init__(self, name: str, dtype: str):
        self._name = name
        self._dtype = dtype

    @property
    def dtype(self) -> str:
        return self._dtype

    def __repr__(self) -> str:
        return f"_DtypeProxy({self._name!r}, dtype={self._dtype!r})"


# ---------- np namespace (dtype subset) ------------------------------------


def _np_promote_types(a: Any, b: Any) -> str:
    return promote_mod.promote_pair(_coerce_dtype(a), _coerce_dtype(b))


def _np_result_type(*xs) -> str:
    if not xs:
        raise DslError("dsl_eval_error", "np.result_type 至少 1 个参数")
    return promote_mod.promote_many([_coerce_dtype(x) for x in xs])


def _coerce_dtype(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, _DtypeProxy):
        return x.dtype
    raise DslError("dsl_eval_error", f"不能把 {type(x).__name__} 解释为 dtype")


class _NpDtypeNamespace:
    """numpy_expr 中 `np` 标识符在 dtype 求值场景下的命名空间。

    - np.promote_types / np.result_type → 延迟到 promote 表
    - np.<dtype> → 返回 dtype 字符串字面量

    所有 dtype 常量从 `promote._ALL_DTYPES` 动态注入（见模块末尾），保证
    schema enum / promote 表 / dtype_eval namespace 单一真值；新增 dtype 时
    只改 promote.py + schemas/op-spec.json 即可。
    """
    promote_types = staticmethod(_np_promote_types)
    result_type = staticmethod(_np_result_type)
    bool_ = "bool"   # numpy 习惯名（np.bool_）；np.bool 已 deprecated


for _dt in promote_mod._ALL_DTYPES:
    setattr(_NpDtypeNamespace, _dt, _dt)


_NP_NAMESPACE = _NpDtypeNamespace()


# ---------- main entry -----------------------------------------------------


def _compile_dtype_rule(rule, field_path):
    if not isinstance(rule, str) or not rule.strip():
        raise DslError("dsl_parse_error", "dtype_rule 必须是非空字符串", field_path)
    try:
        tree = ast.parse(rule, mode="exec")
    except SyntaxError as e:
        raise DslError(
            "dsl_parse_error",
            f"dtype_rule 语法错: {e.msg} (行 {e.lineno})",
            field_path,
        ) from None
    try:
        _ast_sandbox.validate_ast(tree)
    except SandboxError as e:
        raise DslError("dsl_eval_error", e.message, field_path) from None
    return compile(tree, "<dtype_rule>", "exec")


def _exec_dtype_rule(compiled, g, locals_dict, field_path):
    try:
        with _ast_sandbox.timeout(_TIMEOUT_S, on_timeout_code="dtype_eval_timeout"):
            exec(compiled, g, locals_dict)
    except SandboxError as e:
        raise DslError("dsl_eval_error", e.message, field_path) from None
    except DslError:
        raise
    except NameError as e:
        raise DslError(
            "unresolved_symbol",
            f"dtype_rule 引用了未声明的标识符: {e.args[0] if e.args else str(e)}",
            field_path,
        ) from None
    except (TypeError, AttributeError) as e:
        raise DslError("dsl_eval_error", f"dtype_rule 求值失败: {e}", field_path) from None


def evaluate_dtype_rule(
    rule: str,
    *,
    output_name: str,
    input_dtypes: dict[str, str],
    field_path: str = "",
) -> str:
    """Run a numpy_expr dtype_rule. Return the resolved dtype string.

    input_dtypes 是当前被校验的 supported_combinations 这一行的 {input_name: dtype_str}。
    """
    compiled = _compile_dtype_rule(rule, field_path)
    extra: dict[str, Any] = {"np": _NP_NAMESPACE}
    for name, dt in input_dtypes.items():
        extra[name] = _DtypeProxy(name, dt)
    output_slot = pytypes.SimpleNamespace(dtype=None)
    extra[output_name] = output_slot
    g = _ast_sandbox.make_globals(extra)
    locals_dict: dict[str, Any] = {}
    _exec_dtype_rule(compiled, g, locals_dict, field_path)

    if output_slot.dtype is not None:
        return _coerce_result_dtype(output_slot.dtype, output_name, field_path)
    if output_name in locals_dict and not isinstance(locals_dict[output_name], pytypes.SimpleNamespace):
        return _coerce_result_dtype(locals_dict[output_name], output_name, field_path)
    raise DslError(
        "unresolved_symbol",
        f"dtype_rule 未给 {output_name}.dtype 赋值；"
        f"应写 `{output_name}.dtype = <expr>`",
        field_path,
    )


def _coerce_result_dtype(val: Any, output_name: str, field_path: str) -> str:
    if isinstance(val, str):
        return val
    if isinstance(val, _DtypeProxy):
        return val.dtype
    raise DslError(
        "dsl_eval_error",
        f"dtype_rule 中 {output_name}.dtype 必须求值为 dtype 字符串，"
        f"得到 {type(val).__name__}",
        field_path,
    )
