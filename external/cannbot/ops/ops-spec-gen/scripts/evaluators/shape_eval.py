# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""shape_eval — 在 SymbolicShape 上执行 numpy 子集表达式，求出 outputs 的形状。

设计动机：
  早期版本 shape_rule 用项目自定义 DSL（MATMUL_SHAPE / same_as / broadcast 函数），
  下游 agent 必须懂 ops-spec-gen 内部宏才能解释。改为 numpy_expr 后，spec.yaml 写：

      shape_rule: |
        c.shape = (
            np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
            + ((a.shape[-1] if transpose_a else a.shape[-2]),)
            + ((b.shape[-2] if transpose_b else b.shape[-1]),)
        )

  任何懂 numpy 的 agent 都能读、能跑（替换 SymbolicShape → 真 ndarray.shape 即可）。

实现方式：
  在受限 AST 沙箱里 exec 表达式，把每个输入 input 暴露为 _ShapeProxy，其 .shape
  返回一个支持切片 / 负索引 / `+` 拼接的 _ShapeTuple。numpy namespace 只暴露
  broadcast_shapes（复用 broadcast.py 的 numpy_broadcast_n）。
"""

from __future__ import annotations

import ast
import types as pytypes
from typing import Any

from . import _ast_sandbox
from ._ast_sandbox import SandboxError
from .types import Dim, SymbolicShape, DslError
from . import broadcast as bcast_mod


_TIMEOUT_S = 5


# ---------- shape proxy / shape tuple --------------------------------------


class _ShapeTuple:
    """支持切片 / 负索引 / `+` 拼接的不可变 dim 序列。包装 list[Dim]。

    a.shape[:-2] / a.shape[-1] / `(d,)` / shape1 + shape2 都通过这层。
    其行为模拟 numpy ndarray.shape（python tuple）+ 一些扩展（_ShapeTuple + tuple_of_dim）。
    """

    __slots__ = ("_dims", "_folded")

    def __init__(self, dims: list[Dim], folded_name: str | None = None):
        self._dims = list(dims)
        self._folded = folded_name

    @classmethod
    def from_symbolic(cls, sh: SymbolicShape) -> "_ShapeTuple":
        return cls(list(sh.explicit), sh.folded_name)

    def to_symbolic(self) -> SymbolicShape:
        return SymbolicShape(folded_name=self._folded, explicit=list(self._dims))

    def __len__(self) -> int:
        # folded 维 rank 未知；len() 仅返回 explicit 显式部分长度
        return len(self._dims)

    def __getitem__(self, key):
        # 切片：返回新的 _ShapeTuple；保留 folded 当切片含起点
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self._dims))
            sliced = self._dims[key]
            # 切片含起点（start == 0 或 None）→ 保留 folded prefix
            keeps_folded = (key.start is None or key.start == 0) and (step >= 0)
            new_folded = self._folded if keeps_folded else None
            return _ShapeTuple(sliced, new_folded)
        # 整数索引：返回单个 Dim
        if isinstance(key, int):
            try:
                return self._dims[key]
            except IndexError:
                raise DslError(
                    "incompatible_dims",
                    f"shape 索引越界: index={key} 但显式 rank={len(self._dims)}",
                )
        raise DslError("dsl_eval_error", f"shape 索引不支持类型: {type(key).__name__}")

    def __add__(self, other):
        # 允许 _ShapeTuple + _ShapeTuple、_ShapeTuple + tuple/list
        if isinstance(other, _ShapeTuple):
            # 后段叠加：folded 来自前段；后段 folded 不能再出现
            if other._folded is not None:
                raise DslError(
                    "folded_dim_misuse",
                    "拼接的右侧含 folded 维（'...x'）；folded 必须在表达式最左侧",
                )
            return _ShapeTuple(self._dims + other._dims, self._folded)
        if isinstance(other, (tuple, list)):
            new_dims = list(self._dims) + [_coerce_to_dim(x) for x in other]
            return _ShapeTuple(new_dims, self._folded)
        return NotImplemented

    def __radd__(self, other):
        # tuple + _ShapeTuple — 罕见，但合法
        if isinstance(other, (tuple, list)):
            new_dims = [_coerce_to_dim(x) for x in other] + list(self._dims)
            return _ShapeTuple(new_dims, self._folded)
        return NotImplemented

    def __iter__(self):
        # 不暴露 folded（迭代是显式部分）；让 *unpack 行为符合 numpy shape 直觉
        return iter(self._dims)

    def __repr__(self) -> str:
        parts = []
        if self._folded is not None:
            parts.append(f"...{self._folded}")
        parts.extend(repr(d) for d in self._dims)
        return f"_ShapeTuple({', '.join(parts)})"


def _coerce_to_dim(x: Any) -> Dim:
    if isinstance(x, Dim):
        return x
    if isinstance(x, bool):
        raise DslError("dsl_eval_error", f"维度不能是 bool: {x!r}")
    if isinstance(x, int):
        if x < 0:
            raise DslError("dsl_eval_error", f"const 维必须非负: {x}")
        return Dim(kind="const", value=x)
    raise DslError("dsl_eval_error", f"无法把 {type(x).__name__} 解释为 Dim")


class _ShapeProxy:
    """spec input 在 numpy_expr 中的代理对象；只暴露 .shape 属性。"""

    __slots__ = ("_name", "_shape")

    def __init__(self, name: str, sh: SymbolicShape):
        self._name = name
        self._shape = _ShapeTuple.from_symbolic(sh)

    @property
    def shape(self) -> _ShapeTuple:
        return self._shape

    def __repr__(self) -> str:
        return f"_ShapeProxy({self._name!r}, shape={self._shape!r})"


# ---------- numpy namespace (subset) ---------------------------------------


def _np_broadcast_shapes(*shapes) -> _ShapeTuple:
    """模拟 numpy.broadcast_shapes：在 SymbolicShape 上做 numpy 风格广播。"""
    if not shapes:
        raise DslError("dsl_eval_error", "np.broadcast_shapes 至少需要 1 个 shape")
    converted: list[SymbolicShape] = []
    for s in shapes:
        if isinstance(s, _ShapeTuple):
            converted.append(s.to_symbolic())
        elif isinstance(s, (tuple, list)):
            converted.append(SymbolicShape(
                folded_name=None,
                explicit=[_coerce_to_dim(x) for x in s],
            ))
        else:
            raise DslError(
                "dsl_eval_error",
                f"np.broadcast_shapes 参数必须是 shape，得到 {type(s).__name__}",
            )
    out_sh = bcast_mod.numpy_broadcast_n(converted)
    return _ShapeTuple.from_symbolic(out_sh)


class _NpNamespace:
    """numpy_expr 中 `np` 标识符背后的极小命名空间。"""
    broadcast_shapes = staticmethod(_np_broadcast_shapes)


_NP_NAMESPACE = _NpNamespace()


# ---------- evaluator main entry -------------------------------------------


def _compile_shape_rule(rule, field_path):
    if not isinstance(rule, str) or not rule.strip():
        raise DslError("dsl_parse_error", "shape_rule 必须是非空字符串", field_path)
    try:
        tree = ast.parse(rule, mode="exec")
    except SyntaxError as e:
        raise DslError(
            "dsl_parse_error",
            f"shape_rule 语法错: {e.msg} (行 {e.lineno})",
            field_path,
        ) from None
    try:
        _ast_sandbox.validate_ast(tree)
    except SandboxError as e:
        raise DslError(_map_sandbox_code(e.code), e.message, field_path) from None
    return compile(tree, "<shape_rule>", "exec")


def _build_shape_eval_globals(output_name, inputs, attr_values):
    extra: dict[str, Any] = {"np": _NP_NAMESPACE}
    for name, sh in inputs.items():
        extra[name] = _ShapeProxy(name, sh)
    extra.update(attr_values)
    output_slot = pytypes.SimpleNamespace(shape=None)
    extra[output_name] = output_slot
    return _ast_sandbox.make_globals(extra), output_slot


def _exec_shape_rule(compiled, g, locals_dict, field_path):
    try:
        with _ast_sandbox.timeout(_TIMEOUT_S, on_timeout_code="shape_eval_timeout"):
            exec(compiled, g, locals_dict)
    except SandboxError as e:
        raise DslError(_map_sandbox_code(e.code), e.message, field_path) from None
    except DslError:
        raise
    except NameError as e:
        raise DslError(
            "unresolved_symbol",
            f"shape_rule 引用了未声明的标识符: {e.args[0] if e.args else str(e)}",
            field_path,
        ) from None
    except (TypeError, AttributeError) as e:
        raise DslError("dsl_eval_error", f"shape_rule 求值失败: {e}", field_path) from None


def evaluate_shape_rule(
    rule: str,
    *,
    output_name: str,
    inputs: dict[str, SymbolicShape],
    attr_values: dict[str, Any],
    field_path: str = "",
) -> SymbolicShape:
    """Run a numpy_expr shape_rule and return the resolved SymbolicShape.

    Raises DslError on parse / sandbox / runtime issues. The caller (stage 3)
    converts these to findings.
    """
    compiled = _compile_shape_rule(rule, field_path)
    g, output_slot = _build_shape_eval_globals(output_name, inputs, attr_values)
    locals_dict: dict[str, Any] = {}
    _exec_shape_rule(compiled, g, locals_dict, field_path)

    if output_slot.shape is not None:
        return _coerce_eval_result(output_slot.shape, output_name, field_path)
    if output_name in locals_dict and not isinstance(locals_dict[output_name], pytypes.SimpleNamespace):
        return _coerce_eval_result(locals_dict[output_name], output_name, field_path)
    raise DslError(
        "unresolved_symbol",
        f"shape_rule 未给 {output_name}.shape 赋值；"
        f"应写 `{output_name}.shape = <expr>`，且输出名要与 outputs[].name 一致",
        field_path,
    )


def _coerce_eval_result(val: Any, output_name: str, field_path: str) -> SymbolicShape:
    if isinstance(val, _ShapeTuple):
        return val.to_symbolic()
    if isinstance(val, SymbolicShape):
        return val
    if isinstance(val, (tuple, list)):
        return SymbolicShape(folded_name=None, explicit=[_coerce_to_dim(x) for x in val])
    raise DslError(
        "dsl_eval_error",
        f"shape_rule 中 {output_name!r} 必须是 shape（_ShapeTuple / tuple / list），"
        f"得到 {type(val).__name__}",
        field_path,
    )


def _map_sandbox_code(code: str) -> str:
    """Map _ast_sandbox 错误码到 shape_closure 域错误码。"""
    if code == "ast_disallowed":
        return "dsl_eval_error"
    if code == "banned_name":
        return "dsl_eval_error"
    if code == "shape_eval_timeout":
        return "dsl_eval_error"
    return "dsl_eval_error"
