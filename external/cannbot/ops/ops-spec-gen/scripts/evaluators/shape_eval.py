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


def _np_reduce_shape(shape, *, axis=-1, keepdims=False) -> _ShapeTuple:
    """Return the output shape of a numpy-style reduction.

    Folded-prefix shapes have unknown leading rank. For those shapes we can
    safely reduce axes in the explicit suffix via negative axis indexes, which is
    exactly what generated Reduction specs use: ["...x", "R"] with axis=-1.
    """
    if isinstance(shape, _ShapeTuple):
        shape_tuple = shape
    elif isinstance(shape, (tuple, list)):
        shape_tuple = _ShapeTuple([_coerce_to_dim(x) for x in shape])
    else:
        raise DslError("dsl_eval_error", f"np.reduce_shape 参数必须是 shape，得到 {type(shape).__name__}")
    if not isinstance(keepdims, bool):
        raise DslError("dsl_eval_error", f"keepdims 必须是 bool，得到 {type(keepdims).__name__}")

    axes = _normalize_reduce_axes(axis, len(shape_tuple._dims), shape_tuple._folded is not None)
    one = Dim(kind="const", value=1)
    dims: list[Dim] = []
    for index, dim in enumerate(shape_tuple._dims):
        if index in axes:
            if keepdims:
                dims.append(one)
            continue
        dims.append(dim)
    return _ShapeTuple(dims, shape_tuple._folded)


def _normalize_reduce_axes(axis, explicit_rank: int, has_folded_prefix: bool) -> set[int]:
    if axis is None:
        if has_folded_prefix:
            raise DslError(
                "dsl_eval_error",
                "folded rank 未知时不支持 axis=None；请用显式后缀维和负 axis",
            )
        raw_axes = list(range(explicit_rank))
    elif isinstance(axis, bool):
        raise DslError("dsl_eval_error", f"axis 必须是 int/list/tuple/None，得到 bool {axis!r}")
    elif isinstance(axis, int):
        raw_axes = [axis]
    elif isinstance(axis, (tuple, list)):
        raw_axes = list(axis)
    else:
        raise DslError("dsl_eval_error", f"axis 必须是 int/list/tuple/None，得到 {type(axis).__name__}")

    normalized: set[int] = set()
    for raw_axis in raw_axes:
        if isinstance(raw_axis, bool) or not isinstance(raw_axis, int):
            raise DslError("dsl_eval_error", f"axis 必须是 int，得到 {raw_axis!r}")
        if has_folded_prefix:
            if raw_axis >= 0:
                raise DslError(
                    "dsl_eval_error",
                    "folded rank 未知时只支持指向显式后缀维的负 axis",
                )
            axis_index = explicit_rank + raw_axis
        else:
            axis_index = raw_axis + explicit_rank if raw_axis < 0 else raw_axis
        if axis_index < 0 or axis_index >= explicit_rank:
            raise DslError(
                "dsl_eval_error",
                f"reduce axis 越界: axis={raw_axis}，显式 rank={explicit_rank}",
            )
        if axis_index in normalized:
            raise DslError("dsl_eval_error", f"reduce axis 重复: axis={raw_axis}")
        normalized.add(axis_index)
    return normalized


class _NpNamespace:
    """numpy_expr 中 `np` 标识符背后的极小命名空间。"""
    broadcast_shapes = staticmethod(_np_broadcast_shapes)
    reduce_shape = staticmethod(_np_reduce_shape)


_NP_NAMESPACE = _NpNamespace()


# ---------- evaluator main entry -------------------------------------------


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

    rule 形如：
        c.shape = np.broadcast_shapes(a.shape, b.shape)
    或多行：
        c.shape = (
            np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
            + (a.shape[-2],)
            + (b.shape[-1],)
        )
    output_name（取自 outputs[].name）预先以 SimpleNamespace 注入 globals，
    使 `c.shape = ...` 语义可成立。
    """
    if not isinstance(rule, str) or not rule.strip():
        raise DslError("dsl_parse_error", "shape_rule 必须是非空字符串", field_path)

    # AST parse + sandbox validate
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
    compiled = compile(tree, "<shape_rule>", "exec")

    # Build globals: np namespace + each input as _ShapeProxy + attribute defaults
    # + 预创建一个 SimpleNamespace 接收 output_name.shape = ... 赋值
    extra: dict[str, Any] = {"np": _NP_NAMESPACE}
    for name, sh in inputs.items():
        extra[name] = _ShapeProxy(name, sh)
    extra.update(attr_values)
    output_slot = pytypes.SimpleNamespace(shape=None)
    extra[output_name] = output_slot
    g = _ast_sandbox.make_globals(extra)
    locals_dict: dict[str, Any] = {}

    try:
        with _ast_sandbox.timeout(_TIMEOUT_S, on_timeout_code="shape_eval_timeout"):
            exec(compiled, g, locals_dict)
    except SandboxError as e:
        raise DslError(_map_sandbox_code(e.code), e.message, field_path) from None
    except DslError:
        raise
    except NameError as e:
        # 引用了 spec.inputs 之外的标识符 / 未声明 attribute
        raise DslError(
            "unresolved_symbol",
            f"shape_rule 引用了未声明的标识符: {e.args[0] if e.args else str(e)}",
            field_path,
        ) from None
    except (TypeError, AttributeError) as e:
        raise DslError("dsl_eval_error", f"shape_rule 求值失败: {e}", field_path) from None

    # 取出 c.shape；也允许作者把表达式结果直接赋给 output_name（不带 .shape）
    if output_slot.shape is not None:
        return _coerce_eval_result(output_slot.shape, output_name, field_path)
    # fallback：作者写 `c = ...` 而不是 `c.shape = ...`
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
