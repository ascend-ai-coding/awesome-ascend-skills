# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Stage 3 / 4 / 5 — semantic validation driven by numpy_expr evaluators.

Each stage_* function takes (spec_dict) and returns a tuple (status, findings)
where findings is a list of dicts ready for the parent's Finding dataclass.

Stage 3 — shape_closure：按 outputs[].shape_rule_kind 分流：
  * numpy_expr     → 通过 shape_eval.evaluate_shape_rule 求出输出 SymbolicShape
  * data_dependent → 跳过求解；校验 data_dependent_shape: true + shape_bounds.max_elements
  * 缺失           → 报 shape_rule_kind_missing ERROR

Stage 4 — dtype_closure：通过 dtype_eval 在 supported_combinations 各行上求值。

Stage 5 — broadcast_legality：data_dependent 的 outputs 不参与广播闭合验证。
"""

from __future__ import annotations

import ast
import math
import re
import types as pytypes
from typing import Any
from typing import NamedTuple

from . import _ast_sandbox
from ._ast_sandbox import SandboxError
from .types import SymbolicShape, DslError
from .parser import parse_shape_literal
from . import shape_eval, dtype_eval, formula_eval
from . import broadcast as bcast_mod


# ---------- helpers --------------------------------------------------------


def _input_shapes(spec: dict) -> dict[str, SymbolicShape]:
    out = {}
    for inp in spec.get("inputs", []) or []:
        name = inp.get("name")
        sym = (inp.get("shape") or {}).get("symbolic", [])
        try:
            out[name] = parse_shape_literal(sym, field_path=f"inputs[{name}].shape.symbolic")
        except DslError as e:
            out[name] = e  # type: ignore[assignment]
    return out


def _attr_defaults(spec: dict) -> dict[str, Any]:
    return {a.get("name"): a.get("default") for a in (spec.get("attributes") or [])}


def _registered_symbols(spec: dict) -> set[str]:
    return set((spec.get("shape_constraints") or {}).get("symbols", {}) or {})


# ---------- concrete shape consistency helpers ------------------------------


_SHAPE_RULE_PLACEHOLDER_RE = re.compile(
    r"\b(TODO|FIXME|TBD|placeholder)\b|占位|待补|临时|简化",
    re.IGNORECASE,
)

_SKIP_MACHINE_CHECK_KINDS = {"raises_error", "returns_empty"}
_MAX_CONCRETE_INPUT_ELEMENTS = 200_000


class _ShapeConsistencyResult(NamedTuple):
    findings: list[dict]
    covered_outputs: set[str]
    failing_outputs: set[str]


class _ConcreteShapeProxy:
    """Input proxy for concrete shape_rule execution; only exposes .shape."""

    __slots__ = ("_name", "_shape")

    def __init__(self, name: str, shape: tuple[int, ...]):
        self._name = name
        self._shape = tuple(shape)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def __repr__(self) -> str:
        return f"_ConcreteShapeProxy({self._name!r}, shape={self._shape!r})"


class _ConcreteNpNamespace:
    """Small numpy namespace for concrete shape_rule execution."""

    __slots__ = ("_np",)

    def __init__(self, np_mod):
        self._np = np_mod

    def broadcast_shapes(self, *shapes) -> tuple[int, ...]:
        return _coerce_concrete_shape(self._np.broadcast_shapes(*shapes))

    def reduce_shape(self, shape, *, axis=-1, keepdims=False) -> tuple[int, ...]:
        shape = _coerce_concrete_shape(shape)
        axes = _normalize_concrete_reduce_axes(axis, len(shape))
        dims: list[int] = []
        for index, dim in enumerate(shape):
            if index in axes:
                if keepdims:
                    dims.append(1)
                continue
            dims.append(dim)
        return tuple(dims)


def _check_numpy_expr_shape_consistency(spec: dict) -> _ShapeConsistencyResult:
    """Compare numpy_expr shape_rule with formula output shapes on valid boundary cases.

    This catches cases where shape semantics are placed in notes or formula only,
    while outputs[].shape_rule remains simplified. The comparison is
    concrete and case-based, so it also accepts valid shape rules whose arithmetic is
    richer than the current SymbolicShape evaluator can represent.
    """
    outputs = _numpy_expr_outputs(spec)
    if not outputs:
        return _ShapeConsistencyResult([], set(), set())

    formula = _shape_consistency_formula(spec)
    if formula is None:
        return _ShapeConsistencyResult([], set(), set())

    findings: list[dict] = []
    cases = list(_iter_concrete_shape_cases(spec, findings))
    if not cases:
        return _ShapeConsistencyResult(findings, set(), set())

    try:
        import numpy as np
    except ImportError:
        return _ShapeConsistencyResult(findings, set(), set())

    compiled_formula = _compile_shape_consistency_formula(formula)
    if compiled_formula is None:
        return _ShapeConsistencyResult(findings, set(), set())

    covered_outputs: set[str] = set()
    failing_outputs: set[str] = set()
    output_names = {out.get("name") for _, out in outputs}

    for case in cases:
        input_tensors = _build_concrete_input_tensors(spec, case, np)
        if input_tensors is None:
            continue
        attrs = _attr_defaults(spec)
        attrs.update(case["attrs"])

        formula_locals, formula_error = _execute_shape_consistency_formula(
            compiled_formula,
            input_tensors,
            attrs,
            np,
            case,
        )
        if formula_error:
            failing_outputs.update(output_names)
            findings.append(formula_error)
            continue

        for out_index, out in outputs:
            covered, failed, output_findings = _compare_shape_rule_with_formula_output(
                out_index,
                out,
                formula_locals,
                case,
                attrs,
                np,
            )
            if covered:
                covered_outputs.add(out.get("name"))
            if failed:
                failing_outputs.add(out.get("name"))
            findings.extend(output_findings)

    return _ShapeConsistencyResult(findings, covered_outputs, failing_outputs)


def _numpy_expr_outputs(spec: dict) -> list[tuple[int, dict]]:
    return [
        (i, out) for i, out in enumerate(spec.get("outputs") or [])
        if out.get("shape_rule_kind") == "numpy_expr"
    ]


def _shape_consistency_formula(spec: dict) -> str | None:
    math_spec = spec.get("math_semantics") or {}
    if math_spec.get("formula_kind") != "numpy_expr":
        return None
    formula = math_spec.get("formula") or ""
    if not isinstance(formula, str) or not formula.strip():
        return None
    return formula


def _compile_shape_consistency_formula(formula: str):
    try:
        formula_tree = ast.parse(formula, mode="exec")
        _validate_formula_ast_for_shape_consistency(formula_tree)
        return compile(formula_tree, "<shape_consistency_formula>", "exec")
    except (SyntaxError, formula_eval.FormulaError):
        # Stage 8 owns formula syntax/runtime smoke diagnostics. Stage 3 keeps its
        # focus on shape_rule unless the formula can actually serve as an oracle.
        return None


def _execute_shape_consistency_formula(
    compiled_formula,
    input_tensors: dict[str, Any],
    attrs: dict[str, Any],
    np_mod,
    case: dict,
) -> tuple[dict[str, Any], dict | None]:
    formula_locals: dict[str, Any] = {}
    try:
        with _ast_sandbox.timeout(5, on_timeout_code="shape_consistency_timeout"):
            g = _ast_sandbox.make_globals({"np": np_mod, "math": math})
            g.update(input_tensors)
            g.update(attrs)
            exec(compiled_formula, g, formula_locals)
    except Exception as e:
        return {}, {
            "severity": "error",
            "rule_id": "shape_closure.numpy_expr_formula_eval_error",
            "field_path": "math_semantics.formula",
            "message": (
                f"边界样例 #{case['index']} {case['case']!r} 下 formula 不能作为 shape oracle："
                f"{type(e).__name__}: {str(e)[:200]}"
            ),
            "suggested_fix": "修正 formula，或把该边界样例标成 raises_error / returns_empty",
        }
    return formula_locals, None


def _compare_shape_rule_with_formula_output(
    out_index: int,
    out: dict,
    formula_locals: dict[str, Any],
    case: dict,
    attrs: dict[str, Any],
    np_mod,
) -> tuple[bool, bool, list[dict]]:
    name = out.get("name")
    if name not in formula_locals:
        return False, False, []

    try:
        formula_shape = _shape_of_formula_output(formula_locals[name], np_mod)
        rule_shape = _evaluate_shape_rule_concrete(
            out.get("shape_rule", ""),
            output_name=name,
            input_shapes=case["shapes"],
            attr_values=attrs,
            np_mod=np_mod,
            field_path=f"outputs[{out_index}].shape_rule",
        )
    except DslError as e:
        return True, True, [{
            "severity": "error",
            "rule_id": f"shape_closure.{e.code}",
            "field_path": e.field_path or f"outputs[{out_index}].shape_rule",
            "message": (
                f"边界样例 #{case['index']} {case['case']!r} 下 "
                f"shape_rule concrete 求值失败：{e.message}"
            ),
            "suggested_fix": _shape_fix_hint(e.code),
        }]
    except Exception as e:
        return True, True, [{
            "severity": "error",
            "rule_id": "shape_closure.numpy_expr_formula_eval_error",
            "field_path": "math_semantics.formula",
            "message": (
                f"边界样例 #{case['index']} {case['case']!r} 下 "
                f"formula 输出 {name!r} 无法转成 shape："
                f"{type(e).__name__}: {str(e)[:200]}"
            ),
            "suggested_fix": "确认 formula 输出为 ndarray 或 numpy 可转换标量",
        }]

    if rule_shape == formula_shape:
        return True, False, []
    return True, True, [{
        "severity": "error",
        "rule_id": "shape_closure.numpy_expr_shape_mismatch",
        "field_path": f"outputs[{out_index}].shape_rule",
        "message": (
            f"边界样例 #{case['index']} {case['case']!r} 下，"
            f"shape_rule 推导 {name}.shape={rule_shape}，"
            f"但 math_semantics.formula 产出 shape={formula_shape}"
        ),
        "suggested_fix": (
            "把遗漏的 input shape / attribute 分支写进 shape_rule；"
            "若输出真实依赖 input value，则改为 data_dependent 并使用 VariableOutput 范式"
        ),
    }]


def _validate_formula_ast_for_shape_consistency(tree: ast.AST) -> None:
    try:
        _ast_sandbox.validate_ast(tree)
    except SandboxError as e:
        if e.code == "ast_disallowed":
            raise formula_eval.FormulaError("formula_ast_disallowed", e.message) from None
        if e.code == "banned_name":
            raise formula_eval.FormulaError("formula_banned_name", e.message) from None
        raise formula_eval.FormulaError(f"formula_{e.code}", e.message) from None


def _iter_concrete_shape_cases(spec: dict, findings: list[dict] | None = None):
    attr_names = {a.get("name") for a in (spec.get("attributes") or [])}
    input_names = {inp.get("name") for inp in (spec.get("inputs") or [])}
    for index, case in enumerate(spec.get("boundary_conditions") or []):
        machine_check = case.get("machine_check") or {}
        if machine_check.get("kind") in _SKIP_MACHINE_CHECK_KINDS:
            continue
        synth = case.get("synthesize") or {}
        try:
            shapes, attrs = _extract_synthesize_shapes_attrs(synth, attr_names)
        except DslError as e:
            if findings is not None:
                findings.append({
                    "severity": "warning",
                    "rule_id": "shape_closure.synthesize_parse_error",
                    "field_path": f"boundary_conditions[{index}].synthesize",
                    "message": (
                        f"边界样例 #{index} {case.get('case') or ''!r} 的 synthesize "
                        f"不能用于 shape oracle：{e.message}"
                    ),
                    "suggested_fix": "修正 synthesize 中的 input shape / attr 字面量，或标记为 raises_error / returns_empty",
                })
            continue
        if not input_names <= set(shapes):
            continue
        if any(_num_elements(shape) > _MAX_CONCRETE_INPUT_ELEMENTS
               for shape in shapes.values()):
            continue
        yield {
            "index": index,
            "case": case.get("case") or "",
            "shapes": shapes,
            "attrs": attrs,
        }


def _extract_synthesize_shapes_attrs(
    synth: dict,
    attr_names: set[str],
) -> tuple[dict[str, tuple[int, ...]], dict[str, Any]]:
    shapes: dict[str, tuple[int, ...]] = {}
    attrs: dict[str, Any] = {}
    if not isinstance(synth, dict):
        return shapes, attrs

    nested_shapes = synth.get("shapes") or {}
    if isinstance(nested_shapes, dict):
        for name, value in nested_shapes.items():
            shapes[str(name)] = _parse_concrete_shape(value)

    nested_attrs = synth.get("attrs") or synth.get("attr") or {}
    if isinstance(nested_attrs, dict):
        for name, value in nested_attrs.items():
            if name in attr_names:
                attrs[str(name)] = value

    for key, value in synth.items():
        if not isinstance(key, str):
            continue
        if key.endswith(".shape"):
            shapes[key[:-6]] = _parse_concrete_shape(value)
            continue
        if key.startswith("attr."):
            name = key[5:]
            if name in attr_names:
                attrs[name] = value
            continue
        if key in attr_names:
            attrs[key] = value
    return shapes, attrs


def _parse_concrete_shape(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as e:
            raise DslError("dsl_eval_error", f"shape literal 解析失败: {value!r}") from e
    if isinstance(value, int) and not isinstance(value, bool):
        value = [value]
    if not isinstance(value, (tuple, list)):
        raise DslError("dsl_eval_error", f"shape literal 必须是 list/tuple，得到 {value!r}")
    dims: list[int] = []
    for dim in value:
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise DslError("dsl_eval_error", f"shape 维度必须是非负整数，得到 {dim!r}")
        if dim < 0:
            raise DslError("dsl_eval_error", f"shape 维度必须非负，得到 {dim}")
        dims.append(int(dim))
    return tuple(dims)


def _num_elements(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def _build_concrete_input_tensors(spec: dict, case: dict, np_mod):
    combos = (spec.get("dtype_policy") or {}).get("supported_combinations") or []
    in_dtypes = combos[0].get("inputs", {}) if combos else {}
    seed = (spec.get("test_matrix") or {}).get("random", {}).get("seed", 42)
    tensors: dict[str, Any] = {}
    for inp in spec.get("inputs") or []:
        name = inp.get("name")
        shape = case["shapes"].get(name)
        if shape is None:
            return None
        dtype = in_dtypes.get(name) or (inp.get("dtype_set") or ["float32"])[0]
        try:
            tensors[name] = formula_eval._gen_tensor(np_mod, shape, dtype, seed)
        except formula_eval.FormulaError:
            return None
    return tensors


def _shape_of_formula_output(value: Any, np_mod) -> tuple[int, ...]:
    if not isinstance(value, np_mod.ndarray):
        value = np_mod.asarray(value)
    return tuple(int(d) for d in value.shape)


def _evaluate_shape_rule_concrete(
    rule: str,
    *,
    output_name: str,
    input_shapes: dict[str, tuple[int, ...]],
    attr_values: dict[str, Any],
    np_mod,
    field_path: str,
) -> tuple[int, ...]:
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
        raise DslError(shape_eval._map_sandbox_code(e.code), e.message, field_path) from None

    extra: dict[str, Any] = {"np": _ConcreteNpNamespace(np_mod)}
    for name, shape in input_shapes.items():
        extra[name] = _ConcreteShapeProxy(name, shape)
    extra.update(attr_values)
    output_slot = pytypes.SimpleNamespace(shape=None)
    extra[output_name] = output_slot

    locals_dict: dict[str, Any] = {}
    try:
        with _ast_sandbox.timeout(5, on_timeout_code="shape_eval_timeout"):
            exec(compile(tree, "<shape_rule_concrete>", "exec"),
                 _ast_sandbox.make_globals(extra), locals_dict)
    except SandboxError as e:
        raise DslError(shape_eval._map_sandbox_code(e.code), e.message, field_path) from None
    except DslError:
        raise
    except NameError as e:
        raise DslError(
            "unresolved_symbol",
            f"shape_rule 引用了未声明的标识符: {e.args[0] if e.args else str(e)}",
            field_path,
        ) from None
    except Exception as e:
        raise DslError("dsl_eval_error", f"shape_rule 求值失败: {e}", field_path) from None

    if output_slot.shape is not None:
        return _coerce_concrete_shape(output_slot.shape)
    if output_name in locals_dict and not isinstance(locals_dict[output_name], pytypes.SimpleNamespace):
        return _coerce_concrete_shape(locals_dict[output_name])
    raise DslError(
        "unresolved_symbol",
        f"shape_rule 未给 {output_name}.shape 赋值；应写 `{output_name}.shape = <expr>`",
        field_path,
    )


def _coerce_concrete_shape(value: Any) -> tuple[int, ...]:
    if isinstance(value, int) and not isinstance(value, bool):
        value = (value,)
    if not isinstance(value, (tuple, list)):
        raise DslError("dsl_eval_error", f"shape 必须是 tuple/list，得到 {type(value).__name__}")
    dims: list[int] = []
    for dim in value:
        if isinstance(dim, bool):
            raise DslError("dsl_eval_error", f"维度不能是 bool: {dim!r}")
        if not isinstance(dim, int):
            try:
                dim = int(dim)
            except Exception as e:
                raise DslError(
                    "dsl_eval_error",
                    f"无法把 {type(dim).__name__} 解释为维度",
                ) from e
        if dim < 0:
            raise DslError("dsl_eval_error", f"const 维必须非负: {dim}")
        dims.append(int(dim))
    return tuple(dims)


def _normalize_concrete_reduce_axes(axis: Any, rank: int) -> set[int]:
    if axis is None:
        raw_axes = list(range(rank))
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
        axis_index = raw_axis + rank if raw_axis < 0 else raw_axis
        if axis_index < 0 or axis_index >= rank:
            raise DslError("dsl_eval_error", f"reduce axis 越界: axis={raw_axis}，rank={rank}")
        if axis_index in normalized:
            raise DslError("dsl_eval_error", f"reduce axis 重复: axis={raw_axis}")
        normalized.add(axis_index)
    return normalized


def _shape_rule_placeholder_finding(rule: str, field_path: str) -> dict | None:
    if isinstance(rule, str) and _SHAPE_RULE_PLACEHOLDER_RE.search(rule):
        return {
            "severity": "error",
            "rule_id": "shape_closure.shape_rule_placeholder",
            "field_path": field_path,
            "message": (
                "shape_rule 不能含 TODO/占位/简化等占位说明；确定性 shape 必须写成可执行规则"
            ),
            "suggested_fix": "删除占位说明，并把完整 input shape / attribute 依赖写入 shape_rule",
        }
    return None


def _pure_reduction_identity_finding(spec: dict, rule: str, output_name: str, field_path: str) -> dict | None:
    op = spec.get("op") or {}
    paradigms = set(op.get("paradigms") or [])
    if op.get("category") != "Reduction" or "Reduction" not in paradigms:
        return None
    input_names = {inp.get("name") for inp in (spec.get("inputs") or []) if inp.get("name")}
    identity_input = _direct_shape_identity_input(rule, output_name, input_names)
    if identity_input is None:
        return None
    return {
        "severity": "error",
        "rule_id": "shape_closure.reduction_shape_identity_suspicious",
        "field_path": field_path,
        "message": (
            "纯 Reduction 的输出 shape_rule 不能直接写成 input.shape；"
            "一般需要按 dim/keep_dims 移除或保留归约轴"
        ),
        "suggested_fix": (
            f"改为 {output_name}.shape = "
            f"np.reduce_shape({identity_input}.shape, axis=dim, keepdims=keep_dims)"
        ),
    }


def _direct_shape_identity_input(rule: str, output_name: str, input_names: set[str]) -> str | None:
    if not isinstance(rule, str) or not rule.strip():
        return None
    try:
        tree = ast.parse(rule, mode="exec")
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        return None
    assign = tree.body[0]
    if len(assign.targets) != 1:
        return None
    target = assign.targets[0]
    value = assign.value
    if (
        isinstance(target, ast.Attribute)
        and target.attr == "shape"
        and isinstance(target.value, ast.Name)
        and target.value.id == output_name
        and isinstance(value, ast.Attribute)
        and value.attr == "shape"
        and isinstance(value.value, ast.Name)
        and value.value.id in input_names
    ):
        return value.value.id
    return None


def _can_suppress_symbolic_shape_error(
    code: str,
    output_name: str,
    consistency: _ShapeConsistencyResult,
) -> bool:
    # Current SymbolicShape cannot represent general Dim arithmetic (e.g. min(N, M-k)).
    # Concrete boundary conformance is allowed to prove those richer numpy_expr rules.
    return (
        code in {"dsl_eval_error", "incompatible_dims"}
        and output_name in consistency.covered_outputs
        and output_name not in consistency.failing_outputs
    )


# ---------- stage 3 — shape_closure ----------------------------------------


# 错误码语义区分（保留作为外部 / IDE 集成的稳定 API）：
#   * shape_closure.unresolved_symbol  — shape_rule 引用了 spec.inputs 中不存在的名字（FAIL）
#   * shape_closure.unregistered_symbol — 显式维 symbol 出现在 input.shape 里但
#                                          shape_constraints.symbols 没登记（WARN）
#   * shape_closure.folded_dim_misuse  — 折叠维 "...x" 重复出现或不在首位（FAIL）
#   * shape_closure.incompatible_dims  — 求值过程中显式维冲突
#   * shape_closure.dsl_parse_error    — numpy_expr 语法错
#   * shape_closure.dsl_eval_error     — numpy_expr 求值时类型/属性/调用错误
#   * shape_closure.shape_rule_kind_missing  — outputs[].shape_rule_kind 缺失（FAIL）
#   * shape_closure.data_dependent_missing_bounds — VariableOutput 缺 shape_bounds.max_elements
#   * shape_closure.rank_overflow      — 输出 rank 过深
#   * shape_closure.shape_rule_placeholder — shape_rule 含 TODO / 占位 / 简化说明
#   * shape_closure.reduction_shape_identity_suspicious — 纯 Reduction 直接写 input.shape
#   * shape_closure.synthesize_parse_error — boundary synthesize 无法作为 shape oracle（WARN）
#   * shape_closure.numpy_expr_shape_mismatch — concrete 边界样例下 shape_rule
#                                                与 formula 输出 shape 不一致
#   * shape_closure.numpy_expr_formula_eval_error — formula 无法作为边界 shape oracle

def stage_3(spec: dict) -> tuple[str, list[dict]]:
    """Resolve every output's shape; verify symbol registration / data-dependent metadata."""
    findings: list[dict] = []
    inputs = _input_shapes(spec)
    attrs = _attr_defaults(spec)
    registered_symbols = _registered_symbols(spec)
    shape_consistency = _check_numpy_expr_shape_consistency(spec)

    # First pass: report parse errors on inputs themselves
    parsed_inputs: dict[str, SymbolicShape] = {}
    for name, val in inputs.items():
        if isinstance(val, DslError):
            findings.append({
                "severity": "error",
                "rule_id": f"shape_closure.{val.code}",
                "field_path": val.field_path,
                "message": val.message,
                "suggested_fix": "修正 inputs[].shape.symbolic 列表",
            })
            continue
        parsed_inputs[name] = val

    # Verify explicit dim symbols are registered (folded dims need not be)
    for name, shape in parsed_inputs.items():
        for d in shape.explicit:
            if d.kind == "symbol" and d.name not in registered_symbols:
                findings.append({
                    "severity": "warning",
                    "rule_id": "shape_closure.unregistered_symbol",
                    "field_path": f"shape_constraints.symbols.{d.name}",
                    "message": f"显式维 {d.name!r} 出现在 input {name!r}.shape "
                               f"但未在 shape_constraints.symbols 登记",
                    "suggested_fix": f"在 shape_constraints.symbols 中添加 "
                                     f"{d.name}: {{kind: dim, range: [0, INT64_MAX]}}",
                })

    # Solve each output
    for i, out in enumerate(spec.get("outputs", []) or []):
        name = out.get("name")
        kind = out.get("shape_rule_kind")
        path = f"outputs[{i}]"

        if kind is None:
            findings.append({
                "severity": "error",
                "rule_id": "shape_closure.shape_rule_kind_missing",
                "field_path": f"{path}.shape_rule_kind",
                "message": f"output {name!r} 必须声明 shape_rule_kind: numpy_expr / data_dependent / textual_only",
                "suggested_fix": "添加 shape_rule_kind: numpy_expr（或 data_dependent / textual_only）",
            })
            continue

        if kind == "data_dependent":
            paradigms = set((spec.get("op") or {}).get("paradigms") or [])
            category = (spec.get("op") or {}).get("category", "")
            _data_dependent_allowed_categories = {"VariableOutput", "Reduction", "ReductionComposite", "ArgReduce", "IndexGather"}
            if "VariableOutput" not in paradigms and category not in _data_dependent_allowed_categories:
                findings.append({
                    "severity": "error",
                    "rule_id": "shape_closure.data_dependent_requires_variable_output",
                    "field_path": f"{path}.shape_rule_kind",
                    "message": "shape_rule_kind=data_dependent 仅允许用于 VariableOutput 范式"
                               "或 Reduction/ReductionComposite/ArgReduce 类别；"
                               "仅依赖 input shape / attribute 的输出必须写 numpy_expr shape_rule",
                    "suggested_fix": "改为 shape_rule_kind: numpy_expr 并写可执行 shape_rule，"
                                     "或确认算子确为 VariableOutput / Reduction 类",
                })
            _check_data_dependent_output(out, i, findings)
            continue

        if kind == "textual_only":
            # shape_rule 含 format_variants 占位符等不可执行语法；跳过求值
            # 要求：必须有 format_variants 且 shape_rule_description 存在
            fv = (spec.get("math_semantics") or {}).get("format_variants")
            if not fv:
                findings.append({
                    "severity": "error",
                    "rule_id": "shape_closure.textual_only_requires_format_variants",
                    "field_path": f"{path}.shape_rule_kind",
                    "message": "shape_rule_kind=textual_only 仅在 math_semantics.format_variants "
                               "存在时允许使用（shape 因数据排布格式而异）",
                    "suggested_fix": "添加 format_variants 或改为 numpy_expr",
                })
            if not out.get("shape_rule_description"):
                findings.append({
                    "severity": "error",
                    "rule_id": "shape_closure.textual_only_requires_description",
                    "field_path": f"{path}.shape_rule_description",
                    "message": "shape_rule_kind=textual_only 必须配 shape_rule_description "
                               "说明每种格式的具体 shape 规则",
                    "suggested_fix": "添加 shape_rule_description，列出各格式的 shape_rule",
                })
            continue

        if kind != "numpy_expr":
            findings.append({
                "severity": "error",
                "rule_id": "shape_closure.shape_rule_kind_unknown",
                "field_path": f"{path}.shape_rule_kind",
                "message": f"未知 shape_rule_kind: {kind!r}（仅支持 numpy_expr / data_dependent / textual_only）",
                "suggested_fix": "改为 numpy_expr、data_dependent 或 textual_only",
            })
            continue

        # numpy_expr 路径
        rule = out.get("shape_rule", "")
        placeholder = _shape_rule_placeholder_finding(rule, f"{path}.shape_rule")
        if placeholder:
            findings.append(placeholder)
        reduction_identity = _pure_reduction_identity_finding(spec, rule, name, f"{path}.shape_rule")
        if reduction_identity:
            findings.append(reduction_identity)
        try:
            shape = shape_eval.evaluate_shape_rule(
                rule,
                output_name=name,
                inputs=parsed_inputs,
                attr_values=attrs,
                field_path=f"{path}.shape_rule",
            )
            # rank overflow
            max_rank = 16
            for inp in spec.get("inputs", []) or []:
                rr = inp.get("rank_range") or [0, 8]
                if isinstance(rr, list) and len(rr) == 2:
                    max_rank = min(max_rank, int(rr[1]) + 4)
            if shape.rank_min > max_rank:
                findings.append({
                    "severity": "error",
                    "rule_id": "shape_closure.rank_overflow",
                    "field_path": f"{path}.shape_rule",
                    "message": f"输出 {name!r} 显式 rank 下界 {shape.rank_min} 超过限值 {max_rank}",
                    "suggested_fix": "检查 inputs[].rank_range 或 shape_rule 是否过深",
                })
        except DslError as e:
            if _can_suppress_symbolic_shape_error(e.code, name, shape_consistency):
                continue
            findings.append({
                "severity": "error",
                "rule_id": f"shape_closure.{e.code}",
                "field_path": e.field_path or f"{path}.shape_rule",
                "message": e.message,
                "suggested_fix": _shape_fix_hint(e.code),
            })

    findings.extend(shape_consistency.findings)
    status = "FAIL" if any(f["severity"] == "error" for f in findings) else "PASS"
    return status, findings


def _check_data_dependent_output(out: dict, i: int, findings: list[dict]) -> None:
    """data_dependent 输出的声明校验：data_dependent_shape: true + shape_bounds.max_elements。

    形状本身不求解（运行时才知道），但要求作者显式声明上界，下游 tiling / UB 预算才能用。
    """
    path = f"outputs[{i}]"
    if not out.get("data_dependent_shape"):
        findings.append({
            "severity": "error",
            "rule_id": "shape_closure.data_dependent_flag_missing",
            "field_path": f"{path}.data_dependent_shape",
            "message": "shape_rule_kind=data_dependent 必须配 data_dependent_shape: true",
            "suggested_fix": "加 data_dependent_shape: true",
        })
    bounds = out.get("shape_bounds") or {}
    if not bounds.get("max_elements"):
        findings.append({
            "severity": "error",
            "rule_id": "shape_closure.data_dependent_missing_bounds",
            "field_path": f"{path}.shape_bounds.max_elements",
            "message": "data_dependent 输出必须声明 shape_bounds.max_elements 上界",
            "suggested_fix": "加 shape_bounds.max_elements（如 'prod(input.shape) * rank(input)' 或具体整数）",
        })
    if not out.get("shape_rule_description"):
        findings.append({
            "severity": "warning",
            "rule_id": "shape_closure.data_dependent_missing_description",
            "field_path": f"{path}.shape_rule_description",
            "message": "data_dependent 输出建议补 shape_rule_description（自然语言描述形状语义）",
            "suggested_fix": "用一段中文/英文描述输出 shape 与 input 值的关系",
        })


def _shape_fix_hint(code: str) -> str:
    return {
        "dsl_parse_error": "检查 numpy_expr 语法（括号、缩进、表达式形式）",
        "dsl_eval_error": "检查 shape_rule 中 .shape / 切片 / np.broadcast_shapes 调用是否合法",
        "incompatible_dims": "检查相关维度是否一致；contraction 类的 K 维必须同名或同 const",
        "unresolved_symbol": "确认 shape_rule 引用的 input / attribute 名拼写正确",
        "folded_dim_misuse": "折叠维 '...x' 至多出现一次且必须在首位",
        "rank_overflow": "调整 inputs[].rank_range 或简化 shape_rule",
    }.get(code, "查阅 SKILL.md §4.3 / scripts/evaluators/shape_eval.py")


# ---------- stage 4 — dtype_closure ----------------------------------------


def stage_4(spec: dict) -> tuple[str, list[dict]]:
    """For each `supported_combinations` row, evaluate dtype_rule and compare."""
    findings: list[dict] = []
    combinations = (spec.get("dtype_policy") or {}).get("supported_combinations") or []
    outputs = spec.get("outputs") or []

    # Cross-check each combination
    for ci, combo in enumerate(combinations):
        in_dtypes = combo.get("inputs", {}) or {}
        declared_outs = combo.get("outputs", {}) or {}
        for i, out in enumerate(outputs):
            name = out.get("name")
            kind = out.get("dtype_rule_kind", "numpy_expr")  # 默认 numpy_expr
            rule = out.get("dtype_rule", "")
            if kind != "numpy_expr":
                findings.append({
                    "severity": "error",
                    "rule_id": "dtype_closure.dtype_rule_kind_unknown",
                    "field_path": f"outputs[{i}].dtype_rule_kind",
                    "message": f"未知 dtype_rule_kind: {kind!r}（v1 仅支持 numpy_expr）",
                    "suggested_fix": "改为 numpy_expr",
                })
                continue
            try:
                derived = dtype_eval.evaluate_dtype_rule(
                    rule,
                    output_name=name,
                    input_dtypes=in_dtypes,
                    field_path=f"outputs[{i}].dtype_rule",
                )
            except DslError as e:
                findings.append({
                    "severity": "error",
                    "rule_id": f"dtype_closure.{e.code}",
                    "field_path": e.field_path or f"dtype_policy.supported_combinations[{ci}]",
                    "message": f"output {name!r}: {e.message}",
                    "suggested_fix": "检查 dtype_rule 表达式或输入 dtype 组合是否可被 promote 表覆盖",
                })
                continue
            declared = declared_outs.get(name)
            if declared is None:
                findings.append({
                    "severity": "error",
                    "rule_id": "dtype_closure.combination_missing_output",
                    "field_path": f"dtype_policy.supported_combinations[{ci}].outputs.{name}",
                    "message": f"组合 #{ci} 未声明 output {name!r} 的 dtype",
                    "suggested_fix": f"补 outputs.{name}: <dtype>",
                })
                continue
            if declared != derived:
                findings.append({
                    "severity": "error",
                    "rule_id": "dtype_closure.combination_mismatch",
                    "field_path": f"dtype_policy.supported_combinations[{ci}].outputs.{name}",
                    "message": (
                        f"组合 #{ci}: dtype_rule 推得 {derived!r} 但显式表写 {declared!r}"
                    ),
                    "suggested_fix": (
                        f"把 outputs.{name} 改为 {derived!r}，"
                        f"或修正 dtype_rule"
                    ),
                })

    status = "FAIL" if any(f["severity"] == "error" for f in findings) else "PASS"
    return status, findings


# ---------- stage 5 — broadcast_legality -----------------------------------


def stage_5(spec: dict) -> tuple[str, list[dict]]:
    """校验算子计算中的 broadcast 语义是否合法。

    broadcast 描述的是算子计算是否需要对数据进行 broadcast（数据复制/扩展），
    而非仅描述多输入 shape 是否需要对齐。
    索引/元数据张量（axes/axis/dim 等 int 类型输入）不参与 broadcast check，
    仅对数据张量做 broadcast 验证。
    """
    findings: list[dict] = []
    inputs = _input_shapes(spec)
    parsed_inputs: list[SymbolicShape] = []
    for name, val in inputs.items():
        if isinstance(val, DslError):
            return "SKIP", findings
        # 索引/元数据张量不参与广播：名为 axes/axis/dim/indices 且 dtype 仅含整数类型
        inp = next((i for i in (spec.get("inputs") or []) if i.get("name") == name), None)
        if inp is not None:
            dtype_set = set(inp.get("dtype_set") or [])
            index_names = {"axes", "axis", "dim", "indices", "offsets", "size"}
            if name in index_names and dtype_set.issubset({"int32", "int64", "int16", "int8", "uint8"}):
                continue   # 跳过索引张量，不参与 broadcast 验证
        parsed_inputs.append(val)

    if not parsed_inputs:
        return "PASS", findings

    bspec = spec.get("broadcast") or {}
    try:
        bcast_mod.simulate(parsed_inputs, bspec)
    except DslError as e:
        findings.append({
            "severity": "error",
            "rule_id": f"broadcast_legality.{e.code}",
            "field_path": "broadcast",
            "message": e.message,
            "suggested_fix": _bcast_fix_hint(e.code, bspec),
        })

    status = "FAIL" if any(f["severity"] == "error" for f in findings) else "PASS"
    return status, findings


def _bcast_fix_hint(code: str, bspec: dict) -> str:
    kind = (bspec or {}).get("kind", "")
    if code == "incompatible_dims":
        return "检查输入显式维命名是否一致，或改 broadcast.kind 为 numpy"
    if code == "explicit_rules_uncovered":
        return ("explicit broadcast.rules 必须含 trailing + leading 各一条，"
                "且 count 加和覆盖输入 rank")
    if code == "numpy_violation":
        return f"broadcast.kind={kind!r} 但算子计算需要 broadcast 数据；改为 numpy"
    return "查阅 SKILL.md §4.5"
