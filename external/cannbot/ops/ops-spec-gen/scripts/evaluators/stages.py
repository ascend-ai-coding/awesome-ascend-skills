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

from typing import Any

from .types import SymbolicShape, DslError
from .parser import parse_shape_literal
from . import shape_eval, dtype_eval
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

def _stage3_parse_input_shapes(inputs, findings):
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
    return parsed_inputs


def _stage3_check_unregistered_symbols(parsed_inputs, registered_symbols, findings):
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


def _stage3_max_rank_for_spec(spec):
    max_rank = 16
    for inp in spec.get("inputs", []) or []:
        rr = inp.get("rank_range") or [0, 8]
        if isinstance(rr, list) and len(rr) == 2:
            max_rank = min(max_rank, int(rr[1]) + 4)
    return max_rank


def _stage3_solve_numpy_expr(spec, out, name, i, parsed_inputs, attrs, findings):
    path = f"outputs[{i}]"
    rule = out.get("shape_rule", "")
    try:
        shape = shape_eval.evaluate_shape_rule(
            rule,
            output_name=name,
            inputs=parsed_inputs,
            attr_values=attrs,
            field_path=f"{path}.shape_rule",
        )
        max_rank = _stage3_max_rank_for_spec(spec)
        if shape.rank_min > max_rank:
            findings.append({
                "severity": "error",
                "rule_id": "shape_closure.rank_overflow",
                "field_path": f"{path}.shape_rule",
                "message": f"输出 {name!r} 显式 rank 下界 {shape.rank_min} 超过限值 {max_rank}",
                "suggested_fix": "检查 inputs[].rank_range 或 shape_rule 是否过深",
            })
    except DslError as e:
        findings.append({
            "severity": "error",
            "rule_id": f"shape_closure.{e.code}",
            "field_path": e.field_path or f"{path}.shape_rule",
            "message": e.message,
            "suggested_fix": _shape_fix_hint(e.code),
        })


def _stage3_solve_output(spec, out, i, parsed_inputs, attrs, findings):
    name = out.get("name")
    kind = out.get("shape_rule_kind")
    path = f"outputs[{i}]"
    if kind is None:
        findings.append({
            "severity": "error",
            "rule_id": "shape_closure.shape_rule_kind_missing",
            "field_path": f"{path}.shape_rule_kind",
            "message": f"output {name!r} 必须声明 shape_rule_kind: numpy_expr 或 data_dependent",
            "suggested_fix": "添加 shape_rule_kind: numpy_expr（或 data_dependent）",
        })
        return
    if kind == "data_dependent":
        _check_data_dependent_output(out, i, findings)
        return
    if kind != "numpy_expr":
        findings.append({
            "severity": "error",
            "rule_id": "shape_closure.shape_rule_kind_unknown",
            "field_path": f"{path}.shape_rule_kind",
            "message": f"未知 shape_rule_kind: {kind!r}（仅支持 numpy_expr / data_dependent）",
            "suggested_fix": "改为 numpy_expr 或 data_dependent",
        })
        return
    _stage3_solve_numpy_expr(spec, out, name, i, parsed_inputs, attrs, findings)


def stage_3(spec: dict) -> tuple[str, list[dict]]:
    """Resolve every output's shape; verify symbol registration / data-dependent contract."""
    findings: list[dict] = []
    inputs = _input_shapes(spec)
    attrs = _attr_defaults(spec)
    registered_symbols = _registered_symbols(spec)

    parsed_inputs = _stage3_parse_input_shapes(inputs, findings)
    _stage3_check_unregistered_symbols(parsed_inputs, registered_symbols, findings)

    for i, out in enumerate(spec.get("outputs", []) or []):
        _stage3_solve_output(spec, out, i, parsed_inputs, attrs, findings)

    status = "FAIL" if any(f["severity"] == "error" for f in findings) else "PASS"
    return status, findings


def _check_data_dependent_output(out: dict, i: int, findings: list[dict]) -> None:
    """data_dependent 输出的契约校验：data_dependent_shape: true + shape_bounds.max_elements。

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


def _stage4_check_one_output(ci, combo, i, out, in_dtypes, declared_outs, findings):
    name = out.get("name")
    kind = out.get("dtype_rule_kind", "numpy_expr")
    rule = out.get("dtype_rule", "")
    if kind != "numpy_expr":
        findings.append({
            "severity": "error",
            "rule_id": "dtype_closure.dtype_rule_kind_unknown",
            "field_path": f"outputs[{i}].dtype_rule_kind",
            "message": f"未知 dtype_rule_kind: {kind!r}（v1 仅支持 numpy_expr）",
            "suggested_fix": "改为 numpy_expr",
        })
        return
    try:
        derived = dtype_eval.evaluate_dtype_rule(
            rule, output_name=name, input_dtypes=in_dtypes,
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
        return
    declared = declared_outs.get(name)
    if declared is None:
        findings.append({
            "severity": "error",
            "rule_id": "dtype_closure.combination_missing_output",
            "field_path": f"dtype_policy.supported_combinations[{ci}].outputs.{name}",
            "message": f"组合 #{ci} 未声明 output {name!r} 的 dtype",
            "suggested_fix": f"补 outputs.{name}: <dtype>",
        })
        return
    if declared != derived:
        findings.append({
            "severity": "error",
            "rule_id": "dtype_closure.combination_mismatch",
            "field_path": f"dtype_policy.supported_combinations[{ci}].outputs.{name}",
            "message": f"组合 #{ci}: dtype_rule 推得 {derived!r} 但显式表写 {declared!r}",
            "suggested_fix": f"把 outputs.{name} 改为 {derived!r}，或修正 dtype_rule",
        })


def stage_4(spec: dict) -> tuple[str, list[dict]]:
    """For each `supported_combinations` row, evaluate dtype_rule and compare."""
    findings: list[dict] = []
    combinations = (spec.get("dtype_policy") or {}).get("supported_combinations") or []
    outputs = spec.get("outputs") or []

    for ci, combo in enumerate(combinations):
        in_dtypes = combo.get("inputs", {}) or {}
        declared_outs = combo.get("outputs", {}) or {}
        for i, out in enumerate(outputs):
            _stage4_check_one_output(ci, combo, i, out, in_dtypes, declared_outs, findings)

    status = "FAIL" if any(f["severity"] == "error" for f in findings) else "PASS"
    return status, findings


# ---------- stage 5 — broadcast_legality -----------------------------------


def stage_5(spec: dict) -> tuple[str, list[dict]]:
    """Simulate broadcast.kind/rules against input shapes; report incompatibilities."""
    findings: list[dict] = []
    inputs = _input_shapes(spec)
    parsed_inputs: list[SymbolicShape] = []
    for name, val in inputs.items():
        if isinstance(val, DslError):
            return "SKIP", findings
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
        return f"broadcast.kind={kind!r} 但输入 shape 实际需要广播；改为 numpy"
    return "查阅 SKILL.md §4.5"
