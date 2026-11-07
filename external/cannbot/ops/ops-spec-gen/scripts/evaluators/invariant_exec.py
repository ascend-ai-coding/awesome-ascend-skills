# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Stage 11 — invariant value execution.

Run each ``math_semantics.invariants[]`` against the formula output on generated
inputs. Unlike stage 10 (formula vs framework oracle), stage 11 needs **no
external oracle** — it verifies the formula against its own declared
mathematical properties (softmax sum-to-one, relu non-negativity, add
commutativity, matmul/reduce zero-absorption...).

This is the key machine check for the **no-oracle case** (oracle absent /
composition DAG / oracle_call_failed), where stage 10 SKIPs. The paradox —
"the cases most needing form verification are exactly those with no oracle" —
is resolved because a wrong-form formula typically violates a declared
invariant on normal inputs (e.g. a non-normalizing "softmax" form yields
sum != 1).

Executable kinds (this version):
  * value group — elementwise_ge / elementwise_le / elementwise_eq /
                  reduce_equals / range_in / produces_in_set
  * algebraic   — equals_under_swap / equals_input_when_other_is_zero /
                  equals_when_input_is_zero

Skipped (not machine-executable here, reported as info):
  * idempotent (output shape may churn), associative_along_batch, monotone_along
    (needs special inputs), equals_input_when_other_is_identity (composition
    only), equals_after_op (external op), and the structural group
    (shape_equals_macro / no_leak_intermediates / permutation_of_input).

Skips the whole stage when:
  * formula_kind != numpy_expr  (no executable formula)
  * spec has no invariants
  * numpy is not installed
  * the formula cannot be compiled / executed
"""

from __future__ import annotations

import re
from typing import Any

from .formula_eval import (
    FormulaError,
    _resolve_shape,
    _gen_tensor,
    run_formula,
)


# Default tight tolerance for non-tolerance_inherit value comparisons on floats.
# Matches must be exact in spirit; a tiny slack absorbs last-bit noise only.
_DEFAULT_ATOL = 1e-6
_DEFAULT_RTOL = 1e-6


# Kinds this stage can machine-execute. Everything else is reported as info-skip.
_VALUE_KINDS = {
    "elementwise_ge", "elementwise_le", "elementwise_eq",
    "reduce_equals", "range_in", "produces_in_set",
}
_ALGEBRAIC_KINDS = {
    "equals_under_swap",
    "equals_input_when_other_is_zero",
    "equals_when_input_is_zero",
}
_EXECUTABLE_KINDS = _VALUE_KINDS | _ALGEBRAIC_KINDS


# ---------- formula execution (mirrors formula_oracle_equiv._run_formula) ---


def _gen_normal_inputs(np_mod, spec: dict, in_dtypes: dict[str, str],
                       seed: int) -> dict[str, Any] | None:
    """Generate one set of normal random input tensors (no special values).

    Each input gets a distinct derived seed so that multi-input specs do not
    receive identical tensors (which would mask non-commutativity / identity
    violations — e.g. x-y with x==y is always 0).
    """
    tensors: dict[str, Any] = {}
    for idx, inp in enumerate(spec.get("inputs") or []):
        name = inp.get("name")
        sym = (inp.get("shape") or {}).get("symbolic", [])
        shape = _resolve_shape(sym)
        dtype = in_dtypes.get(name) or (inp.get("dtype_set") or ["float32"])[0]
        try:
            arr = _gen_tensor(np_mod, shape, dtype, seed + idx)
            if arr.dtype.kind == "f":
                arr = arr.astype("float32")
            tensors[name] = arr
        except FormulaError:
            return None
    return tensors


def _resolve_axis(axis_expr: Any, spec: dict) -> int | None:
    """Resolve an axis value (${attr.dim} / int literal) to a concrete int."""
    if isinstance(axis_expr, bool):
        return None
    if isinstance(axis_expr, int):
        return axis_expr
    if not isinstance(axis_expr, str):
        return None
    m = re.match(r"^\$\{attr\.(\w+)\}$", axis_expr)
    if m:
        attr_name = m.group(1)
        for a in spec.get("attributes") or []:
            if a.get("name") == attr_name and "default" in a:
                d = a["default"]
                if isinstance(d, int) and not isinstance(d, bool):
                    return d
        return None
    # ${input.X} requires runtime input value — not statically resolvable.
    if axis_expr.startswith("${"):
        return None
    try:
        return int(axis_expr)
    except (ValueError, TypeError):
        return None


def _get_tolerance(spec: dict, inv: dict, in_dtype: str) -> tuple[float, float]:
    """Return (atol, rtol) for a tolerance_inherit invariant; else tight default."""
    if inv.get("tolerance_inherit"):
        per = (spec.get("numerical_tolerance") or {}).get("per_dtype") or {}
        entry = per.get(in_dtype) or {}
        return (float(entry.get("atol", 1e-5)), float(entry.get("rtol", 1e-5)))
    return (_DEFAULT_ATOL, _DEFAULT_RTOL)


def _first_output_name(spec: dict) -> str | None:
    outs = spec.get("outputs") or []
    return outs[0].get("name") if outs else None


_INDEX_INPUT_NAMES = {"axes", "axis", "dim", "indices", "offsets", "size"}
_INT_DTYPES = {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}


def _is_index_input(spec: dict, name: str) -> bool:
    """True if `name` is an index/metadata input (integer axis/indices tensor).

    These must NOT be zeroed by algebraic invariants — zeroing `axis` is
    meaningless and produces false violations (mirrors stage 5 broadcast logic).
    """
    if name not in _INDEX_INPUT_NAMES:
        return False
    for inp in spec.get("inputs") or []:
        if inp.get("name") == name:
            dtype_set = set(inp.get("dtype_set") or [])
            return dtype_set.issubset(_INT_DTYPES)
    return False


# ---------- per-kind checkers ----------------------------------------------


def _check_elementwise_threshold(out, inv, kind, out_name, np_mod) -> list[dict]:
    """elementwise_ge / elementwise_le against a scalar threshold."""
    threshold = inv.get("value")
    findings: list[dict] = []
    if threshold is None:
        return findings
    arr = np_mod.asarray(out)
    finite = ~np_mod.isnan(arr)
    if not finite.any():
        return findings
    farr = arr[finite]
    if kind == "elementwise_ge":
        bad = int((farr < threshold).sum())
        cmp_label, want = "≥", threshold
    else:  # elementwise_le
        bad = int((farr > threshold).sum())
        cmp_label, want = "≤", threshold
    if bad:
        findings.append({
            "severity": "error",
            "rule_id": f"invariant_exec.{kind}_violated",
            "field_path": f"math_semantics.invariants[].{inv.get('name', kind)}",
            "message": (
                f"invariant {inv.get('name', kind)!r} (kind={kind}) 被违反："
                f"输出 {out_name} 有 {bad} 处不满足 out {cmp_label} {want}。"
                f"这通常表明 formula 实现形式与声明的数学性质不符。"
            ),
            "suggested_fix": (
                f"修正 formula 使输出满足 out {cmp_label} {want}，"
                f"或确认 invariant 声明正确"
            ),
        })
    return findings


def _check_elementwise_eq(out, inv, out_name, atol, rtol, np_mod) -> list[dict]:
    value = inv.get("value")
    if value is None:
        return []
    arr = np_mod.asarray(out)
    finite = ~np_mod.isnan(arr)
    if not finite.any():
        return []
    farr = arr[finite].astype("float64")
    abs_diff = np_mod.abs(farr - float(value))
    rel_diff = abs_diff / (abs(float(value)) + 1e-30)
    if (abs_diff.max() > atol) and (rel_diff.max() > rtol):
        return [{
            "severity": "error",
            "rule_id": "invariant_exec.elementwise_eq_violated",
            "field_path": f"math_semantics.invariants[].{inv.get('name', 'elementwise_eq')}",
            "message": (
                f"invariant {inv.get('name', 'elementwise_eq')!r} (elementwise_eq) 被违反："
                f"输出 {out_name} 未逐元素等于 {value}（max_abs={float(abs_diff.max()):.2e}）。"
            ),
            "suggested_fix": f"修正 formula 使输出恒等于 {value}",
        }]
    return []


def _check_reduce_equals(out, inv, spec, out_name, atol, rtol, np_mod) -> list[dict]:
    reducer = inv.get("reducer")
    axis = _resolve_axis(inv.get("axis"), spec)
    value = inv.get("value")
    if reducer is None or axis is None or value is None:
        return [{
            "severity": "info",
            "rule_id": "invariant_exec.reduce_equals_unresolved",
            "field_path": f"math_semantics.invariants[].{inv.get('name', 'reduce_equals')}",
            "message": (
                f"invariant {inv.get('name', 'reduce_equals')!r} 的 reducer/axis/value "
                f"无法解析（reducer={reducer!r}, axis={inv.get('axis')!r}），跳过执行"
            ),
            "suggested_fix": "确保 reducer/axis/value 字段齐全且 axis 可静态解析",
        }]
    arr = np_mod.asarray(out)
    if axis >= arr.ndim or axis < -arr.ndim:
        return []
    fn = {"sum": np_mod.sum, "mean": np_mod.mean, "max": np_mod.max, "min": np_mod.min}.get(reducer)
    if fn is None:
        return []
    reduced = fn(arr, axis=axis, keepdims=False)
    reduced = np_mod.asarray(reduced).astype("float64")
    bad_mask = np_mod.isnan(reduced)
    finite = ~bad_mask
    if finite.any():
        rfin = reduced[finite]
        abs_diff = np_mod.abs(rfin - float(value))
        rel_diff = abs_diff / (abs(float(value)) + 1e-30)
        if (abs_diff.max() > atol) and (rel_diff.max() > rtol):
            return [{
                "severity": "error",
                "rule_id": "invariant_exec.reduce_equals_violated",
                "field_path": f"math_semantics.invariants[].{inv.get('name', 'reduce_equals')}",
                "message": (
                    f"invariant {inv.get('name', 'reduce_equals')!r} 被违反："
                    f"输出 {out_name} 沿 axis={axis} 做 {reducer} 归约后 "
                    f"max_abs_diff={float(abs_diff.max()):.2e}（期望 == {value}）。"
                    f"归一化/分布类算子形式错误通常在此暴露。"
                ),
                "suggested_fix": f"修正 formula 使 {reducer}(out, axis={axis}) == {value}",
            }]
    return []


def _check_range_in(out, inv, out_name, np_mod) -> list[dict]:
    rng = inv.get("range")
    if not isinstance(rng, (list, tuple)) or len(rng) != 2:
        return []
    lo, hi = float(rng[0]), float(rng[1])
    arr = np_mod.asarray(out)
    finite = ~np_mod.isnan(arr)
    if not finite.any():
        return []
    farr = arr[finite]
    bad = int(((farr < lo) | (farr > hi)).sum())
    if bad:
        return [{
            "severity": "error",
            "rule_id": "invariant_exec.range_in_violated",
            "field_path": f"math_semantics.invariants[].{inv.get('name', 'range_in')}",
            "message": (
                f"invariant {inv.get('name', 'range_in')!r} 被违反："
                f"输出 {out_name} 有 {bad} 处落在 [{lo}, {hi}] 之外。"
            ),
            "suggested_fix": f"修正 formula 使输出 ∈ [{lo}, {hi}]",
        }]
    return []


def _check_produces_in_set(out, inv, out_name, np_mod) -> list[dict]:
    s = inv.get("set")
    if not isinstance(s, list):
        return []
    allowed = set(s)
    arr = np_mod.asarray(out)
    bad = int((~np_mod.isin(arr, list(allowed))).sum())
    if bad:
        return [{
            "severity": "error",
            "rule_id": "invariant_exec.produces_in_set_violated",
            "field_path": f"math_semantics.invariants[].{inv.get('name', 'produces_in_set')}",
            "message": (
                f"invariant {inv.get('name', 'produces_in_set')!r} 被违反："
                f"输出 {out_name} 有 {bad} 处取值不在声明集合 {sorted(allowed)} 内。"
            ),
            "suggested_fix": f"修正 formula 使输出仅取值于 {sorted(allowed)}",
        }]
    return []


def _check_equals_under_swap(np_mod, spec, inv, inputs, out_name,
                             atol, rtol) -> list[dict]:
    swap = inv.get("swap") or []
    if len(swap) < 2:
        return []
    a, b = swap[0], swap[1]
    if a not in inputs or b not in inputs:
        return []
    out1 = run_formula(np_mod, spec, inputs)
    swapped = dict(inputs)
    swapped[a], swapped[b] = inputs[b], inputs[a]
    out2 = run_formula(np_mod, spec, swapped)
    if not out1 or not out2 or out_name not in out1 or out_name not in out2:
        return []
    v1 = np_mod.asarray(out1[out_name]).astype("float64")
    v2 = np_mod.asarray(out2[out_name]).astype("float64")
    if v1.shape != v2.shape:
        return [{
            "severity": "error",
            "rule_id": "invariant_exec.equals_under_swap_violated",
            "field_path": f"math_semantics.invariants[].{inv.get('name', 'equals_under_swap')}",
            "message": (
                f"invariant {inv.get('name', 'equals_under_swap')!r} 被违反："
                f"交换 {a}/{b} 后输出 shape 不一致（{v1.shape} vs {v2.shape}）。"
            ),
            "suggested_fix": "确认算子对该输入对满足交换律，或修正 formula",
        }]
    abs_diff = np_mod.abs(v1 - v2)
    rel_diff = abs_diff / (np_mod.abs(v1) + np_mod.abs(v2) + 1e-30)
    if (abs_diff.max() > atol) and (rel_diff.max() > rtol):
        return [{
            "severity": "error",
            "rule_id": "invariant_exec.equals_under_swap_violated",
            "field_path": f"math_semantics.invariants[].{inv.get('name', 'equals_under_swap')}",
            "message": (
                f"invariant {inv.get('name', 'equals_under_swap')!r} 被违反："
                f"交换输入 {a}/{b} 后输出 {out_name} 发散（max_abs={float(abs_diff.max()):.2e}）。"
                f"非交换 formula 被声明为交换律时会在此暴露。"
            ),
            "suggested_fix": "修正 formula 使其对声明输入对满足交换律，或更正 invariant 声明",
        }]
    return []


def _check_identity_when_other_zero(np_mod, spec, inv, inputs, in_dtypes,
                                    seed, out_name, atol, rtol) -> list[dict]:
    identity_input = inv.get("identity_input")
    zero_input = inv.get("zero_input")
    if not identity_input or not zero_input or identity_input not in inputs:
        return []
    # zero_input 必须是数据输入，不能是索引/元数据张量
    if _is_index_input(spec, zero_input):
        return []
    zeroed = dict(inputs)
    shape = inputs[zero_input].shape
    zeroed[zero_input] = np_mod.zeros(shape, dtype="float32")
    out = run_formula(np_mod, spec, zeroed)
    if not out or out_name not in out:
        return []
    produced = np_mod.asarray(out[out_name]).astype("float64")
    expected = np_mod.asarray(inputs[identity_input]).astype("float64")
    if produced.shape != expected.shape:
        return []
    if produced.size == 0:
        return []
    abs_diff = np_mod.abs(produced - expected)
    rel_diff = abs_diff / (np_mod.abs(expected) + 1e-30)
    if (abs_diff.max() > atol) and (rel_diff.max() > rtol):
        return [{
            "severity": "error",
            "rule_id": "invariant_exec.equals_input_when_other_is_zero_violated",
            "field_path": f"math_semantics.invariants[].{inv.get('name', 'equals_input_when_other_is_zero')}",
            "message": (
                f"invariant {inv.get('name', 'equals_input_when_other_is_zero')!r} 被违反："
                f"置零 {zero_input} 后输出 {out_name} 不等于 {identity_input}（max_abs={float(abs_diff.max()):.2e}）。"
            ),
            "suggested_fix": f"修正 formula 使 {zero_input}=0 时输出 == {identity_input}",
        }]
    return []


def _check_equals_when_input_is_zero(np_mod, spec, inv, inputs, out_name,
                                     atol, rtol) -> list[dict]:
    value = inv.get("value")
    if value is None:
        return []
    findings: list[dict] = []
    for name, tensor in inputs.items():
        # 跳过索引/元数据输入（axis/axes/dim/indices 等整数张量）——置零它们无意义
        if _is_index_input(spec, name):
            continue
        zeroed = dict(inputs)
        zeroed[name] = np_mod.zeros(tensor.shape, dtype="float32")
        out = run_formula(np_mod, spec, zeroed)
        if not out or out_name not in out:
            continue
        produced = np_mod.asarray(out[out_name]).astype("float64")
        if produced.size == 0:
            continue  # 空输出（如 IndexGather 零输入→空）无可比元素，跳过
        abs_diff = np_mod.abs(produced - float(value))
        rel_diff = abs_diff / (abs(float(value)) + 1e-30)
        if (abs_diff.max() > atol) and (rel_diff.max() > rtol):
            findings.append({
                "severity": "error",
                "rule_id": "invariant_exec.equals_when_input_is_zero_violated",
                "field_path": f"math_semantics.invariants[].{inv.get('name', 'equals_when_input_is_zero')}",
                "message": (
                    f"invariant {inv.get('name', 'equals_when_input_is_zero')!r} 被违反："
                    f"置零输入 {name} 后输出 {out_name} 不等于 {value}（max_abs={float(abs_diff.max()):.2e}）。"
                ),
                "suggested_fix": f"修正 formula 使任一数据输入为 0 时输出 == {value}",
            })
            break  # one violation is enough to report
    return findings


# ---------- stage 11 --------------------------------------------------------


def stage_11(spec: dict) -> tuple[str, list[dict]]:
    """Execute math_semantics.invariants[] against the formula output.

    See module docstring for the no-oracle rationale and the executable/skipped
    kind sets.
    """
    findings: list[dict] = []

    # --- Prerequisites ---

    formula_kind = (spec.get("math_semantics") or {}).get("formula_kind")
    if formula_kind != "numpy_expr":
        return "SKIP", [{
            "severity": "info",
            "rule_id": "invariant_exec.skipped_non_numpy",
            "field_path": "math_semantics.formula_kind",
            "message": f"formula_kind={formula_kind!r}，stage 11 仅在 numpy_expr 下运行",
            "suggested_fix": None,
        }]

    invariants = (spec.get("math_semantics") or {}).get("invariants") or []
    if not invariants:
        return "SKIP", [{
            "severity": "info",
            "rule_id": "invariant_exec.no_invariants",
            "field_path": "math_semantics.invariants",
            "message": "spec 未声明任何 invariant，stage 11 跳过",
            "suggested_fix": None,
        }]

    try:
        import numpy as np
    except ImportError:
        return "SKIP", [{
            "severity": "info",
            "rule_id": "invariant_exec.numpy_not_installed",
            "field_path": "<env>",
            "message": "numpy 未安装；stage 11 跳过",
            "suggested_fix": "pip install numpy",
        }]

    combos = (spec.get("dtype_policy") or {}).get("supported_combinations") or []
    if not combos:
        return "SKIP", [{
            "severity": "info",
            "rule_id": "invariant_exec.no_combination",
            "field_path": "dtype_policy.supported_combinations",
            "message": "缺 supported_combinations，stage 11 跳过",
            "suggested_fix": None,
        }]
    in_dtypes = combos[0].get("inputs") or {}
    in_dtype = next(iter(in_dtypes.values()), "float32")

    seed = (spec.get("test_matrix") or {}).get("random", {}).get("seed", 42)
    inputs = _gen_normal_inputs(np, spec, in_dtypes, seed)
    if inputs is None:
        return "SKIP", [{
            "severity": "info",
            "rule_id": "invariant_exec.no_inputs",
            "field_path": "<internal>",
            "message": "无法生成测试输入，stage 11 跳过",
            "suggested_fix": None,
        }]

    out_name = _first_output_name(spec)

    # --- Pre-compute formula output for value-group invariants ---

    cached_output: dict[str, Any] | None = None
    cached_fail_reported = False

    def _get_formula_output(inv_name: str) -> Any | None:
        nonlocal cached_output, cached_fail_reported
        if cached_output is not None:
            return cached_output.get(out_name)
        cached_output = run_formula(np, spec, inputs)
        if not cached_output or out_name not in cached_output:
            if not cached_fail_reported:
                findings.append({
                    "severity": "info",
                    "rule_id": "invariant_exec.formula_no_output",
                    "field_path": f"math_semantics.invariants[].{inv_name}",
                    "message": (
                        f"invariant {inv_name!r}：formula 未产出输出 {out_name!r}，跳过"
                    ),
                    "suggested_fix": None,
                })
                cached_fail_reported = True
            return None
        return cached_output[out_name]

    # --- Iterate invariants ---

    for inv in invariants:
        kind = inv.get("kind")
        name = inv.get("name", kind)

        if kind not in _EXECUTABLE_KINDS:
            findings.append({
                "severity": "info",
                "rule_id": "invariant_exec.kind_not_executable",
                "field_path": f"math_semantics.invariants[].{name}",
                "message": (
                    f"invariant {name!r} 的 kind={kind!r} 暂不支持机器执行"
                    f"（结构/组合/复杂代数类），跳过值校验"
                ),
                "suggested_fix": None,
            })
            continue

        atol, rtol = _get_tolerance(spec, inv, in_dtype)

        if kind in _VALUE_KINDS:
            out = _get_formula_output(name)
            if out is None:
                continue
            if kind in ("elementwise_ge", "elementwise_le"):
                findings.extend(_check_elementwise_threshold(out, inv, kind, out_name, np))
            elif kind == "elementwise_eq":
                findings.extend(_check_elementwise_eq(out, inv, out_name, atol, rtol, np))
            elif kind == "reduce_equals":
                findings.extend(_check_reduce_equals(out, inv, spec, out_name, atol, rtol, np))
            elif kind == "range_in":
                findings.extend(_check_range_in(out, inv, out_name, np))
            elif kind == "produces_in_set":
                findings.extend(_check_produces_in_set(out, inv, out_name, np))
        elif kind == "equals_under_swap":
            findings.extend(_check_equals_under_swap(np, spec, inv, inputs, out_name, atol, rtol))
        elif kind == "equals_input_when_other_is_zero":
            findings.extend(_check_identity_when_other_zero(
                np, spec, inv, inputs, in_dtypes, seed, out_name, atol, rtol))
        elif kind == "equals_when_input_is_zero":
            findings.extend(_check_equals_when_input_is_zero(np, spec, inv, inputs, out_name, atol, rtol))

    status = "FAIL" if any(f["severity"] == "error" for f in findings) else "PASS"
    return status, findings
