# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Stage 10 — formula-oracle value equivalence.

When both the spec formula (numpy_expr) and the reference oracle (reachable)
are available, run both on the same inputs — including special values (NaN,
inf, large numbers, signed zeros +0/-0) — and compare outputs. Catches the class of error where
the spec formula uses a *mathematically equivalent but computationally
different* form than the actual framework implementation (e.g. textbook
``m_new = β1·m + (1-β1)·grad`` vs TF incremental ``m = m + (1-β1)·(grad-m)``).
These forms diverge on NaN/inf propagation and last-bit rounding, causing
bit-level mismatch between kernel and golden at downstream black-box testing.

Skips when:
  * formula_kind != 'numpy_expr'  (python_block / textual_only)
  * reference_oracle.absent == true
  * numpy or the oracle framework is not installed
  * oracle uses composition mode (DAG — not yet supported for value comparison)
  * the oracle API call fails (e.g. signature mismatch)

Emits errors when:
  * formula output and oracle output have different NaN patterns
  * formula output and oracle output have different inf/-inf patterns
  * formula output and oracle output have different +0/-0 sign patterns
  * formula output and oracle output diverge beyond tolerance on finite values

Cost note: importing torch can take 1-3 s (cached across stages within one
process). Formula sandbox eval + oracle call on small tensors is < 1 s total.
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
from .oracle_check import _resolve_api_callable, _PLACEHOLDER_RE


# Tolerance for finite-value comparison between formula and oracle.
# We use a moderate tolerance because mathematically-equivalent formulas
# may produce slightly different last-bit results even when the *form* is
# correct (e.g. summation order differences). The critical signal is in
# the NaN/inf pattern, not the last bits.
_FINITE_ATOL = 1e-5
_FINITE_RTOL = 1e-5


def _gen_special_tensors(np, spec: dict, in_dtypes: dict, seed: int) -> list[dict[str, Any]]:
    """Generate multiple sets of input tensors, each with a different special value.

    Returns a list of dicts, each mapping input name → numpy tensor.
    Sets:
      * normal   — baseline random values (no special values)
      * nan      — one position set to NaN
      * inf      — one position set to +inf
      * ninf     — one position set to -inf
      * large    — one position set to a very large value
      * pos_zero — one position set to +0.0
      * neg_zero — one position set to -0.0
    """
    input_names = [inp.get("name") for inp in (spec.get("inputs") or [])]

    # Build the base tensors (same as stage 8)
    base: dict[str, Any] = {}
    for inp in spec.get("inputs") or []:
        name = inp.get("name")
        sym = (inp.get("shape") or {}).get("symbolic", [])
        shape = _resolve_shape(sym)
        dtype = in_dtypes.get(name) or (inp.get("dtype_set") or ["float32"])[0]
        try:
            arr = _gen_tensor(np, shape, dtype, seed)
            if arr.dtype.kind == "f":
                arr = arr.astype("float32")
            base[name] = arr
        except FormulaError:
            base[name] = np.zeros(shape, dtype="float32")

    if not base:
        return []

    # Determine a position to inject special values (first element)
    first_name = input_names[0]
    first_arr = base[first_name]

    # Normal baseline（构造一次，空输入早退与正常路径共用）
    normal_set = {"_label": "normal", **{k: v.copy() for k, v in base.items()}}
    if first_arr.size == 0:
        return [normal_set]

    flat = first_arr.flatten()
    pos = 0  # inject at first element

    test_sets: list[dict[str, Any]] = [normal_set]

    # NaN injection
    nan_set = {k: v.copy() for k, v in base.items()}
    flat_nan = nan_set[first_name].flatten()
    if flat_nan.dtype.kind == "f":
        flat_nan[pos] = float("nan")
        nan_set[first_name] = flat_nan.reshape(first_arr.shape)
        test_sets.append({"_label": "nan", **nan_set})

    # +inf injection
    inf_set = {k: v.copy() for k, v in base.items()}
    flat_inf = inf_set[first_name].flatten()
    if flat_inf.dtype.kind == "f":
        flat_inf[pos] = float("inf")
        inf_set[first_name] = flat_inf.reshape(first_arr.shape)
        test_sets.append({"_label": "inf", **inf_set})

    # -inf injection
    ninf_set = {k: v.copy() for k, v in base.items()}
    flat_ninf = ninf_set[first_name].flatten()
    if flat_ninf.dtype.kind == "f":
        flat_ninf[pos] = float("-inf")
        ninf_set[first_name] = flat_ninf.reshape(first_arr.shape)
        test_sets.append({"_label": "ninf", **ninf_set})

    # Large value injection
    large_set = {k: v.copy() for k, v in base.items()}
    flat_large = large_set[first_name].flatten()
    if flat_large.dtype.kind == "f":
        flat_large[pos] = 1e30
        large_set[first_name] = flat_large.reshape(first_arr.shape)
        test_sets.append({"_label": "large", **large_set})

    # +0 / -0 injection — IEEE 754 signed zeros compare equal but expose their
    # sign via 1/(+0)=+inf vs 1/(-0)=-inf and np.signbit. Different computation
    # forms (e.g. textbook a*x+b*y vs incremental x+(y-x)) produce different
    # zero signs on cancellation / underflow paths.
    for zval, zlabel in ((0.0, "pos_zero"), (-0.0, "neg_zero")):
        zero_set = {k: v.copy() for k, v in base.items()}
        flat_zero = zero_set[first_name].flatten()
        if flat_zero.dtype.kind == "f":
            flat_zero[pos] = zval
            zero_set[first_name] = flat_zero.reshape(first_arr.shape)
            test_sets.append({"_label": zlabel, **zero_set})

    return test_sets


def _resolve_kwargs(spec: dict, oracle: dict, input_tensors: dict[str, Any]) -> dict[str, Any]:
    """Resolve oracle kwargs placeholders to actual values."""
    attr_names = {a.get("name"): a for a in (spec.get("attributes") or [])}
    input_names = {i.get("name") for i in (spec.get("inputs") or [])}

    kwargs: dict[str, Any] = {}
    for kw_name, kw_val in (oracle.get("kwargs") or {}).items():
        if not isinstance(kw_val, str):
            kwargs[kw_name] = kw_val
            continue

        resolved = kw_val
        matches = list(_PLACEHOLDER_RE.finditer(kw_val))
        single_full = len(matches) == 1 and matches[0].group(0) == kw_val
        for m in matches:
            ns = m.group("ns")
            path = m.group("path")
            head = path.split(".")[0]
            if ns == "attr" and head in attr_names:
                attr_def = attr_names[head]
                val = attr_def.get("default")
                if single_full:
                    resolved = val if val is not None else 0
                else:
                    resolved = resolved.replace(m.group(0), str(val) if val is not None else "0")
            elif ns == "input" and head in input_tensors:
                resolved = input_tensors[head]
                break  # can't string-replace a tensor; use directly
            else:
                resolved = resolved.replace(m.group(0), "0")
        kwargs[kw_name] = resolved

    return kwargs


def _to_framework_tensor(framework: str, np_arr, framework_mod):
    """Convert a numpy array to a framework-native tensor."""
    if framework == "numpy":
        return np_arr
    if framework == "torch":
        return framework_mod.from_numpy(np_arr)
    if framework == "tensorflow":
        return framework_mod.constant(np_arr)
    if framework == "jax":
        return framework_mod.numpy.asarray(np_arr)
    # Fallback: try from_numpy / asarray
    if hasattr(framework_mod, "from_numpy"):
        return framework_mod.from_numpy(np_arr)
    if hasattr(framework_mod, "asarray"):
        return framework_mod.asarray(np_arr)
    return np_arr


def _from_framework_tensor(result, np):
    """Convert a framework tensor (or tuple of tensors) back to numpy."""
    if isinstance(result, (tuple, list)):
        return [_from_framework_tensor(r, np) for r in result]
    if hasattr(result, "detach"):  # torch
        return result.detach().cpu().numpy()
    if hasattr(result, "numpy"):  # tensorflow / keras
        return result.numpy()
    if hasattr(result, "astype") and not hasattr(result, "device"):
        # likely already numpy
        return result
    try:
        return np.asarray(result)
    except Exception:
        return None


def _call_oracle_single_api(
    framework_mod, oracle_api, spec: dict, oracle: dict,
    input_tensors: dict[str, Any], np
) -> Any | None:
    """Call the oracle API with the given inputs. Returns output or None on failure."""
    fw_tensors = {}
    framework = oracle.get("framework", "numpy")
    for name, arr in input_tensors.items():
        fw_tensors[name] = _to_framework_tensor(framework, arr, framework_mod)

    kwargs = _resolve_kwargs(spec, oracle, fw_tensors)

    input_names = [inp.get("name") for inp in (spec.get("inputs") or [])]
    positional = [fw_tensors.get(name) for name in input_names if name in fw_tensors]

    try:
        if kwargs:
            result = oracle_api(*positional, **kwargs)
        else:
            result = oracle_api(*positional)
    except Exception:
        return None

    return _from_framework_tensor(result, np)


def _compare_outputs(formula_out: dict[str, Any], oracle_out, np, output_name: str,
                     test_label: str, output_index: int = 0) -> list[dict]:
    """Compare formula output with oracle output for a single output variable.

    Checks:
      1. NaN pattern match
      2. inf/-inf pattern match
      3. Finite value closeness
    """
    findings: list[dict] = []

    # If oracle returned a tuple/list, take the element matching this output's index
    if isinstance(oracle_out, (list, tuple)):
        oracle_arr = oracle_out[output_index] if output_index < len(oracle_out) else None
    else:
        oracle_arr = oracle_out

    if oracle_arr is None:
        return findings

    oracle_arr = np.asarray(oracle_arr) if not isinstance(oracle_arr, type(np.zeros(0))) else oracle_arr

    if output_name not in formula_out:
        return findings

    f_val = formula_out[output_name]

    # Shape check — if shapes don't match, report mismatch and skip value comparison
    if f_val.shape != oracle_arr.shape:
        findings.append({
            "severity": "warning",
            "rule_id": "formula_oracle_equiv.shape_mismatch",
            "field_path": f"outputs[{output_name}]",
            "message": (
                f"formula output shape {f_val.shape} ≠ oracle output shape {oracle_arr.shape} "
                f"on {test_label} inputs; skipping value comparison"
            ),
            "suggested_fix": "检查 formula 是否与 oracle 产出相同 shape",
        })
        return findings

    # NaN pattern comparison
    f_nan = np.isnan(f_val)
    o_nan = np.isnan(oracle_arr)
    if not np.array_equal(f_nan, o_nan):
        diff_count = int((f_nan != o_nan).sum())
        findings.append({
            "severity": "error",
            "rule_id": "formula_oracle_equiv.nan_pattern_divergence",
            "field_path": f"outputs[{output_name}]",
            "message": (
                f"formula 与 oracle 的 NaN 模式不一致（{test_label} 输入）："
                f"formula NaN {int(f_nan.sum())} 处，oracle NaN {int(o_nan.sum())} 处，"
                f"差异 {diff_count} 处。"
                f"这通常表明 formula 使用了与框架实际实现不同的计算形式"
                f"（如教科书合并形式 vs 增量形式），导致 NaN/inf 传播路径不同。"
            ),
            "suggested_fix": (
                "将 formula 改为与 reference_oracle 实际计算形式一致的表达式"
                "（参考 REQUIREMENTS.md 中的框架实际求值公式，而非教科书数学等价形式）"
            ),
        })
        return findings

    # inf/-inf pattern comparison (only on positions where both are not NaN)
    finite_mask = ~f_nan
    f_inf = np.isinf(f_val) & finite_mask
    o_inf = np.isinf(oracle_arr) & finite_mask
    if not np.array_equal(f_inf, o_inf):
        diff_count = int((f_inf != o_inf).sum())
        findings.append({
            "severity": "error",
            "rule_id": "formula_oracle_equiv.inf_pattern_divergence",
            "field_path": f"outputs[{output_name}]",
            "message": (
                f"formula 与 oracle 的 inf 模式不一致（{test_label} 输入）："
                f"formula inf {int(f_inf.sum())} 处，oracle inf {int(o_inf.sum())} 处，"
                f"差异 {diff_count} 处。"
                f"这通常表明 formula 使用了与框架实际实现不同的计算形式。"
            ),
            "suggested_fix": (
                "将 formula 改为与 reference_oracle 实际计算形式一致的表达式"
            ),
        })
        return findings

    # +0/-0 zero-sign pattern comparison — IEEE 754 distinguishes signed zeros:
    # they compare equal (==) but 1/(+0)=+inf vs 1/(-0)=-inf, and np.signbit
    # reveals the difference. Divergence here often signals a different
    # computation form (e.g. a*x + b*y vs x + (y-x) produce different zero signs
    # on cancellation / underflow paths). Only meaningful for float outputs.
    both_finite = finite_mask & ~f_inf
    if f_val.dtype.kind == "f" and oracle_arr.dtype.kind == "f":
        zero_mask = both_finite & (f_val == 0) & (oracle_arr == 0)
        if zero_mask.any():
            f_zero_sign = np.signbit(f_val[zero_mask])
            o_zero_sign = np.signbit(oracle_arr[zero_mask])
            if not np.array_equal(f_zero_sign, o_zero_sign):
                diff_count = int((f_zero_sign != o_zero_sign).sum())
                findings.append({
                    "severity": "error",
                    "rule_id": "formula_oracle_equiv.zero_sign_divergence",
                    "field_path": f"outputs[{output_name}]",
                    "message": (
                        f"formula 与 oracle 的零符号（+0/-0）模式不一致（{test_label} 输入）："
                        f"零位置中符号差异 {diff_count} 处。"
                        f"IEEE 754 区分 +0/-0（1/(+0)=+inf, 1/(-0)=-inf），符号差异通常表明"
                        f"formula 使用了与框架不同的计算形式"
                        f"（如合并形式 vs 增量形式在抵消/下溢路径上产生不同零符号）。"
                    ),
                    "suggested_fix": (
                        "将 formula 改为与 reference_oracle 实际计算形式一致的表达式"
                        "（参考 REQUIREMENTS §6 核心公式与 code_design.md 实现的逐步骤对应）"
                    ),
                })
                return findings

    # Finite value comparison (on positions where both are finite)
    if both_finite.any():
        f_finite = f_val[both_finite].astype("float64")
        o_finite = oracle_arr[both_finite].astype("float64")
        abs_diff = np.abs(f_finite - o_finite)
        rel_diff = abs_diff / (np.abs(o_finite) + 1e-30)
        max_abs = float(abs_diff.max())
        max_rel = float(rel_diff.max())

        if max_abs > _FINITE_ATOL and max_rel > _FINITE_RTOL:
            findings.append({
                "severity": "error",
                "rule_id": "formula_oracle_equiv.value_divergence",
                "field_path": f"outputs[{output_name}]",
                "message": (
                    f"formula 与 oracle 在有限值上发散（{test_label} 输入）："
                    f"max_abs_diff={max_abs:.2e}（阈值 {_FINITE_ATOL:.0e}），"
                    f"max_rel_diff={max_rel:.2e}（阈值 {_FINITE_RTOL:.0e}）。"
                    f"这通常表明 formula 与 oracle 的计算形式不一致。"
                ),
                "suggested_fix": (
                    "将 formula 改为与 reference_oracle 实际计算形式一致的表达式"
                ),
            })

    return findings


def stage_10(spec: dict) -> tuple[str, list[dict]]:
    """Stage 10 — formula-oracle value equivalence on special-value inputs.

    Runs both the spec formula and the reference oracle on the same inputs
    (including NaN, inf, large values, signed zeros +0/-0) and compares
    outputs. Catches formula form errors that are mathematically equivalent
    but computationally different.

    Skips when:
      * formula_kind != numpy_expr
      * oracle absent
      * framework not installed
      * oracle uses composition mode (DAG)
      * oracle API call fails
    """
    findings: list[dict] = []

    # --- Prerequisites ---

    formula_kind = (spec.get("math_semantics") or {}).get("formula_kind")
    if formula_kind != "numpy_expr":
        return "SKIP", [{
            "severity": "info",
            "rule_id": "formula_oracle_equiv.skipped_non_numpy",
            "field_path": "math_semantics.formula_kind",
            "message": f"formula_kind={formula_kind!r}，stage 10 仅在 numpy_expr 下运行",
            "suggested_fix": None,
        }]

    oracle = (spec.get("math_semantics") or {}).get("reference_oracle") or {}
    absent = bool(oracle.get("absent", False))
    if absent:
        return "SKIP", [{
            "severity": "info",
            "rule_id": "formula_oracle_equiv.absent",
            "field_path": "math_semantics.reference_oracle.absent",
            "message": "oracle 显式声明缺失，stage 10 跳过",
            "suggested_fix": None,
        }]

    # Composition mode — not yet supported for value comparison
    if oracle.get("composition"):
        return "SKIP", [{
            "severity": "info",
            "rule_id": "formula_oracle_equiv.composition_not_supported",
            "field_path": "math_semantics.reference_oracle.composition",
            "message": "oracle 使用 composition (DAG) 模式，stage 10 值等价校验暂不支持",
            "suggested_fix": None,
        }]

    try:
        import numpy as np
    except ImportError:
        return "SKIP", [{
            "severity": "info",
            "rule_id": "formula_oracle_equiv.numpy_not_installed",
            "field_path": "<env>",
            "message": "numpy 未安装；stage 10 跳过",
            "suggested_fix": "pip install numpy",
        }]

    framework = oracle.get("framework")
    api_path = oracle.get("api")
    if not framework or not api_path:
        return "SKIP", [{
            "severity": "info",
            "rule_id": "formula_oracle_equiv.incomplete_oracle",
            "field_path": "math_semantics.reference_oracle",
            "message": "oracle framework 或 api 缺失，stage 10 跳过",
            "suggested_fix": None,
        }]

    # Resolve oracle API (reuse stage 9 logic)
    try:
        api_ok, oracle_api = _resolve_api_callable(
            framework, api_path,
            "math_semantics.reference_oracle.api",
            findings,
        )
    except Exception as ex:
        # Framework import may raise RuntimeError (e.g. torch_npu backend crash)
        # rather than ImportError. Treat as framework-not-available → SKIP.
        findings.append({
            "severity": "info",
            "rule_id": "formula_oracle_equiv.framework_import_error",
            "field_path": "math_semantics.reference_oracle.framework",
            "message": (
                f"framework={framework!r} 导入失败（{type(ex).__name__}）；stage 10 跳过。"
                f"如需值等价校验，请修复框架安装或设置 TORCH_DEVICE_BACKEND_AUTOLOAD=0"
            ),
            "suggested_fix": f"修复 {framework} 安装或换用 numpy oracle",
        })
        return "SKIP", findings
    if not api_ok:
        # If framework not installed → SKIP; if API not found → SKIP (stage 9 already reports)
        if any(f["rule_id"] == "oracle_reachable.framework_not_installed" for f in findings):
            findings.append({
                "severity": "info",
                "rule_id": "formula_oracle_equiv.framework_not_installed",
                "field_path": "math_semantics.reference_oracle.framework",
                "message": f"framework={framework!r} 未安装；stage 10 跳过",
                "suggested_fix": f"pip install {framework}",
            })
            return "SKIP", findings
        # API errors already reported by stage 9; don't duplicate
        return "SKIP", [{
            "severity": "info",
            "rule_id": "formula_oracle_equiv.api_unreachable",
            "field_path": "math_semantics.reference_oracle.api",
            "message": "oracle API 不可达，stage 10 跳过（stage 9 已报错）",
            "suggested_fix": None,
        }]

    # --- Generate test tensors with special values ---

    combos = (spec.get("dtype_policy") or {}).get("supported_combinations") or []
    if not combos:
        return "SKIP", [{
            "severity": "info",
            "rule_id": "formula_oracle_equiv.no_combination",
            "field_path": "dtype_policy.supported_combinations",
            "message": "缺 supported_combinations，stage 10 跳过",
            "suggested_fix": None,
        }]
    in_dtypes = combos[0].get("inputs") or {}

    seed = (spec.get("test_matrix") or {}).get("random", {}).get("seed", 42)
    test_sets = _gen_special_tensors(np, spec, in_dtypes, seed)
    if not test_sets:
        return "SKIP", [{
            "severity": "info",
            "rule_id": "formula_oracle_equiv.no_test_inputs",
            "field_path": "<internal>",
            "message": "无法生成测试输入，stage 10 跳过",
            "suggested_fix": None,
        }]

    # Import the oracle framework
    try:
        framework_mod = __import__(framework)
    except Exception:
        # Covers ImportError and RuntimeError (e.g. torch_npu backend crash)
        return "SKIP", [{
            "severity": "info",
            "rule_id": "formula_oracle_equiv.framework_not_installed",
            "field_path": "math_semantics.reference_oracle.framework",
            "message": f"framework={framework!r} 未安装或导入失败；stage 10 跳过",
            "suggested_fix": f"pip install {framework} 或修复框架安装",
        }]

    # --- Run formula and oracle on each test set ---

    output_names = [out.get("name") for out in (spec.get("outputs") or [])]
    oracle_called = False
    compare_findings: list[dict] = []

    for ts in test_sets:
        label = ts.pop("_label", "unknown")
        input_tensors = {k: v for k, v in ts.items() if k != "_label"}

        # Run formula
        formula_out = run_formula(np, spec, input_tensors)
        if formula_out is None:
            continue

        # Call oracle
        oracle_out = _call_oracle_single_api(
            framework_mod, oracle_api, spec, oracle, input_tensors, np
        )
        if oracle_out is None:
            # Oracle call failed for this test set — try next set
            continue

        oracle_called = True

        # Compare each output
        for out_idx, out_name in enumerate(output_names):
            compare_findings.extend(
                _compare_outputs(formula_out, oracle_out, np, out_name, label, out_idx)
            )

    if not oracle_called:
        findings.append({
            "severity": "info",
            "rule_id": "formula_oracle_equiv.oracle_call_failed",
            "field_path": "math_semantics.reference_oracle",
            "message": (
                "无法在测试输入上成功调用 oracle API"
                "（可能因 API 签名差异）；stage 10 值等价校验跳过"
            ),
            "suggested_fix": (
                "检查 oracle.kwargs 中的占位符是否与 API 签名匹配，"
                "或确认 oracle.api 的参数顺序与 spec.inputs 一致"
            ),
        })
        return "SKIP", findings

    # Deduplicate findings (same rule_id + message may appear across test sets)
    seen = set()
    unique_findings: list[dict] = []
    for f in compare_findings:
        key = (f["rule_id"], f.get("field_path", ""), f.get("message", "")[:100])
        if key not in seen:
            seen.add(key)
            unique_findings.append(f)

    findings.extend(unique_findings)
    status = "FAIL" if any(f["severity"] == "error" for f in findings) else "PASS"
    return status, findings
