# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Stage 8 — formula smoke eval.

Run the spec.math_semantics.formula on tiny tensors (default [2,3]) under a
restricted-AST sandbox. Catches:
  * formula syntax errors (ParseError)
  * misspelled numpy ops (e.g. `x.maks(...)` → AttributeError)
  * shape/axis bugs (broadcast errors, wrong dim)
  * formula vs dtype_rule mismatch (output dtype runtime ≠ DSL推导)
  * unintentional all-NaN outputs (e.g. div-by-zero with neutral inputs)

Skips when:
  * formula_kind != 'numpy_expr' (python_block / textual_only)
  * numpy is not installed (lazy import; report SKIP, not FAIL)

Sandbox policy (defense in depth — assume formula authors are trusted, but block
accidental imports / file IO / infinite loops):
  * AST whitelist: Module, Assign, AugAssign, Expr, BinOp, UnaryOp, Compare,
                   BoolOp, Call, Attribute, Subscript, Name, Constant, Tuple,
                   List, Slice, Load, Store, IfExp, And/Or/Not + numeric ops.
  * Banned: Import, ImportFrom, FunctionDef, Lambda, ClassDef, Global, Nonlocal,
            Try, While, For (use np vectorized ops instead).
  * Restricted globals: only `np`, `math`, plus a small set of allowed builtins
                        (`abs`, `min`, `max`, `range`, `len`, `int`, `float`).
  * SIGALRM-based timeout: 5 s wallclock per formula.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any

from . import _ast_sandbox
from ._ast_sandbox import SandboxError


# Lazy numpy import — kept inside the function so import errors become SKIP
# rather than crashing module load.


class FormulaError(Exception):
    """Raised by the sandbox; carries (code, message) for finding emission."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


def _validate_ast(tree: ast.AST) -> None:
    """共享沙箱校验，failure 转换为 formula_* 错误码。"""
    try:
        _ast_sandbox.validate_ast(tree)
    except SandboxError as e:
        # 把通用沙箱错误码改写为 formula 域错误码（保持向后兼容）
        if e.code == "ast_disallowed":
            raise FormulaError("formula_ast_disallowed", e.message) from None
        if e.code == "banned_name":
            raise FormulaError("formula_banned_name", e.message) from None
        raise FormulaError(f"formula_{e.code}", e.message) from None


def _timeout(seconds: int):
    """5 秒沙箱超时；超时事件转 formula_timeout 错误码。"""
    return _ast_sandbox.timeout(seconds, on_timeout_code="formula_timeout")


# ---------- shape literal ↦ numpy shape tuple ------------------------------


def _resolve_shape(symbolic: list, default_dim: int = 3) -> tuple:
    """Pick a concrete shape for a SymbolicShape list.

    Strategy:
      * folded prefix `...x` → 替换为 1 个 default_dim（保证不退化为 0-D，
        让 stage 8 上 `np.nonzero` / `np.fft` 等不接受 scalar 的 API 也能跑）
      * explicit symbol → default_dim (e.g. M, K, N → 3)
      * integer literal → use as-is
    Yields a small shape (rank ≤ original rank + 0..1, all dims ≤ default_dim).
    """
    out: list[int] = []
    for e in symbolic or []:
        if isinstance(e, int) and not isinstance(e, bool):
            out.append(int(e))
        elif isinstance(e, str):
            if e.startswith("..."):
                # 给折叠维一个 1-D 占位；显式维同时存在时不会让 rank 过深
                out.append(default_dim)
                continue
            out.append(default_dim)
        # other types ignored — caller's stage 1 should have caught them
    return tuple(out)


# ---------- generate test tensors ------------------------------------------


def _gen_tensor(np, shape: tuple, dtype: str, seed: int):
    """Generate a small tensor with the given dtype + shape, deterministic seed.

    Stand-in 套路（numpy 没有原生类型时的兼容方案）：
      * bfloat16 → 用 float32 容器（值范围一致，精度差异由 tolerance 兜底）
      * fp8_e4m3fn / fp8_e5m2 / fp8_e8m0 / hifloat8 → 用 float16 容器
      * fp4_e2m1 / fp4_e1m2 → 用 float16 容器，clamp ±6（fp4 仅 16 个有效值）
      * complex32 → complex64 容器
      * int4 → int8 容器，clamp [-8, 7]
      * uint4 → uint8 容器，clamp [0, 15]
      * uint1 → uint8 容器，clamp [0, 1]
    见 _is_dtype_standin_match 处理 stage 8 dtype mismatch 比对。
    """
    rng = np.random.default_rng(seed)
    # 窄浮点 stand-in（fp16 容器；具体 dtype 真实表示由后端处理）
    if dtype in ("float8_e4m3fn", "float8_e5m2", "float8_e8m0", "hifloat8"):
        return rng.standard_normal(size=shape).astype("float16")
    if dtype in ("float4_e2m1", "float4_e1m2"):
        return (rng.standard_normal(size=shape) * 2.0).clip(-6.0, 6.0).astype("float16")
    if dtype.startswith("float") or dtype == "bfloat16":
        np_dtype = "float32" if dtype == "bfloat16" else dtype
        return rng.standard_normal(size=shape).astype(np_dtype)
    if dtype.startswith("complex"):
        # complex32 没有 numpy 原生，用 complex64 容器
        np_dtype = "complex128" if dtype == "complex128" else "complex64"
        return (rng.standard_normal(size=shape) + 1j *
                rng.standard_normal(size=shape)).astype(np_dtype)
    if dtype == "bool":
        return rng.integers(0, 2, size=shape).astype("bool")
    if dtype == "uint1":
        return rng.integers(0, 2, size=shape).astype("uint8")
    if dtype == "int4":
        return rng.integers(-8, 8, size=shape).astype("int8")
    if dtype == "uint4":
        return rng.integers(0, 16, size=shape).astype("uint8")
    if dtype.startswith("int") or dtype.startswith("uint"):
        return rng.integers(-8, 8, size=shape).astype(dtype)
    raise FormulaError(
        "formula_unsupported_dtype",
        f"stage 8 暂不支持 dtype: {dtype!r}",
    )


# 哪些 stand-in 对在 dtype 比对时算"等价"
_DTYPE_STANDIN_PAIRS = frozenset({
    ("bfloat16", "float32"),  # bf16 模拟为 fp32
    ("float8_e4m3fn", "float16"),
    ("float8_e5m2", "float16"),
    ("float8_e8m0", "float16"),
    ("hifloat8", "float16"),
    ("float4_e2m1", "float16"),
    ("float4_e1m2", "float16"),
    ("complex32", "complex64"),
    ("int4", "int8"),
    ("uint4", "uint8"),
    ("uint1", "uint8"),
})


def _is_dtype_standin_match(spec_dtype: str, runtime_dtype: str) -> bool:
    """spec 声明的 dtype 与 runtime 实跑出的 dtype 是否属于已知 stand-in 对。"""
    return (spec_dtype, runtime_dtype) in _DTYPE_STANDIN_PAIRS


# ---------- main entry -----------------------------------------------------


def _stage8_skip_for_kind(formula_kind):
    if formula_kind == "numpy_expr":
        return None
    return [{
        "severity": "info",
        "rule_id": "formula_smoke_eval.skipped_non_numpy",
        "field_path": "math_semantics.formula_kind",
        "message": f"formula_kind={formula_kind!r}，stage 8 仅在 numpy_expr 下运行",
        "suggested_fix": None,
    }]


def _stage8_import_numpy():
    try:
        import numpy as np
        return np, None
    except ImportError:
        return None, [{
            "severity": "info",
            "rule_id": "formula_smoke_eval.numpy_not_installed",
            "field_path": "<env>",
            "message": "numpy 未安装；stage 8 跳过",
            "suggested_fix": "pip install numpy",
        }]


def _stage8_compile(formula):
    """Parse + AST whitelist + compile. Returns (compiled, findings)."""
    if not formula.strip():
        return None, [{
            "severity": "error",
            "rule_id": "formula_smoke_eval.empty_formula",
            "field_path": "math_semantics.formula",
            "message": "formula 为空但 formula_kind=numpy_expr",
            "suggested_fix": "填写 numpy 可 eval 的表达式或把 formula_kind 改为 textual_only",
        }]
    try:
        tree = ast.parse(formula, mode="exec")
        _validate_ast(tree)
        return compile(tree, "<formula>", "exec"), []
    except SyntaxError as e:
        return None, [{
            "severity": "error",
            "rule_id": "formula_smoke_eval.syntax_error",
            "field_path": "math_semantics.formula",
            "message": f"formula 语法错: {e.msg} (行 {e.lineno})",
            "suggested_fix": "检查表达式语法",
        }]
    except FormulaError as e:
        return None, [{
            "severity": "error",
            "rule_id": f"formula_smoke_eval.{e.code}",
            "field_path": "math_semantics.formula",
            "message": e.message,
            "suggested_fix": "把不允许的语法用 numpy 向量化操作替代",
        }]


def _stage8_build_inputs(spec, np, in_dtypes):
    """Build input tensors per spec.inputs. Returns (tensors, findings)."""
    input_tensors: dict[str, Any] = {}
    findings: list[dict] = []
    seed = (spec.get("test_matrix") or {}).get("random", {}).get("seed", 42)
    for inp in spec.get("inputs") or []:
        name = inp.get("name")
        sym = (inp.get("shape") or {}).get("symbolic", [])
        shape = _resolve_shape(sym)
        dtype = in_dtypes.get(name) or (inp.get("dtype_set") or ["float32"])[0]
        try:
            input_tensors[name] = _gen_tensor(np, shape, dtype, seed)
        except FormulaError as e:
            findings.append({
                "severity": "warning",
                "rule_id": f"formula_smoke_eval.{e.code}",
                "field_path": f"inputs[{name}].dtype_set",
                "message": e.message,
                "suggested_fix": "确认 dtype 在 stage 8 支持范围",
            })
    return input_tensors, findings


def _stage8_exec(compiled, input_tensors, attr_values, np):
    """Exec compiled formula under timeout. Returns (locals_dict, findings)."""
    g: dict[str, Any] = _ast_sandbox.make_globals({"np": np, "math": math})
    g.update(input_tensors)
    g.update(attr_values)
    locals_dict: dict[str, Any] = {}
    try:
        with _timeout(5):
            exec(compiled, g, locals_dict)
        return locals_dict, []
    except FormulaError as e:
        return None, [{
            "severity": "error",
            "rule_id": f"formula_smoke_eval.{e.code}",
            "field_path": "math_semantics.formula",
            "message": e.message,
            "suggested_fix": "拆短 formula；避免大循环",
        }]
    except Exception as e:
        msg = re.sub(r"\s+", " ", str(e)).strip()
        return None, [{
            "severity": "error",
            "rule_id": "formula_smoke_eval.numpy_eval_error",
            "field_path": "math_semantics.formula",
            "message": f"{type(e).__name__}: {msg[:300]}",
            "suggested_fix": "检查 numpy API 名 / 参数 / shape 是否对",
        }]


def _stage8_validate_output(name, locals_dict, expected, np):
    """Validate single output: presence, dtype match, NaN sanity."""
    findings: list[dict] = []
    if name not in locals_dict:
        findings.append({
            "severity": "error",
            "rule_id": "formula_smoke_eval.missing_output",
            "field_path": f"outputs[{name}]",
            "message": f"formula 未给变量 {name!r} 赋值",
            "suggested_fix": f"在 formula 中产出 {name}",
        })
        return findings
    val = locals_dict[name]
    if not isinstance(val, np.ndarray):
        try:
            val = np.asarray(val)
        except Exception:
            findings.append({
                "severity": "warning",
                "rule_id": "formula_smoke_eval.output_not_array",
                "field_path": f"outputs[{name}]",
                "message": f"output {name!r} 不是 ndarray (得到 {type(val).__name__})",
                "suggested_fix": "在 formula 末尾用 np.asarray(...) 包一下",
            })
            return findings

    runtime_dtype = str(val.dtype)
    if expected and not _is_dtype_standin_match(expected, runtime_dtype) and runtime_dtype != expected:
        findings.append({
            "severity": "warning",
            "rule_id": "formula_smoke_eval.dtype_mismatch_at_runtime",
            "field_path": f"outputs[{name}]",
            "message": (
                f"运行时 dtype={runtime_dtype} 与 supported_combinations[0] "
                f"声明的 {expected} 不一致（stage 4 是逻辑校验，stage 8 是运行验证）"
            ),
            "suggested_fix": "检查 formula 是否漏了 dtype cast，或修正 supported_combinations",
        })

    if val.size > 0 and np.issubdtype(val.dtype, np.floating):
        nan_ratio = float(np.isnan(val).sum()) / val.size
        if nan_ratio == 1.0:
            findings.append({
                "severity": "warning",
                "rule_id": "formula_smoke_eval.produces_unexpected_nan",
                "field_path": f"outputs[{name}]",
                "message": (
                    f"formula 在中性输入下产出全 NaN（{name}）；"
                    "可能 div-by-zero 或 max-shift 反向"
                ),
                "suggested_fix": "检查公式是否有未设防的除零 / log(0) 等",
            })
    return findings


def stage_8(spec: dict) -> tuple[str, list[dict]]:
    """Run formula on tiny inputs; report runtime / dtype / NaN issues."""
    formula_kind = (spec.get("math_semantics") or {}).get("formula_kind")
    skip_findings = _stage8_skip_for_kind(formula_kind)
    if skip_findings is not None:
        return "SKIP", skip_findings

    np, np_skip = _stage8_import_numpy()
    if np is None:
        return "SKIP", np_skip

    formula = (spec.get("math_semantics") or {}).get("formula", "")
    compiled, findings = _stage8_compile(formula)
    if compiled is None:
        return "FAIL", findings

    combos = (spec.get("dtype_policy") or {}).get("supported_combinations") or []
    if not combos:
        return "FAIL", [{
            "severity": "error",
            "rule_id": "formula_smoke_eval.no_combination",
            "field_path": "dtype_policy.supported_combinations",
            "message": "缺 supported_combinations，stage 8 无法选择 dtype 跑 formula",
            "suggested_fix": "至少声明一条 supported_combinations",
        }]
    in_dtypes = combos[0].get("inputs") or {}
    expected_outs = combos[0].get("outputs") or {}

    input_tensors, input_findings = _stage8_build_inputs(spec, np, in_dtypes)
    findings.extend(input_findings)

    attr_values = {a["name"]: a["default"] for a in (spec.get("attributes") or []) if "default" in a}

    locals_dict, exec_findings = _stage8_exec(compiled, input_tensors, attr_values, np)
    if locals_dict is None:
        findings.extend(exec_findings)
        return "FAIL", findings

    for out in spec.get("outputs") or []:
        name = out.get("name")
        findings.extend(_stage8_validate_output(name, locals_dict, expected_outs.get(name), np))

    status = "FAIL" if any(f["severity"] == "error" for f in findings) else "PASS"
    return status, findings
