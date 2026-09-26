# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Stage 12 — extreme_inputs × formula × machine_check 联合校验。

背景（特殊值断言与公式矛盾的典型缺陷）：stage 2 只做 machine_check.kind / pattern 的白名单校验，
不检查"声明的期望"与"formula 在该 pattern 输入下的实际输出"是否一致。
典型错例：减法算子把 predict=+inf、label=-inf 断言为 produces_nan
——异号无穷相减按 IEEE 754 应为 +inf，仅同号 inf-inf 与 inf*0 产生 NaN。

本 stage 对 extreme_inputs[] 中可机器合成的条目：
  1. 按 synthesize.shapes + patterns + dtype 合成输入张量（registry 语义）；
  2. 以 synthesize.attrs 覆盖属性默认值，在 AST sandbox 中执行 formula；
  3. 将实际输出的 NaN/±inf/有限值模式与 machine_check.kind 比对：
     - produces_nan  -> 所有输出逐元素必须为 NaN（scope: 整张量）或至少一处（无 scope）
     - nan_propagates-> 至少一个输出位置出现 NaN
     - matches_oracle-> formula 与 reference_oracle 在同一输入上模式一致
                       （oracle 不可达 / kwargs 无法感知条目级 attrs 时降级为 INFO 跳过）
  4. kind 不属于上述集合（equals / raises_error / …）-> INFO 跳过，不判 FAIL。

status：任一 ERR -> FAIL；至少一条机器校验通过且无 ERR -> PASS；否则 SKIP。
"""

from __future__ import annotations

import ast
import importlib
import re

from .formula_eval import run_formula
from .formula_oracle_equiv import _call_oracle_single_api, _compare_outputs
from .oracle_check import _resolve_api_callable

_PATTERN_RE = re.compile(r"^(?P<name>[a-z_]+)(\((?P<value>[^()]*)\))?$")

# 可机器校验的 machine_check kind
_CHECKABLE_KINDS = frozenset({"produces_nan", "nan_propagates", "matches_oracle"})

# 输入合成 dtype stand-in（numpy 无原生容器时）
_DTYPE_STANDIN = {
    "bfloat16": "float32",
    "float8_e4m3fn": "float16",
    "float8_e5m2": "float16",
    "float8_e8m0": "float16",
    "hifloat8": "float16",
    "float4_e2m1": "float16",
    "float4_e1m2": "float16",
    "complex32": "complex64",
}


def _info(rule_id, field_path, message):
    return {"severity": "info", "rule_id": rule_id, "field_path": field_path,
            "message": message, "suggested_fix": None}


def _warn(rule_id, field_path, message):
    return {"severity": "warning", "rule_id": rule_id, "field_path": field_path,
            "message": message, "suggested_fix": None}


def _parse_shape(shape_str):
    """'[2, 3]' / '[1]' / '[]' -> tuple；非法返回 None。"""
    try:
        v = ast.literal_eval(str(shape_str))
    except (ValueError, SyntaxError):
        return None
    if not isinstance(v, (list, tuple)) or not all(isinstance(d, int) for d in v):
        return None
    return tuple(v)


def _entry_dtypes(entry, input_names):
    """解析条目级 dtype（dict 或标量两种写法），缺省 float32。"""
    raw = (entry.get("synthesize") or {}).get("dtype")
    per = {n: "float32" for n in input_names}
    if isinstance(raw, str):
        per = {n: raw for n in input_names}
    elif isinstance(raw, dict):
        for n, dt in raw.items():
            if n in per:
                per[n] = str(dt)
    return per


def _apply_pattern(np, arr, pattern_name, value):
    """按 registry 语义就地应用 pattern；返回 False 表示该 pattern 不可合成。"""
    if pattern_name == "inject_nan_one_element":
        if arr.size:
            arr.flat[0] = np.nan
    elif pattern_name == "single_pos_inf":
        if arr.size:
            arr.flat[0] = np.inf
    elif pattern_name == "single_neg_inf":
        if arr.size:
            arr.flat[0] = -np.inf
    elif pattern_name == "all_pos_inf":
        arr.fill(np.inf)
    elif pattern_name == "all_neg_inf":
        arr.fill(-np.inf)
    elif pattern_name == "all_zero":
        arr.fill(0)
    elif pattern_name == "all_same":
        if value is None:
            return False
        try:
            arr.fill(float(value))
        except (TypeError, ValueError):
            return False
    elif pattern_name == "subnormal_only":
        arr.fill(1e-38)  # float32 subnormal 量级
    elif pattern_name == "min_max_alternate":
        if arr.dtype.kind != "f":
            return False
        fi = np.finfo(arr.dtype.type)
        arr[...] = np.where(np.arange(arr.size) % 2 == 0, fi.max, fi.min).reshape(arr.shape)
    elif pattern_name == "denormal_boundary":
        arr.fill(np.finfo("float32").tiny)
    else:
        return False
    return True


def _make_base_tensors(np, shapes, dtypes, input_names, seed):
    """按 shapes/dtype 生成随机基底张量；返回 (tensors, None) 或 (None, err)。"""
    rng = np.random.default_rng(seed)
    tensors = {}
    for name in input_names:
        shape = _parse_shape(shapes.get(name, "[2]"))
        if shape is None:
            return None, "输入 %s 的 shape 规格 %r 无法解析" % (name, shapes.get(name))
        np_dtype = _DTYPE_STANDIN.get(dtypes[name], dtypes[name])
        try:
            tensors[name] = rng.standard_normal(size=shape).astype(np_dtype)
        except TypeError:
            tensors[name] = rng.standard_normal(size=shape).astype("float32")
    return tensors, None


def _synthesize_inputs(np, entry, input_names, seed):
    """按条目合成全部输入。返回 (tensors, None) 或 (None, 错误信息)。"""
    synth = entry.get("synthesize") or {}
    tensors, err = _make_base_tensors(np, synth.get("shapes") or {},
                                      _entry_dtypes(entry, input_names),
                                      input_names, seed)
    if tensors is None:
        return None, err
    for p in synth.get("patterns") or []:
        if not isinstance(p, dict):
            continue
        target = p.get("target")
        if target not in tensors:
            return None, "pattern 目标 %r 不是声明的输入" % (target,)
        m = _PATTERN_RE.match(str(p.get("pattern", "")))
        if not m:
            return None, "pattern %r 不符合命名规则" % (p.get("pattern"),)
        if not _apply_pattern(np, tensors[target], m.group("name"), m.group("value")):
            return None, "pattern %r 暂不支持机器合成" % (p.get("pattern"),)
    return tensors, None


def _fmt_val(v):
    if v != v:  # NaN
        return "NaN"
    if v == float("inf"):
        return "+inf"
    if v == float("-inf"):
        return "-inf"
    return "%.6g" % v


def _check_produces_nan(ctx, formula_out, whole_tensor):
    """whole_tensor=True（scope: 整张量）要求逐元素 NaN；否则至少一处。返回 findings。"""
    np_ = ctx["np"]
    findings = []
    for out_name, arr in formula_out.items():
        if not arr.size:
            continue
        nan_mask = np_.isnan(arr)
        ok = bool(nan_mask.all()) if whole_tensor else bool(nan_mask.any())
        if ok:
            continue
        findings.append(_produces_nan_finding(ctx, out_name, arr,
                                              nan_mask, whole_tensor))
    return findings


def _produces_nan_finding(ctx, out_name, arr, nan_mask, whole_tensor):
    first = arr.flatten()[0]
    return {
        "severity": "error",
        "rule_id": "extreme_check.produces_nan_conflict",
        "field_path": ctx["where"],
        "message": (
            "条目 %r 声明 produces_nan（%s），但 formula 在该 pattern 输入下"
            "输出 %s 的首元素为 %s（NaN 占比 %d/%d）。"
            "常见根因：混淆同号/异号无穷运算——异号 inf-(−inf)=+inf、"
            "同号 inf-inf=NaN、inf*0=NaN；或声明 scope: 整张量 但 NaN 只按位置传播。"
            "期望须按 IEEE 754 语义与真实传播路径推导，不能照搬模板。"
            % (ctx["case"], "整张量" if whole_tensor else "至少一处", out_name,
               _fmt_val(float(first)), int(nan_mask.sum()), arr.size)
        ),
        "suggested_fix": (
            "按 formula 实际输出修正 machine_check（位置传播改用 nan_propagates，"
            "或去掉 scope: 整张量）；若要测整张量 NaN，输入 pattern 须使全部元素"
            "产生 NaN（同号 inf / inf×0 / 全量注入），并显式固定影响输出的全部自由输入"
        ),
    }


def _check_nan_propagates(ctx, formula_out):
    """声明 nan_propagates 时至少一个输出位置出现 NaN。返回 findings。"""
    if formula_out and not any(bool(ctx["np"].isnan(a).any()) for a in formula_out.values()):
        return [{
            "severity": "error",
            "rule_id": "extreme_check.nan_propagates_violated",
            "field_path": ctx["where"],
            "message": "条目 %r 声明 nan_propagates，但 formula 在该 pattern 输入下"
                       "所有输出均未出现 NaN" % ctx["case"],
            "suggested_fix": "检查 formula 的 NaN 传播路径与输入注入模式是否匹配",
        }]
    return []


def _oracle_kwargs_conflict(oracle, attrs_override):
    """条目级 attrs 是否影响 oracle kwargs（literal 或占位引用）→ 无法安全对拍。"""
    if not attrs_override:
        return False
    kwargs = oracle.get("kwargs") or {}
    attr_names = set(attrs_override)
    for k, v in kwargs.items():
        if k in attr_names:
            return True
        if isinstance(v, str) and any(a in v for a in attr_names):
            return True
    return False


def _resolve_oracle_modules(oracle):
    """解析 oracle API 与框架模块；返回 (callable, fw_mod) 或 (None, None)。"""
    api_ok, oracle_api = _resolve_api_callable(
        oracle.get("framework", ""), oracle.get("api") or "",
        "math_semantics.reference_oracle.api", [])
    if not api_ok:
        return None, None
    try:
        fw_mod = importlib.import_module(oracle.get("framework", "numpy"))
    except Exception:
        return None, None
    return oracle_api, fw_mod


def _remap_compare_findings(ctx, formula_out, oracle_out):
    """将 _compare_outputs 结果改写为 extreme_check 命名空间。"""
    findings = []
    for out in ctx["spec"].get("outputs") or []:
        for f in _compare_outputs(formula_out, oracle_out, ctx["np"], out.get("name"),
                                  "extreme"):
            f["rule_id"] = "extreme_check.%s" % f["rule_id"].split(".")[-1]
            f["field_path"] = ctx["where"]
            f["message"] = "条目 %r：%s" % (ctx["case"], f["message"])
            findings.append(f)
    return findings


def _check_matches_oracle(ctx, tensors, formula_out, attrs_override):
    """matches_oracle：formula 与 oracle 同输入对拍；不可达时 INFO 跳过。返回 findings。"""
    oracle = ctx["oracle"]
    where, case = ctx["where"], ctx["case"]
    if _oracle_kwargs_conflict(oracle, attrs_override):
        return [_info("extreme_check.oracle_skipped_attr_override", where + ".attrs",
                      "条目 %r 的 attrs 覆盖会影响 oracle kwargs 取值，"
                      "无法安全对拍；仅确认 formula 可执行" % case)]
    oracle_api, fw_mod = _resolve_oracle_modules(oracle)
    if oracle_api is None:
        return [_info("extreme_check.oracle_unreachable", where,
                      "oracle 不可达或框架不可用，条目 %r 跳过 formula/oracle 对拍" % case)]
    oracle_out = _call_oracle_single_api(fw_mod, oracle_api, ctx["spec"], oracle,
                                         tensors, ctx["np"])
    if oracle_out is None:
        return [_info("extreme_check.oracle_call_failed", where,
                      "条目 %r oracle 调用失败，跳过对拍" % case)]
    return _remap_compare_findings(ctx, formula_out, oracle_out)


def _run_entry_check(ctx, entry, idx, seed):
    """单条目联合校验。返回 (checked, findings)；checked=False 表示条目被跳过。"""
    ctx = dict(ctx, where="extreme_inputs[%d]" % idx,
               case=str(entry.get("case", "#%d" % idx)))
    kind = ((entry.get("machine_check") or {}).get("kind")) or ""
    if kind not in _CHECKABLE_KINDS:
        return False, [_info("extreme_check.skipped_kind",
                             ctx["where"] + ".machine_check.kind",
                             "kind=%r 暂不支持机器复核，条目 %r 跳过"
                             % (kind, ctx["case"]))]
    input_names = [i.get("name") for i in (ctx["spec"].get("inputs") or [])]
    tensors, err = _synthesize_inputs(ctx["np"], entry, input_names, seed + idx)
    if tensors is None:
        return False, [_warn("extreme_check.unsynthesizable",
                             ctx["where"] + ".synthesize",
                             "条目 %r 输入无法机器合成（%s），跳过联合校验"
                             % (ctx["case"], err))]
    attrs_override = (entry.get("synthesize") or {}).get("attrs") or {}
    formula_out = run_formula(ctx["np"], ctx["spec"], tensors,
                              attr_overrides=attrs_override)
    if formula_out is None:
        return False, [_warn("extreme_check.formula_unexecutable", ctx["where"],
                             "条目 %r 的输入下 formula 执行失败，跳过联合校验"
                             % ctx["case"])]
    if kind == "produces_nan":
        scope = (entry.get("machine_check") or {}).get("scope") or ""
        whole = str(scope).strip() == "整张量"
        return True, _check_produces_nan(ctx, formula_out, whole)
    if kind == "nan_propagates":
        return True, _check_nan_propagates(ctx, formula_out)
    return True, _check_matches_oracle(ctx, tensors, formula_out, attrs_override)


def stage_12(spec):
    """Entry point — 见模块 docstring。返回 (status, findings)。"""
    entries = spec.get("extreme_inputs") or []
    if not entries:
        return "SKIP", [_info("extreme_check.no_entries", "extreme_inputs",
                              "spec 未声明 extreme_inputs，stage 12 跳过")]
    ms = spec.get("math_semantics") or {}
    if ms.get("formula_kind") != "numpy_expr":
        return "SKIP", [_info("extreme_check.skipped_non_numpy",
                              "math_semantics.formula_kind",
                              "formula_kind=%r，stage 12 仅在 numpy_expr 下运行"
                              % (ms.get("formula_kind"),))]
    try:
        import numpy as np
    except ImportError:
        return "SKIP", [_info("extreme_check.numpy_not_installed", "<env>",
                              "numpy 未安装；stage 12 跳过")]

    ctx = {"np": np, "spec": spec, "oracle": ms.get("reference_oracle") or {}}
    seed = (spec.get("test_matrix") or {}).get("random", {}).get("seed", 42)
    checked = 0
    findings = []
    for idx, entry in enumerate(entries):
        ok, fs = _run_entry_check(ctx, entry, idx, seed)
        checked += 1 if ok else 0
        findings.extend(fs)

    if any(f["severity"] == "error" for f in findings):
        return "FAIL", findings
    if checked == 0:
        return "SKIP", findings
    return "PASS", findings
