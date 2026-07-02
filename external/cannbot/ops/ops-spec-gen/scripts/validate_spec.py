#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""spec.yaml validator — full 9-stage L0 校验.

Stage 1: jsonschema validation against schemas/op-spec.json.
Stage 2: category ↔ paradigm consistency, mutual exclusion,
         paradigm internal constraints, and white-list checks for
         primitives.op / invariants.kind / machine_check.kind / synthesize.patterns / error_type.
Stage 3: shape_closure — solve outputs[].shape_rule via shape DSL.
Stage 4: dtype_closure — cross-check dtype_rule with supported_combinations.
Stage 5: broadcast_legality — simulate broadcast.kind/rules against input shapes.
Stage 6: boundary_min_set — per-paradigm minimum case set.
Stage 7: tolerance_coverage — per-dtype tolerance covers output dtypes + tightness.
Stage 8: formula_smoke_eval — run formula on tiny tensors via numpy sandbox.
Stage 9: oracle_reachable — real import framework + walk api attribute chain.

Stage interface contract:
  - status ∈ {PASS, FAIL, SKIP}
  - findings: list of {severity, rule_id, field_path, message, suggested_fix}

Usage:
    python3 validate_spec.py path/to/spec.yaml
    python3 validate_spec.py path/to/spec.yaml --json     # machine-readable output
    python3 validate_spec.py path/to/spec.yaml --strict   # exit non-zero on warnings too
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

try:
    import jsonschema
    from jsonschema import Draft202012Validator
except ImportError:
    print("ERROR: jsonschema not installed. Run: pip install jsonschema pyyaml", file=sys.stderr)
    sys.exit(2)

SKILL_DIR = Path(__file__).resolve().parent.parent
REGISTRIES = SKILL_DIR / "registries"
SCHEMA_PATH = SKILL_DIR / "schemas" / "op-spec.json"


@dataclass
class Finding:
    severity: str       # error | warning | info
    rule_id: str
    field_path: str
    message: str
    suggested_fix: str | None = None


@dataclass
class StageResult:
    stage_id: int
    status: str         # PASS | FAIL | SKIP
    findings: list[Finding] = field(default_factory=list)


def _load_yaml(p: Path) -> dict:
    """Load a YAML file. Raises yaml.YAMLError on parse error (caller decides exit code)."""
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get(d: dict, *path, default=None):
    cur: Any = d
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def stage_1(spec: dict) -> StageResult:
    """Validate against op-spec.json (Draft 2020-12)."""
    res = StageResult(stage_id=1, status="PASS")
    schema = _load_schema()
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(spec), key=lambda e: list(e.absolute_path))
    for err in errors:
        path = "/".join(str(p) for p in err.absolute_path) or "<root>"
        res.findings.append(Finding(
            severity="error",
            rule_id="schema_static",
            field_path=path,
            message=err.message,
            suggested_fix=None,
        ))
    if res.findings:
        res.status = "FAIL"
    return res


@functools.lru_cache(maxsize=1)
def _load_schema() -> dict:
    """Cached JSON Schema; deserialized once per process."""
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def stage_2(spec: dict, registries: dict) -> StageResult:
    """category ↔ paradigm consistency + paradigm internal constraints + whitelists.

    Dispatches to 6 focused checks; each appends to res.findings. Overall FAIL if
    any error-severity finding is produced. Sub-checks correspond to the design
    doc's lettered sub-rule groups (A/B/C/D/E/F + the H 'hint' group).
    """
    res = StageResult(stage_id=2, status="PASS")
    cat = _get(spec, "op", "category")
    paradigms = set(_get(spec, "op", "paradigms", default=[]) or [])

    _check_category_paradigm_consistency(cat, paradigms, registries, res.findings)
    _check_mutually_exclusive_paradigms(paradigms, registries, res.findings)
    _check_paradigm_internal_constraints(spec, paradigms, registries, res.findings)
    _check_invariants(spec, registries, res.findings)
    _check_machine_check(spec, registries, res.findings)
    _check_synthesize_patterns(spec, registries, res.findings)
    _check_supported_chips(spec, registries, res.findings)
    _check_anti_pattern_ids(spec, registries, res.findings)
    _check_op_error_codes(spec, registries, res.findings)
    _check_composition_hint(spec, paradigms, res.findings)

    if any(f.severity == "error" for f in res.findings):
        res.status = "FAIL"
    return res


# ---------- stage 2 sub-checks --------------------------------------------


def _check_category_paradigm_consistency(cat, paradigms, registries, findings):
    """A. category 必含 paradigm 是否齐全；fused_composite 额外要求 ≥ 2 条基础 paradigm。"""
    cmap = registries["category_map"]
    required = set(cmap["category_requires_paradigms"].get(cat, []))
    missing = required - paradigms
    if missing:
        findings.append(Finding(
            severity="error",
            rule_id="category_paradigm_consistency.required_paradigm_missing",
            field_path="op.paradigms",
            message=f"category={cat} 必含 paradigm {sorted(missing)}",
            suggested_fix=f"在 op.paradigms 中加入 {sorted(missing)}",
        ))

    if cat == "fused_composite":
        basic = {"Elementwise", "Broadcast", "Reduction", "Contraction",
                 "ArgReduce", "LayoutTransform"}
        if len(paradigms & basic) < 2:
            findings.append(Finding(
                severity="error",
                rule_id="category_paradigm_consistency.fused_composite_basics",
                field_path="op.paradigms",
                message="category=fused_composite 必须含 ≥ 2 条基础 paradigm",
                suggested_fix=f"从 {sorted(basic)} 中至少选 2 条加入 paradigms",
            ))


def _check_mutually_exclusive_paradigms(paradigms, registries, findings):
    """B. paradigm 互斥对（同时出现 ⇒ FAIL）。"""
    cmap = registries["category_map"]
    for pair in cmap.get("mutually_exclusive_paradigms", []):
        if set(pair) <= paradigms:
            findings.append(Finding(
                severity="error",
                rule_id="category_paradigm_consistency.mutually_exclusive",
                field_path="op.paradigms",
                message=f"paradigms 互斥：{pair}",
                suggested_fix=f"删除 {pair} 之一；写冲突另成 atomic_update",
            ))


def _check_paradigm_numerical_stable(spec, findings):
    if not _get(spec, "numerical_stability", "required"):
        findings.append(Finding(
            severity="error",
            rule_id="paradigm_constraint.numerical_stable",
            field_path="numerical_stability.required",
            message="paradigms 含 NumericalStable ⇒ numerical_stability.required 必须为 true",
            suggested_fix="把 numerical_stability.required 改为 true 并补 techniques",
        ))


def _check_paradigm_reduction(spec, findings):
    attrs = spec.get("attributes") or []
    axis_names = {a.get("name") for a in attrs}
    if not (axis_names & {"axis", "dim", "axes"}):
        findings.append(Finding(
            severity="error",
            rule_id="paradigm_constraint.reduction_axis_missing",
            field_path="attributes",
            message="paradigms 含 Reduction（且不含 Recurrence）⇒ 必须有 axis/dim/axes 属性",
            suggested_fix="在 attributes 中添加 axis/dim/axes 之一",
        ))


def _check_paradigm_argreduce(spec, findings):
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from evaluators import dtype_eval as _de
        from evaluators.types import DslError as _DslError
    except ImportError:
        _de = None  # type: ignore
    for i, out in enumerate(spec.get("outputs", [])):
        rule = (out.get("dtype_rule") or "").strip()
        kind = out.get("dtype_rule_kind", "numpy_expr")
        derived: str | None = None
        if _de is not None and kind == "numpy_expr" and rule:
            combos = ((spec.get("dtype_policy") or {})
                      .get("supported_combinations") or [])
            in_dt = combos[0].get("inputs", {}) if combos else {}
            try:
                derived = _de.evaluate_dtype_rule(
                    rule, output_name=out.get("name"),
                    input_dtypes=in_dt,
                )
            except _DslError:
                derived = None
        if derived not in ("int32", "int64"):
            findings.append(Finding(
                severity="error",
                rule_id="paradigm_constraint.argreduce_dtype",
                field_path=f"outputs[{i}].dtype_rule",
                message=f"ArgReduce 输出 dtype 必须为 int32 / int64，当前 dtype_rule={rule!r} 推得 {derived!r}",
                suggested_fix="把 dtype_rule 改为 `<output>.dtype = np.int32` 或 `np.int64`",
            ))


def _check_paradigm_stateful(spec, findings):
    has_state_input = any(inp.get("role") == "state" for inp in spec.get("inputs", []))
    has_inplace_output = any(
        re.match(r"^inplace_with\(", out.get("aliasing") or "")
        for out in spec.get("outputs", [])
    )
    if not (has_state_input or has_inplace_output):
        findings.append(Finding(
            severity="error",
            rule_id="paradigm_constraint.stateful_state_or_inplace",
            field_path="inputs[*].role | outputs[*].aliasing",
            message="Stateful 必须有 inputs.role=state 或 outputs.aliasing=inplace_with(<state_input>)",
            suggested_fix="添加 role: state 的 input，或在某 output 加 aliasing: inplace_with(<state_input>)",
        ))


_QUANT_CANONICAL_FORMS = [
    {"scale", "zero_point"},
    {"scale"},
    {"x1Scale", "x2Scale"},
    {"dequant_scale"},
    {"deqScale"},
]


def _check_paradigm_quantization(spec, findings):
    attr_names = {a.get("name") for a in (spec.get("attributes") or [])}
    input_names = {i.get("name") for i in (spec.get("inputs") or [])}
    all_names = attr_names | input_names
    if not any(form <= all_names for form in _QUANT_CANONICAL_FORMS):
        findings.append(Finding(
            severity="error",
            rule_id="paradigm_constraint.quantization_attrs",
            field_path="attributes",
            message=(
                "Quantization 必须含以下任一组量化参数（attribute 或 input 均可）："
                "{scale, zero_point} / {scale} / {x1Scale, x2Scale} / "
                "{dequant_scale} / {deqScale}"
            ),
            suggested_fix="按算子语义选择对应形式；对称量化只需 scale；双输入分别量化用 x1Scale + x2Scale",
        ))


def _check_paradigm_variable_output(spec, findings):
    for i, out in enumerate(spec.get("outputs", [])):
        if not out.get("data_dependent_shape"):
            findings.append(Finding(
                severity="error",
                rule_id="paradigm_constraint.variable_output_flag",
                field_path=f"outputs[{i}].data_dependent_shape",
                message="VariableOutput 输出必须配 data_dependent_shape: true",
                suggested_fix="在该 output 加 data_dependent_shape: true",
            ))


def _check_paradigm_random_sampling(spec, findings):
    attr_names = {a.get("name") for a in (spec.get("attributes") or [])}
    if "seed" not in attr_names:
        findings.append(Finding(
            severity="error",
            rule_id="paradigm_constraint.random_sampling_seed",
            field_path="attributes",
            message="RandomSampling 必须含 seed 属性",
            suggested_fix="在 attributes 中添加 seed (int64)",
        ))


def _check_paradigm_collective(spec, findings):
    if not _get(spec, "op", "platform_constraints", "requires_hccl"):
        findings.append(Finding(
            severity="error",
            rule_id="paradigm_constraint.collective_hccl",
            field_path="op.platform_constraints.requires_hccl",
            message="CollectiveCommunication 必须声明 op.platform_constraints.requires_hccl: true",
            suggested_fix="在 op 下添加 platform_constraints: {requires_hccl: true}",
        ))


def _check_paradigm_internal_constraints(spec, paradigms, registries, findings):
    """C. 每个 paradigm 的内部结构性要求。

    本函数是该规则的**真值**（category_paradigm_map.yaml 的 paradigm_constraints
    段已删除，因为通用 mini-DSL 表达力不足，且会让维护者误以为改 yaml 即可改逻辑）。
    """
    primitives_wl = set(registries["primitives"])

    if "NumericalStable" in paradigms:
        _check_paradigm_numerical_stable(spec, findings)
    if "FusedComposite" in paradigms:
        _check_fused_composite(spec, primitives_wl, findings)
    if "Reduction" in paradigms and "Recurrence" not in paradigms:
        _check_paradigm_reduction(spec, findings)
    if "ArgReduce" in paradigms:
        _check_paradigm_argreduce(spec, findings)
    if "Stateful" in paradigms:
        _check_paradigm_stateful(spec, findings)
    if "Quantization" in paradigms:
        _check_paradigm_quantization(spec, findings)
    if "VariableOutput" in paradigms:
        _check_paradigm_variable_output(spec, findings)
    if "RandomSampling" in paradigms:
        _check_paradigm_random_sampling(spec, findings)
    if "CollectiveCommunication" in paradigms:
        _check_paradigm_collective(spec, findings)


def _check_fused_primitives(prims, primitives_wl, findings):
    if len(prims) < 2:
        findings.append(Finding(
            severity="error",
            rule_id="paradigm_constraint.fused_composite_min_primitives",
            field_path="math_semantics.composition.primitives",
            message=f"FusedComposite 要求 primitives 长度 ≥ 2（当前 {len(prims)}）",
            suggested_fix="拆分成至少 2 条白名单内原语",
        ))
    for i, prim in enumerate(prims):
        if prim.get("op") not in primitives_wl:
            findings.append(Finding(
                severity="error",
                rule_id="paradigm_constraint.primitive_not_whitelisted",
                field_path=f"math_semantics.composition.primitives[{i}].op",
                message=f"原语 op={prim.get('op')!r} 不在 PRIMITIVE_WHITELIST",
                suggested_fix=f"改为白名单内之一: {sorted(primitives_wl)}",
            ))


def _check_fused_dataflow(spec, comp, prims, findings):
    df = comp.get("dataflow") or {}
    if df.get("no_leak"):
        output_names = {o["name"] for o in spec.get("outputs", [])}
        leaked = set(df.get("intermediates") or []) & output_names
        if leaked:
            findings.append(Finding(
                severity="error",
                rule_id="paradigm_constraint.intermediate_leak",
                field_path="math_semantics.composition.dataflow.intermediates",
                message=f"中间 tensor 泄漏到 outputs: {sorted(leaked)}",
                suggested_fix="把这些名字从 intermediates 移除或重命名 outputs",
            ))

    produced = {inp["name"] for inp in spec.get("inputs", [])}
    for i, prim in enumerate(prims):
        for x in (prim.get("inputs") or []):
            if x not in produced:
                findings.append(Finding(
                    severity="error",
                    rule_id="paradigm_constraint.dataflow_unclosed",
                    field_path=f"math_semantics.composition.primitives[{i}].inputs",
                    message=f"未定义的中间 tensor: {x!r}",
                    suggested_fix=f"先在前序 primitive 的 outputs 中产出 {x}",
                ))
        produced.update(prim.get("outputs") or [])


def _check_fused_composite(spec, primitives_wl, findings):
    """FusedComposite 子项：composition 必填、primitives ≥ 2、白名单、不泄漏、闭合。"""
    comp = _get(spec, "math_semantics", "composition")
    if comp is None:
        findings.append(Finding(
            severity="error",
            rule_id="paradigm_constraint.fused_composite_composition_missing",
            field_path="math_semantics.composition",
            message="paradigms 含 FusedComposite ⇒ math_semantics.composition 必填",
            suggested_fix="添加 composition.primitives (≥2 条) 与 dataflow",
        ))
        return

    prims = comp.get("primitives", []) or []
    _check_fused_primitives(prims, primitives_wl, findings)
    _check_fused_dataflow(spec, comp, prims, findings)


def _check_invariants(spec, registries, findings):
    """D. invariants[].kind 白名单 + required_fields + tolerance_inherit 约束。"""
    invariant_kinds = registries["invariant_kinds"]
    invariants = _get(spec, "math_semantics", "invariants", default=[]) or []
    for i, inv in enumerate(invariants):
        kind = inv.get("kind")
        info = invariant_kinds.get(kind)
        if info is None:
            findings.append(Finding(
                severity="error",
                rule_id="invariant_kind_resolved.unknown_kind",
                field_path=f"math_semantics.invariants[{i}].kind",
                message=f"未知 invariant kind: {kind!r}",
                suggested_fix="使用 registries/invariant_kind_registry.yaml 中的 kind",
            ))
            continue
        for f in info.get("required_fields", []):
            if f not in inv:
                findings.append(Finding(
                    severity="error",
                    rule_id="invariant_kind_resolved.missing_field",
                    field_path=f"math_semantics.invariants[{i}].{f}",
                    message=f"invariant kind={kind} 缺必填字段 {f!r}",
                    suggested_fix=f"补充 {f}",
                ))
        if info.get("forbid_tolerance_inherit") and inv.get("tolerance_inherit") is True:
            findings.append(Finding(
                severity="error",
                rule_id="invariant_kind_resolved.tolerance_inherit_forbidden",
                field_path=f"math_semantics.invariants[{i}].tolerance_inherit",
                message=f"kind={kind} 不允许 tolerance_inherit: true（结构性/离散）",
                suggested_fix="改为 false 或删除该字段",
            ))
        if info.get("require_tolerance_inherit") and not inv.get("tolerance_inherit"):
            findings.append(Finding(
                severity="warning",
                rule_id="invariant_kind_resolved.tolerance_inherit_required",
                field_path=f"math_semantics.invariants[{i}].tolerance_inherit",
                message=f"kind={kind} 建议 tolerance_inherit: true（浮点累加）",
                suggested_fix="添加 tolerance_inherit: true",
            ))


def _check_machine_check(spec, registries, findings):
    """E. boundary/extreme 的 machine_check.kind 白名单 + raises_error.error_type 枚举。"""
    machine_check_kinds = set(registries["machine_check_kinds"])
    error_codes = set(registries["error_codes"])
    for sect in ("boundary_conditions", "extreme_inputs"):
        for i, c in enumerate(spec.get(sect) or []):
            mc = c.get("machine_check") or {}
            kind = mc.get("kind")
            if kind not in machine_check_kinds:
                findings.append(Finding(
                    severity="error",
                    rule_id="machine_check_kind_unknown",
                    field_path=f"{sect}[{i}].machine_check.kind",
                    message=f"未知 machine_check.kind: {kind!r}",
                    suggested_fix="使用 registries/machine_check_kind_registry.yaml 中的 kind",
                ))
            elif kind == "raises_error":
                etype = mc.get("error_type")
                if etype not in error_codes:
                    findings.append(Finding(
                        severity="error",
                        rule_id="error_type_unknown",
                        field_path=f"{sect}[{i}].machine_check.error_type",
                        message=f"未知 error_type: {etype!r}",
                        suggested_fix=f"使用 registries/error_code_enum.yaml 中的枚举值",
                    ))


_PATTERN_RE = re.compile(r"^(?P<name>[a-z_]+)(\([^()]*\))?$")


def _check_synthesize_patterns(spec, registries, findings):
    """F. extreme_inputs.synthesize.patterns[].pattern 白名单（兼容旧 pattern: 顶层写法）。"""
    pattern_names = set(registries["patterns"])
    for i, c in enumerate(spec.get("extreme_inputs") or []):
        synth = c.get("synthesize") or {}
        plist: list[tuple[int | None, str | None]] = []
        if "patterns" in synth and isinstance(synth["patterns"], list):
            plist = [(idx, p.get("pattern")) for idx, p in enumerate(synth["patterns"])]
        elif "pattern" in synth:
            plist = [(None, synth["pattern"])]
            findings.append(Finding(
                severity="warning",
                rule_id="synthesize_legacy_format",
                field_path=f"extreme_inputs[{i}].synthesize",
                message="旧 'pattern:' 顶层写法已过时（v1 兼容期 90 天），建议迁移到 'patterns: [{pattern, target}]'",
                suggested_fix="改为 patterns: [{pattern: <name>, target: <input_name>}, ...]",
            ))
        for idx, pname in plist:
            if not pname:
                continue
            m = _PATTERN_RE.match(pname)
            if not m or m.group("name") not in pattern_names:
                fp = (f"extreme_inputs[{i}].synthesize.patterns[{idx}].pattern"
                      if idx is not None else f"extreme_inputs[{i}].synthesize.pattern")
                findings.append(Finding(
                    severity="error",
                    rule_id="synthesize_pattern_unknown",
                    field_path=fp,
                    message=f"未知 pattern: {pname!r}",
                    suggested_fix=f"使用 registries/synthesize_pattern_registry.yaml 中的 name",
                ))


def _check_supported_chips(spec, registries, findings):
    """G. 平台兼容性校验。

    若 spec 声明 supported_chips（可选），其中任一芯片必须支持所有 inputs.dtype_set
    + supported_combinations 涉及的 dtype。
    """
    chip_ids = _get(spec, "op", "platform_constraints", "supported_chips", default=None)
    if not chip_ids:
        return
    chips = registries.get("chips") or {}
    if not chips:
        return  # registry 缺失走 schema-only 校验，跳过此项

    # 收集 spec 涉及的所有 dtype
    declared_dtypes: set = set()
    for inp in spec.get("inputs") or []:
        declared_dtypes.update(inp.get("dtype_set") or [])
    for combo in (spec.get("dtype_policy") or {}).get("supported_combinations") or []:
        for d in (combo.get("inputs") or {}).values():
            declared_dtypes.add(d)
        for d in (combo.get("outputs") or {}).values():
            declared_dtypes.add(d)

    for chip_id in chip_ids:
        chip = chips.get(chip_id)
        if chip is None:
            findings.append(Finding(
                severity="error",
                rule_id="paradigm_constraint.unknown_chip",
                field_path="op.platform_constraints.supported_chips",
                message=f"未注册的芯片 id: {chip_id!r}",
                suggested_fix="在 registries/chip_registry.yaml 登记，或修正拼写",
            ))
            continue
        unsupported = declared_dtypes - chip["dtypes"]
        if unsupported:
            findings.append(Finding(
                severity="error",
                rule_id="paradigm_constraint.dtype_chip_mismatch",
                field_path="op.platform_constraints.supported_chips",
                message=(
                    f"芯片 {chip_id!r} 不支持以下 dtype: {sorted(unsupported)}；"
                    f"已知该芯片支持: {sorted(chip['dtypes'])}"
                ),
                suggested_fix=(
                    "从 supported_chips 移除不兼容的型号，或从 dtype_set / "
                    "supported_combinations 移除该芯片不支持的 dtype"
                ),
            ))


def _check_anti_pattern_ids(spec, registries, findings):
    """I. numerical_stability.techniques[].anti_pattern_id 必须在 anti_pattern_registry 中。

    schema 已用 ^AP-\\d{3}$ 卡格式；本检查锁定**有效编号集合**，避免 AP-999 / AP-001
    （未分配）混入 spec。
    """
    valid = registries.get("anti_patterns") or set()
    if not valid:
        return  # registry 缺失走 schema-only 校验，跳过此项
    techniques = _get(spec, "numerical_stability", "techniques", default=[]) or []
    for i, tech in enumerate(techniques):
        ap_id = tech.get("anti_pattern_id")
        if ap_id is None:
            continue
        if ap_id not in valid:
            findings.append(Finding(
                severity="error",
                rule_id="paradigm_constraint.unknown_anti_pattern",
                field_path=f"numerical_stability.techniques[{i}].anti_pattern_id",
                message=(
                    f"未注册的 anti_pattern_id: {ap_id!r}；已知 IDs={sorted(valid)}"
                ),
                suggested_fix=(
                    "在 registries/anti_pattern_registry.yaml 中登记新编号，或修正拼写"
                ),
            ))


def _check_op_error_codes(spec, registries, findings):
    """J. boundary_conditions[].machine_check.error_type 必须在 op.error_codes 全集内。

    若 spec 没声明 op.error_codes 但在 boundary/extreme 中使用了 raises_error，
    报 error 强制声明（error_codes 是算子契约的一部分，不是可选描述）。
    """
    declared_raw = _get(spec, "op", "error_codes", default=None)
    declared_set = set(declared_raw or [])

    raises_cases: list[tuple[str, int, str]] = []
    for sect in ("boundary_conditions", "extreme_inputs"):
        for i, case in enumerate(spec.get(sect) or []):
            mc = case.get("machine_check") or {}
            if mc.get("kind") != "raises_error":
                continue
            etype = mc.get("error_type")
            if etype:
                raises_cases.append((sect, i, etype))

    if not declared_set and raises_cases:
        etypes_used = sorted({e for _, _, e in raises_cases})
        findings.append(Finding(
            severity="error",
            rule_id="paradigm_constraint.error_codes_undeclared",
            field_path="op.error_codes",
            message=(
                f"boundary/extreme 中使用了 raises_error (error_type={etypes_used}) "
                f"但 op.error_codes 未声明；error_codes 是算子契约，必须显式列全集"
            ),
            suggested_fix=f"在 op 块下添加 error_codes: {etypes_used}",
        ))
        return

    for sect, i, etype in raises_cases:
        if etype not in declared_set:
            findings.append(Finding(
                severity="error",
                rule_id="paradigm_constraint.error_code_not_declared",
                field_path=f"{sect}[{i}].machine_check.error_type",
                message=(
                    f"error_type={etype!r} 不在 op.error_codes 声明的全集 "
                    f"{sorted(declared_set)} 中"
                ),
                suggested_fix="把该 error_type 加到 op.error_codes，或修正本 case",
            ))


def _check_composition_hint(spec, paradigms, findings):
    """H. composition 存在但 paradigms 缺 FusedComposite ⇒ WARN（不算 FAIL）。"""
    has_comp = _get(spec, "math_semantics", "composition") is not None
    if not has_comp or "FusedComposite" in paradigms:
        return
    prims = _get(spec, "math_semantics", "composition", "primitives", default=[]) or []
    if len(prims) >= 2:
        findings.append(Finding(
            severity="warning",
            rule_id="composition_without_fused_composite",
            field_path="op.paradigms",
            message="composition.primitives ≥ 2 但 paradigms 不含 FusedComposite，建议补 FusedComposite",
            suggested_fix="在 op.paradigms 添加 FusedComposite",
        ))
    elif len(prims) == 1:
        findings.append(Finding(
            severity="warning",
            rule_id="composition_single_primitive",
            field_path="math_semantics.composition",
            message="composition.primitives 只有 1 条，建议删除 composition 段",
            suggested_fix="删除 math_semantics.composition",
        ))


def _stage6_check_paradigm_req(req, paradigm, boundary_cases, extreme_cases, res):
    req_id = req.get("id")
    summary = req.get("summary", req_id)
    if "special_check" in req:
        return
    section = req.get("section", "boundary_conditions")
    cases = boundary_cases if section == "boundary_conditions" else extreme_cases
    text = " | ".join(str(c.get("case", "")) for c in cases)
    tags: set = set()
    for c in cases:
        for t in (c.get("tags") or []):
            tags.add(str(t))
    keywords = req.get("match_any", []) or []
    if req_id in tags:
        return
    if any(kw in text for kw in keywords):
        return
    res.findings.append(Finding(
        severity="error",
        rule_id="boundary_min_set.missing_required_case",
        field_path=f"{section}",
        message=f"paradigm={paradigm} 缺必含 case: {summary} (id={req_id})",
        suggested_fix=(
            f"在 {section} 中添加一条 case，加 `tags: [{req_id}]` "
            f"或描述包含以下任一关键词：{keywords}"
        ),
    ))


def stage_6(spec: dict, registries: dict) -> StageResult:
    """boundary_min_set — 按 paradigms 检查 spec 是否覆盖各范式必含的最低 case 集。

    数据源：registries/boundary_min_cases.yaml；显式 tag 命中或关键词子串兜底命中即算覆盖。
    """
    res = StageResult(stage_id=6, status="PASS")
    paradigms = set(_get(spec, "op", "paradigms", default=[]) or [])
    requirements = (registries.get("boundary_min_cases") or {}).get("paradigm_requirements", {})
    boundary_cases = spec.get("boundary_conditions") or []
    extreme_cases = spec.get("extreme_inputs") or []

    for paradigm in sorted(paradigms):
        for req in requirements.get(paradigm, []) or []:
            _stage6_check_paradigm_req(req, paradigm, boundary_cases, extreme_cases, res)

    if any(f.severity == "error" for f in res.findings):
        res.status = "FAIL"
    return res


# 容差紧度阈值从 registries/tolerance_defaults.yaml 加载（与 generator 共享数据源）
@functools.lru_cache(maxsize=1)
def _tolerance_tight_threshold() -> dict[str, float]:
    data = yaml.safe_load((REGISTRIES / "tolerance_defaults.yaml").read_text(encoding="utf-8"))
    return data.get("tightness_threshold") or {}


def stage_7(spec: dict) -> StageResult:
    """tolerance_coverage — numerical_tolerance.per_dtype 是否覆盖所有声明的输出 dtype。

    与 stage 4 (dtype DSL 求解) 完整版的差别：本 stage 用 dtype_policy.supported_combinations
    显式枚举的输出 dtype 作为代理，stage 4 实现后会改为用 DSL 推导结果。两者通常一致。

    额外做"容差过紧"启发式检查 (WARN)：rtol 显著低于 dtype 单步舍入精度时提示。
    """
    res = StageResult(stage_id=7, status="PASS")

    declared_output_dtypes: set[str] = set()
    for combo in _get(spec, "dtype_policy", "supported_combinations", default=[]) or []:
        for d in (combo.get("outputs") or {}).values():
            declared_output_dtypes.add(d)

    per_dtype = _get(spec, "numerical_tolerance", "per_dtype", default={}) or {}
    declared_tol_dtypes = set(per_dtype.keys())

    uncovered = declared_output_dtypes - declared_tol_dtypes
    for d in sorted(uncovered):
        res.findings.append(Finding(
            severity="error",
            rule_id="tolerance_coverage.uncovered_output_dtype",
            field_path=f"numerical_tolerance.per_dtype.{d}",
            message=f"输出 dtype {d!r} 未在 numerical_tolerance.per_dtype 中声明容差",
            suggested_fix=f"为 {d} 补充 {{rtol, atol, metric}}",
        ))

    # 紧度提示（启发式 WARN）
    thresholds = _tolerance_tight_threshold()
    for d, entry in per_dtype.items():
        threshold = thresholds.get(d)
        if threshold is None:
            continue
        rtol = entry.get("rtol")
        if isinstance(rtol, (int, float)) and 0 < rtol < threshold:
            res.findings.append(Finding(
                severity="warning",
                rule_id="tolerance_coverage.tolerance_too_tight",
                field_path=f"numerical_tolerance.per_dtype.{d}.rtol",
                message=(
                    f"{d} rtol={rtol:.1e} 低于该 dtype 单步舍入量级 ({threshold:.0e})；"
                    "大量累加后实际不可能达到，可能导致 PR 误报"
                ),
                suggested_fix=f"放宽 {d} 的 rtol 到至少 {threshold:.0e}",
            ))

    if any(f.severity == "error" for f in res.findings):
        res.status = "FAIL"
    return res


@functools.lru_cache(maxsize=1)
def load_registries() -> dict:
    """Load all registries from disk; cached per-process."""
    inv = {}
    inv_data = yaml.safe_load((REGISTRIES / "invariant_kind_registry.yaml").read_text(encoding="utf-8"))
    for group in ("value", "algebraic", "structural"):
        for entry in inv_data.get(group, []) or []:
            inv[entry["kind"]] = entry

    return {
        "primitives": yaml.safe_load((REGISTRIES / "primitive_whitelist.yaml").read_text(encoding="utf-8"))["primitives"],
        "invariant_kinds": inv,
        "machine_check_kinds": [
            x["kind"] for x in
            yaml.safe_load((REGISTRIES / "machine_check_kind_registry.yaml").read_text(encoding="utf-8"))["machine_check_kinds"]
        ],
        "patterns": [
            x["name"] for x in
            yaml.safe_load((REGISTRIES / "synthesize_pattern_registry.yaml").read_text(encoding="utf-8"))["patterns"]
        ],
        "error_codes": yaml.safe_load((REGISTRIES / "error_code_enum.yaml").read_text(encoding="utf-8"))["error_codes"],
        "category_map": yaml.safe_load((REGISTRIES / "category_paradigm_map.yaml").read_text(encoding="utf-8")),
        "boundary_min_cases": yaml.safe_load((REGISTRIES / "boundary_min_cases.yaml").read_text(encoding="utf-8")),
        "chips": _load_chip_registry(),
        "anti_patterns": _load_anti_pattern_registry(),
    }


def _load_anti_pattern_registry() -> set:
    """加载 anti_pattern_registry.yaml；返回 {AP-XXX, ...}。"""
    path = REGISTRIES / "anti_pattern_registry.yaml"
    if not path.exists():
        return set()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {x["id"] for x in data.get("anti_patterns") or []}


def _load_chip_registry() -> dict:
    """加载 chip_registry.yaml；返回 {chip_id: {dtypes: set}}。"""
    path = REGISTRIES / "chip_registry.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out = {}
    for c in data.get("chips") or []:
        out[c["id"]] = {
            "dtypes": set(c.get("supported_dtypes") or []),
        }
    return out


def _run_eval_stages(spec: dict) -> tuple[StageResult, ...]:
    """Lazily import the evaluators package; if it fails, return five SKIP stages.

    Returns five StageResult objects in order: stage 3, 4, 5, 8, 9.
    Stage 8 / 9 may individually SKIP based on env (numpy/framework not installed)
    even when the evaluators package itself imports fine.
    """
    try:
        from evaluators import stages as eval_stages
        from evaluators import formula_eval, oracle_check
    except ImportError as e:
        skip_finding = Finding(
            severity="info",
            rule_id="stage_skipped",
            field_path="<evaluators import>",
            message=f"evaluators 子包不可用，stage 3-5/8/9 跳过：{e}",
            suggested_fix="确认 scripts/evaluators/ 子包文件齐全",
        )
        return tuple(
            StageResult(stage_id=sid, status="SKIP", findings=[skip_finding])
            for sid in (3, 4, 5, 8, 9)
        )

    pipeline = (
        (3, eval_stages.stage_3),
        (4, eval_stages.stage_4),
        (5, eval_stages.stage_5),
        (8, formula_eval.stage_8),
        (9, oracle_check.stage_9),
    )
    results: list[StageResult] = []
    for stage_id, fn in pipeline:
        try:
            status, raw_findings = fn(spec)
        except Exception as ex:  # 防御性 — DSL 内部 bug 不应整体打断
            results.append(StageResult(
                stage_id=stage_id,
                status="FAIL",
                findings=[Finding(
                    severity="error",
                    rule_id=f"stage_{stage_id}_internal_error",
                    field_path="<internal>",
                    message=f"stage {stage_id} 内部异常：{type(ex).__name__}: {ex}",
                    suggested_fix="向 ops-spec-gen 维护者反馈完整堆栈",
                )]
            ))
            continue
        findings = [Finding(**f) for f in raw_findings]
        results.append(StageResult(stage_id=stage_id, status=status, findings=findings))
    return tuple(results)


def render_text(stages: list[StageResult], *, quiet: bool = False) -> str:
    """Render stage results as text. quiet=True 时只打 FAIL 的 stage（PASS/SKIP 折叠为一行汇总）。"""
    lines: list[str] = []
    icons = {"PASS": "✓", "FAIL": "✗", "SKIP": "↷"}

    if quiet:
        # 只列 FAIL，PASS/SKIP 计数
        passed = [s for s in stages if s.status == "PASS"]
        skipped = [s for s in stages if s.status == "SKIP"]
        failed = [s for s in stages if s.status == "FAIL"]
        if passed:
            lines.append(f"  {icons['PASS']}  {len(passed)} 个 stage PASS: "
                         + ", ".join(f"stage {s.stage_id}" for s in passed))
        if skipped:
            lines.append(f"  {icons['SKIP']}  {len(skipped)} 个 stage SKIP: "
                         + ", ".join(f"stage {s.stage_id}" for s in skipped))
        for st in failed:
            lines.append(f"\n[stage {st.stage_id}] {icons[st.status]} {st.status}")
            for f in st.findings:
                if f.severity not in ("error", "warning"):
                    continue
                sev = {"error": "ERR ", "warning": "WARN"}[f.severity]
                lines.append(f"  {sev}  {f.rule_id}")
                lines.append(f"        path: {f.field_path}")
                lines.append(f"        msg : {f.message}")
                if f.suggested_fix:
                    lines.append(f"        fix : {f.suggested_fix}")
    else:
        for st in stages:
            lines.append(f"\n[stage {st.stage_id}] {icons[st.status]} {st.status}")
            for f in st.findings:
                sev = {"error": "ERR ", "warning": "WARN", "info": "INFO"}[f.severity]
                lines.append(f"  {sev}  {f.rule_id}")
                lines.append(f"        path: {f.field_path}")
                lines.append(f"        msg : {f.message}")
                if f.suggested_fix:
                    lines.append(f"        fix : {f.suggested_fix}")

    overall = (
        "PASS"
        if all(s.status in ("PASS", "SKIP") for s in stages)
        else "FAIL"
    )
    lines.append(f"\n=== overall: {overall} ===")
    return "\n".join(lines)


def _validate_argparser():
    ap = argparse.ArgumentParser(description="Validate spec.yaml (full 9-stage).")
    ap.add_argument("spec_path")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    ap.add_argument("--strict", action="store_true",
                    help="Exit non-zero on warnings as well as errors")
    ap.add_argument("--stage", action="append", type=int, choices=range(1, 10),
                    metavar="N",
                    help="只跑指定 stage（可多次：--stage 1 --stage 2）；省略时跑全部 9 个")
    ap.add_argument("--quiet", action="store_true",
                    help="只打 FAIL 的 stage（仍输出 overall 行）")
    return ap


def _load_spec_or_exit(spec_path):
    path = Path(spec_path)
    if not path.exists():
        print(f"ERROR: spec file not found: {path}", file=sys.stderr)
        return None, 2
    try:
        spec = _load_yaml(path)
    except yaml.YAMLError as e:
        print(f"ERROR: YAML 解析失败 {path}", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        return None, 2
    if not isinstance(spec, dict):
        print(f"ERROR: {spec_path} did not parse as a YAML mapping", file=sys.stderr)
        return None, 2
    return spec, 0


def _run_selected_stages(spec, selected, registries):
    stages: list[StageResult] = []
    if selected is None or 1 in selected:
        stages.append(stage_1(spec))
    if selected is None or 2 in selected:
        stages.append(stage_2(spec, registries))
    dsl_needed = selected is None or selected & {3, 4, 5, 8, 9}
    if dsl_needed:
        s3, s4, s5, s8, s9 = _run_eval_stages(spec)
        for sid, sr in zip((3, 4, 5, 8, 9), (s3, s4, s5, s8, s9)):
            if selected is None or sid in selected:
                stages.append(sr)
    if selected is None or 6 in selected:
        stages.append(stage_6(spec, registries))
    if selected is None or 7 in selected:
        stages.append(stage_7(spec))
    stages.sort(key=lambda s: s.stage_id)
    return stages


def main() -> int:
    args = _validate_argparser().parse_args()
    spec, err = _load_spec_or_exit(args.spec_path)
    if spec is None:
        return err

    selected = set(args.stage) if args.stage else None
    registries = load_registries()
    stages = _run_selected_stages(spec, selected, registries)

    if args.json:
        print(json.dumps({
            "spec": str(args.spec_path),
            "stages": [{"stage_id": s.stage_id, "status": s.status,
                        "findings": [asdict(f) for f in s.findings]}
                       for s in stages],
        }, ensure_ascii=False, indent=2))
    else:
        print(render_text(stages, quiet=args.quiet))

    has_error = any(f.severity == "error" for s in stages for f in s.findings)
    has_warning = any(f.severity == "warning" for s in stages for f in s.findings)
    if has_error:
        return 1
    if args.strict and has_warning:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
