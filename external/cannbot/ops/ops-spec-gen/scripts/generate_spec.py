#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Interactive algorithm-spec.yaml generator.

Reads registries/*.yaml to get controlled enums (25 categories, 27 paradigms,
category->required-paradigms map), prompts the user for the minimum information
needed, and writes ops/<op>/spec.yaml using templates/spec.yaml.tmpl.

Non-interactive use:
    python3 generate_spec.py \
        --op-name softmax --category reduction_composite \
        --paradigms Reduction,NumericalStable,FusedComposite \
        --inputs x:float16,float32,bfloat16 \
        --outputs y \
        --output-dir /path/to/ops/softmax

After generation, run validate_spec.py to catch the {{TODO}} placeholders that
must be filled in by hand (formula, reference_oracle.api, supported_combinations,
boundary_conditions, etc.).
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

SKILL_DIR = Path(__file__).resolve().parent.parent
REGISTRIES = SKILL_DIR / "registries"
TEMPLATE = SKILL_DIR / "templates" / "spec.yaml.tmpl"


def _load(name: str) -> dict:
    with open(REGISTRIES / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_registries() -> dict:
    return {
        "categories": _load("category_enum.yaml")["categories"],
        "paradigms": [p["name"] for p in _load("paradigm_enum.yaml")["paradigms"]],
        "category_map": _load("category_paradigm_map.yaml"),
        "chips": _load("chip_registry.yaml")["chips"],
    }


def _chip_dtype_table(chips: list[dict]) -> dict[str, set[str]]:
    """chip_id → supported_dtypes set（从 registry 抽出来的查表）。"""
    return {c["id"]: set(c.get("supported_dtypes") or []) for c in chips}


def narrow_chips_by_dtypes(chips: list[dict], dtypes: set[str]) -> list[str]:
    """返回能覆盖所有 dtypes 的芯片 id 列表；dtypes 为空时返回全部 chip id。"""
    if not dtypes:
        return [c["id"] for c in chips]
    out: list[str] = []
    for c in chips:
        supported = set(c.get("supported_dtypes") or [])
        if dtypes.issubset(supported):
            out.append(c["id"])
    return out


def _load_tolerance_defaults() -> dict:
    """Read per-dtype tolerance defaults from the shared registry."""
    return _load("tolerance_defaults.yaml")["defaults"]


@dataclass
class TensorSpec:
    name: str
    dtype_set: list[str]
    rank_range: tuple[int, int] = (0, 8)


@dataclass
class GenInput:
    op_name: str
    description: str
    category: str
    paradigms: list[str]
    inputs: list[TensorSpec]
    outputs: list[str]
    promotion: str = "same_as_first_input"
    broadcast_kind: str = "numpy"
    accumulation_order: str = "none"
    numerical_stability_required: bool = False
    supported_chips: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.supported_chips is None:
            self.supported_chips = []


def parse_tensor_arg(s: str) -> TensorSpec:
    """Parse 'x:float16,float32,bfloat16' or 'x'."""
    if ":" in s:
        name, dtypes = s.split(":", 1)
        dtype_set = [d.strip() for d in dtypes.split(",") if d.strip()]
    else:
        name, dtype_set = s, ["float32"]
    if not re.match(r"^[a-z][a-zA-Z0-9_]*$", name):
        raise ValueError(f"Invalid tensor name: {name}")
    return TensorSpec(name=name, dtype_set=dtype_set)


def prompt(msg: str, default: str = "", allow_empty: bool = False) -> str:
    """Read a line from stdin; type 'q' or Ctrl-D / Ctrl-C to quit.

    allow_empty=True 用于"空回车 = 结束 / 跳过"的交互场景（如列表收集），
    此时不再循环要求重新输入。
    """
    while True:
        try:
            ans = input(f"{msg} [{default}]: " if default else f"{msg}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  ↷ 用户取消生成（EOF / Ctrl-C）")
            raise SystemExit(0)
        if ans.lower() in ("q", "quit", "exit"):
            print("  ↷ 用户取消生成（输入 q）")
            raise SystemExit(0)
        if ans:
            return ans
        if default:
            return default
        if allow_empty:
            return ""


def prompt_choice(msg: str, choices: list[str], default: str = "") -> str:
    """Pick one from a list. Type 'q' / Ctrl-D / Ctrl-C to quit."""
    print(f"{msg}")
    for i, c in enumerate(choices, 1):
        marker = "*" if c == default else " "
        print(f"  {marker} {i:2d}. {c}")
    print("  (输入 q / Ctrl-D 退出生成)")
    while True:
        try:
            ans = input(f"  selection [{default or '?'}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  ↷ 用户取消生成")
            raise SystemExit(0)
        if ans.lower() in ("q", "quit", "exit"):
            print("  ↷ 用户取消生成")
            raise SystemExit(0)
        if not ans and default:
            return default
        if ans.isdigit() and 1 <= int(ans) <= len(choices):
            return choices[int(ans) - 1]
        if ans in choices:
            return ans
        print(f"  ✗ 无效输入 {ans!r}；请输入 1-{len(choices)} 的编号或选项名")


def prompt_multi_choice(msg: str, choices: list[str], defaults: list[str] | None = None) -> list[str]:
    """Pick one or more from a list (comma-separated indices or names).

    空回车 → 返回 defaults（若提供）或全选。q / Ctrl-D 退出生成。
    """
    print(f"{msg}")
    default_set = set(defaults or [])
    for i, c in enumerate(choices, 1):
        marker = "*" if c in default_set else " "
        print(f"  {marker} {i:2d}. {c}")
    hint = "逗号分隔编号或名称；空回车=默认；q 退出"
    print(f"  ({hint})")
    while True:
        try:
            ans = input(f"  selection [{','.join(defaults) if defaults else 'all'}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  ↷ 用户取消生成")
            raise SystemExit(0)
        if ans.lower() in ("q", "quit", "exit"):
            print("  ↷ 用户取消生成")
            raise SystemExit(0)
        if not ans:
            return list(defaults) if defaults else list(choices)
        picked: list[str] = []
        bad: list[str] = []
        for tok in [t.strip() for t in ans.split(",") if t.strip()]:
            if tok.isdigit() and 1 <= int(tok) <= len(choices):
                c = choices[int(tok) - 1]
            elif tok in choices:
                c = tok
            else:
                bad.append(tok)
                continue
            if c not in picked:
                picked.append(c)
        if bad:
            print(f"  ✗ 无效项: {bad}；请重新输入")
            continue
        if not picked:
            print(f"  ✗ 至少选一项")
            continue
        return picked


def _collect_op_basic(reg):
    op_name = prompt("Operator name (lowercase, e.g. softmax)")
    if not re.match(r"^[a-z][a-z0-9_]*$", op_name):
        raise SystemExit(f"Invalid op name: {op_name}")
    description = prompt("One-line description")
    category = prompt_choice("Pick category (single-select, 25 controlled values):",
                             reg["categories"])
    return op_name, description, category


def _collect_paradigms(reg, category):
    required = reg["category_map"]["category_requires_paradigms"].get(category, [])
    print(f"\n  category={category} requires paradigm(s): {required}")
    extra = prompt(
        "Add extra paradigms (comma-separated, blank for none)",
        default="", allow_empty=True).strip()
    paradigms = list(required)
    if extra:
        for p in [x.strip() for x in extra.split(",") if x.strip()]:
            if p not in reg["paradigms"]:
                raise SystemExit(f"Unknown paradigm: {p}. Allowed: {reg['paradigms']}")
            if p not in paradigms:
                paradigms.append(p)
    return paradigms


def _collect_inputs():
    inputs: list[TensorSpec] = []
    print("\n--- Inputs ---")
    while True:
        name = prompt(f"Input #{len(inputs)+1} name (blank to finish)", default="", allow_empty=True)
        if not name:
            if not inputs:
                print("  must have ≥ 1 input")
                continue
            break
        dtypes = prompt("  dtype_set (comma-separated)", default="float16,float32,bfloat16")
        inputs.append(TensorSpec(name=name, dtype_set=[d.strip() for d in dtypes.split(",")]))
    return inputs


def _collect_outputs():
    outputs: list[str] = []
    print("\n--- Outputs ---")
    while True:
        name = prompt(f"Output #{len(outputs)+1} name (blank to finish)", default="", allow_empty=True)
        if not name:
            if not outputs:
                print("  must have ≥ 1 output")
                continue
            break
        outputs.append(name)
    return outputs


def _collect_supported_chips(reg, inputs):
    declared_dtypes: set[str] = set()
    for t in inputs:
        declared_dtypes.update(t.dtype_set)
    candidates = narrow_chips_by_dtypes(reg["chips"], declared_dtypes)
    if not candidates:
        print(f"\n  ⚠ 没有芯片同时支持所有 dtype {sorted(declared_dtypes)}；")
        print(f"    请手动从全集挑选并在生成后修正 dtype_set / supported_combinations")
        candidates = [c["id"] for c in reg["chips"]]
    return prompt_multi_choice(
        f"\nsupported_chips（按 dtype_set {sorted(declared_dtypes)} 自动收窄；多选）",
        candidates,
        defaults=candidates,
    )


def interactive_collect(reg: dict) -> GenInput:
    print("\n=== algorithm-spec.yaml interactive generator ===\n")
    op_name, description, category = _collect_op_basic(reg)
    paradigms = _collect_paradigms(reg, category)
    inputs = _collect_inputs()
    outputs = _collect_outputs()

    promotion = prompt_choice(
        "dtype_policy.promotion",
        ["same_as_first_input", "numpy_promote", "fixed", "upcast_for_accum"],
        default="same_as_first_input")
    broadcast_kind = prompt_choice(
        "broadcast.kind",
        ["none", "numpy", "explicit"],
        default="numpy")
    accumulation_order = prompt_choice(
        "determinism.accumulation_order",
        ["none", "stable_in_axis", "tree", "sequential"],
        default="none")

    supported_chips = _collect_supported_chips(reg, inputs)

    return GenInput(
        op_name=op_name,
        description=description,
        category=category,
        paradigms=paradigms,
        inputs=inputs,
        outputs=outputs,
        promotion=promotion,
        broadcast_kind=broadcast_kind,
        accumulation_order=accumulation_order,
        numerical_stability_required=("NumericalStable" in paradigms),
        supported_chips=supported_chips,
    )


def build_attributes_block(paradigms: list[str], inputs: list[TensorSpec]) -> str:
    """Inject minimum required attrs per paradigm injection table.

    Multiple injectors may apply (e.g. Quantization + RandomSampling). Order
    matters only cosmetically — collected lines are joined.
    """
    blocks: list[str] = []

    if "Reduction" in paradigms and "Recurrence" not in paradigms:
        target = inputs[0].name
        blocks.append(
            "  - name: dim\n"
            "    type: int64\n"
            "    default: -1\n"
            '    semantics: "归约轴；负值按 Python 风格折算"\n'
            "    machine_constraint:\n"
            "      kind: int_in_range_relative_to_rank\n"
            f"      target: {target}\n"
            f'      lower_inclusive: "-rank({target})"\n'
            f'      upper_exclusive: "rank({target})"'
        )

    if "Quantization" in paradigms:
        blocks.append(
            "  - name: scale\n    type: float32\n    default: 1.0\n"
            '    semantics: "量化缩放因子"\n'
            "    machine_constraint: {kind: float_in_range}\n"
            "  - name: zero_point\n    type: int64\n    default: 0\n"
            '    semantics: "量化零点"\n'
            "    machine_constraint: {kind: int_in_range}"
        )

    if "RandomSampling" in paradigms:
        blocks.append(
            "  - name: seed\n    type: int64\n    default: 0\n"
            '    semantics: "随机数种子；固定 seed 保证 bitwise_reproducible"\n'
            "    machine_constraint: {kind: int_in_range}"
        )

    return "\n".join(blocks) if blocks else "  []"


def build_op_block(op_name: str, description: str, category: str,
                   paradigms: list[str],
                   supported_chips: list[str] | None = None,
                   error_codes: list[str] | None = None) -> str:
    """Build the `op:` field.

    发射规则：
      * supported_chips 非空 → 写入 platform_constraints.supported_chips
      * CollectiveCommunication 范式 → 追加 requires_hccl: true
      * error_codes 非空 → 写入 op.error_codes（语义类别）
    """
    parts = [
        f"  name: {op_name}",
        f'  version: "1.0"',
        f'  description: "{description}"',
        f"  category: {category}",
        f"  paradigms: [{', '.join(paradigms)}]",
    ]
    if error_codes:
        parts.append("  error_codes:")
        for ec in error_codes:
            parts.append(f"    - {ec}")
    chips = supported_chips or []
    needs_hccl = "CollectiveCommunication" in paradigms
    if chips or needs_hccl:
        parts.append("  platform_constraints:")
        if chips:
            parts.append(f"    supported_chips: [{', '.join(chips)}]")
        if needs_hccl:
            parts.append("    requires_hccl: true")
    return "\n".join(parts)


def build_inputs_block(inputs: list[TensorSpec], paradigms: list[str]) -> str:
    lines = []
    has_stateful = "Stateful" in paradigms
    for i, inp in enumerate(inputs):
        # 折叠维名直接用 input 全名，避免多 input 同首字母冲突（e.g. xa/xb 都生成 "...x"）
        folded_name = inp.name.lower()
        # Stateful: 第一个 input 自动设为 role: state（占位；用户可改）
        role = "state" if (has_stateful and i == 0) else "tensor"
        lines.append(f"  - name: {inp.name}")
        lines.append(f"    role: {role}")
        lines.append(f"    dtype_set: [{', '.join(inp.dtype_set)}]")
        lines.append(f"    rank_range: [{inp.rank_range[0]}, {inp.rank_range[1]}]")
        lines.append(f"    layout: ND")
        lines.append(f"    shape:")
        lines.append(f'      symbolic: ["...{folded_name}"]')
    return "\n".join(lines)


def build_outputs_block(outputs: list[str], first_input: str, category: str) -> str:
    """生成 outputs 段。VariableOutput 范畴（如 nonzero / unique）走 data_dependent 模板。"""
    lines = []
    is_variable_output = category == "variable_output"
    for out in outputs:
        lines.append(f"  - name: {out}")
        if is_variable_output:
            # data_dependent 模板 — 形状由运行时输入值决定，仅给出语义描述与上界
            lines.append(f"    shape_rule_kind: data_dependent")
            lines.append(f"    data_dependent_shape: true")
            lines.append(f"    shape_rule_description: |")
            lines.append(f"      # TODO: 描述 {out}.shape 与 {first_input} 实际值的关系")
            lines.append(f"      # 例如：{out}.shape = (K, rank({first_input}))，K = count({first_input} != 0)")
            lines.append(f"    shape_bounds:")
            lines.append(f"      max_elements: \"prod({first_input}.shape)\"  # TODO 上界")
            lines.append(f"      output_rank: 2  # TODO")
            lines.append(f"    dtype_rule_kind: numpy_expr")
            lines.append(f"    dtype_rule: |")
            lines.append(f"      {out}.dtype = np.int64  # TODO: VariableOutput 通常输出索引")
        else:
            lines.append(f"    shape_rule_kind: numpy_expr")
            lines.append(f"    shape_rule: |")
            lines.append(f"      {out}.shape = {first_input}.shape  # TODO: numpy 表达式")
            lines.append(f"    dtype_rule_kind: numpy_expr")
            lines.append(f"    dtype_rule: |")
            lines.append(f"      {out}.dtype = {first_input}.dtype  # TODO")
        lines.append(f"    layout: ND")
        lines.append(f"    aliasing: none")
    return "\n".join(lines)


def build_supported_combinations_block(inputs: list[TensorSpec],
                                        outputs: list[str],
                                        promotion: str) -> str:
    """生成 dtype_policy.supported_combinations。

    策略：
      * 同 dtype 直传 — 取所有输入 dtype_set 交集（每个 dtype 一行）
      * promotion=numpy_promote：在交集行外加几行 promote 组合（取 fp16+fp32 / bf16+fp32 这种典型对）
      * 其他 promotion：仅同 dtype 直传，开发者后续补充
    """
    if not inputs or not outputs:
        return "    # TODO: supported_combinations 至少 1 行\n    - {inputs: {x: float32}, outputs: {y: float32}}"

    dtype_intersection = set(inputs[0].dtype_set)
    for inp in inputs[1:]:
        dtype_intersection &= set(inp.dtype_set)
    if not dtype_intersection:
        # 无交集：保守只生成第一个 dtype 一行
        dtype_intersection = {inputs[0].dtype_set[0]}

    lines = []
    for dtype in sorted(dtype_intersection):
        in_part = ", ".join(f"{inp.name}: {dtype}" for inp in inputs)
        out_part = ", ".join(f"{out}: {dtype}" for out in outputs)
        lines.append(f"    - {{inputs: {{{in_part}}}, outputs: {{{out_part}}}}}")

    if promotion == "numpy_promote" and len(inputs) >= 2:
        # 加 fp16 + fp32 / bf16 + fp32 这种典型混合行
        # 为 ≥ 3 input 算子（如 where(cond, x, y)）也生成：把 narrow 给"第一半"
        # input、wide 给"第二半"，简单但能覆盖；用户可手动微调
        mid = len(inputs) // 2 or 1
        for narrow, wide in (("float16", "float32"), ("bfloat16", "float32")):
            # 检查所有 input 都允许其分配的 dtype
            assigned = [narrow if i < mid else wide for i in range(len(inputs))]
            if all(d in inp.dtype_set for d, inp in zip(assigned, inputs)) and narrow != wide:
                in_part = ", ".join(
                    f"{inp.name}: {d}" for d, inp in zip(assigned, inputs)
                )
                out_part = ", ".join(f"{out}: {wide}" for out in outputs)
                lines.append(f"    - {{inputs: {{{in_part}}}, outputs: {{{out_part}}}}}")

    return "\n".join(lines)


# Per-dtype 容差默认值从 registries/tolerance_defaults.yaml 加载
# 这样 generator 与 stage 7 校验器共享同一份数据，避免漂移
def _format_tol_value(v) -> str:
    """Format a numeric value for inline yaml output (preserve scientific notation)."""
    if isinstance(v, (int, bool)) and not isinstance(v, bool) or v == 0:
        return "0"
    if isinstance(v, float):
        if v == 0:
            return "0"
        return f"{v:.1e}"
    return str(v)


def build_per_dtype_tolerance_block(inputs: list[TensorSpec], outputs: list[str],
                                     promotion: str) -> str:
    """生成 numerical_tolerance.per_dtype，覆盖 supported_combinations 中所有出现的 output dtype。"""
    defaults = _load_tolerance_defaults()
    # 与 supported_combinations 一致：先收集会出现的 output dtype
    dtype_intersection = set(inputs[0].dtype_set) if inputs else {"float32"}
    for inp in (inputs[1:] if inputs else []):
        dtype_intersection &= set(inp.dtype_set)
    if not dtype_intersection:
        dtype_intersection = {inputs[0].dtype_set[0]} if inputs else {"float32"}
    out_dtypes = set(dtype_intersection)
    # numpy_promote 时混合 dtype 行的 output dtype 也会出现
    if promotion == "numpy_promote" and len(inputs) >= 2:
        for narrow, wide in (("float16", "float32"), ("bfloat16", "float32")):
            mid = len(inputs) // 2 or 1
            assigned = [narrow if i < mid else wide for i in range(len(inputs))]
            if all(d in inp.dtype_set for d, inp in zip(assigned, inputs)):
                out_dtypes.add(wide)

    lines = []
    for dtype in sorted(out_dtypes):
        entry = defaults.get(dtype) or {"rtol": 1.0e-5, "atol": 1.0e-5, "metric": "max_relative"}
        rtol = _format_tol_value(entry["rtol"])
        atol = _format_tol_value(entry["atol"])
        metric = entry["metric"]
        lines.append(f"    {dtype}: {{rtol: {rtol}, atol: {atol}, metric: {metric}}}")
    return "\n".join(lines)


def build_accumulator_block(category: str, paradigms: list[str]) -> str:
    triggers = {"contraction", "reduction_composite"}
    if category in triggers or (
        "NumericalStable" in paradigms and "Reduction" in paradigms
    ):
        return "  accumulator_dtype: float32"
    return ""


def build_composition_block(paradigms: list[str], inputs: list[TensorSpec],
                            outputs: list[str]) -> str:
    if "FusedComposite" not in paradigms:
        return ""
    in_names = ",".join(inp.name for inp in inputs)
    out_name = outputs[0]
    return f"""
  composition:
    primitives:
      # TODO: 至少 2 条原语；op 必须在 PRIMITIVE_WHITELIST
      - id: prim1
        op: elementwise_unary
        inputs: [{in_names}]
        outputs: [tmp]
      - id: prim2
        op: elementwise_unary
        inputs: [tmp]
        outputs: [{out_name}]
    dataflow:
      intermediates: [tmp]
      no_leak: true
      fusable_groups:
        - [prim1, prim2]"""


# stage 6 要求 — 按 op.paradigms 注入对应必含 case 的占位
# 关键词必须能命中 registries/boundary_min_cases.yaml 中对应 paradigm 的 match_any
# 字段，否则生成的骨架会在 stage 6 直接 FAIL（违背"骨架可用"承诺）。

_PARADIGM_BOUNDARY_TEMPLATES = {
    "Reduction": [
        '  - case: "reduce 轴长度为 1"\n    synthesize: { __FIRST_INPUT__.shape: "[1]" }\n    machine_check: {kind: matches_oracle}',
        '  - case: "rank=0 标量输入"\n    synthesize: { __FIRST_INPUT__.shape: "[]" }\n    machine_check: {kind: matches_oracle}',
        '  - case: "空 Tensor"\n    synthesize: { __FIRST_INPUT__.shape: "[0, 4]" }\n    machine_check: {kind: returns_empty}',
    ],
    "SlidingWindow": [
        '  - case: "stride > kernel"\n    synthesize: { __FIRST_INPUT__.shape: "[1, 1, 8, 8]" }\n    machine_check: {kind: matches_oracle}',
    ],
    "Padding": [
        '  - case: "零 padding 退化"\n    synthesize: { __FIRST_INPUT__.shape: "[2, 3]" }\n    machine_check: {kind: matches_oracle}',
        '  - case: "超大 padding"\n    synthesize: { __FIRST_INPUT__.shape: "[2, 3]" }\n    machine_check: {kind: matches_oracle}',
    ],
    "IndexGather": [
        '  - case: "索引越界（负向）"\n    synthesize: { __FIRST_INPUT__.shape: "[4, 8]" }\n    machine_check: {kind: raises_error, error_type: attribute_value_out_of_range}',
    ],
    "ScatterUpdate": [
        '  - case: "索引越界"\n    synthesize: { __FIRST_INPUT__.shape: "[4, 8]" }\n    machine_check: {kind: raises_error, error_type: attribute_value_out_of_range}',
    ],
    "AtomicUpdate": [
        '  - case: "索引越界"\n    synthesize: { __FIRST_INPUT__.shape: "[4, 8]" }\n    machine_check: {kind: raises_error, error_type: attribute_value_out_of_range}',
    ],
    "SortSelect": [
        '  - case: "k=0"\n    synthesize: { __FIRST_INPUT__.shape: "[8]" }\n    machine_check: {kind: matches_oracle}',
        '  - case: "k=N"\n    synthesize: { __FIRST_INPUT__.shape: "[8]" }\n    machine_check: {kind: matches_oracle}',
    ],
    "Histogram": [
        '  - case: "所有值落同一 bin"\n    synthesize: { __FIRST_INPUT__.shape: "[16]" }\n    machine_check: {kind: matches_oracle}',
    ],
    "Spectral": [
        '  - case: "长度=2 幂次"\n    synthesize: { __FIRST_INPUT__.shape: "[16]" }\n    machine_check: {kind: matches_oracle}',
        '  - case: "非 2 幂次长度"\n    synthesize: { __FIRST_INPUT__.shape: "[15]" }\n    machine_check: {kind: matches_oracle}',
    ],
    "Stateful": [
        '  - case: "跨调用状态保持（first call vs second call）"\n    synthesize: { __FIRST_INPUT__.shape: "[8]" }\n    machine_check: {kind: matches_oracle}',
    ],
}

_PARADIGM_EXTREME_TEMPLATES = {
    "NumericalStable": [
        '  - case: "fp16 上溢边界"\n    synthesize:\n      shapes: { __FIRST_INPUT__: "[8]" }\n      patterns:\n        - {pattern: "all_same(60000.0)", target: __FIRST_INPUT__}\n      dtype: float16\n    machine_check: {kind: matches_oracle}',
    ],
    "IndexGather": [
        '  - case: "全相同索引"\n    synthesize:\n      shapes: { __FIRST_INPUT__: "[8]" }\n      patterns:\n        - {pattern: all_zero, target: __FIRST_INPUT__}\n    machine_check: {kind: matches_oracle}',
    ],
    "AtomicUpdate": [
        '  - case: "索引冲突"\n    synthesize:\n      shapes: { __FIRST_INPUT__: "[8]" }\n      patterns:\n        - {pattern: all_zero, target: __FIRST_INPUT__}\n    machine_check: {kind: matches_oracle}',
    ],
    "MaskPredicate": [
        '  - case: "全 True mask"\n    synthesize:\n      shapes: { __FIRST_INPUT__: "[8]" }\n      patterns:\n        - {pattern: all_zero, target: __FIRST_INPUT__}\n    machine_check: {kind: matches_oracle}',
        '  - case: "全 False mask"\n    synthesize:\n      shapes: { __FIRST_INPUT__: "[8]" }\n      patterns:\n        - {pattern: all_zero, target: __FIRST_INPUT__}\n    machine_check: {kind: matches_oracle}',
    ],
    "ArgReduce": [
        '  - case: "tie / 等值"\n    synthesize:\n      shapes: { __FIRST_INPUT__: "[8]" }\n      patterns:\n        - {pattern: all_zero, target: __FIRST_INPUT__}\n    machine_check: {kind: matches_oracle}',
    ],
    "RandomSampling": [
        '  - case: "deterministic 固定 seed"\n    synthesize:\n      shapes: { __FIRST_INPUT__: "[8]" }\n      patterns:\n        - {pattern: all_zero, target: __FIRST_INPUT__}\n    machine_check: {kind: matches_oracle}',
    ],
    "Quantization": [
        '  - case: "全零输入（zero_point 行为）"\n    synthesize:\n      shapes: { __FIRST_INPUT__: "[8]" }\n      patterns:\n        - {pattern: all_zero, target: __FIRST_INPUT__}\n    machine_check: {kind: matches_oracle}',
    ],
}


def build_extra_boundary_cases(paradigms: list[str], first_input: str) -> str:
    """生成 paradigm 必含的 boundary case 占位，避免 stage 6 FAIL。"""
    lines: list[str] = []
    for p in paradigms:
        for tmpl in _PARADIGM_BOUNDARY_TEMPLATES.get(p, []):
            lines.append(tmpl.replace("__FIRST_INPUT__", first_input))
    return "\n".join(lines)


def build_extra_extreme_cases(paradigms: list[str], first_input: str) -> str:
    """生成 paradigm 必含的 extreme case 占位，避免 stage 6 FAIL。"""
    lines: list[str] = []
    for p in paradigms:
        for tmpl in _PARADIGM_EXTREME_TEMPLATES.get(p, []):
            lines.append(tmpl.replace("__FIRST_INPUT__", first_input))
    return "\n".join(lines)


def infer_error_codes(paradigms: list[str], attributes_block: str) -> list[str]:
    """按 paradigm + 注入的 attributes 推断该算子的默认 error_codes 集合。

    最小集：null_input（任何算子都可能传 nullptr）。其余按结构特征加入：
      * Broadcast / Reduction / Contraction / IndexGather / LayoutTransform
        → 输入 shape 之间存在硬约束，加 shape_mismatch
      * attributes 段已注入（非 "[]"）→ 有可越界的属性，加 attribute_value_out_of_range
    """
    codes: list[str] = ["null_input"]
    shape_paradigms = {
        "Broadcast", "Reduction", "Contraction", "IndexGather",
        "ScatterUpdate", "LayoutTransform", "Padding", "SortSelect",
        "ArgReduce",
    }
    if any(p in shape_paradigms for p in paradigms):
        codes.append("shape_mismatch")
    if attributes_block.strip() and attributes_block.strip() != "[]":
        codes.append("attribute_value_out_of_range")
    return codes


def build_raises_error_boundary_case(error_codes: list[str], first_input: str) -> str:
    """挑一条最贴合 error_codes 的 raises_error 占位 case。

    优先级：shape_mismatch > attribute_value_out_of_range > null_input。
    """
    if "shape_mismatch" in error_codes:
        return (
            '  - case: "输入 shape 不满足算子约束 → 报错（占位，请按算子改具体形状）"\n'
            f'    synthesize: {{ {first_input}.shape: "[2, 3]" }}\n'
            "    machine_check: {kind: raises_error, error_type: shape_mismatch}"
        )
    if "attribute_value_out_of_range" in error_codes:
        return (
            '  - case: "属性取值越界 → 报错（占位，请按算子改具体属性值）"\n'
            f'    synthesize: {{ {first_input}.shape: "[2, 3]" }}\n'
            "    machine_check: {kind: raises_error, error_type: attribute_value_out_of_range}"
        )
    return (
        '  - case: "必填输入为 null → 报错（占位，请按算子改实际入参）"\n'
        f'    synthesize: {{ {first_input}.shape: "[2, 3]" }}\n'
        "    machine_check: {kind: raises_error, error_type: null_input}"
    )


def _build_techniques_text(numerical_stability_required):
    if not numerical_stability_required:
        return "[]"
    return (
        "\n    - name: max_subtraction\n"
        '      applies_to: "TODO"\n'
        '      rationale: "TODO"\n'
        "      anti_pattern_id: AP-004"
    )


def _build_boundary_extreme_cases(gi, first_input, attributes_block):
    extra_boundary_cases = build_extra_boundary_cases(gi.paradigms, first_input)
    extra_extreme_cases = build_extra_extreme_cases(gi.paradigms, first_input)
    error_codes = infer_error_codes(gi.paradigms, attributes_block)
    raises_error_case = build_raises_error_boundary_case(error_codes, first_input)
    extra_boundary_cases = (extra_boundary_cases + "\n" + raises_error_case
                            if extra_boundary_cases.strip()
                            else raises_error_case)
    return extra_boundary_cases, extra_extreme_cases, error_codes


def _build_replacements(gi, first_input, first_output, attributes_block,
                        extra_boundary_cases, extra_extreme_cases, error_codes,
                        techniques):
    return {
        "{{op_name}}": gi.op_name,
        "{{description}}": gi.description,
        "{{category}}": gi.category,
        "{{paradigms_csv}}": ", ".join(gi.paradigms),
        "{{op_block}}": build_op_block(gi.op_name, gi.description,
                                        gi.category, gi.paradigms,
                                        supported_chips=gi.supported_chips,
                                        error_codes=error_codes),
        "{{inputs_block}}": build_inputs_block(gi.inputs, gi.paradigms),
        "{{attributes_block}}": attributes_block,
        "{{outputs_block}}": build_outputs_block(gi.outputs, first_input, gi.category),
        "{{first_input}}": first_input,
        "{{first_output}}": first_output,
        "{{promotion}}": gi.promotion,
        "{{broadcast_kind}}": gi.broadcast_kind,
        "{{accumulation_order}}": gi.accumulation_order,
        "{{numerical_stability_required}}": str(gi.numerical_stability_required).lower(),
        "{{numerical_stability_techniques}}": techniques,
        "{{accumulator_block}}": build_accumulator_block(gi.category, gi.paradigms),
        "{{composition_block}}": build_composition_block(gi.paradigms, gi.inputs, gi.outputs),
        "{{supported_combinations_block}}": build_supported_combinations_block(
            gi.inputs, gi.outputs, gi.promotion),
        "{{per_dtype_tolerance_block}}": build_per_dtype_tolerance_block(
            gi.inputs, gi.outputs, gi.promotion),
        "{{extra_boundary_cases}}": extra_boundary_cases,
        "{{extra_extreme_cases}}": extra_extreme_cases,
    }


def render(gi: GenInput) -> str:
    text = TEMPLATE.read_text(encoding="utf-8")
    first_input = gi.inputs[0].name
    first_output = gi.outputs[0]

    techniques = _build_techniques_text(gi.numerical_stability_required)
    attributes_block = build_attributes_block(gi.paradigms, gi.inputs)
    extra_boundary_cases, extra_extreme_cases, error_codes = _build_boundary_extreme_cases(
        gi, first_input, attributes_block)

    replacements = _build_replacements(
        gi, first_input, first_output, attributes_block,
        extra_boundary_cases, extra_extreme_cases, error_codes, techniques)
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def _build_argparser():
    ap = argparse.ArgumentParser(description="Generate spec.yaml skeleton.")
    ap.add_argument("--op-name")
    ap.add_argument("--category")
    ap.add_argument("--paradigms", help="Comma-separated PascalCase, e.g. Reduction,NumericalStable")
    ap.add_argument("--inputs", help="Comma-separated 'name:dtype1,dtype2'", default="")
    ap.add_argument("--outputs", help="Comma-separated names", default="")
    ap.add_argument("--description", default=None,
                    help="算子一句话描述；省略时生成 'TODO: describe <op_name>'")
    ap.add_argument("--promotion", default="same_as_first_input")
    ap.add_argument("--broadcast-kind", dest="broadcast_kind", default="numpy")
    ap.add_argument("--accumulation-order", dest="accumulation_order", default="none")
    ap.add_argument("--chips", default="",
                    help="Comma-separated chip ids（如 Ascend910B,Ascend950PR）；"
                         "省略时按 inputs.dtype_set 自动收窄到全部兼容芯片")
    ap.add_argument("--output-dir", required=True,
                    help="Directory to write spec.yaml into (created if absent)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing spec.yaml")
    return ap


def _resolve_chips(args, inputs, reg):
    declared_dtypes: set[str] = set()
    for t in inputs:
        declared_dtypes.update(t.dtype_set)
    all_chip_ids = [c["id"] for c in reg["chips"]]
    if args.chips:
        supported_chips = [c.strip() for c in args.chips.split(",") if c.strip()]
        unknown = [c for c in supported_chips if c not in all_chip_ids]
        if unknown:
            raise SystemExit(
                f"Unknown chip id(s): {unknown}. Allowed: {all_chip_ids}"
            )
        return supported_chips
    supported_chips = narrow_chips_by_dtypes(reg["chips"], declared_dtypes)
    if not supported_chips:
        raise SystemExit(
            f"No chip in registry supports all of dtype_set "
            f"{sorted(declared_dtypes)}; pass --chips explicitly."
        )
    return supported_chips


def _build_geninput_from_args(args, reg):
    paradigms = [p.strip() for p in args.paradigms.split(",") if p.strip()]
    if args.inputs:
        inputs = [parse_tensor_arg(s) for s in args.inputs.split(";") if s.strip()]
    else:
        inputs = [TensorSpec(name="x", dtype_set=["float32"])]
    outputs = [s.strip() for s in args.outputs.split(",") if s.strip()] or ["y"]
    supported_chips = _resolve_chips(args, inputs, reg)

    gi = GenInput(
        op_name=args.op_name,
        description=args.description or f"TODO: describe {args.op_name} operator",
        category=args.category,
        paradigms=paradigms,
        inputs=inputs,
        outputs=outputs,
        promotion=args.promotion,
        broadcast_kind=args.broadcast_kind,
        accumulation_order=args.accumulation_order,
        numerical_stability_required=("NumericalStable" in paradigms),
        supported_chips=supported_chips,
    )
    if gi.category not in reg["categories"]:
        raise SystemExit(f"Unknown category: {gi.category}")
    for p in gi.paradigms:
        if p not in reg["paradigms"]:
            raise SystemExit(f"Unknown paradigm: {p}")
    return gi


def main() -> int:
    args = _build_argparser().parse_args()
    reg = load_registries()
    non_interactive = args.op_name and args.category and args.paradigms

    if non_interactive:
        gi = _build_geninput_from_args(args, reg)
    else:
        gi = interactive_collect(reg)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "spec.yaml"
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} exists. Use --force to overwrite.")

    out_path.write_text(render(gi), encoding="utf-8")
    print(f"\n✓ Wrote {out_path}")
    print(f"  category={gi.category}  paradigms={gi.paradigms}")
    print("\nNext steps:")
    print("  1. Fill in TODOs (formula / oracle.api / supported_combinations / boundary cases)")
    print(f"  2. Validate: python3 {SKILL_DIR}/scripts/validate_spec.py {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
