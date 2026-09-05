#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Generate S2P3_test_design.md from standard white-box Step 2 artifacts.

The generator is intentionally operator-agnostic. It renders structured facts
from S2P1/S2P2 artifacts and preserves the LLM/verifier sections when rerun.
"""

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_logger = logging.getLogger(__name__)

SCRIPT_BEGIN = "<!-- BEGIN SCRIPT GENERATED SECTION -->"
SCRIPT_END = "<!-- END SCRIPT GENERATED SECTION -->"
LLM_BEGIN = "<!-- BEGIN LLM ANALYSIS SECTION -->"
LLM_END = "<!-- END LLM ANALYSIS SECTION -->"
VERIFIER_BEGIN = "<!-- BEGIN VERIFIER SECTION -->"
VERIFIER_END = "<!-- END VERIFIER SECTION -->"

REQUIRED_SECTIONS = [
    "## 1. 生成与输入摘要",
    "## 2. 接口与参数模型",
    "## 3. 路径与 Group 覆盖",
    "### 3.1 代码路径全景",
    "### 3.2 测试关注点（groups）",
    "## 4. Case 枚举与一致性校验",
    "## 5. 自动发现的未确认项",
    "## 6. 测试设计分析",
    "### 6.1 事实摘要与设计结论",
    "### 6.2 关键派生变量",
    "### 6.3 执行模式分析",
    "## 7. 风险与补充建议",
    "## 8. Step 3 验证结论（原 §9 验证结论）",
]


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_text_if_exists(path):
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def fmt_cell(value):
    if value is None:
        return "-"
    if isinstance(value, (list, tuple)):
        if not value:
            return "-"
        return ", ".join(fmt_inline(v) for v in value)
    if isinstance(value, dict):
        return "<br>".join(f"`{k}`={fmt_inline(v)}" for k, v in value.items()) or "-"
    text = str(value).replace("\n", "<br>")
    return text.replace("|", "\\|")


def fmt_inline(value):
    if isinstance(value, list):
        return "[" + ", ".join(str(v) for v in value) + "]"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def fmt_code(value):
    return f"`{fmt_cell(value)}`" if value not in (None, "") else "-"


def condition_to_text(cond):
    if "boundary_check" in cond:
        return f"boundary:{cond['boundary_check']}"
    if "expr" in cond:
        rhs = cond.get("value", cond.get("ref", ""))
        return f"{cond['expr']} {cond.get('op', '')} {rhs}".strip()
    var = cond.get("var")
    if not var:
        return json.dumps(cond, ensure_ascii=False)
    if cond.get("op") == "range":
        return f"{var} range {cond.get('min')}..{cond.get('max')}"
    rhs = cond.get("value", cond.get("ref", ""))
    return f"{var}{cond.get('op', '')}{rhs}"


def path_conditions(path):
    conditions = path.get("conditions", [])
    if not conditions:
        return "-"
    return "<br>".join(fmt_cell(condition_to_text(c)) for c in conditions)


def parse_inputs(output_dir):
    files = {
        "S2P1_path_list.json": output_dir / "S2P1_path_list.json",
        "S2P1_operator_model.json": output_dir / "S2P1_operator_model.json",
        "S2P1_low_configs.json": output_dir / "S2P1_low_configs.json",
        "S2P2_param_def.json": output_dir / "S2P2_param_def.json",
        "S2P2_cases.json": output_dir / "S2P2_cases.json",
        "S2P2_traceability.md": output_dir / "S2P2_traceability.md",
    }
    data = {}
    parse_status = {}
    for name, path in files.items():
        if not path.exists():
            parse_status[name] = (False, "missing")
            if name.endswith(".json"):
                raise FileNotFoundError(f"required input missing: {path}")
            data[name] = ""
            continue
        if name.endswith(".json"):
            data[name] = load_json(path)
            parse_status[name] = (True, "json parsed")
        else:
            data[name] = read_text_if_exists(path)
            parse_status[name] = (True, "text loaded")
    return data, parse_status


def collect_stats(path_list, param_def, low_configs, cases):
    if not isinstance(cases, list):
        raise ValueError(f"S2P2_cases.json must be a list, got {type(cases).__name__}")
    paths = path_list.get("paths", [])
    return {
        "path_count": len(paths),
        "group_count": len(param_def.get("groups", [])),
        "case_count": len(cases),
        "low_config_count": len(low_configs),
        "reachability": Counter(p.get("reachability", "missing") for p in paths),
        "case_groups": Counter(c.get("_group", c.get("group", "missing")) for c in cases),
        "case_keys": Counter(c.get("key", "missing") for c in cases),
    }


def render_io_table(title, entries):
    lines = [
        f"#### {title}", "",
        "| 名称 | 类型/必选 | dtype | rank | shape/value |",
        "|------|-----------|-------|------|-------------|",
    ]
    for item in entries:
        dtype = item.get("dtype", {})
        rank = item.get("rank", {})
        shape = item.get("shape", item.get("value_domain", {}))
        param_type = item.get("param_type", item.get("type", "-"))
        lines.append(
            f"| `{fmt_cell(item.get('name'))}` | {fmt_cell(param_type)} | "
            f"{fmt_cell(dtype)} | {fmt_cell(rank)} | {fmt_cell(shape)} |"
        )
    if not entries:
        lines.append("| - | - | - | - | - |")
    return lines


def render_attributes(attrs):
    lines = [
        "#### Attributes", "",
        "| 名称 | 类型 | 默认值 | 范围/约束 | 来源 |",
        "|------|------|--------|-----------|------|",
    ]
    for item in attrs:
        lines.append(
            f"| `{fmt_cell(item.get('name'))}` | {fmt_cell(item.get('type'))} | "
            f"{fmt_cell(item.get('default'))} | "
            f"{fmt_cell(item.get('range', item.get('constraints')))} | "
            f"{fmt_cell(item.get('source'))} |"
        )
    if not attrs:
        lines.append("| - | - | - | - | - |")
    return lines


def render_paths(paths):
    lines = [
        "### 3.1 代码路径全景",
        "",
        "| path | key | reachability | group | conditions | kernel | source |",
        "|------|-----|--------------|-------|------------|--------|--------|",
    ]
    for path in paths:
        lines.append(
            f"| `{fmt_cell(path.get('id'))}` | "
            f"`{fmt_cell(path.get('tiling_key'))}` | "
            f"{fmt_cell(path.get('reachability'))} | "
            f"{fmt_cell(path.get('group'))} | "
            f"{path_conditions(path)} | "
            f"{fmt_cell(path.get('key_instructions', []))} | "
            f"{fmt_cell(path.get('source'))} |"
        )
    if not paths:
        lines.append("| - | - | - | - | - | - | - |")
    return lines


def entry_dimensions(entry):
    return [(k, v) for k, v in entry.items() if k not in {"path", "key"}]


def estimate_entry_count(entry):
    count = 1
    has_dim = False
    for _, value in entry_dimensions(entry):
        has_dim = True
        if isinstance(value, list):
            count *= max(len(value), 1)
    return count if has_dim else 1


def render_groups(groups):
    lines = ["### 3.2 测试关注点（groups）", ""]
    for index, group in enumerate(groups, start=1):
        lines.extend([
            f"#### 3.2.{index} {group.get('id')}",
            "",
            f"**路由条件**：{fmt_cell(group.get('mode'))}",
            "",
            f"**约束**：{fmt_cell(group.get('constraint_note'))}",
            "",
            "| dtype | path | key | 维度字段 | 预估组合数 |",
            "|-------|------|-----|----------|------------|",
        ])
        total = 0
        per_dtype = group.get("per_dtype", {})
        for dtype, entries in per_dtype.items():
            for entry in entries:
                dims = "<br>".join(f"`{k}`={fmt_inline(v)}" for k, v in entry_dimensions(entry)) or "-"
                combo = estimate_entry_count(entry)
                total += combo
                lines.append(
                    f"| `{fmt_cell(dtype)}` | "
                    f"`{fmt_cell(entry.get('path'))}` | "
                    f"`{fmt_cell(entry.get('key'))}` | "
                    f"{fmt_cell(dims)} | {combo} |"
                )
        if not per_dtype:
            lines.append("| - | - | - | - | - |")
        if "group_dims" in group:
            lines.extend(["", f"**group_dims**：{fmt_cell(group.get('group_dims'))}"])
        lines.extend(["", f"**预估组合数**：约 {total}", ""])
    if not groups:
        lines.append("无 group。")
    return lines


def validate_model(path_list, param_def, cases):
    if not isinstance(cases, list):
        raise ValueError(f"S2P2_cases.json must be a list, got {type(cases).__name__}")
    checks = []
    paths = path_list.get("paths", [])
    path_ids = [p.get("id") for p in paths]
    path_id_set = set(path_ids)
    groups = param_def.get("groups", [])
    group_ids = [g.get("id") for g in groups]

    checks.append(("输入文件存在且 JSON 可解析", True, "parse_inputs 已完成"))
    checks.append((
        "path ID 唯一且非空",
        len(path_ids) == len(path_id_set) and all(path_ids),
        f"{len(path_id_set)}/{len(path_ids)}",
    ))
    checks.append(("group ID 和顺序可读取", all(group_ids), ", ".join(str(g) for g in group_ids)))

    reachable_without_group = [
        p.get("id") for p in paths
        if p.get("reachability") == "reachable" and not p.get("group")
    ]
    checks.append(("reachable path 必须有 group", not reachable_without_group, fmt_cell(reachable_without_group)))

    group_path_refs = []
    for group in groups:
        for entries in group.get("per_dtype", {}).values():
            group_path_refs.extend(entry.get("path") for entry in entries)
    missing_group_paths = sorted({p for p in group_path_refs if p not in path_id_set})
    checks.append(("group 引用 path 必须存在", not missing_group_paths, fmt_cell(missing_group_paths)))

    case_path_refs = [c.get("path") for c in cases if "path" in c]
    missing_case_paths = sorted({p for p in case_path_refs if p not in path_id_set})
    checks.append(("case 引用 path 必须存在", not missing_case_paths, fmt_cell(missing_case_paths)))
    return checks


def render_checks(checks):
    lines = ["### 4.2 自动一致性校验", "", "| 检查项 | 结果 | 说明 |", "|--------|------|------|"]
    for name, passed, detail in checks:
        lines.append(f"| {fmt_cell(name)} | {'PASS' if passed else 'FAIL'} | {fmt_cell(detail)} |")
    return lines


def collect_unresolved(path_list, operator_model):
    items = []
    by_status = {}
    for path in path_list.get("paths", []):
        status = path.get("reachability")
        if status in {"disputed", "api_warn", "dead", "api_dead"}:
            by_status.setdefault(status, []).append(path.get("id"))
    for status, ids in sorted(by_status.items()):
        items.append((status, ", ".join(f"`{i}`" for i in ids), f"path_list reachability={status}"))

    exposure = operator_model.get("torch_npu_api_exposure", {})
    for gap in exposure.get("param_gaps", []):
        param = gap.get("aclnn_param") or gap.get("param") or "unknown"
        items.append(("param_gap", f"`{param}`", fmt_cell(gap)))

    checklist = path_list.get("completeness_checklist", {})
    unresolved = checklist.get("unresolved_items", {})
    if isinstance(unresolved, dict):
        for key, detail in unresolved.items():
            items.append(("unresolved_item", f"`{key}`", detail))
    return items


def render_unresolved(items):
    lines = ["## 5. 自动发现的未确认项", ""]
    if not items:
        lines.append("无。")
        return lines
    lines.extend(["| 类型 | 对象 | 说明 |", "|------|------|------|"])
    for item_type, target, detail in items:
        lines.append(f"| {fmt_cell(item_type)} | {fmt_cell(target)} | {fmt_cell(detail)} |")
    return lines


@dataclass
class RenderContext:
    data: dict
    parse_status: dict
    checks: list
    unresolved_items: list


def _render_summary(op_name, output_dir, ctx: RenderContext, stats):
    path_list = ctx.data["S2P1_path_list.json"]
    param_def = ctx.data["S2P2_param_def.json"]
    lines = [SCRIPT_BEGIN, "## 1. 生成与输入摘要", ""]
    lines.extend([
        "| 项目 | 值 |",
        "|------|----|",
        f"| 算子 | `{fmt_cell(op_name)}` |",
        f"| 输出目录 | `{fmt_cell(output_dir)}` |",
        f"| 平台 | {fmt_cell(param_def.get('platform', path_list.get('platform')))} |",
        f"| path 数 | {stats['path_count']} |",
        f"| group 数 | {stats['group_count']} |",
        f"| case 数 | {stats['case_count']} |",
        f"| low config 数 | {stats['low_config_count']} |",
        f"| reachability 分布 | {fmt_cell(dict(stats['reachability']))} |",
        "",
        "### 1.1 输入产物清单",
        "",
        "| 文件 | 状态 | 路径 |",
        "|------|------|------|",
    ])
    for name, (ok, status) in ctx.parse_status.items():
        lines.append(f"| `{name}` | {'OK' if ok else 'MISSING'}: {fmt_cell(status)} | `{output_dir / name}` |")
    return lines


def _render_interface(operator_model):
    lines = ["", "## 2. 接口与参数模型", ""]
    lines.extend(render_io_table("Inputs", operator_model.get("inputs", [])))
    lines.append("")
    lines.extend(render_io_table("Outputs", operator_model.get("outputs", [])))
    lines.append("")
    lines.extend(render_attributes(operator_model.get("attributes", [])))
    if operator_model.get("torch_npu_api_exposure"):
        api_exp = json.dumps(
            operator_model["torch_npu_api_exposure"],
            indent=2, ensure_ascii=False,
        )
        lines.extend(["", "#### API 暴露补充", "", f"```json\n{api_exp}\n```"])
    return lines


def _render_cases(param_def, cases, stats, checks, unresolved_items):
    lines = ["## 4. Case 枚举与一致性校验", "", "### 4.1 Case 枚举摘要", ""]
    lines.extend(["| 项目 | 值 |", "|------|----|"])
    lines.append(f"| case 总数 | {stats['case_count']} |")
    lines.append(f"| 按 group 分布 | {fmt_cell(dict(stats['case_groups']))} |")
    lines.append(f"| 按 key 分布 | {fmt_cell(dict(stats['case_keys']))} |")
    for item in param_def.get("dtype_tensors", []):
        param = item.get("param")
        if param:
            lines.append(f"| 按 `{param}` 分布 | {fmt_cell(dict(Counter(c.get(param, 'missing') for c in cases)))} |")
    lines.append("")
    lines.extend(render_checks(checks))
    lines.append("")
    lines.extend(render_unresolved(unresolved_items))
    lines.append(SCRIPT_END)
    return lines


def render_script_section(op_name, output_dir, ctx: RenderContext):
    path_list = ctx.data["S2P1_path_list.json"]
    operator_model = ctx.data["S2P1_operator_model.json"]
    param_def = ctx.data["S2P2_param_def.json"]
    cases = ctx.data["S2P2_cases.json"]
    stats = collect_stats(path_list, param_def, ctx.data["S2P1_low_configs.json"], cases)

    lines = _render_summary(op_name, output_dir, ctx, stats)
    lines.extend(_render_interface(operator_model))
    lines.extend(["", "## 3. 路径与 Group 覆盖", ""])
    lines.extend(render_paths(path_list.get("paths", [])))
    lines.append("")
    lines.extend(render_groups(param_def.get("groups", [])))
    lines.extend(_render_cases(param_def, cases, stats, ctx.checks, ctx.unresolved_items))
    return "\n".join(lines).rstrip() + "\n"


def default_llm_section():
    return "\n".join([
        LLM_BEGIN,
        "## 6. 测试设计分析",
        "",
        "### 6.1 事实摘要与设计结论",
        "（由 LLM 基于脚本生成区和 Step 2 产物填写；不得修改脚本生成区事实。）",
        "",
        "### 6.2 关键派生变量",
        "（由 LLM 基于 S2P2_traceability.md / source_constraints 填写。）",
        "",
        "### 6.3 执行模式分析",
        "（由 LLM 基于脚本生成区和 traceability 填写；信息不足时应明确说明。）",
        "",
        "## 7. 风险与补充建议",
        "（由 LLM 汇总 api_warn / dead / disputed / param_gaps 等风险与处理建议。）",
        LLM_END,
        "",
    ])


def default_verifier_section():
    return "\n".join([
        VERIFIER_BEGIN,
        "## 8. Step 3 验证结论（原 §9 验证结论）",
        "（Step 3 完成后由 verifier 填写）",
        VERIFIER_END,
        "",
    ])


def replace_section(text, begin, end, new_section):
    if begin not in text or end not in text:
        return None
    start = text.index(begin)
    finish = text.index(end, start) + len(end)
    return text[:start] + new_section.rstrip() + text[finish:]


def assemble_document(op_name, output_path, script_section, overwrite_llm=False, force=False):
    title = f"# {op_name} 白盒测试设计\n\n"
    llm_section = default_llm_section()
    verifier_section = default_verifier_section()

    if force or not output_path.exists():
        return title + script_section + "\n" + llm_section + verifier_section

    existing = output_path.read_text(encoding="utf-8")
    if not existing.startswith("# "):
        existing = title + existing
    updated = replace_section(existing, SCRIPT_BEGIN, SCRIPT_END, script_section)
    if updated is None:
        updated = title + script_section + "\n" + llm_section + verifier_section
    if overwrite_llm:
        updated2 = replace_section(updated, LLM_BEGIN, LLM_END, llm_section)
        updated = updated2 if updated2 is not None else updated.rstrip() + "\n\n" + llm_section
    if VERIFIER_BEGIN not in updated or VERIFIER_END not in updated:
        updated = updated.rstrip() + "\n\n" + verifier_section
    return updated


def validate_document(text, path_list, param_def):
    errors = []
    for marker in [SCRIPT_BEGIN, SCRIPT_END, LLM_BEGIN, LLM_END, VERIFIER_BEGIN, VERIFIER_END]:
        if marker not in text:
            errors.append(f"missing marker: {marker}")
    for section in REQUIRED_SECTIONS:
        if section not in text:
            errors.append(f"missing section: {section}")
    for path in path_list.get("paths", []):
        pid = path.get("id")
        if pid and f"`{pid}`" not in text:
            errors.append(f"missing path id in document: {pid}")
    for group in param_def.get("groups", []):
        gid = group.get("id")
        if gid and str(gid) not in text:
            errors.append(f"missing group id in document: {gid}")
    return errors


def main():
    logging.basicConfig(format="%(message)s")
    parser = argparse.ArgumentParser(description="Generate S2P3_test_design.md from Step 2 artifacts")
    parser.add_argument("--op-name", required=True, help="Operator name")
    parser.add_argument("--op-path", required=True, help="Operator source path")
    parser.add_argument("--output-dir", help="Output directory (default: {op_path}/tests/whitebox)")
    parser.add_argument("--force", action="store_true", help="Rewrite the whole document")
    parser.add_argument("--overwrite-llm-section", action="store_true", help="Reset the LLM analysis section")
    args = parser.parse_args()

    op_path = Path(args.op_path).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else op_path / "tests" / "whitebox"
    output_path = output_dir / "S2P3_test_design.md"

    data, parse_status = parse_inputs(output_dir)
    path_list = data["S2P1_path_list.json"]
    param_def = data["S2P2_param_def.json"]
    cases = data["S2P2_cases.json"]
    operator_model = data["S2P1_operator_model.json"]
    if not isinstance(cases, list):
        parser.error(f"S2P2_cases.json must be a list, got {type(cases).__name__}")

    checks = validate_model(path_list, param_def, cases)
    failed_checks = [name for name, passed, _ in checks if not passed]
    if failed_checks:
        raise ValueError("model validation failed: " + "; ".join(failed_checks))

    unresolved_items = collect_unresolved(path_list, operator_model)
    ctx = RenderContext(data, parse_status, checks, unresolved_items)
    script_section = render_script_section(args.op_name, output_dir, ctx)
    document = assemble_document(
        args.op_name,
        output_path,
        script_section,
        overwrite_llm=args.overwrite_llm_section,
        force=args.force,
    )
    doc_errors = validate_document(document, path_list, param_def)
    if doc_errors:
        raise ValueError("document validation failed:\n" + "\n".join(f"  - {e}" for e in doc_errors))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    stats = collect_stats(path_list, param_def, data["S2P1_low_configs.json"], cases)
    _logger.info("Generated %s", output_path)
    _logger.info("Paths: %d", stats["path_count"])
    _logger.info("Groups: %d", stats["group_count"])
    _logger.info("Cases: %d", stats["case_count"])
    _logger.info("Reachability: %s", dict(stats["reachability"]))


if __name__ == "__main__":
    main()
