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
"""build_path_list.py — 从 path_config.json + scout_k.json 组装 S2P1_path_list.json

脚本完全通用，不含任何算子特定逻辑。所有算子特定信息通过 path_config.json 声明。

功能：
1. 合并降级路径到 paths 数组
2. 查 scout_k.json 按 tiling_key 补全 kernel 行号 → source 字段
3. 从 glossary 数组提取 variable_classification，补全到每条路径
4. 孤儿 dispatch 检测（scout_k keys − 已声明 keys → D{N} dead 路径）
5. 组装 completeness_checklist（dispatch_coverage 自动填充，其余从 config 复制）
6. 生成 S2P1_tiling_glossary.md
7. name 自动生成（path_{id}_key{tiling_key}）
8. conditions 紧凑格式解析（字符串 → 标准 JSON 数组）
9. schema 校验

用法：
  python3 build_path_list.py \\
    --config S2P1_path_config.json \\
    --scout-k S2P0_scout_k.json \\
    --output-dir tests/whitebox/
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass

_logger = logging.getLogger(__name__)


@dataclass
class _BuildResult:
    paths: list
    total_scout_keys: int
    declared_keys: set
    orphan_keys: list
    tiling_no_kernel: list
    errors: list


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_VALID_CATEGORIES = ("input_variable", "caller_option", "internal_variable")


def extract_vc(glossary):
    """从 glossary 数组的 category 字段提取 variable_classification。"""
    vc = {"input_variables": [], "caller_options": [], "internal_variables": []}
    for entry in glossary:
        cat = entry.get("category")
        var = entry["tiling_var"]
        if cat == "input_variable":
            vc["input_variables"].append(var)
        elif cat == "caller_option":
            vc["caller_options"].append(var)
        elif cat == "internal_variable":
            vc["internal_variables"].append(var)
    return vc


def _parse_value(s):
    """解析右侧值：引号→字符串常量，true/false→布尔，数字→int/float，其他→None(变量)"""
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1]
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return None


_CMP_OPS = r"==|!=|>=|<=|>|<"


def parse_condition(line):
    """解析单行紧凑条件为标准 JSON dict。

    支持格式：
    - { 开头 → JSON 对象原样解析（混合模式兜底）
    - boundary:EXPR → {"boundary_check": "EXPR"}
    - VAR range MIN..MAX → range 条件
    - VAR in [LIST] → in 条件
    - VAR%NUM OP VAL → mod_eq 条件
    - EXPR%VAR OP VAL → expr 条件
    - VAR OP RHS → 比较条件（RHS 数字/引号/布尔→value，标识符→ref）
    """
    line = line.strip()
    if not line:
        return None

    # 混合模式：以 { 开头 → JSON 对象
    if line.startswith("{"):
        return json.loads(line)

    # boundary_check
    if line.startswith("boundary:"):
        return {"boundary_check": line[len("boundary:"):]}

    # range: VAR range MIN..MAX
    m = re.match(r"^(\w+)\s+range\s+(\d+)\.\.(\d+)$", line)
    if m:
        return {"var": m.group(1), "op": "range",
                "min": int(m.group(2)), "max": int(m.group(3))}

    # in: VAR in [v1,v2,...]
    m = re.match(r"^(\w+)\s+in\s+\[(.*)\]$", line)
    if m:
        raw_vals = [v.strip() for v in m.group(2).split(",") if v.strip()]
        vals = []
        for v in raw_vals:
            parsed = _parse_value(v)
            vals.append(parsed if parsed is not None else v.strip('"\''))
        return {"var": m.group(1), "op": "in", "value": vals}

    # mod_eq: VAR%NUM==VAL (% 右侧是数字)
    m = re.match(rf"^(\w+)%(\d+)({_CMP_OPS})(.+)$", line)
    if m:
        return {"var": m.group(1), "op": "mod_eq",
                "divisor": int(m.group(2)),
                "remainder": _parse_value(m.group(4))}

    # expr: EXPR%VAR OP VAL (% 右侧是标识符)
    m = re.match(rf"^(.+?)%(\w+)({_CMP_OPS})(.+)$", line)
    if m:
        return {"expr": f"{m.group(1)} % {m.group(2)}",
                "op": m.group(3),
                "value": _parse_value(m.group(4))}

    # 比较运算符: VAR OP RHS
    m = re.match(rf"^(\w+)({_CMP_OPS})(.+)$", line)
    if m:
        var, op, rhs = m.group(1), m.group(2), m.group(3)
        val = _parse_value(rhs)
        if val is not None:
            return {"var": var, "op": op, "value": val}
        return {"var": var, "op": op, "ref": rhs}

    raise ValueError(f"Cannot parse condition: {line}")


def parse_conditions(cond_str):
    """解析紧凑格式条件字符串为标准 JSON 数组。兼容已为数组的旧格式。"""
    if isinstance(cond_str, list):
        return cond_str
    if not isinstance(cond_str, str) or not cond_str.strip():
        return []
    result = []
    for line in cond_str.strip().split("\n"):
        parsed = parse_condition(line)
        if parsed is not None:
            result.append(parsed)
    return result


def build_key_line_map(scout_k):
    """从 scout_k.json 构建 {tiling_key: kernel_line} 映射，仅含 active keys。"""
    key_line_map = {}
    for entry in scout_k.get("entries", []):
        for key in entry.get("keys", []):
            if key.get("active", False):
                key_line_map[int(key["value"])] = key["line"]
    return key_line_map


def merge_degradations(paths, degradations):
    """将降级路径合并到 paths 数组。降级路径的 conditions = parent conditions + trigger。

    name 自动生成（path_{id}_key{tiling_key}）。
    trigger 支持紧凑格式字符串，为 None 时跳过。
    """
    path_by_id = {p["id"]: p for p in paths}
    for deg in degradations:
        parent = path_by_id.get(deg["parent_id"])
        if parent is None:
            raise ValueError(f"degradation parent_id '{deg['parent_id']}' not found in paths")
        # Parse parent conditions if still string (enrich_paths not yet run)
        parent_conditions = parent.get("conditions", [])
        if isinstance(parent_conditions, str):
            parent_conditions = parse_conditions(parent_conditions)
        trigger = deg["trigger"]
        if isinstance(trigger, str):
            trigger = parse_condition(trigger)
        if trigger is not None:
            merged_conditions = list(parent_conditions) + [trigger]
        else:
            merged_conditions = list(parent_conditions)
        deg_name = deg.get("name", f"path_{deg['id']}_key{deg.get('tiling_key', '?')}")
        paths.append({
            "id": deg["id"],
            "name": deg_name,
            "tiling_key": deg["tiling_key"],
            "conditions": merged_conditions,
            "kernel_class": deg["kernel_class"],
            "tiling_line": deg["tiling_line"],
            "degraded_from": parent["tiling_key"],
        })
    return paths


def enrich_paths(paths, config, key_line_map):
    """富化每条路径：自动生成 name、解析 conditions、补全 source/key_instructions/vc。"""
    tiling_file = config["tiling_file"]
    kernel_file = config["kernel_file"]
    vc = extract_vc(config.get("glossary", []))
    tiling_no_kernel = []

    for path in paths:
        # name 自动生成
        path.setdefault("name", f"path_{path.get('id', '?')}_key{path.get('tiling_key', '?')}")

        # conditions 紧凑格式解析
        cond = path.get("conditions")
        if isinstance(cond, str):
            path["conditions"] = parse_conditions(cond)

        key = path.get("tiling_key")

        # source 字段
        tiling_line = path.pop("tiling_line", None)
        if key in key_line_map:
            kernel_line = key_line_map[key]
            path["source"] = f"{tiling_file}:{tiling_line} -> {kernel_file}:{kernel_line}"
        else:
            path["source"] = f"{tiling_file}:{tiling_line} (no kernel dispatch)"
            if key is not None:
                tiling_no_kernel.append(key)

        # key_instructions（从 kernel_class 转换）
        kernel_class = path.pop("kernel_class", None)
        if kernel_class:
            path["key_instructions"] = [kernel_class]
        else:
            path["key_instructions"] = []

        # variable_classification（从 glossary 提取）
        path["input_variables"] = list(vc["input_variables"])
        path["caller_options"] = list(vc["caller_options"])
        path["internal_variables"] = list(vc["internal_variables"])

    return tiling_no_kernel


def detect_orphans(key_line_map, declared_keys, orphan_explanations):
    """孤儿检测：scout_k active keys − 已声明 keys → D{N} dead 路径。"""
    orphan_keys = sorted(set(key_line_map.keys()) - declared_keys)
    orphan_paths = []
    for i, key in enumerate(orphan_keys):
        explanation = orphan_explanations.get(str(key), {})
        orphan_paths.append({
            "id": f"D{i + 1}",
            "name": explanation.get("name", f"path_D{i + 1}_key{key}"),
            "group": None,
            "reachability": "dead",
            "dead_reason": "kernel_has_impl_but_no_tiling_path",
            "dead_detail": explanation.get(
                "dead_detail",
                f"tiling 无法产生此 key({key})，具体原因未声明"
            ),
            "conditions": [],
            "input_variables": [],
            "caller_options": [],
            "internal_variables": [],
            "key_instructions": explanation.get("key_instructions", []),
            "source": f"kernel_dispatch:{key_line_map[key]} (无 tiling 源码对应)",
            "tiling_key": key,
        })
    return orphan_paths, orphan_keys


def build_checklist(config, orphan_keys, tiling_no_kernel, total_scout_keys, total_paths):
    """组装 completeness_checklist。dispatch_coverage 自动填充，其余从 config 复制。"""
    checklist = dict(config.get("completeness_checklist", {}))

    # 合并脚本检测的 tiling_no_kernel 和 LLM 声明的 tiling_no_kernel_keys
    declared_tnk = config.get("tiling_no_kernel_keys", [])
    all_tnk = sorted(set(tiling_no_kernel + declared_tnk))

    declared_count = total_scout_keys - len(orphan_keys)
    unexplained_orphans = [k for k in orphan_keys if str(k) not in config.get("orphan_explanations", {})]
    # 孤儿全部有 explanation → covered（已作为 dead 路径回收）
    # 存在未解释孤儿 → missing
    dispatch_status = "covered" if not unexplained_orphans else "missing"
    checklist["dispatch_coverage"] = {
        "status": dispatch_status,
        "evidence": [
            f"scout_k active keys: {total_scout_keys}, "
            f"declared: {declared_count}, orphan: {len(orphan_keys)} (all as dead paths), "
            f"tiling_no_kernel: {all_tnk}"
        ],
    }

    unresolved = {}
    if all_tnk:
        unresolved["tiling_no_kernel_keys"] = (
            f"Keys {all_tnk} set by tiling but no TILING_KEY_IS dispatch in kernel."
        )
    if unexplained_orphans:
        unresolved["unexplained_orphan_keys"] = (
            f"Keys {unexplained_orphans} have no explanation in orphan_explanations."
        )
    if unresolved:
        checklist["unresolved_items"] = unresolved

    return checklist


def generate_glossary_md(config):
    """从 glossary 数组生成 S2P1_tiling_glossary.md。"""
    op_name = config["operator"]
    entries = config.get("glossary", [])

    lines = [
        f"# Tiling 变量含义表",
        "",
        f"> 算子：{op_name}",
        "> 数据来源：tiling 源码变量提取",
        "",
        "| tiling_var | semantic_name | category | type | desc | shape_contribution |",
        "|------------|--------------|----------|------|------|--------------------|",
    ]

    category_order = {"input_variable": 0, "caller_option": 1, "internal_variable": 2}
    sorted_entries = sorted(entries, key=lambda e: (category_order.get(e["category"], 99), e["tiling_var"]))

    for e in sorted_entries:
        shape_contribution = format_shape_contribution(e.get("shape_contribution"))
        lines.append(
            f"| `{e['tiling_var']}` | {e['semantic_name']} | {e['category']} | "
            f"{e['type']} | {e['desc']} | {shape_contribution} |"
        )

    return "\n".join(lines) + "\n"


def format_shape_contribution(value):
    if value is None:
        return "-"
    parts = [f"shape_relation={value.get('shape_relation', '')}"]
    if value.get("representative"):
        parts.append(f"representative={value['representative']}")
    return "; ".join(parts)


def _validate_glossary(glossary):
    """校验 glossary 条目，返回 (glossary_vars, errors)。"""
    errors = []
    glossary_vars = set()
    for entry in glossary:
        var = entry.get("tiling_var", "")
        glossary_vars.add(var)
        cat = entry.get("category")
        if cat not in _VALID_CATEGORIES:
            errors.append(f"[INVALID_CATEGORY] glossary entry '{var}': "
                          f"category '{cat}' not in {_VALID_CATEGORIES}")
        if "shape_contribution" not in entry:
            errors.append(f"[MISSING_SHAPE_CONTRIBUTION] glossary entry '{var}': missing shape_contribution")
            continue
        shape_contribution = entry.get("shape_contribution")
        if shape_contribution is None:
            continue
        if not isinstance(shape_contribution, dict):
            errors.append(
                f"[INVALID_SHAPE_CONTRIBUTION] glossary entry '{var}':"
                " shape_contribution must be null or object"
            )
            continue
        if not shape_contribution.get("shape_relation"):
            errors.append(
                f"[INVALID_SHAPE_CONTRIBUTION] glossary entry '{var}':"
                " shape_contribution.shape_relation is required"
            )
    return glossary_vars, errors


def _check_condition_vars(conditions, glossary_vars, pid):
    """检查 conditions 中的变量是否在 glossary 中有记录。"""
    errors = []
    for cond in conditions:
        for field_name in ("var", "ref"):
            v = cond.get(field_name)
            if v and v not in glossary_vars:
                errors.append(f"[UNKNOWN_VAR] {pid}: variable '{v}' not in glossary")
    return errors


def validate(paths, key_line_map, glossary):
    """schema 校验，返回错误列表。"""
    glossary_vars, errors = _validate_glossary(glossary)
    ids = set()

    for p in paths:
        pid = p.get("id", "")
        is_dead = pid.startswith("D")

        if pid in ids:
            errors.append(f"[DUP_ID] {pid}: duplicate id")
        ids.add(pid)

        if not pid:
            errors.append(f"[MISSING_ID] path has no id: {p.get('name', '?')}")

        if not is_dead and not p.get("conditions"):
            errors.append(f"[EMPTY_COND] {pid}: conditions is empty")

        if not is_dead:
            errors.extend(_check_condition_vars(p.get("conditions", []), glossary_vars, pid))

        if not p.get("source"):
            errors.append(f"[MISSING_SRC] {pid}: source is empty")

        if "tiling_key" not in p:
            errors.append(f"[MISSING_KEY] {pid}: tiling_key is missing")
        elif not is_dead and p.get("tiling_key") not in key_line_map:
            errors.append(f"[KEY_NOT_IN_KERNEL] {pid}: tiling_key={p.get('tiling_key')} not in scout_k active keys")

    return errors


def _write_outputs(output_dir, output, config):
    import os
    os.makedirs(output_dir, exist_ok=True)
    path_list_path = os.path.join(output_dir, "S2P1_path_list.json")
    with open(path_list_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    glossary_path = os.path.join(output_dir, "S2P1_tiling_glossary.md")
    with open(glossary_path, "w", encoding="utf-8") as f:
        f.write(generate_glossary_md(config))


def _report_results(result: _BuildResult):
    counts = Counter(
        "orphan" if p["id"].startswith("D") else
        "degraded" if "d" in p["id"] else
        "normal"
        for p in result.paths
    )
    _logger.info("S2P1_path_list.json written: %d paths (%d normal, %d degraded, %d orphan)",
                 len(result.paths), counts.get("normal", 0), counts.get("degraded", 0), counts.get("orphan", 0))
    _logger.info("S2P1_tiling_glossary.md written")
    _logger.info("scout_k active keys: %d, declared: %d, orphan: %d, tiling_no_kernel: %d",
                 result.total_scout_keys, len(result.declared_keys),
                 len(result.orphan_keys), len(result.tiling_no_kernel))
    if result.errors:
        _logger.info("Validation ERRORS:")
        for e in result.errors:
            _logger.info("  %s", e)
    else:
        _logger.info("Validation PASSED")
    if result.tiling_no_kernel:
        _logger.info("tiling_no_kernel keys: %s", result.tiling_no_kernel)
    if result.orphan_keys:
        _logger.info("orphan keys: %s", result.orphan_keys)


def main():
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(description="Build S2P1_path_list.json from path_config + scout_k")
    parser.add_argument("--config", required=True, help="Path to S2P1_path_config.json")
    parser.add_argument("--scout-k", required=True, help="Path to S2P0_scout_k.json")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    config = load_json(args.config)
    scout_k = load_json(args.scout_k)
    key_line_map = build_key_line_map(scout_k)
    total_scout_keys = len(key_line_map)

    paths = merge_degradations(list(config["paths"]), config.get("degradations", []))
    tiling_no_kernel = enrich_paths(paths, config, key_line_map)
    declared_keys = {p.get("tiling_key") for p in paths if "tiling_key" in p}
    orphan_paths, orphan_keys = detect_orphans(
        key_line_map, declared_keys, config.get("orphan_explanations", {})
    )
    paths.extend(orphan_paths)
    checklist = build_checklist(config, orphan_keys, tiling_no_kernel, total_scout_keys, len(paths))
    errors = validate(paths, key_line_map, config.get("glossary", []))

    output = {
        "paths": paths,
        "source_constraints": config.get("source_constraints", []),
        "completeness_checklist": checklist,
    }
    _write_outputs(args.output_dir, output, config)
    build_result = _BuildResult(
        paths=paths,
        total_scout_keys=total_scout_keys,
        declared_keys=declared_keys,
        orphan_keys=orphan_keys,
        tiling_no_kernel=tiling_no_kernel,
        errors=errors,
    )
    _report_results(build_result)
    sys.exit(2 if errors else 0)


if __name__ == "__main__":
    main()
