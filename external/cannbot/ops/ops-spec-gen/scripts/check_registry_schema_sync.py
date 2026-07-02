#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""check_registry_schema_sync — 校验 registries/*.yaml 与 schemas/op-spec.json 的 enum 同步。

背景：schema enum 与 registry 当前需要手工同步；漂移会让 spec 看起来通过 stage 1 但
stage 2 报名字未知，或反过来。本脚本批量检查所有已对应 enum，差异 ⇒ exit 1。

不在本脚本范围（这些 registry 的语义不是简单 enum）：
  * category_paradigm_map.yaml — 映射，不是单一 enum
  * boundary_min_cases.yaml    — 数据驱动 stage 6
  * tolerance_defaults.yaml    — 默认值，不是 enum
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


SKILL_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = SKILL_ROOT / "schemas" / "op-spec.json"
REG_DIR = SKILL_ROOT / "registries"


# 每条规则：(registry 文件, 取出 set 的 callable, schema 中 dotted JSON pointer 起点)
def _from_categories(reg: dict) -> set:
    return set(reg["categories"])


def _from_paradigms(reg: dict) -> set:
    # paradigms 列表元素是 dict {name, group, priority}
    return {item["name"] for item in reg["paradigms"]}


def _from_primitives(reg: dict) -> set:
    # primitives 列表元素是裸字符串
    return set(reg["primitives"])


def _from_invariant_kinds(reg: dict) -> set:
    out: set = set()
    for group in ("value", "algebraic", "structural"):
        for item in reg.get(group, []) or []:
            out.add(item["kind"])
    return out


def _from_machine_check(reg: dict) -> set:
    return {item["kind"] for item in reg["machine_check_kinds"]}


def _from_synthesize(reg: dict) -> set:
    # patterns 列表元素是 dict {name, takes_value}
    return {item["name"] for item in reg["patterns"]}


def _from_frameworks(reg: dict) -> set:
    return set(reg["frameworks"])


def _from_chips(reg: dict) -> set:
    return {c["id"] for c in reg["chips"]}


def _from_error_codes(reg: dict) -> set:
    return set(reg["error_codes"])


# schema enum 取值器：传入 schema dict，返回对应 enum 的 set
def _schema_category(s: dict) -> set:
    return set(s["properties"]["op"]["properties"]["category"]["enum"])


def _schema_paradigm(s: dict) -> set:
    return set(s["properties"]["op"]["properties"]["paradigms"]["items"]["enum"])


def _schema_primitive(s: dict) -> set:
    return set(s["properties"]["math_semantics"]["properties"]["composition"]
               ["properties"]["primitives"]["items"]["properties"]["op"]["enum"])


def _schema_invariant_kind(s: dict) -> set:
    return set(s["properties"]["math_semantics"]["properties"]["invariants"]
               ["items"]["properties"]["kind"]["enum"])


def _schema_machine_check(s: dict) -> set:
    # 在 $defs.case_with_check.properties.machine_check.properties.kind
    return set(s["$defs"]["case_with_check"]["properties"]["machine_check"]
               ["properties"]["kind"]["enum"])


def _schema_framework(s: dict) -> set:
    return set(s["properties"]["math_semantics"]["properties"]
               ["reference_oracle"]["properties"]["framework"]["enum"])


def _schema_chips(s: dict) -> set:
    return set(s["properties"]["op"]["properties"]
               ["platform_constraints"]["properties"]["supported_chips"]["items"]["enum"])


def _schema_error_codes(s: dict) -> set:
    return set(s["properties"]["op"]["properties"]["error_codes"]["items"]["enum"])


CHECKS = [
    ("category", "category_enum.yaml", _from_categories, _schema_category),
    ("paradigm", "paradigm_enum.yaml", _from_paradigms, _schema_paradigm),
    ("primitive_op", "primitive_whitelist.yaml", _from_primitives, _schema_primitive),
    ("invariant_kind", "invariant_kind_registry.yaml", _from_invariant_kinds, _schema_invariant_kind),
    ("machine_check", "machine_check_kind_registry.yaml", _from_machine_check, _schema_machine_check),
    ("framework", "framework_oracle_registry.yaml", _from_frameworks, _schema_framework),
    ("chip", "chip_registry.yaml", _from_chips, _schema_chips),
    ("error_code", "error_code_enum.yaml", _from_error_codes, _schema_error_codes),
]


def main(argv: list[str] | None = None) -> int:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    failures: list[str] = []

    for label, reg_file, reg_extractor, schema_extractor in CHECKS:
        reg_path = REG_DIR / reg_file
        reg = yaml.safe_load(reg_path.read_text(encoding="utf-8"))
        try:
            reg_set = reg_extractor(reg)
        except Exception as e:
            failures.append(f"[{label}] 解析 registry {reg_file} 失败: {e}")
            continue
        try:
            schema_set = schema_extractor(schema)
        except Exception as e:
            failures.append(f"[{label}] 解析 schema enum 失败: {e}")
            continue

        only_in_reg = reg_set - schema_set
        only_in_schema = schema_set - reg_set
        if only_in_reg or only_in_schema:
            failures.append(
                f"[{label}] 不同步:\n"
                f"    仅在 {reg_file}: {sorted(only_in_reg)}\n"
                f"    仅在 schema:    {sorted(only_in_schema)}"
            )

    if failures:
        print("registry/schema enum 同步失败：", file=sys.stderr)
        for f in failures:
            print(f, file=sys.stderr)
        return 1

    print(f"OK: {len(CHECKS)} 项 enum 全部同步")
    return 0


if __name__ == "__main__":
    sys.exit(main())
