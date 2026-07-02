#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""dump_rule_ids — 抓取 scripts/ 下所有 rule_id 字面量，生成 references/error-codes.md。

工作模式：
  python3 scripts/dump_rule_ids.py            # 打印 markdown 到 stdout
  python3 scripts/dump_rule_ids.py --write    # 写回 references/error-codes.md
  python3 scripts/dump_rule_ids.py --check    # 比对现存文件；不一致 exit 1（CI 守门）

抓取范围：
  * "rule_id": "..." / rule_id="..." 字面量（不含 f-string）
  * DslError("<code>", ...) — 这些 code 在 stages.py 中按 "<stage>.<code>" 复用
所以输出包含两份：直接 rule_id + evaluator 错误码引用一栏。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = SKILL_ROOT / "scripts"
DOC = SKILL_ROOT / "references" / "error-codes.md"


# rule_id: "..."  /  rule_id = "..."
_RULE_RE = re.compile(r'rule_id\s*[":=]+\s*"([a-z_][a-z0-9_.]+)"')
# DslError("<code>", ...
_DSL_RE = re.compile(r'DslError\(\s*"([a-z_][a-z0-9_]*)"')


def _scan() -> tuple[set[str], set[str]]:
    rule_ids: set[str] = set()
    dsl_codes: set[str] = set()
    for p in SCRIPTS.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        text = p.read_text(encoding="utf-8")
        for m in _RULE_RE.finditer(text):
            rule_ids.add(m.group(1))
        for m in _DSL_RE.finditer(text):
            dsl_codes.add(m.group(1))
    return rule_ids, dsl_codes


def _group_by_stage(ids: set[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for rid in sorted(ids):
        if "." in rid:
            stage, _ = rid.split(".", 1)
        else:
            stage = "<misc>"
        grouped.setdefault(stage, []).append(rid)
    return grouped


_STAGE_LABEL = {
    "schema_static": "Stage 1 schema_static",
    "category_paradigm_consistency": "Stage 2 category_paradigm_consistency",
    "paradigm_constraint": "Stage 2 paradigm_constraint",
    "invariant_kind_resolved": "Stage 2 invariant_kind_resolved",
    "shape_closure": "Stage 3 shape_closure",
    "dtype_closure": "Stage 4 dtype_closure",
    "broadcast_legality": "Stage 5 broadcast_legality",
    "boundary_min_set": "Stage 6 boundary_min_set",
    "tolerance_coverage": "Stage 7 tolerance_coverage",
    "formula_smoke_eval": "Stage 8 formula_smoke_eval",
    "oracle_reachable": "Stage 9 oracle_reachable",
}


def render(rule_ids: set[str], dsl_codes: set[str]) -> str:
    grouped = _group_by_stage(rule_ids)
    lines = [
        "# Validator rule_id 目录",
        "",
        "> 由 `scripts/dump_rule_ids.py` 自动生成；改动 validator 后跑 `--write` 同步。CI 由",
        "> `tests/test_doc_drift.py` 守门：源文件中的 rule_id 与本文件不一致即 fail。",
        "",
        "本目录列出 9-stage L0 校验器输出的所有 `rule_id` 字面量。新增 finding 时把字符串",
        "登记到对应 stage；若是 evaluator（numpy AST 求值器）内部抛 `DslError`，code 会在 stages.py 拼成",
        "`<stage>.<code>`，所以**新增 evaluator 错误码必须同时拓宽 SKILL.md §4.x 表**。",
        "",
    ]

    seen = set()
    for stage_key, label in _STAGE_LABEL.items():
        if stage_key not in grouped:
            continue
        seen.add(stage_key)
        lines.append(f"## {label}")
        lines.append("")
        for rid in grouped[stage_key]:
            lines.append(f"- `{rid}`")
        lines.append("")

    other = sorted(set(grouped) - seen)
    if other:
        lines.append("## 其他 / 未归类")
        lines.append("")
        for k in other:
            for rid in grouped[k]:
                lines.append(f"- `{rid}`")
        lines.append("")

    lines.append("## Evaluator 错误码（`DslError(code, ...)`）")
    lines.append("")
    lines.append("以下错误码由 `scripts/evaluators/` 内部抛出，被 stage 3-5 / 8 / 9 包装为 `<stage>.<code>`。")
    lines.append("")
    for code in sorted(dsl_codes):
        lines.append(f"- `{code}`")
    lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    args = p.parse_args(argv)

    rule_ids, dsl_codes = _scan()
    text = render(rule_ids, dsl_codes)

    if args.write:
        DOC.write_text(text, encoding="utf-8")
        print(f"WROTE: {DOC} ({len(rule_ids)} rule_ids + {len(dsl_codes)} DSL codes)")
        return 0
    if args.check:
        existing = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
        if existing == text:
            print(f"OK: {len(rule_ids)} rule_ids + {len(dsl_codes)} DSL codes")
            return 0
        print("DRIFT: references/error-codes.md 与源码不一致；跑：", file=sys.stderr)
        print("  python3 scripts/dump_rule_ids.py --write", file=sys.stderr)
        return 1
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
