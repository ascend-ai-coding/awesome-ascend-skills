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
"""Step 3 Diversity Report — analyze S2P1_cases.json parameter diversity and basic info."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def load_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "cases" in data:
        return list(data["cases"])
    raise ValueError(f"invalid cases file: {path}")


def analyze_basic(cases: list[dict[str, Any]]) -> dict[str, Any]:
    reachability = Counter(c.get("reachability", "unknown") for c in cases)
    path_ids = set(c.get("path_id") for c in cases if c.get("path_id"))
    tiling_keys = set(c.get("tiling_key") for c in cases if c.get("tiling_key") is not None)
    return {
        "total_cases": len(cases),
        "distinct_paths": len(path_ids),
        "distinct_tiling_keys": len(tiling_keys),
        "reachability": dict(reachability),
    }


def analyze_diversity(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    all_keys: set[str] = set()
    for c in cases:
        all_keys.update(c.get("case_values", {}).keys())

    results = []
    for key in sorted(all_keys):
        values = [c["case_values"][key] for c in cases if key in c.get("case_values", {})]
        distinct = {_hashable(v) for v in values}
        entry: dict[str, Any] = {"parameter": key, "distinct": len(distinct)}

        numeric = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if numeric and len(numeric) == len(values):
            entry["type"] = "float" if any(isinstance(v, float) for v in values) else "int"
            entry["range"] = f"[{min(numeric)}, {max(numeric)}]"
        else:
            entry["type"] = "str"
            str_vals = sorted(str(v) for v in distinct)
            if len(str_vals) <= 10:
                entry["range"] = ", ".join(str_vals)
            else:
                entry["range"] = f"{len(str_vals)} values"

        results.append(entry)
    return results


def _hashable(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    return value


def analyze_paths(cases: list[dict[str, Any]]) -> dict[str, Any]:
    path_witnesses: Counter[str] = Counter()
    for c in cases:
        pid = c.get("path_id")
        if pid:
            path_witnesses[pid] += 1
    if not path_witnesses:
        return {"paths": 0, "min_witnesses": 0, "max_witnesses": 0, "avg_witnesses": 0}
    counts = list(path_witnesses.values())
    return {
        "paths": len(path_witnesses),
        "min_witnesses": min(counts),
        "max_witnesses": max(counts),
        "avg_witnesses": round(sum(counts) / len(counts), 1),
    }


def analyze_reachability(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Any]] = {}
    for c in cases:
        r = c.get("reachability", "unknown")
        groups.setdefault(r, []).append(c.get("tiling_key"))

    results = []
    for r in sorted(groups):
        keys = sorted(set(k for k in groups[r] if k is not None))
        key_str = ", ".join(str(k) for k in keys)
        results.append({"reachability": r, "cases": len(groups[r]), "tiling_keys": key_str})
    return results


def analyze_path_coverage(path_list_path: Path) -> dict[str, list[int]] | None:
    if not path_list_path.exists():
        return None
    data = json.loads(path_list_path.read_text(encoding="utf-8"))
    reachable_keys = sorted(set(
        p["tiling_key"] for p in data.get("paths", [])
        if p.get("reachability") == "reachable"
    ))
    tiling_no_kernel_keys = sorted(set(
        p["tiling_key"] for p in data.get("paths", [])
        if p.get("reachability") == "tiling_no_kernel"
    ))
    orphan_keys = sorted(set(
        orphan["tiling_key"] for orphan in data.get("kernel_orphans", [])
    ))
    return {
        "reachable": reachable_keys,
        "tiling_no_kernel": tiling_no_kernel_keys,
        "orphan": orphan_keys,
    }


def build_conclusion(diversity: list[dict[str, Any]]) -> tuple[str, list[str]]:
    warnings = []
    for entry in diversity:
        if entry["distinct"] <= 1:
            warnings.append(f"parameter '{entry['parameter']}' is constant (distinct={entry['distinct']})")
    if warnings:
        return "WARN", warnings
    return "PASS", []


def render_md(
    operator: str,
    conclusion: str,
    warnings: list[str],
    basic: dict[str, Any],
    diversity: list[dict[str, Any]],
    paths: dict[str, Any],
    reachability: list[dict[str, Any]],
    path_coverage: dict[str, list[int]] | None,
) -> str:
    lines = [f"# S3 多样性报告 — {operator}\n"]

    warning_items = ", ".join(warnings) if warnings else "none"
    lines.append("\n## Agent Summary\n\n| 字段 | 值 |\n|---|---|\n")
    lines.append(f"| cases | {basic['total_cases']} |\n")
    lines.append(f"| paths | {basic['distinct_paths']} |\n")
    lines.append(f"| params | {len(diversity)} |\n")
    lines.append(f"| warnings | {len(warnings)} |\n")
    lines.append(f"| warning_items | {warning_items} |\n")

    lines.append(f"\n## 结论\n\n**{conclusion}**")
    if conclusion == "PASS":
        lines.append(" — 所有参数均有多样性覆盖。\n")
    else:
        lines.append(" — 存在多样性不足：\n")
        for w in warnings:
            lines.append(f"- {w}\n")

    lines.append("\n## 基本信息\n\n| 指标 | 值 |\n|---|---|\n")
    lines.append(f"| 用例总数 | {basic['total_cases']} |\n")
    lines.append(f"| 路径数 | {basic['distinct_paths']} |\n")
    lines.append(f"| Tiling Key 数 | {basic['distinct_tiling_keys']} |\n")
    for r, cnt in sorted(basic["reachability"].items()):
        lines.append(f"| {r} | {cnt} |\n")

    if path_coverage:
        lines.append("\n## 路径覆盖\n")
        r_keys = path_coverage["reachable"]
        t_keys = path_coverage["tiling_no_kernel"]
        o_keys = path_coverage["orphan"]
        lines.append(f"\n### 可达路径（{len(r_keys)} 条）\n\n")
        lines.append(", ".join(str(k) for k in r_keys) + "\n")
        lines.append(f"\n### Tiling 可达 / Kernel 不可达（{len(t_keys)} 条）\n\n")
        lines.append(", ".join(str(k) for k in t_keys) + "\n" if t_keys else "无\n")
        lines.append(f"\n### Kernel Orphan（{len(o_keys)} 条）\n\n")
        lines.append(", ".join(str(k) for k in o_keys) + "\n" if o_keys else "无\n")

    lines.append("\n## 参数多样性\n\n| 参数 | 去重数 | 类型 | 范围 |\n|---|---|---|---|\n")
    for entry in diversity:
        lines.append(f"| {entry['parameter']} | {entry['distinct']} | {entry['type']} | {entry['range']} |\n")

    lines.append("\n## 路径分布\n\n| 指标 | 值 |\n|---|---|\n")
    labels = {
        "paths": "路径数",
        "min_witnesses": "最小 witness",
        "max_witnesses": "最大 witness",
        "avg_witnesses": "平均 witness",
    }
    for k, v in paths.items():
        lines.append(f"| {labels.get(k, k)} | {v} |\n")

    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze S2P1_cases.json diversity.")
    parser.add_argument("--cases", required=True, help="S2P1_cases.json path")
    parser.add_argument("--path-list", default=None, help="S2P1_path_list.json path (optional)")
    parser.add_argument("--output", default="S3_diversity_report.md")
    parser.add_argument("--operator", default="unknown")
    args = parser.parse_args()

    cases_path = Path(args.cases)
    if not cases_path.exists():
        print(f"ERROR: {cases_path} not found", file=sys.stderr)
        return 1

    try:
        cases = load_cases(cases_path)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    basic = analyze_basic(cases)
    diversity = analyze_diversity(cases)
    paths = analyze_paths(cases)
    reachability = analyze_reachability(cases)
    conclusion, warnings = build_conclusion(diversity)

    path_coverage = None
    if args.path_list:
        path_coverage = analyze_path_coverage(Path(args.path_list))

    md = render_md(args.operator, conclusion, warnings, basic, diversity, paths, reachability, path_coverage)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")

    print(f"S3 diversity report: {conclusion}")
    print(f"  cases={basic['total_cases']}, paths={basic['distinct_paths']}, params={len(diversity)}")
    if warnings:
        for w in warnings:
            print(f"  WARN: {w}", file=sys.stderr)
    print(f"Written -> {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
