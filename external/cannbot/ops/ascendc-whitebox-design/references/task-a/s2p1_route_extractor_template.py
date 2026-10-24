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
"""Task A Route Evaluator template.

Copy this file with cp as S2P1_route_extract.py and implement only the dynamic region.
Do not rewrite the static framework.
Run: python3 S2P1_route_extract.py --operator <op> --cases S2P1_route_cases.json [--cases-output S2P1_cases.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = "S2P1_path_list.task-a"


@dataclass(frozen=True)
class Context:
    operator: str
    root: Path
    output_path: Path
    cases_path: Path
    cases_output_path: Path


def fail(message: str) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run route evaluators and build S2P1 path graph.")
    parser.add_argument("--operator", required=True)
    parser.add_argument("--cases", required=True, help="Route cases JSON input")
    parser.add_argument("--output", default="S2P1_path_list.json")
    parser.add_argument("--cases-output", default="S2P1_cases.json")
    return parser.parse_args()


def build_context(args: argparse.Namespace) -> Context:
    def resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else ROOT / path

    return Context(args.operator, ROOT, resolve(args.output), resolve(args.cases), resolve(args.cases_output))


def load_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "cases" in data:
        return list(data["cases"])
    raise ValueError(f"invalid route cases file: {path}")


def stable_id(obj: Any, prefix: str) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:8]
    return f"{prefix}_{digest}"


def run_evaluators(ctx: Context, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for index, case in enumerate(cases):
        case_id = str(case.get("id", f"RC{index}"))
        validate_route_case(case, case_id)
        tiling = eval_tiling_route(case)
        if tiling is None:
            results.append({"case_id": case_id, "tiling": None, "kernel": None})
            continue
        kernel = eval_kernel_route(case, tiling)
        results.append({"case_id": case_id, "tiling": tiling, "kernel": kernel})
    return results


def dedup_table(results: list[dict[str, Any]], field: str, prefix: str) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    witness_map: dict[str, list[str]] = {}
    for result in results:
        fact = result[field]
        if fact is None:
            continue
        hint = str(fact.get("id_hint") or stable_id(fact, prefix))
        if hint not in table:
            entry = dict(fact)
            entry.pop("id_hint", None)
            entry["id"] = hint
            table[hint] = entry
            witness_map[hint] = []
        witness_map[hint].append(result["case_id"])
    for fact_id, witnesses in witness_map.items():
        table[fact_id]["witness_route_case_ids"] = sorted(set(witnesses))
    return table


def build_paths(
    results: list[dict[str, Any]],
    tiling_paths: dict[str, dict[str, Any]],
    kernel_branches: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    edge_map: dict[str, dict[str, Any]] = {}
    unresolved: list[dict[str, Any]] = []

    for result in results:
        case_id = result["case_id"]
        tiling = result["tiling"]
        kernel = result["kernel"]

        if tiling is None:
            unresolved.append({
                "id": stable_id({"case_id": case_id}, "R"),
                "side": "tiling",
                "reason": "eval_tiling_route returned None for route case",
                "evidence": [],
                "witness_route_case_ids": [case_id],
            })
            continue

        t_id = str(tiling.get("id_hint") or stable_id(tiling, "T"))
        t_key = int(tiling.get("tiling_key", -1))

        if kernel is None:
            edge_key = f"{t_id}|missing"
            if edge_key not in edge_map:
                edge_map[edge_key] = {
                    "id": f"P{len(edge_map)}",
                    "name": f"path_P{len(edge_map)}_{t_id}_missing",
                    "reachability": "tiling_no_kernel",
                    "tiling_path_id": t_id,
                    "kernel_branch_id": None,
                    "tiling_key": t_key,
                    "reason": "no kernel dispatch reached by evaluator",
                    "witness_route_case_ids": [],
                }
            edge_map[edge_key]["witness_route_case_ids"].append(case_id)
            continue

        k_id = str(kernel.get("id_hint") or stable_id(kernel, "K"))
        edge_key = f"{t_id}|{k_id}"
        if edge_key not in edge_map:
            edge_map[edge_key] = {
                "id": f"P{len(edge_map)}",
                "name": f"path_P{len(edge_map)}_{t_id}_{k_id}",
                "reachability": "reachable",
                "tiling_path_id": t_id,
                "kernel_branch_id": k_id,
                "tiling_key": t_key,
                "witness_route_case_ids": [],
            }
        edge_map[edge_key]["witness_route_case_ids"].append(case_id)

    paths = sorted(edge_map.values(), key=lambda e: e["id"])
    return paths, unresolved


def build_cases_output(
    cases: list[dict[str, Any]],
    paths: list[dict[str, Any]],
    unresolved: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    case_to_path: dict[str, dict[str, Any]] = {}
    for path in paths:
        for cid in path.get("witness_route_case_ids", []):
            case_to_path[cid] = path
    case_to_unresolved: dict[str, dict[str, Any]] = {}
    for item in unresolved:
        for cid in item.get("witness_route_case_ids", []):
            case_to_unresolved[cid] = item

    records: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        case_id = str(case.get("id", f"RC{index}"))
        path = case_to_path.get(case_id)
        record: dict[str, Any] = {"id": case_id}
        if path:
            record["path_id"] = path["id"]
            record["path_name"] = path.get("name")
            record["tiling_key"] = path.get("tiling_key")
            record["tiling_path_id"] = path.get("tiling_path_id")
            record["kernel_branch_id"] = path.get("kernel_branch_id")
            record["reachability"] = path.get("reachability")
        elif case_id in case_to_unresolved:
            record["path_id"] = None
            record["path_name"] = None
            record["tiling_key"] = None
            record["tiling_path_id"] = None
            record["kernel_branch_id"] = None
            record["reachability"] = "unresolved"
        else:
            record["path_id"] = None
            record["path_name"] = None
            record["tiling_key"] = None
            record["tiling_path_id"] = None
            record["kernel_branch_id"] = None
            record["reachability"] = "unmapped"
        record["case_values"] = case.get("case_values", {})
        record["purpose"] = case.get("purpose", "")
        record["evidence"] = case.get("evidence", [])
        records.append(record)
    return [r for r in records if r.get("reachability") == "reachable"]


def build_kernel_orphans(
    ctx: Context, kernel_branches: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    catalog = build_kernel_branch_catalog(ctx)
    hit_ids = set(kernel_branches.keys())
    orphans = []
    for item in catalog:
        hint = str(item.get("id_hint") or stable_id(item, "K"))
        if hint in hit_ids:
            continue
        orphans.append({
            "id": f"D{len(orphans)}",
            "reachability": "dead",
            "kernel_branch_id": hint,
            "tiling_key": int(item.get("tiling_key", -1)),
            "dead_reason": "kernel branch never reached by route cases",
        })
    return orphans


def normalize_issues(items: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    issues = []
    for index, item in enumerate(items):
        issue = dict(item)
        issue["id"] = str(issue.pop("id_hint", issue.get("id", f"{prefix}{index}")))
        issues.append(issue)
    return sorted(issues, key=lambda i: (str(i.get("side", "")), i["id"]))


def build_completeness_checklist(
    ctx: Context,
    route_case_count: int,
    tiling_paths: dict[str, dict[str, Any]],
    kernel_branches: dict[str, dict[str, Any]],
    paths: list[dict[str, Any]],
    orphans: list[dict[str, Any]],
    unresolved: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "route_case_coverage": {"status": "covered", "count": route_case_count},
        "tiling_path_coverage": {"status": "covered", "count": len(tiling_paths)},
        "kernel_dispatch_coverage": {
            "status": "covered",
            "kernel_branches": len(kernel_branches),
            "kernel_orphans": len(orphans),
        },
        "mapping_coverage": {
            "status": "unresolved" if unresolved else "covered",
            "paths": len(paths),
            "mapped": sum(p["reachability"] == "reachable" for p in paths),
            "tiling_no_kernel": sum(p["reachability"] == "tiling_no_kernel" for p in paths),
            "kernel_orphans": len(orphans),
            "unresolved": len(unresolved),
        },
        "source_scope_compliance": build_source_scope_compliance(ctx),
    }


def build_path_list(ctx: Context, cases: list[dict[str, Any]]) -> dict[str, Any]:
    results = run_evaluators(ctx, cases)
    tiling_paths = dedup_table(results, "tiling", "T")
    kernel_branches = dedup_table(results, "kernel", "K")
    paths, eval_unresolved = build_paths(results, tiling_paths, kernel_branches)
    orphans = build_kernel_orphans(ctx, kernel_branches)
    unsupported: list[dict[str, Any]] = []
    unresolved = normalize_issues(eval_unresolved, "R")

    data = {
        "schema_version": SCHEMA_VERSION,
        "operator": ctx.operator,
        "source": build_source_info(ctx),
        "tiling_paths": tiling_paths,
        "kernel_branches": kernel_branches,
        "paths": paths,
        "kernel_orphans": orphans,
        "unsupported": unsupported,
        "unresolved": unresolved,
        "completeness_checklist": build_completeness_checklist(
            ctx, len(cases), tiling_paths, kernel_branches, paths, orphans, unresolved
        ),
    }
    validate_path_list(data)
    return data


def validate_path_list(data: dict[str, Any]) -> None:
    _require(data.get("schema_version") == SCHEMA_VERSION, "schema_version mismatch")
    _require(isinstance(data.get("operator"), str) and data["operator"], "operator required")
    for field in ("source", "tiling_paths", "kernel_branches"):
        _require(isinstance(data.get(field), dict), f"{field} must be object")
    for field in ("paths", "kernel_orphans", "unsupported", "unresolved"):
        _require(isinstance(data.get(field), list), f"{field} must be list")
    _require(isinstance(data.get("completeness_checklist"), dict), "completeness_checklist must be object")

    for path in data["paths"]:
        _require(path.get("tiling_path_id") in data["tiling_paths"], f"path {path.get('id')}: tiling_path_id not found")
        k_id = path.get("kernel_branch_id")
        if k_id is not None:
            _require(k_id in data["kernel_branches"], f"path {path.get('id')}: kernel_branch_id not found")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def normalize_case_value_fields() -> frozenset[str]:
    """Load and validate the operator-specific fixed case_values contract."""
    fields = get_case_value_fields()
    _require(isinstance(fields, (set, frozenset)), "get_case_value_fields must return a set")
    _require(bool(fields), "get_case_value_fields must return a non-empty set")
    normalized = frozenset(fields)
    invalid = sorted(f for f in normalized if not isinstance(f, str) or not f)
    _require(not invalid, f"case_values fields must be non-empty strings: {invalid}")
    return normalized


def validate_case_values(case_values: Any, owner: str) -> None:
    """Require every route case to use the configured field contract."""
    _require(isinstance(case_values, dict), f"{owner}: case_values must be object")
    actual = set(case_values)
    expected = set(normalize_case_value_fields())
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    _require(not missing, f"{owner}: missing case_values fields: {missing}")
    _require(not unknown, f"{owner}: unknown case_values fields: {unknown}")


def build_source_scope_compliance(ctx: Context) -> dict[str, Any]:
    """Static helper: build compliance checklist from source info.
    
    This function does not need to be modified in Step 2.
    It calls build_source_info(ctx) from the dynamic region.
    """
    source = build_source_info(ctx)
    return {
        "status": "covered",
        "source_scope": source.get("source_scope"),
        "tiling_p0_files": source.get("tiling_p0_files", []),
        "kernel_p0_dispatch_files": source.get("kernel_p0_dispatch_files", []),
    }


def validate_route_case(case: dict[str, Any], case_id: str) -> None:
    validate_case_values(case.get("case_values"), case_id)


def main() -> int:
    args = parse_args()
    try:
        ctx = build_context(args)
        cases = load_cases(ctx.cases_path)
        data = build_path_list(ctx, cases)
        ctx.output_path.parent.mkdir(parents=True, exist_ok=True)
        with ctx.output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        cases_out = build_cases_output(cases, data["paths"], data["unresolved"])
        with ctx.cases_output_path.open("w", encoding="utf-8") as f:
            json.dump(cases_out, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(
            f"S2P1_path_list.json written: "
            f"tiling_paths={len(data['tiling_paths'])}, "
            f"kernel_branches={len(data['kernel_branches'])}, "
            f"paths={len(data['paths'])}, "
            f"orphans={len(data['kernel_orphans'])}, "
            f"unresolved={len(data['unresolved'])}"
        )
        print(f"S2P1_cases.json written: {len(cases_out)} cases -> {ctx.cases_output_path}")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return fail(str(exc))
    return 0


# =========================
# Dynamic region: implement for each operator
#
# get_case_value_fields() is the single field contract for all route cases.
# The generator imports it from this module instead of maintaining a copy.
# eval_tiling_route(case) and eval_kernel_route(case, tiling) must read tiling
# branch variables from case["case_values"]. Task A evaluator only classifies
# tiling/kernel routes; it does not construct or validate complete operator instances.
# S2P1_case_generator.py owns case generation and this evaluator only evaluates routes.
# =========================


def get_case_value_fields() -> frozenset[str]:
    """Return the complete fixed set of external route fields from Step 1."""
    raise NotImplementedError("TODO(dynamic): implement get_case_value_fields")


def build_source_info(ctx: Context) -> dict[str, Any]:
    """Return source-scope metadata."""
    raise NotImplementedError("TODO(dynamic): implement build_source_info")


def eval_tiling_route(case: dict[str, Any]) -> dict[str, Any] | None:
    """Simulate tiling P0 dispatch from case["case_values"]."""
    raise NotImplementedError("TODO(dynamic): implement eval_tiling_route")


def eval_kernel_route(case: dict[str, Any], tiling: dict[str, Any]) -> dict[str, Any] | None:
    """Simulate kernel P0 dispatch from case["case_values"] and tiling."""
    raise NotImplementedError("TODO(dynamic): implement eval_kernel_route")


def build_kernel_branch_catalog(ctx: Context) -> list[dict[str, Any]]:
    """Return full kernel dispatch catalog for orphan detection."""
    raise NotImplementedError("TODO(dynamic): implement build_kernel_branch_catalog")


if __name__ == "__main__":
    sys.exit(main())
