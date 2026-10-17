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
"""Task A Route Case Generator template.

Copy this file with cp as S2P1_case_generator.py and implement only the dynamic region.
Do not rewrite the static framework.
Run: python3 S2P1_case_generator.py --operator <op> [--output S2P1_route_cases.json] [--seed 42]
Requires S2P1_route_extract.py in the same directory for validation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = "S2P1_route_cases.task-a"
EVALUATOR_FILENAME = "S2P1_route_extract.py"


# =========================
# Static framework
# =========================

OPERATOR_NAME = "{{OPERATOR}}"

_USED_CASE_VALUES: dict[str, set[Any]] = {}


def pick_distinct_value(field: str, candidates: list[Any]) -> Any:
    """Pick the first unused value for a field, falling back to the first candidate."""
    if not candidates:
        raise ValueError(f"{field}: candidates must not be empty")
    used = _USED_CASE_VALUES.setdefault(field, set())
    for value in candidates:
        if value not in used:
            used.add(value)
            return value
    value = candidates[0]
    used.add(value)
    return value


def integer_candidates(
    lo: int,
    hi: int,
    *,
    count: int = 4,
    include_bounds: bool = True,
    offset: int = 0,
) -> list[int]:
    """Build deterministic integer candidates inside [lo, hi]."""
    if lo > hi:
        raise ValueError(f"invalid integer range: [{lo}, {hi}]")

    values: list[int] = []
    if include_bounds:
        values.extend([lo, hi])
    for i in range(count):
        values.append(lo + offset + i)

    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        clamped = min(max(value, lo), hi)
        if clamped not in seen:
            seen.add(clamped)
            result.append(clamped)
    return result


def rotate_candidates(values: list[Any], *, offset: int = 0) -> list[Any]:
    """Rotate finite candidate values deterministically."""
    if not values:
        raise ValueError("values must not be empty")
    shift = offset % len(values)
    return values[shift:] + values[:shift]


def build_tiling_case_values(
    case_spec: dict[str, Any], case_value_fields: frozenset[str]
) -> dict[str, Any]:
    """Return the case_values payload for a candidate route case."""
    case_values = dict(case_spec.get("case_values", {}))
    validate_case_values(case_values, case_value_fields, "candidate")
    return case_values


def build_case_object(
    case_id: str,
    case_spec: dict[str, Any],
    case_values: dict[str, Any],
    tiling: dict[str, Any],
    kernel: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the final route case object written to S2P1_route_cases.json."""
    seed_constraints = dict(case_spec.get("seed_constraints", {}))
    branch_group = case_spec.get("branch_group")
    sub_branch = case_spec.get("sub_branch")
    if branch_group is not None:
        seed_constraints["branch_group"] = branch_group
    if sub_branch is not None:
        seed_constraints["sub_branch"] = sub_branch
    expected = case_spec.get("expected", {})
    if expected:
        seed_constraints["expected"] = expected
    seed_constraints["tiling_key"] = tiling.get("tiling_key")
    seed_constraints["kernel_reachable"] = kernel is not None
    return {
        "id": case_id,
        "case_values": case_values,
        "seed_constraints": seed_constraints,
        "purpose": case_spec.get("purpose", ""),
        "evidence": list(case_spec.get("evidence", [])),
    }


def fail(message: str) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return 1


def normalize_case_value_fields(fields: Any) -> frozenset[str]:
    if not isinstance(fields, (set, frozenset)) or not fields:
        raise ValueError("get_case_value_fields must return a non-empty set")
    normalized = frozenset(fields)
    invalid = sorted(f for f in normalized if not isinstance(f, str) or not f)
    if invalid:
        raise ValueError(f"case_values fields must be non-empty strings: {invalid}")
    return normalized


def load_evaluator() -> tuple[Callable, Callable, frozenset[str]]:
    path = ROOT / EVALUATOR_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"evaluator not found: {path}")
    spec = importlib.util.spec_from_file_location("route_extract", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    get_case_value_fields = getattr(mod, "get_case_value_fields", None)
    if not callable(get_case_value_fields):
        raise ValueError("evaluator get_case_value_fields must be implemented")
    case_value_fields = normalize_case_value_fields(get_case_value_fields())
    return mod.eval_tiling_route, mod.eval_kernel_route, case_value_fields


def validate_case_values(
    case_values: Any, case_value_fields: frozenset[str], owner: str
) -> None:
    """Require an exact match with the evaluator's field contract."""
    if not isinstance(case_values, dict):
        raise TypeError(f"{owner}: case_values must be object")
    actual = set(case_values)
    expected = set(case_value_fields)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise ValueError(f"{owner}: missing case_values fields: {missing}")
    if unknown:
        raise ValueError(f"{owner}: unknown case_values fields: {unknown}")


def validate_route_case_spec(
    case_spec: dict[str, Any], index: int, case_value_fields: frozenset[str]
) -> None:
    if not isinstance(case_spec, dict):
        raise TypeError(f"candidate {index}: case spec must be object")
    case_values = case_spec.get("case_values")
    if not isinstance(case_values, dict):
        raise TypeError(f"candidate {index}: case_values must be object")
    validate_case_values(case_values, case_value_fields, f"candidate {index}")


def route_matches(case_spec: dict[str, Any], tiling: dict[str, Any], kernel: dict[str, Any] | None) -> bool:
    expected = case_spec.get("expected", {})
    if not isinstance(expected, dict):
        return False

    if "tiling_key" in expected and tiling.get("tiling_key") != expected["tiling_key"]:
        return False

    expected_strategy = expected.get("strategy_vars", {})
    if not isinstance(expected_strategy, dict):
        return False
    actual_strategy = tiling.get("strategy_vars", {})
    if not isinstance(actual_strategy, dict):
        return False
    for name, value in expected_strategy.items():
        if actual_strategy.get(name) != value:
            return False

    if "reachability" in expected:
        actual = "reachable" if kernel is not None else "tiling_only"
        if actual != expected["reachability"]:
            return False

    if kernel is not None:
        if "kernel_branch_id" in expected:
            kernel_id = kernel.get("id_hint") or kernel.get("kernel_branch_id") or kernel.get("id")
            if kernel_id != expected["kernel_branch_id"]:
                return False
    elif expected.get("reachability") == "reachable":
        return False

    return True


def generate_cases() -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    """Generate route cases from operator-specific candidate specs."""
    eval_tiling, eval_kernel, case_value_fields = load_evaluator()
    candidates = generate_candidate_cases()
    cases: list[dict[str, Any]] = []
    target_counts: dict[str, int] = {}
    failed: list[str] = []
    kernel_reachable = 0
    tiling_only = 0

    for index, case_spec in enumerate(candidates):
        validate_route_case_spec(case_spec, index, case_value_fields)

        matched = False
        last_error = ""
        # Candidate values are selected by generate_candidate_cases(). A retry
        # of the same case cannot provide a new sample, so evaluate once and
        # report a mismatch instead of repeating identical attempts.
        try:
            case_values = build_tiling_case_values(case_spec, case_value_fields)
            case_obj: dict[str, Any] = {"case_values": case_values}

            tiling = eval_tiling(case_obj)
            if tiling is None:
                last_error = "eval_tiling_route returned None"
            else:
                kernel = eval_kernel(case_obj, tiling)
                if not route_matches(case_spec, tiling, kernel):
                    last_error = "route mismatch"
                else:
                    matched = True
                    case_id = str(case_spec.get("id", f"RC{len(cases)}"))
                    cases.append(build_case_object(case_id, case_spec, case_values, tiling, kernel))

                    branch_group = case_spec.get("branch_group")
                    sub_branch = case_spec.get("sub_branch")
                    if branch_group and sub_branch:
                        target_name = f"{branch_group}/{sub_branch}"
                    else:
                        target_name = str(branch_group or case_spec.get("purpose", case_id))
                    target_counts[target_name] = target_counts.get(target_name, 0) + 1
                    if kernel is not None:
                        kernel_reachable += 1
                    else:
                        tiling_only += 1
        except Exception as exc:
            last_error = str(exc)

        if not matched:
            suffix = f": {last_error}" if last_error else ""
            failed.append(f"candidate[{index}] {case_spec}{suffix}")

    stats = {
        "total": len(cases),
        "target_counts": target_counts,
        "kernel_reachable": kernel_reachable,
        "tiling_only": tiling_only,
        "failed": failed,
    }
    return cases, stats, failed


def print_report(stats: dict[str, Any], failed: list[str]) -> None:
    print(f"Generated {stats['total']} route cases")
    for target, count in sorted(stats["target_counts"].items()):
        print(f"  {target}: {count}")
    print(f"  kernel_reachable={stats['kernel_reachable']}, tiling_only={stats['tiling_only']}")
    if failed:
        print(f"  FAILED ({len(failed)}):", file=sys.stderr)
        for f in failed:
            print(f"    {f}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate route cases from branch recipes.")
    parser.add_argument("--operator", required=True)
    parser.add_argument("--output", default="S2P1_route_cases.json")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    try:
        cases, stats, failed = generate_cases()
        data = {
            "schema_version": SCHEMA_VERSION,
            "operator": args.operator,
            "generated_by": "case_generator",
            "cases": cases,
        }
        if failed:
            data["failed"] = failed
        output_path = ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print_report(stats, failed)
        print(f"Written -> {output_path}")
        return 1 if failed else 0
    except Exception as exc:
        return fail(str(exc))


# =========================
# Dynamic region: implement for each operator
#
# case_values must contain the complete fixed field set returned by
# S2P1_route_extract.py:get_case_value_fields(). Do not put internally derived
# route facts in this object. Task A only generates route witnesses needed to
# trigger tiling/kernel dispatch. The complete operator instance is constructed
# by Step 5 Mapper.
# =========================


def generate_candidate_cases() -> list[dict[str, Any]]:
    """Return operator-specific route candidate cases.

    Each candidate should contain at least:
    - branch_group: tiling-side major branch name
    - sub_branch: optional sub-branch name inside branch_group
    - case_values: the complete fixed set of external route fields
    - expected: optional route expectations for validation
    - seed_constraints: optional witness metadata
    - purpose: short coverage description
    - evidence: source references
    """
    raise NotImplementedError("TODO(dynamic): implement generate_candidate_cases")


if __name__ == "__main__":
    sys.exit(main())
