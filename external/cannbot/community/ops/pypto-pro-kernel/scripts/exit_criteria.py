# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Validate and evaluate user-defined optimization criteria; no implicit targets."""
from __future__ import annotations

import math
import re

OPERATORS = {"<", "<=", ">", ">=", "==", "between"}
BASES = {"hardware", "model"}
AGGREGATIONS = {"each", "min", "max", "mean"}


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")


def validate_exit_criteria(tree, case_names=None):
    """Return leaf conditions, rejecting ambiguity and empty/vacuously true groups."""
    leaves = []
    seen = set()

    def visit(node, depth=0):
        if depth > 12 or len(leaves) > 128:
            raise ValueError("exit criteria exceed the supported expression size")
        if not isinstance(node, dict):
            raise ValueError("each exit criterion must be an object")
        groups = set(node) & {"all", "any"}
        if groups:
            if len(groups) != 1 or len(node) != 1:
                raise ValueError("a criteria group has exactly one all/any field")
            children = node[next(iter(groups))]
            if not isinstance(children, list) or not children:
                raise ValueError("all/any criteria must contain at least one condition")
            for child in children:
                visit(child, depth + 1)
            return
        required = {"id", "metric", "operator", "threshold", "unit", "basis", "cases",
                    "aggregation", "source"}
        if set(node) != required:
            raise ValueError(f"criterion fields must be {sorted(required)}")
        for key in ("id", "metric", "unit", "source"):
            if not isinstance(node[key], str) or not node[key].strip():
                raise ValueError(f"criterion {key} must be a non-empty string")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", node["id"]) or node["id"] in seen:
            raise ValueError("criterion ids must be unique lower_snake_case names")
        seen.add(node["id"])
        if node["operator"] not in OPERATORS or node["basis"] not in BASES:
            raise ValueError("unsupported criterion operator or evidence basis")
        if node["aggregation"] not in AGGREGATIONS:
            raise ValueError("unsupported criterion case aggregation")
        threshold = node["threshold"]
        if node["operator"] == "between":
            if not isinstance(threshold, list) or len(threshold) != 2:
                raise ValueError("between requires [lower, upper]")
            for value in threshold:
                _number(value, "threshold")
            if threshold[0] > threshold[1]:
                raise ValueError("criterion interval is reversed")
        else:
            _number(threshold, "threshold")
        cases = node["cases"]
        if (not isinstance(cases, list) or not cases or
                any(not isinstance(x, str) or not x for x in cases) or len(set(cases)) != len(cases)):
            raise ValueError("criterion cases must be a non-empty array of unique case names")
        if case_names is not None and not set(cases) <= set(case_names):
            raise ValueError("criterion refers to a case outside the SPEC contract")
        leaves.append(node)

    if tree is not None:
        visit(tree)
    return leaves


def _compare(value, operator, threshold):
    return {"<": lambda: value < threshold, "<=": lambda: value <= threshold,
            ">": lambda: value > threshold, ">=": lambda: value >= threshold,
            "==": lambda: value == threshold,
            "between": lambda: threshold[0] <= value <= threshold[1]}[operator]()


def evaluate_exit_criteria(tree, observations):
    """Three-valued evaluation of already authenticated observations from ONE candidate.

    The caller verifies artifact/source hashes. Here incompatible units/bases, duplicate
    records and missing cases cannot accidentally produce PASS. Observations are records
    with metric, case, value, unit, basis and source fields.
    """
    leaves = validate_exit_criteria(tree)
    if not leaves:
        return {"status": "NOT_REQUESTED", "met": False, "conditions": []}
    details = {}
    for condition in leaves:
        values, missing = [], []
        for case in condition["cases"]:
            matches = [row for row in observations if row.get("metric") == condition["metric"]
                       and row.get("case") == case and row.get("unit") == condition["unit"]
                       and row.get("basis") == condition["basis"]
                       and row.get("source") == condition["source"]]
            if len(matches) != 1:
                missing.append(case)
                continue
            value = matches[0].get("value")
            try:
                _number(value, "observed metric")
            except ValueError:
                missing.append(case)
                continue
            values.append(value)
        if missing:
            details[condition["id"]] = {"id": condition["id"], "status": "UNKNOWN",
                                       "missing_or_ambiguous_cases": missing}
            continue
        aggregation = condition["aggregation"]
        tested = values if aggregation == "each" else [
            {"min": min, "max": max, "mean": lambda xs: sum(xs) / len(xs)}[aggregation](values)]
        passed = all(_compare(value, condition["operator"], condition["threshold"]) for value in tested)
        details[condition["id"]] = {"id": condition["id"], "status": "PASS" if passed else "FAIL",
                                   "values": values, "tested_values": tested}

    def combine(node):
        if "id" in node:
            return details[node["id"]]["status"]
        mode = next(iter(node))
        statuses = [combine(child) for child in node[mode]]
        decisive = "FAIL" if mode == "all" else "PASS"
        if decisive in statuses:
            return decisive
        if "UNKNOWN" in statuses:
            return "UNKNOWN"
        return "PASS" if mode == "all" else "FAIL"

    status = combine(tree)
    return {"status": status, "met": status == "PASS", "conditions": list(details.values())}
