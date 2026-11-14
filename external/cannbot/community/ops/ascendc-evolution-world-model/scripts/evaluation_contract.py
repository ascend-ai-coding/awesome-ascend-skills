#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# ----------------------------------------------------------------------------------------------------------
"""Shared evaluation evidence for refine, artifact checks and lineage.

See references/evaluation_contract.md for format precedence and quality policy.
This module parses evidence only; case coverage, canonical timing and multi-shape
parent eligibility remain responsibilities of their existing consumers.
"""

import json
import math
from typing import Optional

QUALITY_RANK = {"good": 0, "acceptable": 1, "noisy": 2, "unknown": 3}
_RESULT_FIELDS = ("compilation_success", "precision_passed", "speedup")


def _number(value) -> Optional[float]:
    """Accept finite JSON numbers; booleans are not measurements."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def measurement_quality(data: dict) -> str:
    """Explicit quality wins; otherwise derive from non-negative CV percent."""
    explicit = data.get("measurement_quality")
    if explicit is not None:
        return (
            explicit
            if isinstance(explicit, str) and explicit in QUALITY_RANK
            else "unknown"
        )
    cv = _number(data.get("cv_pct"))
    if cv is None or cv < 0:
        return "unknown"
    if cv < 5:
        return "good"
    return "acceptable" if cv < 15 else "noisy"


def _pipeline(data: dict):
    evolved = data.get("evolved")
    # Explicit nested null records a failed collection; do not revive stale data.
    if isinstance(evolved, dict) and "pipeline" in evolved:
        value = evolved["pipeline"]
    else:
        value = data.get("pipeline")
    return value if isinstance(value, dict) else None


def _quality_fields(data: dict, preferred: dict) -> dict:
    return {
        key: preferred[key] if key in preferred else data.get(key)
        for key in ("measurement_quality", "cv_pct")
    }


def _single_result(data: dict, comparison: dict) -> Optional[dict]:
    # evolved.pipeline alone is NOT a comparison-format evaluation.
    nested = any(key in comparison for key in _RESULT_FIELDS)
    fields = comparison if nested else data
    if not any(key in fields for key in _RESULT_FIELDS):
        return None
    compile_ok = fields.get("compilation_success")
    precision_ok = fields.get("precision_passed")
    if nested and precision_ok is None and isinstance(data.get("evolved"), dict):
        precision_ok = data["evolved"].get("precision_passed")
    speedup = fields.get("speedup")
    if nested and speedup is None:
        speedup = data.get("speedup")
    return {
        "compile": compile_ok,
        "precision": precision_ok,
        "speedup": speedup,
        "measurement_quality": measurement_quality(_quality_fields(data, fields)),
        "_source": "flat_with_comparison" if nested else "flat",
    }


def _multi_quality(data: dict, comparison: dict, targets: list) -> str:
    global_fields = _quality_fields(data, comparison)
    global_present = any(value is not None for value in global_fields.values())
    qualities = [measurement_quality(global_fields)] if global_present else []
    for target in targets:
        has_quality = any(
            target.get(key) is not None for key in ("measurement_quality", "cv_pct")
        )
        if has_quality or not global_present:
            qualities.append(measurement_quality(target))
    return max(qualities, key=QUALITY_RANK.get)


def _multi_result(data: dict, comparison: dict) -> Optional[dict]:
    shapes = data["shape_results"]
    if not isinstance(shapes, dict):
        return None
    targets = shapes.get("target")
    generalization = shapes.get("generalization", [])
    aggregate = data.get("aggregate") or {}
    if (
        not isinstance(targets, list)
        or not targets
        or not isinstance(generalization, list)
        or not isinstance(aggregate, dict)
    ):
        return None
    for shape in targets + generalization:
        if not isinstance(shape, dict) or any(
            type(shape.get(key)) is not bool for key in _RESULT_FIELDS[:2]
        ):
            return None
    compile_ok = all(shape["compilation_success"] for shape in targets + generalization)
    precision_ok = all(shape["precision_passed"] for shape in targets + generalization)
    # Explicit aggregate failure cannot be overridden by passing shape details.
    for key in _RESULT_FIELDS[:2]:
        if key in comparison and type(comparison[key]) is not bool:
            return None
    compile_ok = compile_ok and comparison.get("compilation_success", True)
    precision_ok = precision_ok and comparison.get("precision_passed", True)
    speeds = [_number(shape.get("speedup")) for shape in targets]
    speedup = (
        min(speeds)
        if all(value is not None and value > 0 for value in speeds)
        else None
    )
    if "target_min_speedup" in aggregate:
        aggregate_min = _number(aggregate["target_min_speedup"])
        if aggregate_min is None or aggregate_min <= 0:
            if compile_ok and precision_ok:
                return None
        elif speedup is not None:
            # Neither a stale aggregate nor one fast shape may hide a slow target.
            speedup = min(speedup, aggregate_min)
    if compile_ok and precision_ok and speedup is None:
        return None
    result = {
        "compile": compile_ok,
        "precision": precision_ok,
        "speedup": speedup,
        "measurement_quality": _multi_quality(data, comparison, targets),
        "_source": "multi_shape",
    }
    if generalization:
        result["_generalization_compile"] = all(
            s["compilation_success"] for s in generalization
        )
        result["_generalization_precision"] = all(
            s["precision_passed"] for s in generalization
        )
        warnings = []
        if not result["_generalization_compile"]:
            warnings.append("generalization_compile_failed")
        if not result["_generalization_precision"]:
            warnings.append("generalization_precision_failed")
        if aggregate.get("any_generalization_regression"):
            warnings.append("generalization_regression")
        if warnings:
            result["_warnings"] = warnings
    return result


def normalize_eval_result(data) -> Optional[dict]:
    """Parse flat, comparison or multi-shape results without mutating the input.

    Return None for malformed/unrecognized evidence. Failures may omit timing;
    passing results need a finite positive speedup and explicit boolean flags.
    Missing or invalid measurement quality remains unknown, never good.
    """
    if not isinstance(data, dict):
        return None
    for key in ("evolved", "aggregate"):
        if data.get(key) is not None and not isinstance(data[key], dict):
            return None
    comparison = data.get("comparison")
    if comparison is None:
        comparison = {}
    if not isinstance(comparison, dict):
        return None
    if "shape_results" in data:
        result = _multi_result(data, comparison)
    else:
        result = _single_result(data, comparison)
    if result is None or any(
        type(result[key]) is not bool for key in ("compile", "precision")
    ):
        return None
    speedup = _number(result["speedup"])
    if result["compile"] and result["precision"] and (speedup is None or speedup <= 0):
        return None
    result["speedup"] = speedup if speedup is not None and speedup > 0 else None
    result["pipeline"] = _pipeline(data)
    return result


def load_eval_result(path: str) -> Optional[dict]:
    """Load normalized evidence; missing/partial JSON is not successful evidence."""
    try:
        with open(path, encoding="utf-8") as stream:
            return normalize_eval_result(json.load(stream))
    except (OSError, ValueError):
        return None
