#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Expand Mapper-v1 low cases into high data_range cases."""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

NON_NORMAL_DATA_RANGES = (
    "zero",
    "extreme",
    "negative",
    "tiny_pos",
    "all_ones",
    "near_zero",
    "with_inf",
    "with_nan",
)


def load_json(base_dir: Path, rel_path: str) -> Any:
    with (base_dir / rel_path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def dump_json(base_dir: Path, rel_path: str, payload: Any) -> None:
    with (base_dir / rel_path).open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def clone_case(source_case: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(source_case)


def mapped_inputs(source_case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    case_inputs = source_case.get("inputs")
    if not isinstance(case_inputs, dict):
        raise ValueError(f"{source_case.get('id', '<unknown>')}: inputs must be an object")
    return case_inputs


def validate_supported(ranges: Any, label: str) -> list[str]:
    if not isinstance(ranges, list):
        raise ValueError(f"{label}: supported must be a list")
    result: list[str] = []
    seen: set[str] = set()
    for item in ranges:
        if not isinstance(item, str):
            raise ValueError(f"{label}: data_range must be a string")
        if item not in NON_NORMAL_DATA_RANGES:
            raise ValueError(f"{label}: unsupported data_range {item!r}")
        if item in seen:
            raise ValueError(f"{label}: duplicate data_range {item!r}")
        seen.add(item)
        result.append(item)
    return result


def load_policy(root: Path) -> dict[str, Any]:
    policy = load_json(root, "S5_data_range_policy.json")
    if not isinstance(policy, dict):
        raise ValueError("S5_data_range_policy.json: root must be an object")
    if policy.get("version") != 1 or policy.get("mode") != "per_input_cyclic":
        raise ValueError("S5_data_range_policy.json: expected version=1 and mode=per_input_cyclic")
    inputs = policy.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("S5_data_range_policy.json: inputs must be a non-empty object")
    normalized: dict[str, Any] = {}
    for name, spec in inputs.items():
        if not isinstance(name, str) or not isinstance(spec, dict):
            raise ValueError("S5_data_range_policy.json: invalid input spec")
        if set(spec) != {"participates", "supported"}:
            raise ValueError(f"S5_data_range_policy.json: input {name} must only contain participates and supported")
        if not isinstance(spec.get("participates"), bool):
            raise ValueError(f"S5_data_range_policy.json: input {name}.participates must be bool")
        supported = validate_supported(spec.get("supported"), f"S5_data_range_policy.json: input {name}")
        if not spec["participates"] and supported:
            raise ValueError(f"S5_data_range_policy.json: input {name}.supported must be []")
        normalized[name] = {"participates": spec["participates"], "supported": supported}
    return {"version": 1, "mode": "per_input_cyclic", "inputs": normalized}


def reset_input_range(descriptor: dict[str, Any]) -> None:
    descriptor["data_range"] = "normal"
    if descriptor.get("kind") == "tensor_list":
        for child in descriptor.get("tensors", []):
            if isinstance(child, dict):
                child["data_range"] = "normal"


def normalize_input_ranges(case: dict[str, Any]) -> dict[str, Any]:
    normalized = clone_case(case)
    for descriptor in mapped_inputs(normalized).values():
        reset_input_range(descriptor)
    return normalized


def set_input_range(case: dict[str, Any], input_name: str, range_name: str) -> None:
    descriptor = mapped_inputs(case).get(input_name)
    if not isinstance(descriptor, dict):
        raise ValueError(f"{case.get('id', '<unknown>')}: missing input {input_name}")
    descriptor["data_range"] = range_name
    if descriptor.get("kind") == "tensor_list":
        for child in descriptor.get("tensors", []):
            if isinstance(child, dict):
                child["data_range"] = range_name


def participants(case: dict[str, Any], policy: dict[str, Any]) -> list[tuple[str, list[str]]]:
    case_inputs = mapped_inputs(case)
    policy_inputs = policy["inputs"]
    missing = set(case_inputs) - set(policy_inputs)
    extra = set(policy_inputs) - set(case_inputs)
    if missing:
        raise ValueError(f"{case.get('id', '<unknown>')}: policy missing inputs {sorted(missing)}")
    if extra:
        raise ValueError(f"{case.get('id', '<unknown>')}: policy has unknown inputs {sorted(extra)}")
    return [
        (name, spec["supported"])
        for name, spec in policy_inputs.items()
        if spec["participates"] and spec["supported"]
    ]


def make_range_variants(case: dict[str, Any], policy: dict[str, Any]) -> list[dict[str, Any]]:
    items = participants(case, policy)
    if not items:
        return []
    count = max(len(ranges) for _, ranges in items)
    variants: list[dict[str, Any]] = []
    for index in range(count):
        range_by_input = {name: ranges[index % len(ranges)] for name, ranges in items}
        variant = normalize_input_ranges(case)
        variant["id"] = f"{case['id']}_range{index:02d}"
        variant["source"] = "range"
        meta = variant.setdefault("meta", {})
        if not isinstance(meta, dict):
            raise ValueError(f"{case.get('id', '<unknown>')}: meta must be an object")
        meta.update({
            "base_id": case["id"],
            "variant_kind": "range",
            "range_mode": "per_input_cyclic",
            "range_index": index,
            "range_by_input": range_by_input,
        })
        for input_name, range_name in range_by_input.items():
            set_input_range(variant, input_name, range_name)
        variants.append(variant)
    return variants


def assert_unique_ids(case_list: list[dict[str, Any]], label: str) -> None:
    seen_ids: set[str] = set()
    for entry in case_list:
        current_id = entry.get("id")
        if not isinstance(current_id, str):
            raise ValueError(f"{label}: case id must be a string")
        if current_id in seen_ids:
            raise ValueError(f"duplicate id in {label}: {current_id}")
        seen_ids.add(current_id)


def build_high_cases(root: Path, low_cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    policy = load_policy(root)
    high_cases = [normalize_input_ranges(case) for case in low_cases]
    range_count = 0
    for case in low_cases:
        variants = make_range_variants(case, policy)
        high_cases.extend(variants)
        range_count += len(variants)
    assert_unique_ids(high_cases, "S5_cases_high.json")
    return high_cases, range_count


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Expand Mapper-v1 low cases into high data_range cases.")
    parser.add_argument("--whitebox-dir", required=True, help="Directory containing S5_cases_low.json.")
    args = parser.parse_args()

    whitebox_dir = Path(args.whitebox_dir).resolve()
    low_cases = load_json(whitebox_dir, "S5_cases_low.json")
    if not isinstance(low_cases, list):
        raise SystemExit("S5_cases_low.json: root must be a JSON list")
    high_cases, range_count = build_high_cases(whitebox_dir, low_cases)
    dump_json(whitebox_dir, "S5_cases_high.json", high_cases)
    _logger.info("PASS: expanded high data_range cases")
    _logger.info(f"low_cases={len(low_cases)} range_cases={range_count} high_cases={len(high_cases)}")


if __name__ == "__main__":
    main()
