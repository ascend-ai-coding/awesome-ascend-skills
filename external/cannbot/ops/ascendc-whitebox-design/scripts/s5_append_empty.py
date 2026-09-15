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
"""Append Mapper-v1 empty low cases and write final S5_cases_low.json."""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

INPUT_FILE = "S5_mapped_cases_low_shape.json"
OUTPUT_FILE = "S5_cases_low.json"


def load_json(root: Path, path: str) -> Any:
    with (root / path).open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(root: Path, path: str, data: Any) -> None:
    with (root / path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def clone_case(case: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(case)


def mapped_inputs(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    inputs = case.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError(f"{case.get('id', '<unknown>')}: inputs must be an object")
    return inputs


def assert_unique_ids(cases: list[dict[str, Any]], label: str) -> None:
    seen: set[str] = set()
    for case in cases:
        case_id = case.get("id")
        if not isinstance(case_id, str):
            raise ValueError(f"{label}: case id must be a string")
        if case_id in seen:
            raise ValueError(f"duplicate id in {label}: {case_id}")
        seen.add(case_id)


def empty_tensor_list_descriptor(descriptor: dict[str, Any]) -> None:
    descriptor["tensor_count"] = 0
    descriptor["tensors"] = []


def empty_tensor_descriptor(descriptor: dict[str, Any]) -> None:
    shape = descriptor.get("shape")
    if isinstance(shape, list):
        if shape:
            descriptor["shape"] = [0, *shape[1:]]
        else:
            descriptor["shape"] = [0]
    else:
        descriptor["shape"] = [0]


def output_matches_input(output: dict[str, Any], input_descriptor: dict[str, Any]) -> bool:
    if output.get("kind") != input_descriptor.get("kind"):
        return False
    if output.get("kind") == "tensor_list":
        return output.get("tensor_count") == input_descriptor.get("tensor_count")
    return output.get("shape") == input_descriptor.get("shape")


def empty_matching_outputs(case: dict[str, Any], seed: dict[str, Any], input_names: tuple[str, ...]) -> None:
    for input_name in input_names:
        seed_input = mapped_inputs(seed)[input_name]
        for output in case.get("outputs", {}).values():
            if not isinstance(output, dict):
                continue
            if not output_matches_input(output, seed_input):
                continue
            if output.get("kind") == "tensor_list":
                empty_tensor_list_descriptor(output)
            elif output.get("kind") == "tensor":
                empty_tensor_descriptor(output)


def tensor_list_seed_groups(shape_cases: list[dict[str, Any]]) -> list[tuple[tuple[str, ...], dict[str, Any]]]:
    result: list[tuple[tuple[str, ...], dict[str, Any]]] = []
    seen: set[tuple[str, ...]] = set()
    for case in shape_cases:
        names = tuple(name for name, desc in mapped_inputs(case).items() if desc.get("kind") == "tensor_list")
        if names and names not in seen:
            seen.add(names)
            result.append((names, case))
    return result


def tensor_seeds(shape_cases: list[dict[str, Any]]) -> list[tuple[tuple[str, ...], dict[str, Any]]]:
    result: list[tuple[tuple[str, ...], dict[str, Any]]] = []
    seen: set[str] = set()
    for case in shape_cases:
        for name, desc in mapped_inputs(case).items():
            if name in seen or desc.get("kind") != "tensor":
                continue
            seen.add(name)
            result.append(((name,), case))
    return result


def make_empty_case(seed: dict[str, Any], input_names: tuple[str, ...], index: int) -> dict[str, Any]:
    case = clone_case(seed)
    case["id"] = f"low_case_empty_{index:02d}"
    case["source"] = "empty"
    meta = case.setdefault("meta", {})
    if not isinstance(meta, dict):
        raise ValueError(f"{seed.get('id', '<unknown>')}: meta must be an object")
    meta.update({"base_id": seed["id"], "variant_kind": "empty", "empty_inputs": list(input_names)})

    for input_name in input_names:
        descriptor = mapped_inputs(case)[input_name]
        if descriptor.get("kind") == "tensor_list":
            empty_tensor_list_descriptor(descriptor)
        elif descriptor.get("kind") == "tensor":
            empty_tensor_descriptor(descriptor)
        else:
            raise ValueError(f"{case['id']}: unsupported input kind for {input_name}")
    empty_matching_outputs(case, seed, input_names)
    return case


def build_final_low_cases(shape_cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    if not shape_cases:
        raise ValueError(f"{INPUT_FILE}: must contain at least one shape case")
    non_shape = [case.get("id", "<unknown>") for case in shape_cases if case.get("source") != "shape"]
    if non_shape:
        raise ValueError(f"{INPUT_FILE}: expected source='shape', got {non_shape}")
    seeds = tensor_list_seed_groups(shape_cases) + tensor_seeds(shape_cases)
    empty_cases = [make_empty_case(seed, input_names, index) for index, (input_names, seed) in enumerate(seeds)]
    final_cases = shape_cases + empty_cases
    assert_unique_ids(final_cases, OUTPUT_FILE)
    return final_cases, len(empty_cases)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Append Mapper-v1 empty low cases.")
    parser.add_argument("--whitebox-dir", required=True, help=f"Directory containing {INPUT_FILE}.")
    parser.add_argument("--force", action="store_true", help=f"Overwrite existing {OUTPUT_FILE}.")
    args = parser.parse_args()

    whitebox_dir = Path(args.whitebox_dir).resolve()
    output_path = whitebox_dir / OUTPUT_FILE
    if output_path.exists() and not args.force:
        raise SystemExit(f"{OUTPUT_FILE} already exists; use --force to regenerate it")
    raw_cases = load_json(whitebox_dir, INPUT_FILE)
    if not isinstance(raw_cases, list):
        raise SystemExit(f"{INPUT_FILE}: root must be a JSON list")
    final_cases, empty_count = build_final_low_cases(raw_cases)
    dump_json(whitebox_dir, OUTPUT_FILE, final_cases)
    _logger.info("PASS: appended empty cases")
    _logger.info(
        f"input={INPUT_FILE} output={OUTPUT_FILE} shape_cases={len(raw_cases)} "
        f"empty_cases_added={empty_count} low_cases={len(final_cases)}"
    )


if __name__ == "__main__":
    main()
