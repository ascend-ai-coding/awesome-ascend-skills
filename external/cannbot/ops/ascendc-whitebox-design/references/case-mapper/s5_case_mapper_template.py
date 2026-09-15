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
"""Generate Step 5.2 shape-only low mapped cases.

Copy this file to an operator whitebox directory as S5_case_mapper.py.
Only implement the dynamic region at the end of this file.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)


ROOT = Path(__file__).resolve().parent


# =========================
# Static region: do not rewrite
# =========================


def load_json(path: str) -> Any:
    with (ROOT / path).open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str, data: Any) -> None:
    with (ROOT / path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def input_tensor(
    dtype: str,
    shape: list[int] | None,
    param_type: str = "REQUIRED",
    data_range: str = "normal",
    fmt: str = "ND",
    **extra: Any,
) -> dict[str, Any]:
    tensor = {
        "kind": "tensor",
        "dtype": dtype,
        "format": fmt,
        "shape": None if shape is None else list(shape),
        "param_type": param_type,
        "data_range": data_range,
    }
    tensor.update(extra)
    return tensor


def dynamic_input_tensor(
    dtype: str,
    tensors: list[dict[str, Any]],
    data_range: str = "normal",
    fmt: str = "ND",
    **extra: Any,
) -> dict[str, Any]:
    tensor = {
        "kind": "tensor_list",
        "dtype": dtype,
        "format": fmt,
        "param_type": "DYNAMIC",
        "tensor_count": len(tensors),
        "data_range": data_range,
        "tensors": tensors,
    }
    tensor.update(extra)
    return tensor


def output_tensor(
    dtype: str,
    shape: list[int] | None,
    param_type: str = "REQUIRED",
    fmt: str = "ND",
    **extra: Any,
) -> dict[str, Any]:
    tensor = {
        "kind": "tensor",
        "dtype": dtype,
        "format": fmt,
        "shape": None if shape is None else list(shape),
        "param_type": param_type,
    }
    tensor.update(extra)
    return tensor


def output_tensor_list(
    dtype: str,
    tensors: list[dict[str, Any]],
    fmt: str = "ND",
    **extra: Any,
) -> dict[str, Any]:
    tensor = {
        "kind": "tensor_list",
        "dtype": dtype,
        "format": fmt,
        "param_type": "DYNAMIC",
        "tensor_count": len(tensors),
        "tensors": tensors,
    }
    tensor.update(extra)
    return tensor


def clone_case(case: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(case)


def path_case_id(index: int, case: dict[str, Any]) -> str:
    return str(case.get("id") or f"case{index:05d}")


def network_case_id(index: int, config: dict[str, Any]) -> str:
    return str(config.get("id") or f"network{index:05d}")


def network_record(config: dict[str, Any], index: int) -> dict[str, Any]:
    record = dict(config)
    record.setdefault("id", network_case_id(index, config))
    return record


def mapped_inputs(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    inputs = case.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError(f"{case.get('id', '<unknown>')}: inputs must be an object")
    return inputs


def _reset_tensor_range(tensor: dict[str, Any]) -> None:
    tensor["data_range"] = "normal"
    if tensor.get("kind") == "tensor_list":
        for child in tensor.get("tensors", []):
            if isinstance(child, dict) and "data_range" in child:
                child["data_range"] = "normal"


def normalize_input_ranges(case: dict[str, Any]) -> dict[str, Any]:
    normalized = clone_case(case)
    for tensor in mapped_inputs(normalized).values():
        _reset_tensor_range(tensor)
    return normalized


def _require_exact_keys(obj: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(obj.keys())
    if actual != expected:
        raise ValueError(f"{label}: expected keys {sorted(expected)}, got {sorted(actual)}")


def _validate_tensor_shape(shape: Any, label: str) -> None:
    if shape is None:
        return
    if not isinstance(shape, list) or not all(isinstance(dim, int) for dim in shape):
        raise ValueError(f"{label}: shape must be a list[int] or null")


def _validate_single_tensor(tensor: dict[str, Any], label: str, *, allow_data_range: bool) -> None:
    expected = {"kind", "dtype", "format", "shape", "param_type"}
    if allow_data_range:
        expected.add("data_range")
    _require_exact_keys(tensor, expected, label)
    _validate_tensor_shape(tensor.get("shape"), label)
    if not isinstance(tensor.get("param_type"), str) or not tensor["param_type"]:
        raise ValueError(f"{label}: param_type must be a non-empty string")
    if allow_data_range:
        if not isinstance(tensor.get("data_range"), str) or not tensor["data_range"]:
            raise ValueError(f"{label}: data_range must be a non-empty string")
    elif "data_range" in tensor:
        raise ValueError(f"{label}: data_range is not allowed")


def _validate_tensor_list_children(tensor: dict[str, Any], label: str, *, allow_data_range: bool) -> None:
    child_expected = {"kind", "dtype", "format", "shape"}
    if allow_data_range:
        child_expected.add("data_range")
    for child_index, child in enumerate(tensor["tensors"]):
        if not isinstance(child, dict):
            raise ValueError(f"{label}: child {child_index} must be an object")
        _require_exact_keys(child, child_expected, f"{label}: child {child_index}")
        _validate_tensor_shape(child.get("shape"), f"{label}: child {child_index}")
        if child.get("kind") != "tensor":
            raise ValueError(f"{label}: child {child_index} kind must be tensor")
        if not isinstance(child.get("dtype"), str) or not child["dtype"]:
            raise ValueError(f"{label}: child {child_index} dtype must be a non-empty string")
        if not isinstance(child.get("format"), str) or not child["format"]:
            raise ValueError(f"{label}: child {child_index} format must be a non-empty string")
        if allow_data_range:
            if not isinstance(child.get("data_range"), str) or not child["data_range"]:
                raise ValueError(f"{label}: child {child_index} data_range must be a non-empty string")
        elif "data_range" in child:
            raise ValueError(f"{label}: child {child_index} data_range is not allowed")


def _validate_tensor_list_descriptor(tensor: dict[str, Any], label: str, *, allow_data_range: bool) -> None:
    expected = {"kind", "dtype", "format", "param_type", "tensor_count", "tensors"}
    if allow_data_range:
        expected.add("data_range")
    _require_exact_keys(tensor, expected, label)
    if tensor.get("param_type") != "DYNAMIC":
        raise ValueError(f"{label}: param_type must be DYNAMIC")
    if not isinstance(tensor.get("tensor_count"), int) or tensor["tensor_count"] < 0:
        raise ValueError(f"{label}: tensor_count must be a non-negative int")
    if not isinstance(tensor.get("tensors"), list):
        raise ValueError(f"{label}: tensors must be a list")
    if tensor["tensor_count"] != len(tensor["tensors"]):
        raise ValueError(f"{label}: tensor_count must equal len(tensors)")
    if allow_data_range:
        if not isinstance(tensor.get("data_range"), str) or not tensor["data_range"]:
            raise ValueError(f"{label}: data_range must be a non-empty string")
    elif "data_range" in tensor:
        raise ValueError(f"{label}: data_range is not allowed")
    _validate_tensor_list_children(tensor, label, allow_data_range=allow_data_range)


def _validate_tensor_descriptor(
    tensor: dict[str, Any], label: str, *, allow_data_range: bool, kind: str
) -> None:
    if tensor.get("kind") != kind:
        raise ValueError(f"{label}: kind must be {kind}")
    if not isinstance(tensor.get("dtype"), str) or not tensor["dtype"]:
        raise ValueError(f"{label}: dtype must be a non-empty string")
    if not isinstance(tensor.get("format"), str) or not tensor["format"]:
        raise ValueError(f"{label}: format must be a non-empty string")
    if kind == "tensor":
        _validate_single_tensor(tensor, label, allow_data_range=allow_data_range)
    else:
        _validate_tensor_list_descriptor(tensor, label, allow_data_range=allow_data_range)


def _validate_mapped_case_fields(case: dict[str, Any], label: str) -> None:
    if not isinstance(case, dict):
        raise ValueError(f"{label}: case must be an object")
    expected = {"id", "source", "attributes", "const_inputs", "inputs", "outputs", "meta"}
    _require_exact_keys(case, expected, label)
    if not isinstance(case["id"], str):
        raise ValueError(f"{label}: id must be a string")
    if not isinstance(case["source"], str):
        raise ValueError(f"{label}: source must be a string")
    if not isinstance(case["attributes"], dict) or not isinstance(case["const_inputs"], dict):
        raise ValueError(f"{label}: attributes and const_inputs must be objects")
    if not isinstance(case["inputs"], dict) or not isinstance(case["outputs"], dict):
        raise ValueError(f"{label}: inputs and outputs must be objects")
    if not isinstance(case["meta"], dict):
        raise ValueError(f"{label}: meta must be an object")
    if "supported_data_ranges" in case["meta"]:
        raise ValueError(f"{label}: meta.supported_data_ranges is not allowed")


def _validate_mapped_input(name: str, tensor: dict[str, Any], label: str) -> None:
    if not isinstance(tensor, dict):
        raise ValueError(f"{label}: input {name} must be an object")
    kind = tensor.get("kind")
    if kind == "tensor_list" or tensor.get("param_type") == "DYNAMIC":
        _validate_tensor_descriptor(tensor, f"{label}: input {name}", allow_data_range=True, kind="tensor_list")
    else:
        _validate_tensor_descriptor(tensor, f"{label}: input {name}", allow_data_range=True, kind="tensor")


def _validate_mapped_outputs(outputs: dict[str, Any], label: str) -> None:
    for name, tensor in outputs.items():
        if not isinstance(tensor, dict):
            raise ValueError(f"{label}: output {name} must be an object")
        kind = tensor.get("kind")
        if kind == "tensor_list" or tensor.get("param_type") == "DYNAMIC":
            _validate_tensor_descriptor(tensor, f"{label}: output {name}", allow_data_range=False, kind="tensor_list")
        else:
            _validate_tensor_descriptor(tensor, f"{label}: output {name}", allow_data_range=False, kind="tensor")


def validate_mapped_case(case: Any, label: str) -> dict[str, Any]:
    _validate_mapped_case_fields(case, label)
    for name, tensor in case["inputs"].items():
        _validate_mapped_input(name, tensor, label)
    _validate_mapped_outputs(case["outputs"], label)
    return normalize_input_ranges(case)


def map_path_case(case: dict[str, Any], index: int) -> dict[str, Any]:
    mapped = build_low_base_case(case, "path", index)
    mapped.setdefault("id", path_case_id(index, case))
    mapped.setdefault("source", "path")
    mapped["id"] = str(mapped["id"])
    mapped["source"] = "path"
    return validate_mapped_case(mapped, mapped["id"])


def map_network_case(record: dict[str, Any], index: int) -> dict[str, Any]:
    mapped = build_low_base_case(record, "network", index)
    mapped.setdefault("id", network_case_id(index, record))
    mapped.setdefault("source", "network")
    mapped["id"] = str(mapped["id"])
    mapped["source"] = "network"
    return validate_mapped_case(mapped, mapped["id"])


def make_shape_variants(case: dict[str, Any]) -> list[dict[str, Any]]:
    source = case.get("source")
    if source == "path":
        return [make_path_shape_case(case)]
    if source == "network":
        return [make_network_shape_case(case)]
    raise ValueError(f"{case.get('id', '<unknown>')}: unsupported source {source!r}")


def assert_unique_ids(cases: list[dict[str, Any]], label: str) -> None:
    seen: set[str] = set()
    for case in cases:
        case_id = case["id"]
        if case_id in seen:
            raise ValueError(f"duplicate id in {label}: {case_id}")
        seen.add(case_id)


def build_low_shape_cases(
    path_cases: list[dict[str, Any]], network_cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base_cases = path_cases + network_cases
    shape_cases: list[dict[str, Any]] = []

    for case in base_cases:
        shape_variants = make_shape_variants(case)
        if not shape_variants:
            raise ValueError(f"{case['id']}: make_shape_variants must return at least one shape low case")
        for variant in shape_variants:
            if not isinstance(variant, dict):
                raise ValueError(f"{case['id']}: shape variant must be an object")
            variant.setdefault("source", "shape")
            variant.setdefault("meta", {})
            if not isinstance(variant["meta"], dict):
                raise ValueError(f"{case['id']}: shape variant meta must be an object")
            variant["source"] = "shape"
            variant["id"] = f"low_case_{len(shape_cases):02d}"
            variant["meta"].setdefault("base_id", case["id"])
            variant["meta"].setdefault("variant_kind", "shape")
            shape_cases.append(validate_mapped_case(variant, variant["id"]))

    assert_unique_ids(shape_cases, "S5_mapped_cases_low_shape.json")
    return shape_cases


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path_cases = load_json("S2P2_cases.json")
    network_configs = load_json("S2P1_low_configs.json")
    mapped_path = [map_path_case(case, index) for index, case in enumerate(path_cases)]
    mapped_network = [
        map_network_case(network_record(config, index), index)
        for index, config in enumerate(network_configs)
    ]
    low_shape_cases = build_low_shape_cases(mapped_path, mapped_network)

    dump_json("S5_mapped_cases_path.json", mapped_path)
    dump_json("S5_mapped_cases_network.json", mapped_network)
    dump_json("S5_mapped_cases_low_shape.json", low_shape_cases)
    _logger.info(
        f"wrote {len(mapped_path)} path cases, {len(mapped_network)} mapped network cases,"
        f" {len(low_shape_cases)} low shape cases"
    )


# =========================
# Dynamic region: operator-specific
# =========================


def build_low_base_case(record: dict[str, Any], source: str, index: int) -> dict[str, Any]:
    """TODO(operator-specific): construct one complete base mapped case."""
    raise NotImplementedError


def make_path_shape_case(case: dict[str, Any]) -> dict[str, Any]:
    """TODO(operator-specific): map one path base case to one shape low case."""
    raise NotImplementedError


def make_network_shape_case(case: dict[str, Any]) -> dict[str, Any]:
    """TODO(operator-specific): map one network base case to one shape low case."""
    raise NotImplementedError


def derive_outputs(
    inputs: dict[str, Any],
    attributes: dict[str, Any],
    const_inputs: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """TODO(operator-specific): derive complete V1 output descriptors."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
