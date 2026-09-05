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
"""Generate TTK kernel CSV from Mapper-v1 final low/high case files."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)


KERNEL_COLUMNS = [
    "testcase_name", "network_name", "op_name",
    "input_shapes", "input_dtypes", "input_formats",
    "output_shapes", "output_dtypes", "output_formats",
    "input_ori_shapes", "input_ori_formats",
    "output_ori_shapes", "output_ori_formats",
    "attributes", "input_data_ranges", "precision_tolerances",
    "absolute_precision",
    "output_inplace_indexes", "output_shape_unknown_indexes",
    "is_enabled", "remark", "soc_series", "priority",
    "dump_file_prefix", "manual_input_binaries", "manual_golden_binaries",
]

DTYPE_MAX = {
    "float16": 65504.0,
    "float": 3.4e38,
    "float32": 3.4e38,
    "bfloat16": 3.3895e38,
    "int32": 2147483647,
    "int64": 9223372036854775807,
}
PRECISION_TOLERANCE = {
    "float16": (0.001, 0.001),
    "bfloat16": (0.001, 0.001),
    "float": (0.0001, 0.0001),
    "float32": (0.0001, 0.0001),
}


class KernelCsvError(RuntimeError):
    pass


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_cases(path: str) -> list[dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise KernelCsvError(f"{path} must be a Step 5 final case list")
    return data


def py_literal(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        if math.isnan(value):
            return "float('nan')"
        if math.isinf(value):
            return "float('inf')" if value > 0 else "float('-inf')"
        return repr(value)
    if isinstance(value, int):
        return repr(value)
    if isinstance(value, tuple):
        return tuple_literal(value)
    if isinstance(value, list):
        return tuple_literal(tuple(value))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return repr(value)


def tuple_literal(values: tuple[Any, ...] | list[Any]) -> str:
    values = tuple(values)
    if not values:
        return "()"
    inner = ", ".join(py_literal(v) for v in values)
    if len(values) == 1:
        inner += ","
    return f"({inner})"


def shape_entry(shape: Any, field_name: str) -> tuple[int, ...] | None:
    if shape is None:
        return None
    if not isinstance(shape, list) or not all(isinstance(dim, int) for dim in shape):
        raise KernelCsvError(f"{field_name}.shape must be list[int] or null")
    return tuple(shape)


def get_tensor_entries(container: Any, case_id: str, field_name: str) -> list[tuple[str, dict[str, Any]]]:
    if not isinstance(container, dict):
        raise KernelCsvError(f"case {case_id}: {field_name} must be an object")
    result = []
    for name, spec in container.items():
        if not isinstance(spec, dict):
            raise KernelCsvError(f"case {case_id}: {field_name}.{name} must be an object")
        result.append((name, spec))
    return result


def tensor_children(spec: dict[str, Any], name: str) -> list[dict[str, Any]]:
    tensors = spec.get("tensors")
    if not isinstance(tensors, list):
        raise KernelCsvError(f"{name}.tensors must be a list")
    children: list[dict[str, Any]] = []
    for index, child in enumerate(tensors):
        if not isinstance(child, dict):
            raise KernelCsvError(f"{name}.tensors[{index}] must be an object")
        children.append(child)
    return children


def tensor_shape_entry(spec: dict[str, Any], name: str) -> tuple[Any, ...] | None:
    kind = spec.get("kind")
    if kind == "tensor_list":
        return tuple(shape_entry(child.get("shape"), f"{name}.tensors[]") for child in tensor_children(spec, name))
    if kind == "tensor":
        return shape_entry(spec.get("shape"), name)
    raise KernelCsvError(f"{name}.kind must be tensor or tensor_list")


def _string_field(spec: dict[str, Any], field: str, name: str) -> str:
    value = spec.get(field)
    if not isinstance(value, str) or not value:
        raise KernelCsvError(f"{name}.{field} must be a non-empty string")
    return value


def _compressed_or_expanded(values: list[str]) -> tuple[str, ...]:
    if not values:
        return ()
    if all(value == values[0] for value in values):
        return (values[0],)
    return tuple(values)


def tensor_dtype_entry(spec: dict[str, Any], name: str) -> str | tuple[str, ...]:
    kind = spec.get("kind")
    if kind == "tensor_list":
        children = tensor_children(spec, name)
        if not children:
            return (_string_field(spec, "dtype", name),)
        return _compressed_or_expanded([_string_field(child, "dtype", f"{name}.tensors[]") for child in children])
    if kind == "tensor":
        return _string_field(spec, "dtype", name)
    raise KernelCsvError(f"{name}.kind must be tensor or tensor_list")


def tensor_format_entry(spec: dict[str, Any], name: str) -> str | tuple[str, ...]:
    kind = spec.get("kind")
    if kind == "tensor_list":
        children = tensor_children(spec, name)
        if not children:
            return (_string_field(spec, "format", name),)
        return _compressed_or_expanded([_string_field(child, "format", f"{name}.tensors[]") for child in children])
    if kind == "tensor":
        return _string_field(spec, "format", name)
    raise KernelCsvError(f"{name}.kind must be tensor or tensor_list")


def _range_normal(rng: random.Random) -> tuple[float, float]:
    a, b = rng.uniform(-10, 10), rng.uniform(-10, 10)
    return (round(min(a, b), 2), round(max(a, b), 2))


def _range_negative(rng: random.Random) -> tuple[float, float]:
    a, b = rng.uniform(-100, -0.01), rng.uniform(-100, -0.01)
    return (round(min(a, b), 2), round(max(a, b), 2))


def data_range_to_ttk(data_range: str, dtype_str: str, rng: random.Random) -> tuple[Any, Any]:
    if data_range == "normal":
        return _range_normal(rng)
    if data_range == "zero":
        return (0, 0)
    if data_range == "extreme":
        mx = DTYPE_MAX.get(dtype_str, 3.4e38)
        return (mx * 0.9, mx)
    if data_range == "negative":
        return _range_negative(rng)
    if data_range == "tiny_pos":
        return (1e-7, 1e-5)
    if data_range == "all_ones":
        return (1, 1)
    if data_range == "near_zero":
        return (-0.01, 0.01)
    if data_range == "with_inf":
        return (1, float("inf"))
    if data_range == "with_nan":
        return (float("nan"), float("nan"))
    raise KernelCsvError(f"unsupported data_range {data_range!r}")


def input_range_entry(spec: dict[str, Any], name: str, rng: random.Random) -> tuple[Any, ...]:
    kind = spec.get("kind")
    if kind == "tensor_list":
        ranges = []
        for child in tensor_children(spec, name):
            data_range = _string_field(child, "data_range", f"{name}.tensors[]")
            dtype = _string_field(child, "dtype", f"{name}.tensors[]")
            ranges.append(data_range_to_ttk(data_range, dtype, rng))
        return tuple(ranges)
    if kind == "tensor":
        data_range = _string_field(spec, "data_range", name)
        dtype = _string_field(spec, "dtype", name)
        return data_range_to_ttk(data_range, dtype, rng)
    raise KernelCsvError(f"{name}.kind must be tensor or tensor_list")


def tolerance_for_dtype(dtype: str) -> tuple[float, float]:
    return PRECISION_TOLERANCE.get(dtype, (0.001, 0.001))


def tolerance_entry(spec: dict[str, Any], name: str) -> tuple[Any, ...]:
    kind = spec.get("kind")
    if kind == "tensor_list":
        return tuple(
            tolerance_for_dtype(_string_field(child, "dtype", f"{name}.tensors[]"))
            for child in tensor_children(spec, name)
        )
    if kind == "tensor":
        return tolerance_for_dtype(_string_field(spec, "dtype", name))
    raise KernelCsvError(f"{name}.kind must be tensor or tensor_list")


def case_attributes(case: dict[str, Any], case_id: str) -> dict[str, Any]:
    attributes = case.get("attributes")
    const_inputs = case.get("const_inputs")
    if not isinstance(attributes, dict) or not isinstance(const_inputs, dict):
        raise KernelCsvError(f"case {case_id}: attributes and const_inputs must be objects")
    merged = dict(attributes)
    merged.update(const_inputs)
    return merged


def build_remark(case: dict[str, Any]) -> str:
    meta = case.get("meta", {}) if isinstance(case.get("meta", {}), dict) else {}
    parts = [f"original_id={case.get('id', '')}", f"source={case.get('source', '')}"]
    for key in ("source_kind", "path", "tiling_key", "network_name", "variant_kind"):
        if key in meta:
            parts.append(f"{key}={meta[key]}")
    return "; ".join(parts)


def build_row(case: dict[str, Any], index: int, op_name: str, rng: random.Random) -> dict[str, str]:
    case_id = str(case.get("id", f"case{index:05d}"))
    inputs = get_tensor_entries(case.get("inputs"), case_id, "inputs")
    outputs = get_tensor_entries(case.get("outputs"), case_id, "outputs")

    input_shapes = tuple(tensor_shape_entry(spec, name) for name, spec in inputs)
    input_dtypes = tuple(tensor_dtype_entry(spec, name) for name, spec in inputs)
    input_formats = tuple(tensor_format_entry(spec, name) for name, spec in inputs)
    output_shapes = tuple(tensor_shape_entry(spec, name) for name, spec in outputs)
    output_dtypes = tuple(tensor_dtype_entry(spec, name) for name, spec in outputs)
    output_formats = tuple(tensor_format_entry(spec, name) for name, spec in outputs)
    input_data_ranges = tuple(input_range_entry(spec, name, rng) for name, spec in inputs)
    precision_tolerances = tuple(tolerance_entry(spec, name) for name, spec in outputs)

    meta = case.get("meta", {}) if isinstance(case.get("meta", {}), dict) else {}
    return {
        "testcase_name": f"case{index:05d}",
        "network_name": str(meta.get("network_name", "")),
        "op_name": op_name,
        "input_shapes": tuple_literal(input_shapes),
        "input_dtypes": tuple_literal(input_dtypes),
        "input_formats": tuple_literal(input_formats),
        "output_shapes": tuple_literal(output_shapes),
        "output_dtypes": tuple_literal(output_dtypes),
        "output_formats": tuple_literal(output_formats),
        "input_ori_shapes": "",
        "input_ori_formats": "",
        "output_ori_shapes": "",
        "output_ori_formats": "",
        "attributes": py_literal(case_attributes(case, case_id)),
        "input_data_ranges": tuple_literal(input_data_ranges),
        "precision_tolerances": tuple_literal(precision_tolerances),
        "absolute_precision": "1e-8",
        "output_inplace_indexes": "()",
        "output_shape_unknown_indexes": "()",
        "is_enabled": "True",
        "remark": build_remark(case),
        "soc_series": "",
        "priority": "0",
        "dump_file_prefix": "",
        "manual_input_binaries": "()",
        "manual_golden_binaries": "()",
    }


def write_csv(cases: list[dict[str, Any]], output_path: str, op_name: str) -> int:
    rng = random.Random(42)
    rows = [build_row(case, idx, op_name, rng) for idx, case in enumerate(cases)]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=KERNEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TTK kernel CSV from Mapper-v1 final case files.")
    parser.add_argument("--op-name", required=True, help="Normalized operator name, e.g. add_rms_norm")
    parser.add_argument("--low-cases", required=True, help="Path to S5_cases_low.json")
    parser.add_argument("--high-cases", required=True, help="Path to S5_cases_high.json")
    parser.add_argument("--low-csv", required=True, help="Output low CSV path")
    parser.add_argument("--high-csv", required=True, help="Output high CSV path")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    low_cases = load_cases(args.low_cases)
    high_cases = load_cases(args.high_cases)
    low_count = write_csv(low_cases, args.low_csv, args.op_name)
    high_count = write_csv(high_cases, args.high_csv, args.op_name)
    _logger.info("PASS: wrote %d low rows to %s", low_count, Path(args.low_csv))
    _logger.info("PASS: wrote %d high rows to %s", high_count, Path(args.high_csv))


if __name__ == "__main__":
    main()
