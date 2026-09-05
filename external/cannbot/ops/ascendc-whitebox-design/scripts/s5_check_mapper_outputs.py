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
"""Validate Mapper-v1 final low/high case JSON structure."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)


FINAL_JSON_FILES = (
    "S5_cases_low.json",
    "S5_cases_high.json",
)

TOP_LEVEL_FIELDS = ("id", "source", "attributes", "const_inputs", "inputs", "outputs", "meta")
INPUT_TENSOR_FIELDS = ("kind", "dtype", "format", "shape", "param_type", "data_range")
OUTPUT_TENSOR_FIELDS = ("kind", "dtype", "format", "shape", "param_type")
INPUT_TENSOR_LIST_FIELDS = ("kind", "dtype", "format", "param_type", "tensor_count", "data_range", "tensors")
OUTPUT_TENSOR_LIST_FIELDS = ("kind", "dtype", "format", "param_type", "tensor_count", "tensors")
INPUT_TENSOR_LIST_CHILD_FIELDS = ("kind", "dtype", "format", "shape", "data_range")
OUTPUT_TENSOR_LIST_CHILD_FIELDS = ("kind", "dtype", "format", "shape")


@dataclass
class ErrorCtx:
    errors: list[str]
    file_name: str
    case_id: str


def add_error(errors: list[str], file_name: str, case_id: str, field_path: str, message: str) -> None:
    errors.append(f"{file_name}: {case_id}: {field_path}: {message}")


def load_json(path: Path, errors: list[str]) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001 - report parse/open failure as validation error
        add_error(errors, path.name, "<file>", "$", f"failed to parse JSON: {exc}")
        return None


def check_exact_fields(obj: dict[str, Any], fields: tuple[str, ...], ctx: ErrorCtx, field_path: str) -> None:
    expected = set(fields)
    actual = set(obj)
    for field in fields:
        if field not in obj:
            add_error(ctx.errors, ctx.file_name, ctx.case_id, f"{field_path}.{field}", "missing field")
    for field in sorted(actual - expected):
        add_error(ctx.errors, ctx.file_name, ctx.case_id, f"{field_path}.{field}", "unexpected field")


def check_string(value: Any, ctx: ErrorCtx, field_path: str) -> None:
    if not isinstance(value, str):
        add_error(ctx.errors, ctx.file_name, ctx.case_id, field_path, "must be a string")


def check_non_empty_string(value: Any, ctx: ErrorCtx, field_path: str) -> None:
    if not isinstance(value, str) or not value:
        add_error(ctx.errors, ctx.file_name, ctx.case_id, field_path, "must be a non-empty string")


def check_object(value: Any, ctx: ErrorCtx, field_path: str) -> bool:
    if not isinstance(value, dict):
        add_error(ctx.errors, ctx.file_name, ctx.case_id, field_path, "must be an object")
        return False
    return True


def check_shape(value: Any, ctx: ErrorCtx, field_path: str) -> None:
    if value is None:
        return
    if not isinstance(value, list) or not all(isinstance(dim, int) for dim in value):
        add_error(ctx.errors, ctx.file_name, ctx.case_id, field_path, "shape must be list[int] or null")


def check_tensor_common(tensor: dict[str, Any], ctx: ErrorCtx, field_path: str) -> None:
    if tensor.get("kind") != "tensor":
        add_error(ctx.errors, ctx.file_name, ctx.case_id, f"{field_path}.kind", "expected 'tensor'")
    if "dtype" in tensor:
        check_non_empty_string(tensor["dtype"], ctx, f"{field_path}.dtype")
    if "format" in tensor:
        check_non_empty_string(tensor["format"], ctx, f"{field_path}.format")
    if "shape" in tensor:
        check_shape(tensor["shape"], ctx, f"{field_path}.shape")


def check_tensor_input(tensor: dict[str, Any], ctx: ErrorCtx, field_path: str) -> None:
    check_exact_fields(tensor, INPUT_TENSOR_FIELDS, ctx, field_path)
    check_tensor_common(tensor, ctx, field_path)
    if "param_type" in tensor:
        check_string(tensor["param_type"], ctx, f"{field_path}.param_type")
    if "data_range" in tensor:
        check_string(tensor["data_range"], ctx, f"{field_path}.data_range")


def check_tensor_output(tensor: dict[str, Any], ctx: ErrorCtx, field_path: str) -> None:
    check_exact_fields(tensor, OUTPUT_TENSOR_FIELDS, ctx, field_path)
    check_tensor_common(tensor, ctx, field_path)
    if "param_type" in tensor:
        check_string(tensor["param_type"], ctx, f"{field_path}.param_type")


def check_tensor_list_common(
    tensor_list: dict[str, Any], ctx: ErrorCtx, field_path: str
) -> tuple[int | None, list[Any] | None]:
    if tensor_list.get("kind") != "tensor_list":
        add_error(ctx.errors, ctx.file_name, ctx.case_id, f"{field_path}.kind", "expected 'tensor_list'")
    if "dtype" in tensor_list:
        check_non_empty_string(tensor_list["dtype"], ctx, f"{field_path}.dtype")
    if "format" in tensor_list:
        check_non_empty_string(tensor_list["format"], ctx, f"{field_path}.format")
    if "param_type" in tensor_list:
        check_string(tensor_list["param_type"], ctx, f"{field_path}.param_type")

    tensor_count = tensor_list.get("tensor_count")
    tensors = tensor_list.get("tensors")
    if not isinstance(tensor_count, int):
        add_error(ctx.errors, ctx.file_name, ctx.case_id, f"{field_path}.tensor_count", "must be an int")
        tensor_count = None
    if not isinstance(tensors, list):
        add_error(ctx.errors, ctx.file_name, ctx.case_id, f"{field_path}.tensors", "must be a list")
        tensors = None
    if tensor_count is not None and tensors is not None and tensor_count != len(tensors):
        add_error(ctx.errors, ctx.file_name, ctx.case_id, f"{field_path}.tensor_count", "must equal len(tensors)")
    return tensor_count, tensors


def check_input_tensor_list_child(child: Any, ctx: ErrorCtx, field_path: str) -> None:
    if not check_object(child, ctx, field_path):
        return
    check_exact_fields(child, INPUT_TENSOR_LIST_CHILD_FIELDS, ctx, field_path)
    check_tensor_common(child, ctx, field_path)
    if "data_range" in child:
        check_string(child["data_range"], ctx, f"{field_path}.data_range")


def check_output_tensor_list_child(child: Any, ctx: ErrorCtx, field_path: str) -> None:
    if not check_object(child, ctx, field_path):
        return
    check_exact_fields(child, OUTPUT_TENSOR_LIST_CHILD_FIELDS, ctx, field_path)
    check_tensor_common(child, ctx, field_path)


def check_tensor_list_input(tensor_list: dict[str, Any], ctx: ErrorCtx, field_path: str) -> None:
    check_exact_fields(tensor_list, INPUT_TENSOR_LIST_FIELDS, ctx, field_path)
    _, tensors = check_tensor_list_common(tensor_list, ctx, field_path)
    if "data_range" in tensor_list:
        check_string(tensor_list["data_range"], ctx, f"{field_path}.data_range")
    if tensors is None:
        return
    for index, child in enumerate(tensors):
        check_input_tensor_list_child(child, ctx, f"{field_path}.tensors[{index}]")


def check_tensor_list_output(tensor_list: dict[str, Any], ctx: ErrorCtx, field_path: str) -> None:
    check_exact_fields(tensor_list, OUTPUT_TENSOR_LIST_FIELDS, ctx, field_path)
    _, tensors = check_tensor_list_common(tensor_list, ctx, field_path)
    if tensors is None:
        return
    for index, child in enumerate(tensors):
        check_output_tensor_list_child(child, ctx, f"{field_path}.tensors[{index}]")


def check_input_descriptor(descriptor: Any, ctx: ErrorCtx, field_path: str) -> None:
    if not check_object(descriptor, ctx, field_path):
        return
    kind = descriptor.get("kind")
    if kind == "tensor":
        check_tensor_input(descriptor, ctx, field_path)
    elif kind == "tensor_list":
        check_tensor_list_input(descriptor, ctx, field_path)
    else:
        add_error(ctx.errors, ctx.file_name, ctx.case_id, f"{field_path}.kind", "expected 'tensor' or 'tensor_list'")


def check_output_descriptor(descriptor: Any, ctx: ErrorCtx, field_path: str) -> None:
    if not check_object(descriptor, ctx, field_path):
        return
    kind = descriptor.get("kind")
    if kind == "tensor":
        check_tensor_output(descriptor, ctx, field_path)
    elif kind == "tensor_list":
        check_tensor_list_output(descriptor, ctx, field_path)
    else:
        add_error(ctx.errors, ctx.file_name, ctx.case_id, f"{field_path}.kind", "expected 'tensor' or 'tensor_list'")


def check_case_schema(case: Any, index: int, file_name: str, errors: list[str]) -> None:
    case_id = f"[{index}]"
    ctx = ErrorCtx(errors, file_name, case_id)
    if not check_object(case, ctx, "$"):
        return

    case_id = str(case.get("id", f"[{index}]"))
    ctx = ErrorCtx(errors, file_name, case_id)
    check_exact_fields(case, TOP_LEVEL_FIELDS, ctx, "$")

    if "id" in case:
        check_string(case["id"], ctx, "id")
    if "source" in case:
        check_string(case["source"], ctx, "source")

    for field in ("attributes", "const_inputs", "inputs", "outputs", "meta"):
        if field in case:
            check_object(case[field], ctx, field)

    inputs = case.get("inputs")
    if isinstance(inputs, dict):
        for input_name, descriptor in inputs.items():
            check_input_descriptor(descriptor, ctx, f"inputs.{input_name}")

    outputs = case.get("outputs")
    if isinstance(outputs, dict):
        for output_name, descriptor in outputs.items():
            check_output_descriptor(descriptor, ctx, f"outputs.{output_name}")


def check_json_file(path: Path, errors: list[str]) -> int:
    if not path.is_file():
        add_error(errors, path.name, "<file>", "$", "required output json is missing")
        return 0

    data = load_json(path, errors)
    if data is None:
        return 0
    if not isinstance(data, list):
        add_error(errors, path.name, "<file>", "$", "must be a JSON list")
        return 0

    for index, case in enumerate(data):
        check_case_schema(case, index, path.name, errors)
    return len(data)


def check_files(paths: list[Path]) -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    stats = {path.name: check_json_file(path, errors) for path in paths}
    return errors, stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Validate Mapper-v1 final low/high case JSON structure.")
    parser.add_argument("--whitebox-dir", help="Directory containing S5_cases_low.json and S5_cases_high.json.")
    parser.add_argument(
        "--json-file",
        action="append",
        default=[],
        help="Specific case JSON file to validate. Can be passed multiple times.",
    )
    args = parser.parse_args()

    if bool(args.whitebox_dir) == bool(args.json_file):
        parser.error("pass exactly one of --whitebox-dir or --json-file")

    if args.whitebox_dir:
        whitebox_dir = Path(args.whitebox_dir).resolve()
        paths = [whitebox_dir / name for name in FINAL_JSON_FILES]
    else:
        paths = [Path(name).resolve() for name in args.json_file]

    errors, stats = check_files(paths)
    if errors:
        _logger.info("FAIL: mapper schema validation errors")
        for error in errors:
            _logger.info(error)
        raise SystemExit(1)

    _logger.info("PASS: mapper schema accepted")
    for path in paths:
        _logger.info(f"{path.name} cases={stats[path.name]}")


if __name__ == "__main__":
    main()
