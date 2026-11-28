#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Load and validate the single JSON machine contract in ``SPEC.md``."""

from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any

BLOCK_RE = re.compile(r"(?ms)^```json[ \t]+machine-contract[ \t]*\r?\n(.*?)^```[ \t]*$")
PLACEHOLDER_RE = re.compile(r"\{\{[^{}]+\}\}")
NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
SYMBOL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
REQUIRED = {
    "schema_version", "op_name", "formula", "supported_dtypes", "inputs", "outputs",
    "default_params", "tolerance", "dynamic_axes_ranges", "shape_constraints", "p0_cases",
}
OPTIONAL = {"perf_target", "exit_criteria"}
TENSOR_FIELDS = {"name", "shape", "dtype", "value_range"}
CASE_FIELDS = {"name", "params", "input_shapes", "output_shapes"}
CASE_DTYPE_FIELDS = {"input_dtypes", "output_dtypes"}
CASE_SPECIAL_FIELDS = {"input_special_values"}
SPECIAL_VALUES = {"-inf", "+inf", "nan"}
DTYPE_VOCAB = {
    "bfloat16", "float16", "float32", "float64", "int8", "uint8", "int16",
    "int32", "int64", "bool",
}


class SpecContractError(ValueError):
    """The SPEC cannot be consumed as a canonical contract."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SpecContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _extract(text: str) -> dict[str, Any]:
    if PLACEHOLDER_RE.search(text):
        raise SpecContractError("unresolved template placeholder remains")
    blocks = BLOCK_RE.findall(text)
    if len(blocks) != 1:
        raise SpecContractError("SPEC.md must contain exactly one fenced `json machine-contract` block")
    try:
        value = json.loads(
            blocks[0], object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SpecContractError(f"non-finite JSON number: {token}")),
        )
    except json.JSONDecodeError as exc:
        raise SpecContractError(f"invalid machine-contract JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SpecContractError("machine contract must be a JSON object")
    return value


def _keys(value: dict[str, Any], required: set[str], optional: set[str], where: str) -> None:
    missing, unknown = sorted(required - value.keys()), sorted(value.keys() - required - optional)
    if missing:
        raise SpecContractError(f"{where} missing fields: {', '.join(missing)}")
    if unknown:
        raise SpecContractError(f"{where} unknown fields: {', '.join(unknown)}")


def _number(value: Any, where: str, *, non_negative: bool = False) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SpecContractError(f"{where} must be a finite number")
    if non_negative and value < 0:
        raise SpecContractError(f"{where} must be non-negative")
    return value


def _eval_node(node: ast.AST, env: dict[str, int], where: str) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise SpecContractError(f"{where} uses unbound shape symbol {node.id}")
        return env[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_node(node.operand, env, where)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv)):
        left, right = _eval_node(node.left, env, where), _eval_node(node.right, env, where)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0:
            raise SpecContractError(f"{where} divides by zero")
        return left // right
    raise SpecContractError(f"{where} permits only names, integers, +, -, *, // and parentheses")


def _shape(value: Any, where: str, *, concrete: bool = False) -> list[int | str]:
    if not isinstance(value, list):
        raise SpecContractError(f"{where} must be a JSON array")
    if value == ["..."]:
        if concrete:
            raise SpecContractError(f"{where} must be concrete")
        return value
    if "..." in value:
        raise SpecContractError(f"{where} rank wildcard must be the sole dimension")
    for dim in value:
        if isinstance(dim, bool) or not isinstance(dim, (int, str)):
            raise SpecContractError(f"{where} dimensions must be integers or expressions")
        if isinstance(dim, int) and dim <= 0:
            raise SpecContractError(f"{where} dimensions must be positive")
        if isinstance(dim, str):
            if concrete or not dim.strip():
                raise SpecContractError(f"{where} must be concrete")
            try:
                tree = ast.parse(dim, mode="eval")
            except SyntaxError as exc:
                raise SpecContractError(f"invalid shape expression in {where}: {dim}") from exc
            _check_shape_syntax(tree.body, where)
    return value


def _check_shape_syntax(node: ast.AST, where: str) -> None:
    """Validate the expression grammar without inventing values for symbols."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return
    if isinstance(node, ast.Name):
        return
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        _check_shape_syntax(node.operand, where)
        return
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv)):
        _check_shape_syntax(node.left, where)
        _check_shape_syntax(node.right, where)
        return
    raise SpecContractError(
        f"{where} permits only names, integers, +, -, *, // and parentheses"
    )


def _shape_symbols(tensors: list[dict[str, Any]]) -> set[str]:
    symbols: set[str] = set()
    for tensor in tensors:
        for dim in tensor["shape"]:
            if isinstance(dim, str):
                if dim == "...":
                    continue
                symbols.update(
                    node.id for node in ast.walk(ast.parse(dim, mode="eval"))
                    if isinstance(node, ast.Name)
                )
    return symbols


def _eval_dim(value: int | str, env: dict[str, int], where: str) -> int:
    result = value if isinstance(value, int) else _eval_node(ast.parse(value, mode="eval").body, env, where)
    if result <= 0:
        raise SpecContractError(f"{where} evaluates to non-positive dimension {result}")
    return result


def _tensors(value: Any, where: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise SpecContractError(f"{where} must be a non-empty array")
    names: set[str] = set()
    for index, item in enumerate(value):
        label = f"{where}[{index}]"
        if not isinstance(item, dict):
            raise SpecContractError(f"{label} must be an object")
        _keys(item, TENSOR_FIELDS, {"is_list"}, label)
        if not isinstance(item.get("is_list", False), bool):
            raise SpecContractError(f"{label}.is_list must be boolean")
        name = item["name"]
        if not isinstance(name, str) or not NAME_RE.fullmatch(name) or name in names:
            raise SpecContractError(f"{label}.name must be a unique lower_snake_case name")
        names.add(name)
        _shape(item["shape"], f"{label}.shape")
        if not isinstance(item["dtype"], str) or not NAME_RE.fullmatch(item["dtype"]):
            raise SpecContractError(f"{label}.dtype must be a canonical dtype name")
        bounds = item["value_range"]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise SpecContractError(f"{label}.value_range must be [min, max]")
        low, high = (_number(bound, f"{label}.value_range") for bound in bounds)
        if low > high:
            raise SpecContractError(f"{label}.value_range minimum exceeds maximum")
    return value


def _formula(value: Any) -> str:
    """Keep formula validation structural; semantic review belongs to the agents."""
    if not isinstance(value, str) or not value.strip():
        raise SpecContractError("formula must be a non-empty string")
    return value.strip()


def _validate_case_header(case: dict[str, Any], index: int,
                          defaults: dict[str, Any], where: str) -> None:
    if not isinstance(case["name"], str) or not NAME_RE.fullmatch(case["name"]):
        raise SpecContractError(f"{where}.name must be lower_snake_case")
    if not isinstance(case["params"], dict) or list(case["params"]) != list(defaults):
        raise SpecContractError(f"{where}.params names/order must equal default_params")
    if index == 0 and case["params"] != defaults:
        raise SpecContractError("the first P0 case params must equal default_params")


def _member_shapes(meta: dict[str, Any], value: Any, where: str) -> list:
    """The shape of a list is an ordered, non-empty array of member shapes."""
    if not meta.get("is_list", False):
        return [value]
    if not isinstance(value, list) or not value or any(not isinstance(s, list) for s in value):
        raise SpecContractError(f"{where} must be a non-empty array of TensorList member shapes")
    if len({len(s) for s in value}) != 1 or not value[0]:
        raise SpecContractError(f"{where} TensorList members must have one common positive rank")
    return value


def _validate_case_shapes(case: dict[str, Any],
                          tensors: tuple[list[dict[str, Any]], list[dict[str, Any]]],
                          where: str) -> None:
    for field, metadata in zip(("input_shapes", "output_shapes"), tensors):
        shapes, names = case[field], [item["name"] for item in metadata]
        if not isinstance(shapes, dict) or list(shapes) != names:
            raise SpecContractError(f"{where}.{field} names/order must match the contract")
        for meta in metadata:
            name = meta["name"]
            for shape in _member_shapes(meta, shapes[name], f"{where}.{field}.{name}"):
                _shape(shape, f"{where}.{field}.{name}", concrete=True)


def _validate_case_dtypes(case: dict[str, Any], index: int,
                          tensors: tuple[list[dict[str, Any]], list[dict[str, Any]]],
                          where: str) -> None:
    present = CASE_DTYPE_FIELDS & case.keys()
    if present and present != CASE_DTYPE_FIELDS:
        raise SpecContractError(f"{where} must declare input_dtypes and output_dtypes together")
    for field, metadata in zip(("input_dtypes", "output_dtypes"), tensors):
        if field not in case:
            continue
        names = [item["name"] for item in metadata]
        values = case[field]
        if not isinstance(values, dict) or list(values) != names:
            raise SpecContractError(f"{where}.{field} names/order must match the contract")
        if any(dtype not in DTYPE_VOCAB for dtype in values.values()):
            raise SpecContractError(f"{where}.{field} must use canonical dtypes")
        if index == 0 and values != {item["name"]: item["dtype"] for item in metadata}:
            raise SpecContractError("the first P0 case dtypes must equal tensor defaults")


def _validate_case_special_values(case: dict[str, Any], inputs: list[dict[str, Any]], where: str) -> None:
    if "input_special_values" not in case:
        return
    mapping = case["input_special_values"]
    names = {item["name"] for item in inputs}
    if not isinstance(mapping, dict) or not mapping or not set(mapping) <= names:
        raise SpecContractError(f"{where}.input_special_values must map declared input names")
    dtypes = case.get("input_dtypes", {item["name"]: item["dtype"] for item in inputs})
    for name, values in mapping.items():
        if (not isinstance(values, list) or not values
                or any(not isinstance(value, str) or value not in SPECIAL_VALUES for value in values)
                or len(values) != len(set(values))):
            raise SpecContractError(f"{where}.input_special_values.{name} must list unique -inf, +inf or nan")
        if dtypes[name] not in {"float16", "bfloat16", "float32", "float64"}:
            raise SpecContractError(f"{where}.input_special_values.{name} requires a floating input dtype")


def _bind_shape_symbol(expression: int | str, concrete: int,
                       env: dict[str, int], where: str) -> None:
    if not isinstance(expression, str) or not SYMBOL_RE.fullmatch(expression):
        return
    if expression in env and env[expression] != concrete:
        raise SpecContractError(f"{where} has inconsistent {expression}")
    env[expression] = concrete


def _case_environment(case: dict[str, Any], inputs: list[dict[str, Any]],
                      where: str) -> dict[str, int]:
    env = {
        key: value for key, value in case["params"].items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
    for meta in inputs:
        if meta["shape"] == ["..."]:
            continue
        for actual in _member_shapes(meta, case["input_shapes"][meta["name"]], where):
            if len(actual) != len(meta["shape"]):
                raise SpecContractError(f"{where} input {meta['name']} rank mismatch")
            for expression, concrete in zip(meta["shape"], actual):
                _bind_shape_symbol(expression, concrete, env, where)
    return env


def _validate_case_expected_shapes(
        case: dict[str, Any],
        tensors: tuple[list[dict[str, Any]], list[dict[str, Any]]],
        env: dict[str, int], where: str) -> None:
    for field, metadata in zip(("input_shapes", "output_shapes"), tensors):
        for meta in metadata:
            if meta["shape"] == ["..."]:
                continue
            actual = case[field][meta["name"]]
            expected = [
                _eval_dim(dim, env, f"{where}.{meta['name']}")
                for dim in meta["shape"]
            ]
            for member in _member_shapes(meta, actual, where):
                if member != expected:
                    raise SpecContractError(
                        f"{where}.{meta['name']} shape {member} does not equal {expected}")


def _validate_case_ranges(case: dict[str, Any], env: dict[str, int],
                          ranges: dict[str, list[int]], where: str) -> None:
    for symbol, bounds in ranges.items():
        if symbol in env and not bounds[0] <= env[symbol] <= bounds[1]:
            raise SpecContractError(f"{where} {symbol}={env[symbol]} is outside {bounds}")


def _case(case: Any, index: int, tensors: tuple[list[dict[str, Any]], list[dict[str, Any]]],
          defaults: dict[str, Any], ranges: dict[str, list[int]]) -> dict[str, Any]:
    where = f"p0_cases[{index}]"
    if not isinstance(case, dict):
        raise SpecContractError(f"{where} must be an object")
    _keys(case, CASE_FIELDS, CASE_DTYPE_FIELDS | CASE_SPECIAL_FIELDS, where)
    _validate_case_header(case, index, defaults, where)
    _validate_case_shapes(case, tensors, where)
    _validate_case_dtypes(case, index, tensors, where)
    _validate_case_special_values(case, tensors[0], where)
    env = _case_environment(case, tensors[0], where)
    _validate_case_expected_shapes(case, tensors, env, where)
    _validate_case_ranges(case, env, ranges, where)
    return case


def _validate_contract_header(contract: dict[str, Any]) -> None:
    if contract["schema_version"] != 1:
        raise SpecContractError("schema_version must be 1")
    if not isinstance(contract["op_name"], str) or not NAME_RE.fullmatch(contract["op_name"]):
        raise SpecContractError("op_name must be lower_snake_case")


def _invalid_dtypes() -> SpecContractError:
    return SpecContractError(
        "supported_dtypes must be a non-empty unique list of canonical dtypes")


def _supported_dtypes(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise _invalid_dtypes()
    if not value:
        raise _invalid_dtypes()
    if len(set(value)) != len(value):
        raise _invalid_dtypes()
    if any(dtype not in DTYPE_VOCAB for dtype in value):
        raise _invalid_dtypes()
    return value


def _validate_tensor_dtypes(dtypes: list[str], inputs: list[dict[str, Any]],
                            outputs: list[dict[str, Any]],
                            cases: list[dict[str, Any]]) -> None:
    observed = [item["dtype"] for item in inputs + outputs]
    for case in cases:
        for field, metadata in zip(("input_dtypes", "output_dtypes"), (inputs, outputs)):
            observed.extend(case.get(field, {item["name"]: item["dtype"] for item in metadata}).values())
    observed_dtypes = list(dict.fromkeys(observed))
    if dtypes != observed_dtypes:
        raise SpecContractError(
            "supported_dtypes must list tensor and P0 case dtypes in first-appearance order")


def _default_params(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpecContractError(
            "default_params must contain lower_snake_case JSON scalar fields")
    if not all(NAME_RE.fullmatch(key) for key in value):
        raise SpecContractError(
            "default_params must contain lower_snake_case JSON scalar fields")
    if any(isinstance(item, (dict, list)) for item in value.values()):
        raise SpecContractError("default_params must contain lower_snake_case JSON scalar fields")
    return value


def _validate_tolerance(value: Any) -> None:
    if not isinstance(value, dict):
        raise SpecContractError("tolerance must be an object")
    if value == {"policy": "pro_scheme_a"}:
        return
    _keys(value, {"atol", "rtol"}, {"integer_atol"}, "tolerance")
    for key in ("atol", "rtol"):
        _number(value[key], f"tolerance.{key}", non_negative=True)
    if "integer_atol" in value:
        amount = value["integer_atol"]
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise SpecContractError("tolerance.integer_atol must be a non-negative integer")


def _validate_perf_target(value: Any) -> None:
    if value is None:
        return
    target = _number(value, "perf_target")
    if target <= 0:
        raise SpecContractError("perf_target must be null or a positive finite number")


def _valid_range_bounds(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    if len(value) != 2:
        return False
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        return False
    if value[0] <= 0:
        return False
    return value[0] <= value[1]


def _dynamic_ranges(value: Any) -> dict[str, list[int]]:
    if not isinstance(value, dict) or not all(SYMBOL_RE.fullmatch(key) for key in value):
        raise SpecContractError("dynamic_axes_ranges must be a shape-symbol object")
    for symbol, bounds in value.items():
        if not _valid_range_bounds(bounds):
            raise SpecContractError(
                f"dynamic_axes_ranges.{symbol} must be [positive_min, max]")
    return value


def _input_anchors(inputs: list[dict[str, Any]]) -> set[str]:
    anchors: set[str] = set()
    for item in inputs:
        for dim in item["shape"]:
            if isinstance(dim, str) and SYMBOL_RE.fullmatch(dim):
                anchors.add(dim)
    return anchors


def _validate_dynamic_symbols(ranges: dict[str, list[int]], used_symbols: set[str],
                              inputs: list[dict[str, Any]]) -> None:
    if set(ranges) != used_symbols:
        raise SpecContractError(
            "dynamic_axes_ranges names must exactly match shape symbols: "
            + ", ".join(sorted(used_symbols)))
    anchors = _input_anchors(inputs)
    if not used_symbols <= anchors:
        raise SpecContractError(
            "each shape symbol must appear as a standalone dimension in at least one input: "
            + ", ".join(sorted(used_symbols - anchors)))


def _validate_shape_constraints(value: Any) -> None:
    if not isinstance(value, list):
        raise SpecContractError("shape_constraints must be an array of non-empty strings")
    if not all(isinstance(item, str) and item for item in value):
        raise SpecContractError("shape_constraints must be an array of non-empty strings")


def _validate_cases(value: Any,
                    tensors: tuple[list[dict[str, Any]], list[dict[str, Any]]],
                    defaults: dict[str, Any],
                    ranges: dict[str, list[int]]) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise SpecContractError("p0_cases must be a non-empty array")
    cases = [
        _case(case, index, tensors, defaults, ranges)
        for index, case in enumerate(value)
    ]
    if len({case["name"] for case in cases}) != len(cases):
        raise SpecContractError("p0_cases names must be unique")
    return cases


def _validate(contract: dict[str, Any]) -> dict[str, Any]:
    _keys(contract, REQUIRED, OPTIONAL, "machine contract")
    _validate_contract_header(contract)
    dtypes = _supported_dtypes(contract["supported_dtypes"])
    inputs, outputs = _tensors(contract["inputs"], "inputs"), _tensors(contract["outputs"], "outputs")
    if {item["name"] for item in inputs} & {item["name"] for item in outputs}:
        raise SpecContractError("input and output names must be distinct")
    _formula(contract["formula"])
    defaults = _default_params(contract["default_params"])
    _validate_tolerance(contract["tolerance"])
    _validate_perf_target(contract.get("perf_target"))
    ranges = _dynamic_ranges(contract["dynamic_axes_ranges"])
    used_symbols = _shape_symbols(inputs + outputs)
    _validate_dynamic_symbols(ranges, used_symbols, inputs)
    _validate_shape_constraints(contract["shape_constraints"])
    cases = _validate_cases(contract["p0_cases"], (inputs, outputs), defaults, ranges)
    _validate_tensor_dtypes(dtypes, inputs, outputs, cases)
    if contract.get("exit_criteria") is not None:
        # Load by file location so callers using importlib do not need to modify sys.path.
        import importlib.util
        module_spec = importlib.util.spec_from_file_location(
            "pypto_exit_criteria", Path(__file__).with_name("exit_criteria.py"))
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        try:
            module.validate_exit_criteria(contract["exit_criteria"], [case["name"] for case in cases])
        except ValueError as exc:
            raise SpecContractError(f"invalid exit_criteria: {exc}") from exc
    result = dict(contract)
    result["p0_shapes"] = [cases[0]["input_shapes"][item["name"]] for item in inputs]
    return result


def validate_text(text: str) -> list[str]:
    try:
        _validate(_extract(text))
        return []
    except SpecContractError as exc:
        return [str(exc)]


def load_spec_contract_text(text: str) -> dict[str, Any]:
    """Return the validated canonical mapping from SPEC markdown text."""
    return _validate(_extract(text))


def load_spec_contract(path: Path) -> dict[str, Any]:
    """Return the validated mapping used by optional SPEC consumers."""
    try:
        return load_spec_contract_text(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SpecContractError(str(exc)) from exc


def _cli_logger() -> logging.Logger:
    """Return an isolated stdout logger with the same output contract as print."""
    logger = logging.Logger(f"{__name__}.cli", level=logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _emit_cli(logger: logging.Logger, level: int, message: str, *args: Any) -> None:
    """Emit through logging without inheriting global logger suppression."""
    record = logger.makeRecord(
        logger.name, level, __file__, 0, message, args, None,
    )
    logger.handle(record)


def main() -> int:
    logger = _cli_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    args = parser.parse_args()
    try:
        load_spec_contract(args.spec)
    except SpecContractError as exc:
        _emit_cli(logger, logging.ERROR, "FAIL: %s", exc)
        return 1
    _emit_cli(logger, logging.INFO, "PASS: SPEC JSON machine contract is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
