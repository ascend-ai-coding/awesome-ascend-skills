#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Validate custom/<op>/eval/module_interfaces.yaml wiring (Step A.5.1).

Codifies the six validity rules previously checked by hand (Step A.5.1):

  1. Every inputs[*].source == "primary" name exists in primary_inputs.
  2. Every inputs[*].source == "module_j" has j < current module id, and the
     referenced name exists in module_j.outputs (no forward/self reference).
  3. Every final_outputs[*].source == "module_j" has j <= N, and the
     referenced name exists in module_j.outputs.
  4. No two outputs share the same (module_id, name) key.
  5. Shape expressions parse using only + - * // and name/int tokens.
  6. dtype strings are from the allowed vocabulary.

On FAIL a rejection block is appended to MEMORY.md and the run stops so the
upstream design can be revised.

Assumed schema (the skeleton emitter and this validator must agree):

    primary_inputs:
      - {name: x, shape: "[B, T]", dtype: float32}
    modules:
      - id: 1
        inputs:  [{name: x, source: primary}]
        outputs: [{name: h1, shape: "[B, T]", dtype: float32}]
      - id: 2
        inputs:  [{name: h1, source: module_1}]
        outputs: [{name: y, shape: "[B, T]", dtype: float32}]
    final_outputs:
      - {name: y, source: module_2}

Usage::

    python validate_yaml.py custom/<op>/eval/module_interfaces.yaml [--json]
    python validate_yaml.py --self-test

Exit code: 0 if valid, 1 if any rule fails (or on load/parse error).
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import sys
from pathlib import Path

# Emit on stdout with a bare (message-only) format so JSON / summary output
# stays machine-parseable for the caller (which reads stdout).
_LOGGER = logging.getLogger("validate_yaml")

DTYPE_VOCAB = {
    "float32",
    "float16",
    "bfloat16",
    "int32",
    "int64",
    "bool",
    "int",
    "int8",
}
_MODULE_RE = re.compile(r"^module_(\d+)$")
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Name,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.FloorDiv,
    ast.USub,
    ast.UAdd,
    ast.Constant,
)


def _shape_dims(shape) -> list[str]:
    """Normalize a shape field (string '[B, T]' or list) into dim expressions."""
    if isinstance(shape, (list, tuple)):
        return [str(d) for d in shape]
    s = str(shape).strip().strip("[]")
    return [d.strip() for d in s.split(",") if d.strip()]


def _dim_parses(expr: str) -> bool:
    """True if a dim expression uses only + - * // and name/int tokens."""
    if expr.lstrip("-").isdigit():
        return True
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return False
        if isinstance(node, ast.Constant) and not isinstance(node.value, int):
            return False
    return True


def _validate_shape_and_dtype(entry, location, errors):
    for dimension in _shape_dims(entry.get("shape", "")):
        if not _dim_parses(dimension):
            errors.append(f"rule5: {location} shape dim '{dimension}' not parseable")
    dtype = entry.get("dtype")
    if dtype is not None and dtype not in DTYPE_VOCAB:
        errors.append(f"rule6: {location} dtype '{dtype}' not in {sorted(DTYPE_VOCAB)}")


def _validate_primary_inputs(primary_inputs, errors):
    for primary_input in primary_inputs:
        if not isinstance(primary_input, dict):
            continue
        name = primary_input.get("name")
        _validate_shape_and_dtype(primary_input, f"primary input '{name}'", errors)


def _index_outputs(modules, errors):
    outputs_by_id: dict[int, set] = {}
    seen_keys: set = set()
    for module in modules:
        module_id = module.get("id")
        names = set()
        for output in module.get("outputs", []) or []:
            name = output.get("name")
            names.add(name)
            key = (module_id, name)
            if key in seen_keys:
                errors.append(f"rule4: duplicate output key {key}")
            seen_keys.add(key)
            _validate_shape_and_dtype(
                output, f"module {module_id} output '{name}'", errors
            )
        outputs_by_id[module_id] = names
    return outputs_by_id


def _validate_module_input(module_id, module_input, primary, outputs_by_id, errors):
    source = module_input.get("source")
    name = module_input.get("name")
    if source == "primary":
        if name not in primary:
            errors.append(
                f"rule1: module {module_id} input '{name}' "
                "source=primary not in primary_inputs"
            )
        return
    source_match = _MODULE_RE.match(str(source or ""))
    if not source_match:
        errors.append(
            f"rule2: module {module_id} input '{name}' has invalid source '{source}'"
        )
        return
    source_id = int(source_match.group(1))
    if not (isinstance(module_id, int) and source_id < module_id):
        errors.append(
            f"rule2: module {module_id} input '{name}' references "
            f"module_{source_id} (must be < {module_id})"
        )
    elif name not in outputs_by_id.get(source_id, set()):
        errors.append(
            f"rule2: module {module_id} input '{name}' not produced by module_{source_id}"
        )


def _validate_module_inputs(modules, primary, outputs_by_id, errors):
    for module in modules:
        module_id = module.get("id")
        for module_input in module.get("inputs", []) or []:
            _validate_module_input(
                module_id,
                module_input,
                primary,
                outputs_by_id,
                errors,
            )


def _validate_final_outputs(final_outputs, module_count, outputs_by_id, errors):
    for final_output in final_outputs:
        name = final_output.get("name")
        source = final_output.get("source")
        source_match = _MODULE_RE.match(str(source or ""))
        if not source_match:
            errors.append(f"rule3: final_output '{name}' has invalid source '{source}'")
            continue
        source_id = int(source_match.group(1))
        if source_id > module_count:
            errors.append(
                f"rule3: final_output '{name}' references module_{source_id} > N={module_count}"
            )
        elif name not in outputs_by_id.get(source_id, set()):
            errors.append(
                f"rule3: final_output '{name}' not produced by module_{source_id}"
            )


def validate(spec: dict) -> list[str]:
    """Return a list of violation strings; empty list means valid."""
    if not isinstance(spec, dict):
        return ["interface must be a mapping"]
    for field in ("primary_inputs", "modules", "final_outputs"):
        rows = spec.get(field)
        if (
            not isinstance(rows, list)
            or not rows
            or any(not isinstance(row, dict) for row in rows)
        ):
            return [f"{field} must be a non-empty list of mappings"]
    for i, module in enumerate(spec["modules"], 1):
        if type(module.get("id")) is not int or module["id"] != i:
            return ["module ids must be consecutive integers starting at 1"]
        for field in ("inputs", "outputs"):
            rows = module.get(field)
            if (
                not isinstance(rows, list)
                or not rows
                or any(not isinstance(row, dict) for row in rows)
            ):
                return [f"module {i} {field} must be a non-empty list of mappings"]
    errors: list[str] = []
    primary_inputs = spec.get("primary_inputs", [])
    primary = {item.get("name") for item in primary_inputs if isinstance(item, dict)}
    _validate_primary_inputs(primary_inputs, errors)
    modules = spec.get("modules", []) or []
    outputs_by_id = _index_outputs(modules, errors)
    _validate_module_inputs(modules, primary, outputs_by_id, errors)
    _validate_final_outputs(
        spec.get("final_outputs", []) or [], len(modules), outputs_by_id, errors
    )

    return errors


def _load(path: Path) -> dict:
    try:
        import yaml  # type: ignore  # noqa: PLC0415 -- optional CLI dependency
    except ImportError as exc:
        raise RuntimeError("PyYAML not available; cannot read YAML file.") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


_VALID = {
    "primary_inputs": [{"name": "x", "shape": "[B, T]", "dtype": "float32"}],
    "modules": [
        {
            "id": 1,
            "inputs": [{"name": "x", "source": "primary"}],
            "outputs": [{"name": "h1", "shape": "[B, T]", "dtype": "float32"}],
        },
        {
            "id": 2,
            "inputs": [{"name": "h1", "source": "module_1"}],
            "outputs": [{"name": "y", "shape": "[B, T]", "dtype": "float32"}],
        },
    ],
    "final_outputs": [{"name": "y", "source": "module_2"}],
}


def _self_test_cases() -> list[tuple[str, dict, str]]:
    """The (name, spec, expected rule) triples the self-test validates against."""
    import copy  # noqa: PLC0415 -- used only by the self-test path

    cases: list[tuple[str, dict, str]] = []
    cases.append(("valid", _VALID, ""))  # expect no error

    c1 = copy.deepcopy(_VALID)
    c1["modules"][0]["inputs"][0]["name"] = "zzz"
    cases.append(("rule1_unknown_primary", c1, "rule1"))
    c2 = copy.deepcopy(_VALID)
    c2["modules"][0]["inputs"][0] = {"name": "y", "source": "module_2"}
    cases.append(("rule2_forward_ref", c2, "rule2"))
    c3 = copy.deepcopy(_VALID)
    c3["final_outputs"][0]["source"] = "module_9"
    cases.append(("rule3_out_of_range", c3, "rule3"))
    c4 = copy.deepcopy(_VALID)
    c4["modules"][1]["outputs"].append(
        {"name": "y", "shape": "[B, T]", "dtype": "float32"}
    )
    c4["modules"][1]["id"] = 2  # two (2,'y') keys
    c4["modules"][1]["outputs"][0]["name"] = "y"
    cases.append(("rule4_dup_key", c4, "rule4"))
    c5 = copy.deepcopy(_VALID)
    c5["modules"][0]["outputs"][0]["shape"] = "[B ** 2, T]"
    cases.append(("rule5_bad_shape", c5, "rule5"))
    c6 = copy.deepcopy(_VALID)
    c6["modules"][0]["outputs"][0]["dtype"] = "float8"
    cases.append(("rule6_bad_dtype", c6, "rule6"))
    c7 = copy.deepcopy(_VALID)
    c7["primary_inputs"][0]["shape"] = "[B ** 2, T]"
    cases.append(("rule5_bad_primary_shape", c7, "rule5"))
    c8 = copy.deepcopy(_VALID)
    c8["primary_inputs"][0]["dtype"] = "float8"
    cases.append(("rule6_bad_primary_dtype", c8, "rule6"))
    c9 = {
        "primary_inputs": [{"name": "x", "shape": "[B, T]", "dtype": "int8"}],
        "modules": [
            {
                "id": 1,
                "inputs": [{"name": "x", "source": "primary"}],
                "outputs": [{"name": "y", "shape": "[B, T]", "dtype": "int8"}],
            },
        ],
        "final_outputs": [{"name": "y", "source": "module_1"}],
    }
    cases.append(("rule6_int8_single_module", c9, ""))  # expect no error
    return cases


def _self_test() -> int:
    bad = 0
    for name, spec, want in _self_test_cases():
        errs = validate(spec)
        if not want:
            ok = not errs
        else:
            ok = any(e.startswith(want) for e in errs)
        bad += 0 if ok else 1
        _LOGGER.error(
            "%s %s: %s", "PASS" if ok else "FAIL", name, errs if errs else "no errors"
        )
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml", nargs="?", type=Path)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    if args.self_test:
        return _self_test()
    if not args.yaml:
        ap.error("yaml path required (or --self-test)")
    spec = _load(args.yaml)
    errors = validate(spec or {})
    if args.json:
        _LOGGER.error(
            json.dumps(
                {"status": "PASS" if not errors else "FAIL", "violations": errors},
                indent=2,
            )
        )
    elif errors:
        _LOGGER.error("FAIL:")
        for e in errors:
            _LOGGER.info("  - %s", e)
    else:
        _LOGGER.info("PASS")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
