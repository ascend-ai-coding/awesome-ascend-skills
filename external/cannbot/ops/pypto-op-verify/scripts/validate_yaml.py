#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
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

DTYPE_VOCAB = {"float32", "float16", "bfloat16", "int32", "int64", "bool", "int"}
_MODULE_RE = re.compile(r"^module_(\d+)$")
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.USub, ast.UAdd, ast.Constant,
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


def _shape_dim_errors(mid, name, shape) -> list[str]:
    """Rule 5: every shape dim expression must parse using only + - * // and name/int tokens."""
    errors: list[str] = []
    for dim in _shape_dims(shape):
        if not _dim_parses(dim):
            errors.append(f"rule5: module {mid} output '{name}' shape dim '{dim}' not parseable")
    return errors


def _index_module_outputs(modules: list) -> tuple[list[str], dict]:
    """Rules 4, 5, 6 over module outputs; also index output names by module id."""
    errors: list[str] = []
    outputs_by_id: dict[int, set] = {}
    seen_keys: set = set()
    for m in modules:
        mid = m.get("id")
        names = set()
        for o in m.get("outputs", []) or []:
            name = o.get("name")
            names.add(name)
            key = (mid, name)
            if key in seen_keys:  # rule 4
                errors.append(f"rule4: duplicate output key {key}")
            seen_keys.add(key)
            errors.extend(_shape_dim_errors(mid, name, o.get("shape", "")))  # rule 5
            dt = o.get("dtype")  # rule 6
            if dt is not None and dt not in DTYPE_VOCAB:
                errors.append(f"rule6: module {mid} output '{name}' dtype '{dt}' not in {sorted(DTYPE_VOCAB)}")
        outputs_by_id[mid] = names
    return errors, outputs_by_id


def _input_wiring_error(mid, inp, primary, outputs_by_id) -> list[str]:
    """Rules 1 & 2 for a single module input; returns the 0-or-1 element error list."""
    src = inp.get("source")
    name = inp.get("name")
    if src == "primary":
        if name not in primary:  # rule 1
            return [f"rule1: module {mid} input '{name}' source=primary not in primary_inputs"]
        return []
    mt = _MODULE_RE.match(str(src or ""))
    if not mt:
        return [f"rule2: module {mid} input '{name}' has invalid source '{src}'"]
    j = int(mt.group(1))
    if not (isinstance(mid, int) and j < mid):  # rule 2: forward/self ref
        return [f"rule2: module {mid} input '{name}' references module_{j} (must be < {mid})"]
    if name not in outputs_by_id.get(j, set()):
        return [f"rule2: module {mid} input '{name}' not produced by module_{j}"]
    return []


def _check_input_wiring(modules: list, primary: set, outputs_by_id: dict) -> list[str]:
    """Rules 1 & 2: every module input resolves to primary or an earlier module output."""
    errors: list[str] = []
    for m in modules:
        mid = m.get("id")
        for inp in m.get("inputs", []) or []:
            errors.extend(_input_wiring_error(mid, inp, primary, outputs_by_id))
    return errors


def _check_final_outputs(final_outputs: list, outputs_by_id: dict, n: int) -> list[str]:
    """Rule 3: every final output references an in-range module that produces the named value."""
    errors: list[str] = []
    for fo in final_outputs:
        name = fo.get("name")
        src = fo.get("source")
        mt = _MODULE_RE.match(str(src or ""))
        if not mt:
            errors.append(f"rule3: final_output '{name}' has invalid source '{src}'")
            continue
        j = int(mt.group(1))
        if j > n:
            errors.append(f"rule3: final_output '{name}' references module_{j} > N={n}")
        elif name not in outputs_by_id.get(j, set()):
            errors.append(f"rule3: final_output '{name}' not produced by module_{j}")
    return errors


def validate(spec: dict) -> list[str]:
    """Return a list of violation strings; empty list means valid."""
    errors: list[str] = []
    primary = {p.get("name") for p in spec.get("primary_inputs", []) if isinstance(p, dict)}
    modules = spec.get("modules", []) or []
    n = len(modules)

    output_errors, outputs_by_id = _index_module_outputs(modules)
    errors.extend(output_errors)
    errors.extend(_check_input_wiring(modules, primary, outputs_by_id))
    errors.extend(_check_final_outputs(spec.get("final_outputs", []) or [], outputs_by_id, n))
    return errors


def _load(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyYAML not available; cannot read YAML file.") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


_VALID = {
    "primary_inputs": [{"name": "x", "shape": "[B, T]", "dtype": "float32"}],
    "modules": [
        {"id": 1, "inputs": [{"name": "x", "source": "primary"}],
         "outputs": [{"name": "h1", "shape": "[B, T]", "dtype": "float32"}]},
        {"id": 2, "inputs": [{"name": "h1", "source": "module_1"}],
         "outputs": [{"name": "y", "shape": "[B, T]", "dtype": "float32"}]},
    ],
    "final_outputs": [{"name": "y", "source": "module_2"}],
}


def _self_test() -> int:
    import copy
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
    c4["modules"][1]["outputs"].append({"name": "y", "shape": "[B, T]", "dtype": "float32"})
    c4["modules"][1]["id"] = 2  # two (2,'y') keys
    c4["modules"][1]["outputs"][0]["name"] = "y"
    cases.append(("rule4_dup_key", c4, "rule4"))
    c5 = copy.deepcopy(_VALID)
    c5["modules"][0]["outputs"][0]["shape"] = "[B ** 2, T]"
    cases.append(("rule5_bad_shape", c5, "rule5"))
    c6 = copy.deepcopy(_VALID)
    c6["modules"][0]["outputs"][0]["dtype"] = "float8"
    cases.append(("rule6_bad_dtype", c6, "rule6"))

    bad = 0
    for name, spec, want in cases:
        errs = validate(spec)
        if want == "":
            ok = not errs
        else:
            ok = any(e.startswith(want) for e in errs)
        bad += 0 if ok else 1
        _LOGGER.info("%s %s: %s", "PASS" if ok else "FAIL", name,
                     errs if errs else "no errors")
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
        _LOGGER.info(json.dumps(
            {"status": "PASS" if not errors else "FAIL", "violations": errors}, indent=2))
    elif errors:
        _LOGGER.info("FAIL:")
        for e in errors:
            _LOGGER.info("  - %s", e)
    else:
        _LOGGER.info("PASS")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
