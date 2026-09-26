#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2024-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Validate a snapshot_manifest.yaml against its schema.

Validates against the schema documented in intermediate-snapshot-automation.md.
Produces a typed dict suitable for snapshot_generator.py and snapshot_bisect.py
to consume.

Usage::

    from snapshot_manifest_schema import load_manifest
    manifest = load_manifest("custom/<op>/_debug/snapshot_manifest.yaml")

The manifest is written by @pypto-op-debugger on first snapshot-driven bisection.
It is op-specific but its shape is constrained so the generator and bisect runner
bind to a single contract.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError as e:
    raise ImportError(
        "snapshot_manifest_schema requires PyYAML. "
        "Install via `pip install pyyaml --break-system-packages` on the NPU host."
    ) from e


# -----------------------------------------------------------------------------
# Allowed vocabulary
# -----------------------------------------------------------------------------

PROBE_POINTS = {"before_nt_loop", "inside_nt_loop", "after_nt_loop"}
SUPPORTED_MODES = {"sim", "npu"}
SUPPORTED_DTYPES = {"float32", "float16", "bfloat16", "int32", "int64", "bool"}


# -----------------------------------------------------------------------------
# Errors
# -----------------------------------------------------------------------------

class SnapshotManifestError(ValueError):
    """Raised when a snapshot manifest fails schema validation."""


# -----------------------------------------------------------------------------
# Validators
# -----------------------------------------------------------------------------

def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SnapshotManifestError(msg)


def _validate_intermediate(idx: int, entry: Dict[str, Any], op: str) -> Dict[str, Any]:
    _require(isinstance(entry, dict), f"intermediates[{idx}] must be a mapping")
    _require("name" in entry, f"intermediates[{idx}] missing 'name'")
    _require("shape" in entry, f"intermediates[{idx}] missing 'shape'")
    _require("probe_point" in entry, f"intermediates[{idx}] missing 'probe_point'")

    name = entry["name"]
    _require(isinstance(name, str) and name.isidentifier(),
             f"intermediates[{idx}].name must be a valid Python identifier, got {name!r}")
    _require(not name.startswith("inspection_"),
             f"intermediates[{idx}].name must NOT start with 'inspection_' "
             f"(the generator adds that prefix). Got {name!r}.")

    shape = entry["shape"]
    _require(isinstance(shape, list) and all(isinstance(d, (str, int)) for d in shape),
             f"intermediates[{idx}].shape must be a list of int|str, got {shape!r}")

    pp = entry["probe_point"]
    _require(pp in PROBE_POINTS,
             f"intermediates[{idx}].probe_point must be one of {sorted(PROBE_POINTS)}, got {pp!r}")

    dtype = entry.get("dtype", "float32")
    _require(dtype in SUPPORTED_DTYPES,
             f"intermediates[{idx}].dtype must be one of {sorted(SUPPORTED_DTYPES)}, got {dtype!r}")

    # Optional golden-side expression for the probe point. If absent, the
    # generator will stub the golden companion and require @pypto-op-debugger to fill it in
    # manually — emit a warning later.
    golden_expression = entry.get("golden_expression")
    if golden_expression is not None:
        _require(isinstance(golden_expression, str),
                 f"intermediates[{idx}].golden_expression must be a string (torch expression)")

    return {
        "name": name,
        "shape": shape,
        "probe_point": pp,
        "dtype": dtype,
        "golden_expression": golden_expression,
    }


def load_manifest(path: str | Path) -> Dict[str, Any]:
    """
    Read + validate a snapshot manifest. Returns a normalized dict:

      {
        "op":     str,
        "module": str,          # e.g. "M2"
        "case":   str,          # case id from adversarial_suite.json
        "modes":  list[str],    # subset of {"sim","npu"}; default ["sim","npu"]
        "intermediates": list[dict],   # each with name/shape/probe_point/dtype/golden_expression
        "atol":   float,        # default 1e-3
        "rtol":   float,        # default 1e-3
        "notes":  str,          # optional human description
      }

    Raises SnapshotManifestError on any schema violation.
    """
    path = Path(path)
    _require(path.exists(), f"snapshot manifest not found at {path}")
    with path.open("r") as f:
        raw = yaml.safe_load(f)
    _require(isinstance(raw, dict), f"snapshot manifest root must be a mapping, got {type(raw).__name__}")

    _require("op" in raw, "missing 'op'")
    _require("module" in raw, "missing 'module'")
    _require("case" in raw, "missing 'case'")
    _require("intermediates" in raw, "missing 'intermediates'")

    op = raw["op"]
    _require(isinstance(op, str) and op.isidentifier(),
             f"'op' must be a valid Python identifier, got {op!r}")

    module = raw["module"]
    _require(isinstance(module, str) and module.startswith("M") and module[1:].isdigit(),
             f"'module' must match 'M<digits>' (e.g. 'M2'), got {module!r}")

    case = raw["case"]
    _require(isinstance(case, str), f"'case' must be a string, got {case!r}")

    modes = raw.get("modes", ["sim", "npu"])
    _require(isinstance(modes, list) and all(m in SUPPORTED_MODES for m in modes),
             f"'modes' must be a subset of {sorted(SUPPORTED_MODES)}, got {modes!r}")

    intermediates_raw = raw["intermediates"]
    _require(isinstance(intermediates_raw, list) and len(intermediates_raw) > 0,
             "'intermediates' must be a non-empty list")
    intermediates = [_validate_intermediate(i, e, op) for i, e in enumerate(intermediates_raw)]
    names = [e["name"] for e in intermediates]
    _require(len(names) == len(set(names)),
             f"intermediates have duplicate names: {[n for n in names if names.count(n) > 1]}")

    return {
        "op": op,
        "module": module,
        "case": case,
        "modes": modes,
        "intermediates": intermediates,
        "atol": float(raw.get("atol", 1e-3)),
        "rtol": float(raw.get("rtol", 1e-3)),
        "notes": raw.get("notes", ""),
    }


# -----------------------------------------------------------------------------
# CLI entry point — validate a manifest standalone
# -----------------------------------------------------------------------------

def _main() -> int:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Validate a snapshot manifest YAML.")
    ap.add_argument("manifest", help="path to snapshot_manifest.yaml")
    ap.add_argument("--print", action="store_true", help="print normalized manifest as JSON")
    args = ap.parse_args()
    try:
        m = load_manifest(args.manifest)
    except SnapshotManifestError as err:
        logging.error("[INVALID] %s", err)
        return 1
    logging.info(
        "[OK] manifest for op=%s module=%s case=%s intermediates=%d modes=%s",
        m["op"], m["module"], m["case"], len(m["intermediates"]), m["modes"],
    )
    if args.print:
        logging.info(json.dumps(m, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
