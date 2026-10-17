#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Generate a module_interfaces.yaml skeleton for completion by hand.

Codifies the deterministic parts of the YAML previously transcribed by
hand: schema scaffolding, primary_inputs
names (from the golden signature), and composition_verification (atol / rtol /
shapes from SPEC.md front matter). The JUDGEMENT part — the modules[] boundaries
and final_outputs wiring — is left as TODO stubs to fill from
DESIGN.md §0.5 dataflow breakpoints.

After filling modules[], validate the result with the sibling validate_yaml.py.

Usage::

    python gen_module_interfaces.py custom/<op>/<op>_golden.py \
        --spec custom/<op>/SPEC.md --op <op> > custom/<op>/eval/module_interfaces.yaml

    python gen_module_interfaces.py --self-test
"""

from __future__ import annotations

import argparse
import ast
import logging
import sys
from pathlib import Path

_LOGGER = logging.getLogger("gen_module_interfaces")

DEFAULT_DTYPE = "float32"


def _golden_param_names(golden_src: str) -> list[str]:
    """Return the argument names of the *_golden function (excluding self)."""
    tree = ast.parse(golden_src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.endswith("_golden"):
            return [a.arg for a in node.args.args if a.arg != "self"]
    return []


def _parse_front_matter(content: str) -> dict:
    """Extract the YAML front matter (between the first two --- fences)."""
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return {}
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyYAML not available; cannot read SPEC front matter.") from exc
    return yaml.safe_load("\n".join(lines[1:end])) or {}


def _shape_for(name: str, p0_shapes: list) -> str:
    """Best-effort shape for a primary input from SPEC p0_shapes (dict form maps
    by name; otherwise emit a TODO)."""
    for item in p0_shapes or []:
        if isinstance(item, dict) and name in item:
            return str(list(item[name]))
    return "TODO  # fill dims (concrete ints or symbolic names from primary_inputs)"


def build_skeleton(op: str, params: list[str], meta: dict) -> str:
    dtypes = meta.get("supported_dtypes") or [DEFAULT_DTYPE]
    default_dtype = dtypes[0] if isinstance(dtypes, list) and dtypes else DEFAULT_DTYPE
    p0_shapes = meta.get("p0_shapes", [])
    atol = meta.get("atol", 1e-3)
    rtol = meta.get("rtol", 1e-3)

    lines = [
        "schema_version: 1",
        f"op: {op}",
        "",
        "# primary_inputs mirror the <op>_golden signature (names auto-filled).",
        "primary_inputs:",
    ]
    for name in params:
        lines.append(f"  - name: {name}")
        lines.append(f"    shape: {_shape_for(name, p0_shapes)}")
        lines.append(f"    dtype: {default_dtype}")
    if not params:
        lines.append("  []  # TODO: golden signature had no params — check the golden")

    lines += [
        "",
        "# TODO (judgement): split into modules per DESIGN.md §0.5",
        "# breakpoints. source is 'primary' or 'module_<j>' with j < current id.",
        "modules:",
        "  - id: 1",
        "    name: TODO",
        "    description: TODO",
        "    inputs:",
        "      - {name: TODO, source: primary}",
        "    outputs:",
        f"      - {{name: TODO, shape: TODO, dtype: {default_dtype}}}",
        "",
        "# TODO: one entry per golden return value, keyed to producer.",
        "final_outputs:",
        "  - {name: TODO, source: module_1}",
        "",
        "composition_verification:",
        f"  atol: {atol}",
        f"  rtol: {rtol}",
        "  seeds: [0, 1, 2]",
        "  shapes:",
    ]
    if p0_shapes:
        for item in p0_shapes:
            lines.append(f"    - {item}")
    else:
        lines.append("    - TODO  # representative shapes from SPEC.md p0_shapes")
    return "\n".join(lines) + "\n"


def _self_test() -> int:
    golden = "import torch\ndef relu_golden(x, scale):\n    return torch.relu(x) * scale\n"
    spec = (
        "---\n"
        "supported_dtypes: [float16, float32]\n"
        "p0_shapes: [{x: [1, 64], scale: [1]}]\n"
        "atol: 0.001\nrtol: 0.001\n"
        "---\nbody\n"
    )
    params = _golden_param_names(golden)
    meta = _parse_front_matter(spec)
    out = build_skeleton("relu", params, meta)
    ok = (params == ["x", "scale"]
          and "op: relu" in out and "name: x" in out and "name: scale" in out
          and "dtype: float16" in out and "atol: 0.001" in out and "modules:" in out)
    _LOGGER.info("%s self-test: params=%s", "PASS" if ok else "FAIL", params)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("golden", nargs="?", type=Path)
    ap.add_argument("--spec", type=Path)
    ap.add_argument("--op")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    if args.self_test:
        return _self_test()
    if not args.golden:
        ap.error("golden path required (or --self-test)")
    params = _golden_param_names(args.golden.read_text(encoding="utf-8"))
    meta = _parse_front_matter(args.spec.read_text(encoding="utf-8")) if args.spec else {}
    op = args.op or args.golden.stem.replace("_golden", "")
    _LOGGER.info(build_skeleton(op, params, meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
