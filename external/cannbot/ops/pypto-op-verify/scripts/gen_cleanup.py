#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Module-integration cleanup (L1) — mechanical, no implementation pass.

L1 cleanup produces two mechanical artifacts previously hand-written:
  1. `<op>_impl.py` — the final cumulative `<op>_module<suffix_N>_impl.py` with
     the wrapper symbol renamed `<op>_module<suffix_N>_wrapper` → `<op>_wrapper`
     and staging comments / stubs stripped. Same kernel logic, same bodies.
  2. `README.md` — reader-facing usage doc filled from SPEC.md front matter.

Run this instead of a hand cleanup pass; the integrated impl is still
verified downstream by the final E2E gate.

Usage::

    python gen_cleanup.py --op relu \
        --final-impl custom/relu/modules/relu_module123_impl.py \
        --spec custom/relu/SPEC.md --out-dir custom/relu

    python gen_cleanup.py --self-test
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

_LOGGER = logging.getLogger("gen_cleanup")

_STUB_RE = re.compile(r"^\s*#\s*STUB\b.*$\n?", re.MULTILINE)


def integrate_impl(op: str, final_impl_src: str, final_suffix: str) -> str:
    """Rename the final module wrapper to `<op>_wrapper` and strip stub comments."""
    src = final_impl_src.replace(f"{op}_module{final_suffix}_wrapper", f"{op}_wrapper")
    src = _STUB_RE.sub("", src)
    header = (f"# Integrated production kernel for {op} "
              f"(consolidated from modules/{op}_module{final_suffix}_impl.py).\n"
              f"# Exports {op}_wrapper. Same kernel logic as the final staged module.\n")
    return header + src


def build_readme(op: str, meta: dict, params: list[str]) -> str:
    dtypes = meta.get("supported_dtypes") or ["float32"]
    p0 = meta.get("p0_shapes") or []
    args = ", ".join(params or ["x"])
    shape_str = ", ".join(str(s) for s in p0) if p0 else "see SPEC.md"
    lines = [
        f"# {op}",
        "",
        "PyPTO kernel. Auto-generated cleanup doc (see SPEC.md for the full spec).",
        "",
        "## Signature",
        "",
        "```python",
        f"from {op}_impl import {op}_wrapper",
        f"out = {op}_wrapper({args})",
        "```",
        "",
        "## Supported dtypes / shapes",
        "",
        f"- dtypes: {', '.join(str(d) for d in dtypes)}",
        f"- P0 shapes: {shape_str}",
        "",
        "## Required environment",
        "",
        "- `TILE_FWK_DEVICE_ID` — target NPU device id",
        "- `LD_LIBRARY_PATH`, `PTO_TILE_LIB_CODE_PATH` — PyPTO runtime libraries",
        "",
        "## Validation",
        "",
        f"Run `python custom/{op}/test_{op}.py` for the end-to-end precision check.",
        "",
    ]
    return "\n".join(lines)


def _parse_front_matter(content: str) -> dict:
    rows = content.splitlines()
    if not rows or rows[0].strip() != "---":
        return {}
    end = next((i for i in range(1, len(rows)) if rows[i].strip() == "---"), None)
    if end is None:
        return {}
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyYAML not available; cannot read SPEC front matter.") from exc
    return yaml.safe_load("\n".join(rows[1:end])) or {}


def _golden_params(golden: Path, op: str) -> list[str]:
    import ast
    try:
        tree = ast.parse(golden.read_text(encoding="utf-8"))
    except OSError:
        return []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == f"{op}_golden":
            return [a.arg for a in node.args.args if a.arg != "self"]
    return []


def _self_test() -> int:
    final = ("def relu_module1_wrapper(x):\n"
             "    # STUB: until M2 verified; golden-fed tensor\n"
             "    return _kernel(x)\n")
    out = integrate_impl("relu", final, "1")
    ok_impl = ("relu_wrapper(x)" in out and "relu_module1_wrapper" not in out
               and "STUB" not in out)
    readme = build_readme("relu", {"supported_dtypes": ["float16"],
                                   "p0_shapes": [[1024, 128]]}, ["x"])
    ok_readme = ("# relu" in readme and "float16" in readme
                 and "relu_wrapper(x)" in readme and "test_relu.py" in readme)
    passed = ok_impl and ok_readme
    _LOGGER.info("%s self-test: impl=%s readme=%s", "PASS" if passed else "FAIL",
                 ok_impl, ok_readme)
    return 0 if passed else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op")
    ap.add_argument("--final-impl", type=Path, help="modules/<op>_module<suffix_N>_impl.py")
    ap.add_argument("--final-suffix", help="cumulative suffix of the final module, e.g. 123")
    ap.add_argument("--spec", type=Path)
    ap.add_argument("--golden", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    if args.self_test:
        return _self_test()
    if not (args.op and args.final_impl and args.final_suffix):
        ap.error("--op, --final-impl, --final-suffix required (or --self-test)")
    meta = _parse_front_matter(args.spec.read_text(encoding="utf-8")) if args.spec else {}
    params = _golden_params(args.golden, args.op) if args.golden else []
    impl_out = integrate_impl(args.op, args.final_impl.read_text(encoding="utf-8"),
                              args.final_suffix)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.op}_impl.py").write_text(impl_out, encoding="utf-8")
    (args.out_dir / "README.md").write_text(build_readme(args.op, meta, params), encoding="utf-8")
    _LOGGER.info("wrote %s_impl.py + README.md to %s", args.op, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
