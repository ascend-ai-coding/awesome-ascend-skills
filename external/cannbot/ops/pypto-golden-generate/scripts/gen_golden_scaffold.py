#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2024-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Deterministic Stage-2 golden scaffold generator (CPU-only).

Reads a structured SPEC.md (YAML front matter + §5 input/output tables) and the
golden template, then emits `<op>_golden.py` with the *deterministic* parts
filled in:

  - file/function names (`{op}` placeholders)
  - `def <op>_golden(<tensor args...>, <scalar kwargs from default_params>)`
  - `_make_inputs(device)` constructing every tensor arg at its P0 concrete shape
  - `_validate()` harness wired to `_make_inputs` (run + finite check)

The op-specific math body, value/structure-constrained inputs, and op-specific
property checks stay as `# TODO:` markers for the mathematician (LLM) to fill.

Pure standard library (re / ast / json). No torch, no NPU — runs anywhere.
"""
from __future__ import annotations

import argparse
import ast
import logging
import os
import re
import sys
from typing import Any

_LOGGER = logging.getLogger("gen_golden_scaffold")


# ---------------------------------------------------------------------------
# Front matter parsing (minimal YAML subset: scalars, [lists], {dicts})
# ---------------------------------------------------------------------------

def _coerce(val: str) -> Any:
    import json
    val = val.strip().strip("`")
    if not val:
        return None
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        pass
    # Quote bareword dict keys so `{eps: 1e-5}` and JSON `{"k": false}` both load.
    fixed = re.sub(r'([{,]\s*)([A-Za-z_]\w*)\s*:', r'\1"\2":', val)
    for loader in (lambda s: ast.literal_eval(s), json.loads):
        try:
            return loader(fixed)
        except (ValueError, SyntaxError, TypeError):
            continue
    return val.strip('"\'')


def parse_front_matter(text: str) -> dict:
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not m:
        raise ValueError("SPEC.md has no YAML front matter (--- block).")
    fm: dict = {}
    for line in m.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, _, rest = line.partition(":")
        if not _:
            continue
        fm[key.strip()] = _coerce(rest)
    return fm


# ---------------------------------------------------------------------------
# Markdown table parsing for §5 input / nn.Parameter / output specs
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "bfloat16": "torch.bfloat16", "bf16": "torch.bfloat16",
    "float16": "torch.float16", "fp16": "torch.float16", "half": "torch.float16",
    "float32": "torch.float32", "fp32": "torch.float32", "float": "torch.float32",
    "float64": "torch.float64", "fp64": "torch.float64",
    "int8": "torch.int8", "uint8": "torch.uint8",
    "int16": "torch.int16", "int32": "torch.int32", "int64": "torch.int64",
    "bool": "torch.bool",
}
_INT_DTYPES = {"torch.int8", "torch.uint8", "torch.int16", "torch.int32",
               "torch.int64", "torch.bool"}


def _norm_name(cell: str) -> str:
    # "`x0`" -> "x0";  "`W` (weight)" -> "W";  "x (x0)" -> "x"
    cell = cell.replace("`", "").strip()
    return re.split(r"[\s(]", cell, 1)[0]


def _norm_dtype(cell: str) -> str:
    key = cell.replace("`", "").strip().lower().split()[0] if cell.strip() else ""
    key = key.replace("torch.", "")           # torch.float32 -> float32
    return _DTYPE_MAP.get(key, "torch.float32")


def _split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


# header cell -> logical column. Tolerates 变量 / 参数名, leading `#` index col,
# and dtype/torch.dtype variants.
_NAME_HEADERS = ("变量", "参数名", "参数", "name")
_SHAPE_HEADERS = ("shape",)
_DTYPE_HEADERS = ("dtype",)


def _header_cols(header_cells: list[str]) -> dict | None:
    cols: dict = {}
    for idx, h in enumerate(header_cells):
        hl = h.strip().lower()
        if any(k in hl for k in _NAME_HEADERS) and "name" not in cols:
            cols["name"] = idx
        elif any(k == hl for k in _SHAPE_HEADERS) and "shape" not in cols:
            cols["shape"] = idx
        elif any(k in hl for k in _DTYPE_HEADERS) and "dtype" not in cols:
            cols["dtype"] = idx
    return cols if {"name", "shape", "dtype"} <= cols.keys() else None


_TABLE_WINDOW = 20  # lines to scan after a heading for its Shape table


def _collect_table_rows(lines: list[str], start: int, cols: dict) -> list[dict]:
    """Collect name/shape/dtype rows from a markdown table body starting at ``start``."""
    rows: list[dict] = []
    k = start
    while k < len(lines) and lines[k].lstrip().startswith("|"):
        cells = _split_row(lines[k])
        if len(cells) > max(cols.values()):
            rows.append({"name": _norm_name(cells[cols["name"]]),
                         "shape": cells[cols["shape"]].replace("`", "").strip(),
                         "dtype": _norm_dtype(cells[cols["dtype"]])})
        k += 1
    return rows


def _parse_table_after(text: str, header_regex: str) -> list[dict]:
    """Return name/shape/dtype rows of the first NON-EMPTY Shape table that
    follows a heading matching header_regex.

    For each heading match we scan a bounded window for a markdown table whose
    header carries a Shape column; column positions are resolved from that
    header row (so leading `#` index columns and 变量/参数名 naming both work).
    Iterating over every heading match — and skipping matches whose window has
    no table (e.g. a prose mention) — makes this robust to keyword collisions."""
    lines = text.splitlines()
    # A heading must be a real heading/label line — NOT a markdown table row.
    # (e.g. an input row cell mentioning "nn.Parameter" must not be treated as
    # a parameter-table heading, which would then capture the 输出规格 table.)
    matches = [i for i, l in enumerate(lines)
               if re.search(header_regex, l) and not l.lstrip().startswith("|")]
    for hi in matches:
        for j in range(hi + 1, min(hi + 1 + _TABLE_WINDOW, len(lines))):
            ln = lines[j]
            if not (ln.lstrip().startswith("|") and "shape" in ln.lower()):
                continue
            cols = _header_cols(_split_row(ln))
            if cols is None:
                break
            rows = _collect_table_rows(lines, j + 2, cols)  # j + 2 skips header + separator
            if rows:
                return rows
            break  # table found but empty — try the next heading match
    return []


# ---------------------------------------------------------------------------
# Codegen helpers
# ---------------------------------------------------------------------------

def _ctor(shape: list[int], dtype: str) -> tuple[str, bool]:
    """Return (constructor_expr, needs_constraint_todo)."""
    shp = ", ".join(str(s) for s in shape)
    if dtype in _INT_DTYPES:
        if dtype == "torch.bool":
            expr = f"torch.randint(0, 2, ({shp},), dtype={dtype}, device=device)"
        elif dtype == "torch.uint8":
            expr = f"torch.randint(0, 256, ({shp},), dtype={dtype}, device=device)"
        else:
            expr = f"torch.randint(0, 8, ({shp},), dtype={dtype}, device=device)"
        return expr, True  # integer tensors are usually indices/quantized -> constrain
    return f"torch.randn({shp}, dtype={dtype}, device=device)", False


def _build_signature(op: str, tensors: list[dict], kwargs: dict) -> str:
    lines = [f"def {op}_golden("]
    for t in tensors:
        lines.append(f"    {t['name']}: torch.Tensor,")
    for k, v in kwargs.items():
        ann = "int" if isinstance(v, int) and not isinstance(v, bool) else \
              "bool" if isinstance(v, bool) else \
              "float" if isinstance(v, float) else "object"
        lines.append(f"    {k}: {ann} = {v!r},")
    lines.append("):")
    return "\n".join(lines)


def _build_make_inputs(tensors: list[dict], kwargs: dict) -> str:
    body = ["def _make_inputs(device):",
            '    """Construct the P0 case. Tensor shapes are taken verbatim from',
            "    SPEC.md front-matter p0_shapes / the §5 nn.Parameter table.",
            "",
            "    NOTE: integer tensors below are emitted as generic randint and are",
            "    flagged `# TODO: constrain` — indices / quantized / cache tensors",
            "    usually need legal values (see golden-generate SKILL §4).",
            '    """']
    names = []
    for t in tensors:
        shape = t.get("concrete")
        if shape is None:
            body.append(f"    # TODO: resolve concrete shape for "
                        f"{t['name']} (symbolic {t['shape']})")
            body.append(f"    {t['name']} = None  # TODO")
            names.append(t["name"])
            continue
        expr, constrain = _ctor(shape, t["dtype"])
        tag = "  # TODO: constrain" if constrain else ""
        body.append(f"    {t['name']} = {expr}{tag}")
        names.append(t["name"])
    kw = ", ".join(f'"{k}": {v!r}' for k, v in kwargs.items())
    body.append(f"    args = [{', '.join(names)}]")
    body.append(f"    kwargs = {{{kw}}}")
    body.append("    return args, kwargs")
    return "\n".join(body)


def _build_validate(op: str) -> str:
    return "\n".join([
        "def _validate():",
        '    """Deterministic harness: run golden on the P0 case + finiteness.',
        "    Fill op-specific property/range checks at the TODO markers below.",
        '    """',
        "    device = _get_device()",
        "    print('=' * 60)",
        f"    print('{op}_golden 验证报告')",
        "    print('=' * 60)",
        "    print(f'Device: {device}')",
        "",
        "    args, kwargs = _make_inputs(device)",
        f"    outs = {op}_golden(*args, **kwargs)",
        "    outs = outs if isinstance(outs, (tuple, list)) else (outs,)",
        "",
        "    print('\\n[finiteness]')",
        "    for i, o in enumerate(outs):",
        "        ok = torch.isfinite(o).all().item()",
        "        print(f'  out[{i}] shape={tuple(o.shape)} finite={ok} '",
        "              f'... {\"PASS\" if ok else \"FAIL\"}')",
        "        assert ok, f'out[{i}] contains NaN/Inf'",
        "",
        "    # TODO: output shape-equality vs SPEC §5 output table",
        "    # TODO: value-range checks derived from the formula (§6)",
        "    # TODO: math properties (monotonicity / symmetry / conservation)",
        "    # TODO: generalization sampling over dynamic axes",
        "",
        "    print('\\n' + '=' * 60)",
        "    print('验证完成')",
        "    print('=' * 60)",
    ])


REQUIRED_FM = ["op_name", "p0_shapes"]


def generate(spec_path: str, template_path: str) -> str:
    with open(spec_path, encoding="utf-8") as f:
        text = f.read()
    fm = parse_front_matter(text)
    missing = [k for k in REQUIRED_FM if not fm.get(k)]
    if missing:
        raise ValueError(f"SPEC.md missing required front-matter fields: {missing}")
    op = fm["op_name"]
    p0 = fm["p0_shapes"]
    kwargs = fm.get("default_params") or {}

    # Input table heading varies: **输入规格** / ### 3.1 输入张量 (Wrapper 接口)
    inputs = _parse_table_after(text, r"输入规格|输入张量")
    # Parameter / weight table heading varies across SPECs:
    #   nn.Parameter 参数 / 模型参数 (state_dict) / 权重规格
    # (NOT "Init 参数" — those are scalars, captured via default_params.)
    params = _parse_table_after(text, r"nn\.Parameter|模型参数|state_dict|权重规格")
    if not inputs:
        raise ValueError("could not locate §5 输入规格 table.")

    # Zip p0_shapes with the input rows (p0_shapes describes the forward inputs).
    for i, row in enumerate(inputs):
        row["concrete"] = list(p0[i]) if i < len(p0) else None
    # nn.Parameter rows carry concrete shapes in the table itself.
    for row in params:
        try:
            row["concrete"] = ast.literal_eval(row["shape"])
        except (ValueError, SyntaxError):
            row["concrete"] = None

    tensors = inputs + params

    with open(template_path, encoding="utf-8") as f:
        template = f.read()
    out = template.replace("{op}", op)

    # Replace the placeholder signature line (already {op}-substituted above).
    out = out.replace(f"def {op}_golden(x: torch.Tensor) -> torch.Tensor:",
                      _build_signature(op, tensors, kwargs), 1)

    # Swap the default _make_inputs (lines from `def _make_inputs` to its
    # `return [x], {}`) with the generated one.
    out = _replace_block(out, "def _make_inputs(device):",
                         _build_make_inputs(tensors, kwargs))
    out = _replace_block(out, "def _validate():", _build_validate(op))
    return out


def _replace_block(src: str, header: str, new_block: str) -> str:
    """Replace from `header` line up to (excluding) the next top-level `def `,
    `if __name__`, or the `# ===` banner that follows."""
    lines = src.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith(header)), None)
    if start is None:
        return src
    end = len(lines)
    _stops = ("def ", "if __name__", "# ====", "# ----")
    for j in range(start + 1, len(lines)):
        if any(lines[j].startswith(s) for s in _stops):
            end = j
            break
    return "\n".join(lines[:start] + new_block.splitlines() + [""] + lines[end:])


# ---------------------------------------------------------------------------

_SELF_TEST_SPEC = """---
schema_version: 1
op_name: demo_add
p0_shapes: [[8, 1024], [8, 1024]]
default_params: {'eps': 1e-6}
dynamic_axes: ['M']
dynamic_axes_ranges: [[1, 128]]
tolerance: {'rtol': 0.004, 'atol': 0.004}
---

### 5. 输入输出规格

**输入规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| a (x0) | [M, 1024] | float32 | M | left |
| b (x1) | [M, 1024] | float32 | M | right |

**输出规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| y (y0) | [M, 1024] | float32 | M | sum |
"""


def _self_test(template_path: str) -> int:
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False,
                                     encoding="utf-8") as f:
        f.write(_SELF_TEST_SPEC)
        spec = f.name
    code = generate(spec, template_path)
    os.unlink(spec)
    checks = {
        "signature": "def demo_add_golden(" in code and "a: torch.Tensor" in code
        and "eps: float = 1e-06" in code,
        "make_inputs": "a = torch.randn(8, 1024, dtype=torch.float32" in code,
        "validate_wired": "args, kwargs = _make_inputs(device)" in code,
        "compiles": _compiles(code),
        "no_placeholder": "{op}" not in code,
    }
    for k, v in checks.items():
        _LOGGER.info("  [%s] %s", "PASS" if v else "FAIL", k)
    return 0 if all(checks.values()) else 1


def _compiles(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        _LOGGER.info("    SyntaxError: %s", e)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Deterministic golden scaffold generator")
    ap.add_argument("--spec", help="path to SPEC.md")
    ap.add_argument("--template", required=True, help="path to golden-template.py")
    ap.add_argument("--out", help="output path for <op>_golden.py (default: stdout)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    if a.self_test:
        return _self_test(a.template)
    if not a.spec:
        ap.error("--spec is required unless --self-test")
    try:
        code = generate(a.spec, a.template)
    except ValueError as e:
        _LOGGER.error("[FAIL] %s", e)
        return 1
    if not _compiles(code):
        _LOGGER.error("[FAIL] generated scaffold does not parse")
        return 1
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(code)
        _LOGGER.info("[OK] wrote %s", a.out)
    else:
        _LOGGER.info(code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
