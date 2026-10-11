#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Derive Stage 3 design parameters from a golden file (SKILL.md Round 0/2/4).

Codifies the deterministic parts of ``pypto-op-design/SKILL.md``:
  - Round 0  — complexity signals L/O, total_complexity, module_count
  - Round 2  — pre-Stage-7 tile-shape baseline, per-op UB estimate,
               expression-expansion bound
  - Round 4  — tiling-layer checklist verdicts (machine-checkable subset)

Human-judged signals stay human: pass loop-carried state groups via
``--state-groups`` and cross-tile reduce count via ``--cross-tile-reduce``
(SKILL.md Round 0 defines both as manual counts).

Usage::

    python derive_design_params.py <op_golden.py> \
        --state-groups 1 --cross-tile-reduce 1 \
        [--dtype bf16] [--ub-bytes 196608] [--json]

    python derive_design_params.py --self-test

Output: human-readable summary (default) or JSON (``--json``) with keys
``signals``, ``total_complexity``, ``module_count``, ``path``,
``tile_baseline``, ``checks``.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import subprocess
import sys
from pathlib import Path

# Emit on stdout with a bare (message-only) format so JSON / summary output
# stays machine-parseable for the caller (verifier reads stdout).
_LOGGER = logging.getLogger("derive_design_params")

UB_BYTES_DEFAULT = 192 * 1024
EXPANSION_LIMIT = 18000
LINES_PER_UNIT = 30.0
OPS_PER_UNIT = 3.0
L0_THRESHOLD = 1.3
LINES_CAP_DIVISOR = 12
VEC_TILE_MIN, VEC_TILE_MAX = 16, 64
CUBE_TILE = 128
DTYPE_BYTES = {"fp32": 4, "float32": 4, "fp16": 2, "float16": 2, "bf16": 2, "bfloat16": 2}


def count_effective_lines(golden: Path) -> int:
    """Delegate to the bundled count_golden_lines.py (single source of truth)."""
    script = Path(__file__).with_name("count_golden_lines.py")
    out = subprocess.run(
        [sys.executable, str(script), str(golden)],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip())


def count_matmuls(golden: Path) -> int:
    """AST count of torch.matmul / @ / torch.einsum / torch.bmm calls."""
    tree = ast.parse(golden.read_text(encoding="utf-8"))
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            n += 1
        elif isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name in {"matmul", "bmm", "einsum", "mm"}:
                n += 1
    return n


def derive(lines: int, matmuls: int, state_groups: int, cross_tile_reduce: int) -> dict:
    sig_l = lines / LINES_PER_UNIT
    sig_o = (matmuls + cross_tile_reduce) / OPS_PER_UNIT
    total = max(sig_l, float(state_groups), sig_o)
    if total < L0_THRESHOLD:
        module_count = 1
    else:
        module_count = max(1, min(round(total), math.ceil(lines / LINES_CAP_DIVISOR)))
    return {
        "signals": {"L": round(sig_l, 2), "S": state_groups, "O": round(sig_o, 2),
                    "effective_lines": lines, "matmul_count": matmuls,
                    "cross_tile_reduce_count": cross_tile_reduce},
        "total_complexity": round(total, 2),
        "module_count": module_count,
        "path": "L0" if module_count == 1 else "L1",
    }


def tile_baseline(dtype: str, ub_bytes: int) -> dict:
    bytes_per = DTYPE_BYTES[dtype.lower()]
    vec = [VEC_TILE_MIN, VEC_TILE_MAX]
    # 最重 op 保守估算: reduce/expand tensor_count = 4
    worst_tile = VEC_TILE_MAX * VEC_TILE_MAX
    ub_per_op = worst_tile * bytes_per * 4
    return {
        "cube_tile": [CUBE_TILE, CUBE_TILE],
        "vec_tile_range": vec,
        "tail_alignment": 8 if bytes_per == 4 else 16,
        "ub_bytes": ub_bytes,
        "worst_op_ub_bytes": ub_per_op,
        "checks": {
            "ub_fits_worst_op": ub_per_op <= ub_bytes,
            "expansion_limit": EXPANSION_LIMIT,
            "tile_params_static": True,  # 常量基线: cube tile 必须是编译期常量
        },
    }


def self_test() -> int:
    # SKILL.md §0.6 验算表
    cases = [
        ("GELU", 4, 0, 0, 0, 1), ("Softmax", 6, 0, 0, 2, 1),
        ("Layernorm", 7, 0, 0, 2, 1), ("attention_fwd", 5, 0, 2, 2, 1),
        ("attention_bwd", 17, 0, 6, 4, 2), ("FA_fwd", 28, 1, 2, 1, 1),
        ("FA_bwd", 45, 2, 4, 2, 2), ("gated_delta_rule", 40, 2, 4, 2, 2),
        ("mamba_ssm", 55, 2, 6, 2, 3),
    ]
    bad = 0
    for name, lines, s, mm, red, want in cases:
        got = derive(lines, mm, s, red)["module_count"]
        ok = got == want
        bad += 0 if ok else 1
        _LOGGER.info("%s %s: module_count=%s expected=%s",
                     "PASS" if ok else "FAIL", name, got, want)
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("golden", nargs="?", type=Path)
    ap.add_argument("--state-groups", type=int, default=0)
    ap.add_argument("--cross-tile-reduce", type=int, default=0)
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--ub-bytes", type=int, default=UB_BYTES_DEFAULT)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    if args.self_test:
        return self_test()
    if not args.golden:
        ap.error("golden path required (or --self-test)")
    lines = count_effective_lines(args.golden)
    result = derive(lines, count_matmuls(args.golden), args.state_groups, args.cross_tile_reduce)
    result["tile_baseline"] = tile_baseline(args.dtype, args.ub_bytes)
    if args.json:
        _LOGGER.info(json.dumps(result, indent=2))
    else:
        _LOGGER.info("total_complexity=%s module_count=%s path=%s (L=%s S=%s O=%s)",
                     result["total_complexity"], result["module_count"], result["path"],
                     result["signals"]["L"], result["signals"]["S"], result["signals"]["O"])
        _LOGGER.info("cube_tile=%s vec_tile=[%s,%s] ub_fits_worst_op=%s",
                     CUBE_TILE, VEC_TILE_MIN, VEC_TILE_MAX,
                     result["tile_baseline"]["checks"]["ub_fits_worst_op"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
