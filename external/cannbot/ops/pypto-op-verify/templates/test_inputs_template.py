# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Template: custom/<op>/eval/test_inputs.py
#
# The ONLY op-specific file in the eval/ harness. Copy to
# custom/<op>/eval/test_inputs.py and fill the two op-specific blocks marked
# "AGENT FILLS": PRIMARY_INPUT_ORDER and the body of make_inputs.
#
# Contract (Step B.1):
#   - PRIMARY_INPUT_ORDER mirrors module_interfaces.yaml primary_inputs order;
#     the runner maps make_inputs() keys to positional args through it.
#   - make_inputs keys MUST match the golden's parameter names.
#   - ALL tensors are created directly on the NPU device via device=DEVICE at
#     construction time — never CPU-first then .npu()/.to(...). This eliminates
#     the "inputs on mixed devices" failure class at its source.

from __future__ import annotations

import os

import torch

DEVICE = torch.device(f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', '0'))}")

_DTYPES = {
    "float32": torch.float32, "fp32": torch.float32,
    "float16": torch.float16, "fp16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "int32": torch.int32, "int64": torch.int64, "bool": torch.bool,
}

# ── AGENT FILLS (1/2): primary input order, mirroring module_interfaces.yaml ──
PRIMARY_INPUT_ORDER: list[str] = ["x"]


def _dtype_from_str(dtype_field) -> torch.dtype:
    """Resolve a case 'dtype' field (str, or dict with 'default'/per-name)."""
    if isinstance(dtype_field, dict):
        dtype_field = dtype_field.get("default", "float32")
    return _DTYPES[str(dtype_field).lower()]


def _resolve_shape(shape_field: dict) -> dict[str, list[int]]:
    """Resolve symbolic shape exprs (e.g. 'S/BT+1') against the case dims.

    Numeric dims (B, T, ...) are provided directly in the case 'shape' dict;
    any string value is evaluated as an arithmetic expression over those dims
    using only + - * // (no names beyond the declared dims)."""
    dims = {k: v for k, v in shape_field.items()
            if isinstance(v, int) and not k.startswith("_")}
    resolved: dict[str, list[int]] = {}
    for name, spec in shape_field.items():
        if name.startswith("_") or not isinstance(spec, (list, tuple, str)):
            continue
        axes = spec if isinstance(spec, (list, tuple)) else [spec]
        out = []
        for ax in axes:
            out.append(int(ax) if isinstance(ax, int)
                       else int(eval(str(ax), {"__builtins__": {}}, dims)))  # noqa: S307
        resolved[name] = out
    return resolved


def make_inputs(case: dict) -> dict[str, "torch.Tensor | int"]:
    """Build all primary inputs for a case, every tensor on DEVICE."""
    # AGENT FILLS (2/2): construct each primary input below. Use the canonical
    # pattern — randn / zeros / etc. placed on the DEVICE and seeded from the
    # case. Read op-specific knobs (gate_mode, h0_mode, boundary flags,
    # cancellation_stress) from the case dict; keys MUST match the golden's
    # parameter names.
    torch.manual_seed(case.get("seed", 42))
    dtype = _dtype_from_str(case.get("dtype", "float32"))
    shapes = _resolve_shape(case.get("shape", {}))

    # Example (replace with this op's actual inputs):
    inputs: dict[str, "torch.Tensor | int"] = {
        "x": torch.randn(shapes.get("x", [shapes.get("B", 1)]), dtype=dtype, device=DEVICE),
    }

    # cancellation_stress hook (Step B.6): engineer near-equal operands.
    cs = case.get("cancellation_stress")
    if cs:
        # deterministic under cs["seed"]; tighten the subtractive pair in
        # cs["expression"] to within cs["relative_gap"]. No-op for ops without
        # subtraction. Fill per op when applicable.
        pass

    return inputs
