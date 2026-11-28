# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Operator-agnostic cost model: the FLOP/byte denominators for MBU/MFU, WITHOUT per-op registration.

The FLOP and byte numbers a roofline needs come from two fundamentally different places:

- ``hbm_bytes`` (the MBU numerator) has two operator-agnostic sources:
  * ``bytes_from_io`` -- sum ``prod(shape) * dtype_bytes`` over every I/O tensor. Holds
    for ANY op with a declared shape, but ESTIMATES: it counts the full declared tensor,
    so it over-counts when the traffic moved != the shape (in-place KV-cache that writes
    a few tokens of a huge cache; weights served from L2). No profiling needed.
  * ``bytes_from_profiling`` -- main_mem bandwidth * duration, the MEASURED HBM traffic.
    Correct even for in-place/cache/L2-resident ops, because it reads what actually
    crossed the boundary. Needs profiling collected with ``--aic-metrics=Memory``.

- ``flops`` (the MFU numerator) CANNOT be inferred from shapes: an ``add`` and a
  ``matmul`` over the same shapes differ by orders of magnitude. So FLOPs come from
  a *measurement* -- the profiler's per-op arithmetic counter -- via
  ``flops_from_profiling``, or are supplied explicitly. FLOPs are optional; without
  them we still report MBU (the binding metric for the common memory-bound case).

This replaces the former per-operator registry (``@register("matmul")`` etc.): there
is no registry and no seed formulas. Adding an operator means passing its I/O shapes
(for bytes) and pointing at its profiling row (for FLOPs) -- no code change.

Consumed by ``compute_efficiency.py``.
"""

import csv
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Cost:
    """Intrinsic work of one operator invocation.

    ``flops`` may be None when no measurement/explicit value is available; consumers
    then report MBU only. ``unit`` says which compute engine does the FLOPs so MFU
    divides by the right peak: "cube" (matmul/mma), "vector" (elementwise/reduction),
    or "mixed" (cube-dominated fused op; treated as cube for the ceiling).
    """

    hbm_bytes: float
    flops: Optional[float] = None
    label: str = ""
    unit: str = "cube"

    # testkit-compatible aliases so this drops into metrics.py-style consumers.
    @property
    def total_flops(self) -> Optional[float]:
        return self.flops

    @property
    def total_bytes(self) -> float:
        return self.hbm_bytes


# dtype string -> bytes. Key set mirrors msprof_perf_summary.py:_JSONL_DTYPE_MAP so the
# same case descriptors feed both. fp8 family added (1 byte) for Ascend low-precision ops.
DTYPE_BYTES: Dict[str, int] = {
    "fp16": 2,
    "float16": 2,
    "half": 2,
    "bf16": 2,
    "bfloat16": 2,
    "fp32": 4,
    "float32": 4,
    "float": 4,
    "fp64": 8,
    "float64": 8,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int": 4,
    "int64": 8,
    "uint64": 8,
    "long": 8,
    "fp8": 1,
    "float8_e4m3": 1,
    "float8_e5m2": 1,
    "hifloat8": 1,
    "complex64": 8,
    "complex128": 16,
}
_DEFAULT_DTYPE_BYTES = 2  # assume 2-byte (bf16/fp16) when dtype is unknown/missing


def _prod(shape) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


def tensor_bytes(t: Dict[str, Any]) -> int:
    """Bytes of one tensor descriptor ``{"shape":[...], "dtype":"..."}``.

    shape None or empty -> 0 (e.g. an unspecified optional bias). Unknown dtype falls
    back to 2 bytes.
    """
    shape = t.get("shape")
    if not shape:
        return 0
    nb = DTYPE_BYTES.get(str(t.get("dtype", "")).lower(), _DEFAULT_DTYPE_BYTES)
    return _prod(shape) * nb


def _has_unknown_dtype(items: List[Dict[str, Any]]) -> bool:
    for t in items:
        if t.get("type", "tensor") != "tensor":
            continue
        if not t.get("shape"):
            continue
        if str(t.get("dtype", "")).lower() not in DTYPE_BYTES:
            return True
    return False


def bytes_from_io(inputs, outputs=None) -> int:
    """Sum bytes over all I/O tensors -- the operator-agnostic MBU numerator.

    Accepts the project's standard case descriptors (see msprof_perf_summary.py:658):
    each item is ``{"type": "tensor", "shape": [...], "dtype": "..."}``. Non-tensor
    items (scalars/attrs) are skipped. ``outputs`` is optional but recommended -- for
    a memory-bound op the output write is a real part of the byte traffic.
    """
    items = list(inputs or []) + list(outputs or [])
    total = 0
    for t in items:
        if t.get("type", "tensor") != "tensor":
            continue
        total += tensor_bytes(t)
    return total


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        return list(csv.DictReader(fh))


def _safe_float(val, default: float = 0.0) -> float:
    if val is None:
        return default
    s = str(val).strip()
    if s in ("", "N/A", "NA", "-"):
        return default
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


def _pick_op_row(rows, op_name):
    """Select the profiling row for the target op (mirrors msprof pick_target_row)."""
    if not rows:
        raise ValueError("empty op_summary csv")
    if op_name:
        for r in rows:
            if r.get("Op Name", "").strip() == op_name:
                return r
        raise ValueError(f"op '{op_name}' not found in op_summary (available: {[r.get('Op Name') for r in rows]})")
    core = [r for r in rows if any(k in r.get("Task Type", "") for k in ("AI_CORE", "AIV", "MIX"))]
    core = core or rows
    if len(core) > 1:
        raise ValueError(
            f"op_summary has {len(core)} compute rows; pass op_name to select one "
            f"(available: {[r.get('Op Name') for r in core]})"
        )
    return core[0]


def flops_from_profiling(
    op_summary_csv: str, op_name: Optional[str] = None, field: str = "aic_cube_fops", scale: float = 1.0
) -> float:
    """Read the measured FLOP count from a profiler op_summary CSV.

    ``field`` defaults to the cube arithmetic counter ``aic_cube_fops`` (the column
    msprof_perf_summary.py reads at line 443). Its exact semantics (whether it already
    counts multiply+add as 2, and its unit) must be validated against real profiling
    output on the SoC under test -- hence both ``field`` and ``scale`` are configurable
    and ``scale`` defaults to 1.0 (no assumption).
    """
    if not os.path.isfile(op_summary_csv):
        raise FileNotFoundError(f"op_summary csv not found: {op_summary_csv}")
    row = _pick_op_row(_read_csv_rows(op_summary_csv), op_name)
    if field not in row:
        raise KeyError(f"field '{field}' not in op_summary columns: {sorted(row)}")
    return _safe_float(row[field]) * scale


# main-memory (HBM) bandwidth columns msprof emits under --aic-metrics=Memory.
# Summing read+write over both the cube (aic) and vector (aiv) cores gives the total
# traffic that crossed the HBM boundary. Absent columns read as 0.
_MEM_BW_FIELDS = (
    "aic_main_mem_read_bw(GB/s)",
    "aic_main_mem_write_bw(GB/s)",
    "aiv_main_mem_read_bw(GB/s)",
    "aiv_main_mem_write_bw(GB/s)",
)


def bytes_from_profiling(
    op_summary_csv: str,
    op_name: Optional[str] = None,
    bw_fields=_MEM_BW_FIELDS,
    duration_field: str = "Task Duration(us)",
    scale: float = 1.0,
) -> float:
    """Read the *measured* HBM bytes from a profiler op_summary CSV.

    ``bytes = (sum of main_mem read/write bandwidths in GB/s) * Task_Duration(us) * 1e3``.
    This is the only operator-agnostic byte source that stays correct when the declared
    tensor shapes do NOT equal the traffic actually moved -- e.g. an in-place KV-cache
    update that writes only a few tokens of a huge cache tensor, or weights that are
    served from L2 (they don't show up in main_mem bandwidth). Requires the profiling to
    be collected with ``--aic-metrics=Memory`` (the basic op_summary lacks these columns).

    The row's own Task Duration is used so bandwidth and time come from the same capture
    (self-consistent); ``scale`` is a manual override kept for parity with
    ``flops_from_profiling`` and defaults to 1.0.
    """
    if not os.path.isfile(op_summary_csv):
        raise FileNotFoundError(f"op_summary csv not found: {op_summary_csv}")
    row = _pick_op_row(_read_csv_rows(op_summary_csv), op_name)
    dur_us = _safe_float(row.get(duration_field))
    if dur_us <= 0:
        raise ValueError(f"row has no positive '{duration_field}' to scale bandwidth by")
    bw_sum = sum(_safe_float(row.get(f)) for f in bw_fields)  # GB/s
    if bw_sum <= 0:
        raise ValueError(
            "no main_mem bandwidth in op_summary (all of "
            f"{list(bw_fields)} are 0/absent). Collect profiling with --aic-metrics=Memory, "
            "or fall back to --io/--case shape-derived bytes."
        )
    return bw_sum * dur_us * 1e3 * scale  # GB/s * us * 1e3 = bytes


def compute_cost(io: Dict[str, Any], flops: Optional[float] = None, unit: str = "cube", label: str = "") -> Cost:
    """Assemble a Cost from an I/O descriptor (-> bytes) and an optional FLOP count.

    ``io`` = ``{"inputs": [...], "outputs": [...]}`` (outputs optional). This is the
    single entry point: bytes are always derived generically; flops are passed through
    (from profiling / explicit / None).
    """
    inputs = io.get("inputs", [])
    outputs = io.get("outputs")
    hbm_bytes = bytes_from_io(inputs, outputs)
    if not label:
        n_in = sum(1 for t in inputs if t.get("type", "tensor") == "tensor")
        n_out = sum(1 for t in (outputs or []) if t.get("type", "tensor") == "tensor")
        label = f"op in={n_in} out={n_out}"
        if _has_unknown_dtype(list(inputs) + list(outputs or [])):
            label += " [dtype? 假定2B]"
    return Cost(hbm_bytes=hbm_bytes, flops=flops, unit=unit, label=label)
