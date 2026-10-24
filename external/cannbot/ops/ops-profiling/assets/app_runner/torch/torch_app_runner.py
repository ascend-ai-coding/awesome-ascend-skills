#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Generic torch application runner for the ops-profiling msprof protocol.

Copy this file into the target project (e.g. ``runner/torch_app_runner.py``)
and edit ONLY the ``TARGET ADAPTER`` block below.  The generic part must stay
untouched: it enforces the contract from
``references/application_runner_contract.md``:

  --probe-warmup-op <ArgMax|Max>   probe path (256 MiB fp16 tensor, real op, sync)
  --case-ids <csv ids> --samples 10 --execution-manifest <json>
  --warmup-mode framework_ops|compiled --warmup-op <ArgMax|Max|OpsProfilingWarmup>

Per case: construct inputs once, then exactly 10 adjacent pairs of
(warmup -> synchronize -> target -> synchronize).  No Python timing; msprof
op_summary is the only time source.
"""

import argparse
import csv
import json
import os
import sys
import tempfile

# ---------------------------------------------------------------------------
# TARGET ADAPTER (the only block to edit per project)
# ---------------------------------------------------------------------------
# If the target needs a project harness/extension, put it on sys.path, e.g.:
# sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness"))
import torch  # noqa: E402
import torch_npu  # noqa: E402

# import custom_ops  # noqa: E402  <- project's torch extension module

TARGET_OPERATOR = "Add"

# Exact profiler Op Names of every kernel launched by one target invocation,
# in launch order.  Discover with a one-case msprof run; keep updated.
# A torch binding typically also launches its output-allocation kernels
# (e.g. aclnnInplaceZero_*): include them, they are part of this runner's
# measurement scope.
TARGET_KERNEL_SEQUENCE = ["aclnnAdd_AddAiCore_Add"]


def parse_case_attrs(attrs):
    """CSV ``attrs`` cell (``k=v;k=v``) -> target input parameters tuple."""
    out = {}
    for kv in attrs.split(";"):
        key, val = kv.split("=")
        out[key] = int(val)
    return (out["m"], out["n"])


def make_case_inputs(params, seed=10):
    """Construct target inputs on the target device.  Must not launch any
    kernel whose Op Name collides with the warmup or target sequence.
    """
    m, n = params
    torch.manual_seed(seed)
    return [torch.rand((m, n), dtype=torch.float32).npu(), torch.rand((m, n), dtype=torch.float32).npu()]


def invoke_target(inputs):
    """One target invocation.  All kernels it launches must be covered by
    TARGET_KERNEL_SEQUENCE.  Standalone Add example; a real project binding
    replaces this with e.g. custom_ops.my_op_npu(*inputs).
    """
    x, y = inputs
    return torch.add(x, y)


# ---------------------------------------------------------------------------
# END TARGET ADAPTER (generic runner below; do not edit)
# ---------------------------------------------------------------------------

OPS_WARMUP_SHAPE = (128, 1024, 1024)
OPS_WARMUP_BYTES = 256 * 1024 * 1024

WARMUP_KERNEL_SEQUENCE = {
    "ArgMax": ["aclnnArgMax_ArgMaxV2AiCore_ArgMaxV2", "aclnnArgMax_CastAiCore_Cast"],
    "Max": ["aclnnMax_MaxAiCore_Max"],
}

IMPLEMENTATION = {
    "ArgMax": "torch_argmax_256mib_fp16_v1",
    "Max": "torch_max_256mib_fp16_v1",
}


def make_ops_warmup_tensor(device):
    value = torch.empty(OPS_WARMUP_SHAPE, dtype=torch.float16, device=device)
    assert value.numel() * value.element_size() == OPS_WARMUP_BYTES
    return value


def run_ops_warmup(value, warmup_op):
    if warmup_op == "ArgMax":
        return torch.argmax(value)
    if warmup_op == "Max":
        return torch.max(value)
    raise ValueError(f"unexpected ops warmup: {warmup_op}")


def load_cases(path):
    cases = {}
    order = []
    with open(path, newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            attrs = row.get("attrs", "")
            case_id = str(row["case_id"]).strip()
            cases[case_id] = parse_case_attrs(attrs) if attrs else None
            order.append(case_id)
    return cases, order


def compiled_warmup_call(device):
    from ops_profiling_warmup import compiled_warmup, make_warmup_tensor

    tensor = make_warmup_tensor(device)
    compiled_warmup(tensor)


def probe_warmup(warmup_op):
    device = "npu"
    value = make_ops_warmup_tensor(device)
    torch.npu.synchronize()
    run_ops_warmup(value, warmup_op)
    torch.npu.synchronize()
    print(f"[torch-app-runner] probe ok: {warmup_op} on {OPS_WARMUP_SHAPE} fp16", flush=True)


def atomic_write_json(path, value):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".exec-manifest.", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=None)
    parser.add_argument("--case-ids", default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--execution-manifest", default=None)
    parser.add_argument("--warmup-mode", default=None, choices=[None, "framework_ops", "compiled"])
    parser.add_argument("--warmup-op", default=None)
    parser.add_argument("--probe-warmup-op", default=None)
    args = parser.parse_args()

    if args.probe_warmup_op is not None:
        probe_warmup(args.probe_warmup_op)
        return 0

    for name, value in (
        ("--cases", args.cases),
        ("--case-ids", args.case_ids),
        ("--samples", args.samples),
        ("--execution-manifest", args.execution_manifest),
        ("--warmup-mode", args.warmup_mode),
        ("--warmup-op", args.warmup_op),
    ):
        if value is None:
            parser.error(f"{name} is required outside probe mode")
    if args.samples != 10:
        parser.error("--samples must be exactly 10")

    cases, order = load_cases(args.cases)
    requested = [part.strip() for part in args.case_ids.split(",") if part.strip()]
    missing = [case_id for case_id in requested if case_id not in cases]
    if missing:
        parser.error(f"unknown case ids: {missing}")
    selected = requested if requested else order

    device = "npu"
    torch_npu.npu.set_device(int(os.getenv("NPU_DEVICE_ID", os.getenv("ASCEND_DEVICE_ID", "0"))))
    torch.npu.synchronize()

    if args.warmup_mode == "framework_ops":
        warmup_tensor = make_ops_warmup_tensor(device)
        warmup_sequence = WARMUP_KERNEL_SEQUENCE[args.warmup_op]
        implementation = IMPLEMENTATION[args.warmup_op]
    else:
        warmup_sequence = ["OpsProfilingWarmup"]
        implementation = "ops_profiling_warmup_v1"
    torch.npu.synchronize()

    case_runs = []
    for case_id in selected:
        inputs = make_case_inputs(cases[case_id])
        torch.npu.synchronize()
        for _ in range(args.samples):
            if args.warmup_mode == "framework_ops":
                run_ops_warmup(warmup_tensor, args.warmup_op)
            else:
                compiled_warmup_call(device)
            torch.npu.synchronize()
            invoke_target(inputs)
            torch.npu.synchronize()
        case_runs.append(
            {
                "case_id": case_id,
                "warmup_kernel_sequence": warmup_sequence,
                "target_kernel_sequence": TARGET_KERNEL_SEQUENCE,
                "iteration_kernel_sequence": warmup_sequence + TARGET_KERNEL_SEQUENCE,
            }
        )
        print(f"[torch-app-runner] case={case_id} pairs={args.samples} done", flush=True)

    manifest = {
        "schema_version": 5,
        "samples_per_case": args.samples,
        "target_operator": TARGET_OPERATOR,
        "warmup": {
            "mode": args.warmup_mode,
            "logical_op": args.warmup_op,
            "implementation": implementation,
            "tensor_bytes": OPS_WARMUP_BYTES,
            "tensor_allocation": {
                "shape": list(OPS_WARMUP_SHAPE),
                "dtype": "float16",
                "elements": OPS_WARMUP_BYTES // 2,
            },
        },
        "case_runs": case_runs,
    }
    atomic_write_json(args.execution_manifest, manifest)
    print("[torch-app-runner] all cases done; execution manifest written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
