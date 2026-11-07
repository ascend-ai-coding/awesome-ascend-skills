# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import csv
import json
from pathlib import Path
from collections.abc import Sequence
from typing import Any, NamedTuple, cast

from models import HostApiCall, KernelInvocation, OperatorStats, StepTrace, TorchOpTiming
from parser_base import find_newest_csv, parse_api_statistic, parse_op_statistic, parse_optional_float
from pipeline_utils import build_kernel_invocation, build_pipeline_stage


class TorchNpuProfileArtifacts(NamedTuple):
    """Parsed torch_npu profiler artifacts with legacy tuple unpacking support."""

    operators: list[OperatorStats]
    invocations: list[KernelInvocation]
    torch_ops: list[TorchOpTiming]
    step_traces: list[StepTrace]
    host_api: list[HostApiCall]
    timeline: list[dict[str, Any]]
    source_files: dict[str, str | None]


def parse_kernel_details(csv_path: Path) -> list[KernelInvocation]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []

        # torch-npu-profiler uses: Name, Type, Accelerator Core, Duration(us), Wait Time(us), Block Dim
        name_col = "Name" if "Name" in fieldnames else None
        duration_col = "Duration(us)" if "Duration(us)" in fieldnames else None
        wait_col = "Wait Time(us)" if "Wait Time(us)" in fieldnames else None
        block_dim_col = "Block Dim" if "Block Dim" in fieldnames else None

        invocations: list[KernelInvocation] = []
        for row in reader:
            op_name = (row.get(name_col, "") if name_col else "").strip()
            if not op_name:
                continue
            duration = parse_optional_float(row[duration_col]) if duration_col else None
            wait_time = parse_optional_float(row[wait_col]) if wait_col else None
            block_dim = int(parse_optional_float(row[block_dim_col]) or 0) if block_dim_col else 0

            pipeline = build_pipeline_stage(fieldnames, row)
            invocations.append(build_kernel_invocation(op_name, duration, wait_time, block_dim, pipeline))

    return invocations



def parse_operator_details(csv_path: Path) -> list[TorchOpTiming]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []

        has_device = "Device Self Duration(us)" in fieldnames
        has_aicore = "Device Self Duration With AICore(us)" in fieldnames
        ops: list[TorchOpTiming] = []
        for row in reader:
            timing = _operator_timing_from_row(row, has_device, has_aicore)
            if timing is not None:
                ops.append(timing)

    return ops


def _operator_timing_from_row(
    row: dict[str, str],
    has_device: bool,
    has_aicore: bool,
) -> TorchOpTiming | None:
    name = row.get("Name", "").strip()
    if not name:
        return None
    device_self, device_total = _device_timings(row, has_device, has_aicore)
    return TorchOpTiming(
        name=name,
        host_self_us=parse_optional_float(row.get("Host Self Duration(us)")) or 0.0,
        host_total_us=parse_optional_float(row.get("Host Total Duration(us)")) or 0.0,
        device_self_us=device_self,
        device_total_us=device_total,
    )


def _device_timings(
    row: dict[str, str],
    has_device: bool,
    has_aicore: bool,
) -> tuple[float, float]:
    device_self = parse_optional_float(row.get("Device Self Duration(us)")) if has_device else 0.0
    device_total = parse_optional_float(row.get("Device Total Duration(us)")) if has_device else 0.0
    if not has_aicore:
        return device_self or 0.0, device_total or 0.0
    aicore_self = parse_optional_float(row.get("Device Self Duration With AICore(us)")) or 0.0
    if aicore_self <= 0:
        return device_self or 0.0, device_total or 0.0
    aicore_total = parse_optional_float(row.get("Device Total Duration With AICore(us)")) or 0.0
    return aicore_self, aicore_total


def parse_step_trace(csv_path: Path) -> list[StepTrace]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        traces: list[StepTrace] = []
        for row in reader:
            traces.append(
                StepTrace(
                    step=int(row.get("Step", "0")),
                    computing_us=parse_optional_float(row.get("Computing")) or 0.0,
                    communication_not_overlapped_us=parse_optional_float(
                        row.get("Communication(Not Overlapped)")
                    ) or 0.0,
                    overlapped_us=parse_optional_float(row.get("Overlapped")) or 0.0,
                    communication_us=parse_optional_float(row.get("Communication")) or 0.0,
                    free_us=parse_optional_float(row.get("Free")) or 0.0,
                    stage_us=parse_optional_float(row.get("Stage")) or 0.0,
                    bubble_us=parse_optional_float(row.get("Bubble")) or 0.0,
                    communication_not_overlapped_exclude_receive_us=parse_optional_float(
                        row.get("Communication(Not Overlapped and Exclude Receive)")
                    ) or 0.0,
                    preparing_us=parse_optional_float(row.get("Preparing")) or 0.0,
                )
            )
    return traces


def parse_trace_view(json_path: Path) -> list[dict[str, Any]]:
    data: Any = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return cast(list[dict[str, Any]], data)
    if isinstance(data, dict) and "traceEvents" in data:
        return cast(list[dict[str, Any]], data["traceEvents"])
    return []


class TorchNpuProfilerParser:
    """Parser for torch_npu.profiler ASCEND_PROFILER_OUTPUT/ artifacts."""

    @staticmethod
    def parse(artifacts_dir: Path) -> TorchNpuProfileArtifacts:
        files = _torch_profile_files(artifacts_dir)
        return TorchNpuProfileArtifacts(
            parse_op_statistic(_required_op_statistic(files)),
            _parse_optional_csv(files["kernel_details"], parse_kernel_details),
            _parse_optional_csv(files["operator_details"], parse_operator_details),
            _parse_optional_csv(files["step_trace_time"], parse_step_trace),
            _parse_optional_csv(files["api_statistic"], parse_api_statistic),
            _parse_optional_trace(files["trace_view"]),
            _source_file_names(files),
        )


def _torch_profile_files(artifacts_dir: Path) -> dict[str, Path | None]:
    return {
        "op_statistic": find_newest_csv(artifacts_dir, "op_statistic"),
        "kernel_details": find_newest_csv(artifacts_dir, "kernel_details"),
        "operator_details": find_newest_csv(artifacts_dir, "operator_details"),
        "step_trace_time": find_newest_csv(artifacts_dir, "step_trace_time"),
        "api_statistic": find_newest_csv(artifacts_dir, "api_statistic"),
        "trace_view": _latest_trace_view(artifacts_dir),
    }


def _latest_trace_view(artifacts_dir: Path) -> Path | None:
    trace_views = sorted(artifacts_dir.glob("trace_view.json"))
    return trace_views[-1] if trace_views else None


def _required_op_statistic(files: dict[str, Path | None]) -> Path:
    op_statistic = files["op_statistic"]
    if op_statistic is None:
        raise FileNotFoundError("No op_statistic CSV found in profiler artifacts")
    return op_statistic


def _parse_optional_csv(path: Path | None, parser: Any) -> list[Any]:
    return parser(path) if path is not None else []


def _parse_optional_trace(path: Path | None) -> list[dict[str, Any]]:
    return parse_trace_view(path) if path is not None else []


def _source_file_names(files: dict[str, Path | None]) -> dict[str, str | None]:
    return {name: path.name if path is not None else None for name, path in files.items()}
