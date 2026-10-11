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

from models import HostApiCall, KernelInvocation, OperatorStats, TaskRecord
from parser_base import find_newest_csv, parse_api_statistic, parse_op_statistic, parse_optional_float
from pipeline_utils import build_kernel_invocation, build_pipeline_stage

_OP_NAME_COLUMNS = ("OP Type", "Op Name", "Name")
_DURATION_COLUMNS = ("Task Duration(us)", "Duration(us)")
_WAIT_COLUMNS = ("Task Wait Time(us)", "Wait Time(us)")
_BLOCK_DIM_COLUMNS = ("Block Dim", "Mix Block Dim")

_TASK_NAME_COLUMNS = ("kernel_name", "Kernel Name", "Op Name", "OP Type")
_TASK_DURATION_COLUMNS = ("task_time(us)", "Task Duration(us)")
_TASK_START_COLUMNS = ("task_start(us)", "Task Start Time(us)")
_TASK_STOP_COLUMNS = ("task_stop(us)", "Task Stop Time(us)")
_TASK_TYPE_COLUMNS = ("kernel_type", "Kernel Type")


class MsprofArtifacts(NamedTuple):
    """Parsed msprof artifacts; tuple semantics preserve existing callers."""

    operators: list[OperatorStats]
    invocations: list[KernelInvocation]
    task_records: list[TaskRecord]
    host_api: list[HostApiCall]
    timeline: list[dict[str, Any]]
    source_files: dict[str, str | None]


def _find_column(fieldnames: Sequence[str], candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in fieldnames:
            return col
    return None


def parse_op_summary(csv_path: Path) -> list[KernelInvocation]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        name_col = _find_column(fieldnames, _OP_NAME_COLUMNS)
        duration_col = _find_column(fieldnames, _DURATION_COLUMNS)
        wait_col = _find_column(fieldnames, _WAIT_COLUMNS)
        block_dim_col = _find_column(fieldnames, _BLOCK_DIM_COLUMNS)

        invocations: list[KernelInvocation] = []
        for row in reader:
            op_name = (row.get(name_col, "") if name_col else "").strip()
            if not op_name:
                continue
            duration = parse_optional_float(row[duration_col]) if duration_col else 0.0
            wait_time = parse_optional_float(row[wait_col]) if wait_col else 0.0
            block_dim = int(parse_optional_float(row[block_dim_col]) or 0) if block_dim_col else 0

            pipeline = build_pipeline_stage(fieldnames, row)
            invocations.append(build_kernel_invocation(op_name, duration, wait_time, block_dim, pipeline))

    return invocations


def parse_task_time(csv_path: Path) -> list[TaskRecord]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        name_col = _find_column(fieldnames, _TASK_NAME_COLUMNS)
        duration_col = _find_column(fieldnames, _TASK_DURATION_COLUMNS)
        start_col = _find_column(fieldnames, _TASK_START_COLUMNS)
        stop_col = _find_column(fieldnames, _TASK_STOP_COLUMNS)
        type_col = _find_column(fieldnames, _TASK_TYPE_COLUMNS)

        records: list[TaskRecord] = []
        for row in reader:
            kernel_name = (row.get(name_col, "") if name_col else "").strip()
            if not kernel_name or kernel_name == "N/A":
                continue
            records.append(
                TaskRecord(
                    kernel_name=kernel_name,
                    kernel_type=(row.get(type_col, "") if type_col else "").strip(),
                    task_time_us=parse_optional_float(row[duration_col]) or 0.0 if duration_col else 0.0,
                    task_start_us=parse_optional_float(row[start_col]) or 0.0 if start_col else 0.0,
                    task_stop_us=parse_optional_float(row[stop_col]) or 0.0 if stop_col else 0.0,
                )
            )

    return records


def parse_msprof_json(json_path: Path) -> list[dict[str, Any]]:
    data: Any = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return cast(list[dict[str, Any]], data)
    if isinstance(data, dict) and "traceEvents" in data:
        return cast(list[dict[str, Any]], data["traceEvents"])
    return []


class MsprofParser:
    """Parser for msprof PROF_*/mindstudio_profiler_output/ artifacts."""

    @staticmethod
    def parse(artifacts_dir: Path) -> MsprofArtifacts:
        op_statistic_csv = find_newest_csv(artifacts_dir, "op_statistic")
        if op_statistic_csv is None:
            raise FileNotFoundError(f"No op_statistic CSV found in {artifacts_dir}")

        op_summary_csv = find_newest_csv(artifacts_dir, "op_summary")
        task_time_csv = find_newest_csv(artifacts_dir, "task_time")
        api_statistic_csv = find_newest_csv(artifacts_dir, "api_statistic")
        msprof_json_files = sorted(artifacts_dir.glob("msprof_*.json"))

        operators = parse_op_statistic(op_statistic_csv)
        invocations = parse_op_summary(op_summary_csv) if op_summary_csv else []
        task_records = parse_task_time(task_time_csv) if task_time_csv else []
        host_api = parse_api_statistic(api_statistic_csv) if api_statistic_csv else []

        timeline: list[dict[str, Any]] = []
        if msprof_json_files:
            timeline = parse_msprof_json(msprof_json_files[-1])

        source_files: dict[str, str | None] = {
            "op_statistic": op_statistic_csv.name,
            "op_summary": op_summary_csv.name if op_summary_csv else None,
            "task_time": task_time_csv.name if task_time_csv else None,
            "api_statistic": api_statistic_csv.name if api_statistic_csv else None,
            "msprof_json": msprof_json_files[-1].name if msprof_json_files else None,
        }

        return MsprofArtifacts(
            operators,
            invocations,
            task_records,
            host_api,
            timeline,
            source_files,
        )
