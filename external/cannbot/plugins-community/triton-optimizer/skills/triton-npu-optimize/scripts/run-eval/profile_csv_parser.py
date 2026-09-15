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
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from perf_artifacts import PerfMetrics, PerfOpRow, format_latency_value


@dataclass(frozen=True)
class ParsedProfileCsvRows:
    source_path: Path
    ops: list[PerfOpRow]
    total_time_us: float
    total_op_avg_time_us: float


@dataclass(frozen=True)
class OpStatisticCsvRow:
    op_type: str
    native_avg_time_us: float
    total_time_us: float


@dataclass(frozen=True)
class _OpStatisticParseData:
    rows: list[OpStatisticCsvRow]
    total_time_us: float
    positive_counts: list[int]


def _new_duration_map() -> dict[str, float]:
    return {}


def _new_op_order() -> list[str]:
    return []


@dataclass
class KernelDetailsAggregation:
    total_time_us: float = 0.0
    total_duration_us_by_op: dict[str, float] = field(default_factory=_new_duration_map)
    op_order: list[str] = field(default_factory=_new_op_order)
    total_duration_us_by_step: dict[str, float] = field(default_factory=_new_duration_map)

    @property
    def observed_step_count(self) -> int:
        return len(self.total_duration_us_by_step)

    def record(self, *, op_type: str, duration_us: float, step_id: str | None) -> None:
        self.total_time_us += duration_us
        if step_id:
            self.total_duration_us_by_step[step_id] = (
                self.total_duration_us_by_step.get(step_id, 0.0) + duration_us
            )
        if op_type not in self.total_duration_us_by_op:
            self.total_duration_us_by_op[op_type] = 0.0
            self.op_order.append(op_type)
        self.total_duration_us_by_op[op_type] += duration_us

    def build_ops(self, *, divisor: int) -> list[PerfOpRow]:
        ops: list[PerfOpRow] = []
        for op_type in self.op_order:
            avg_time_us = self.total_duration_us_by_op[op_type] / divisor
            ops.append({"op_type": op_type, "avg_time_us": float(format_latency_value(avg_time_us))})
        return ops


def find_optional_profile_csv(profile_root: Path, filename: str) -> Path | None:
    matches = sorted(path for path in profile_root.rglob(filename) if path.is_file())
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def find_latest_op_statistic_csv(profile_root: Path) -> Path | None:
    matches = sorted(path for path in profile_root.rglob("op_statistic_*.csv") if path.is_file())
    plain_path = find_optional_profile_csv(profile_root, "op_statistic.csv")
    if plain_path is not None:
        matches.append(plain_path)
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def parse_op_statistic_csv(
    csv_path: Path,
    *,
    active_count: int | None = None,
    verbose: bool = False,
    stderr: TextIO = sys.stderr,
) -> ParsedProfileCsvRows:
    if active_count is not None and active_count <= 0:
        raise ValueError(f"active_count must be > 0 when parsing {csv_path}")
    parsed = _read_op_statistic_rows(csv_path)
    if not parsed.rows:
        raise ValueError(f"No rows found in {csv_path}")
    step_count = _resolve_op_statistic_step_count(
        parsed.positive_counts,
        fallback_active_count=active_count,
        csv_path=csv_path,
        verbose=verbose,
        stderr=stderr,
    )
    ops = _op_statistic_perf_rows(parsed.rows, step_count)
    if verbose:
        print(
            f"[metrics] op_statistic.csv {csv_path}: {len(parsed.rows)} ops, "
            f"total_time_us={parsed.total_time_us}",
            file=stderr,
        )
    return ParsedProfileCsvRows(
        source_path=csv_path,
        ops=ops,
        total_time_us=parsed.total_time_us,
        total_op_avg_time_us=sum(float(row["avg_time_us"]) for row in ops),
    )


def _read_op_statistic_rows(csv_path: Path) -> _OpStatisticParseData:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        _require_csv_columns(csv_path, fieldnames, ("OP Type", "Avg Time(us)", "Total Time(us)"))
        rows: list[OpStatisticCsvRow] = []
        total_time_us, positive_counts = 0.0, []
        for row in reader:
            parsed_row, count = _parse_op_statistic_row(row, csv_path, has_count="Count" in fieldnames)
            rows.append(parsed_row)
            total_time_us += parsed_row.total_time_us
            if count is not None and count > 0:
                positive_counts.append(count)
    return _OpStatisticParseData(rows, total_time_us, positive_counts)


def _require_csv_columns(csv_path: Path, fieldnames: list[str], columns: tuple[str, ...]) -> None:
    for column in columns:
        if column not in fieldnames:
            raise ValueError(f"Missing required column '{column}' in {csv_path}")


def _parse_op_statistic_row(
    row: dict[str, str | None],
    csv_path: Path,
    *,
    has_count: bool,
) -> tuple[OpStatisticCsvRow, int | None]:
    op_type = (row.get("OP Type") or "").strip()
    if not op_type:
        raise ValueError(f"Empty 'OP Type' value in {csv_path}")
    parsed = OpStatisticCsvRow(
        op_type=op_type,
        native_avg_time_us=_parse_float_field(row.get("Avg Time(us)"), "Avg Time(us)", csv_path),
        total_time_us=_parse_float_field(row.get("Total Time(us)"), "Total Time(us)", csv_path),
    )
    count = _parse_int_field(row.get("Count"), "Count", csv_path) if has_count else None
    return parsed, count


def _op_statistic_perf_rows(rows: list[OpStatisticCsvRow], step_count: int | None) -> list[PerfOpRow]:
    ops: list[PerfOpRow] = []
    for row in rows:
        average = row.native_avg_time_us
        if step_count is not None:
            average = float(format_latency_value(row.total_time_us / step_count))
        ops.append({"op_type": row.op_type, "avg_time_us": average})
    return ops


def parse_kernel_details_csv(
    csv_path: Path,
    *,
    active_count: int,
    verbose: bool = False,
    stderr: TextIO = sys.stderr,
) -> ParsedProfileCsvRows:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "Name" not in fieldnames:
            raise ValueError(f"Missing required column 'Name' in {csv_path}")
        if "Duration(us)" not in fieldnames:
            raise ValueError(f"Missing required column 'Duration(us)' in {csv_path}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"Profiler kernel details are empty: {csv_path}")

    aggregation = KernelDetailsAggregation()
    has_step_id_column = "Step Id" in fieldnames
    for row in rows:
        op_name = (row.get("Name") or "").strip()
        if not op_name:
            raise ValueError(f"Encountered empty operator name in {csv_path}")
        duration = _parse_float_field(row.get("Duration(us)"), "Duration(us)", csv_path)
        step_id = (row.get("Step Id") or "").strip() if has_step_id_column else None
        aggregation.record(op_type=op_name, duration_us=duration, step_id=step_id or None)

    # Step Id is the authoritative per-iteration boundary when present. The
    # benchmark's active_count is only a fallback for older exports that omit it.
    divisor = aggregation.observed_step_count if aggregation.observed_step_count else active_count
    ops = aggregation.build_ops(divisor=divisor)

    if verbose:
        if aggregation.observed_step_count and aggregation.observed_step_count != active_count:
            print(
                f"[metrics] kernel_details.csv {csv_path}: observed_step_count={aggregation.observed_step_count} "
                f"differs from active_count={active_count}",
                file=stderr,
            )
        print(
            f"[metrics] kernel_details.csv {csv_path}: "
            f"{len(ops)} unique ops, total_time_us={aggregation.total_time_us}",
            file=stderr,
        )
    return ParsedProfileCsvRows(
        source_path=csv_path,
        ops=ops,
        total_time_us=aggregation.total_time_us,
        total_op_avg_time_us=sum(float(row["avg_time_us"]) for row in ops),
    )


def resolve_perf_metrics(
    ops: list[PerfOpRow],
    kernel_names: list[str],
    *,
    total_op_avg_time_us: float | None = None,
    verbose: bool = False,
    stderr: TextIO = sys.stderr,
) -> PerfMetrics:
    matched_op_types = _resolve_matching_op_types(ops, kernel_names)
    matched_avg_times = [
        float(row["avg_time_us"]) for row in ops if str(row["op_type"]) in matched_op_types
    ]
    if verbose and not matched_avg_times:
        op_types = [str(row["op_type"]) for row in ops]
        print(
            f"[metrics] no resolved kernels matched profiler operators. "
            f"resolved_kernels={kernel_names}, profiler_ops={op_types}",
            file=stderr,
        )
    metrics: PerfMetrics = {
        "kernel_avg_time_us": sum(matched_avg_times) if matched_avg_times else None,
        "ops": [
            {
                "op_type": row["op_type"],
                "avg_time_us": row["avg_time_us"],
            }
            for row in ops
        ],
    }
    if total_op_avg_time_us is not None:
        metrics["total_op_avg_time_us"] = total_op_avg_time_us
    return metrics


def _resolve_matching_op_types(ops: list[PerfOpRow], kernel_names: list[str]) -> set[str]:
    op_types = {str(row["op_type"]) for row in ops}
    matched_op_types: set[str] = set()
    for kernel_name in kernel_names:
        if kernel_name in op_types:
            matched_op_types.add(kernel_name)
            continue
        alias = f"{kernel_name}_kernel"
        if alias in op_types:
            matched_op_types.add(alias)
    return matched_op_types


def _resolve_op_statistic_step_count(
    positive_counts: list[int],
    *,
    fallback_active_count: int | None,
    csv_path: Path,
    verbose: bool,
    stderr: TextIO,
) -> int | None:
    """Resolve the per-step divisor for op_statistic fallback rows.

    op_statistic.csv has no Step Id column, so we cannot directly count active
    profiler iterations the way kernel_details.csv can. Instead we infer the
    step multiplicity from the per-op Count values:

    - each Count is the number of launches observed for one op in the export
    - elementwise/helper ops often run N times per benchmark step
    - the greatest common divisor across positive Count values is therefore the
      best proxy for the number of exported active steps

    For the current profiler outputs we want the benchmark contract to remain
    authoritative when it is available, so active_count is the primary divisor
    and the inferred GCD is used as a consistency check plus fallback.
    """
    inferred_step_count: int | None = None
    if positive_counts:
        inferred_step_count = positive_counts[0]
        for count in positive_counts[1:]:
            inferred_step_count = math.gcd(inferred_step_count, count)
        if inferred_step_count <= 1:
            inferred_step_count = None

    if fallback_active_count is not None:
        if verbose and inferred_step_count is not None and inferred_step_count != fallback_active_count:
            print(
                f"[metrics] op_statistic.csv {csv_path}: inferred_step_count={inferred_step_count} "
                f"differs from active_count={fallback_active_count}; "
                f"using active_count={fallback_active_count} as the benchmark-provided step proxy",
                file=stderr,
            )
        if verbose and inferred_step_count is None:
            print(
                f"[metrics] op_statistic.csv {csv_path}: could not infer step count from Count values; "
                f"using active_count={fallback_active_count}",
                file=stderr,
            )
        return fallback_active_count

    return inferred_step_count


def _parse_float_field(raw_value: object, field_name: str, csv_path: Path) -> float:
    stripped = "" if raw_value is None else str(raw_value).strip()
    if not stripped:
        raise ValueError(f"Empty '{field_name}' value in {csv_path}")
    return float(stripped)


def _parse_int_field(raw_value: object, field_name: str, csv_path: Path) -> int:
    stripped = "" if raw_value is None else str(raw_value).strip()
    if not stripped:
        raise ValueError(f"Empty '{field_name}' value in {csv_path}")
    return int(stripped)
