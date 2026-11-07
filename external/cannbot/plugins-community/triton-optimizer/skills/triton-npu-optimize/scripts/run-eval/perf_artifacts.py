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

import json
import math
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TextIO, TypedDict, Union, cast


class PerfOpRow(TypedDict):
    op_type: str
    avg_time_us: float


class PerfMetricsRequired(TypedDict):
    kernel_avg_time_us: float | None
    ops: list[PerfOpRow]


class PerfMetrics(PerfMetricsRequired, total=False):
    total_op_avg_time_us: float | None


@dataclass(frozen=True)
class PerfCaseRecord:
    case_label: str
    kernel_names: list[str]
    kernel_source: str
    metrics: PerfMetrics | None = None
    error_message: str | None = None
    case_wall_clock_seconds: float | None = None
    bench_mode: str | None = None


ComparisonMode = Literal["latency", "total-op"]
MetricSource = Literal["auto", "kernel", "total-op", "all"]


@dataclass(frozen=True)
class PerfEntry:
    display_value: str
    numeric_value: float
    comparison_mode: ComparisonMode


@dataclass(frozen=True)
class PerfParseOutcome:
    entries: dict[str, PerfEntry]
    skipped_latency_errors: dict[str, str]


@dataclass(frozen=True)
class PerfPairOutcome:
    baseline_entries: dict[str, PerfEntry]
    compare_entries: dict[str, PerfEntry]
    skipped_latency_errors: dict[str, str]


@dataclass(frozen=True)
class MetricSourceSectionResult:
    metric_source: str
    rendered_output: str


@dataclass(frozen=True)
class _ComparisonData:
    pair_outcome: PerfPairOutcome
    baseline_outcome: PerfParseOutcome
    compare_outcome: PerfParseOutcome
    baseline: dict[str, float]
    compare: dict[str, float]
    comparable_ids: tuple[str, ...]
    invalid_metric_errors: dict[str, str]


class PerfValueMap(dict[str, float]):
    def __init__(
        self,
        values: dict[str, float],
        *,
        comparison_modes: dict[str, ComparisonMode],
    ) -> None:
        super().__init__(values)
        self.comparison_modes = comparison_modes


RequiredLatencyIds = Union[Collection[str], dict[str, PerfEntry], PerfValueMap]
PerfPairValues = tuple[dict[str, float], dict[str, float], dict[str, ComparisonMode]]
RequiredLatencyIdResolution = tuple[set[str], dict[str, ComparisonMode]]
PerfMetricSummary = tuple[float | None, float | None]


def _read_bench_mode_from_jsonl(path: Path) -> str | None:
    """Return the bench_mode from the first non-empty JSONL record.

    Also verifies that all records in the file have the same bench_mode.
    A mix of modes within one file is treated as corrupt data.
    """
    try:
        with path.open(encoding="utf-8") as handle:
            return _read_consistent_bench_mode(path, handle)
    except (json.JSONDecodeError, OSError):
        return None


def _read_consistent_bench_mode(path: Path, handle: TextIO) -> str | None:
    mode: str | None = None
    for line_no, line in enumerate(handle, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        record_mode = json.loads(stripped).get("bench_mode")
        if mode is None:
            mode = record_mode
            continue
        if mode != record_mode:
            raise ValueError(
                f"{path}: mixed bench_mode in same file: line {line_no} has bench_mode={record_mode}, "
                f"but earlier records have bench_mode={mode}"
            )
    return mode


def _check_cross_mode(baseline_perf: Path, compare_perf: Path) -> None:
    try:
        baseline_mode = _read_bench_mode_from_jsonl(baseline_perf)
        compare_mode = _read_bench_mode_from_jsonl(compare_perf)
    except ValueError as exc:
        raise ValueError(f"cannot compare results: {exc}") from exc
    if baseline_mode is None and compare_mode is None:
        return
    if baseline_mode is None:
        raise ValueError(
            f"cannot compare results: candidate has bench_mode={compare_mode} "
            f"but baseline has no bench_mode (pre-perf-counter record)"
        )
    if compare_mode is None:
        raise ValueError(
            f"cannot compare results: baseline has bench_mode={baseline_mode} "
            f"but candidate has no bench_mode (pre-perf-counter record)"
        )
    if baseline_mode != compare_mode:
        raise ValueError(
            f"cannot compare results from different bench modes: "
            f"baseline={baseline_mode}, candidate={compare_mode}"
        )


def compare_perf_files(
    baseline_perf: Path,
    compare_perf: Path,
    *,
    skip_latency_errors: bool = False,
    metric_source: MetricSource = "auto",
) -> int:
    if metric_source == "all":
        return _compare_perf_files_all(
            baseline_perf,
            compare_perf,
            skip_latency_errors=skip_latency_errors,
        )
    try:
        data = _prepare_comparison_data(
            baseline_perf,
            compare_perf,
            skip_latency_errors=skip_latency_errors,
            metric_source=metric_source,
        )
    except ValueError as exc:
        print(f"FAIL: {exc}")
        return 1
    rendered, succeeded = _render_comparison(data, metric_source=metric_source)
    print(rendered, end="")
    return 0 if succeeded else 1


def _prepare_comparison_data(
    baseline_perf: Path,
    compare_perf: Path,
    *,
    skip_latency_errors: bool,
    metric_source: MetricSource,
) -> _ComparisonData:
    _check_cross_mode(baseline_perf, compare_perf)
    baseline_outcome, compare_outcome = _parse_comparison_outcomes(
        baseline_perf, compare_perf, skip_latency_errors, metric_source
    )
    pair_outcome = _normalize_perf_pair_for_comparison(
        baseline_perf=baseline_perf,
        compare_perf=compare_perf,
        baseline_entries=baseline_outcome.entries,
        compare_entries=compare_outcome.entries,
        metric_source=metric_source,
        tolerate_latency_errors=skip_latency_errors,
    )
    return _comparison_data_from_pair(
        pair_outcome,
        baseline_outcome=baseline_outcome,
        compare_outcome=compare_outcome,
        baseline_perf=baseline_perf,
        compare_perf=compare_perf,
        skip_latency_errors=skip_latency_errors,
    )


def _parse_comparison_outcomes(
    baseline_perf: Path,
    compare_perf: Path,
    skip_latency_errors: bool,
    metric_source: MetricSource,
) -> tuple[PerfParseOutcome, PerfParseOutcome]:
    baseline_outcome = _parse_perf_entries_for_comparison(
        baseline_perf,
        skip_latency_errors=skip_latency_errors,
        metric_source=metric_source,
    )
    compare_outcome = _parse_required_perf_entries_for_comparison(
        compare_perf,
        baseline_outcome.entries,
        skip_latency_errors=skip_latency_errors,
        metric_source=metric_source,
    )
    return baseline_outcome, compare_outcome


def _comparison_data_from_pair(pair_outcome: PerfPairOutcome, **values: object) -> _ComparisonData:
    baseline_outcome = cast(PerfParseOutcome, values["baseline_outcome"])
    compare_outcome = cast(PerfParseOutcome, values["compare_outcome"])
    baseline_perf = cast(Path, values["baseline_perf"])
    compare_perf = cast(Path, values["compare_perf"])
    skip_latency_errors = bool(values["skip_latency_errors"])
    comparable_ids = tuple(
        sorted(set(pair_outcome.baseline_entries) & set(pair_outcome.compare_entries))
    )
    baseline, compare = _comparison_numeric_values(pair_outcome, comparable_ids)
    invalid_errors = _collect_invalid_metric_errors(
        baseline,
        compare,
        baseline_perf=baseline_perf,
        compare_perf=compare_perf,
    )
    if invalid_errors and not skip_latency_errors:
        first_id = sorted(invalid_errors)[0]
        raise ValueError(invalid_errors.get(first_id, "invalid performance metric"))
    if invalid_errors:
        comparable_ids = tuple(item for item in comparable_ids if item not in invalid_errors)
        baseline, compare = _comparison_numeric_values(pair_outcome, comparable_ids)
    return _ComparisonData(
        pair_outcome=pair_outcome,
        baseline_outcome=baseline_outcome,
        compare_outcome=compare_outcome,
        baseline=baseline,
        compare=compare,
        comparable_ids=comparable_ids,
        invalid_metric_errors=invalid_errors,
    )


def _comparison_numeric_values(
    pair_outcome: PerfPairOutcome,
    comparable_ids: tuple[str, ...],
) -> tuple[dict[str, float], dict[str, float]]:
    baseline = {
        item: pair_outcome.baseline_entries[item].numeric_value for item in comparable_ids
    }
    compare = {
        item: pair_outcome.compare_entries[item].numeric_value for item in comparable_ids
    }
    return baseline, compare


def _comparison_skipped_errors(data: _ComparisonData) -> dict[str, str]:
    return {
        **data.baseline_outcome.skipped_latency_errors,
        **data.compare_outcome.skipped_latency_errors,
        **data.pair_outcome.skipped_latency_errors,
        **data.invalid_metric_errors,
    }


def _render_comparison(
    data: _ComparisonData,
    *,
    metric_source: MetricSource,
    section_name: str | None = None,
) -> tuple[str, bool]:
    lines = ["Perf comparison:"]
    if section_name is not None:
        lines.insert(0, f"Metric source section: {section_name}")
    for latency_id in data.comparable_ids:
        baseline_entry = data.pair_outcome.baseline_entries[latency_id]
        compare_entry = data.pair_outcome.compare_entries[latency_id]
        lines.append(
            f"{latency_id}: baseline={baseline_entry.display_value}, "
            f"compare={compare_entry.display_value}, "
            f"delta={_format_delta_percent(data.baseline[latency_id], data.compare[latency_id])}"
        )
    avg_improvement, geomean_speedup = _summarize_perf_metrics(
        data.baseline, data.compare
    )
    lines.append(f"Avg improvement: {_format_improvement_percent(avg_improvement)}")
    lines.append(f"Geomean speedup: {_format_speedup(geomean_speedup)}")
    compared_entries = {
        item: data.pair_outcome.baseline_entries[item] for item in data.comparable_ids
    }
    lines.append(
        f"Metric source: {_summarize_metric_source(compared_entries, metric_source=metric_source)}"
    )
    skipped_errors = _comparison_skipped_errors(data)
    if skipped_errors:
        lines.append(f"FAIL: skipped {len(skipped_errors)} latency entries due to latency errors")
        lines.extend(skipped_errors.get(item, "") for item in sorted(skipped_errors))
        return "\n".join(lines) + "\n", False
    lines.append(f"PASS: compared {len(data.baseline)} latency entries")
    return "\n".join(lines) + "\n", True


def _compare_perf_files_all(
    baseline_perf: Path,
    compare_perf: Path,
    *,
    skip_latency_errors: bool,
) -> int:
    try:
        _check_cross_mode(baseline_perf, compare_perf)
    except ValueError as exc:
        print(f"FAIL: {exc}")
        return 1
    section_results: list[MetricSourceSectionResult] = []
    for section_metric_source in ("kernel", "total-op"):
        section_results.append(
            _compare_perf_section(
                baseline_perf,
                compare_perf,
                section_metric_source,
                skip_latency_errors=skip_latency_errors,
            )
        )
    for section_result in section_results:
        print(section_result.rendered_output, end="")
    all_passed = section_results and all("\nPASS:" in item.rendered_output for item in section_results)
    return 0 if all_passed else 1


def _compare_perf_section(
    baseline_perf: Path,
    compare_perf: Path,
    metric_source: str,
    *,
    skip_latency_errors: bool,
) -> MetricSourceSectionResult:
    typed_source = cast(MetricSource, metric_source)
    try:
        data = _prepare_comparison_data(
            baseline_perf,
            compare_perf,
            skip_latency_errors=skip_latency_errors,
            metric_source=typed_source,
        )
    except ValueError as exc:
        return MetricSourceSectionResult(
            metric_source=metric_source,
            rendered_output=f"Metric source section: {metric_source}\nFAIL: {exc}\n",
        )
    rendered, _succeeded = _render_comparison(
        data, metric_source=typed_source, section_name=metric_source
    )
    return MetricSourceSectionResult(metric_source=metric_source, rendered_output=rendered)


def parse_perf_file(path: Path) -> dict[str, float]:
    return _parse_perf_file(path)


def parse_required_perf_file(path: Path, required_latency_ids: RequiredLatencyIds) -> dict[str, float]:
    return _parse_required_perf_file(path, required_latency_ids)


def parse_perf_file_for_metric_source(
    path: Path,
    *,
    metric_source: MetricSource = "auto",
) -> dict[str, float]:
    if metric_source == "all":
        raise ValueError("parse_perf_file_for_metric_source does not support metric_source='all'")
    outcome = _parse_perf_entries_for_comparison(
        path,
        skip_latency_errors=False,
        metric_source=metric_source,
    )
    return PerfValueMap(
        {latency_id: entry.numeric_value for latency_id, entry in outcome.entries.items()},
        comparison_modes={
            latency_id: entry.comparison_mode for latency_id, entry in outcome.entries.items()
        },
    )


def parse_required_perf_file_for_metric_source(
    path: Path,
    required_latency_ids: RequiredLatencyIds,
    *,
    metric_source: MetricSource = "auto",
) -> dict[str, float]:
    if metric_source == "all":
        raise ValueError(
            "parse_required_perf_file_for_metric_source does not support metric_source='all'"
        )
    outcome = _parse_required_perf_entries_for_comparison(
        path,
        required_latency_ids,
        skip_latency_errors=False,
        metric_source=metric_source,
    )
    return PerfValueMap(
        {latency_id: entry.numeric_value for latency_id, entry in outcome.entries.items()},
        comparison_modes={
            latency_id: entry.comparison_mode for latency_id, entry in outcome.entries.items()
        },
    )


def parse_perf_pair_for_comparison(
    baseline_perf: Path,
    compare_perf: Path,
    *,
    metric_source: MetricSource = "auto",
) -> PerfPairValues:
    if metric_source == "all":
        raise ValueError("parse_perf_pair_for_comparison does not support metric_source='all'")
    _check_cross_mode(baseline_perf, compare_perf)
    baseline_outcome = _parse_perf_entries_for_comparison(
        baseline_perf,
        skip_latency_errors=False,
        metric_source=metric_source,
    )
    compare_outcome = _parse_required_perf_entries_for_comparison(
        compare_perf,
        baseline_outcome.entries,
        skip_latency_errors=False,
        metric_source=metric_source,
    )
    pair_outcome = _normalize_perf_pair_for_comparison(
        baseline_perf=baseline_perf,
        compare_perf=compare_perf,
        baseline_entries=baseline_outcome.entries,
        compare_entries=compare_outcome.entries,
        metric_source=metric_source,
        tolerate_latency_errors=False,
    )
    baseline_values = {
        latency_id: entry.numeric_value
        for latency_id, entry in pair_outcome.baseline_entries.items()
    }
    compare_values = {
        latency_id: entry.numeric_value
        for latency_id, entry in pair_outcome.compare_entries.items()
    }
    invalid_metric_errors = _collect_invalid_metric_errors(
        baseline_values,
        compare_values,
        baseline_perf=baseline_perf,
        compare_perf=compare_perf,
    )
    if invalid_metric_errors:
        first_latency_id = sorted(invalid_metric_errors)[0]
        raise ValueError(invalid_metric_errors[first_latency_id])
    comparison_modes: dict[str, ComparisonMode] = {
        latency_id: entry.comparison_mode
        for latency_id, entry in pair_outcome.baseline_entries.items()
    }
    return baseline_values, compare_values, comparison_modes


def perf_output_path(operator_file: Path) -> Path:
    return operator_file.parent / f"{operator_file.stem}_perf.txt"


def write_perf_lines(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def format_latency_value(value: float) -> str:
    rendered = f"{value:.6f}".rstrip("0").rstrip(".")
    if "." not in rendered:
        rendered += ".0"
    return rendered


def render_perf_case_records(records: list[PerfCaseRecord], **values: object) -> list[str]:
    rendered: list[str] = []
    for record in records:
        rendered.extend(
            render_perf_case_record(
                record,
                **values,
            )
        )
    return rendered


def render_perf_case_record(record: PerfCaseRecord, **values: object) -> list[str]:
    latency_prefix = str(values["latency_prefix"])
    raw_prefix = str(values["raw_prefix"])
    resolved_kernels_prefix = str(values["resolved_kernels_prefix"])
    kernel_source_prefix = str(values["kernel_source_prefix"])
    latency_error_prefix = str(values["latency_error_prefix"])
    missing_kernel_match_error = str(values["missing_kernel_match_error"])
    elapsed_id_prefix = str(values.get("elapsed_id_prefix", ""))
    case_label = record.case_label
    lines = [f"{latency_prefix}-{case_label}: {_format_case_latency_value(record)}"]
    if record.case_wall_clock_seconds is not None:
        elapsed_id = f"{elapsed_id_prefix}-{case_label}" if elapsed_id_prefix else case_label
        lines.append(f"# elapsed-seconds-{elapsed_id}: {record.case_wall_clock_seconds:.6f}")
    if record.metrics is not None:
        raw_payload = json.dumps({"ops": record.metrics["ops"]}, separators=(",", ":"))
        lines.append(f"# {raw_prefix}-{case_label}: {raw_payload}")
        if record.metrics["kernel_avg_time_us"] is None:
            lines.append(f"# {latency_error_prefix}-{case_label}: {missing_kernel_match_error}")
    if record.error_message is not None:
        lines.append(f"# {latency_error_prefix}-{case_label}: {record.error_message}")
    lines.append(f"# {resolved_kernels_prefix}-{case_label}: {','.join(record.kernel_names)}")
    lines.append(f"# {kernel_source_prefix}-{case_label}: {record.kernel_source}")
    return lines


def render_perf_case_record_jsonl(
    record: PerfCaseRecord,
    *,
    missing_kernel_match_error: str | None = None,
) -> str:
    metrics = record.metrics
    kernel_avg_time_us: float | None = None
    ops: list[PerfOpRow] | None = None
    total_op_avg_time_us: float | None = None
    if metrics is not None:
        kernel_avg_time_us = metrics["kernel_avg_time_us"]
        ops = metrics["ops"]
        explicit_total_op_avg_time_us = metrics.get("total_op_avg_time_us")
        if explicit_total_op_avg_time_us is not None:
            total_op_avg_time_us = explicit_total_op_avg_time_us
        elif ops:
            total_op_avg_time_us = sum(op["avg_time_us"] for op in ops)
        elif record.bench_mode == "perf-counter":
            total_op_avg_time_us = kernel_avg_time_us
        else:
            total_op_avg_time_us = 0.0
    error_message = record.error_message
    if error_message is None and metrics is not None and kernel_avg_time_us is None:
        error_message = missing_kernel_match_error
    payload: dict[str, object] = {
        "case_label": record.case_label,
        "kernel_names": record.kernel_names,
        "kernel_source": record.kernel_source,
        "kernel_avg_time_us": kernel_avg_time_us,
        "ops": ops,
        "total_op_avg_time_us": total_op_avg_time_us,
        "error_message": error_message,
        "case_wall_clock_seconds": record.case_wall_clock_seconds,
        "bench_mode": record.bench_mode,
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def render_perf_case_records_jsonl(
    records: list[PerfCaseRecord],
    *,
    missing_kernel_match_error: str | None = None,
) -> list[str]:
    return [
        render_perf_case_record_jsonl(
            record,
            missing_kernel_match_error=missing_kernel_match_error,
        )
        for record in records
    ]


def _format_case_latency_value(record: PerfCaseRecord) -> str:
    if record.metrics is None or record.metrics["kernel_avg_time_us"] is None:
        return "NA"
    return format_latency_value(record.metrics["kernel_avg_time_us"])


def _parse_perf_file(path: Path) -> dict[str, float]:
    entries = _parse_perf_entries(path)
    return PerfValueMap(
        {latency_id: entry.numeric_value for latency_id, entry in entries.items()},
        comparison_modes={latency_id: entry.comparison_mode for latency_id, entry in entries.items()},
    )


def _parse_perf_entries(path: Path) -> dict[str, PerfEntry]:
    return _parse_perf_entries_strict(path).entries


def _parse_perf_entries_strict(path: Path) -> PerfParseOutcome:
    return _parse_perf_entries_impl(path, tolerate_latency_errors=False)


def _parse_perf_entries_for_comparison(
    path: Path,
    *,
    skip_latency_errors: bool,
    metric_source: MetricSource,
) -> PerfParseOutcome:
    return _parse_perf_entries_impl(
        path,
        tolerate_latency_errors=skip_latency_errors,
        metric_source=metric_source,
    )


def _parse_perf_entries_impl(
    path: Path,
    *,
    tolerate_latency_errors: bool,
    metric_source: MetricSource = "auto",
) -> PerfParseOutcome:
    lines = path.read_text(encoding="utf-8").splitlines()
    if _is_jsonl_perf_file(lines):
        return _parse_perf_entries_from_jsonl(
            path,
            lines,
            tolerate_latency_errors=tolerate_latency_errors,
            metric_source=metric_source,
        )
    return _parse_text_perf_entries(path, lines, tolerate_latency_errors, metric_source)


def _is_jsonl_perf_file(lines: list[str]) -> bool:
    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped:
            return stripped.startswith("{")
    return False


def _parse_text_perf_entries(
    path: Path,
    lines: list[str],
    tolerate_latency_errors: bool,
    metric_source: MetricSource,
) -> PerfParseOutcome:
    raw_totals = _parse_raw_op_statistic_totals(path, lines)
    latency_errors = _parse_latency_errors(path, lines)
    entries: dict[str, PerfEntry] = {}
    skipped_latency_errors: dict[str, str] = {}
    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"{path}:{line_no} is not a 'latency-<id>: <value>' line")
        key, value = line.split(":", 1)
        latency_id = key.strip()
        if not latency_id.startswith("latency-"):
            raise ValueError(f"{path}:{line_no} does not start with 'latency-'")
        value_text = value.strip()
        if latency_id in entries:
            raise ValueError(f"{path}:{line_no} duplicates latency id '{latency_id}'")
        uncomparable_error = _get_uncomparable_latency_error(
            path, line_no, latency_id, latency_errors
        )
        if uncomparable_error is not None:
            if tolerate_latency_errors:
                skipped_latency_errors[latency_id] = uncomparable_error
                continue
            raise ValueError(uncomparable_error)
        entries[latency_id] = _build_perf_entry_for_source(
            path=path,
            line_no=line_no,
            latency_id=latency_id,
            value_text=value_text,
            raw_totals=raw_totals,
            metric_source=metric_source,
        )
    if not entries and not skipped_latency_errors:
        raise ValueError(f"{path} did not contain any latency-<id>: <value> entries")
    return PerfParseOutcome(entries=entries, skipped_latency_errors=skipped_latency_errors)


def _parse_required_perf_file(path: Path, required_latency_ids: RequiredLatencyIds) -> dict[str, float]:
    entries = _parse_required_perf_entries(path, required_latency_ids)
    values: dict[str, float] = {}
    for latency_id, entry in entries.items():
        values[latency_id] = entry.numeric_value
    return values


def _parse_required_perf_entries(
    path: Path, required_latency_ids: RequiredLatencyIds
) -> dict[str, PerfEntry]:
    return _parse_required_perf_entries_strict(path, required_latency_ids).entries


def _parse_required_perf_entries_strict(
    path: Path, required_latency_ids: RequiredLatencyIds
) -> PerfParseOutcome:
    return _parse_required_perf_entries_impl(
        path, required_latency_ids, tolerate_latency_errors=False
    )


def _parse_required_perf_entries_for_comparison(
    path: Path,
    required_latency_ids: RequiredLatencyIds,
    *,
    skip_latency_errors: bool,
    metric_source: MetricSource,
) -> PerfParseOutcome:
    return _parse_required_perf_entries_impl(
        path,
        required_latency_ids,
        tolerate_latency_errors=skip_latency_errors,
        metric_source=metric_source,
    )


def _parse_required_perf_entries_impl(
    path: Path,
    required_latency_ids: RequiredLatencyIds,
    *,
    tolerate_latency_errors: bool,
    metric_source: MetricSource = "auto",
) -> PerfParseOutcome:
    required_ids, comparison_modes = _resolve_required_latency_requirements(required_latency_ids)
    if not required_ids:
        return PerfParseOutcome(entries={}, skipped_latency_errors={})

    lines = path.read_text(encoding="utf-8").splitlines()
    if _is_jsonl_perf_file(lines):
        outcome = _parse_perf_entries_from_jsonl(
            path,
            lines,
            tolerate_latency_errors=tolerate_latency_errors,
            metric_source=metric_source,
            required_ids=required_ids,
            comparison_modes=comparison_modes,
        )
        _raise_for_missing_required_ids(path, required_ids, outcome)
        return outcome
    outcome = _parse_required_text_perf_entries(
        path,
        lines,
        required_ids=required_ids,
        comparison_modes=comparison_modes,
        tolerate_latency_errors=tolerate_latency_errors,
        metric_source=metric_source,
    )
    _raise_for_missing_required_ids(path, required_ids, outcome)
    return outcome


def _parse_required_text_perf_entries(path: Path, lines: list[str], **values: object) -> PerfParseOutcome:
    required_ids = cast(set[str], values["required_ids"])
    comparison_modes = cast(dict[str, ComparisonMode], values["comparison_modes"])
    tolerate_latency_errors = bool(values["tolerate_latency_errors"])
    metric_source = cast(MetricSource, values["metric_source"])
    raw_totals = _parse_raw_op_statistic_totals(path, lines)
    latency_errors = _parse_latency_errors(path, lines)
    entries: dict[str, PerfEntry] = {}
    skipped_latency_errors: dict[str, str] = {}
    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        latency_id = key.strip()
        matched_latency_id = _resolve_required_latency_id_match(latency_id, required_ids)
        if matched_latency_id is None:
            continue
        value_text = value.strip()
        if matched_latency_id in entries:
            raise ValueError(f"{path}:{line_no} duplicates latency id '{matched_latency_id}'")
        uncomparable_error = _get_uncomparable_latency_error(
            path, line_no, latency_id, latency_errors
        )
        if uncomparable_error is not None:
            if tolerate_latency_errors:
                skipped_latency_errors[matched_latency_id] = uncomparable_error
                continue
            raise ValueError(uncomparable_error)
        effective_metric_source = (
            "total-op"
            if metric_source == "auto" and comparison_modes[matched_latency_id] == "total-op"
            else metric_source
        )
        entries[matched_latency_id] = _build_perf_entry_for_source(
            path=path,
            line_no=line_no,
            latency_id=latency_id,
            value_text=value_text,
            raw_totals=raw_totals,
            metric_source=effective_metric_source,
        )

    return PerfParseOutcome(entries=entries, skipped_latency_errors=skipped_latency_errors)


def _raise_for_missing_required_ids(
    path: Path,
    required_ids: set[str],
    outcome: PerfParseOutcome,
) -> None:
    missing_ids = sorted(required_ids - set(outcome.entries) - set(outcome.skipped_latency_errors))
    if missing_ids:
        raise ValueError(f"{path} is missing required latency ids: {missing_ids}")


def _parse_perf_entries_from_jsonl(path: Path, lines: list[str], **values: object) -> PerfParseOutcome:
    tolerate_latency_errors = bool(values["tolerate_latency_errors"])
    metric_source = cast(MetricSource, values.get("metric_source", "auto"))
    required_ids = cast(set[str] | None, values.get("required_ids"))
    comparison_modes = cast(dict[str, ComparisonMode] | None, values.get("comparison_modes"))
    entries: dict[str, PerfEntry] = {}
    skipped_latency_errors: dict[str, str] = {}
    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        record = _parse_jsonl_record(path, line_no, line)
        latency_id, matched_id = _resolve_jsonl_latency_id(record, required_ids)
        if matched_id is None:
            continue
        if matched_id in entries:
            raise ValueError(f"{path}:{line_no} duplicates latency id '{matched_id}'")
        error = _jsonl_comparison_error(path, line_no, latency_id, record)
        if error is not None:
            if tolerate_latency_errors:
                skipped_latency_errors[matched_id] = error
                continue
            raise ValueError(error)
        entries[matched_id] = _jsonl_perf_entry(
            path, line_no, latency_id, record,
            metric_source=_jsonl_metric_source(metric_source, comparison_modes, matched_id),
        )
    if not entries and not skipped_latency_errors:
        if required_ids is not None:
            return PerfParseOutcome(entries={}, skipped_latency_errors={})
        raise ValueError(f"{path} did not contain any latency-<id>: <value> entries")
    return PerfParseOutcome(entries=entries, skipped_latency_errors=skipped_latency_errors)


def _parse_jsonl_record(path: Path, line_no: int, line: str) -> dict[str, object]:
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{line_no} has invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{path}:{line_no} JSONL record must be an object")
    return cast(dict[str, object], parsed)


def _resolve_jsonl_latency_id(
    record: dict[str, object],
    required_ids: set[str] | None,
) -> tuple[str, str | None]:
    latency_id = f"latency-{record['case_label']}"
    if required_ids is None:
        return latency_id, latency_id
    return latency_id, _resolve_required_latency_id_match(latency_id, required_ids)


def _jsonl_comparison_error(
    path: Path,
    line_no: int,
    latency_id: str,
    record: dict[str, object],
) -> str | None:
    error_value = record.get("error_message")
    if error_value is not None and not isinstance(error_value, str):
        raise ValueError(f"{path}:{line_no} 'error_message' must be a string or null")
    if error_value is None or error_value.startswith("no resolved kernels matched"):
        return None
    return f"{path}:{line_no} cannot compare '{latency_id}' because 'error_message: {error_value}' is present"


def _jsonl_metric_source(
    metric_source: MetricSource,
    comparison_modes: dict[str, ComparisonMode] | None,
    matched_id: str,
) -> MetricSource:
    if metric_source == "auto" and comparison_modes is not None:
        if comparison_modes.get(matched_id) == "total-op":
            return "total-op"
    return metric_source


def _jsonl_perf_entry(
    path: Path,
    line_no: int,
    latency_id: str,
    record: dict[str, object],
    *,
    metric_source: MetricSource,
) -> PerfEntry:
    kernel_value = record.get("kernel_avg_time_us")
    total_op_value = record.get("total_op_avg_time_us")
    if kernel_value is not None:
        return _jsonl_kernel_entry(
            path, line_no, latency_id,
            kernel_value=kernel_value,
            total_op_value=total_op_value,
            metric_source=metric_source,
        )
    if total_op_value is not None:
        return _jsonl_total_op_entry(path, line_no, latency_id, total_op_value, metric_source)
    raise ValueError(f"{path}:{line_no} has no usable metric for '{latency_id}'")


def _jsonl_kernel_entry(path: Path, line_no: int, latency_id: str, **values: object) -> PerfEntry:
    kernel_value = values["kernel_value"]
    total_op_value = values["total_op_value"]
    metric_source = cast(MetricSource, values["metric_source"])
    if metric_source != "total-op":
        value = float(cast("int | float | str", kernel_value))
        return PerfEntry(format_latency_value(value), value, "latency")
    if total_op_value is None:
        raise ValueError(
            f"{path}:{line_no} requires total-op for '{latency_id}' "
            "under --metric-source total-op but total_op_avg_time_us is null"
        )
    value = float(cast("int | float | str", total_op_value))
    return PerfEntry(_format_total_op_display(value), value, "total-op")


def _jsonl_total_op_entry(
    path: Path,
    line_no: int,
    latency_id: str,
    total_op_value: object,
    metric_source: MetricSource,
) -> PerfEntry:
    if metric_source == "kernel":
        raise ValueError(
            f"{path}:{line_no} requires kernel latency for '{latency_id}' under --metric-source kernel"
        )
    value = float(cast("int | float | str", total_op_value))
    return PerfEntry(f"NA ({_format_total_op_display(value)})", value, "total-op")


def _parse_raw_op_statistic_totals(path: Path, lines: list[str]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line.startswith("# raw-op-statistic-"):
            continue
        body = line[1:].strip()
        if ":" not in body:
            raise ValueError(f"{path}:{line_no} is not a '# raw-op-statistic-<id>: <json>' line")
        key, value = body.split(":", 1)
        raw_stat_id = key.strip()
        latency_id = f"latency-{raw_stat_id.removeprefix('raw-op-statistic-')}"
        if latency_id in totals:
            raise ValueError(f"{path}:{line_no} duplicates raw-op statistic for '{latency_id}'")
        try:
            payload = json.loads(value.strip())
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} has invalid raw-op-statistic JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_no} raw-op-statistic JSON must be an object")
        payload_dict = cast(dict[str, object], payload)
        ops = payload_dict.get("ops")
        if not isinstance(ops, list):
            raise ValueError(f"{path}:{line_no} raw-op-statistic JSON is missing an 'ops' list")
        total = 0.0
        typed_ops = cast(list[object], ops)
        for op in typed_ops:
            if not isinstance(op, dict):
                raise ValueError(f"{path}:{line_no} raw-op-statistic ops entries must be objects")
            op_dict = cast(dict[str, object], op)
            avg_time_us = op_dict.get("avg_time_us")
            if not isinstance(avg_time_us, (int, float)):
                raise ValueError(
                    f"{path}:{line_no} raw-op-statistic ops entries must include numeric 'avg_time_us'"
                )
            total += float(avg_time_us)
        totals[latency_id] = total
    return totals


def _parse_latency_errors(path: Path, lines: list[str]) -> dict[str, str]:
    errors: dict[str, str] = {}
    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line.startswith("# latency-error-"):
            continue
        body = line[1:].strip()
        if ":" not in body:
            raise ValueError(f"{path}:{line_no} is not a '# latency-error-<id>: <message>' line")
        key, value = body.split(":", 1)
        error_id = key.strip()
        latency_id = f"latency-{error_id.removeprefix('latency-error-')}"
        if latency_id in errors:
            raise ValueError(f"{path}:{line_no} duplicates latency error for '{latency_id}'")
        errors[latency_id] = value.strip()
    return errors


def _resolve_required_latency_requirements(
    required_latency_ids: RequiredLatencyIds,
) -> RequiredLatencyIdResolution:
    required_ids = set(required_latency_ids)
    comparison_modes: dict[str, ComparisonMode] = {
        latency_id: "latency" for latency_id in required_ids
    }
    if isinstance(required_latency_ids, dict):
        typed_required_latency_ids = cast(object, required_latency_ids)
        for latency_id in required_ids:
            value = cast(dict[str, object], typed_required_latency_ids)[latency_id]
            if isinstance(value, PerfEntry):
                comparison_modes[latency_id] = value.comparison_mode
        return required_ids, comparison_modes
    raw_modes = (
        required_latency_ids.comparison_modes
        if isinstance(required_latency_ids, PerfValueMap)
        else None
    )
    if raw_modes is not None:
        for latency_id in required_ids:
            mode = raw_modes.get(latency_id)
            if mode in ("latency", "total-op"):
                comparison_modes[latency_id] = mode
    return required_ids, comparison_modes


def _resolve_required_latency_id_match(
    latency_id: str,
    required_ids: set[str],
) -> str | None:
    return latency_id if latency_id in required_ids else None


def _rebuild_entry_for_mode(
    path: Path,
    latency_id: str,
    *,
    target_mode: ComparisonMode,
) -> PerfEntry:
    metric_source: MetricSource = "kernel" if target_mode == "latency" else "total-op"
    return _parse_required_perf_entries_for_comparison(
        path,
        {latency_id},
        skip_latency_errors=False,
        metric_source=metric_source,
    ).entries[latency_id]


def _require_raw_total(
    path: Path,
    line_no: int,
    latency_id: str,
    raw_totals: dict[str, float],
    *,
    reason: str = "to provide total-op fallback",
) -> float:
    total = raw_totals.get(latency_id)
    if total is None:
        raise ValueError(
            f"{path}:{line_no} requires '# raw-op-statistic-{latency_id.removeprefix('latency-')}: ...' {reason}"
        )
    return total


def _build_perf_entry_for_source(*, path: Path, line_no: int, latency_id: str, **values: object) -> PerfEntry:
    value_text = str(values["value_text"])
    raw_totals = cast(dict[str, float], values["raw_totals"])
    metric_source = cast(MetricSource, values["metric_source"])
    if metric_source == "kernel":
        return _kernel_perf_entry(path, line_no, latency_id, value_text)

    if metric_source == "total-op":
        return _total_op_perf_entry(
            path,
            line_no,
            latency_id,
            value_text=value_text,
            raw_totals=raw_totals,
            reason=f"for '{latency_id}' under --metric-source total-op",
        )

    if value_text == "NA":
        return _total_op_perf_entry(path, line_no, latency_id, value_text=value_text, raw_totals=raw_totals)
    return _kernel_perf_entry(path, line_no, latency_id, value_text)


def _kernel_perf_entry(
    path: Path,
    line_no: int,
    latency_id: str,
    value_text: str,
) -> PerfEntry:
    if value_text == "NA":
        raise ValueError(
            f"{path}:{line_no} requires kernel latency for '{latency_id}' under --metric-source kernel"
        )
    return PerfEntry(
        display_value=value_text,
        numeric_value=_parse_latency_number(path, line_no, value_text),
        comparison_mode="latency",
    )


def _total_op_perf_entry(path: Path, line_no: int, latency_id: str, **values: object) -> PerfEntry:
    value_text = str(values["value_text"])
    raw_totals = cast(dict[str, float], values["raw_totals"])
    reason = str(values.get("reason", "to provide total-op fallback"))
    total_op_value = _require_raw_total(path, line_no, latency_id, raw_totals, reason=reason)
    display_value = _format_total_op_display(total_op_value)
    if value_text == "NA":
        display_value = f"NA ({display_value})"
    return PerfEntry(display_value, total_op_value, "total-op")


def _parse_latency_number(path: Path, line_no: int, value_text: str) -> float:
    try:
        return float(value_text)
    except ValueError as exc:
        raise ValueError(f"{path}:{line_no} has invalid latency value '{value_text}'") from exc


def _get_uncomparable_latency_error(
    path: Path,
    line_no: int,
    latency_id: str,
    latency_errors: dict[str, str],
) -> str | None:
    error_message = latency_errors.get(latency_id)
    if error_message is None or error_message.startswith("no resolved kernels matched"):
        return None
    return (
        f"{path}:{line_no} cannot compare '{latency_id}' because "
        f"'# latency-error-{latency_id.removeprefix('latency-')}: {error_message}' is present"
    )


def _format_total_op_display(value: float) -> str:
    return f"total-op={format_latency_value(value)}"


def _format_delta_percent(baseline: float, compare: float) -> str:
    if baseline == 0:
        if compare == 0:
            return "0.00%"
        return "inf"
    delta = ((compare - baseline) / baseline) * 100.0
    return f"{delta:.2f}%"


def _collect_invalid_metric_errors(
    baseline: dict[str, float],
    compare: dict[str, float],
    *,
    baseline_perf: Path,
    compare_perf: Path,
) -> dict[str, str]:
    errors: dict[str, str] = {}
    for latency_id in sorted(set(baseline) & set(compare)):
        baseline_value = baseline[latency_id]
        compare_value = compare[latency_id]
        if baseline_value <= 0:
            errors[latency_id] = (
                f"{baseline_perf} cannot compare '{latency_id}' because baseline timing "
                f"{baseline_value} must be > 0"
            )
            continue
        if compare_value <= 0:
            errors[latency_id] = (
                f"{compare_perf} cannot compare '{latency_id}' because compare timing "
                f"{compare_value} must be > 0"
            )
    return errors


def _normalize_perf_pair_for_comparison(**values: object) -> PerfPairOutcome:
    baseline_perf = cast(Path, values["baseline_perf"])
    compare_perf = cast(Path, values["compare_perf"])
    baseline_entries = cast(dict[str, PerfEntry], values["baseline_entries"])
    compare_entries = cast(dict[str, PerfEntry], values["compare_entries"])
    metric_source = cast(MetricSource, values["metric_source"])
    tolerate_latency_errors = bool(values["tolerate_latency_errors"])
    if metric_source != "auto":
        return PerfPairOutcome(
            baseline_entries=dict(baseline_entries),
            compare_entries=dict(compare_entries),
            skipped_latency_errors={},
        )

    normalized_baseline = dict(baseline_entries)
    normalized_compare = dict(compare_entries)
    skipped_latency_errors: dict[str, str] = {}
    shared_ids = sorted(set(normalized_baseline) & set(normalized_compare))
    for latency_id in shared_ids:
        baseline_entry = normalized_baseline[latency_id]
        compare_entry = normalized_compare[latency_id]
        if baseline_entry.comparison_mode == compare_entry.comparison_mode:
            continue
        try:
            normalized_baseline[latency_id] = _rebuild_entry_for_mode(
                baseline_perf,
                latency_id,
                target_mode="total-op",
            )
            normalized_compare[latency_id] = _rebuild_entry_for_mode(
                compare_perf,
                latency_id,
                target_mode="total-op",
            )
        except ValueError as exc:
            if not tolerate_latency_errors:
                raise
            skipped_latency_errors[latency_id] = str(exc)
            normalized_baseline.pop(latency_id, None)
            normalized_compare.pop(latency_id, None)
    return PerfPairOutcome(
        baseline_entries=normalized_baseline,
        compare_entries=normalized_compare,
        skipped_latency_errors=skipped_latency_errors,
    )


def _summarize_perf_metrics(
    baseline: dict[str, float],
    compare: dict[str, float],
) -> PerfMetricSummary:
    pairs = [(baseline[latency_id], compare[latency_id]) for latency_id in sorted(baseline)]
    if not pairs:
        return None, None
    if any(baseline_value <= 0 or compare_value <= 0 for baseline_value, compare_value in pairs):
        return None, None

    improvements = [
        (baseline_value - compare_value) / baseline_value
        for baseline_value, compare_value in pairs
    ]
    ratios = [baseline_value / compare_value for baseline_value, compare_value in pairs]
    avg_improvement = sum(improvements) / len(improvements)
    geomean_speedup = math.exp(sum(math.log(ratio) for ratio in ratios) / len(ratios))
    return avg_improvement, geomean_speedup


def _format_improvement_percent(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value * 100:+.1f}%"


def _format_speedup(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.2f}x"


def _summarize_metric_source(
    entries: dict[str, PerfEntry],
    *,
    metric_source: MetricSource = "auto",
) -> str:
    if not entries:
        return "unknown"
    if metric_source == "kernel":
        return "kernel"
    if metric_source == "total-op":
        return "total-op"
    modes = {entry.comparison_mode for entry in entries.values()}
    if modes == {"latency"}:
        return "kernel"
    if modes == {"total-op"}:
        return "total-op"
    return "mixed (kernel + total-op fallback)"
