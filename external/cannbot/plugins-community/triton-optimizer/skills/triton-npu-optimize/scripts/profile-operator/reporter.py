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
from pathlib import Path
from typing import Any, cast

from models import (
    BoundClassificationKind,
    CoreTypeAggregate,
    HostApiCall,
    HostApiSummary,
    KernelInvocation,
    OperatorStats,
    OperatorTypeKind,
    ParsedProfile,
    TaskTimelineSummary,
)
from parser_base import detect_profile_mode
from msprof_parser import MsprofParser
from torch_npu_profiler_parser import TorchNpuProfilerParser

_TRANSFER_HINTS = (
    "copy", "memcpy", "transdata", "dma", "load", "store", "move",
)


def _format_number(value: float) -> str:
    if value.is_integer():
        return f"{value:.1f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        padded = row + [""] * (len(headers) - len(row))
        lines.append("| " + " | ".join(padded[: len(headers)]) + " |")
    return "\n".join(lines)


def _normalize_core_type(core_type: str) -> str:
    lowered = core_type.strip().lower()
    if "scalar" in lowered:
        return "scalar"
    if "vector" in lowered:
        return "vector"
    if "cube" in lowered or "aicore" in lowered or "ai_core" in lowered:
        return "cube"
    return "other"


def _is_data_movement_op(op_type: str) -> bool:
    lowered = op_type.lower()
    return any(hint in lowered for hint in _TRANSFER_HINTS)


def _compute_data_movement_hotspots(operators: list[OperatorStats]) -> list[OperatorStats]:
    hotspots = [op for op in operators if _is_data_movement_op(op.op_type)]
    return sorted(hotspots, key=lambda op: op.total_time_us, reverse=True)


def _classify_operator_type(
    core_type: str,
    ratio_avgs: dict[str, float],
    cube_utilization_avg: float | None,
) -> tuple[OperatorTypeKind, list[str], str]:
    vector_ratio = ratio_avgs.get("aiv_vec_ratio", 0.0)
    cube_ratio = max(ratio_avgs.get("aic_mac_ratio", 0.0), cube_utilization_avg or 0.0)
    signals: list[str] = []
    if vector_ratio >= 30.0:
        signals.append(f"high aiv_vec_ratio={_format_number(vector_ratio)}")
    if cube_ratio >= 20.0:
        signals.append(f"high cube-side activity={_format_number(cube_ratio)}")

    if vector_ratio >= 30.0 and cube_ratio >= 20.0:
        kind: OperatorTypeKind = "mix"
    elif cube_ratio >= 20.0:
        kind = "cube"
    elif vector_ratio >= 20.0:
        kind = "vector"
    else:
        kind: OperatorTypeKind = cast(OperatorTypeKind, _normalize_core_type(core_type))
        signals.append(f"fallback to target core type {core_type}")

    source = "op_summary" if signals and "fallback" not in signals[-1] else "op_statistic"
    return kind, signals, source


def _classify_bound(ratio_avgs: dict[str, float]) -> tuple[BoundClassificationKind, dict[str, float], list[str]]:
    compute_score = ratio_avgs.get("aic_mac_ratio", 0.0) + ratio_avgs.get("aiv_vec_ratio", 0.0)
    memory_score = (
        ratio_avgs.get("aic_mte1_ratio", 0.0)
        + ratio_avgs.get("aic_mte2_ratio", 0.0)
        + ratio_avgs.get("aic_mte3_ratio", 0.0)
        + ratio_avgs.get("aiv_mte2_ratio", 0.0)
        + ratio_avgs.get("aiv_mte3_ratio", 0.0)
    )
    scalar_score = ratio_avgs.get("aic_scalar_ratio", 0.0) + ratio_avgs.get("aiv_scalar_ratio", 0.0)

    scores = {"compute": compute_score, "memory": memory_score, "scalar": scalar_score}

    if scalar_score >= max(compute_score, memory_score) and scalar_score >= 20.0:
        classification: BoundClassificationKind = "scalar-overhead"
        reasoning = ["scalar-side ratios dominate the visible pipeline ratios"]
    elif memory_score >= compute_score + 5.0:
        classification = "memory-bound"
        reasoning = ["MTE-side ratios exceed compute-side ratios"]
    elif compute_score >= 70.0 and scalar_score < 20.0 and memory_score < 25.0:
        classification = "compute-bound"
        reasoning = ["compute-side ratios dominate with limited scalar and MTE pressure"]
    else:
        classification = "mixed"
        reasoning = ["no single pipeline family dominates strongly enough"]

    return classification, scores, reasoning


def _select_target(operators: list[OperatorStats], target_op: str | None) -> tuple[OperatorStats, bool]:
    if target_op is not None:
        for op in operators:
            if op.op_type == target_op:
                return op, False
        available_operators = sorted({op.op_type for op in operators})
        available_text = ", ".join(available_operators) if available_operators else "(none)"
        raise ValueError(
            f"Target operator not found in op_statistic: {target_op}. "
            f"Available operators: {available_text}"
        )
    return max(operators, key=lambda op: op.total_time_us), True


def _aggregate_core_types(operators: list[OperatorStats]) -> CoreTypeAggregate:
    agg = CoreTypeAggregate()
    for op in operators:
        bucket = _normalize_core_type(op.core_type)
        agg.raw_core_types.setdefault(bucket, [])
        if op.core_type not in agg.raw_core_types[bucket]:
            agg.raw_core_types[bucket].append(op.core_type)
        if bucket == "cube":
            agg.cube_total_us += op.total_time_us
            agg.cube_ratio_pct += op.ratio_percent
        elif bucket == "vector":
            agg.vector_total_us += op.total_time_us
            agg.vector_ratio_pct += op.ratio_percent
        elif bucket == "scalar":
            agg.scalar_total_us += op.total_time_us
            agg.scalar_ratio_pct += op.ratio_percent
        else:
            agg.other_total_us += op.total_time_us
            agg.other_ratio_pct += op.ratio_percent
    return agg


def _compute_pipeline_averages(invocations: list[KernelInvocation], target_op: str) -> dict[str, float]:
    ratio_values: dict[str, list[float]] = {
        "aic_mac_ratio": [],
        "aic_scalar_ratio": [],
        "aic_mte1_ratio": [],
        "aic_mte2_ratio": [],
        "aic_mte3_ratio": [],
        "aiv_vec_ratio": [],
        "aiv_scalar_ratio": [],
        "aiv_mte2_ratio": [],
        "aiv_mte3_ratio": [],
    }
    for inv in invocations:
        if inv.op_name != target_op or inv.pipeline is None:
            continue
        ratio_values["aic_mac_ratio"].append(inv.pipeline.aic_mac_ratio)
        ratio_values["aic_scalar_ratio"].append(inv.pipeline.aic_scalar_ratio)
        ratio_values["aic_mte1_ratio"].append(inv.pipeline.aic_mte1_ratio)
        ratio_values["aic_mte2_ratio"].append(inv.pipeline.aic_mte2_ratio)
        ratio_values["aic_mte3_ratio"].append(inv.pipeline.aic_mte3_ratio)
        ratio_values["aiv_vec_ratio"].append(inv.pipeline.aiv_vec_ratio)
        ratio_values["aiv_scalar_ratio"].append(inv.pipeline.aiv_scalar_ratio)
        ratio_values["aiv_mte2_ratio"].append(inv.pipeline.aiv_mte2_ratio)
        ratio_values["aiv_mte3_ratio"].append(inv.pipeline.aiv_mte3_ratio)

    return {key: sum(vals) / len(vals) if vals else 0.0 for key, vals in ratio_values.items()}


def _count_operators_by_bucket(operators: list[OperatorStats]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for op in operators:
        bucket = _normalize_core_type(op.core_type)
        counts[bucket] = counts.get(bucket, 0) + 1
    return counts


def _check_launch_related(host_api: list[HostApiCall]) -> bool:
    launch_keywords = ("launch", "memcpy", "synchronize", "binary", "stream")
    return any(
        any(kw in call.api_name.lower() for kw in launch_keywords)
        for call in host_api
    )


def _parse_profile_inputs(profile_path: Path) -> dict[str, Any]:
    mode, artifacts_dir = detect_profile_mode(profile_path)
    if mode == "msprof":
        parsed = MsprofParser.parse(artifacts_dir)
        return {
            "mode": mode, "operators": parsed.operators, "invocations": parsed.invocations,
            "task_records": parsed.task_records, "torch_ops": None, "step_traces": None,
            "host_api": parsed.host_api, "timeline": parsed.timeline, "source_files": parsed.source_files,
        }
    parsed = TorchNpuProfilerParser.parse(artifacts_dir)
    return {
        "mode": mode, "operators": parsed.operators, "invocations": parsed.invocations,
        "task_records": None, "torch_ops": parsed.torch_ops, "step_traces": parsed.step_traces,
        "host_api": parsed.host_api, "timeline": parsed.timeline, "source_files": parsed.source_files,
    }


def _summarize_task_timeline(task_records: Any, target_op: str) -> TaskTimelineSummary | None:
    target_tasks = sorted(
        (task for task in task_records or [] if task.kernel_name == target_op),
        key=lambda task: task.task_start_us,
    )
    if not target_tasks:
        return None
    gaps = [
        current.task_start_us - previous.task_stop_us
        for previous, current in zip(target_tasks, target_tasks[1:])
    ]
    positive_gaps = [gap for gap in gaps if gap >= 0]
    return TaskTimelineSummary(
        matched_rows=len(target_tasks),
        total_task_time_us=sum(task.task_time_us for task in target_tasks),
        span_us=target_tasks[-1].task_stop_us - target_tasks[0].task_start_us,
        total_gap_us=sum(positive_gaps),
        max_gap_us=max(positive_gaps, default=0.0),
        overlap_count=sum(gap < 0 for gap in gaps),
    )


def _profile_json_payload(profile: ParsedProfile) -> dict[str, Any]:
    target = _require_profile_target(profile)
    invocations = _target_invocations(profile, target.op_type)
    return {
        **_profile_json_metadata(profile, target),
        "op_summary": _op_summary_stats(invocations),
        "core_type_totals": _core_type_totals(profile),
        "data_movement_hotspots": [
            _operator_json(operator)
            for operator in _compute_data_movement_hotspots(profile.operators)
        ],
        "top_ops": [_operator_json(operator) for operator in profile.top_operators],
        "operator_type_guess": _operator_type_json(profile),
        "bound_analysis": _bound_analysis_json(profile),
        "pipeline_signals": _pipeline_signals(invocations),
        "task_timeline_signals": _task_timeline_signals(profile),
        "host_api_signals": _host_api_signals(profile),
        "msprof_timeline_signals": {"stream_like_tracks": profile.stream_like_tracks},
        "binary_signals": {},
    }


def _require_profile_target(profile: ParsedProfile) -> OperatorStats:
    if profile.target is None:
        raise ValueError("ParsedProfile.target is None")
    return profile.target


def _target_invocations(profile: ParsedProfile, target_op: str) -> list[KernelInvocation]:
    return [invocation for invocation in profile.invocations if invocation.op_name == target_op]


def _profile_json_metadata(profile: ParsedProfile, target: OperatorStats) -> dict[str, Any]:
    source_files = profile.source_files
    return {
        "profile_dir": profile.profile_dir,
        "op_statistic_file": source_files.get("op_statistic"),
        "op_summary_file": source_files.get("op_summary") or source_files.get("kernel_details"),
        "task_time_file": source_files.get("task_time"),
        "api_statistic_file": source_files.get("api_statistic"),
        "msprof_json_file": source_files.get("msprof_json") or source_files.get("trace_view"),
        "target_operator": target.op_type,
        "selection": _selection_text(profile.target_inferred),
        "target_row": _operator_json(target),
    }


def _selection_text(target_inferred: bool) -> str:
    if target_inferred:
        return "inferred from the hottest `op_statistic` row by `Total Time(us)`"
    return "matched the explicit `--target-op` value"


def _operator_json(operator: OperatorStats) -> dict[str, Any]:
    return {
        "op_type": operator.op_type,
        "core_type": operator.core_type,
        "count": float(operator.count),
        "total_time_us": operator.total_time_us,
        "min_time_us": operator.min_time_us,
        "avg_time_us": operator.avg_time_us,
        "max_time_us": operator.max_time_us,
        "ratio_percent": operator.ratio_percent,
    }


def _op_summary_stats(invocations: list[KernelInvocation]) -> dict[str, Any]:
    durations = [invocation.duration_us for invocation in invocations if invocation.duration_us > 0]
    return {
        "matched_rows": len(invocations),
        "total_duration_us": sum(durations) if durations else None,
        "avg_duration_us": sum(durations) / len(durations) if durations else None,
        "min_duration_us": min(durations) if durations else None,
        "max_duration_us": max(durations) if durations else None,
        "note": None,
    }


def _core_type_totals(profile: ParsedProfile) -> dict[str, Any]:
    aggregate = profile.core_type_aggregate
    if aggregate is None:
        return {}
    counts = _count_operators_by_bucket(profile.operators)
    totals: dict[str, Any] = {}
    for bucket, total, ratio, raw_types in _core_type_total_rows(aggregate):
        if total > 0 or raw_types:
            totals[bucket] = {
                "total_time_us": total,
                "ratio_percent": ratio,
                "count": counts.get(bucket, 0),
                "raw_core_types": raw_types,
            }
    return totals


def _core_type_total_rows(
    aggregate: CoreTypeAggregate,
) -> tuple[tuple[str, float, float, list[str]], ...]:
    return (
        ("cube", aggregate.cube_total_us, aggregate.cube_ratio_pct, aggregate.raw_core_types.get("cube", [])),
        ("vector", aggregate.vector_total_us, aggregate.vector_ratio_pct, aggregate.raw_core_types.get("vector", [])),
        ("scalar", aggregate.scalar_total_us, aggregate.scalar_ratio_pct, aggregate.raw_core_types.get("scalar", [])),
        ("other", aggregate.other_total_us, aggregate.other_ratio_pct, aggregate.raw_core_types.get("other", [])),
    )


def _operator_type_json(profile: ParsedProfile) -> dict[str, Any]:
    return {
        "kind": profile.operator_type,
        "signals": profile.operator_type_signals,
        "source": profile.operator_type_source,
    }


def _bound_analysis_json(profile: ParsedProfile) -> dict[str, Any]:
    return {
        "classification": profile.bound_classification,
        "scores": profile.bound_scores,
        "reasoning": profile.bound_reasoning,
    }


def _task_timeline_signals(profile: ParsedProfile) -> dict[str, Any]:
    if profile.task_timeline is None:
        return {"matched_rows": 0, "max_gap_us": 0.0}
    return {
        "matched_rows": profile.task_timeline.matched_rows,
        "max_gap_us": profile.task_timeline.max_gap_us,
    }


def _host_api_signals(profile: ParsedProfile) -> dict[str, Any]:
    top_apis = sorted(profile.host_api_calls, key=lambda call: call.time_us, reverse=True)[:10]
    launch_related = profile.host_api_summary.launch_related_present if profile.host_api_summary else False
    return {
        "launch_related_present": launch_related,
        "top_apis": [
            {"api_name": call.api_name, "time_us": call.time_us, "count": call.count, "avg_us": call.avg_us}
            for call in top_apis
        ],
    }


def _pipeline_signals(invocations: list[KernelInvocation]) -> dict[str, Any]:
    ratios: dict[str, list[float]] = {}
    waits: list[float] = []
    cubes: list[float] = []
    block_dims: set[int] = set()
    for invocation in invocations:
        _collect_pipeline_values(invocation, ratios, waits, cubes, block_dims)
    signals: dict[str, Any] = {}
    if ratios:
        signals["ratios"] = {name: _numeric_summary(values) for name, values in ratios.items()}
    if waits:
        signals["task_wait_time_us"] = _numeric_summary(waits)
    if cubes:
        signals["cube_utilization_percent"] = _numeric_summary(cubes)
    if block_dims:
        signals["block_dim"] = {"observed_values": sorted(block_dims)}
    return signals


def _collect_pipeline_values(
    invocation: KernelInvocation,
    ratios: dict[str, list[float]],
    waits: list[float],
    cubes: list[float],
    block_dims: set[int],
) -> None:
    if invocation.wait_time_us > 0:
        waits.append(invocation.wait_time_us)
    if invocation.block_dim > 0:
        block_dims.add(invocation.block_dim)
    if invocation.pipeline is None:
        return
    pipeline = invocation.pipeline
    for name in _PIPELINE_RATIO_NAMES:
        ratios.setdefault(name, []).append(getattr(pipeline, name))
    cubes.append(pipeline.cube_utilization)


_PIPELINE_RATIO_NAMES = (
    "aic_mac_ratio", "aic_scalar_ratio", "aic_mte1_ratio", "aic_mte2_ratio", "aic_mte3_ratio",
    "aiv_vec_ratio", "aiv_scalar_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio",
)


def _numeric_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "avg": sum(values) / len(values),
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "total": sum(values),
    }


def _markdown_top_rows(profile: ParsedProfile) -> list[list[str]]:
    return [
        [
            op.op_type,
            op.core_type,
            _format_number(float(op.count)),
            _format_number(op.total_time_us),
            _format_number(op.avg_time_us),
            _format_number(op.ratio_percent),
        ]
        for op in profile.top_operators
    ]


def _markdown_core_rows(profile: ParsedProfile) -> list[list[str]]:
    aggregate = profile.core_type_aggregate
    if aggregate is None:
        return []
    counts = _count_operators_by_bucket(profile.operators)
    rows: list[list[str]] = []
    for bucket, total, ratio, raw_types in _core_type_total_rows(aggregate):
        if total > 0 or raw_types:
            rows.append(
                [
                    bucket,
                    ", ".join(sorted(raw_types)),
                    _format_number(float(counts.get(bucket, 0))),
                    _format_number(total),
                    _format_number(ratio),
                ]
            )
    return rows


def _markdown_movement_rows(profile: ParsedProfile) -> list[list[str]]:
    return [
        [
            op.op_type,
            op.core_type,
            _format_number(op.total_time_us),
            _format_number(op.ratio_percent),
        ]
        for op in _compute_data_movement_hotspots(profile.operators)
    ]


def _markdown_header(profile: ParsedProfile, target: OperatorStats) -> list[str]:
    source_files = profile.source_files
    op_summary_file = source_files.get("op_summary") or source_files.get("kernel_details")
    return [
        "# Ascend NPU Operator Profile Summary",
        "",
        f"- Profile directory: `{profile.profile_dir}`",
        f"- `op_statistic` file: `{source_files.get('op_statistic', 'not found')}`",
        f"- `op_summary` file: `{op_summary_file or 'not found'}`",
        f"- Target operator: `{target.op_type}`",
        f"- Selection: {_selection_text(profile.target_inferred)}",
        "",
        "## Operator timing",
        "",
        f"- Core type: `{target.core_type}`",
        f"- Invocation count: `{_format_number(float(target.count))}`",
        f"- Total time: `{_format_number(target.total_time_us)} us`",
        f"- Average time: `{_format_number(target.avg_time_us)} us`",
        f"- Min time: `{_format_number(target.min_time_us)} us`",
        f"- Max time: `{_format_number(target.max_time_us)} us`",
        f"- Runtime ratio: `{_format_number(target.ratio_percent)}%`",
    ]


def _markdown_op_summary(profile: ParsedProfile, target: OperatorStats) -> list[str]:
    invocations = _target_invocations(profile, target.op_type)
    stats = _op_summary_stats(invocations)
    lines = ["", "## op_summary cross-check", "", f"- Matched op_summary rows: `{stats['matched_rows']}`"]
    if stats["total_duration_us"] is not None:
        lines.extend(
            [
                f"- Summed task duration: `{_format_number(cast(float, stats['total_duration_us']))} us`",
                f"- Average task duration: `{_format_number(cast(float, stats['avg_duration_us']))} us`",
                f"- Min task duration: `{_format_number(cast(float, stats['min_duration_us']))} us`",
                f"- Max task duration: `{_format_number(cast(float, stats['max_duration_us']))} us`",
            ]
        )
    elif not invocations:
        lines.append("- Note: No invocations matched the target operator.")
    return lines


def _markdown_tail(profile: ParsedProfile, core_rows: list[list[str]], movement_rows: list[list[str]]) -> list[str]:
    task = profile.task_timeline
    launch_present = profile.host_api_summary.launch_related_present if profile.host_api_summary else False
    return [
        f"- Operator type guess: `{profile.operator_type}`",
        f"- Bound analysis: `{profile.bound_classification}`",
        "",
        "## Core type totals",
        "",
        _markdown_table(["Bucket", "Raw core types", "Count", "Total Time(us)", "Ratio(%)"], core_rows),
        "",
        "## Data movement hotspots",
        "",
        _markdown_table(["OP Type", "Core Type", "Total Time(us)", "Ratio(%)"], movement_rows)
        if movement_rows else "_No transfer-like hotspots matched the default heuristics._",
        "",
        "## Layered profiler signals",
        "",
        f"- Task timeline matched rows: `{task.matched_rows if task else 0}`",
        f"- Max task gap: `{_format_number(task.max_gap_us if task else 0.0)} us`",
        f"- Host launch-related APIs present: `{launch_present}`",
        f"- msprof tracks: `{profile.stream_like_tracks}`",
        "- Binary signals available: `False`",
        "",
        "## Top operators by total time",
        "",
        _markdown_table(
            ["OP Type", "Core Type", "Count", "Total Time(us)", "Avg Time(us)", "Ratio(%)"],
            _markdown_top_rows(profile),
        ),
    ]


class ProfileReporter:
    @staticmethod
    def load_profile(
        profile_path: Path,
        target_op: str | None = None,
        top_count: int = 5,
    ) -> ParsedProfile:
        inputs = _parse_profile_inputs(profile_path)
        operators = cast(list[OperatorStats], inputs["operators"])
        invocations = cast(list[KernelInvocation], inputs["invocations"])
        target, inferred = _select_target(operators, target_op)
        ratio_avgs = _compute_pipeline_averages(invocations, target.op_type)

        cube_utilization_values = [
            inv.pipeline.cube_utilization
            for inv in invocations
            if inv.op_name == target.op_type and inv.pipeline is not None
        ]
        cube_util_avg = sum(cube_utilization_values) / len(cube_utilization_values) if cube_utilization_values else None

        op_type_kind, op_type_signals, op_type_source = _classify_operator_type(
            target.core_type, ratio_avgs, cube_util_avg
        )
        bound_kind, bound_scores, bound_reasoning = _classify_bound(ratio_avgs)

        return ParsedProfile(
            bench_mode=cast(str, inputs["mode"]),
            profile_dir=str(profile_path.resolve()),
            source_files=cast(dict[str, str | None], inputs["source_files"]),
            operators=operators,
            invocations=invocations,
            host_api_calls=cast(list[HostApiCall], inputs["host_api"]),
            task_records=inputs["task_records"],
            torch_op_timing=inputs["torch_ops"],
            step_trace=inputs["step_traces"],
            target=target,
            target_inferred=inferred,
            top_operators=sorted(operators, key=lambda op: op.total_time_us, reverse=True)[:top_count],
            core_type_aggregate=_aggregate_core_types(operators),
            bound_classification=bound_kind,
            bound_scores=bound_scores,
            bound_reasoning=bound_reasoning,
            operator_type=op_type_kind,
            operator_type_signals=op_type_signals,
            operator_type_source=op_type_source,
            task_timeline=_summarize_task_timeline(inputs["task_records"], target.op_type),
            host_api_summary=HostApiSummary(
                launch_related_present=_check_launch_related(cast(list[HostApiCall], inputs["host_api"]))
            ),
            stream_like_tracks=len({event.get("tid") for event in inputs["timeline"] if event.get("tid") is not None}),
        )

    @staticmethod
    def _render_markdown(profile: ParsedProfile) -> str:
        target = _require_profile_target(profile)
        core_rows = _markdown_core_rows(profile)
        movement_rows = _markdown_movement_rows(profile)
        lines = _markdown_header(profile, target)
        lines.extend(_markdown_op_summary(profile, target))
        lines.extend(_markdown_tail(profile, core_rows, movement_rows))
        return "\n".join(lines) + "\n"

    @staticmethod
    def _build_json_payload(profile: ParsedProfile) -> dict[str, Any]:
        return _profile_json_payload(profile)

    def build_report(
        self,
        profile_path: str | Path,
        target_op: str | None = None,
        top_count: int = 5,
        output_format: str = "markdown",
    ) -> str:
        profile = self.load_profile(Path(profile_path), target_op=target_op, top_count=top_count)
        if output_format == "json":
            return self._render_json(profile)
        if output_format != "markdown":
            raise ValueError(f"Unsupported output format: {output_format}")
        return self._render_markdown(profile)

    def _render_json(self, profile: ParsedProfile) -> str:
        return json.dumps(self._build_json_payload(profile), indent=2, sort_keys=True) + "\n"


def build_report(
    profile_path: str | Path,
    target_op: str | None = None,
    top_count: int = 5,
    output_format: str = "markdown",
) -> str:
    return ProfileReporter().build_report(
        profile_path,
        target_op=target_op,
        top_count=top_count,
        output_format=output_format,
    )
