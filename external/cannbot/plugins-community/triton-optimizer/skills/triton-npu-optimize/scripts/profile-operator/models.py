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

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Per-operator aggregated statistics (available in both modes)
# ---------------------------------------------------------------------------

@dataclass
class OperatorStats:
    """Aggregated per-operator-type timing from op_statistic CSV."""
    op_type: str
    core_type: str
    count: int
    total_time_us: float
    min_time_us: float
    avg_time_us: float
    max_time_us: float
    ratio_percent: float


# ---------------------------------------------------------------------------
# Per-invocation pipeline breakdown (available in both modes)
# ---------------------------------------------------------------------------

@dataclass
class PipelineStage:
    """Pipeline ratios from one kernel invocation.

    All ratios are percentages (0-100 range).
    Available for compute ops (AI_CORE, AI_VECTOR_CORE).
    """
    aic_mac_ratio: float = 0.0
    aic_scalar_ratio: float = 0.0
    aic_mte1_ratio: float = 0.0
    aic_mte2_ratio: float = 0.0
    aic_mte3_ratio: float = 0.0
    aiv_vec_ratio: float = 0.0
    aiv_scalar_ratio: float = 0.0
    aiv_mte2_ratio: float = 0.0
    aiv_mte3_ratio: float = 0.0
    cube_utilization: float = 0.0
    block_dim: int = 0


@dataclass
class KernelInvocation:
    """Per-invocation kernel data from op_summary (msprof) or kernel_details (torch-npu-profiler)."""
    op_name: str
    duration_us: float
    wait_time_us: float
    block_dim: int
    pipeline: PipelineStage | None = None  # None for non-compute ops


@dataclass
class TaskRecord:
    """Task scheduler timeline record. msprof only."""
    kernel_name: str
    kernel_type: str
    task_time_us: float
    task_start_us: float
    task_stop_us: float


# ---------------------------------------------------------------------------
# Host API calls (available in both modes)
# ---------------------------------------------------------------------------

@dataclass
class HostApiCall:
    """Host-side AscendCL/Runtime API timing."""
    api_name: str
    level: str
    time_us: float
    count: int
    avg_us: float


# ---------------------------------------------------------------------------
# Torch-NPU-profiler-only artifacts
# ---------------------------------------------------------------------------

@dataclass
class TorchOpTiming:
    """PyTorch-level operator timing (host + device view). torch-npu-profiler only."""
    name: str
    host_self_us: float
    host_total_us: float
    device_self_us: float
    device_total_us: float


@dataclass
class StepTrace:
    """Per-step compute/communication breakdown. torch-npu-profiler only."""
    step: int
    computing_us: float
    communication_not_overlapped_us: float
    overlapped_us: float
    communication_us: float
    free_us: float
    stage_us: float
    bubble_us: float
    communication_not_overlapped_exclude_receive_us: float
    preparing_us: float


# ---------------------------------------------------------------------------
# Classification results
# ---------------------------------------------------------------------------

OperatorTypeKind = Literal["cube", "vector", "mix", "scalar", "other", "unknown"]
BoundClassificationKind = Literal["compute-bound", "memory-bound", "scalar-overhead", "mixed", "unknown"]


@dataclass
class CoreTypeAggregate:
    """Aggregated timing by normalized core type."""
    cube_total_us: float = 0.0
    cube_ratio_pct: float = 0.0
    vector_total_us: float = 0.0
    vector_ratio_pct: float = 0.0
    scalar_total_us: float = 0.0
    scalar_ratio_pct: float = 0.0
    other_total_us: float = 0.0
    other_ratio_pct: float = 0.0
    raw_core_types: dict[str, list[str]] = field(default_factory=lambda: {})


@dataclass
class TaskTimelineSummary:
    """Aggregated task timeline signals."""
    matched_rows: int = 0
    total_task_time_us: float | None = None
    span_us: float | None = None
    total_gap_us: float = 0.0
    max_gap_us: float = 0.0
    overlap_count: int = 0


@dataclass
class HostApiSummary:
    """Aggregated host API signals."""
    launch_related_present: bool = False


# ---------------------------------------------------------------------------
# The top-level profile result
# ---------------------------------------------------------------------------

BenchMode = Literal["msprof", "torch-npu-profiler"]


@dataclass
class ParsedProfile:
    """Complete parsed profile from either profiling mode.

    Fields are None when not available in the source mode.
    """
    bench_mode: BenchMode

    # Metadata
    profile_dir: str
    """Path to the profile root directory."""
    source_files: dict[str, str | None] = field(default_factory=lambda: {})

    # Always available
    operators: list[OperatorStats] = field(default_factory=lambda: [])
    invocations: list[KernelInvocation] = field(default_factory=lambda: [])
    host_api_calls: list[HostApiCall] = field(default_factory=lambda: [])
    """Host-side API call timing."""

    # msprof only
    task_records: list[TaskRecord] | None = None
    """Task scheduler timeline (task_time)."""

    # torch-npu-profiler only
    torch_op_timing: list[TorchOpTiming] | None = None
    """PyTorch operator view (operator_details)."""
    step_trace: list[StepTrace] | None = None
    """Per-step compute/communication breakdown."""

    # Computed / derived
    target: OperatorStats | None = None
    """Selected target operator (hottest or user-specified)."""
    target_inferred: bool = True
    """Whether target was auto-selected (True) or user-specified (False)."""
    top_operators: list[OperatorStats] = field(default_factory=lambda: [])
    core_type_aggregate: CoreTypeAggregate | None = None
    bound_classification: BoundClassificationKind = "unknown"
    bound_scores: dict[str, float] = field(default_factory=lambda: {})
    bound_reasoning: list[str] = field(default_factory=lambda: [])
    operator_type: OperatorTypeKind = "unknown"
    operator_type_signals: list[str] = field(default_factory=lambda: [])
    """Operator type classification signals."""
    operator_type_source: str = "none"
    """Where the operator type evidence came from."""
    task_timeline: TaskTimelineSummary | None = None
    """Aggregated task timeline signals (msprof only)."""
    host_api_summary: HostApiSummary | None = None
    stream_like_tracks: int = 0
    """Aggregated host API signals."""
