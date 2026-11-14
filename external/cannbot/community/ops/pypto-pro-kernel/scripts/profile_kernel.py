# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Measure native NPU tasks, pipe diagnostics and paired full-call wall latency.

Import with runpy.run_path(<skill>/scripts/profile_kernel.py), then call
measure_native(lambda: run(...)). Allocate/copy inputs before the callback.
The callback must repeat the same launch sequence of native tasks on one stream;
task sums do not include host dispatch time or gaps between tasks.
"""

import csv
from pathlib import Path
import statistics
import tempfile
import math
import time


def _task_rows(path, *, repeats, tasks_per_call):
    """Validate and order one fixed launch sequence without dropping CSV fields."""
    if any(type(value) is not int or value < 1 for value in (repeats, tasks_per_call)):
        raise ValueError("repeats and tasks_per_call must be positive integers")
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"Start Time(us)", "Duration(us)"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"unsupported profiler CSV columns: {reader.fieldnames}")
        rows = list(reader)
    expected = repeats * tasks_per_call
    if len(rows) != expected:
        raise ValueError(
            f"expected {expected} native tasks, got {len(rows)}; check callback work and capture completeness"
        )
    times = [(float(row["Start Time(us)"]), float(row["Duration(us)"])) for row in rows]
    if any(
        not math.isfinite(start) or not math.isfinite(duration) or duration <= 0
        for start, duration in times
    ):
        raise ValueError(
            "profiler task timestamps/durations must be finite, with positive durations"
        )
    if len({start for start, _ in times}) != len(rows):
        raise ValueError("coincident task start times make native task order ambiguous")
    return sorted(rows, key=lambda row: float(row["Start Time(us)"]))


def _task_calls(path, *, repeats, tasks_per_call):
    """Read ordered per-call durations; positions are native tasks, not DSL phases."""
    rows = _task_rows(path, repeats=repeats, tasks_per_call=tasks_per_call)
    return tuple(
        tuple(float(row["Duration(us)"]) for row in rows[slice(i, i + tasks_per_call)])
        for i in range(0, len(rows), tasks_per_call)
    )


def task_pipe_statistics(path, *, repeats, tasks_per_call):
    """Keep exported counter names/units, missing samples and native task positions.

    These task-level aggregates do not resolve individual cores. Ratios overlap
    and are not additive; inactive AIC/AIV families can contain exported zeros.
    """
    rows = _task_rows(path, repeats=repeats, tasks_per_call=tasks_per_call)
    fields = [
        key
        for key in rows[0]
        if key.startswith(("aic_", "aiv_", "aicore_")) or key == "cube_utilization(%)"
    ]
    result = []
    for position in range(tasks_per_call):
        selected = rows[position::tasks_per_call]
        counters = {}
        for field in fields:
            values = []
            for row in selected:
                raw = (row.get(field) or "").strip()
                if raw.upper() in (
                    "",
                    "NA",
                    "N/A",
                    "NULL",
                    "NONE",
                    "NAN",
                    "INF",
                    "+INF",
                    "-INF",
                ):
                    continue
                try:
                    value = float(raw)
                except ValueError as exc:
                    raise ValueError(
                        f"unsupported counter value for {field}: {raw!r}"
                    ) from exc
                if math.isfinite(value):
                    values.append(value)
            counters[field] = {
                "median": statistics.median(values) if values else None,
                "samples": len(values),
                "missing_samples": repeats - len(values),
            }
        identity = {
            key: sorted({row[key] for row in selected if row.get(key)})
            for key in (
                "Name",
                "Type",
                "Accelerator Core",
                "Block Num",
                "Mix Block Num",
            )
        }
        result.append(
            {
                "position": position + 1,
                "identity": identity,
                "available": any(item["samples"] for item in counters.values()),
                "counters": counters,
            }
        )
    return result


def task_samples(path, *, repeats, tasks_per_call):
    """Parse one capture strictly; missing/extra tasks never become zero cost."""
    return tuple(
        sum(call)
        for call in _task_calls(path, repeats=repeats, tasks_per_call=tasks_per_call)
    )


def task_statistics(path, *, repeats, tasks_per_call):
    """Summarize whole-call sums and each native task's one-based position.

    Caller guarantees one stream and the same launch sequence per repetition.
    A position is not an inferred source function, phase, or hardware pipe.
    """
    return _statistics(
        _task_calls(path, repeats=repeats, tasks_per_call=tasks_per_call)
    )


def _statistics(calls):
    repeats, tasks_per_call = len(calls), len(calls[0])
    samples = tuple(sum(call) for call in calls)
    per_task = []
    for position in range(tasks_per_call):
        durations = tuple(call[position] for call in calls)
        per_task.append(
            {
                "position": position + 1,
                "median_us": statistics.median(durations),
                "min_us": min(durations),
                "max_us": max(durations),
            }
        )
    return {
        "metric": "native_task_sum_us",
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "repeats": repeats,
        "tasks_per_call": tasks_per_call,
        "per_task": per_task,
    }


def _paired_statistics(calls, before_tasks, after_tasks, order):
    if order not in ("AB", "BA"):
        raise ValueError("order must be AB (baseline first) or BA (candidate first)")
    baseline, candidate = [], []
    for call in calls:
        if order == "AB":
            baseline.append(sum(call[:before_tasks]))
            candidate.append(sum(call[before_tasks:]))
        else:
            candidate.append(sum(call[:after_tasks]))
            baseline.append(sum(call[after_tasks:]))

    def summary(values):
        return {
            "median_us": statistics.median(values),
            "min_us": min(values),
            "max_us": max(values),
        }

    wins = sum(after < before for before, after in zip(baseline, candidate))
    return {
        "order": order,
        "pairs": len(calls),
        "baseline": summary(baseline),
        "candidate": summary(candidate),
        "median_ratio": statistics.median(
            after / before for before, after in zip(baseline, candidate)
        ),
        "candidate_wins": wins,
        "win_fraction": wins / len(calls),
    }


def paired_task_statistics(path, *, repeats, before_tasks, after_tasks, order="AB"):
    """Parse paired calls with possibly different native task counts, without NPU imports."""
    if any(
        type(value) is not int or value < 1
        for value in (repeats, before_tasks, after_tasks)
    ):
        raise ValueError(
            "repeats, before_tasks and after_tasks must be positive integers"
        )
    calls = _task_calls(
        path, repeats=repeats, tasks_per_call=before_tasks + after_tasks
    )
    return _paired_statistics(calls, before_tasks, after_tasks, order)


def _runtime(*, pipe_metrics=False):
    import torch
    import torch_npu

    profiler = torch_npu.profiler
    required = (
        "_ExperimentalConfig",
        "ExportType",
        "ProfilerLevel",
        "ProfilerActivity",
        "profile",
        "tensorboard_trace_handler",
    )
    missing = [name for name in required if not hasattr(profiler, name)]
    if missing:
        raise RuntimeError(
            f"torch_npu {torch_npu.__version__}: unavailable profiler capabilities {missing}; no timing established"
        )
    try:
        metrics = (
            {"aic_metrics": profiler.AiCMetrics.PipeUtilization} if pipe_metrics else {}
        )
        config = profiler._ExperimentalConfig(
            export_type=[profiler.ExportType.Text],
            profiler_level=profiler.ProfilerLevel.Level1,
            **metrics,
        )
    except (AttributeError, TypeError) as exc:
        raise RuntimeError(
            f"torch_npu {torch_npu.__version__}: unsupported profiler configuration"
        ) from exc
    return torch, torch_npu, profiler, config


def _capture_calls(call, *, repeats, tasks_per_call, runtime, pipe_metrics=False):
    torch, _, profiler, config = runtime
    with tempfile.TemporaryDirectory(prefix="native-task-profile-") as directory:
        with profiler.profile(
            activities=[profiler.ProfilerActivity.NPU],
            with_stack=False,
            record_shapes=False,
            profile_memory=False,
            experimental_config=config,
            on_trace_ready=profiler.tensorboard_trace_handler(
                directory, analyse_flag=True
            ),
        ):
            for _ in range(repeats):
                call()
            torch.npu.synchronize()
        files = list(Path(directory).rglob("kernel_details.csv"))
        if len(files) != 1:
            raise RuntimeError(
                f"expected one kernel_details.csv, found {len(files)}; no device timing established"
            )
        calls = _task_calls(files[0], repeats=repeats, tasks_per_call=tasks_per_call)
        if pipe_metrics:
            pipes = task_pipe_statistics(
                files[0], repeats=repeats, tasks_per_call=tasks_per_call
            )
            return calls, pipes
    return calls


def measure_native(call, *, warmup=3, repeats=20, tasks_per_call=1):
    """Return whole-call and per-task durations, excluding event/host intervals.

    Caller owns device selection, external board lock and timeout. This function
    invokes call warmup + repeats times; validate numerical results separately.
    """
    if any(
        type(value) is not int or value < 1
        for value in (warmup, repeats, tasks_per_call)
    ):
        raise ValueError("warmup, repeats and tasks_per_call must be positive integers")
    runtime = _runtime()
    torch, torch_npu, _, _ = runtime
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    calls = _capture_calls(
        call, repeats=repeats, tasks_per_call=tasks_per_call, runtime=runtime
    )
    return {
        **_statistics(calls),
        "torch": str(torch.__version__),
        "torch_npu": str(torch_npu.__version__),
    }


def measure_pipes(call, *, warmup=3, repeats=20, tasks_per_call=1):
    """Collect warmed task-level PipeUtilization alongside complete-call durations.

    Use per-core msprof op output when these aggregates cannot explain a tail.
    Caller owns immutable inputs, numerical validation, device lock and timeout.
    """
    if any(
        type(value) is not int or value < 1
        for value in (warmup, repeats, tasks_per_call)
    ):
        raise ValueError("warmup, repeats and tasks_per_call must be positive integers")
    runtime = _runtime(pipe_metrics=True)
    torch, torch_npu, _, _ = runtime
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    calls, pipes = _capture_calls(
        call,
        repeats=repeats,
        tasks_per_call=tasks_per_call,
        runtime=runtime,
        pipe_metrics=True,
    )
    return {
        **_statistics(calls),
        "pipe_scope": "task_aggregate",
        "pipe_counters": pipes,
        "torch": str(torch.__version__),
        "torch_npu": str(torch_npu.__version__),
    }


def compare_native(baseline, candidate, *, warmup=3, repeats=20, tasks_per_call=(1, 1)):
    """Measure AB and BA in one process; ratios pair complete native-task sums.

    tasks_per_call is (baseline_count, candidate_count), independent of order.
    Callbacks must use equivalent immutable inputs, repeat a fixed task sequence
    on one stream, and overwrite their outputs. Check correctness before calling.
    Caller owns the device, board lock and timeout. Captures are always temporary.
    """
    if not isinstance(tasks_per_call, (tuple, list)) or len(tasks_per_call) != 2:
        raise ValueError(
            "tasks_per_call must contain baseline and candidate task counts"
        )
    if any(
        type(value) is not int or value < 1
        for value in (warmup, repeats, *tasks_per_call)
    ):
        raise ValueError(
            "warmup, repeats and both task counts must be positive integers"
        )
    before_tasks, after_tasks = tasks_per_call
    runtime = _runtime()
    torch, torch_npu, _, _ = runtime
    for _ in range(warmup):
        baseline()
        candidate()
    torch.npu.synchronize()
    results = []
    for order in ("AB", "BA"):
        first, second = (
            (baseline, candidate) if order == "AB" else (candidate, baseline)
        )

        def pair():
            first()
            second()

        calls = _capture_calls(
            pair,
            repeats=repeats,
            tasks_per_call=before_tasks + after_tasks,
            runtime=runtime,
        )
        results.append(_paired_statistics(calls, before_tasks, after_tasks, order))
    return {
        "metric": "native_task_sum_us",
        "orders": results,
        "tasks_per_call": {"baseline": before_tasks, "candidate": after_tasks},
        "warmup": warmup,
        "pairs_per_order": repeats,
        "torch": str(torch.__version__),
        "torch_npu": str(torch_npu.__version__),
    }


def compare_wall(baseline, candidate, *, warmup=3, repeats=20):
    """Pair warmed, synchronized full-call latency in AB and BA order.

    Each callback is invoked warmup + 2*repeats times. Prepare equivalent states
    outside the callbacks, including fresh pools for single in-place updates.
    This measures host dispatch, execution and the final synchronization; it
    does not collect profiler task durations or establish numerical correctness.
    Caller owns device selection, the board lock and an otherwise idle stream.
    """
    if any(type(value) is not int or value < 1 for value in (warmup, repeats)):
        raise ValueError("warmup and repeats must be positive integers")
    import torch
    import torch_npu

    for _ in range(warmup):
        baseline()
        candidate()
    torch.npu.synchronize()
    orders = []
    for order in ("AB", "BA"):
        callbacks = (baseline, candidate) if order == "AB" else (candidate, baseline)
        pairs = []
        for _ in range(repeats):
            durations = []
            for callback in callbacks:
                torch.npu.synchronize()
                start = time.perf_counter_ns()
                callback()
                torch.npu.synchronize()
                duration = (time.perf_counter_ns() - start) / 1000
                if duration <= 0:
                    raise RuntimeError("wall-clock duration must be positive")
                durations.append(duration)
            pairs.append(tuple(durations))
        orders.append(_paired_statistics(pairs, 1, 1, order))
    return {
        "metric": "synchronized_wall_us",
        "orders": orders,
        "warmup": warmup,
        "pairs_per_order": repeats,
        "torch": str(torch.__version__),
        "torch_npu": str(torch_npu.__version__),
    }
