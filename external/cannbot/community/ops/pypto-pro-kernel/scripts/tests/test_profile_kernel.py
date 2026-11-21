# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "profile_kernel", Path(__file__).resolve().parents[1] / "profile_kernel.py"
)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def capture(tmp_path, rows, header="Start Time(us),Duration(us)\n"):
    path = tmp_path / "kernel_details.csv"
    path.write_text(header + "\n".join(rows) + "\n")
    return path


def test_orders_tasks_and_sums_only_within_each_call(tmp_path):
    path = capture(tmp_path, ["4,3", "1,1", "3,2", "2,4"])
    assert profile.task_samples(path, repeats=2, tasks_per_call=2) == (5, 5)


@pytest.mark.parametrize("rows", [["1,2"], ["1,2", "3,2", "5,2"]])
def test_missing_or_extra_tasks_fail_instead_of_biasing_average(tmp_path, rows):
    with pytest.raises(ValueError, match="expected 2 native tasks"):
        profile.task_samples(capture(tmp_path, rows), repeats=2, tasks_per_call=1)


@pytest.mark.parametrize("row", ["nan,1", "1,nan", "inf,1", "1,inf", "1,0", "1,-1"])
def test_invalid_timestamps_and_durations_fail(tmp_path, row):
    with pytest.raises(ValueError, match="finite"):
        profile.task_samples(capture(tmp_path, [row]), repeats=1, tasks_per_call=1)


def test_unknown_units_are_not_silently_interpreted_as_microseconds(tmp_path):
    with pytest.raises(ValueError, match="unsupported profiler CSV"):
        profile.task_samples(
            capture(tmp_path, ["1,2"], "Start Time(ns),Duration(ns)\n"),
            repeats=1,
            tasks_per_call=1,
        )


@pytest.mark.parametrize("measure", [profile.measure_native, profile.measure_pipes])
@pytest.mark.parametrize(
    "options",
    [{"warmup": 0}, {"repeats": 0}, {"tasks_per_call": -1}, {"repeats": True}],
)
def test_bad_counts_fail_before_importing_device_runtime(options, measure):
    with pytest.raises(ValueError, match="positive integers"):
        measure(lambda: None, **options)


def test_native_task_positions_keep_per_call_pairing_after_sorting(tmp_path):
    path = capture(tmp_path, ["50,8", "10,2", "60,30", "30,4", "20,10", "40,20"])
    result = profile.task_statistics(path, repeats=3, tasks_per_call=2)
    assert result["metric"] == "native_task_sum_us"
    assert result["median_us"] == 24
    assert (result["min_us"], result["max_us"]) == (12, 38)
    assert result["per_task"] == [
        {"position": 1, "median_us": 4, "min_us": 2, "max_us": 8},
        {"position": 2, "median_us": 20, "min_us": 10, "max_us": 30},
    ]


def test_sum_median_is_not_replaced_by_sum_of_position_medians(tmp_path):
    path = capture(tmp_path, ["1,1", "2,100", "3,2", "4,2", "5,100", "6,1"])
    result = profile.task_statistics(path, repeats=3, tasks_per_call=2)
    assert result["median_us"] == 101
    assert sum(task["median_us"] for task in result["per_task"]) == 4


def test_single_task_breakdown_agrees_with_total(tmp_path):
    result = profile.task_statistics(
        capture(tmp_path, ["2,9", "1,3"]), repeats=2, tasks_per_call=1
    )
    assert result["per_task"] == [
        {"position": 1, "median_us": 6, "min_us": 3, "max_us": 9}
    ]
    assert result["median_us"] == 6


@pytest.mark.parametrize("reader", [profile.task_samples, profile.task_statistics])
def test_coincident_starts_do_not_invent_an_execution_order(tmp_path, reader):
    path = capture(tmp_path, ["1,2", "1,3"])
    with pytest.raises(ValueError, match="order ambiguous"):
        reader(path, repeats=1, tasks_per_call=2)


@pytest.mark.parametrize(
    "options",
    [
        {"repeats": 0, "tasks_per_call": 1},
        {"repeats": 1, "tasks_per_call": 0},
        {"repeats": True, "tasks_per_call": 1},
    ],
)
def test_statistics_reject_invalid_call_counts_before_reading(options):
    with pytest.raises(ValueError, match="positive integers"):
        profile.task_statistics("unread.csv", **options)


def test_breakdown_still_rejects_incomplete_capture(tmp_path):
    with pytest.raises(ValueError, match="expected 4 native tasks"):
        profile.task_statistics(
            capture(tmp_path, ["1,2", "2,3", "3,4"]), repeats=2, tasks_per_call=2
        )


@pytest.mark.parametrize("order", ["AB", "BA"])
def test_paired_ratios_use_complete_calls_with_unequal_task_counts(tmp_path, order):
    pairs = [([1, 100], [50]), ([2, 2], [8]), ([100, 1], [90])]
    values = [
        duration
        for before, after in pairs
        for duration in (before + after if order == "AB" else after + before)
    ]
    rows = [f"{number},{value}" for number, value in enumerate(values)]
    result = profile.paired_task_statistics(
        capture(tmp_path, rows), repeats=3, before_tasks=2, after_tasks=1, order=order
    )
    assert result["order"] == order and result["pairs"] == 3
    assert result["baseline"]["median_us"] == 101
    assert result["candidate"]["median_us"] == 50
    assert result["median_ratio"] == pytest.approx(90 / 101)
    assert result["candidate_wins"] == 2 and result["win_fraction"] == pytest.approx(
        2 / 3
    )


def test_paired_ties_are_not_counted_as_wins(tmp_path):
    result = profile.paired_task_statistics(
        capture(tmp_path, ["1,2", "2,2"]), repeats=1, before_tasks=1, after_tasks=1
    )
    assert result["median_ratio"] == 1 and result["candidate_wins"] == 0


def test_paired_order_must_be_explicit_and_known(tmp_path):
    with pytest.raises(ValueError, match="order must be AB"):
        profile.paired_task_statistics(
            capture(tmp_path, ["1,2", "2,3"]),
            repeats=1,
            before_tasks=1,
            after_tasks=1,
            order="guess",
        )


@pytest.mark.parametrize(
    "options",
    [
        {"tasks_per_call": (1,)},
        {"tasks_per_call": (0, 1)},
        {"tasks_per_call": (1, True)},
        {"repeats": 0},
    ],
)
def test_bad_pair_configuration_fails_before_runtime_import(options):
    with pytest.raises(ValueError):
        profile.compare_native(lambda: None, lambda: None, **options)


def test_native_pair_callbacks_retain_each_capture_order(monkeypatch):
    from types import SimpleNamespace

    trace = []
    callbacks = []
    runtime = (
        SimpleNamespace(
            npu=SimpleNamespace(synchronize=lambda: None), __version__="test"
        ),
        SimpleNamespace(__version__="test"),
        None,
        None,
    )
    monkeypatch.setattr(profile, "_runtime", lambda: runtime)

    def record_capture(callback, **options):
        callbacks.append(callback)
        return []

    monkeypatch.setattr(profile, "_capture_calls", record_capture)
    monkeypatch.setattr(profile, "_paired_statistics", lambda *args: {})
    profile.compare_native(
        lambda: trace.append("A"), lambda: trace.append("B"), warmup=1, repeats=1
    )
    trace.clear()
    for captured in callbacks:
        captured()
    assert trace == ["A", "B", "B", "A"]


def test_pipe_counters_follow_task_positions_even_with_identical_symbols(tmp_path):
    path = capture(
        tmp_path,
        [
            "4,8,kernel,AI_CORE,4,1.0,40,75",
            "1,2,kernel,AI_CORE,4,0.1,10,25",
            "3,6,kernel,AI_CORE,4,0.3,30,35",
            "2,4,kernel,AI_CORE,4,0.8,20,65",
        ],
        "Start Time(us),Duration(us),Name,Accelerator Core,Block Num,"
        "aic_mac_ratio,aicore_time(us),cube_utilization(%)\n",
    )
    first, second = profile.task_pipe_statistics(path, repeats=2, tasks_per_call=2)
    assert first["position"] == 1 and second["position"] == 2
    assert first["identity"]["Name"] == ["kernel"]
    assert first["identity"]["Block Num"] == ["4"]
    assert first["counters"]["aic_mac_ratio"] == {
        "median": 0.2,
        "samples": 2,
        "missing_samples": 0,
    }
    assert second["counters"]["aic_mac_ratio"]["median"] == 0.9
    assert first["counters"]["aicore_time(us)"]["median"] == 20
    assert second["counters"]["cube_utilization(%)"]["median"] == 70


@pytest.mark.parametrize(
    "values,median,count",
    [
        (["NA", "N/A", "", "nan", "inf"], None, 0),
        (["NA", "", "0", "0.4", "-inf"], 0.2, 2),
    ],
)
def test_missing_pipe_samples_are_explicit_and_real_zero_is_retained(
    tmp_path, values, median, count
):
    path = capture(
        tmp_path,
        [f"{i},1,{value}" for i, value in enumerate(values)],
        "Start Time(us),Duration(us),aiv_vec_ratio\n",
    )
    result = profile.task_pipe_statistics(path, repeats=5, tasks_per_call=1)[0]
    assert result["available"] == bool(count)
    assert result["counters"]["aiv_vec_ratio"] == {
        "median": median,
        "samples": count,
        "missing_samples": 5 - count,
    }


def test_capture_without_pipe_columns_reports_unavailable(tmp_path):
    result = profile.task_pipe_statistics(
        capture(tmp_path, ["1,2"]), repeats=1, tasks_per_call=1
    )[0]
    assert result["available"] is False and result["counters"] == {}


def test_unknown_counter_format_is_not_reinterpreted(tmp_path):
    path = capture(tmp_path, ["1,2,15%"], "Start Time(us),Duration(us),aic_mac_ratio\n")
    with pytest.raises(ValueError, match="unsupported counter value"):
        profile.task_pipe_statistics(path, repeats=1, tasks_per_call=1)


@pytest.mark.parametrize(
    "rows,message",
    [
        (["1,2,0.3"], "expected 2 native tasks"),
        (["1,2,0.3", "1,3,0.4"], "order ambiguous"),
    ],
)
def test_pipe_capture_keeps_count_and_order_guards(tmp_path, rows, message):
    path = capture(tmp_path, rows, "Start Time(us),Duration(us),aic_mac_ratio\n")
    with pytest.raises(ValueError, match=message):
        profile.task_pipe_statistics(path, repeats=2, tasks_per_call=1)


def test_explicit_pipe_collection_requires_the_metric_capability(monkeypatch):
    import sys
    from types import SimpleNamespace

    options = []
    profiler = SimpleNamespace(
        _ExperimentalConfig=lambda **kwargs: options.append(kwargs),
        ExportType=SimpleNamespace(Text="text"),
        ProfilerLevel=SimpleNamespace(Level1="level1"),
        ProfilerActivity=object(),
        profile=object(),
        tensorboard_trace_handler=object(),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(__version__="probe", profiler=profiler),
    )
    profile._runtime()
    assert "aic_metrics" not in options[-1]
    with pytest.raises(RuntimeError, match="unsupported profiler configuration"):
        profile._runtime(pipe_metrics=True)
    profiler.AiCMetrics = SimpleNamespace(PipeUtilization="pipe")
    profile._runtime(pipe_metrics=True)
    assert options[-1]["aic_metrics"] == "pipe"


def test_memory_group_does_not_fall_back_to_pipe_collection(monkeypatch):
    import sys
    from types import SimpleNamespace

    options = []
    profiler = SimpleNamespace(
        _ExperimentalConfig=lambda **kwargs: options.append(kwargs),
        ExportType=SimpleNamespace(Text="text"),
        ProfilerLevel=SimpleNamespace(Level1="level1"),
        ProfilerActivity=object(),
        profile=object(),
        tensorboard_trace_handler=object(),
        AiCMetrics=SimpleNamespace(PipeUtilization="pipe"),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(__version__="probe", profiler=profiler),
    )
    with pytest.raises(RuntimeError, match="unsupported profiler configuration"):
        profile._runtime(metric="Memory")
    profiler.AiCMetrics.Memory = "memory"
    profile._runtime(metric="Memory")
    assert options[-1]["aic_metrics"] == "memory"


@pytest.mark.parametrize(
    "options",
    [{"metric": "Unknown"}, {"metric": None}, {"metric": "Memory", "repeats": 0}],
)
def test_metric_request_validation_precedes_device_runtime(monkeypatch, options):
    import builtins

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name not in ("torch", "torch_npu")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(ValueError):
        profile.measure_metrics(lambda: None, **options)


def test_l2_counter_zeros_and_missing_samples_remain_distinct(tmp_path):
    path = capture(
        tmp_path,
        ["1,3,0,N/A", "2,4,0,1"],
        "Start Time(us),Duration(us),aiv_read_local_l2_hit,aiv_read_local_l2_miss\n",
    )
    counters = profile.task_pipe_statistics(path, repeats=2, tasks_per_call=1)[0][
        "counters"
    ]
    assert counters["aiv_read_local_l2_hit"] == {
        "median": 0,
        "samples": 2,
        "missing_samples": 0,
    }
    assert counters["aiv_read_local_l2_miss"] == {
        "median": 1,
        "samples": 1,
        "missing_samples": 1,
    }


def fake_capture_runtime(monkeypatch, tmp_path, rows):
    from contextlib import contextmanager
    from types import SimpleNamespace

    monkeypatch.setattr(profile.tempfile, "tempdir", str(tmp_path))

    @contextmanager
    def capture_profile(**kwargs):
        directory = Path(kwargs["on_trace_ready"])
        (directory / "raw-data.bin").write_bytes(b"capture evidence")
        try:
            yield
        finally:
            if rows is not None:
                capture(directory, rows)

    profiler = SimpleNamespace(
        ProfilerActivity=SimpleNamespace(NPU="npu"),
        profile=capture_profile,
        tensorboard_trace_handler=lambda directory, **kwargs: directory,
    )
    torch = SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: None))
    return torch, None, profiler, None


def test_metric_collection_keeps_complete_call_sums_and_missing_counter_scope(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    torch, _, profiler, config = fake_capture_runtime(
        monkeypatch, tmp_path, ["1,1", "2,2", "3,3", "4,4"]
    )
    torch.__version__ = "probe"
    monkeypatch.setattr(
        profile,
        "_runtime",
        lambda **kwargs: (
            torch,
            SimpleNamespace(__version__="probe"),
            profiler,
            config,
        ),
    )
    calls = []
    result = profile.measure_metrics(
        lambda: calls.append(1), metric="Memory", warmup=2, repeats=2, tasks_per_call=2
    )
    assert len(calls) == 4
    assert result["median_us"] == 5 and result["tasks_per_call"] == 2
    assert (
        result["counter_group"] == "Memory"
        and result["counter_scope"] == "task_aggregate"
    )
    assert all(not item["available"] for item in result["counters"])
    assert not list(tmp_path.glob("native-task-profile-*"))


@pytest.mark.parametrize(
    "failure,exception",
    [
        ("missing_export", RuntimeError),
        ("incomplete_tasks", ValueError),
        ("callback", LookupError),
    ],
)
def test_failed_capture_preserves_evidence_and_original_exception(
    monkeypatch, tmp_path, caplog, failure, exception
):
    rows = None if failure == "missing_export" else ["1,2"]
    runtime = fake_capture_runtime(monkeypatch, tmp_path, rows)

    def call():
        if failure == "callback":
            raise LookupError("callback failure")

    with pytest.raises(exception):
        profile._capture_calls(call, repeats=2, tasks_per_call=1, runtime=runtime)
    directories = list(tmp_path.glob("native-task-profile-*"))
    assert len(directories) == 1
    assert (directories[0] / "raw-data.bin").read_bytes() == b"capture evidence"
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.name == profile.__name__
    assert record.levelname == "ERROR"
    assert str(directories[0]) in record.getMessage()
    assert "no device timing established" in record.getMessage()


@pytest.mark.parametrize("pipe_metrics", [False, True])
def test_successful_capture_returns_validated_samples_and_removes_artifacts(
    monkeypatch, tmp_path, pipe_metrics
):
    runtime = fake_capture_runtime(monkeypatch, tmp_path, ["1,2", "3,4"])
    result = profile._capture_calls(
        lambda: None,
        repeats=2,
        tasks_per_call=1,
        runtime=runtime,
        pipe_metrics=pipe_metrics,
    )
    calls = result[0] if pipe_metrics else result
    assert [sum(call) for call in calls] == [2, 4]
    assert not list(tmp_path.glob("native-task-profile-*"))


@pytest.mark.parametrize("options", [{"warmup": 0}, {"repeats": -1}, {"repeats": True}])
def test_wall_bad_counts_fail_before_runtime_import(options):
    with pytest.raises(ValueError, match="positive integers"):
        profile.compare_wall(lambda: None, lambda: None, **options)


def test_wall_pairs_include_completion_but_exclude_prior_stream_work(monkeypatch):
    import sys
    from types import SimpleNamespace

    clock = [0]
    calls = []

    def sync():
        clock[0] += 1000

    def callback(name, elapsed):
        calls.append(name)
        clock[0] += elapsed

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(__version__="test", npu=SimpleNamespace(synchronize=sync)),
    )
    # No profiler API is needed for this separately named metric.
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(__version__="test"))
    monkeypatch.setattr(profile.time, "perf_counter_ns", lambda: clock[0])
    result = profile.compare_wall(
        lambda: callback("A", 8000), lambda: callback("B", 4000), warmup=2, repeats=2
    )
    assert calls == list("ABABABABBABA")
    assert calls.count("A") == calls.count("B") == 2 + 2 * 2
    assert result["metric"] == "synchronized_wall_us"
    assert [item["order"] for item in result["orders"]] == ["AB", "BA"]
    for item in result["orders"]:
        assert item["baseline"]["median_us"] == 9
        assert item["candidate"]["median_us"] == 5
        assert item["median_ratio"] == pytest.approx(5 / 9)
        assert item["candidate_wins"] == item["pairs"] == 2


def test_wall_failed_callback_does_not_produce_timing(monkeypatch):
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: None)),
    )
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())

    def failed():
        raise RuntimeError("device failure")

    with pytest.raises(RuntimeError, match="device failure"):
        profile.compare_wall(failed, lambda: None)


@pytest.mark.parametrize(
    "options", [{"warmup": 0}, {"repeats": True}, {"tasks_per_call": (1, 1)}]
)
def test_configuration_screen_rejects_bad_counts_before_runtime(options):
    with pytest.raises(ValueError, match="positive integers"):
        profile.compare_configurations(
            lambda: None, {"config": lambda: None}, **options
        )


@pytest.mark.parametrize(
    "candidates", [{}, [], {"": lambda: None}, {1: lambda: None}, {"x": None}]
)
def test_configuration_screen_requires_named_callables(candidates):
    with pytest.raises(ValueError):
        profile.compare_configurations(lambda: None, candidates)


@pytest.mark.parametrize("missing_task", [False, True])
def test_configuration_screen_keeps_labels_call_sums_and_reversed_pair_order(
    monkeypatch, tmp_path, missing_task
):
    from contextlib import contextmanager
    from types import SimpleNamespace

    active = []
    captures = []
    invocations = []
    monkeypatch.setattr(profile.tempfile, "tempdir", str(tmp_path))

    @contextmanager
    def recording(**kwargs):
        rows = []
        active.append(rows)
        try:
            yield
        finally:
            active.pop()
            captures.append(rows)
            # CSV ordering is independent of actual task start order.
            capture(Path(kwargs["on_trace_ready"]), list(reversed(rows)))

    def callback(name, durations):
        invocations.append(name)
        if active:
            if missing_task and name == "fast":
                durations = durations[:-1]
            for duration in durations:
                active[-1].append(f"{len(active[-1]) + 1},{duration}")

    torch = SimpleNamespace(
        __version__="test", npu=SimpleNamespace(synchronize=lambda: None)
    )
    torch_npu = SimpleNamespace(__version__="test")
    profiler = SimpleNamespace(
        ProfilerActivity=SimpleNamespace(NPU="npu"),
        profile=recording,
        tensorboard_trace_handler=lambda directory, **kwargs: directory,
    )
    monkeypatch.setattr(profile, "_runtime", lambda: (torch, torch_npu, profiler, None))

    def baseline():
        return callback("base", (10, 90))

    candidates = {
        "fast": lambda: callback("fast", (20, 30)),
        "slow": lambda: callback("slow", (80, 120)),
    }
    if missing_task:
        with pytest.raises(ValueError, match="expected 24 native tasks, got 21"):
            profile.compare_configurations(
                baseline, candidates, warmup=2, repeats=3, tasks_per_call=2
            )
        assert len(list(tmp_path.glob("native-task-profile-*"))) == 1
        return

    result = profile.compare_configurations(
        baseline, candidates, warmup=2, repeats=3, tasks_per_call=2
    )
    assert len(captures) == 2
    assert invocations == (
        ["base", "fast", "slow"] * 2
        + ["base", "fast", "base", "slow"] * 3
        + ["slow", "base", "fast", "base"] * 3
    )
    assert result["candidate_order"] == {"AB": ["fast", "slow"], "BA": ["slow", "fast"]}
    assert result["metric"] == "native_task_sum_us" and result["tasks_per_call"] == 2
    for name, duration, ratio, wins in [("fast", 50, 0.5, 3), ("slow", 200, 2.0, 0)]:
        orders = result["candidates"][name]["orders"]
        assert [item["order"] for item in orders] == ["AB", "BA"]
        for item in orders:
            assert item["baseline"]["median_us"] == 100
            assert item["candidate"]["median_us"] == duration
            assert item["median_ratio"] == ratio and item["candidate_wins"] == wins
            assert item["pairs"] == 3
    assert not list(tmp_path.glob("native-task-profile-*"))
