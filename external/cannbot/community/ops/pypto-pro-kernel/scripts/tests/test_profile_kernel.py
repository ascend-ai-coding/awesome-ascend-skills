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
