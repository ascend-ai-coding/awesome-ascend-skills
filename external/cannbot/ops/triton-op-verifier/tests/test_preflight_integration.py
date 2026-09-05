#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
# ----------------------------------------------------------------------------------------------------------
"""test_preflight_integration.py — pytest integration tests for NPU preflight gates.

Run:  pytest test_preflight_integration.py -v
      (or: python3 -m pytest test_preflight_integration.py -v)

Tests verify.py and benchmark.py blocked execution paths plus result persistence.
"""

from __future__ import annotations

import json
from pathlib import Path

import benchmark
import verify


BLOCKED_PREFLIGHT = {
    "status": "not_ready",
    "exit_code": 1,
    "duration_ms": 1.0,
    "checks": [
        {
            "name": "memory",
            "status": "fail",
            "duration_ms": 0.1,
            "error": "available NPU memory is below the configured minimum",
        }
    ],
}


def test_verify_writes_structured_b_class_result_before_loading_modules(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, "_check_baseline_integrity", None)
    monkeypatch.setattr(verify, "run_preflight", lambda **kwargs: BLOCKED_PREFLIGHT)

    passed, total = verify.verify_implementations("demo", str(tmp_path))

    assert (passed, total) == (0, 0)
    result = json.loads((tmp_path / "verify_result.json").read_text())
    assert result["failure_class"] == "B"
    assert result["npu_preflight"]["status"] == "not_ready"
    assert result["failures"][0]["error_type"] == "NpuPreflightError"
    assert json.loads((tmp_path / "npu_preflight.json").read_text())["status"] == "not_ready"


def test_benchmark_blocks_before_runtime_import_and_serializes_result(tmp_path, monkeypatch):
    calls = []

    def fake_preflight(**kwargs):
        calls.append(kwargs)
        return BLOCKED_PREFLIGHT

    monkeypatch.setattr(benchmark, "run_preflight", fake_preflight)
    config = benchmark.BenchmarkConfig(op_name="demo", verify_dir=str(tmp_path))

    result = benchmark.benchmark_implementations(config)
    result_dict = benchmark.result_to_dict(result)

    assert len(calls) == 1
    assert result.failure_class == "B"
    assert result.total_cases == 0
    assert result.npu_preflight["status"] == "not_ready"
    assert result_dict["failure_class"] == "B"
    assert result_dict["npu_preflight"]["status"] == "not_ready"
    assert (tmp_path / "npu_preflight.json").is_file()


def test_preflight_result_writer_creates_parent_directories(tmp_path):
    result_path = tmp_path / "phase" / "verify" / "npu_preflight.json"
    verify.write_preflight_result(BLOCKED_PREFLIGHT, result_path)

    assert result_path.is_file()
    assert json.loads(result_path.read_text())["exit_code"] == 1
