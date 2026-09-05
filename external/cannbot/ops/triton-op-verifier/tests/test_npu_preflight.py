#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
# ----------------------------------------------------------------------------------------------------------
"""test_npu_preflight.py — pytest UT for npu_preflight.py.

Run:  pytest test_npu_preflight.py -v
      (or: python3 -m pytest test_npu_preflight.py -v)

Tests runtime/device/memory readiness decisions and CLI exit-code mapping.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import npu_preflight


class FakeNPU:
    def __init__(self, *, available=True, count=1, current=0, free=8 * 1024**3, total=16 * 1024**3):
        self.available = available
        self.count = count
        self.current = current
        self.free = free
        self.total = total

    def is_available(self):
        return self.available

    def device_count(self):
        return self.count

    def current_device(self):
        return self.current

    def mem_get_info(self, device_index):
        assert device_index == self.current
        return self.free, self.total


def _runtime(npu: FakeNPU):
    torch = SimpleNamespace(npu=npu)
    torch_npu = SimpleNamespace(npu=npu)

    def import_module(name):
        if name == "torch":
            return torch
        if name == "torch_npu":
            return torch_npu
        raise ModuleNotFoundError(name)

    return import_module


def _smi(*args, **kwargs):
    return SimpleNamespace(returncode=0, stdout="NPU Arch: DAV_2201\n", stderr="")


def test_ready_when_runtime_and_memory_are_available():
    npu = FakeNPU()
    result = npu_preflight.run_preflight(
        min_free_mb=1024,
        import_module=_runtime(npu),
        run_command=_smi,
    )

    assert result["status"] == "ready"
    assert result["exit_code"] == npu_preflight.EXIT_READY
    assert result["checks"][-1]["status"] == "pass"
    memory = next(item for item in result["checks"] if item["name"] == "memory")
    assert memory["free_mb"] == 8192.0
    assert all("duration_ms" in item for item in result["checks"])
    assert result["duration_ms"] >= 0


def test_not_ready_when_free_memory_is_below_threshold():
    npu = FakeNPU(free=512 * 1024**2)
    result = npu_preflight.run_preflight(
        min_free_mb=1024,
        import_module=_runtime(npu),
        run_command=_smi,
    )

    assert result["status"] == "not_ready"
    assert result["exit_code"] == npu_preflight.EXIT_NOT_READY
    memory = next(item for item in result["checks"] if item["name"] == "memory")
    assert memory["status"] == "fail"


def test_indeterminate_when_memory_cannot_be_read():
    npu = FakeNPU()

    def broken_mem_get_info(device_index):
        raise RuntimeError("memory query unavailable")

    npu.mem_get_info = broken_mem_get_info
    result = npu_preflight.run_preflight(
        import_module=_runtime(npu),
        run_command=_smi,
    )

    assert result["status"] == "indeterminate"
    assert result["exit_code"] == npu_preflight.EXIT_INDETERMINATE
    memory = next(item for item in result["checks"] if item["name"] == "memory")
    assert memory["status"] == "unknown"


def test_not_ready_when_npu_is_unavailable():
    npu = FakeNPU(available=False)
    result = npu_preflight.run_preflight(
        import_module=_runtime(npu),
        run_command=_smi,
    )

    assert result["status"] == "not_ready"
    assert result["exit_code"] == npu_preflight.EXIT_NOT_READY
    available = next(item for item in result["checks"] if item["name"] == "npu_available")
    assert available["status"] == "fail"
    memory = next(item for item in result["checks"] if item["name"] == "memory")
    assert memory["status"] == "skipped"


def test_smi_failure_is_reported_as_warning_without_masking_runtime_readiness():
    npu = FakeNPU()

    def missing_smi(*args, **kwargs):
        raise FileNotFoundError("npu-smi")

    result = npu_preflight.run_preflight(
        import_module=_runtime(npu),
        run_command=missing_smi,
    )

    assert result["status"] == "ready"
    smi = next(item for item in result["checks"] if item["name"] == "npu_smi")
    assert smi["status"] == "warn"


def test_invalid_device_is_not_ready():
    npu = FakeNPU(count=1, current=1)
    result = npu_preflight.run_preflight(
        import_module=_runtime(npu),
        run_command=_smi,
    )

    assert result["status"] == "not_ready"
    device = next(item for item in result["checks"] if item["name"] == "device_visibility")
    assert device["status"] == "fail"
    memory = next(item for item in result["checks"] if item["name"] == "memory")
    assert memory["status"] == "skipped"


def test_missing_runtime_import_is_not_ready():
    def missing_runtime(name):
        raise ModuleNotFoundError(name)

    result = npu_preflight.run_preflight(import_module=missing_runtime, run_command=_smi)

    assert result["status"] == "not_ready"
    assert result["exit_code"] == npu_preflight.EXIT_NOT_READY


def test_writes_result_json(tmp_path: Path):
    npu = FakeNPU()
    result = npu_preflight.run_preflight(import_module=_runtime(npu), run_command=_smi)
    output_path = tmp_path / "nested" / "npu_preflight.json"

    npu_preflight.write_preflight_result(result, output_path)

    assert output_path.is_file()
    assert npu_preflight.json.loads(output_path.read_text())["status"] == "ready"


def test_loads_options_from_config(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text('{"npu_preflight": {"min_free_mb": 256, "smi_timeout_seconds": 2}}')

    assert npu_preflight.load_preflight_options(str(config_path)) == {
        "min_free_mb": 256.0,
        "smi_timeout_seconds": 2.0,
    }


def test_smi_timeout_is_a_warning():
    npu = FakeNPU()

    def timeout(*args, **kwargs):
        raise npu_preflight.subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    result = npu_preflight.run_preflight(
        import_module=_runtime(npu),
        run_command=timeout,
    )

    assert result["status"] == "ready"
    smi = next(item for item in result["checks"] if item["name"] == "npu_smi")
    assert smi["status"] == "warn"
