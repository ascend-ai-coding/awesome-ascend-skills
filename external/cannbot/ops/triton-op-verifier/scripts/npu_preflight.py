#!/usr/bin/env python3
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

import argparse
import dataclasses
import importlib
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable

from _log_utils import setup_logger as _setup_logger_shared


EXIT_READY = 0
EXIT_NOT_READY = 1
EXIT_INDETERMINATE = 2
EXIT_INTERNAL_ERROR = 3

DEFAULT_MIN_FREE_MB = 1.0
DEFAULT_SMI_TIMEOUT_SECONDS = 5.0

logger = logging.getLogger(__name__)


def _now() -> float:
    return time.perf_counter()


def _duration_ms(start: float, clock: Callable[[], float]) -> float:
    return round(max(0.0, (clock() - start) * 1000.0), 3)


def _check(name: str, fn: Callable[[], tuple[str, dict[str, Any]]], clock: Callable[[], float]) -> dict[str, Any]:
    started = clock()
    try:
        status, details = fn()
        result: dict[str, Any] = {"name": name, "status": status, "duration_ms": _duration_ms(started, clock)}
        result.update(details)
        return result
    except Exception as exc:
        return {
            "name": name,
            "status": "fail",
            "duration_ms": _duration_ms(started, clock),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _skipped(name: str, reason: str) -> dict[str, Any]:
    return {"name": name, "status": "skipped", "duration_ms": 0.0, "reason": reason}


@dataclasses.dataclass
class RuntimeImport:
    """torch / torch_npu 运行时导入结果：模块对象（失败为 None）与各自的检查记录。"""
    torch: Any | None
    torch_npu: Any | None
    torch_result: dict[str, Any]
    torch_npu_result: dict[str, Any]

    @property
    def imported(self) -> bool:
        return self.torch_result["status"] == "pass" and self.torch_npu_result["status"] == "pass"

    @property
    def import_checks(self) -> list[dict[str, Any]]:
        return [self.torch_result, self.torch_npu_result]


def _import_runtime(
    import_module: Callable[[str], Any],
    clock: Callable[[], float],
) -> RuntimeImport:
    def import_one(name: str) -> tuple[Any | None, dict[str, Any]]:
        started = clock()
        try:
            module = import_module(name)
            return module, {
                "name": f"{name}_import",
                "status": "pass",
                "duration_ms": _duration_ms(started, clock),
            }
        except Exception as exc:
            return None, {
                "name": f"{name}_import",
                "status": "fail",
                "duration_ms": _duration_ms(started, clock),
                "error": f"{type(exc).__name__}: {exc}",
            }

    torch, torch_result = import_one("torch")
    torch_npu, torch_npu_result = import_one("torch_npu")
    return RuntimeImport(torch, torch_npu, torch_result, torch_npu_result)


def _runtime_device(torch: Any, torch_npu: Any) -> Any | None:
    if torch is not None:
        candidate = getattr(torch, "npu", None)
        if candidate is not None:
            return candidate
    if torch_npu is not None:
        return getattr(torch_npu, "npu", None)
    return None


def _call_mem_get_info(npu: Any, device_index: int) -> tuple[Any, Any]:
    mem_get_info = getattr(npu, "mem_get_info")
    try:
        return mem_get_info(device_index)
    except TypeError:
        return mem_get_info()


def _valid_memory(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _run_smi(
    run_command: Callable[..., Any],
    timeout_seconds: float,
) -> tuple[str, dict[str, Any]]:
    try:
        completed = run_command(
            ["npu-smi", "info", "-m"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        return "warn", {"error": f"npu-smi not found: {exc}"}
    except subprocess.TimeoutExpired:
        return "warn", {"error": f"npu-smi timed out after {timeout_seconds:.3f}s"}
    except Exception as exc:
        return "warn", {"error": f"{type(exc).__name__}: {exc}"}

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    details: dict[str, Any] = {"returncode": completed.returncode}
    if stdout:
        details["stdout"] = stdout[-4000:]
    if stderr:
        details["stderr"] = stderr[-2000:]
    if completed.returncode != 0:
        details["error"] = "npu-smi returned a non-zero exit code"
        return "warn", details
    return "pass", details


def _device_memory_checks(
    npu: Any,
    torch_npu: Any,
    min_free_mb: float,
    clock: Callable[[], float],
) -> list[dict[str, Any]]:
    """运行时已就绪后的 npu_available / device_visibility / memory 级联检查。"""
    checks = [_check("npu_available", lambda: _check_npu_available(torch_npu), clock)]
    if checks[-1]["status"] != "pass":
        checks.extend(
            [
                _skipped("device_visibility", "NPU runtime is unavailable"),
                _skipped("memory", "NPU runtime is unavailable"),
            ]
        )
        return checks

    device_result = _check("device_visibility", lambda: _check_device(npu), clock)
    checks.append(device_result)
    if device_result["status"] != "pass":
        checks.append(_skipped("memory", "NPU device is unavailable"))
        return checks

    checks.append(
        _check(
            "memory",
            lambda: _check_memory(npu, int(device_result["device_index"]), min_free_mb),
            clock,
        )
    )
    return checks


def _runtime_readiness_checks(
    runtime: RuntimeImport,
    min_free_mb: float,
    clock: Callable[[], float],
) -> list[dict[str, Any]]:
    """按运行时导入结果分流的 npu_available / device_visibility / memory 检查。"""
    if not runtime.imported:
        return [
            _skipped("npu_available", "required runtime import failed"),
            _skipped("device_visibility", "required runtime import failed"),
            _skipped("memory", "required runtime import failed"),
        ]

    npu = _runtime_device(runtime.torch, runtime.torch_npu)
    if npu is None:
        return [
            _skipped("npu_available", "torch_npu runtime is unavailable"),
            _skipped("device_visibility", "NPU runtime is unavailable"),
            _skipped("memory", "NPU runtime is unavailable"),
        ]

    return _device_memory_checks(npu, runtime.torch_npu, min_free_mb, clock)


def _summarize_status(checks: list[dict[str, Any]]) -> tuple[str, int]:
    """由各项检查结果汇总整体状态与退出码（npu_smi 仅为警告，不参与判定）。"""
    required_checks = [result for result in checks if result["name"] != "npu_smi"]
    hard_fail = any(result["status"] in {"fail", "skipped"} for result in required_checks)
    unknown = any(result["status"] == "unknown" for result in required_checks)
    if hard_fail:
        return "not_ready", EXIT_NOT_READY
    if unknown:
        return "indeterminate", EXIT_INDETERMINATE
    return "ready", EXIT_READY


def run_preflight(
    *,
    min_free_mb: float = DEFAULT_MIN_FREE_MB,
    smi_timeout_seconds: float = DEFAULT_SMI_TIMEOUT_SECONDS,
    import_module: Callable[[str], Any] = importlib.import_module,
    run_command: Callable[..., Any] = subprocess.run,
    clock: Callable[[], float] = _now,
) -> dict[str, Any]:
    started = clock()
    runtime = _import_runtime(import_module, clock)

    checks: list[dict[str, Any]] = runtime.import_checks
    checks.extend(_runtime_readiness_checks(runtime, min_free_mb, clock))
    checks.append(_check("npu_smi", lambda: _run_smi(run_command, smi_timeout_seconds), clock))

    status, exit_code = _summarize_status(checks)

    return {
        "status": status,
        "exit_code": exit_code,
        "duration_ms": _duration_ms(started, clock),
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        or os.environ.get("ASCEND_VISIBLE_DEVICES"),
        "checks": checks,
    }


def load_preflight_options(config_path: str | None = None) -> dict[str, float]:
    path = Path(config_path) if config_path else Path.cwd() / "config.json"
    try:
        with path.open(encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        config = {}
    options = config.get("npu_preflight", {}) if isinstance(config, dict) else {}
    if not isinstance(options, dict):
        options = {}
    return {
        "min_free_mb": float(options.get("min_free_mb", DEFAULT_MIN_FREE_MB)),
        "smi_timeout_seconds": float(options.get("smi_timeout_seconds", DEFAULT_SMI_TIMEOUT_SECONDS)),
    }


def write_preflight_result(result: dict[str, Any], output_path: str | os.PathLike[str]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _check_npu_available(torch_npu: Any) -> tuple[str, dict[str, Any]]:
    available = bool(torch_npu.npu.is_available())
    if not available:
        return "fail", {"error": "torch_npu.npu.is_available() returned false"}
    return "pass", {}


def _check_device(npu: Any) -> tuple[str, dict[str, Any]]:
    device_count_fn = getattr(npu, "device_count")
    current_device_fn = getattr(npu, "current_device")
    count = int(device_count_fn())
    current = int(current_device_fn())
    if count <= 0:
        return "fail", {"device_count": count, "error": "no visible NPU device"}
    if current < 0 or current >= count:
        return "fail", {"device_count": count, "device_index": current, "error": "current device index is invalid"}
    return "pass", {"device_count": count, "device_index": current}


def _check_memory(npu: Any, device_index: int, min_free_mb: float) -> tuple[str, dict[str, Any]]:
    try:
        free_bytes, total_bytes = _call_mem_get_info(npu, device_index)
    except Exception as exc:
        return "unknown", {"error": f"memory query failed: {type(exc).__name__}: {exc}"}
    if not _valid_memory(free_bytes) or not _valid_memory(total_bytes):
        return "unknown", {"error": "NPU free/total memory is unavailable or invalid"}
    free_bytes = float(free_bytes)
    total_bytes = float(total_bytes)
    if total_bytes <= 0 or free_bytes < 0 or free_bytes > total_bytes:
        return "unknown", {
            "free_bytes": free_bytes,
            "total_bytes": total_bytes,
            "error": "NPU memory values are invalid",
        }
    required_bytes = float(min_free_mb) * 1024 * 1024
    details = {
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "min_free_bytes": int(required_bytes),
        "free_mb": round(free_bytes / (1024 * 1024), 3),
        "total_mb": round(total_bytes / (1024 * 1024), 3),
    }
    if free_bytes < required_bytes:
        details["error"] = "available NPU memory is below the configured minimum"
        return "fail", details
    return "pass", details


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check Ascend NPU runtime and free-memory readiness.")
    parser.add_argument(
        "--min-free-mb", type=float, default=DEFAULT_MIN_FREE_MB, help="Minimum free NPU memory in MiB."
    )
    parser.add_argument(
        "--smi-timeout-sec",
        type=float,
        default=DEFAULT_SMI_TIMEOUT_SECONDS,
        help="npu-smi command timeout in seconds.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the JSON result.")
    return parser


def main() -> int:
    _setup_logger_shared(logger)
    args = _build_parser().parse_args()
    if not math.isfinite(args.min_free_mb) or args.min_free_mb < 0:
        logger.error("--min-free-mb must be a finite non-negative number")
        return EXIT_INDETERMINATE
    if not math.isfinite(args.smi_timeout_sec) or args.smi_timeout_sec <= 0:
        logger.error("--smi-timeout-sec must be a finite positive number")
        return EXIT_INDETERMINATE

    try:
        result = run_preflight(
            min_free_mb=args.min_free_mb,
            smi_timeout_seconds=args.smi_timeout_sec,
        )
    except Exception as exc:
        result = {
            "status": "indeterminate",
            "exit_code": EXIT_INTERNAL_ERROR,
            "duration_ms": 0.0,
            "checks": [],
            "error": f"{type(exc).__name__}: {exc}",
        }

    indent = 2 if args.pretty else None
    logger.info("%s", json.dumps(result, ensure_ascii=False, indent=indent, sort_keys=False))
    return int(result["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
