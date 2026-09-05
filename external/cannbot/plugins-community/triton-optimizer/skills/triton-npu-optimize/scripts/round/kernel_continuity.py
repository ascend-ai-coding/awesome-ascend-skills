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

import re
from dataclasses import dataclass
from pathlib import Path

_TRITON_IMPORT_RE = re.compile(r"(?m)^\s*(?:import\s+triton\b|from\s+triton\b)")
_TRITON_DECORATOR_RE = re.compile(r"(?m)^\s*@\s*triton\.(?:jit|autotune|heuristics)\b")
_TRITON_LANGUAGE_RE = re.compile(r"(?m)\btriton\.language\b|\btl\.")
_TRITON_LAUNCH_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\s*\[[^\]]+\]\s*\(", re.DOTALL)

_TILELANG_IMPORT_RE = re.compile(r"(?m)^\s*(?:import\s+tilelang\b|from\s+tilelang\b)")
_TILELANG_DECORATOR_RE = re.compile(r"(?m)^\s*@\s*(?:T\.prim_func|jit)\b")
_TILELANG_LANGUAGE_RE = re.compile(r"(?m)\btilelang\.language\b|\bT\.\b")
_TILELANG_LAUNCH_RE = re.compile(r"(?m)with\s+T\.Kernel\s*\([^)]*\)\s*as\s+[\w\(]")


@dataclass(frozen=True)
class KernelContinuityResult:
    ok: bool
    reason: str | None


def analyze_triton_kernel_continuity(operator_path: Path) -> KernelContinuityResult:
    source = operator_path.read_text(encoding="utf-8")
    triton_check_pass = _has_kernel_launch(source, _TRITON_LAUNCH_RE) and _has_triton_signal(source)
    tilelang_check_pass = _has_kernel_launch(source, _TILELANG_LAUNCH_RE) and _has_tilelang_signal(source)

    if triton_check_pass or tilelang_check_pass:
        return KernelContinuityResult(ok=True, reason=None)

    return KernelContinuityResult(
        ok=False,
        reason="round operator no longer preserves a recognizable Ascend kernel launch path",
    )


def _has_kernel_launch(source: str, launch_re: re.Pattern[str]) -> bool:
    return bool(launch_re.search(source))


def _has_triton_signal(source: str) -> bool:
    return bool(
        _TRITON_IMPORT_RE.search(source)
        or _TRITON_DECORATOR_RE.search(source)
        or _TRITON_LANGUAGE_RE.search(source)
    )


def _has_tilelang_signal(source: str) -> bool:
    return bool(
        _TILELANG_IMPORT_RE.search(source)
        or _TILELANG_DECORATOR_RE.search(source)
        or _TILELANG_LANGUAGE_RE.search(source)
    )
