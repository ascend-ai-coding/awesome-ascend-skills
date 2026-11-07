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

from pathlib import Path


_BATCH_OPTIMIZE_EXCLUDED_PREFIXES = ("test_", "differential_test_", "bench_", "opt_")
_BATCH_OPTIMIZE_EXCLUDED_NAMES = {"__init__.py"}


def resolve_workspace_operator_file(workspace: Path) -> Path:
    candidates = [
        path for path in sorted(workspace.iterdir()) if is_workspace_operator_candidate(path)
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(f"no candidate operator file found in workspace: {workspace}")
    raise ValueError(f"multiple candidate operator files found in workspace: {workspace}")


def expected_round_operator_name(workspace: Path) -> str:
    operator_file = resolve_workspace_operator_file(workspace)
    return f"opt_{operator_file.name}"


def expected_round_perf_name(workspace: Path) -> str:
    operator_file = resolve_workspace_operator_file(workspace)
    return f"opt_{operator_file.stem}_perf.txt"


def is_workspace_operator_candidate(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix != ".py":
        return False
    if path.name in _BATCH_OPTIMIZE_EXCLUDED_NAMES:
        return False
    return not path.name.startswith(_BATCH_OPTIMIZE_EXCLUDED_PREFIXES)
