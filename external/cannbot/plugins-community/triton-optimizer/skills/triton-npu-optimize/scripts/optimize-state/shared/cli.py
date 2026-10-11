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

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterator

from shared.models import OptimizeCheckResult


def print_json_payload(payload: dict[str, object]) -> None:
    """Keep machine-readable Agent output consistent across state commands."""
    import json

    print(json.dumps(payload, ensure_ascii=True))


def print_check_result(result: OptimizeCheckResult) -> int:
    print_json_payload(build_check_payload(result))
    return 0 if result.status == "pass" else 1


def print_workflow_failure(*, kind: str, issue: str, guideline: str) -> int:
    print_json_payload(
        build_workflow_failure_payload(kind=kind, issue=issue, guideline=guideline)
    )
    return 1


def build_check_payload(result: OptimizeCheckResult) -> dict[str, object]:
    payload: dict[str, object] = {
        "kind": result.kind,
        "status": result.status,
        "issues": list(result.issues),
        "guideline": result.summary,
    }
    if result.next_option is not None:
        payload["next_option"] = result.next_option
    return payload


def build_workflow_failure_payload(
    *,
    kind: str,
    issue: str,
    guideline: str,
) -> dict[str, object]:
    return {
        "kind": kind,
        "status": "fail",
        "issues": [issue],
        "guideline": guideline,
    }


@contextmanager
def temporary_module_path(path: Path) -> Iterator[None]:
    path_text = str(path)
    added = path_text not in sys.path
    if added:
        sys.path.insert(0, path_text)
    try:
        yield
    finally:
        if added:
            sys.path.remove(path_text)


def load_module_from_script(script_path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        with temporary_module_path(script_path.parent):
            spec.loader.exec_module(module)
    finally:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
    return module
