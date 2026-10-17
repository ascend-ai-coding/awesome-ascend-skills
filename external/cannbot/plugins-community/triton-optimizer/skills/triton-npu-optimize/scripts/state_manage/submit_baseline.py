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
from datetime import datetime, timezone
from pathlib import Path

from baseline.check import check_baseline
from shared.cli import print_check_result, print_workflow_failure
from state_manage.state_machine import bootstrap_state, mark_baseline_passed


def build_parser(*, prog_name: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog_name or Path(__file__).name)
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("submit-baseline")
    baseline.add_argument("--baseline-dir", required=True)
    return parser


def _workflow_failure_guideline(message: str) -> str:
    if "workflow state" in message:
        return (
            "The temporary optimize workflow state is invalid. Do not continue to round work. "
            "Ask the runner to restart the optimize session so the runner-managed workflow state "
            "can be rebuilt cleanly."
        )
    return (
        "Baseline validation passed, but workflow-state advancement failed. Restart the "
        "optimize session before continuing."
    )


def _bootstrap_missing_workflow_state(state_path: Path) -> None:
    bootstrap_state(
        state_path,
        run_id=_bootstrap_run_id(),
        baseline_reused=False,
    )


def _bootstrap_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"submit-baseline-{timestamp}"


def main(argv: list[str] | None = None, *, prog_name: str | None = None) -> int:
    parser = build_parser(prog_name=prog_name)
    args = parser.parse_args(argv)
    baseline_dir = Path(args.baseline_dir).expanduser().resolve()
    result = check_baseline(baseline_dir)
    state_path = baseline_dir.parent / ".triton-agent" / "state.json"
    if result.status == "pass":
        try:
            if not state_path.exists():
                _bootstrap_missing_workflow_state(state_path)
            mark_baseline_passed(state_path)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            return print_workflow_failure(
                kind="baseline",
                issue=str(exc),
                guideline=_workflow_failure_guideline(str(exc)),
            )
    return print_check_result(result)


if __name__ == "__main__":
    raise SystemExit(main())
