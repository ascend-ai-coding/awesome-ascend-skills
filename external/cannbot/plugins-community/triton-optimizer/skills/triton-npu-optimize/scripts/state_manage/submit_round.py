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
from pathlib import Path

from round.check import check_round
from shared.cli import build_check_payload, print_check_result, print_json_payload, print_workflow_failure
from shared.results import build_check_result
from state_manage.state_machine import complete_round


def build_parser(*, prog_name: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog_name or Path(__file__).name)
    subparsers = parser.add_subparsers(dest="command", required=True)

    round_parser = subparsers.add_parser("submit-round")
    round_parser.add_argument("--round-dir", required=True)
    round_parser.add_argument("--current-round", type=int, default=None)
    round_parser.add_argument("--final-round", type=int, default=None)
    round_parser.add_argument(
        "--optimize-target",
        choices=("kernel", "operator"),
        default=None,
    )
    return parser


def _workflow_failure_guideline(message: str) -> str:
    if (
        "workflow state is not available" in message
        or ".triton-agent/state.json" in message
    ):
        return (
            "Optimize workflow state is unavailable. Use the staged "
            "`triton-npu-optimize-state` skill's `submit-baseline` subcommand to repair "
            "session state, then reopen this `opt-round-N/` with `start-round` before "
            "retrying `submit-round`."
        )
    if (
        "workflow phase is awaiting_round_start" in message
        or "current_round=None" in message
        or "missing workflow state entry" in message
    ):
        return (
            "This round has not been formally started yet. Use the staged "
            "`triton-npu-optimize-state` skill's `start-round` subcommand for this "
            "`opt-round-N/` before running `submit-round`."
        )
    if "cannot complete non-active round" in message or "workflow state current_round=" in message:
        return (
            "The requested round is not the active workflow round. Finish the active round, or "
            "use `triton-npu-optimize-state` `start-round` to open the intended round "
            "before submitting it."
        )
    if "workflow state" in message:
        return (
            "The temporary optimize workflow state is invalid. Stop this attempt and restart "
            "the optimize session so the runner-managed workflow state is rebuilt cleanly."
        )
    return (
        "Round validation passed, but workflow-state completion failed. Repair the optimize "
        "session before continuing."
    )


def _missing_round_directory_payload(round_dir: Path) -> dict[str, object]:
    result = build_check_result(
        kind="round",
        status="fail",
        issues=(f"missing round directory: {round_dir.name}",),
        summary=(
            f"round check requires fixes: missing round directory: {round_dir.name}. "
            "Create or reopen the expected `opt-round-N/` directory before submitting the round."
        ),
    )
    return build_check_payload(result)


def main(argv: list[str] | None = None, *, prog_name: str | None = None) -> int:
    parser = build_parser(prog_name=prog_name)
    args = parser.parse_args(argv)
    round_dir = Path(args.round_dir).expanduser().resolve()
    if not round_dir.is_dir():
        print_json_payload(_missing_round_directory_payload(round_dir))
        return 1

    result = check_round(
        round_dir,
        current_round=args.current_round,
        final_round=args.final_round,
        optimize_target=args.optimize_target,
    )
    state_path = round_dir.parent / ".triton-agent" / "state.json"
    if result.status == "pass":
        if not state_path.exists():
            return print_workflow_failure(
                kind="round",
                issue="optimize workflow state is not available",
                guideline=_workflow_failure_guideline("optimize workflow state is not available"),
            )
        try:
            complete_round(
                state_path,
                round_dir.name,
                current_round_arg=args.current_round,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            return print_workflow_failure(
                kind="round",
                issue=str(exc),
                guideline=_workflow_failure_guideline(str(exc)),
            )
    return print_check_result(result)


if __name__ == "__main__":
    raise SystemExit(main())
