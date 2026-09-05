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
import json
from pathlib import Path

from state_manage.state_machine import (
    ANALYSIS_POLICIES,
    ROUND_STRATEGIES,
    set_current_round_state as set_current_round_state_in_workflow,
)


def build_parser(*, prog_name: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog_name or Path(__file__).name)
    subparsers = parser.add_subparsers(dest="command", required=True)

    update = subparsers.add_parser("set-current-round-state")
    update.add_argument("--round-strategy", choices=ROUND_STRATEGIES)
    update.add_argument("--analysis-policy", choices=ANALYSIS_POLICIES)
    update.add_argument("--reason", required=True)
    return parser


def _build_failure_payload(issue: str, guideline: str) -> dict[str, object]:
    return {
        "status": "fail",
        "issues": [issue],
        "guideline": guideline,
    }


def _workflow_failure_guideline(message: str) -> str:
    if (
        "workflow state is not available" in message
        or ".triton-agent/state.json" in message
    ):
        return (
            "Optimize workflow state is unavailable. Use the staged "
            "`triton-npu-optimize-state` skill's `submit-baseline` subcommand to repair "
            "session state, then reopen the intended `opt-round-N/` with `start-round` "
            "before retrying `set-current-round-state`."
        )
    if "no optimize round is currently active" in message:
        return (
            "No optimize round is currently active. Start the next round first before "
            "changing round strategy state."
        )
    if "state update would be a no-op" in message:
        return (
            "This state update would be a no-op. Keep the current round state or change "
            "the strategy or analysis policy before retrying."
        )
    if "analysis_policy cannot become shallower" in message:
        return (
            "The requested analysis_policy would become shallower. Keep the current or a "
            "deeper analysis policy within the same round."
        )
    if "set-current-round-state requires" in message:
        return "Provide --round-strategy and/or --analysis-policy together with --reason."
    if "workflow state" in message:
        return (
            "The temporary optimize workflow state is invalid. Stop this attempt and restart "
            "the optimize session so the runner can rebuild the temporary workflow state."
        )
    return (
        "This set-current-round-state request could not be applied. Repair the optimize "
        "session and retry."
    )


def _find_state_path_from_cwd(cwd: Path) -> Path | None:
    for candidate_dir in (cwd, *cwd.parents):
        state_path = candidate_dir / ".triton-agent" / "state.json"
        if state_path.exists():
            return state_path
    return None


def _run_state_update(args: argparse.Namespace) -> dict[str, object]:
    state_path = _find_state_path_from_cwd(Path.cwd())
    if state_path is None:
        raise RuntimeError("optimize workflow state is not available")
    return set_current_round_state_in_workflow(
        state_path,
        round_strategy=args.round_strategy,
        analysis_policy=args.analysis_policy,
        reason=args.reason,
    )


def _build_success_payload(workflow_result: dict[str, object]) -> dict[str, object]:
    round_name = str(workflow_result["round"])
    payload: dict[str, object] = {
        "status": "pass",
        "round": round_name,
        "guideline": (
            f"Round strategy state for {round_name} is now updated. "
            "Use the new state as the active same-round contract."
        ),
        "round_strategy": workflow_result["round_strategy"],
        "analysis_policy": workflow_result["analysis_policy"],
        "reason": workflow_result["reason"],
    }
    _copy_optional_fields(
        workflow_result,
        payload,
        "previous_round_strategy",
        "previous_analysis_policy",
        "warnings",
    )
    return payload


def _copy_optional_fields(
    source: dict[str, object],
    destination: dict[str, object],
    *field_names: str,
) -> None:
    for field_name in field_names:
        if field_name in source:
            destination[field_name] = source[field_name]


def main(argv: list[str] | None = None, *, prog_name: str | None = None) -> int:
    args = build_parser(prog_name=prog_name).parse_args(argv)
    try:
        payload = _build_success_payload(_run_state_update(args))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        payload = _build_failure_payload(str(exc), _workflow_failure_guideline(str(exc)))
    print(json.dumps(payload, ensure_ascii=True))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
