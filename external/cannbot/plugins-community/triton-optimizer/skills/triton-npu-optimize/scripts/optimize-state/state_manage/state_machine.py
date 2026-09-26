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

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import tempfile
from typing import cast

WORKFLOW_SCHEMA_VERSION = 1

PHASE_BASELINE = "baseline"
PHASE_AWAITING_ROUND_START = "awaiting_round_start"
PHASE_ROUND_ACTIVE = "round_active"
_PHASES = {PHASE_BASELINE, PHASE_AWAITING_ROUND_START, PHASE_ROUND_ACTIVE}

ROUND_DIR_PATTERN = re.compile(r"opt-round-(\d+)$")

ROUND_STRATEGIES = (
    "exploration",
    "structural_change",
    "focused_tuning",
    "stabilization",
    "plateau_review",
)
# Order encodes increasing required evidence depth, shallowest first.
ANALYSIS_POLICIES = (
    "pattern_entry",
    "profile_required",
    "ir_required",
    "compiler_source_required",
)
UPDATED_BY_VALUES = ("start-round", "set-current-round-state")
_ANALYSIS_POLICY_ORDER = {
    name: index for index, name in enumerate(ANALYSIS_POLICIES)
}
_WARNING_WORTHY_STRATEGY_TRANSITIONS = {
    ("structural_change", "exploration"): (
        "Returning from structural_change to exploration is unusual; "
        "confirm the previous rewrite direction is no longer justified."
    ),
    ("focused_tuning", "exploration"): (
        "Returning from focused_tuning to exploration is unusual; "
        "confirm the round really needs a broader search again."
    ),
    ("plateau_review", "focused_tuning"): (
        "Leaving plateau_review for focused_tuning is unusual; "
        "confirm the plateau conclusion has been resolved."
    ),
    ("plateau_review", "structural_change"): (
        "Leaving plateau_review for structural_change is unusual; "
        "confirm a new structural hypothesis is now evidence-backed."
    ),
}
_UNUSUAL_STRATEGY_POLICY_COMBINATIONS = {
    ("exploration", "compiler_source_required"): (
        "The combination exploration + compiler_source_required is unusual; "
        "compiler-source depth is often too deep for an exploratory round."
    ),
    ("structural_change", "pattern_entry"): (
        "The combination structural_change + pattern_entry is unusual; "
        "structural rewrites often need deeper evidence than pattern_entry alone."
    ),
    ("plateau_review", "pattern_entry"): (
        "The combination plateau_review + pattern_entry is unusual; plateau review usually follows deeper evidence."
    ),
}


@dataclass(frozen=True)
class _StrategyRequest:
    round_strategy: str
    analysis_policy: str
    reason: str


@dataclass(frozen=True)
class _StateUpdateMirror:
    round_dir: str
    request: _StrategyRequest
    reason: str
    previous: _StrategyRequest | None


@dataclass(frozen=True)
class _RoundActivation:
    payload: dict[str, object]
    rounds: dict[str, object]
    round_number: int
    round_dir: str
    request: _StrategyRequest
    warnings: list[str]


@dataclass(frozen=True)
class _RoundStateUpdate:
    round_key: str
    round_entry: dict[str, object]
    existing_state: dict[str, str] | None
    previous_request: _StrategyRequest
    next_request: _StrategyRequest
    reason: str
    warnings: list[str]


def load_state(state_path: Path) -> dict[str, object]:
    try:
        raw_payload = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed workflow state JSON: {exc}") from exc
    if not isinstance(raw_payload, dict):
        raise ValueError("workflow state must be a JSON object")
    payload = cast(dict[str, object], raw_payload)
    _validate_state(payload)
    return payload


def bootstrap_state(
    state_path: Path,
    *,
    run_id: str,
    baseline_reused: bool,
) -> None:
    payload: dict[str, object] = {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "run_id": run_id,
        "phase": PHASE_AWAITING_ROUND_START if baseline_reused else PHASE_BASELINE,
        "current_round": None,
        "baseline": {
            "status": "passed" if baseline_reused else "pending",
            "submitted_at": None,
        },
        "rounds": {},
    }
    _atomic_write_json(state_path, payload)


def mark_baseline_passed(state_path: Path) -> None:
    payload = load_state(state_path)
    baseline = _require_baseline_dict(payload)
    baseline["status"] = "passed"
    baseline["submitted_at"] = _utc_now()
    payload["phase"] = PHASE_AWAITING_ROUND_START
    payload["current_round"] = None
    _atomic_write_json(state_path, payload)


def start_round(
    state_path: Path,
    round_dir: str,
    *,
    round_strategy: str,
    analysis_policy: str,
    reason: str,
) -> dict[str, object]:
    request = _normalize_start_round_inputs(round_strategy, analysis_policy, reason)
    warnings = _build_strategy_state_warnings(
        round_strategy=request.round_strategy,
        analysis_policy=request.analysis_policy,
    )
    payload = load_state(state_path)
    round_number = _parse_round_number(round_dir)
    round_key = str(round_number)
    rounds = _require_rounds_dict(payload)
    _ensure_baseline_passed(payload)
    if _is_matching_active_round(
        payload,
        rounds,
        round_number,
        request,
    ):
        return _start_round_result(
            round_dir=round_dir,
            round_strategy=request.round_strategy,
            analysis_policy=request.analysis_policy,
            reason=request.reason,
            warnings=warnings,
        )
    _ensure_round_can_start(payload, rounds, round_key, round_dir)
    return _activate_and_mirror_round(
        state_path,
        _RoundActivation(payload, rounds, round_number, round_dir, request, warnings),
    )


def _activate_and_mirror_round(
    state_path: Path,
    update: _RoundActivation,
) -> dict[str, object]:
    _activate_round(
        update.payload,
        update.rounds,
        update.round_number,
        update.round_dir,
        update.request,
    )
    _atomic_write_json(state_path, update.payload)
    result_warnings = _mirror_start_round_update(
        state_path,
        update.round_dir,
        update.request,
        update.warnings,
    )
    return _start_round_result(
        round_dir=update.round_dir,
        round_strategy=update.request.round_strategy,
        analysis_policy=update.request.analysis_policy,
        reason=update.request.reason,
        warnings=result_warnings,
    )


def set_current_round_state(
    state_path: Path,
    *,
    round_strategy: str | None = None,
    analysis_policy: str | None = None,
    reason: str,
) -> dict[str, object]:
    normalized_reason = _validate_reason(reason)
    if round_strategy is None and analysis_policy is None:
        raise ValueError(
            "set-current-round-state requires --round-strategy and/or --analysis-policy"
        )
    payload = load_state(state_path)
    _current_round, round_key, round_entry = _get_active_round_entry(payload)
    existing_state = _get_optional_strategy_state(round_entry)
    previous_request, next_request = (
        _resolve_strategy_state_update(existing_state, round_strategy, analysis_policy)
    )
    _validate_strategy_state_update(
        existing_state,
        previous_request,
        next_request,
    )
    warnings = _build_transition_warnings(
        previous_round_strategy=previous_request.round_strategy,
        next_round_strategy=next_request.round_strategy,
        next_analysis_policy=next_request.analysis_policy,
    )
    round_entry["strategy_state"] = _build_strategy_state(
        round_strategy=next_request.round_strategy,
        analysis_policy=next_request.analysis_policy,
        reason=normalized_reason,
        updated_by="set-current-round-state",
    )
    _atomic_write_json(state_path, payload)
    return _build_round_state_update_result(
        state_path,
        _RoundStateUpdate(
            round_key,
            round_entry,
            existing_state,
            previous_request,
            next_request,
            normalized_reason,
            warnings,
        ),
    )


def _build_round_state_update_result(
    state_path: Path,
    update: _RoundStateUpdate,
) -> dict[str, object]:
    round_dir = _require_round_dir(update.round_entry, update.round_key)
    mirror = _StateUpdateMirror(
        round_dir=round_dir,
        request=update.next_request,
        reason=update.reason,
        previous=update.previous_request if update.existing_state is not None else None,
    )
    result_warnings = _mirror_round_state_update(state_path, mirror, update.warnings)
    return _set_current_round_state_result(
        round_dir=round_dir,
        previous_round_strategy=(
            update.previous_request.round_strategy
            if update.existing_state is not None
            else None
        ),
        next_round_strategy=update.next_request.round_strategy,
        previous_analysis_policy=(
            update.previous_request.analysis_policy
            if update.existing_state is not None
            else None
        ),
        next_analysis_policy=update.next_request.analysis_policy,
        reason=update.reason,
        warnings=result_warnings,
    )


def complete_round(
    state_path: Path,
    round_dir: str,
    *,
    current_round_arg: int | None = None,
) -> None:
    payload = load_state(state_path)
    round_number = _parse_round_number(round_dir)
    round_key = str(round_number)
    rounds = _require_rounds_dict(payload)

    if payload["phase"] != PHASE_ROUND_ACTIVE:
        raise ValueError(
            f"cannot complete {round_dir} while workflow phase is {payload['phase']}"
        )
    if payload.get("current_round") != round_number:
        raise ValueError(
            f"workflow state current_round={payload.get('current_round')} does not match {round_dir}"
        )
    if current_round_arg is not None and current_round_arg != round_number:
        raise ValueError(
            f"--current-round={current_round_arg} does not match workflow state round {round_number}"
        )

    round_entry_obj = rounds.get(round_key)
    if not isinstance(round_entry_obj, dict):
        raise ValueError(f"missing workflow state entry for {round_dir}")
    round_entry = cast(dict[str, object], round_entry_obj)
    if round_entry.get("status") != "active":
        raise ValueError(f"cannot complete non-active round {round_dir}")

    round_entry["status"] = "passed"
    round_entry["ended_at"] = _utc_now()
    payload["phase"] = PHASE_AWAITING_ROUND_START
    payload["current_round"] = None
    _atomic_write_json(state_path, payload)


def render_phase_summary(state_path: Path) -> str:
    payload = load_state(state_path)
    baseline = _require_baseline_dict(payload)
    reused = baseline.get("status") == "passed" and baseline.get("submitted_at") is None
    baseline_source = "pending"
    if baseline.get("status") == "passed":
        baseline_source = "reused" if reused else "freshly passed in this run"
    lines = [f"Current phase: {payload['phase']}"]
    current_round = payload.get("current_round")
    lines.append(
        f"Current round: {current_round}" if current_round is not None else "Current round: none"
    )
    if payload["phase"] == PHASE_ROUND_ACTIVE and isinstance(current_round, int):
        round_entry_obj = _require_rounds_dict(payload).get(str(current_round))
        if isinstance(round_entry_obj, dict):
            strategy_state = _get_optional_strategy_state(cast(dict[str, object], round_entry_obj))
            if strategy_state is None:
                lines.append("Current round strategy state: missing")
            else:
                lines.append(
                    f"Current round strategy: {strategy_state['round_strategy']}"
                )
                lines.append(
                    f"Required analysis depth: {strategy_state['analysis_policy']}"
                )
                lines.append(
                    f"Current round reason: {strategy_state['reason']}"
                )
    lines.append(f"Baseline source: {baseline_source}")
    return "\n".join(lines)


def write_round_timings_archive(state_path: Path, archive_path: Path) -> bool:
    payload = load_state(state_path)
    rows: list[dict[str, object]] = []
    for round_key, round_state in sorted(
        _require_rounds_dict(payload).items(),
        key=lambda item: int(item[0]),
    ):
        if not isinstance(round_state, dict):
            raise ValueError(f"workflow state round {round_key} must be an object")
        round_state_dict = cast(dict[str, object], round_state)
        if round_state_dict.get("status") != "passed":
            continue
        started_at = round_state_dict.get("started_at")
        ended_at = round_state_dict.get("ended_at")
        if not isinstance(started_at, str) or not started_at:
            raise ValueError(f"completed round {round_key} is missing started_at")
        if not isinstance(ended_at, str) or not ended_at:
            raise ValueError(f"completed round {round_key} is missing ended_at")
        rows.append(
            {
                "round": int(round_key),
                "started_at": started_at,
                "ended_at": ended_at,
            }
        )
    if not rows:
        return False
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_text(
        json.dumps(rows, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return True


def _validate_state(payload: dict[str, object]) -> None:
    phase = _validate_state_header(payload)
    baseline = _require_baseline_dict(payload)
    _validate_baseline_state(baseline)
    rounds = _require_rounds_dict(payload)
    _validate_active_round_reference(payload, rounds, phase)
    for round_key, round_state in rounds.items():
        _validate_round_entry(round_key, round_state)


def _normalize_start_round_inputs(
    round_strategy: str,
    analysis_policy: str,
    reason: str,
) -> _StrategyRequest:
    return _StrategyRequest(
        round_strategy=_validate_round_strategy(round_strategy),
        analysis_policy=_validate_analysis_policy(analysis_policy),
        reason=_validate_reason(reason),
    )


def _ensure_baseline_passed(payload: dict[str, object]) -> None:
    if _require_baseline_dict(payload).get("status") != "passed":
        raise ValueError("cannot start a round before baseline.status=passed")


def _is_matching_active_round(
    payload: dict[str, object],
    rounds: dict[str, object],
    round_number: int,
    request: _StrategyRequest,
) -> bool:
    if payload.get("phase") != PHASE_ROUND_ACTIVE:
        return False
    if payload.get("current_round") != round_number:
        return False
    round_entry = rounds.get(str(round_number))
    if not isinstance(round_entry, dict) or round_entry.get("status") != "active":
        return False
    strategy_state = _get_optional_strategy_state(cast(dict[str, object], round_entry))
    if strategy_state is None:
        raise ValueError(
            "active round is missing strategy_state; use set-current-round-state to initialize it"
        )
    matches = (
        strategy_state["round_strategy"] == request.round_strategy
        and strategy_state["analysis_policy"] == request.analysis_policy
        and strategy_state["reason"] == request.reason
    )
    if not matches:
        raise ValueError("cannot reinitialize active round with different strategy state")
    return True


def _ensure_round_can_start(
    payload: dict[str, object],
    rounds: dict[str, object],
    round_key: str,
    round_dir: str,
) -> None:
    if payload.get("phase") != PHASE_AWAITING_ROUND_START:
        raise ValueError(
            f"cannot start {round_dir} while workflow phase is {payload['phase']}"
        )
    existing = rounds.get(round_key)
    if isinstance(existing, dict) and existing.get("status") == "passed":
        raise ValueError(f"cannot reopen completed round {round_dir}")


def _activate_round(
    payload: dict[str, object],
    rounds: dict[str, object],
    round_number: int,
    round_dir: str,
    request: _StrategyRequest,
) -> None:
    payload["phase"] = PHASE_ROUND_ACTIVE
    payload["current_round"] = round_number
    rounds[str(round_number)] = {
        "status": "active",
        "round_dir": round_dir,
        "started_at": _utc_now(),
        "ended_at": None,
        "strategy_state": _build_strategy_state(
            round_strategy=request.round_strategy,
            analysis_policy=request.analysis_policy,
            reason=request.reason,
            updated_by="start-round",
        ),
    }


def _mirror_start_round_update(
    state_path: Path,
    round_dir: str,
    request: _StrategyRequest,
    warnings: list[str],
) -> list[str]:
    result_warnings = list(warnings)
    try:
        _append_state_update_block(
            _attempts_path_for_round(state_path, round_dir),
            source="start-round",
            round_strategy=request.round_strategy,
            analysis_policy=request.analysis_policy,
            reason=request.reason,
            warnings=result_warnings,
        )
    except OSError as exc:
        result_warnings.append(_history_mirror_warning(exc))
    return result_warnings


def _get_active_round_entry(
    payload: dict[str, object],
) -> tuple[int, str, dict[str, object]]:
    current_round = payload.get("current_round")
    if payload.get("phase") != PHASE_ROUND_ACTIVE or not isinstance(current_round, int):
        raise ValueError("no optimize round is currently active")
    round_key = str(current_round)
    round_entry = _require_rounds_dict(payload).get(round_key)
    if not isinstance(round_entry, dict):
        raise ValueError(f"missing workflow state entry for opt-round-{current_round}")
    if round_entry.get("status") != "active":
        raise ValueError(f"cannot update non-active round opt-round-{current_round}")
    return current_round, round_key, cast(dict[str, object], round_entry)


def _resolve_strategy_state_update(
    existing_state: dict[str, str] | None,
    round_strategy: str | None,
    analysis_policy: str | None,
) -> tuple[_StrategyRequest, _StrategyRequest]:
    if existing_state is None:
        if round_strategy is None or analysis_policy is None:
            raise ValueError(
                "legacy active round is missing strategy_state; provide both --round-strategy and --analysis-policy"
            )
        previous = _StrategyRequest(
            round_strategy=_validate_round_strategy(round_strategy),
            analysis_policy=_validate_analysis_policy(analysis_policy),
            reason="",
        )
    else:
        previous = _StrategyRequest(
            round_strategy=existing_state["round_strategy"],
            analysis_policy=existing_state["analysis_policy"],
            reason=existing_state["reason"],
        )
    next_strategy = (
        _validate_round_strategy(round_strategy)
        if round_strategy is not None
        else previous.round_strategy
    )
    next_policy = (
        _validate_analysis_policy(analysis_policy)
        if analysis_policy is not None
        else previous.analysis_policy
    )
    return previous, _StrategyRequest(next_strategy, next_policy, "")


def _validate_strategy_state_update(
    existing_state: dict[str, str] | None,
    previous: _StrategyRequest,
    next_request: _StrategyRequest,
) -> None:
    if existing_state is None:
        return
    if (
        previous.round_strategy == next_request.round_strategy
        and previous.analysis_policy == next_request.analysis_policy
    ):
        raise ValueError("state update would be a no-op")
    if _ANALYSIS_POLICY_ORDER[next_request.analysis_policy] < _ANALYSIS_POLICY_ORDER[previous.analysis_policy]:
        raise ValueError("analysis_policy cannot become shallower within the same round")


def _require_round_dir(round_entry: dict[str, object], round_key: str) -> str:
    round_dir = round_entry.get("round_dir")
    if not isinstance(round_dir, str) or not round_dir:
        raise ValueError(f"workflow state round {round_key} is missing round_dir")
    return round_dir


def _mirror_round_state_update(
    state_path: Path,
    mirror: _StateUpdateMirror,
    warnings: list[str],
) -> list[str]:
    result_warnings = list(warnings)
    try:
        _append_state_update_block(
            _attempts_path_for_round(state_path, mirror.round_dir),
            source="set-current-round-state",
            round_strategy=mirror.request.round_strategy,
            analysis_policy=mirror.request.analysis_policy,
            reason=mirror.reason,
            previous_round_strategy=_previous_round_strategy(mirror.previous),
            previous_analysis_policy=_previous_analysis_policy(mirror.previous),
            warnings=result_warnings,
        )
    except OSError as exc:
        result_warnings.append(_history_mirror_warning(exc))
    return result_warnings


def _previous_round_strategy(previous: _StrategyRequest | None) -> str:
    return previous.round_strategy if previous is not None else "<unset>"


def _previous_analysis_policy(previous: _StrategyRequest | None) -> str:
    return previous.analysis_policy if previous is not None else "<unset>"


def _history_mirror_warning(error: OSError) -> str:
    return (
        "attempts.md history mirror could not be updated: "
        f"{error}. Workflow state remains authoritative."
    )


def _validate_state_header(payload: dict[str, object]) -> str:
    schema_version = payload.get("schema_version")
    if schema_version != WORKFLOW_SCHEMA_VERSION:
        raise ValueError(f"unsupported workflow state schema_version: {schema_version!r}")
    phase = payload.get("phase")
    if not isinstance(phase, str) or phase not in _PHASES:
        raise ValueError(f"unknown workflow state phase: {phase!r}")
    return phase


def _validate_baseline_state(baseline: dict[str, object]) -> None:
    baseline_status = baseline.get("status")
    if baseline_status not in {"pending", "passed"}:
        raise ValueError(f"unknown baseline status: {baseline_status!r}")
    submitted_at = baseline.get("submitted_at")
    if submitted_at is not None and not isinstance(submitted_at, str):
        raise ValueError("baseline.submitted_at must be a string or null")


def _validate_active_round_reference(
    payload: dict[str, object],
    rounds: dict[str, object],
    phase: str,
) -> None:
    current_round = payload.get("current_round")
    if phase != PHASE_ROUND_ACTIVE:
        if current_round is not None:
            raise ValueError(f"phase={phase} requires current_round=null")
        return
    if not isinstance(current_round, int):
        raise ValueError("phase=round_active requires a non-null integer current_round")
    current_entry = rounds.get(str(current_round))
    if not isinstance(current_entry, dict):
        raise ValueError(f"phase=round_active requires rounds[{current_round}]")
    if current_entry.get("status") != "active":
        raise ValueError(f"phase=round_active requires active state for round {current_round}")
    if not isinstance(current_entry.get("started_at"), str) or not current_entry["started_at"]:
        raise ValueError(f"active round {current_round} must have started_at")


def _validate_round_entry(round_key: str, round_state: object) -> None:
    if not isinstance(round_state, dict):
        raise ValueError(f"workflow state round {round_key} must be an object")
    entry = cast(dict[str, object], round_state)
    status = entry.get("status")
    if status not in {"active", "passed"}:
        raise ValueError(f"unknown round status for {round_key}: {status!r}")
    _validate_round_entry_identity(round_key, entry)
    _validate_round_entry_timing(round_key, entry, cast(str, status))
    strategy_state = entry.get("strategy_state")
    if strategy_state is not None:
        _validate_strategy_state(strategy_state, round_key=round_key)


def _validate_round_entry_identity(round_key: str, entry: dict[str, object]) -> None:
    round_dir = _require_round_dir(entry, round_key)
    if _parse_round_number(round_dir) != int(round_key):
        raise ValueError(
            f"workflow state round key {round_key} does not match round_dir {round_dir}"
        )


def _validate_round_entry_timing(
    round_key: str,
    entry: dict[str, object],
    status: str,
) -> None:
    started_at = entry.get("started_at")
    if not isinstance(started_at, str) or not started_at:
        raise ValueError(f"workflow state round {round_key} is missing started_at")
    ended_at = entry.get("ended_at")
    if ended_at is not None and not isinstance(ended_at, str):
        raise ValueError(f"workflow state round {round_key} ended_at must be a string or null")
    if status == "passed" and ended_at is None:
        raise ValueError(f"completed round {round_key} is missing ended_at")


def _validate_strategy_state(strategy_state: object, *, round_key: str) -> None:
    if not isinstance(strategy_state, dict):
        raise ValueError(f"workflow state round {round_key} strategy_state must be an object")
    strategy_state_dict = cast(dict[str, object], strategy_state)
    round_strategy = strategy_state_dict.get("round_strategy")
    if round_strategy not in ROUND_STRATEGIES:
        raise ValueError(
            f"workflow state round {round_key} has unknown round_strategy: {round_strategy!r}"
        )
    analysis_policy = strategy_state_dict.get("analysis_policy")
    if analysis_policy not in ANALYSIS_POLICIES:
        raise ValueError(
            f"workflow state round {round_key} has unknown analysis_policy: {analysis_policy!r}"
        )
    reason = strategy_state_dict.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"workflow state round {round_key} strategy_state.reason must be a non-empty string")
    updated_at = strategy_state_dict.get("updated_at")
    if not isinstance(updated_at, str) or not updated_at:
        raise ValueError(f"workflow state round {round_key} strategy_state.updated_at must be a non-empty string")
    updated_by = strategy_state_dict.get("updated_by")
    if updated_by not in UPDATED_BY_VALUES:
        raise ValueError(
            f"workflow state round {round_key} strategy_state.updated_by must be one of {UPDATED_BY_VALUES}"
        )


def _require_baseline_dict(payload: dict[str, object]) -> dict[str, object]:
    baseline = payload.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("workflow state baseline must be an object")
    return cast(dict[str, object], baseline)


def _require_rounds_dict(payload: dict[str, object]) -> dict[str, object]:
    rounds = payload.get("rounds")
    if not isinstance(rounds, dict):
        raise ValueError("workflow state rounds must be an object")
    return cast(dict[str, object], rounds)


def _get_optional_strategy_state(round_entry: dict[str, object]) -> dict[str, str] | None:
    strategy_state = round_entry.get("strategy_state")
    if strategy_state is None:
        return None
    if not isinstance(strategy_state, dict):
        raise ValueError("workflow round strategy_state must be an object")
    strategy_state_dict = cast(dict[str, object], strategy_state)
    round_strategy = strategy_state_dict.get("round_strategy")
    analysis_policy = strategy_state_dict.get("analysis_policy")
    reason = strategy_state_dict.get("reason")
    if not isinstance(round_strategy, str) or round_strategy not in ROUND_STRATEGIES:
        raise ValueError(f"workflow round strategy_state.round_strategy is invalid: {round_strategy!r}")
    if not isinstance(analysis_policy, str) or analysis_policy not in ANALYSIS_POLICIES:
        raise ValueError(f"workflow round strategy_state.analysis_policy is invalid: {analysis_policy!r}")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("workflow round strategy_state.reason must be a non-empty string")
    return {
        "round_strategy": round_strategy,
        "analysis_policy": analysis_policy,
        "reason": reason.strip(),
    }


def _validate_round_strategy(round_strategy: str) -> str:
    normalized = round_strategy.strip()
    if normalized not in ROUND_STRATEGIES:
        raise ValueError(f"unknown round_strategy: {round_strategy!r}")
    return normalized


def _validate_analysis_policy(analysis_policy: str) -> str:
    normalized = analysis_policy.strip()
    if normalized not in ANALYSIS_POLICIES:
        raise ValueError(f"unknown analysis_policy: {analysis_policy!r}")
    return normalized


def _validate_reason(reason: str) -> str:
    normalized = reason.strip()
    if not normalized:
        raise ValueError("reason is required")
    return normalized


def _build_strategy_state(
    *,
    round_strategy: str,
    analysis_policy: str,
    reason: str,
    updated_by: str,
) -> dict[str, object]:
    return {
        "round_strategy": round_strategy,
        "analysis_policy": analysis_policy,
        "reason": reason,
        "updated_at": _utc_now(),
        "updated_by": updated_by,
    }


def _build_strategy_state_warnings(
    *,
    round_strategy: str,
    analysis_policy: str,
) -> list[str]:
    warning = _UNUSUAL_STRATEGY_POLICY_COMBINATIONS.get(
        (round_strategy, analysis_policy)
    )
    return [warning] if warning is not None else []


def _build_transition_warnings(
    *,
    previous_round_strategy: str,
    next_round_strategy: str,
    next_analysis_policy: str,
) -> list[str]:
    warnings: list[str] = []
    transition_warning = _WARNING_WORTHY_STRATEGY_TRANSITIONS.get(
        (previous_round_strategy, next_round_strategy)
    )
    if transition_warning is not None:
        warnings.append(transition_warning)
    warnings.extend(
        _build_strategy_state_warnings(
            round_strategy=next_round_strategy,
            analysis_policy=next_analysis_policy,
        )
    )
    return warnings


def _append_state_update_block(
    attempts_path: Path,
    *,
    source: str,
    round_strategy: str,
    analysis_policy: str,
    reason: str,
    warnings: list[str],
    previous_round_strategy: str | None = None,
    previous_analysis_policy: str | None = None,
) -> None:
    lines = [f"## State Update {_utc_now()}", f"- Source: {source}"]
    if previous_round_strategy is None and previous_analysis_policy is None:
        lines.extend(
            [
                f"- Round strategy: {round_strategy}",
                f"- Analysis policy: {analysis_policy}",
            ]
        )
    else:
        lines.extend(
            [
                f"- Round strategy: {previous_round_strategy} -> {round_strategy}",
                f"- Analysis policy: {previous_analysis_policy} -> {analysis_policy}",
            ]
        )
    lines.append(f"- Reason: {reason}")
    for warning in warnings:
        lines.append(f"- Warning: {warning}")
    block = "\n".join(lines) + "\n"
    attempts_path.parent.mkdir(parents=True, exist_ok=True)
    existing = attempts_path.read_text(encoding="utf-8") if attempts_path.exists() else ""
    separator = ""
    if existing:
        separator = "\n" if existing.endswith("\n") else "\n\n"
    attempts_path.write_text(existing + separator + block, encoding="utf-8")


def _attempts_path_for_round(state_path: Path, round_dir: str) -> Path:
    return state_path.parent.parent / round_dir / "attempts.md"


def _start_round_result(
    *,
    round_dir: str,
    round_strategy: str,
    analysis_policy: str,
    reason: str,
    warnings: list[str],
) -> dict[str, object]:
    result: dict[str, object] = {
        "round": round_dir,
        "round_strategy": round_strategy,
        "analysis_policy": analysis_policy,
        "reason": reason,
    }
    if warnings:
        result["warnings"] = warnings
    return result


def _set_current_round_state_result(
    *,
    round_dir: str,
    previous_round_strategy: str | None,
    next_round_strategy: str,
    previous_analysis_policy: str | None,
    next_analysis_policy: str,
    reason: str,
    warnings: list[str],
) -> dict[str, object]:
    result: dict[str, object] = {
        "round": round_dir,
        "round_strategy": next_round_strategy,
        "analysis_policy": next_analysis_policy,
        "reason": reason,
    }
    if previous_round_strategy is not None:
        result["previous_round_strategy"] = previous_round_strategy
    if previous_analysis_policy is not None:
        result["previous_analysis_policy"] = previous_analysis_policy
    if warnings:
        result["warnings"] = warnings
    return result


def _parse_round_number(round_dir: str) -> int:
    match = ROUND_DIR_PATTERN.fullmatch(round_dir)
    if match is None:
        raise ValueError(f"invalid round directory name: {round_dir!r}")
    round_number = int(match.group(1), 10)
    if round_number < 1:
        raise ValueError(f"round number must be >= 1: {round_dir!r}")
    return round_number


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    try:
        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
