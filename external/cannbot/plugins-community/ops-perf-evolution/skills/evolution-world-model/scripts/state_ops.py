#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# ----------------------------------------------------------------------------------------------------------
"""state_ops.py — Per-evolution runtime state machine.

Manages `<evo_dir>/state.json`, a runtime cursor decoupled from
`world_model.json` (which holds decision-tree evidence). The state file
records *where the agent is right now* (stage, current round, partial
progress, drift status) so that:

  1. Stop / PreToolUse hooks can validate stage/artifact consistency.
  2. An agent re-entering after a crash can resume from `stage` precisely.

This is intentionally separate from `session_anchor.py` (per-op identity
lock at `output/.ops-evo_current_session_{op_name}.json`) — anchors are
identity, this file is execution cursor.

CLI subcommands:
  init           --evo-dir <dir> --agent <lingxi-evo|ops-evo>
                 --session-id <id> --max-rounds <int>
                 Create a fresh state.json. Refuses to overwrite unless --force.
  read           --evo-dir <dir>
                 Print state.json to stdout (pretty JSON).
  write-stage    --evo-dir <dir> --stage <name> [--round <int>]
                 Update stage field; optionally set current_round atomically.
  write-partial  --evo-dir <dir> --parallel-idx <int> --status <name>
                 Update partial_status[idx] = status.
  reset-partial  --evo-dir <dir>
                 Clear partial_status (called when entering round_refine).
  set-verdict    --evo-dir <dir> --verdict <advanced|stalled|regressed|unknown>
                 Update last_mainline_verdict and bump mainline_stall_count.
  set-drift      --evo-dir <dir> --status <normal|replan_required>
                 Update drift_status.
  mark-must-run  --evo-dir <dir> --step <name>
                 Append <name> to must_run_before_next_round.
  clear-must-run --evo-dir <dir> --step <name>
                 Remove <name> from must_run_before_next_round.
  validate       --evo-dir <dir> [--check-stage-artifacts]
                 Sanity-check schema + (optionally) cross-check stage vs disk.

Stage state machine (allowed values, single source of truth):
  init → shared_prep → [baseline_build] → wm_init
  → [round loop, repeats N times:]
      round_gate → round_select → round_generate
      → round_refine → round_react → round_checkpoint
  → finalize → report → done
  Terminal failures: aborted
  Special: drift_replan (inserted between rounds when drift_status=replan_required)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any

LOGGER = logging.getLogger(__name__)

# 数据输出专用 logger：CLI 结果（JSON/表格/报告）走 stdout（agent 调用协议通道），
# 与 LOGGER（stderr 进度/警告）分离，避免 lint G.LOG.02 误报 print。
DATA_LOGGER = logging.getLogger(f"{__name__}.data")
_DATA_HANDLER = logging.StreamHandler(sys.stdout)
_DATA_HANDLER.setFormatter(logging.Formatter("%(message)s"))
DATA_LOGGER.addHandler(_DATA_HANDLER)
DATA_LOGGER.propagate = False
DATA_LOGGER.setLevel(logging.INFO)


SCHEMA_VERSION = "1.0"

STAGES_INIT = ("init", "shared_prep", "baseline_build", "wm_init")
STAGES_ROUND = (
    "round_gate",
    "round_select",
    "round_generate",
    "round_refine",
    "round_react",
    "round_checkpoint",
)
STAGES_FINAL = ("finalize", "report", "done", "aborted", "drift_replan")
ALL_STAGES = STAGES_INIT + STAGES_ROUND + STAGES_FINAL

VALID_AGENTS = ("lingxi-evo", "ops-evo")
VALID_VERDICTS = ("advanced", "stalled", "regressed", "unknown")
VALID_DRIFT = ("normal", "replan_required")
VALID_PARTIAL_STATUS = ("pending", "running", "completed", "failed")


def _now_iso() -> str:
    tz = timezone(timedelta(hours=8))
    return datetime.now(tz).strftime("%Y-%m-%dT%H:%M:%S%z")


def _state_path(evo_dir: str) -> str:
    return os.path.join(evo_dir, "state.json")


class StateOpsError(Exception):
    """state_ops 的致命错误，由 main 统一转换为进程退出码。"""

    def __init__(self, msg: str, code: int = 1):
        super().__init__(msg)
        self.code = code


def _abort(msg: str, code: int = 1) -> None:
    raise StateOpsError(msg, code)


def _read_state(evo_dir: str) -> dict:
    path = _state_path(evo_dir)
    if not os.path.isfile(path):
        _abort(f"state.json not found at {path}; run `state_ops.py init` first")
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            _abort(f"state.json is not valid JSON ({e}); manual repair required")
            return {}  # unreachable


def _write_state(evo_dir: str, state: dict) -> None:
    state["last_updated_at"] = _now_iso()
    path = _state_path(evo_dir)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_init(args: argparse.Namespace) -> None:
    evo_dir = os.path.abspath(args.evo_dir)
    if not os.path.isdir(evo_dir):
        _abort(f"evo-dir does not exist or is not a directory: {evo_dir}")

    if args.agent not in VALID_AGENTS:
        _abort(f"agent must be one of {VALID_AGENTS}, got {args.agent}")

    path = _state_path(evo_dir)
    if os.path.exists(path) and not args.force:
        _abort(f"state.json already exists at {path}; pass --force to overwrite")

    state = {
        "schema_version": SCHEMA_VERSION,
        "session_id": args.session_id,
        "evo_dir": evo_dir,
        "agent": args.agent,
        "stage": "init",
        "current_round": 0,
        "max_rounds": args.max_rounds,
        "expected_parallel_num": args.parallel_num,
        "round_started_at": None,
        "round_finished_at": None,
        "partial_status": {},
        "mainline_stall_count": 0,
        "last_mainline_verdict": "unknown",
        "drift_status": "normal",
        "must_run_before_next_round": [],
        "read_keys": [],          # v3.2 Phase D2: do-not-reread 集合（source_key 列表）
        "last_updated_at": None,
    }
    _write_state(evo_dir, state)
    LOGGER.info("state_ops init: wrote %s", path)
    LOGGER.info("  agent=%s session_id=%s max_rounds=%s",
                args.agent, args.session_id, args.max_rounds)


def cmd_read(args: argparse.Namespace) -> None:
    state = _read_state(os.path.abspath(args.evo_dir))
    DATA_LOGGER.info("%s", json.dumps(state, ensure_ascii=False, indent=2))


def cmd_write_stage(args: argparse.Namespace) -> None:
    if args.stage not in ALL_STAGES:
        _abort(f"stage must be one of {ALL_STAGES}, got {args.stage}")

    evo_dir = os.path.abspath(args.evo_dir)
    state = _read_state(evo_dir)
    prev_stage = state.get("stage")
    state["stage"] = args.stage

    if args.round is not None:
        state["current_round"] = args.round

    # Round boundary timestamps
    if args.stage == "round_gate" and prev_stage != "round_gate":
        state["round_started_at"] = _now_iso()
        state["round_finished_at"] = None
    if args.stage == "round_checkpoint":
        state["round_finished_at"] = _now_iso()

    # When entering round_refine, partial_status must be all-completed; warn otherwise
    if args.stage == "round_refine":
        ps = state.get("partial_status", {})
        unfinished = [k for k, v in ps.items() if v not in ("completed", "failed")]
        if unfinished:
            LOGGER.warning(
                "[WARNING] state_ops write-stage: entering round_refine but partials "
                "%s not in (completed, failed)",
                unfinished,
            )

    _write_state(evo_dir, state)
    LOGGER.info("state_ops: stage %s → %s (round=%s)",
                prev_stage, args.stage, state['current_round'])


def cmd_write_partial(args: argparse.Namespace) -> None:
    if args.status not in VALID_PARTIAL_STATUS:
        _abort(f"status must be one of {VALID_PARTIAL_STATUS}, got {args.status}")

    evo_dir = os.path.abspath(args.evo_dir)
    state = _read_state(evo_dir)
    state.setdefault("partial_status", {})[str(args.parallel_idx)] = args.status
    _write_state(evo_dir, state)
    LOGGER.info("state_ops: partial[%s] = %s", args.parallel_idx, args.status)


def cmd_reset_partial(args: argparse.Namespace) -> None:
    evo_dir = os.path.abspath(args.evo_dir)
    state = _read_state(evo_dir)
    state["partial_status"] = {}
    _write_state(evo_dir, state)
    LOGGER.info("state_ops: partial_status cleared")


def cmd_set_verdict(args: argparse.Namespace) -> None:
    if args.verdict not in VALID_VERDICTS:
        _abort(f"verdict must be one of {VALID_VERDICTS}, got {args.verdict}")

    evo_dir = os.path.abspath(args.evo_dir)
    state = _read_state(evo_dir)
    state["last_mainline_verdict"] = args.verdict
    if args.verdict in ("stalled", "regressed"):
        state["mainline_stall_count"] = int(state.get("mainline_stall_count", 0)) + 1
    elif args.verdict == "advanced":
        state["mainline_stall_count"] = 0
    _write_state(evo_dir, state)
    LOGGER.info("state_ops: verdict=%s stall_count=%s",
                args.verdict, state['mainline_stall_count'])


def cmd_set_drift(args: argparse.Namespace) -> None:
    if args.status not in VALID_DRIFT:
        _abort(f"drift status must be one of {VALID_DRIFT}, got {args.status}")

    evo_dir = os.path.abspath(args.evo_dir)
    state = _read_state(evo_dir)
    state["drift_status"] = args.status
    _write_state(evo_dir, state)
    LOGGER.info("state_ops: drift_status = %s", args.status)


def cmd_mark_must_run(args: argparse.Namespace) -> None:
    evo_dir = os.path.abspath(args.evo_dir)
    state = _read_state(evo_dir)
    pending = state.setdefault("must_run_before_next_round", [])
    if args.step not in pending:
        pending.append(args.step)
    _write_state(evo_dir, state)
    LOGGER.info("state_ops: must_run += %s (now: %s)", args.step, pending)


def cmd_clear_must_run(args: argparse.Namespace) -> None:
    evo_dir = os.path.abspath(args.evo_dir)
    state = _read_state(evo_dir)
    pending = [s for s in state.get("must_run_before_next_round", []) if s != args.step]
    state["must_run_before_next_round"] = pending
    _write_state(evo_dir, state)
    LOGGER.info("state_ops: must_run -= %s (now: %s)", args.step, pending)


# ═══════════════════════════════════════════════════════════════════
# v3.2 Phase D2: read_keys do-not-reread 接口
# ═══════════════════════════════════════════════════════════════════


def cmd_add_read_keys(args: argparse.Namespace) -> None:
    """追加 source_keys 到 state.read_keys（去重）。"""
    evo_dir = os.path.abspath(args.evo_dir)
    state = _read_state(evo_dir)

    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    if not keys:
        _abort("--keys must be non-empty (comma-separated)")

    existing = state.get("read_keys") or []
    # 去重保序
    merged = list(dict.fromkeys(existing + keys))
    added = [k for k in keys if k not in existing]

    state["read_keys"] = merged
    _write_state(evo_dir, state)
    LOGGER.info("state_ops add-read-keys: added %d new key(s); "
                "total now %d", len(added), len(merged))
    for k in added:
        LOGGER.info("  + %s", k)


def cmd_get_read_keys(args: argparse.Namespace) -> None:
    """输出 state.read_keys 当前内容（用于 partial-prompt 注入或调试）。"""
    evo_dir = os.path.abspath(args.evo_dir)
    state = _read_state(evo_dir)
    keys = state.get("read_keys") or []

    if args.format == "json":
        DATA_LOGGER.info("%s", json.dumps({"read_keys": keys, "count": len(keys)},
                                          indent=2, ensure_ascii=False))
    elif args.format == "lines":
        for k in keys:
            DATA_LOGGER.info("%s", k)
    else:  # markdown
        if not keys:
            DATA_LOGGER.info("(本 session 暂无已读 source_keys)")
        else:
            DATA_LOGGER.info("## Excluded — 本 session 已读 %d 项，不重读", len(keys))
            for k in keys:
                DATA_LOGGER.info("- %s", k)


def cmd_clear_read_keys(args: argparse.Namespace) -> None:
    """清空 read_keys（drift 漂移强制扩搜索时用）。"""
    evo_dir = os.path.abspath(args.evo_dir)
    state = _read_state(evo_dir)
    n_before = len(state.get("read_keys") or [])
    state["read_keys"] = []
    _write_state(evo_dir, state)
    LOGGER.info("state_ops clear-read-keys: cleared %d keys", n_before)


def _check_partial_artifact(idx: str, status: str, stage: str,
                             rd: int, evo_dir: str) -> str | None:
    """检查单个 partial 的状态与产物一致性，返回问题描述或 None。"""
    if status not in ("completed", "failed"):
        return (f"stage={stage} but partial[{idx}] status={status} "
                f"(expected completed/failed)")
    eval_path = os.path.join(
        evo_dir, f"round_{rd}", f"parallel_{idx}", "evaluation_results.json"
    )
    if status == "completed" and not os.path.isfile(eval_path):
        return f"missing artifact: {eval_path}"
    return None


def _check_stage_artifacts(state: dict, evo_dir: str) -> list[str]:
    """Cross-check stage against on-disk artifacts. Returns list of issues (empty = ok)."""
    issues: list[str] = []
    stage = state.get("stage")
    rd = int(state.get("current_round", 0))

    # round_generate_done equivalent: stage == round_refine implies all partials
    # have produced evaluation_results.json
    if stage in ("round_refine", "round_react", "round_checkpoint"):
        partials = state.get("partial_status", {})
        for idx, status in partials.items():
            issue = _check_partial_artifact(idx, status, stage, rd, evo_dir)
            if issue:
                issues.append(issue)

    # When in round_refine, world_model.json must exist
    if stage in STAGES_ROUND and stage != "round_gate":
        wm_path = os.path.join(evo_dir, "world_model.json")
        if not os.path.isfile(wm_path):
            issues.append(f"missing world_model.json at {wm_path}")

    # current_round must not exceed max_rounds + 1 (we allow finalize at max+0)
    max_r = int(state.get("max_rounds", 0))
    if rd > max_r + 1:
        issues.append(f"current_round={rd} exceeds max_rounds={max_r}")

    return issues


def cmd_validate(args: argparse.Namespace) -> None:
    evo_dir = os.path.abspath(args.evo_dir)
    state = _read_state(evo_dir)

    issues: list[str] = []
    # Schema sanity
    if state.get("schema_version") != SCHEMA_VERSION:
        issues.append(f"schema_version mismatch: {state.get('schema_version')} != {SCHEMA_VERSION}")
    if state.get("stage") not in ALL_STAGES:
        issues.append(f"invalid stage: {state.get('stage')}")
    if state.get("agent") not in VALID_AGENTS:
        issues.append(f"invalid agent: {state.get('agent')}")
    if state.get("drift_status") not in VALID_DRIFT:
        issues.append(f"invalid drift_status: {state.get('drift_status')}")
    if state.get("last_mainline_verdict") not in VALID_VERDICTS:
        issues.append(f"invalid last_mainline_verdict: {state.get('last_mainline_verdict')}")

    # evo_dir consistency
    if os.path.abspath(state.get("evo_dir", "")) != evo_dir:
        issues.append(
            f"evo_dir field mismatch: {state.get('evo_dir')} != {evo_dir}"
        )

    if args.check_stage_artifacts:
        issues.extend(_check_stage_artifacts(state, evo_dir))

    for i in issues:
        LOGGER.error("  - %s", i)
    if issues:
        return 2
    LOGGER.info("state_ops validate: OK")
    return 0


# ---------------------------------------------------------------------------
# Inference: rebuild state from filesystem evidence
# ---------------------------------------------------------------------------
# state.json's LLM-untrusted fields (stage, current_round, partial_status) are
# re-derived from filesystem evidence every time Stop hook fires or wm_ops
# select/refine runs. The agent prompt is no longer responsible for keeping
# these fields correct.
#
# Trust sources (in order):
#   1. world_model.json: session.actual_rounds_completed, requested_rounds
#      (these are wm_ops's own writes, fully trusted)
#   2. Filesystem: round_N/ directories and parallel_K/evaluation_results.json
#      existence
#
# LLM-trusted fields NOT touched by infer:
#   - drift_status (set by wm_ops refine + cmd_select)
#   - mainline_stall_count (currently unused by hook, reserved for verdict path)
#   - last_mainline_verdict (currently unused, reserved for verdict path)
#   - must_run_before_next_round (set by hook/wm_ops, not LLM)
#   - max_rounds, agent, session_id, evo_dir, schema_version (set at init)


def _read_subagent_exit_marker(parallel_dir: str) -> dict | None:
    """Read .subagent_exit_status marker from a parallel directory.

    Returns the parsed dict, or None if marker is missing or corrupt.
    Written by loop-subagent-stop.sh after a subagent terminates.
    """
    marker_path = os.path.join(parallel_dir, ".subagent_exit_status")
    if not os.path.isfile(marker_path):
        return None
    try:
        with open(marker_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _list_round_numbers(evo_dir: str, entries: list) -> list[int]:
    """从目录项中枚举 round_N 目录，返回升序的轮次号列表。"""
    numbers = []
    for d in entries:
        if not d.startswith("round_"):
            continue
        suffix = d[len("round_"):]
        if suffix.isdigit() and os.path.isdir(os.path.join(evo_dir, d)):
            numbers.append(int(suffix))
    return sorted(numbers)


def _infer_one_partial(round_dir: str, pd: str) -> str:
    """推断单个 parallel 目录的 partial 状态。"""
    parallel_dir = os.path.join(round_dir, pd)
    eval_path = os.path.join(parallel_dir, "evaluation_results.json")
    if os.path.isfile(eval_path) and os.path.getsize(eval_path) > 0:
        return "completed"
    # Check subagent exit marker before defaulting to "running"
    marker = _read_subagent_exit_marker(parallel_dir)
    if marker is not None and marker.get("exit_status", "") in ("timeout", "error", "killed"):
        return "failed"
    # - exit_status == "completed" but no results: possible write race
    # - marker 缺失：crash-safe default
    # 两种情况都保守标记 "running"（blocks premature stop）
    return "running"


def _infer_partial_status(round_dir: str, parallel_dirs: list) -> tuple[dict, int]:
    """遍历 parallel 目录推断 partial_status，返回 (status_dict, completed_count)。"""
    partial_status: dict[str, str] = {}
    completed = 0
    for pd in parallel_dirs:
        idx = pd[len("parallel_"):]
        status = _infer_one_partial(round_dir, pd)
        partial_status[idx] = status
        if status == "completed":
            completed += 1
    return partial_status, completed


def _has_evolution_report(entries: list) -> bool:
    """检查目录项中是否存在 evolution-report_*.html（最终报告产物）。"""
    return any(
        name.startswith("evolution-report_") and name.endswith(".html")
        for name in entries
    )


@dataclass
class _StageDecisionInput:
    """_decide_stage 的决策输入封装。"""
    entries: list
    actual_rounds: int
    requested_rounds: int
    max_round: int
    completed: int
    total: int


def _decide_stage(di: _StageDecisionInput) -> str:
    """按决策表推导 stage（见 _infer_state_from_filesystem docstring）。"""
    if di.total == 0:
        return "round_select"
    if di.completed < di.total:
        return "round_generate"
    if di.actual_rounds < di.max_round:
        # All partials done but refine hasn't run yet
        return "round_generate"
    if di.actual_rounds == di.max_round:
        if di.requested_rounds > 0 and di.max_round >= di.requested_rounds:
            # All requested rounds finished. Distinguish finalize vs done by
            # checking for evolution-report*.html artifact — produced only by
            # the `evolution-report` skill after the final report step. This
            # gives us an objective filesystem-derived signal that the entire
            # pipeline (including step 6 report) has completed, rather than
            # relying on the agent to explicitly mark stage=done.
            return "done" if _has_evolution_report(di.entries) else "finalize"
        return "round_checkpoint"
    # actual_rounds > max_round — inconsistent; conservatively pick
    # round_checkpoint (refine likely ran for higher round but dirs missing)
    return "round_checkpoint"


def _infer_state_from_filesystem(evo_dir: str) -> dict:
    """Re-derive runtime state from filesystem evidence + world_model.json.

    Returns a partial state dict containing only the LLM-untrusted fields:
      {"stage": <name>, "current_round": <int>, "partial_status": {...}}

    Does NOT read state.json's existing stage/partial_status fields — by
    design. Callers merge this with existing state.json, preserving
    drift_status / stall_count / must_run_before_next_round.

    Stage decision table (in priority order):
      1. No world_model.json    → "shared_prep"  (agent hasn't run wm_init yet)
      2. No round_N directory   → "wm_init"
      3. round_N exists, no parallel_K dirs   → "round_select"
      4. some parallel_K's evaluation_results.json missing → "round_generate"
      5. all parallels done but actual_rounds_completed < max_round → "round_generate"
         (partials done but refine not run; next action = refine)
      6. actual_rounds == max_round == requested_rounds AND
         evolution-report_*.html exists in evo_dir → "done"
      7. actual_rounds == max_round == requested_rounds AND no report file → "finalize"
      8. actual_rounds == max_round < requested_rounds → "round_checkpoint"
         (refine done, ready for next round)
    """
    wm_path = os.path.join(evo_dir, "world_model.json")
    if not os.path.isfile(wm_path):
        return {"stage": "shared_prep", "current_round": 0, "partial_status": {}}

    try:
        with open(wm_path, "r", encoding="utf-8") as f:
            wm = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"stage": "shared_prep", "current_round": 0, "partial_status": {}}

    sess = wm.get("session", {}) or {}
    actual_rounds = int(sess.get("actual_rounds_completed", 0) or 0)
    requested_rounds = int(sess.get("requested_rounds", 0) or 0)

    # Enumerate round_N/ directories
    try:
        entries = os.listdir(evo_dir)
    except OSError:
        entries = []
    round_numbers = _list_round_numbers(evo_dir, entries)
    max_round = round_numbers[-1] if round_numbers else 0

    if max_round == 0:
        return {"stage": "wm_init", "current_round": 0, "partial_status": {}}

    # Inspect partials of max_round
    round_dir = os.path.join(evo_dir, f"round_{max_round}")
    try:
        round_entries = sorted(os.listdir(round_dir))
    except OSError:
        round_entries = []
    parallel_dirs = [
        d for d in round_entries
        if d.startswith("parallel_") and os.path.isdir(os.path.join(round_dir, d))
    ]

    partial_status, completed = _infer_partial_status(round_dir, parallel_dirs)
    stage = _decide_stage(_StageDecisionInput(
        entries, actual_rounds, requested_rounds,
        max_round, completed, len(parallel_dirs)))

    return {
        "stage": stage,
        "current_round": max_round,
        "partial_status": partial_status,
    }


def cmd_infer(args: argparse.Namespace) -> None:
    """Re-infer state from filesystem and overwrite LLM-untrusted fields.

    Preserves drift_status / mainline_stall_count / last_mainline_verdict /
    must_run_before_next_round / max_rounds / agent / session_id / evo_dir.
    """
    evo_dir = os.path.abspath(args.evo_dir)
    if not os.path.isdir(evo_dir):
        _abort(f"evo-dir does not exist: {evo_dir}")

    inferred = _infer_state_from_filesystem(evo_dir)

    state_path = _state_path(evo_dir)
    if not os.path.isfile(state_path):
        # Bootstrap mode: agent hasn't called init yet. Build a minimal state.
        # This path is intentionally limited — hook callers should usually
        # see state.json already exist (init ran in step 3.5).
        if not args.bootstrap:
            _abort(
                f"state.json missing at {state_path}; "
                "pass --bootstrap to create a minimal state, "
                "or run `state_ops.py init` first"
            )
        state = {
            "schema_version": SCHEMA_VERSION,
            "session_id": "unknown",
            "evo_dir": evo_dir,
            "agent": "unknown",
            "stage": inferred["stage"],
            "current_round": inferred["current_round"],
            "max_rounds": 0,
            "round_started_at": None,
            "round_finished_at": None,
            "partial_status": inferred["partial_status"],
            "mainline_stall_count": 0,
            "last_mainline_verdict": "unknown",
            "drift_status": "normal",
            "must_run_before_next_round": [],
            "last_updated_at": None,
        }
    else:
        state = _read_state(evo_dir)
        # Overwrite only the LLM-untrusted fields
        prev_stage = state.get("stage")
        prev_round = state.get("current_round")
        state["stage"] = inferred["stage"]
        state["current_round"] = inferred["current_round"]
        state["partial_status"] = inferred["partial_status"]
        if not args.quiet:
            LOGGER.info(
                "state_ops infer: stage %s → %s, "
                "current_round %s → %s, "
                "partial_status=%s",
                prev_stage, inferred['stage'],
                prev_round, inferred['current_round'],
                inferred['partial_status'],
            )

    _write_state(evo_dir, state)
    if args.print_state:
        DATA_LOGGER.info("%s", json.dumps(state, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Public Python API (for use by hooks and other Python tools)
# ---------------------------------------------------------------------------

def read_state(evo_dir: str) -> dict | None:
    """Read state.json. Returns None if missing (not an error for hooks)."""
    path = _state_path(evo_dir)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return None


def _repo_root_from_script() -> str | None:
    """Best-effort repo-root derivation from this script's location.

    state_ops.py lives at <repo>/plugins-community/ops-perf-evolution/skills/evolution-world-model/scripts/.
    Walk upward from __file__ until a directory containing the ops
    subtree is found. Used as a cwd-independent fallback so the anchor scan
    works even when the calling hook's cwd is not the repo root.
    Returns None if no such directory is found.
    """
    cur = os.path.dirname(os.path.abspath(__file__))
    while cur and cur != os.path.dirname(cur):
        if os.path.isdir(os.path.join(cur, ".claude", "skills")):
            return cur
        cur = os.path.dirname(cur)
    return None


def _scan_anchor_for_evo_dir(output_dir: str) -> str | None:
    """Scan output_dir for .ops-evo_current_session_*.json anchor files.

    `output_dir` must be an absolute path (resolved by the caller, either
    from a repo-root signal seen during the upward walk or from
    _repo_root_from_script) so the scan does not depend on the process cwd.
    Only anchors ≤24h old are considered to avoid stale matches.

    Returns the evo_dir from the most recently modified valid anchor,
    or None if no anchors exist / all are stale.
    """
    import glob
    import time

    pattern = os.path.join(output_dir, ".ops-evo_current_session_*.json")
    anchor_files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    now = time.time()

    for anchor_path in anchor_files:
        # Staleness check: skip anchors older than 24 hours
        try:
            anchor_age = now - os.path.getmtime(anchor_path)
        except OSError:
            continue
        if anchor_age > 86400:
            continue

        try:
            with open(anchor_path, "r", encoding="utf-8") as f:
                anchor = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        evo_dir = anchor.get("evo_dir", "")
        if evo_dir and os.path.isdir(evo_dir) and os.path.isfile(os.path.join(evo_dir, "state.json")):
            return os.path.abspath(evo_dir)

    return None


def find_evo_dir(start: str | None = None) -> str | None:
    """Walk upward from `start` (cwd by default) looking for state.json.

    If walk-up fails, scan output/.ops-evo_current_session_*.json anchors
    as a fallback (agents running from repo root won't find state.json via
    upward walk). The fallback resolves an absolute output/ path from
    either a repo-root signal seen during the walk or this script's own
    location, so it no longer depends on the process cwd.

    Stops at filesystem root. Returns the directory containing state.json,
    or None if not found.
    """
    cur = os.path.abspath(start or os.getcwd())
    repo_root_signal: str | None = None
    while True:
        if os.path.isfile(os.path.join(cur, "state.json")):
            return cur
        # Record the first ancestor that owns an output/ dir — that is the
        # repo root, where session anchors live. Collected while we already
        # walk upward, so the fallback scan gets an absolute path for free.
        if repo_root_signal is None and os.path.isdir(os.path.join(cur, "output")):
            repo_root_signal = cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    # Fallback: scan session anchors under <repo_root>/output. Prefer the
    # repo root detected during the walk; otherwise derive it from this
    # script's location (cwd-independent).
    repo_root = repo_root_signal or _repo_root_from_script()
    if repo_root is None:
        return None
    return _scan_anchor_for_evo_dir(os.path.join(repo_root, "output"))


def check_stage_artifacts(state: dict, evo_dir: str) -> list[str]:
    """Public wrapper around _check_stage_artifacts for hook use."""
    return _check_stage_artifacts(state, evo_dir)


def infer_state_from_filesystem(evo_dir: str) -> dict:
    """Public wrapper around _infer_state_from_filesystem.

    Returns {"stage", "current_round", "partial_status"} from disk evidence.
    Does not write anything — caller decides how to merge.
    """
    return _infer_state_from_filesystem(evo_dir)


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------

def _register_state_commands(sub):
    """注册状态读写类子命令：init / read / write-stage / write-partial / reset-partial。"""
    p = sub.add_parser("init", help="Create a fresh state.json")
    p.add_argument("--evo-dir", required=True)
    p.add_argument("--agent", required=True, choices=VALID_AGENTS)
    p.add_argument("--session-id", required=True)
    p.add_argument("--max-rounds", type=int, required=True)
    p.add_argument("--parallel-num", type=int, default=0,
                   help="Expected number of parallel partials per round (used for R7 anti-skip check). 0 = no check.")
    p.add_argument("--force", action="store_true", help="Overwrite existing state.json")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("read", help="Print state.json")
    p.add_argument("--evo-dir", required=True)
    p.set_defaults(func=cmd_read)

    p = sub.add_parser("write-stage", help="Update stage field")
    p.add_argument("--evo-dir", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--round", type=int, default=None)
    p.set_defaults(func=cmd_write_stage)

    p = sub.add_parser("write-partial", help="Update partial_status[idx]")
    p.add_argument("--evo-dir", required=True)
    p.add_argument("--parallel-idx", type=int, required=True)
    p.add_argument("--status", required=True, choices=VALID_PARTIAL_STATUS)
    p.set_defaults(func=cmd_write_partial)

    p = sub.add_parser("reset-partial", help="Clear partial_status")
    p.add_argument("--evo-dir", required=True)
    p.set_defaults(func=cmd_reset_partial)


def _register_flag_commands(sub):
    """注册标志位类子命令：set-verdict / set-drift / mark-must-run / clear-must-run。"""
    p = sub.add_parser("set-verdict", help="Update last_mainline_verdict")
    p.add_argument("--evo-dir", required=True)
    p.add_argument("--verdict", required=True, choices=VALID_VERDICTS)
    p.set_defaults(func=cmd_set_verdict)

    p = sub.add_parser("set-drift", help="Update drift_status")
    p.add_argument("--evo-dir", required=True)
    p.add_argument("--status", required=True, choices=VALID_DRIFT)
    p.set_defaults(func=cmd_set_drift)

    p = sub.add_parser("mark-must-run", help="Append step to must_run_before_next_round")
    p.add_argument("--evo-dir", required=True)
    p.add_argument("--step", required=True)
    p.set_defaults(func=cmd_mark_must_run)

    p = sub.add_parser("clear-must-run", help="Remove step from must_run_before_next_round")
    p.add_argument("--evo-dir", required=True)
    p.add_argument("--step", required=True)
    p.set_defaults(func=cmd_clear_must_run)


def _register_keys_infer_commands(sub):
    """注册校验/read_keys/infer 类子命令。"""
    p = sub.add_parser("validate", help="Validate state.json schema and artifacts")
    p.add_argument("--evo-dir", required=True)
    p.add_argument("--check-stage-artifacts", action="store_true")
    p.set_defaults(func=cmd_validate)

    # v3.2 Phase D2: read_keys do-not-reread 接口
    p = sub.add_parser("add-read-keys", help="Append source_keys to state.read_keys (dedup)")
    p.add_argument("--evo-dir", required=True)
    p.add_argument("--keys", required=True,
                   help="逗号分隔的 source_keys 列表（如 'evolution-strategies#card/P1_double_buffer,...'）")
    p.set_defaults(func=cmd_add_read_keys)

    p = sub.add_parser("get-read-keys", help="Print state.read_keys (for prompt injection)")
    p.add_argument("--evo-dir", required=True)
    p.add_argument("--format", choices=["json", "lines", "markdown"], default="markdown",
                   help="输出格式（默认 markdown，可注入 prompt）")
    p.set_defaults(func=cmd_get_read_keys)

    p = sub.add_parser("clear-read-keys", help="Clear state.read_keys (drift breaker)")
    p.add_argument("--evo-dir", required=True)
    p.set_defaults(func=cmd_clear_read_keys)

    p = sub.add_parser(
        "infer",
        help="Re-derive stage/current_round/partial_status from filesystem evidence",
    )
    p.add_argument("--evo-dir", required=True)
    p.add_argument("--print-state", action="store_true",
                   help="Print the merged state.json to stdout after writing")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress stderr summary line")
    p.add_argument("--bootstrap", action="store_true",
                   help="Create a minimal state.json if it does not exist")
    p.set_defaults(func=cmd_infer)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Per-evolution runtime state machine (state.json)."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _register_state_commands(sub)
    _register_flag_commands(sub)
    _register_keys_infer_commands(sub)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args()
    try:
        result = args.func(args)
    except StateOpsError as e:
        LOGGER.error("state_ops: FATAL — %s", e)
        sys.exit(e.code)
    sys.exit(result if isinstance(result, int) else 0)


if __name__ == "__main__":
    main()
