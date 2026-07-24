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
"""
wm_ops.py — World Model CLI operations for LINGXI evolution.

CLI subcommands:
  select          --path <wm.json> --n <int>      Select top-N open nodes by utility score
  validate        --path <wm.json>                Validate invariants (non-zero exit = errors)
  summary         --path <wm.json> [--max-chars 1200]  Compact summary for sub-agent injection
  deep-profiling  --wm-path <wm.json> --node-id <id> --work-dir <dir> --op-name <name>
                  Run deep profiling analysis and write evidence to world model
  refine          --wm-path <wm.json> --round <int> --results-dir <dir> --parallel-map <json>
                  Deterministic world model update after a round (score, children, stagnation)
  diagnose        --wm-path <wm.json> --node-id <id> --failure-type <type> --failure-reason <str>
                  Write failure diagnosis for a node (impl_error → fix child, strategy_infeasible → seal)
"""

import argparse
import json
import logging
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Optional

from wm_log import DATA_LOGGER

LOGGER = logging.getLogger(__name__)

# Ensure sibling scripts (profiling_evidence, state_ops, …) are importable when
# wm_ops.py is loaded via importlib (e.g. tests/test_e2e_multishape_smoke.py).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Ledger 模块（v3.2 Phase D1 artifacts + finalize-ledger），从 wm_ops 拆分
from wm_ledger import (  # noqa: E402
    _collect_ancestry_failed_strategies,
    _maybe_write_ledger,
    _all_known_strategy_ids,
    cmd_finalize_ledger,
)

# Audit 模块（diagnose / filter-candidates / validate-diagnosis / verify-notes），从 wm_ops 拆分
from wm_audit import (  # noqa: E402
    cmd_diagnose,
    cmd_filter_candidates,
    cmd_validate_diagnosis,
    cmd_verify_notes,
)


# v3.2 Stage 3 收口：expected_gain 通过 compute_utility 注入 select 排序
# 纯函数，无 LLM。父无 facts/diagnosis 或子无 strategy_combination 时返回 0.0，
# utility 退化为旧公式 → 完全向后兼容。
try:
    from profiling_evidence import compute_expected_gain as _compute_expected_gain
except ImportError:
    _compute_expected_gain = None


# ---------------------------------------------------------------------------
# Multi-shape gating helpers
# ---------------------------------------------------------------------------

# Strategy that signals shape-specialized variant approach (P-ShapeSpec-01)
P_SHAPE_SPEC = "P-ShapeSpec-01"

GATING_FAILED = "failed"
GATING_TARGET_REGRESSION = "target_regression"
GATING_GENERALIZATION_REGRESSION = "generalization_regression"
GATING_PARTIAL_PASSED = "partial_passed"
GATING_FULLY_PASSED = "fully_passed"


def _is_multi_shape_eval(eval_result: dict) -> bool:
    """Tell whether the evaluation result was produced by the multi-shape pipeline.

    Detected by presence of aggregate or shape_results keys. Old single-shape
    pipeline only produces baseline/evolved/comparison.
    """
    return isinstance(eval_result, dict) and (
        "aggregate" in eval_result or "shape_results" in eval_result
    )


def _node_shape_divergence(node: dict) -> float:
    """(max - min) / max across the node's target speedups; 0 if not multi-shape."""
    agg = node.get("aggregate") or {}
    tmax = agg.get("target_max_speedup")
    tmin = agg.get("target_min_speedup")
    if not (isinstance(tmax, (int, float)) and isinstance(tmin, (int, float))):
        return 0.0
    if tmax <= 0:
        return 0.0
    return max(0.0, (tmax - tmin) / tmax)


def _inject_shape_spec_into_strategies(strategies: list) -> list:
    """Append P-ShapeSpec-01 to a strategy_combination if not already present."""
    s = list(strategies or [])
    if P_SHAPE_SPEC not in s:
        s.append(P_SHAPE_SPEC)
    return s


# ---------------------------------------------------------------------------
# Optional state.json sync with state_ops.py
# ---------------------------------------------------------------------------
# wm_ops subcommands operate on world_model.json. The runtime state cursor
# lives in <evo_dir>/state.json (see state_ops.py / state_schema.md).
# These helpers infer evo_dir from the wm_path and re-derive state as a
# side effect. All failures are silently ignored — state.json is optional;
# wm_ops remains backward-compatible if it is absent.

def _maybe_update_state_stage(wm_path: str, stage: str, round_num: Optional[int] = None) -> None:
    """Best-effort: if <dirname(wm_path)>/state.json exists, update its stage.

    Silently noops if state.json missing or state_ops import fails. Never
    raises — wm_ops must not break when state.json infrastructure is absent.
    """
    try:
        evo_dir = os.path.dirname(os.path.abspath(wm_path))
        state_path = os.path.join(evo_dir, "state.json")
        if not os.path.isfile(state_path):
            return
        # Import lazily to avoid hard dependency at module load
        from state_ops import _read_state, _write_state, ALL_STAGES  # type: ignore
        if stage not in ALL_STAGES:
            return
        state = _read_state(evo_dir)
        prev = state.get("stage")
        state["stage"] = stage
        if round_num is not None:
            state["current_round"] = round_num
        _write_state(evo_dir, state)
        LOGGER.info("  [state] stage %s → %s%s", prev, stage,
                    f" (round={round_num})" if round_num is not None else "")
    except Exception:
        # state.json sync is optional; never let it break wm_ops
        pass


# ---------------------------------------------------------------------------
# Drift circuit breaker
# ---------------------------------------------------------------------------
# 当全局 stagnation_count 或分支 stagnation_count_vs_base 超过
# DRIFT_THRESHOLD 时，将 state.drift_status 置为 "replan_required"，使
# 下一轮的 GATE 步骤转向 drift_replan 流程（open_exploration 饱和 +
# 强制新来源读取）。当停滞恢复时自动清回 "normal"。

DRIFT_THRESHOLD = 2  # consecutive stalled rounds to trigger drift


def _decide_drift_status(wm: dict) -> tuple[str, str]:
    """Return (new_drift_status, reason) based on world model stagnation counters.

    Args:
        wm: world model dict (post-refine state).

    Returns:
        (status, reason):
          status ∈ {"replan_required", "normal"}
          reason: human-readable one-liner for logging
    """
    sc = int(wm.get("stagnation_count", 0))
    scvb = int(wm.get("stagnation_count_vs_base", 0))

    if sc >= DRIFT_THRESHOLD:
        return "replan_required", (
            f"stagnation_count={sc} ≥ {DRIFT_THRESHOLD}: "
            f"no global-best progress for {sc} consecutive rounds"
        )
    if scvb >= DRIFT_THRESHOLD:
        return "replan_required", (
            f"stagnation_count_vs_base={scvb} ≥ {DRIFT_THRESHOLD}: "
            f"no variant beat its parent for {scvb} consecutive rounds"
        )
    return "normal", f"stagnation_count={sc}, vs_base={scvb} below threshold"


def _maybe_update_drift_status(wm_path: str, wm: dict) -> None:
    """Best-effort: detect stall and write state.drift_status.

    Called from cmd_refine after world_model.json is updated. Noops if
    state.json missing or import fails. Mirrors the noop discipline of
    _maybe_update_state_stage.
    """
    try:
        evo_dir = os.path.dirname(os.path.abspath(wm_path))
        state_path = os.path.join(evo_dir, "state.json")
        if not os.path.isfile(state_path):
            return
        from state_ops import _read_state, _write_state  # type: ignore

        state = _read_state(evo_dir)
        prev = state.get("drift_status", "normal")
        new_status, reason = _decide_drift_status(wm)

        if prev == new_status:
            return  # idempotent — no-op log spam

        state["drift_status"] = new_status
        _write_state(evo_dir, state)
        marker = "[DRIFT]" if new_status == "replan_required" else "[drift cleared]"
        LOGGER.info("  %s drift_status %s → %s (%s)", marker, prev, new_status, reason)
    except Exception as e:
        # drift status 更新失败不影响主流程，仅记录
        LOGGER.debug("_maybe_* suppressed: %s", e)


def _read_state_field_safe(wm_path: str, field: str) -> Any:
    """Read a top-level field from <evo_dir>/state.json. Returns None on any error."""
    try:
        evo_dir = os.path.dirname(os.path.abspath(wm_path))
        state_path = os.path.join(evo_dir, "state.json")
        if not os.path.isfile(state_path):
            return None
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        return state.get(field)
    except Exception:
        return None


def _maybe_infer_state(wm_path: str) -> None:
    """Re-derive state.stage/current_round/partial_status from filesystem
    evidence (replaces an older single-field _maybe_update_state_stage helper).

    Noop if state.json missing. Drift_status / stall_count / must_run are
    preserved (those are wm_ops/setup's responsibility, not LLM-trusted state).
    """
    try:
        evo_dir = os.path.dirname(os.path.abspath(wm_path))
        state_path = os.path.join(evo_dir, "state.json")
        if not os.path.isfile(state_path):
            return
        from state_ops import _read_state, _write_state, _infer_state_from_filesystem  # type: ignore
        inferred = _infer_state_from_filesystem(evo_dir)
        state = _read_state(evo_dir)
        prev_stage = state.get("stage")
        prev_round = state.get("current_round")
        state["stage"] = inferred["stage"]
        state["current_round"] = inferred["current_round"]
        state["partial_status"] = inferred["partial_status"]
        _write_state(evo_dir, state)
        if prev_stage != inferred["stage"] or prev_round != inferred["current_round"]:
            LOGGER.info(
                "  [state] inferred: stage %s → %s, round %s → %s",
                prev_stage, inferred['stage'], prev_round, inferred['current_round'],
            )
    except Exception as e:
        # state infer 失败不影响主流程，仅记录
        LOGGER.debug("_maybe_* suppressed: %s", e)


def _maybe_clear_drift(wm_path: str) -> None:
    """Drift_status auto-clear after SELECT consumed the drift signal.

    Called from cmd_select right after a drift-aware selection runs. The drift
    signal is a one-shot trigger; consuming it must reset state.drift_status to
    "normal" so subsequent rounds aren't repeatedly diverted.
    """
    try:
        evo_dir = os.path.dirname(os.path.abspath(wm_path))
        state_path = os.path.join(evo_dir, "state.json")
        if not os.path.isfile(state_path):
            return
        from state_ops import _read_state, _write_state  # type: ignore
        state = _read_state(evo_dir)
        if state.get("drift_status") == "replan_required":
            state["drift_status"] = "normal"
            _write_state(evo_dir, state)
            LOGGER.info("  [drift consumed] drift_status replan_required → normal")
    except Exception as e:
        # drift 清除失败不影响主流程，仅记录
        LOGGER.debug("_maybe_* suppressed: %s", e)


def _profiling_complete(parallel_dir: str) -> bool:
    """Check whether msprof profiling produced its key artifact in a parallel dir.

    The canonical evidence is `parallel_K/profiling/.../op_summary_*.csv` —
    this CSV is what ascendc-profiling-analysis actually consumes. If absent,
    profiling either was skipped or failed silently.

    Returns True if at least one op_summary_*.csv exists under profiling/.
    """
    prof_dir = os.path.join(parallel_dir, "profiling")
    if not os.path.isdir(prof_dir):
        return False
    # Walk the directory tree; profiling output is nested deep
    for _, _, files in os.walk(prof_dir):
        for f in files:
            if f.startswith("op_summary_") and f.endswith(".csv"):
                return True
    return False


def _read_precision_passed(eval_path: str) -> bool:
    """读取 evaluation_results.json 的 precision_passed，读取失败返回 False。"""
    if not os.path.isfile(eval_path):
        return False
    try:
        with open(eval_path, "r", encoding="utf-8") as f:
            eres = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return bool(eres.get("precision_passed"))


def _collect_missing_profiling(results_dir: str, parallel_map: dict) -> list:
    """收集跑了成功评估但缺 profiling 输出的 parallel 下标。

    只关心实际跑成功的 partial — 编译失败的自然没有 profiling。
    """
    missing = []
    for p_idx_str in parallel_map:
        parallel_dir = os.path.join(results_dir, f"parallel_{p_idx_str}")
        eval_path = os.path.join(parallel_dir, "evaluation_results.json")
        if not _read_precision_passed(eval_path):
            continue
        if not _profiling_complete(parallel_dir):
            missing.append(p_idx_str)
    return missing


def _maybe_mark_profiling_skipped(wm_path: str, results_dir: str, parallel_map: dict) -> None:

    """R9: detect if msprof was skipped for any passed partial.

    For each parallel slot whose evaluation_results.json reports
    precision_passed=True (i.e. the kernel ran), check whether
    profiling/.../op_summary_*.csv exists. If any passed partial lacks
    profiling, mark `msprof` into state.must_run_before_next_round so the
    next round's GATE / Stop hook R4 blocks until profiling is rerun.
    """
    try:
        evo_dir = os.path.dirname(os.path.abspath(wm_path))
        state_path = os.path.join(evo_dir, "state.json")
        if not os.path.isfile(state_path):
            return

        missing = _collect_missing_profiling(results_dir, parallel_map)

        if not missing:
            return

        from state_ops import _read_state, _write_state  # type: ignore
        state = _read_state(evo_dir)
        pending = state.setdefault("must_run_before_next_round", [])
        if "msprof" not in pending:
            pending.append("msprof")
            _write_state(evo_dir, state)
            LOGGER.warning(
                "  [R9] msprof missing for passed partial(s) %s; "
                "must_run_before_next_round += msprof",
                missing,
            )
    except Exception as e:
        # profiling skipped 标记失败不影响主流程，仅记录
        LOGGER.debug("_maybe_* suppressed: %s", e)


def _read_precision_flag(eval_path: str):
    """读取 evaluation_results.json 的 precision_passed 标志。

    返回 True/False；文件缺失或解析失败返回 None（不计入统计）。
    """
    if not os.path.isfile(eval_path):
        return None
    try:
        with open(eval_path, "r", encoding="utf-8") as f:
            eres = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    return bool(eres.get("precision_passed", True))


def _count_precision_failures(results_dir: str, parallel_map: dict) -> tuple:
    """统计本轮 partial 的精度评估结果，返回 (total, failed)。"""
    total = 0
    failed = 0
    for p_idx_str in parallel_map:
        eval_path = os.path.join(
            results_dir, f"parallel_{p_idx_str}", "evaluation_results.json")
        flag = _read_precision_flag(eval_path)
        if flag is None:
            continue
        total += 1
        if not flag:
            failed += 1
    return total, failed


def _maybe_warn_precision_failures(parallel_map: dict, results_dir: str) -> None:

    """R10 (warn-only): if ≥50% of partials failed precision, alert.

    Pure stderr warning — does not block. Surfaces large-scale precision
    regressions early so the user notices before agent claims success.
    """
    try:
        total, failed = _count_precision_failures(results_dir, parallel_map)
        if total >= 2 and failed * 2 >= total:
            LOGGER.warning(
                "  [R10 WARN] %d/%d partials failed precision in this round. "
                "Review evaluation_results.json before claiming success.",
                failed, total,
            )
    except Exception as e:
        # precision 失败告警失败不影响主流程，仅记录
        LOGGER.debug("_maybe_* suppressed: %s", e)


# Ensure state_ops is importable when wm_ops is run as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Core algorithm: utility computation
# ---------------------------------------------------------------------------

def _baseline_mismatch_penalty(node: dict, wm: dict) -> float:
    """A5: soft penalty for nodes misaligned with root-level baseline_evidence.

    Returns:
      0.0 if wm has no baseline_evidence (fallback to pre-A5 behavior)
      -2.0 if node's strategy_combination intersects baseline anti_strategies
      -1.0 if node has a non-empty strategy_combination that has NO intersection
           with baseline suggested_strategies (primary + secondary)
      0.0 otherwise

    Cast so the penalty cannot flip cross-layer ordering (see design doc):
    abs(-2.0) < typical parent_score differential and w_root_explore, so
    a misaligned node with a very high-scoring parent can still outrank an
    aligned node with a weak parent — this is intentional.
    """
    be = wm.get("baseline_evidence") if isinstance(wm, dict) else None
    if not be or not isinstance(be, dict):
        return 0.0
    strats = node.get("strategy_combination") or []
    if not strats:
        return 0.0
    s_set = set(strats)
    anti = set(be.get("anti_strategies") or [])
    if s_set & anti:
        return -2.0
    suggested = set(be.get("suggested_strategies") or [])
    if suggested and not (s_set & suggested):
        return -1.0
    return 0.0


def compute_utility(node: dict, nodes: dict, wm: Optional[dict] = None) -> float:
    """Compute utility score for a given node using the unified world model formula.

    Unified formula (authoritative — all other docs reference this):
        utility = 3.0 × parent_score
                + 2.5 × (5 - difficulty)
                + 0.75 × depth
                + w_root_explore       (2.0 if parent is root, else 0.0)
                + w_evidence           (1.5 if parent has profiling_evidence, else 0.0)
                + w_baseline_mismatch  (A5: 0.0 / -1.0 / -2.0, requires wm)
                + w_close_to_target    (multi-shape: +0.5 if parent partial_passed
                                       with score >= target_speedup × 0.85)
                + w_shape_divergence   (multi-shape: +1.0 if parent shape_divergence >= 0.20
                                       AND node strategy_combination contains P-ShapeSpec-01)
                + w_expected_gain      (v3.2 Stage 3: min(3.0, 2.0 × log1p(expected_gain)),
                                       computed from parent.facts × parent.diagnosis.labels ×
                                       node.strategy_combination via compute_expected_gain)

    Rationale for each term:
        parent_score:          Exploit — children of high-performing nodes are more promising
        difficulty:            Prefer easier implementations first (low-hanging fruit)
        depth:                 Mild encouragement for depth-first exploitation
        w_root_explore:        Ensure first-layer breadth is explored before deep-diving
        w_evidence:            Prioritize nodes whose parent has instruction-level profiling
        w_baseline_mismatch:   Soft-penalize strategies misaligned with baseline bottleneck
        w_close_to_target:     Multi-shape — keep pushing parents that are "this close" to target
        w_shape_divergence:    Multi-shape — encourage P-ShapeSpec-01 children when target shapes diverge
        w_expected_gain:       v3.2 Stage 3 forward-looking reward — strategies whose triggers
                               intersect parent.diagnosis.bottleneck_labels and address larger
                               facts ratios get higher utility. Amdahl upper bound; cap at 3.0
                               (same magnitude as w_root_explore) to avoid single-slot dominance.
                               Decays via log1p to keep huge gains from monopolizing.
    """
    parent_id = node.get("parent_id") or "root"
    parent = nodes.get(parent_id, {})
    parent_score = parent.get("score") or 1.0
    difficulty = node.get("difficulty", 3)
    depth = node.get("depth", 1)
    w_root_explore = 2.0 if parent_id == "root" else 0.0
    w_evidence = 1.5 if parent.get("profiling_evidence") else 0.0
    w_baseline_mismatch = _baseline_mismatch_penalty(node, wm) if wm else 0.0

    # multi-shape weights (parent-derived)
    w_close = 0.0
    w_div = 0.0
    parent_gating = parent.get("gating")
    target_speedup = None
    if isinstance(wm, dict):
        target_speedup = (wm.get("shape_targets") or {}).get("target_speedup_threshold")
        if target_speedup is None:
            target_speedup = wm.get("target_speedup")
    is_partial_gating = parent_gating == GATING_PARTIAL_PASSED
    has_numeric_targets = (isinstance(target_speedup, (int, float))
                           and isinstance(parent_score, (int, float)))
    meets_threshold = has_numeric_targets and parent_score >= target_speedup * 0.85
    if is_partial_gating and meets_threshold:
        w_close = 0.5

    if _node_shape_divergence(parent) >= 0.20:
        if P_SHAPE_SPEC in (node.get("strategy_combination") or []):
            w_div = 1.0

    # v3.2 Stage 3: expected_gain — facts × diagnosis labels × strategy reverse-lookup
    w_expected_gain = 0.0
    if _compute_expected_gain is not None:
        parent_facts = parent.get("facts")
        parent_diag = parent.get("diagnosis") or {}
        parent_labels = parent_diag.get("bottleneck_labels") or []
        gain = _compute_expected_gain(
            node.get("strategy_combination") or [],
            parent_facts,
            parent_labels,
        )
        if gain > 0:
            w_expected_gain = min(3.0, 2.0 * math.log1p(gain))

    return (3.0 * parent_score
            + 2.5 * (5 - difficulty)
            + 0.75 * depth
            + w_root_explore
            + w_evidence
            + w_baseline_mismatch
            + w_close
            + w_div
            + w_expected_gain)


# ---------------------------------------------------------------------------
# Optimization type inference
# ---------------------------------------------------------------------------

_BANDWIDTH_STRATEGIES = frozenset({
    "P1", "P7", "P10", "P11",
    # P19-P26: buffer management + DataCopy params
    "P19", "P20", "P21", "P22",
    "P24", "P25", "P26",
    # P32-P33: special copy patterns
    "P32", "P33",
    # P34-P45: buffer resident / reuse
    "P34", "P35", "P37", "P38", "P40",
    "P41", "P43", "P44",
    "P45",
    # P49, P52: hardware dequant, L2 cache hint
    "P49", "P52",
    # P53-P88: bandwidth-related (data movement, format, buffer, cache)
    "P53", "P56", "P59", "P60", "P61", "P63",
    "P64", "P65", "P66", "P69", "P70", "P71",
    "P74", "P76", "P78", "P81", "P83", "P85",
})
_TILING_STRATEGIES = frozenset({
    "P2", "P4", "P5", "P8",
    # P28-P30: sync & pipeline control
    "P28", "P29", "P30",
    # P47, P51: diagonal scheduling, dynamic core ratio
    "P47", "P51",
    # P53-P88: tiling-related (sync, partition, pipeline, core ratio)
    "P54", "P55", "P57", "P58", "P67", "P68",
    "P72", "P73", "P75", "P77", "P80",
    "P84", "P86", "P87", "P88",
})

_REGISTER_OPT_STRATEGIES = frozenset()
_VF_FUSION_STRATEGIES = frozenset()
_INSTRUCTION_SCHED_STRATEGIES = frozenset()


def infer_optimization_type(strategy_combination: list, mode: str = "strategy_guided") -> str:
    """Infer the performance optimization type from strategy combination.

    Returns one of: "bandwidth", "tiling", "algorithm",
    "register_opt", "vf_fusion", "instruction_sched".
    D/A-series strategies are ignored (precision constraints, not perf directions).
    P19-P88 (data movement / CV fusion / advanced) are classified as bandwidth, tiling (sync),
    or algorithm (default).
    A5 R-series strategies map to register_opt/vf_fusion/instruction_sched/bandwidth/algorithm.
    """
    if mode in ("open_exploration", "profiling_driven"):
        return "algorithm"
    s = set(strategy_combination) if strategy_combination else set()
    bw = len(s & _BANDWIDTH_STRATEGIES)
    tl = len(s & _TILING_STRATEGIES)
    ro = len(s & _REGISTER_OPT_STRATEGIES)
    vf = len(s & _VF_FUSION_STRATEGIES)
    isc = len(s & _INSTRUCTION_SCHED_STRATEGIES)
    counts = {"bandwidth": bw, "tiling": tl, "register_opt": ro,
              "vf_fusion": vf, "instruction_sched": isc}
    best_type = max(counts, key=counts.get)
    if counts.get(best_type, 0) > 0:
        return best_type
    return "algorithm"


# ---------------------------------------------------------------------------
# soft prune
# ---------------------------------------------------------------------------

def soft_prune_dead_branches(nodes: dict) -> list:
    """Soft-prune open nodes under sealed ancestors.

    An ancestor is considered "sealed" if any of:
      - status=="failed" AND difficulty>=5 (impl-level seal, legacy behavior)
      - direction_sealed is True (direction-level seal set by diagnose on a
        passed node whose direction has been semantically disproven)

    Walks each open node's parent chain upward. If a sealed ancestor is found
    before a healthy (passed but non-sealed / completed) ancestor, the open
    node is soft-pruned by setting its difficulty to 5 (never deleted).

    Returns list of pruned node IDs.
    """
    pruned = []
    for nid, nd in list(nodes.items()):
        if nd.get("status") != "open":
            continue
        current_id = nd.get("parent_id")
        should_prune = False
        visited = set()
        while current_id and current_id in nodes and current_id not in visited:
            visited.add(current_id)
            ancestor = nodes[current_id]
            sealed_failed = (ancestor.get("status") == "failed"
                             and ancestor.get("difficulty", 0) >= 5)
            sealed_direction = bool(ancestor.get("direction_sealed"))
            if sealed_failed or sealed_direction:
                should_prune = True
                break
            # A passed/completed ancestor without direction_sealed is healthy;
            # stop the upward walk (don't cross a healthy branch).
            if ancestor.get("status") in ("passed", "completed"):
                break
            current_id = ancestor.get("parent_id")
        if should_prune:
            nd["difficulty"] = 5
            pruned.append(nid)
    return pruned


# ---------------------------------------------------------------------------
# soft demote (A4: bridging stale-direction signal → SELECT without hard-prune)
# ---------------------------------------------------------------------------


def _is_soft_disproven(nd: dict, nodes: dict, stag_threshold: float) -> bool:
    """判断 passed 节点是否"软证伪"：未实质超越父节点且瓶颈未转移。"""
    if nd.get("status") != "passed":
        return False
    if nd.get("bottleneck_shift"):
        return False
    parent_id = nd.get("parent_id")
    if not parent_id:
        return False
    parent = nodes.get(parent_id, {})
    parent_score = parent.get("score") or 1.0
    self_score = nd.get("score") or 0.0
    # Weak improvement under the current measurement-quality threshold.
    # Absolute regression (self < parent) is subsumed since stag_threshold ≥ 1.0.
    return self_score < parent_score * stag_threshold


def _demote_one_open_node(cid: str, ch: dict, round_num: Optional[int]) -> bool:
    """对单个 open 后代执行 difficulty+1（幂等：同轮不重复提升）。

    返回是否实际执行了提升。
    """
    marker = ch.get("demoted_in_round")
    if round_num is not None and marker == round_num:
        return False  # already demoted this round, skip bump but still walk children
    old = ch.get("difficulty", 3) or 3
    ch["difficulty"] = min(4, old + 1)
    if round_num is not None:
        ch["demoted_in_round"] = round_num
    return True


def _demote_descendants(nd: dict, nodes: dict, round_num: Optional[int]) -> list:
    """BFS 遍历 nd 的后代，对 open 节点执行 difficulty+1 降级。

    不跨越 passed 后代（其子树由自身指标评判）。
    """
    demoted = []
    frontier = list(nd.get("children", []))
    visited = set()
    while frontier:
        cid = frontier.pop()
        if cid in visited or cid not in nodes:
            continue
        visited.add(cid)
        ch = nodes[cid]
        # Don't cross a passed descendant (its own sub-tree is judged
        # by its own metrics).
        if ch.get("status") == "passed":
            continue
        # Only demote open nodes; also require idempotence on round.
        if ch.get("status") == "open" and _demote_one_open_node(cid, ch, round_num):
            demoted.append(cid)
        frontier.extend(ch.get("children", []) or [])
    return demoted


def soft_demote_stale_directions(
    wm: dict,
    quality: str = "good",
    round_num: Optional[int] = None,
) -> list:
    """Penalize open descendants of passed nodes whose speedup barely beat
    their parent AND whose bottleneck did not shift.

    Signal:
      A passed node n with n.score < parent.score × stag_threshold is
      "soft disproven" when bottleneck_shift is not set — the variant ran,
      but it neither improved meaningfully nor opened a new bottleneck
      front. The direction has not been hard-disproven (that is A6 via
      diagnose), but exploring further down this branch is lower value
      than exploring elsewhere.

    Effect:
      All open descendants of such nodes get difficulty += 1, capped at 4
      (never 5 — that's reserved for hard seal via diagnose). This lets
      compute_utility naturally deprioritize them vs. siblings with
      stronger parents, while keeping them selectable as a last resort.

    Idempotence:
      Marks each demoted node with demoted_in_round so repeated refine
      calls in the same round do not re-inflate difficulty.

    Returns list of demoted node IDs.
    """
    stag_threshold = _THRESHOLDS.get(quality, _THRESHOLDS["good"])["stagnation"]
    nodes = wm.get("decision_tree", {}).get("nodes", {}) or {}
    demoted = []

    for nd in list(nodes.values()):
        if not _is_soft_disproven(nd, nodes, stag_threshold):
            continue
        demoted.extend(_demote_descendants(nd, nodes, round_num))

    return demoted


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------

def _strategy_signature(strategies) -> frozenset:
    """Canonicalize a strategy_combination for sibling collision detection."""
    if not strategies:
        return frozenset()
    return frozenset(str(s).strip() for s in strategies if s)


def _sig_jaccard(sig_a: frozenset, sig_b: frozenset) -> float:
    """Jaccard similarity between two strategy signatures."""
    if not sig_a and not sig_b:
        return 1.0
    if not sig_a or not sig_b:
        return 0.0
    inter = len(sig_a & sig_b)
    union = len(sig_a | sig_b)
    return inter / union if union else 0.0


# A3 hard-filter threshold: two sibling variants with jaccard > this AND same
# optimization_type are considered a collision and the later one is skipped.
# 0.6 catches "one of two strategies swapped" for length-2 combos and "two of
# three kept" for length-3 combos — both are semantically redundant work.
_COLLISION_JACCARD_THRESHOLD = 0.6


def _collides_with_selected(
    candidate: dict, selected: list
) -> bool:
    """A3: True if candidate's strategy signature jaccard-overlaps with an
    already-selected sibling AND they share optimization_type.

    Same-parent is NOT a requirement for collision — two variants on different
    parents doing literally the same strategy combination still waste a slot.
    """
    cand_sig = _strategy_signature(candidate.get("strategy_combination"))
    if not cand_sig:
        # Empty-strategy nodes (e.g. free exploration) are never collision-filtered
        return False
    cand_type = candidate.get("optimization_type")
    for _nid, sel_nd in selected:
        sel_sig = _strategy_signature(sel_nd.get("strategy_combination"))
        if not sel_sig:
            continue
        if cand_type and sel_nd.get("optimization_type") != cand_type:
            continue
        if _sig_jaccard(cand_sig, sel_sig) > _COLLISION_JACCARD_THRESHOLD:
            return True
    return False


def _parent_eligible(nodes: dict, nid: str) -> bool:
    """Check whether selecting this open node would derive from a parent that
    is marked ineligible (target_shape_regression / generalization_regression).

    Backwards-compatible: nodes without parent_eligible (old single-shape
    evaluations) are treated as eligible.
    """
    nd = nodes.get(nid, {})
    parent_id = nd.get("parent_id") or "root"
    parent = nodes.get(parent_id, {})
    # Only enforce the gate when the parent has been evaluated under the
    # multi-shape pipeline. Older parents (no aggregate / no gating) pass.
    if parent.get("aggregate") is None and parent.get("gating") is None:
        return True
    # Defaults to True if explicit field absent — minimize blast radius for
    # legacy nodes that pre-date this field.
    return bool(parent.get("parent_eligible", True))


def _split_by_mode(open_nodes: list) -> tuple:
    """按 mode 分组为 (strategy_guided, open_exploration)。

    open_exploration 组同时包含 open_exploration 和 profiling_driven 模式。
    """
    strategy_guided = []
    open_exploration = []
    for nid, nd in open_nodes:
        mode = nd.get("mode", "strategy_guided")
        if mode in ("open_exploration", "profiling_driven"):
            open_exploration.append((nid, nd))
        else:
            strategy_guided.append((nid, nd))
    return strategy_guided, open_exploration


class _BranchQuota:
    """Branch diversity constraint：限制每个 parent_id 贡献的槽位数。"""

    def __init__(self, strategy_guided: list, n: int):
        sg_parent_ids = set(
            (nd.get("parent_id") or "root") for _, nd in strategy_guided
        )
        num_active_branches = max(1, len(sg_parent_ids))
        self.max_per_branch = max(1, -(-n // num_active_branches))  # ceil division
        self.branch_count: dict[str, int] = {}

    def ok(self, nd: dict) -> bool:
        """Check if the node's parent branch still has capacity."""
        pid = nd.get("parent_id") or "root"
        return self.branch_count.get(pid, 0) < self.max_per_branch

    def add(self, nd: dict) -> None:
        """Record that a slot was allocated from this node's parent branch."""
        pid = nd.get("parent_id") or "root"
        self.branch_count[pid] = self.branch_count.get(pid, 0) + 1


def _select_sg_guaranteed(strategy_guided: list, sg_slots: int, quota: _BranchQuota,
                          selected_sg: list, used_ids: set):
    """保底轮：每种 optimization_type 的开放节点先各占 1 槽。

    覆盖 A3 类型（bandwidth, tiling, algorithm）和 A5 类型
    （register_opt, vf_fusion, instruction_sched）。
    """
    type_best: dict[str, tuple[str, dict]] = {}
    for nid, nd in strategy_guided:
        t = nd.get("optimization_type") or infer_optimization_type(
            nd.get("strategy_combination", []),
            nd.get("mode", "strategy_guided"),
        )
        if t not in type_best:
            type_best[t] = (nid, nd)  # strategy_guided already sorted desc

    for t in type_best:
        if len(selected_sg) >= sg_slots:
            break
        nid, nd = type_best[t]
        if quota.ok(nd) and not _collides_with_selected(nd, selected_sg):
            selected_sg.append((nid, nd))
            used_ids.add(nid)
            quota.add(nd)


def _select_sg_slots(strategy_guided: list, sg_slots: int, quota: _BranchQuota) -> list:
    """strategy_guided 槽位选择：保底轮 + utility 排序 + 两级放宽兜底。"""
    selected_sg: list[tuple[str, dict]] = []
    used_ids: set[str] = set()

    _select_sg_guaranteed(strategy_guided, sg_slots, quota, selected_sg, used_ids)

    # --- Remaining slots: utility-driven with branch diversity + A3 collision filter ---
    for nid, nd in strategy_guided:
        if len(selected_sg) >= sg_slots:
            break
        if (nid not in used_ids
                and quota.ok(nd)
                and not _collides_with_selected(nd, selected_sg)):
            selected_sg.append((nid, nd))
            used_ids.add(nid)
            quota.add(nd)

    # --- Fallback: if branch+collision constraints left slots unfilled, relax ---
    # Relax order: (a) keep collision filter, drop branch constraint;
    #              (b) drop both (last resort — avoid empty slots)
    if len(selected_sg) < sg_slots:
        for nid, nd in strategy_guided:
            if len(selected_sg) >= sg_slots:
                break
            if nid not in used_ids and not _collides_with_selected(nd, selected_sg):
                selected_sg.append((nid, nd))
                used_ids.add(nid)
    if len(selected_sg) < sg_slots:
        for nid, nd in strategy_guided:
            if len(selected_sg) >= sg_slots:
                break
            if nid not in used_ids:
                selected_sg.append((nid, nd))
                used_ids.add(nid)

    return selected_sg, used_ids


def _fill_oe_slots(selected: list, open_exploration: list, strategy_guided: list,
                   oe_slots: int, used_ids: set):
    """open_exploration 专用槽：oe 节点优先，不足时回退 strategy_guided。"""
    # Fill up to oe_slots with open_exploration nodes; if not enough,
    # fall back to remaining strategy_guided nodes (collision-filtered).
    available_oe = [(nid, nd) for nid, nd in open_exploration if nid not in used_ids]
    oe_taken = 0
    for nid, nd in available_oe[:oe_slots]:
        selected.append((nid, nd))
        used_ids.add(nid)
        oe_taken += 1

    while oe_taken < oe_slots:
        remaining = [
            (nid, nd) for nid, nd in strategy_guided
            if nid not in used_ids and not _collides_with_selected(nd, selected)
        ]
        if not remaining:
            remaining = [(nid, nd) for nid, nd in strategy_guided if nid not in used_ids]
        if not remaining:
            break
        nid, nd = remaining[0]
        selected.append((nid, nd))
        used_ids.add(nid)
        oe_taken += 1


def _pad_free_placeholders(selected: list, n: int):
    """开放节点不足时用自由探索占位补齐。"""
    while len(selected) < n:
        idx = len(selected)
        selected.append((
            f"free_{idx}",
            {
                "id": f"free_{idx}",
                "mode": "strategy_guided",
                "status": "open",
                "strategy_combination": [],
                "description": "自由探索方向，基于已有经验选择多样化策略",
                "parent_id": "root",
                "difficulty": 3,
                "depth": 1,
                "score": None,
                "solution_ref": None,
                "parent_code_ref": None,
                "children": [],
            }
        ))


def _build_result_entry(i: int, nid: str, nd: dict, nodes: dict, wm: dict) -> dict:
    """组装 select 结果项（含父节点信息）。"""
    parent_id = nd.get("parent_id") or "root"
    parent_node = nodes.get(parent_id, {})
    parent_score = parent_node.get("score") or 1.0
    parent_solution_ref = parent_node.get("solution_ref")  # may be None
    # Resolve parent profiling one-liner
    pi = parent_node.get("profiling_insight")
    parent_profiling_one_liner = (
        pi.get("profiling_one_liner") if isinstance(pi, dict) else None
    )

    return {
        "parallel_index": i,
        "node_id": nid,
        "utility": round(compute_utility(nd, nodes, wm), 4),
        "mode": nd.get("mode", "strategy_guided"),
        "description": nd.get("description", ""),
        "strategy_combination": nd.get("strategy_combination", []),
        "parent_id": parent_id,
        "parent_score": parent_score,
        "parent_solution_ref": parent_solution_ref,
        "parent_profiling_one_liner": parent_profiling_one_liner,
        "difficulty": nd.get("difficulty", 3),
        "depth": nd.get("depth", 1),
    }


def select_nodes(wm: dict, n: int, force_open_exploration_min: Optional[int] = None) -> list:
    """
    Select top-N open nodes by utility score with open_exploration slot reservation
    and branch diversity constraint.

    Slot allocation (matches lingxi-evo/SKILL.md §4.0):
      - oe_slots = max(1, ⌈n/4⌉) reserved for open_exploration / profiling_driven
      - sg_slots = n - oe_slots for strategy_guided, sorted by utility descending
      - oe slots fall back to next strategy_guided if not enough oe nodes exist

    Branch diversity constraint:
      - Each parent_id contributes at most ceil(n / num_active_branches) slots
      - Prevents all slots from clustering on a single high-scoring branch
      - Falls back to unconstrained fill if constraint leaves slots empty

    Drift mode:
      - When `force_open_exploration_min` is set (e.g. by cmd_select when
        state.drift_status == "replan_required"), oe_slots is raised to at
        least that many. Default oe_slots formula still applies as a floor.
      - Typical drift call: force_open_exploration_min = max(1, ⌈n/2⌉) so at
        least half the round goes to fresh exploration when search stalls.

    If n == 1, the single slot goes to the highest-utility node regardless of mode.

    Pads with free-exploration placeholders if open nodes are insufficient.

    Returns a list of dicts with parallel_index, node_id, utility, mode, and
    all node fields needed for prompt construction.
    """
    nodes: dict = wm.get("decision_tree", {}).get("nodes", {})

    open_nodes = [
        (nid, nd) for nid, nd in nodes.items()
        if nd.get("status") == "open" and _parent_eligible(nodes, nid)
    ]
    strategy_guided, open_exploration = _split_by_mode(open_nodes)

    # Sort each group by utility descending (wm passed so A5 baseline
    # mismatch penalty can fire when wm.baseline_evidence is present)
    strategy_guided.sort(key=lambda x: -compute_utility(x[1], nodes, wm))
    open_exploration.sort(key=lambda x: -compute_utility(x[1], nodes, wm))

    selected: list[tuple[str, dict]] = []

    if n == 1:
        # Single slot: pick the overall best regardless of mode
        all_sorted = sorted(open_nodes, key=lambda x: -compute_utility(x[1], nodes, wm))
        if all_sorted:
            selected.append(all_sorted[0])
    else:
        # P0-a: ⌈n/4⌉ slots reserved for open_exploration (min 1), rest for strategy_guided
        oe_slots = max(1, -(-n // 4))
        # Drift mode raises oe_slots to force fresh-direction exploration
        if force_open_exploration_min is not None:
            oe_slots = max(oe_slots, min(force_open_exploration_min, n))
        sg_slots = n - oe_slots

        quota = _BranchQuota(strategy_guided, n)
        selected_sg, used_ids = _select_sg_slots(strategy_guided, sg_slots, quota)
        selected.extend(selected_sg)

        _fill_oe_slots(selected, open_exploration, strategy_guided, oe_slots, used_ids)

    _pad_free_placeholders(selected, n)

    return [
        _build_result_entry(i, nid, nd, nodes, wm)
        for i, (nid, nd) in enumerate(selected)
    ]


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

_LEGAL_BOTTLENECKS = {
    "mte2_stall", "mte3_stall", "tiling_imbalance",
    "scalar_loading", "scalar_compute",
    "compute_bound", "near_optimal",
    "no_overlap", "partial_overlap",
    "undersize_transfer", "bus_contention", "icache_miss",
}


def _check_inv1_parent_refs(nodes: dict) -> list[str]:
    """Invariant 1: parent_id 必须指向存在的节点。"""
    errors = []
    for nid, nd in nodes.items():
        pid = nd.get("parent_id")
        if pid is not None and pid not in nodes:
            errors.append(
                f"Invariant 1 FAIL: node '{nid}' has parent_id='{pid}' "
                f"which does not exist in nodes"
            )
    return errors


def _check_inv2_best_score(wm: dict) -> list[str]:
    """Invariant 2: best_score >= 1.0。"""
    best_score = wm.get("best_score", 1.0)
    if best_score is None or best_score < 1.0:
        return [f"Invariant 2 FAIL: best_score={best_score} is less than 1.0"]
    return []


def _check_inv3_legal_status(nodes: dict) -> list[str]:
    """Invariant 3: status 取值合法。"""
    legal_statuses = {"open", "in_progress", "passed", "failed", "completed"}
    errors = []
    for nid, nd in nodes.items():
        status = nd.get("status")
        if status not in legal_statuses:
            errors.append(
                f"Invariant 3 FAIL: node '{nid}' has illegal status='{status}'"
            )
    return errors


def _has_open_descendant(nodes: dict, nid: str, all_open_ids: set) -> bool:
    """BFS 检查节点是否存在 open 状态的后代。"""
    nd = nodes.get(nid, {})
    queue = list(nd.get("children", []))
    visited: set[str] = set()
    while queue:
        cid = queue.pop(0)
        if cid in visited:
            continue
        visited.add(cid)
        if cid in all_open_ids:
            return True
        cnode = nodes.get(cid, {})
        queue.extend(cnode.get("children", []))
    return False


def _check_inv4_continuation(nodes: dict) -> list[str]:
    """Invariant 4: passed 节点必须有 ≥1 个 open 后代（延续性）。"""
    errors = []
    all_open_ids = {nid for nid, nd in nodes.items() if nd.get("status") == "open"}
    for nid, nd in nodes.items():
        if nd.get("status") != "passed":
            continue
        children_ids = nd.get("children", [])
        has_open_child = any(cid in all_open_ids for cid in children_ids)
        if children_ids and not has_open_child:
            if not _has_open_descendant(nodes, nid, all_open_ids):
                errors.append(
                    f"Invariant 4 FAIL: passed node '{nid}' has no open "
                    f"descendant (continuation invariant violated)"
                )
    return errors


def _check_inv5_stagnation_counters(wm: dict) -> list[str]:
    """Invariant 5: stagnation 计数器是非负整数。"""
    errors = []
    for key in ("stagnation_count", "stagnation_count_vs_base"):
        val = wm.get(key)
        if not isinstance(val, int) or val < 0:
            errors.append(
                f"Invariant 5 FAIL: '{key}'={val!r} is not a non-negative integer"
            )
    return errors


def _check_inv6_profiling_evidence(nodes: dict) -> list[str]:
    """Invariant 6: profiling_evidence 字段结构合法。"""
    errors = []
    for nid, nd in nodes.items():
        pe = nd.get("profiling_evidence")
        if pe is None:
            continue
        if not isinstance(pe, dict):
            errors.append(
                f"Invariant 6 FAIL: node '{nid}' has profiling_evidence "
                f"that is not a dict: {type(pe).__name__}"
            )
            continue
        required_keys = {"bottleneck_type", "suggested_strategies"}
        missing = required_keys - set(pe.keys())
        if missing:
            errors.append(
                f"Invariant 6 FAIL: node '{nid}' profiling_evidence "
                f"missing required keys: {missing}"
            )
        bt = pe.get("bottleneck_type")
        if bt and bt not in _LEGAL_BOTTLENECKS:
            errors.append(
                f"Invariant 6 FAIL: node '{nid}' profiling_evidence "
                f"has illegal bottleneck_type='{bt}'"
            )
    return errors


def _check_inv7_optimization_type(nodes: dict) -> list[str]:
    """Invariant 7: optimization_type 取值合法（若存在）。"""
    legal_opt_types = {"bandwidth", "tiling", "algorithm",
                       "register_opt", "vf_fusion", "instruction_sched"}
    errors = []
    for nid, nd in nodes.items():
        ot = nd.get("optimization_type")
        if ot is not None and ot not in legal_opt_types:
            errors.append(
                f"Invariant 7 FAIL: node '{nid}' has illegal "
                f"optimization_type='{ot}'"
            )
    return errors


def _check_inv8_session(wm: dict) -> list[str]:
    """Invariant 8: session 身份锚点存在且合法。"""
    sess = wm.get("session")
    if not isinstance(sess, dict):
        return ["Invariant 8 FAIL: session field missing or not a dict"]
    errors = []
    for key in ("session_id", "start_time", "evo_dir", "op_name"):
        if not sess.get(key):
            errors.append(f"Invariant 8 FAIL: session.{key} is empty or missing")
    actual = sess.get("actual_rounds_completed")
    requested = sess.get("requested_rounds")
    if actual is not None and requested is not None and actual > requested:
        errors.append(
            f"Invariant 8 FAIL: actual_rounds_completed({actual}) > "
            f"requested_rounds({requested})"
        )
    return errors


def _check_inv9_baseline_evidence(wm: dict) -> list[str]:
    """Invariant 9: baseline_evidence（root 级）结构合法（若存在）。"""
    be = wm.get("baseline_evidence")
    if be is None:
        return []
    if not isinstance(be, dict):
        return [
            f"Invariant 9 FAIL: baseline_evidence is not a dict: "
            f"{type(be).__name__}"
        ]
    errors = []
    bt = be.get("bottleneck_type")
    if bt is not None and bt not in _LEGAL_BOTTLENECKS:
        errors.append(
            f"Invariant 9 FAIL: baseline_evidence has illegal "
            f"bottleneck_type='{bt}'"
        )
    if "suggested_strategies" not in be:
        errors.append(
            "Invariant 9 FAIL: baseline_evidence missing "
            "'suggested_strategies' field"
        )
    return errors


def validate(wm: dict) -> list[str]:
    """Validate world model invariants. Returns list of error strings (empty = valid).

    Invariants checked:
      1. All parent_id values point to existing nodes
      2. best_score >= 1.0
      3. status values are legal
      4. All passed nodes have at least 1 open child (continuation invariant)
      5. stagnation_count and stagnation_count_vs_base are non-negative integers
      6. profiling_evidence field structure is valid
      7. optimization_type values are legal (if present)
      8. session identity anchor present and valid
      9. baseline_evidence (root-level) structure is valid (if present)
    """
    nodes: dict = wm.get("decision_tree", {}).get("nodes", {})
    errors: list[str] = []
    errors.extend(_check_inv1_parent_refs(nodes))
    errors.extend(_check_inv2_best_score(wm))
    errors.extend(_check_inv3_legal_status(nodes))
    errors.extend(_check_inv4_continuation(nodes))
    errors.extend(_check_inv5_stagnation_counters(wm))
    errors.extend(_check_inv6_profiling_evidence(nodes))
    errors.extend(_check_inv7_optimization_type(nodes))
    errors.extend(_check_inv8_session(wm))
    errors.extend(_check_inv9_baseline_evidence(wm))
    return errors


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def _find_best_path(wm: dict) -> str:
    """Trace the best path from root to the highest-scoring passed node."""
    nodes: dict = wm.get("decision_tree", {}).get("nodes", {})

    # Find node with best score
    best_nid = None
    best_score = -1.0
    for nid, nd in nodes.items():
        sc = nd.get("score")
        if sc is not None and sc > best_score:
            best_score = sc
            best_nid = nid

    if best_nid is None:
        return "root(baseline)"

    # Trace path from best node up to root
    path = []
    current = best_nid
    while current is not None:
        nd = nodes.get(current, {})
        sc = nd.get("score")
        label = f"{current}({sc:.2f}x)" if sc is not None else current
        path.append(label)
        current = nd.get("parent_id")

    path.reverse()
    return " → ".join(path)


def _count_unexplored_root_branches(wm: dict) -> str:
    """Summarize root's direct children that are still open."""
    nodes: dict = wm.get("decision_tree", {}).get("nodes", {})
    root = nodes.get("root", {})
    open_children = []
    for cid in root.get("children", []):
        cnode = nodes.get(cid, {})
        if cnode.get("status") == "open":
            strats = "+".join(cnode.get("strategy_combination", []) or ["free"])
            open_children.append(f"{cid}({strats})")
    return ", ".join(open_children) if open_children else "(none)"


def summary(wm: dict, max_chars: int = 1200) -> str:
    """
    Generate a compact world model summary for sub-agent prompt injection.

    Output is plain text, ≤ max_chars characters.
    """
    nodes: dict = wm.get("decision_tree", {}).get("nodes", {})
    open_count = sum(1 for nd in nodes.values() if nd.get("status") == "open")
    passed_count = sum(1 for nd in nodes.values() if nd.get("status") == "passed")
    failed_count = sum(1 for nd in nodes.values() if nd.get("status") == "failed")

    stagnation_count = wm.get("stagnation_count", 0)
    stagnation_window = wm.get("stagnation_window", 2)
    best_score = wm.get("best_score", 1.0)
    kernel_summary = wm.get("kernel_summary", "N/A")
    open_questions = wm.get("open_questions", [])

    active_path = _find_best_path(wm)
    unexplored_roots = _count_unexplored_root_branches(wm)

    lines = [
        "[World Model Summary]",
        f"Best: {best_score}x | Stagnation: {stagnation_count}/{stagnation_window}",
        f"Open nodes: {open_count} | Passed: {passed_count} | Failed: {failed_count}",
        f"Active path: {active_path}",
        f"Unexplored root branches: {unexplored_roots}",
    ]

    if open_questions:
        lines.append("Key findings:")
        for q in open_questions[:5]:
            lines.append(f"  - {q}")

    text = "\n".join(lines)

    # Truncate to max_chars if needed
    if len(text) > max_chars:
        text = text[:max_chars - 3] + "..."

    return text


# ---------------------------------------------------------------------------
# refine — deterministic world model update after a round
# ---------------------------------------------------------------------------


def _safe_float(val, default: float = 0.0) -> float:
    """Coerce a value to float, returning default on failure."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


# Dynamic thresholds based on measurement quality
_THRESHOLDS = {
    "good": {"improve": 1.05, "stagnation": 1.02},
    "acceptable": {"improve": 1.08, "stagnation": 1.03},
    "noisy": {"improve": 1.15, "stagnation": 1.05},
}


def _generate_child_id(parent_id: str, nodes: dict, suffix: str = "") -> str:
    """Generate a unique child node ID."""
    base = f"{parent_id}_{suffix}" if suffix else f"{parent_id}_c"
    if base not in nodes:
        return base
    for i in range(1, 100):
        candidate = f"{base}{i}"
        if candidate not in nodes:
            return candidate
    return f"{parent_id}_{suffix}_{id(nodes) % 10000}"


@dataclass
class ChildNodeSpec:
    """_make_child_node 的节点规格封装。"""
    child_id: str
    mode: str
    description: str
    strategy_combination: list = None
    optimization_type: str = None
    difficulty_delta: int = 0


def _make_child_node(parent: dict, spec: ChildNodeSpec) -> dict:
    """Create a child node dict with inherited fields."""
    difficulty = min(4, max(1, parent.get("difficulty", 3) + spec.difficulty_delta))
    optimization_type = spec.optimization_type
    if optimization_type is None:
        optimization_type = parent.get("optimization_type", "algorithm")
    return {
        "id": spec.child_id,
        "mode": spec.mode,
        "strategy_combination": spec.strategy_combination or [],
        "description": spec.description,
        "optimization_type": optimization_type,
        "difficulty": difficulty,
        "depth": parent.get("depth", 1) + 1,
        "parent_id": parent["id"],
        "status": "open",
        "score": None,
        "solution_ref": None,
        "children": [],
        "failure_type": None,
        "failure_reason": None,
        "retry_count": 0,
        "profiling_insight": None,
        "profiling_evidence": None,
    }




_QUALITY_RANK = {"good": 0, "acceptable": 1, "noisy": 2}


@dataclass
class _RefineContext:
    """refine 单轮处理过程中的共享状态。"""
    nodes: dict
    round_num: int
    results_dir: str
    summary_lines: list
    pending_diagnosis: list
    round_passed: int = 0
    round_failed: int = 0
    round_best_speedup: float = 0.0
    worst_quality: str = "good"


def _reset_stale_in_progress(nodes: dict, current_node_ids: set) -> list:
    """重置被上轮 SELECT 标记 in_progress 但本轮未执行的节点为 open。"""
    stale_reset = []
    for nid, nd in nodes.items():
        if nd.get("status") == "in_progress" and nid not in current_node_ids:
            nd["status"] = "open"
            stale_reset.append(nid)
    return stale_reset


def _infer_bottleneck(pipeline: dict, bottleneck: str) -> str:
    """bottleneck 为 unknown 时按 pipeline 占比推断。"""
    if bottleneck != "unknown" or not pipeline:
        return bottleneck
    mte2 = _safe_float(pipeline.get("aiv_mte2_ratio")) or _safe_float(pipeline.get("mte2_pct"))
    vec = _safe_float(pipeline.get("aiv_vec_ratio")) or _safe_float(pipeline.get("vec_pct"))
    scalar = _safe_float(pipeline.get("aiv_scalar_ratio")) or _safe_float(pipeline.get("scalar_pct"))
    # Values may be ratios (0-1) or percentages (0-100); normalize
    if max(mte2, vec, scalar) <= 1.0 and max(mte2, vec, scalar) > 0:
        mte2 *= 100
        vec *= 100
        scalar *= 100
    if mte2 > 50:
        return "memory_bound"
    if vec > 60:
        return "compute_bound"
    if scalar > 30:
        return "scalar_bound"
    return "balanced"


def _build_profiling_one_liner(pipeline: dict, bottleneck: str, speedup: float) -> str:
    if pipeline:
        parts = [f"{k}={v}" for k, v in sorted(pipeline.items()) if v]
        if parts:
            return " | ".join(parts[:4]) + f" | bottleneck={bottleneck}"
    return f"speedup={speedup:.2f}x | bottleneck={bottleneck}"


def _fill_multi_shape_fields(node: dict, eval_result: dict, ms_active: bool):
    """填充多形态评估字段（gating/aggregate/shape_results/parent_eligible）。"""
    if not ms_active:
        # Single-shape (legacy) — derive defaults for forwards compat
        node.setdefault("parent_eligible", True)
        node.setdefault("target_shape_regression", False)
        return
    ms_gating = eval_result.get("gating")
    ms_aggregate = eval_result.get("aggregate")
    ms_shape_results = eval_result.get("shape_results")
    node["gating"] = ms_gating
    node["aggregate"] = ms_aggregate
    node["shape_results"] = ms_shape_results
    target_regression = bool(
        (ms_aggregate or {}).get("any_target_regression")
    )
    node["target_shape_regression"] = target_regression
    # parent_eligible: only True when no target shape regressed.
    # generalization_regression at this stage does not automatically
    # forbid using the node as parent (the supervisor decides
    # downstream); target_regression always forbids.
    node["parent_eligible"] = not target_regression
    if target_regression:
        # Override status display reason — node still ran, but
        # parent_eligible=false will keep SELECT from picking it.
        failed_shapes = [
            r.get("name") for r in (ms_shape_results or {}).get("target", [])
            if isinstance(r.get("speedup"), (int, float)) and r["speedup"] < 1.0
        ]
        node["failure_reason"] = (
            f"target_shape_regression: shapes {failed_shapes} speedup < 1.0x; "
            f"suggest {P_SHAPE_SPEC}"
        )


def _attach_csv_evidence(node: dict, pipeline: dict, bottleneck: str):
    """A1: CSV-level profiling_evidence 合成 + Stage 1 extract_facts。

    Synthesize profiling_analysis from pipeline ratios and run it
    through extract_profiling_evidence so the hard mapping table
    (BOTTLENECK_STRATEGY_MAP) is consulted every round, not only
    when deep-profiling is conditionally triggered.
    """
    try:
        from profiling_evidence import (
            synthesize_analysis_from_pipeline, extract_profiling_evidence,
            extract_facts,
        )
    except ImportError:
        _pr = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if _pr not in sys.path:
            sys.path.insert(0, _pr)
        from profiling_evidence import (
            synthesize_analysis_from_pipeline, extract_profiling_evidence,
            extract_facts,
        )

    pa = synthesize_analysis_from_pipeline(pipeline, bottleneck_hint=bottleneck)
    csv_evidence = extract_profiling_evidence({"profiling_analysis": pa}) if pa else None
    # Don't overwrite deep-profiling evidence (has pattern_type etc.)
    if csv_evidence and not node.get("profiling_evidence"):
        csv_evidence["_source"] = "csv_synth"
        node["profiling_evidence"] = csv_evidence
        # Also mirror into profiling_insight.recommended_strategies so
        # prompts that read that legacy field see the mapping too.
        sug = csv_evidence.get("suggested_strategies") or []
        node["profiling_insight"]["recommended_strategies"] = sug[:5]

    # --- v3.2 Stage 1: extract_facts (pure facts, no bottleneck conclusion) ---
    # 与 csv_evidence 并行，独立写入 node["facts"]。
    # 上游 LLM 在 refine prompt 中读取 node["facts"]，输出 diagnosis (Stage 2)。
    if not node.get("facts"):
        facts_input = {"profiling_analysis": pa} if pa else {}
        if pipeline:
            facts_input["pipeline"] = pipeline
        facts = extract_facts(facts_input)
        if facts:
            node["facts"] = facts


@dataclass
class _PassedEvalInfo:
    """PASSED 变体的评估信息封装。"""
    speedup: float
    mq: str
    one_liner: str
    bottleneck: str


def _gen_sg_children(node: dict, node_id: str, nodes: dict,
                     speedup: float, substantial: bool):
    """strategy_guided 模式子节点：显著提升生 2 个深度探索，否则 1 个延续探索。"""
    if substantial:
        for i in range(2):
            cid = _generate_child_id(node_id, nodes, f"s{i+1}")
            sc = list(node.get("strategy_combination", []))
            desc = f"深度探索: 在 {node_id}({speedup:.2f}x) 基础上继续优化"
            child = _make_child_node(node, ChildNodeSpec(
                cid, "strategy_guided", desc, sc))
            nodes[cid] = child
            node.setdefault("children", []).append(cid)
    else:
        cid = _generate_child_id(node_id, nodes, "cont")
        desc = f"延续探索: {node_id}({speedup:.2f}x) 无显著提升，尝试变化"
        child = _make_child_node(node, ChildNodeSpec(
            cid, "strategy_guided", desc))
        nodes[cid] = child
        node.setdefault("children", []).append(cid)


def _gen_oe_child(node: dict, node_id: str, nodes: dict,
                  speedup: float, substantial: bool):
    """open_exploration 模式子节点：固定生 1 个。"""
    cid = _generate_child_id(node_id, nodes, "x1")
    if substantial:
        desc = f"在开放探索({node_id}, {speedup:.2f}x)基础上继续自主推理"
    else:
        desc = f"延续探索: {node_id}({speedup:.2f}x) 开放探索方向"
    child = _make_child_node(node, ChildNodeSpec(
        cid, "open_exploration", desc))
    nodes[cid] = child
    node.setdefault("children", []).append(cid)


def _gen_pd_continue_child(node: dict, node_id: str, nodes: dict,
                           speedup: float, substantial: bool):
    """profiling_driven 模式子节点：仅显著提升时生 1 个。"""
    if not substantial:
        return
    cid = _generate_child_id(node_id, nodes, "x1")
    desc = f"Profiling驱动({node_id}, {speedup:.2f}x)已解决瓶颈，继续探索"
    child = _make_child_node(node, ChildNodeSpec(
        cid, "open_exploration", desc))
    nodes[cid] = child
    node.setdefault("children", []).append(cid)


def _gen_profiling_driven_child(node: dict, node_id: str, nodes: dict,
                                bottleneck: str, one_liner: str):
    """profiling_driven 补充子节点：瓶颈明确且尚无 pd 子节点时生成。"""
    if bottleneck in ("balanced", "unknown"):
        return
    existing_pd = [
        c for c in node.get("children", [])
        if nodes.get(c, {}).get("mode") == "profiling_driven"
    ]
    if existing_pd:
        return
    pd_id = _generate_child_id(node_id, nodes, "pd1")
    pd_desc = f"[Profiling驱动] 针对 {bottleneck}: {one_liner}"
    opt_type = "bandwidth" if "memory" in bottleneck else (
        "tiling" if "compute" in bottleneck else "algorithm"
    )
    pd_child = _make_child_node(node, ChildNodeSpec(
        pd_id, "profiling_driven", pd_desc,
        optimization_type=opt_type,
    ))
    nodes[pd_id] = pd_child
    node.setdefault("children", []).append(pd_id)


def _gen_children_for_passed(node: dict, node_id: str, nodes: dict,
                             info: _PassedEvalInfo):

    """为 PASSED 节点按模式生成子节点（含 profiling_driven 补充）。"""
    speedup = info.speedup
    bottleneck = info.bottleneck
    one_liner = info.one_liner
    # Bottleneck shift detection
    parent_id = node.get("parent_id", "root")
    parent_node = nodes.get(parent_id, {})
    parent_pi = parent_node.get("profiling_insight")
    if parent_pi and isinstance(parent_pi, dict):
        parent_bn = parent_pi.get("bottleneck")
        if parent_bn and parent_bn != bottleneck:
            node["bottleneck_shift"] = f"{parent_bn} → {bottleneck}"

    # Dynamic threshold
    thresholds = _THRESHOLDS.get(info.mq, _THRESHOLDS["good"])
    improve_threshold = thresholds["improve"]
    parent_score = parent_node.get("score") or 1.0
    substantial = speedup > parent_score * improve_threshold

    # Generate child nodes
    mode = node.get("mode", "strategy_guided")
    if mode == "strategy_guided":
        _gen_sg_children(node, node_id, nodes, speedup, substantial)
    elif mode == "open_exploration":
        _gen_oe_child(node, node_id, nodes, speedup, substantial)
    elif mode == "profiling_driven":
        _gen_pd_continue_child(node, node_id, nodes, speedup, substantial)

    _gen_profiling_driven_child(node, node_id, nodes, bottleneck, one_liner)


def _merge_evidence_into_children(node: dict, node_id: str, nodes: dict):
    """A1 (cont'd): merge evidence into this round's new children.

    v3.3 sliding-window + LLM-hint: each strategy_guided/profiling_driven
    child gets parent ancestry kept (minus offset slot) + new suggested
    slots, capped at K=5; LLM next_round_hint.prefer wins new slots,
    avoid + ancestry failures join anti. open_exploration children keep
    sc=[] (truly free) — only carry ancestry_avoid forward.
    """
    if not node.get("profiling_evidence"):
        return
    try:
        from profiling_evidence import MergeConfig, merge_strategies_with_evidence
    except ImportError:
        _pr = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if _pr not in sys.path:
            sys.path.insert(0, _pr)
        from profiling_evidence import MergeConfig, merge_strategies_with_evidence

    ev = node["profiling_evidence"]
    hint = (node.get("diagnosis") or {}).get("next_round_hint") or {}
    ancestry_failed = _collect_ancestry_failed_strategies(node_id, nodes)
    sib_idx = 0
    for cid in node.get("children", []):
        ch = nodes.get(cid)
        if not ch or ch.get("status") != "open":
            continue
        # Skip children created before this round (have scores or were
        # touched by prior rounds) — only touch nodes with empty
        # solution_ref and no profiling data of their own.
        if ch.get("score") is not None or ch.get("profiling_insight"):
            continue
        if ch.get("mode") == "open_exploration":
            # Keep sc=[]; only forward avoid set so its own children
            # never re-derive a known dead-end. No strategy injection.
            avoid = hint.get("avoid", []) or []
            if avoid:
                ch.setdefault("ancestry_avoid", [])
                ch["ancestry_avoid"] = sorted(set(ch["ancestry_avoid"]) | set(avoid))
            continue
        old = ch.get("strategy_combination", [])
        ch["strategy_combination"] = merge_strategies_with_evidence(
            old, ev,
            ancestry_failed=ancestry_failed,
            parent_hint=hint,
            config=MergeConfig(offset=sib_idx),
        )
        sib_idx += 1


def _inject_shape_spec_into_children(node: dict, nodes: dict):
    """Multi-shape P-ShapeSpec-01 auto-injection.

    When this node's target shapes regressed, push children toward
    shape-specialized variants by injecting P-ShapeSpec-01. open_exploration
    children keep sc=[]; they get the constraint via description only.
    """
    if not node.get("target_shape_regression"):
        return
    for cid in node.get("children", []):
        ch = nodes.get(cid)
        if not ch or ch.get("status") != "open":
            continue
        if ch.get("score") is not None:
            continue
        if ch.get("mode") != "open_exploration":
            ch["strategy_combination"] = _inject_shape_spec_into_strategies(
                ch.get("strategy_combination", [])
            )
        base_desc = ch.get("description", "")
        note = "考虑用 shape-specialized branch 隔离改动，保护其它 target shape 不退化"
        if note not in base_desc:
            ch["description"] = (base_desc + " | " if base_desc else "") + note


@dataclass
class _VariantEvalInput:
    """单个变体评估输入封装。"""
    node: dict
    node_id: str
    p_idx: int
    eval_result: dict
    speedup: float


def _process_passed_variant(vei: _VariantEvalInput, ctx: _RefineContext):
    """处理 PASSED 变体：填字段、profiling_insight、evidence、子节点。"""
    node, node_id, p_idx, eval_result, speedup = (
        vei.node, vei.node_id, vei.p_idx, vei.eval_result, vei.speedup)
    node["status"] = "passed"
    node["score"] = round(speedup, 4)
    node["solution_ref"] = f"round_{ctx.round_num}/parallel_{p_idx}"
    ctx.round_passed += 1
    if speedup > ctx.round_best_speedup:
        ctx.round_best_speedup = speedup

    # Multi-shape fields (only when eval came from multi-shape pipeline)
    ms_active = _is_multi_shape_eval(eval_result)
    _fill_multi_shape_fields(node, eval_result, ms_active)

    # Extract profiling_insight from evaluation_results.json
    # ops-evo 写的是 {"evolved": {...}} 嵌套结构；lingxi-evo/lingxi-partial
    # 写的是扁平结构（直接把 pipeline / bottleneck 放顶层）。兼容两种。
    evolved = eval_result.get("evolved") or eval_result
    pipeline = evolved.get("pipeline", {})
    bottleneck = _infer_bottleneck(pipeline, evolved.get("bottleneck", "unknown"))

    # Build profiling_insight
    one_liner = _build_profiling_one_liner(pipeline, bottleneck, speedup)
    node["profiling_insight"] = {
        "bottleneck": bottleneck,
        "pipeline": pipeline,
        "recommended_strategies": [],
        "profiling_one_liner": one_liner,
    }

    comp = eval_result.get("comparison", {})
    mq = comp.get("measurement_quality", "good")

    _attach_csv_evidence(node, pipeline, bottleneck)
    _gen_children_for_passed(node, node_id, ctx.nodes,
                             _PassedEvalInfo(speedup, mq, one_liner, bottleneck))
    _merge_evidence_into_children(node, node_id, ctx.nodes)
    _inject_shape_spec_into_children(node, ctx.nodes)

    ctx.summary_lines.append(
        f"  p{p_idx} [{node_id}]: PASS {speedup:.2f}x "
        f"({bottleneck}, quality={mq})"
    )


def _process_failed_variant(node: dict, node_id: str, p_idx: int,
                            eval_result: dict, ctx: _RefineContext):
    """处理 FAILED 变体：标失败、记 pending_diagnosis。"""
    comp = eval_result.get("comparison", {})
    compilation_ok = comp.get("compilation_success", False)
    precision_ok = comp.get("precision_passed", False)
    node["status"] = "failed"
    error = comp.get("precision_message", eval_result.get("error", "unknown"))
    node["failure_reason"] = str(error)[:200]
    ctx.round_failed += 1

    ctx.pending_diagnosis.append({
        "node_id": node_id,
        "parallel_index": p_idx,
        "compilation_success": compilation_ok,
        "precision_passed": precision_ok,
        "error": str(error)[:200],
        "implementation_note_path": os.path.join(
            ctx.results_dir, f"parallel_{p_idx}", "implementation_note.txt"
        ),
    })

    ctx.summary_lines.append(
        f"  p{p_idx} [{node_id}]: FAIL "
        f"(compile={'OK' if compilation_ok else 'FAIL'}, "
        f"precision={'OK' if precision_ok else 'FAIL'})"
    )


def _process_one_variant(node_id: str, p_idx_str: str, ctx: _RefineContext):
    """处理单个变体：SKIP / no-results / PASS / FAIL 分发。"""
    p_idx = int(p_idx_str)
    node = ctx.nodes.get(node_id)
    if node is None:
        ctx.summary_lines.append(f"  p{p_idx} [{node_id}]: SKIP (node not found)")
        ctx.round_failed += 1
        return

    # Read evaluation_results.json
    eval_path = os.path.join(ctx.results_dir, f"parallel_{p_idx}", "evaluation_results.json")
    if not os.path.isfile(eval_path):
        node["status"] = "failed"
        node["failure_reason"] = "evaluation_results.json not found"
        ctx.summary_lines.append(f"  p{p_idx} [{node_id}]: FAIL (no results)")
        ctx.round_failed += 1
        ctx.pending_diagnosis.append({
            "node_id": node_id, "parallel_index": p_idx,
            "reason": "evaluation_results.json not found",
        })
        return

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_result = json.load(f)

    comp = eval_result.get("comparison", {})
    compilation_ok = comp.get("compilation_success", False)
    precision_ok = comp.get("precision_passed", False)
    speedup = comp.get("speedup", 0.0) or 0.0
    mq = comp.get("measurement_quality", "good")
    if _QUALITY_RANK.get(mq, 0) > _QUALITY_RANK.get(ctx.worst_quality, 0):
        ctx.worst_quality = mq

    # ── Multi-shape pipeline: prefer new fields when present ──
    ms_active = _is_multi_shape_eval(eval_result)
    ms_aggregate = eval_result.get("aggregate") if ms_active else None
    if ms_active and isinstance(ms_aggregate, dict):
        # Use min(target speedups) as the node's effective speedup
        ms_min = ms_aggregate.get("target_min_speedup")
        if isinstance(ms_min, (int, float)) and ms_min > 0:
            speedup = ms_min

    if compilation_ok and precision_ok and speedup > 0:
        _process_passed_variant(
            _VariantEvalInput(node, node_id, p_idx, eval_result, speedup), ctx)
    else:
        _process_failed_variant(node, node_id, p_idx, eval_result, ctx)


@dataclass
class _StagnationInput:
    """_update_stagnation 的输入封装。"""
    round_best_speedup: float
    best_score_before: float
    worst_quality: str


def _update_stagnation(wm: dict, nodes: dict, parallel_map: dict,
                       si: _StagnationInput):
    """停滞计数：全局 stagnation_count + vs_base（是否有变体跑赢父节点）。"""
    round_best_speedup = si.round_best_speedup
    best_score_before = si.best_score_before
    stag_threshold = _THRESHOLDS.get(si.worst_quality, _THRESHOLDS["good"])["stagnation"]
    if round_best_speedup <= best_score_before * stag_threshold:
        wm["stagnation_count"] = wm.get("stagnation_count", 0) + 1
    else:
        wm["stagnation_count"] = 0

    # Stagnation vs base (did any variant beat its parent?)
    any_beat_parent = False
    for node_id in parallel_map.values():
        node = nodes.get(node_id, {})
        if node.get("status") == "passed" and node.get("score"):
            parent_id = node.get("parent_id", "root")
            parent_score = nodes.get(parent_id, {}).get("score") or 1.0
            if node["score"] > parent_score:
                any_beat_parent = True
                break
    if any_beat_parent:
        wm["stagnation_count_vs_base"] = 0
    else:
        wm["stagnation_count_vs_base"] = wm.get("stagnation_count_vs_base", 0) + 1


def refine(
    wm: dict,
    round_num: int,
    results_dir: str,
    parallel_map: dict,
    task_type: str = "vector",
) -> dict:
    """Deterministic world model update after a round of evolution.

    Handles: score update, profiling_insight extraction, bottleneck shift
    detection, child node generation, stagnation counting.

    Does NOT handle (requires LLM): failure diagnosis (impl_error vs
    strategy_infeasible), open_questions update (Analyze).

    Args:
        wm: world model dict (modified in-place and returned)
        round_num: current round number
        results_dir: directory containing parallel_0/, parallel_1/, etc.
        parallel_map: {"0": "n1", "1": "n2", "2": "x0"} parallel_index→node_id
        task_type: vector/cube/cv-mix/unknown

    Returns:
        dict with keys: round_summary (str), pending_diagnosis (list),
        best_score_before (float), best_score_after (float)
    """
    nodes = wm.setdefault("decision_tree", {}).setdefault("nodes", {})

    # Stale in_progress cleanup: nodes marked in_progress by a previous SELECT
    # but not in the current parallel_map are stale (their sub-agent failed
    # silently). Reset to "open" so they can be re-selected in future rounds.
    stale_reset = _reset_stale_in_progress(nodes, set(parallel_map.values()))

    best_score_before = wm.get("best_score", 1.0) or 1.0
    ctx = _RefineContext(
        nodes=nodes,
        round_num=round_num,
        results_dir=results_dir,
        summary_lines=[],
        pending_diagnosis=[],
    )
    round_total = len(parallel_map)

    for p_idx_str, node_id in parallel_map.items():
        _process_one_variant(node_id, p_idx_str, ctx)

    # --- Top-level updates ---
    best_score_after = max(best_score_before, ctx.round_best_speedup)
    wm["best_score"] = round(best_score_after, 4)

    # Update session anchor: actual_rounds_completed
    sess = wm.setdefault("session", {})
    sess["actual_rounds_completed"] = max(
        sess.get("actual_rounds_completed", 0), round_num
    )

    _update_stagnation(wm, nodes, parallel_map, _StagnationInput(
        ctx.round_best_speedup, best_score_before, ctx.worst_quality))

    # Build round summary
    stale_line = f"  [cleanup] Reset stale in_progress to open: {stale_reset}\n" if stale_reset else ""
    round_summary = (
        stale_line
        + f"Round {round_num}: {ctx.round_passed}/{round_total} passed, "
        f"{ctx.round_failed}/{round_total} failed, "
        f"best={ctx.round_best_speedup:.2f}x, "
        f"global_best={best_score_after:.2f}x, "
        f"stagnation={wm.get('stagnation_count', 0)}/{wm.get('stagnation_count_vs_base', 0)}\n"
        + "\n".join(ctx.summary_lines)
    )

    return {
        "round_summary": round_summary,
        "pending_diagnosis": ctx.pending_diagnosis,
        "best_score_before": best_score_before,
        "best_score_after": best_score_after,
        "worst_quality": ctx.worst_quality,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def cmd_select(args: argparse.Namespace) -> None:
    with open(args.path, "r", encoding="utf-8") as f:
        wm = json.load(f)

    # Drift breaker — if state.drift_status == replan_required, force
    # at least ⌈n/2⌉ open_exploration slots so the search escapes local minima.
    force_oe = None
    drift = _read_state_field_safe(args.path, "drift_status")
    if drift == "replan_required":
        force_oe = max(1, -(-args.n // 2))  # ceil(n/2)
        LOGGER.warning(
            "  [DRIFT] state.drift_status=replan_required → force "
            "open_exploration_min=%d (of n=%d)",
            force_oe, args.n,
        )

    result = select_nodes(wm, args.n, force_open_exploration_min=force_oe)
    DATA_LOGGER.info("%s", json.dumps(result, ensure_ascii=False, indent=2))
    # Re-infer state from filesystem (replaces older single-field write helper)
    _maybe_infer_state(args.path)
    # Drift signal is one-shot; auto-clear after SELECT consumed it
    if drift == "replan_required":
        _maybe_clear_drift(args.path)


def cmd_validate(args: argparse.Namespace) -> int:
    with open(args.path, "r", encoding="utf-8") as f:
        wm = json.load(f)
    errors = validate(wm)
    if errors:
        DATA_LOGGER.info("Validation FAILED:")
        for e in errors:
            DATA_LOGGER.info("  %s", e)
        return 1
    DATA_LOGGER.info("Validation PASSED: all invariants satisfied.")
    return 0


def cmd_summary(args: argparse.Namespace) -> None:
    with open(args.path, "r", encoding="utf-8") as f:
        wm = json.load(f)
    DATA_LOGGER.info("%s", summary(wm, max_chars=args.max_chars))


def _run_deep_profiling_script(args) -> tuple:
    """Step 1+2: 运行 run_deep_profiling.py 并读取 profiling_evidence。

    Returns (evidence, error_msg)；成功时 error_msg 为 None。
    """
    script_path = os.path.join(
        os.path.dirname(__file__), "..", "..",
        ".claude", "skills", "ascendc-profiling-analysis", "scripts",
        "run_deep_profiling.py",
    )
    script_path = os.path.abspath(script_path)
    result_path = os.path.join(args.work_dir, "deep_profiling_result.json")

    cmd = [
        sys.executable, script_path,
        "--work-dir", args.work_dir,
        "--op-name", args.op_name,
        "--output", result_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        return None, f"deep-profiling: run_deep_profiling.py failed:\n{proc.stderr}"

    with open(result_path, "r", encoding="utf-8") as f:
        prof_result = json.load(f)

    evidence = prof_result.get("profiling_evidence")
    if not evidence:
        return None, "deep-profiling: no profiling_evidence produced"
    return evidence, None


def _merge_evidence_into_open_children(node: dict, nodes: dict, evidence: dict) -> list:
    """Step 4: 把 evidence 策略合并进 open 子节点（open_exploration 除外）。

    Returns 被更新的子节点 id 列表。
    """
    try:
        from profiling_evidence import MergeConfig, merge_strategies_with_evidence
    except ImportError:
        _proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        sys.path.insert(0, _proj_root)
        from profiling_evidence import MergeConfig, merge_strategies_with_evidence

    updated_children = []
    sib_idx = 0
    for cid in node.get("children", []):
        child = nodes.get(cid)
        if not (child and child.get("status") == "open"):
            continue
        # open_exploration children stay free (sc=[]); others get capped merge
        if child.get("mode") == "open_exploration":
            continue
        old_strats = child.get("strategy_combination", [])
        child["strategy_combination"] = merge_strategies_with_evidence(
            old_strats, evidence, config=MergeConfig(offset=sib_idx)
        )
        updated_children.append(cid)
        sib_idx += 1
    return updated_children


def cmd_deep_profiling(args: argparse.Namespace) -> int:
    """Run deep profiling on a node and write results to world_model.json."""

    evidence, error = _run_deep_profiling_script(args)
    if error:
        LOGGER.error("%s", error)
        return 1

    # Step 3: Write evidence into world_model.json node
    with open(args.wm_path, "r", encoding="utf-8") as f:
        wm = json.load(f)

    nodes = wm.get("decision_tree", {}).get("nodes", {})
    node = nodes.get(args.node_id)
    if not node:
        LOGGER.error("deep-profiling: node '%s' not found", args.node_id)
        return 1

    node["profiling_evidence"] = evidence

    # Step 4: Optionally merge strategies into open children
    if args.merge_children:
        updated_children = _merge_evidence_into_open_children(node, nodes, evidence)
        if updated_children:
            LOGGER.info("deep-profiling: updated strategies for children: %s",
                        updated_children)

    # Step 5: Write back
    with open(args.wm_path, "w", encoding="utf-8") as f:
        json.dump(wm, f, ensure_ascii=False, indent=2)

    LOGGER.info("deep-profiling: wrote profiling_evidence to node '%s'", args.node_id)
    LOGGER.info("  bottleneck_type: %s", evidence.get('bottleneck_type'))
    LOGGER.info("  suggested_strategies: %s", evidence.get('suggested_strategies'))
    return 0


def cmd_attach_baseline_evidence(args: argparse.Namespace) -> None:
    """Attach root-level baseline_evidence to world_model.json from
    baseline_evaluation.json's pipeline/bottleneck fields.

    Called once per evolution session, right after Phase 3.6 baseline profiling
    produces baseline_evaluation.json. Subsequent SELECT rounds then use
    wm["baseline_evidence"] to penalize nodes misaligned with the baseline
    bottleneck (compute_utility w_baseline_mismatch) and inject the Baseline
    row into partial-agent prompts.

    Gracefully no-ops (writes baseline_evidence=None) when the baseline lacks
    pipeline data — all downstream consumers handle None by falling back.
    """

    try:
        from profiling_evidence import (
            synthesize_analysis_from_pipeline, extract_profiling_evidence,
        )
    except ImportError:
        _proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        sys.path.insert(0, _proj_root)
        from profiling_evidence import (
            synthesize_analysis_from_pipeline, extract_profiling_evidence,
        )

    with open(args.baseline_eval, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    # baseline_evaluation.json written by evaluate_ops_direct.py contains a
    # top-level "baseline" dict with pipeline/bottleneck; some variants nest
    # under "evolved" when baseline is the only run. Try both.
    baseline = eval_data.get("baseline") or eval_data.get("evolved") or {}
    pipeline = baseline.get("pipeline") or {}
    bottleneck_hint = baseline.get("bottleneck")

    with open(args.wm_path, "r", encoding="utf-8") as f:
        wm = json.load(f)

    if not pipeline:
        wm["baseline_evidence"] = None
        with open(args.wm_path, "w", encoding="utf-8") as f:
            json.dump(wm, f, ensure_ascii=False, indent=2)
        LOGGER.warning("attach-baseline-evidence: no pipeline data in baseline; "
                       "wrote baseline_evidence=null")
        return

    profiling_analysis = synthesize_analysis_from_pipeline(pipeline, bottleneck_hint)
    evidence = extract_profiling_evidence({"profiling_analysis": profiling_analysis})

    if not evidence:
        wm["baseline_evidence"] = None
        LOGGER.warning("attach-baseline-evidence: evidence synthesis returned None; "
                       "wrote baseline_evidence=null")
    else:
        wm["baseline_evidence"] = evidence
        LOGGER.info("attach-baseline-evidence: baseline bottleneck_type=%s, "
                    "suggested=%s, anti=%s",
                    evidence.get('bottleneck_type'),
                    evidence.get('suggested_strategies')[:5],
                    evidence.get('anti_strategies'))

    with open(args.wm_path, "w", encoding="utf-8") as f:
        json.dump(wm, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════════
# v3.2 Phase D1: Ledger 三件套（attempt-ledger.md + lineage.jsonl）
# ═══════════════════════════════════════════════════════════════════




































def cmd_refine(args: argparse.Namespace) -> None:
    """Deterministic world model update after a round."""

    with open(args.wm_path, "r", encoding="utf-8") as f:
        wm = json.load(f)

    parallel_map = json.loads(args.parallel_map)

    result = refine(
        wm=wm,
        round_num=args.round,
        results_dir=args.results_dir,
        parallel_map=parallel_map,
        task_type=args.task_type,
    )

    # Write updated world_model.json
    with open(args.wm_path, "w", encoding="utf-8") as f:
        json.dump(wm, f, ensure_ascii=False, indent=2)

    # A4: soft-demote open descendants of passed-but-stale-direction nodes.
    # Runs BEFORE soft_prune so that demoted nodes that happen to be under a
    # sealed ancestor still get hard-pruned correctly.
    nodes = wm.get("decision_tree", {}).get("nodes", {})
    demoted = soft_demote_stale_directions(
        wm, quality=result.get("worst_quality", "good"), round_num=args.round
    )
    if demoted:
        with open(args.wm_path, "w", encoding="utf-8") as f:
            json.dump(wm, f, ensure_ascii=False, indent=2)
        LOGGER.info("refine: soft-demoted %d stale-direction descendants: %s",
                    len(demoted), demoted)

    # Soft prune dead branches after refine
    pruned = soft_prune_dead_branches(nodes)
    if pruned:
        # Write again with pruned nodes
        with open(args.wm_path, "w", encoding="utf-8") as f:
            json.dump(wm, f, ensure_ascii=False, indent=2)
        LOGGER.info("refine: soft-pruned %d orphaned open nodes: %s",
                    len(pruned), pruned)

    # Write pending_diagnosis.json if there are failed nodes
    if result["pending_diagnosis"]:
        diag_path = os.path.join(args.results_dir, "pending_diagnosis.json")
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(result["pending_diagnosis"], f, ensure_ascii=False, indent=2)
        LOGGER.info("Pending diagnosis written to: %s", diag_path)

    # Print round summary
    DATA_LOGGER.info("%s", result["round_summary"])
    # Re-infer state from filesystem after refine writes wm.session.actual_rounds_completed
    _maybe_infer_state(args.wm_path)
    # Drift circuit breaker — auto-set state.drift_status based on
    # post-refine stagnation counters. Noops if state.json absent.
    _maybe_update_drift_status(args.wm_path, wm)
    # R9: detect missing profiling artifacts and gate next round
    _maybe_mark_profiling_skipped(args.wm_path, args.results_dir, parallel_map)
    # R10 (warn-only): flag if ≥50% partials failed precision
    _maybe_warn_precision_failures(parallel_map, args.results_dir)
    # v3.2 Phase D1: 追加 attempt-ledger.md + lineage.jsonl
    _maybe_write_ledger(args.wm_path, wm, args.round, parallel_map, args.results_dir)


def cmd_session(args: argparse.Namespace) -> None:
    """Write or update session identity anchor in world_model.json.

    Called once at the start of evolution (step 3 init) to pin the session
    to a unique directory and prevent later steps from dynamically
    discovering unrelated historical directories.
    """
    import datetime as dt

    if os.path.isfile(args.wm_path):
        with open(args.wm_path, "r", encoding="utf-8") as f:
            wm = json.load(f)
    else:
        wm = {}

    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
    evo_dir = os.path.abspath(args.evo_dir)

    wm["session"] = {
        "session_id": args.session_id,
        "start_time": now.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requested_rounds": args.requested_rounds,
        "actual_rounds_completed": 0,
        "evo_dir": evo_dir,
        "op_name": args.op_name,
    }

    os.makedirs(os.path.dirname(args.wm_path), exist_ok=True)
    with open(args.wm_path, "w", encoding="utf-8") as f:
        json.dump(wm, f, ensure_ascii=False, indent=2)

    LOGGER.info("session: anchored to %s", evo_dir)
    LOGGER.info("  session_id=%s, requested_rounds=%s",
                args.session_id, args.requested_rounds)
    # Re-infer state from filesystem; session step initializes wm.json
    # which lets the next infer call resolve stage from "shared_prep" to "wm_init"
    _maybe_infer_state(args.wm_path)


def cmd_session_verify(args: argparse.Namespace) -> int:
    """Verify that a directory belongs to the current session.

    Exits with non-zero code if the evo_dir does NOT match the session anchor.
    Used in step 5 (final report) to prevent attribution errors.
    """

    if not os.path.isfile(args.wm_path):
        LOGGER.error("session-verify: FATAL — world_model.json not found at %s",
                     args.wm_path)
        return 1

    with open(args.wm_path, "r", encoding="utf-8") as f:
        wm = json.load(f)

    sess = wm.get("session")
    if not isinstance(sess, dict):
        LOGGER.error("session-verify: FATAL — session field missing in world_model.json")
        return 2

    expected_dir = os.path.abspath(sess.get("evo_dir", ""))
    actual_dir = os.path.abspath(args.evo_dir)

    if expected_dir != actual_dir:
        LOGGER.error(
            "session-verify: FATAL — directory mismatch\n"
            "  expected (from session anchor): %s\n"
            "  actual (provided):              %s\n"
            "  Do NOT dynamically discover directories. Use the evo_dir "
            "from session.anchor only.",
            expected_dir, actual_dir,
        )
        return 3

    actual_rounds = sess.get("actual_rounds_completed", 0)
    requested_rounds = sess.get("requested_rounds", 0)
    if actual_rounds < requested_rounds:
        LOGGER.warning(
            "[WARNING] session-verify: WARNING — only %d/%d "
            "rounds completed. Report must clearly state this.",
            actual_rounds, requested_rounds,
        )

    LOGGER.info("session-verify: OK — %s matches session anchor", actual_dir)
    LOGGER.info("  rounds: %d/%d completed", actual_rounds, requested_rounds)
    return 0








































def cmd_prune(args: argparse.Namespace) -> None:
    """Standalone soft-prune: set difficulty=5 on open nodes under sealed ancestors."""
    with open(args.path, "r", encoding="utf-8") as f:
        wm = json.load(f)

    nodes = wm.get("decision_tree", {}).get("nodes", {})
    pruned = soft_prune_dead_branches(nodes)

    if pruned:
        with open(args.path, "w", encoding="utf-8") as f:
            json.dump(wm, f, ensure_ascii=False, indent=2)
        LOGGER.info("prune: soft-pruned %d nodes: %s", len(pruned), pruned)
    else:
        LOGGER.info("prune: no orphaned open nodes found")






















def _register_query_commands(subparsers):
    """注册查询类子命令：select / validate / summary。"""
    p_select = subparsers.add_parser("select", help="Select top-N open nodes")
    p_select.add_argument("--path", required=True, help="Path to world_model.json")
    p_select.add_argument("--n", type=int, required=True, help="Number of nodes to select")
    p_select.set_defaults(func=cmd_select)

    p_validate = subparsers.add_parser("validate", help="Validate invariants")
    p_validate.add_argument("--path", required=True, help="Path to world_model.json")
    p_validate.set_defaults(func=cmd_validate)

    p_summary = subparsers.add_parser("summary", help="Compact summary for prompt injection")
    p_summary.add_argument("--path", required=True, help="Path to world_model.json")
    p_summary.add_argument(
        "--max-chars", type=int, default=1200,
        help="Maximum characters in output (default: 1200)"
    )
    p_summary.set_defaults(func=cmd_summary)


def _register_evidence_commands(subparsers):
    """注册证据类子命令：deep-profiling / attach-baseline-evidence / refine / diagnose。"""
    p_dp = subparsers.add_parser("deep-profiling", help="Run deep profiling and write to world model")
    p_dp.add_argument("--wm-path", required=True, help="Path to world_model.json")
    p_dp.add_argument("--node-id", required=True, help="Node ID to write profiling_evidence to")
    p_dp.add_argument("--work-dir", required=True, help="Operator work directory (with build artifacts)")
    p_dp.add_argument("--op-name", required=True, help="Operator name")
    p_dp.add_argument("--merge-children", action="store_true",
                       help="Merge evidence strategies into open child nodes")
    p_dp.set_defaults(func=cmd_deep_profiling)

    p_abe = subparsers.add_parser(
        "attach-baseline-evidence",
        help="Write root-level baseline_evidence from baseline_evaluation.json",
    )
    p_abe.add_argument("--wm-path", required=True, help="Path to world_model.json")
    p_abe.add_argument(
        "--baseline-eval", required=True,
        help="Path to baseline_evaluation.json (from evaluate_ops_direct.py)",
    )
    p_abe.set_defaults(func=cmd_attach_baseline_evidence)

    p_refine = subparsers.add_parser(
        "refine", help="Deterministic world model update after a round"
    )
    p_refine.add_argument("--wm-path", required=True, help="Path to world_model.json")
    p_refine.add_argument("--round", type=int, required=True, help="Round number")
    p_refine.add_argument("--results-dir", required=True,
                          help="Directory containing parallel_0/, parallel_1/, etc.")
    p_refine.add_argument("--parallel-map", required=True,
                          help='JSON: {"0":"n1","1":"n2","2":"x0"}')
    p_refine.add_argument("--task-type", default="vector",
                          choices=["vector", "cube", "cv-mix", "unknown"])
    p_refine.set_defaults(func=cmd_refine)

    p_diag = subparsers.add_parser(
        "diagnose", help="Write failure diagnosis for a node"
    )
    p_diag.add_argument("--wm-path", required=True, help="Path to world_model.json")
    p_diag.add_argument("--node-id", required=True, help="Failed node ID")
    p_diag.add_argument("--failure-type", required=True,
                        choices=["impl_error", "strategy_infeasible"])
    p_diag.add_argument("--failure-reason", required=True, help="One-line reason")
    p_diag.set_defaults(func=cmd_diagnose)


def _register_audit_commands(subparsers):
    """注册审计类子命令：verify-notes / prune / finalize-ledger / validate-diagnosis。"""
    p_vn = subparsers.add_parser(
        "verify-notes",
        help="Verify implementation_note.txt presence + size across all partials (FG6)",
    )
    p_vn.add_argument("--evo-dir", required=True, help="Evolution output directory root")
    p_vn.add_argument("--format", choices=["text", "json"], default="text",
                      help="Output format (default: text)")
    p_vn.add_argument("--strict", action="store_true",
                      help="Exit code 2 if any partial fails (default: warn only)")
    p_vn.set_defaults(func=cmd_verify_notes)

    p_prune = subparsers.add_parser(
        "prune", help="Soft-prune open nodes under sealed ancestors"
    )
    p_prune.add_argument("--path", required=True, help="Path to world_model.json")
    p_prune.set_defaults(func=cmd_prune)

    p_fl = subparsers.add_parser(
        "finalize-ledger",
        help="Refresh lineage.jsonl + regenerate attempt-ledger.md with latest node.diagnosis",
    )
    p_fl.add_argument("--wm-path", required=True, help="Path to world_model.json")
    p_fl.add_argument("--evo-dir", default=None,
                      help="Evolution output directory (default: dir containing wm-path)")
    p_fl.set_defaults(func=cmd_finalize_ledger)

    p_vd = subparsers.add_parser(
        "validate-diagnosis",
        help="v3.2: 校验节点 diagnosis 字段 (bottleneck_labels ⊂ 18 项词表 + confidence + text)",
    )
    p_vd.add_argument("--wm-path", required=True, help="Path to world_model.json")
    p_vd.add_argument("--node-id", default=None, help="单个节点 ID（默认校验所有节点）")
    p_vd.add_argument("--strict", action="store_true", help="发现问题时 exit 2")
    p_vd.set_defaults(func=cmd_validate_diagnosis)


def _register_filter_session_commands(subparsers):
    """注册过滤与会话类子命令：filter-candidates / session / session-verify。"""
    p_fc = subparsers.add_parser(
        "filter-candidates",
        help="v3.2: 用 Preconditions YAML 过滤候选策略 ID 列表，可选写入 node.filtered_by",
    )
    p_fc.add_argument("--candidate-ids", required=True,
                      help="逗号分隔的候选策略 ID（如 'P1,P5,P10'）")
    p_fc.add_argument("--kernel-dir", required=True,
                      help="算子源码根目录（含 op_kernel/ 和 op_host/）")
    p_fc.add_argument("--baseline-eval", default=None,
                      help="baseline_evaluation.json 路径（profiling_metric 检查需要）")
    p_fc.add_argument("--precond-dir", default=None,
                      help="Preconditions YAML 目录（默认从 evolution-strategies skill 读）")
    p_fc.add_argument("--wm-path", default=None,
                      help="world_model.json 路径（配合 --node-id 写 filtered_by）")
    p_fc.add_argument("--node-id", default=None,
                      help="节点 ID（写 filtered_by）")
    p_fc.add_argument("--summary", action="store_true",
                      help="打印简短摘要而非 JSON")
    p_fc.set_defaults(func=cmd_filter_candidates)

    p_sess = subparsers.add_parser(
        "session",
        help="Write session identity (session_id/evo_dir/requested_rounds) into world_model.json",
    )
    p_sess.add_argument("--wm-path", required=True, help="Path to world_model.json")
    p_sess.add_argument("--session-id", required=True)
    p_sess.add_argument("--evo-dir", required=True)
    p_sess.add_argument("--requested-rounds", type=int, default=5)
    p_sess.set_defaults(func=cmd_session)

    p_sv = subparsers.add_parser(
        "session-verify",
        help="Verify evo_dir matches the session anchor in world_model.json",
    )
    p_sv.add_argument("--wm-path", required=True, help="Path to world_model.json")
    p_sv.add_argument("--evo-dir", required=True)
    p_sv.set_defaults(func=cmd_session_verify)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="World Model CLI operations for LINGXI evolution."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _register_query_commands(subparsers)
    _register_evidence_commands(subparsers)
    _register_audit_commands(subparsers)
    _register_filter_session_commands(subparsers)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args()
    result = args.func(args)
    sys.exit(result if isinstance(result, int) else 0)


if __name__ == "__main__":
    main()
