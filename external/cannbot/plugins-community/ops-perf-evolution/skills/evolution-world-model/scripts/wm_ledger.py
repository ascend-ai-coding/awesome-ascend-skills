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
"""wm_ledger.py — Ledger artifacts 模块（从 wm_ops.py 拆分）。

职责：
- attempt-ledger.md / lineage.jsonl 的追加写（_maybe_write_ledger）
- finalize-ledger：用最新 node.diagnosis 刷新 lineage + 重建 ledger.md +
  应用 next_round_hint + 回填 candidate_sources（cmd_finalize_ledger）
- strategy ID → source_key 反查（INDEX.json 缓存）
- evaluation_results.json 多结构解析
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

LOGGER = logging.getLogger(__name__)

_LEDGER_INDEX_CACHE = None


def _collect_ancestry_failed_strategies(node_id: str, nodes: dict) -> set:
    """v3.3: walk node→root, collect strategies used by any failed ancestor.

    Used to auto-add ancestry failures to the anti set when deriving children,
    so the search never re-tries a combination that already failed upstream.
    """
    failed: set = set()
    cur = nodes.get(node_id)
    seen = set()
    while cur and cur.get("parent_id") and cur["id"] not in seen:
        seen.add(cur["id"])
        cur = nodes.get(cur["parent_id"])
        if cur and cur.get("status") in ("failed", "failed_compile", "failed_precision"):
            failed.update(cur.get("strategy_combination", []) or [])
    return failed


def _load_index_from_candidates(candidates: list):
    """按候选路径依次尝试加载 INDEX.json，全部失败返回 {}。"""
    for p in candidates:
        if not os.path.isfile(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
    return {}


def _load_strategies_index_for_ledger():

    """Cache 加载 INDEX.json 用于 strategy ID → source_key 反查。"""
    global _LEDGER_INDEX_CACHE
    if _LEDGER_INDEX_CACHE is not None:
        return _LEDGER_INDEX_CACHE

    # 多路径搜寻
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(os.getcwd(), "plugins-community/ops-perf-evolution/skills/"
                                 "evolution-strategies/references/INDEX.json"),
        os.path.join(here, "..", "..", "plugins-community/ops-perf-evolution/skills/"
                                     "evolution-strategies/references/INDEX.json"),
        os.path.join(here, "..", "..", "..", "plugins-community/ops-perf-evolution/skills/"
                                        "evolution-strategies/references/INDEX.json"),
    ]
    _LEDGER_INDEX_CACHE = _load_index_from_candidates(candidates)
    return _LEDGER_INDEX_CACHE


def _id_to_source_keys(strategy_id: str) -> list:
    """把 strategy ID (e.g. 'P1') 转为 source_keys 列表（含 card + 可选 playbook）。"""
    idx = _load_strategies_index_for_ledger()
    if not idx:
        # Fallback: 没有 INDEX，构造一个最小 source_key（猜测格式）
        return [f"evolution-strategies#card/{strategy_id}"]

    keys: list = []
    for c in idx.get("cards", []):
        if c.get("id") == strategy_id:
            keys.append(c.get("source_key"))
    for pb in idx.get("playbooks", []):
        if pb.get("id") == strategy_id:
            keys.append(pb.get("source_key"))
    return keys or [f"evolution-strategies#card/{strategy_id}"]


def _all_known_strategy_ids() -> set:
    """Strategy IDs known to INDEX.json (cards). Empty set means INDEX absent →
    callers should treat ID legality as 'cannot verify, accept'. Always allow
    P-ShapeSpec-01 (architectural constraint, not a card)."""
    idx = _load_strategies_index_for_ledger()
    ids = {c.get("id") for c in idx.get("cards", []) if c.get("id")}
    ids.add("P-ShapeSpec-01")
    return ids


def _parse_multi_shape_eval(data: dict, targets: list, generalization: list) -> dict:
    """解析 multi_shape 结构的评估结果。"""
    # target 是否全部通过（multi target 时取交集）
    all_compile = all(t.get("compilation_success") for t in targets)
    all_precision = all(t.get("precision_passed") for t in targets)
    # speedup 取 aggregate 优先，targets[0] 兜底
    agg = data.get("aggregate") or {}
    speedup = (
        agg.get("target_geo_mean_speedup")
        or agg.get("target_min_speedup")
        or targets[0].get("speedup")
    )
    result = {
        "compile": all_compile,
        "precision": all_precision,
        "speedup": speedup,
        "_source": "multi_shape",
    }
    # F-G2: generalization 状态（不影响主 compile/precision，作为 warnings）
    if generalization:
        gen_compile = all(g.get("compilation_success") for g in generalization)
        gen_precision = all(g.get("precision_passed") for g in generalization)
        warnings: list[str] = []
        if not gen_compile:
            warnings.append("generalization_compile_failed")
        if not gen_precision:
            warnings.append("generalization_precision_failed")
        # 也加上 regression 信号
        if agg.get("any_generalization_regression"):
            warnings.append("generalization_regression")
        result["_generalization_compile"] = gen_compile
        result["_generalization_precision"] = gen_precision
        if warnings:
            result["_warnings"] = warnings
    return result


def _parse_comparison_eval(data: dict) -> Optional[dict]:
    """解析 flat_with_comparison 结构，字段全空返回 None。"""
    comparison = data.get("comparison") or {}
    evolved = data.get("evolved") or {}
    if not (comparison or evolved):
        return None
    compile_val = comparison.get("compilation_success")
    precision_val = (
        comparison.get("precision_passed")
        if comparison.get("precision_passed") is not None
        else evolved.get("precision_passed")
    )
    speedup_val = comparison.get("speedup") or data.get("speedup")
    # 仅当至少有一个字段非 None 时才认为是 comparison 结构
    if not any(v is not None for v in (compile_val, precision_val, speedup_val)):
        return None
    return {
        "compile": compile_val,
        "precision": precision_val,
        "speedup": speedup_val,
        "_source": "flat_with_comparison",
    }


def _load_eval_results_for_parallel(results_dir: str, pidx_str: str) -> Optional[dict]:

    """读取 round_N/parallel_K/evaluation_results.json，兼容多种 evaluator 输出结构。

    返回归一化后的 dict: {compile, precision, speedup, _source, _warnings?}
    若文件不存在或解析失败返回 None。

    支持的结构（按检测优先级）:

    1. multi_shape (新 ops-evo, e.g. rms_norm):
       data.shape_results.target[i].{compilation_success, precision_passed, speedup}
       data.shape_results.generalization[i].* (可选，跑了泛化时存在)
       data.aggregate.target_geo_mean_speedup
       → 主 compile/precision 来自 target；generalization 状态作为 _warnings

    2. flat_with_comparison (旧 ops-evo 单 shape, e.g. lightning_indexer_grad):
       data.comparison.{compilation_success, precision_passed, speedup}
       data.evolved.precision_passed (兜底)

    3. flat (扁平兜底，最简单的 evaluator 输出):
       data.{compilation_success, precision_passed, speedup}
    """
    path = os.path.join(results_dir, f"parallel_{pidx_str}", "evaluation_results.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    # 优先 multi-shape 结构（shape_results.target/generalization）
    shape_results = data.get("shape_results") or {}
    targets = shape_results.get("target") or []
    generalization = shape_results.get("generalization") or []

    if targets:
        return _parse_multi_shape_eval(data, targets, generalization)

    # F-G1: 旧 flat with comparison 结构（lightning_indexer_grad 类）
    result = _parse_comparison_eval(data)
    if result is not None:
        return result

    # 兜底：最扁平结构
    return {
        "compile": data.get("compilation_success"),
        "precision": data.get("precision_passed"),
        "speedup": data.get("speedup"),
        "_source": "flat",
    }


def _write_ledger_header(f, op_name: str):
    """第一次创建 ledger.md 时写表头。"""
    f.write(f"# Attempt Ledger — {op_name}\n\n")
    f.write("v3.2 ledger 自动追加产物。\n")
    f.write("`source_keys` 列出本变体所用策略的 source_key（多个 ;` 分隔），")
    f.write("便于反向追溯到 cards/preconditions/playbooks 文件。\n\n")
    f.write("| round | parallel | node_id | strategies | source_keys | "
            "compile | precision | speedup | filtered_by | diagnosis_labels |\n")
    f.write("|---|---|---|---|---|---|---|---|---|---|\n")


def _resolve_compile_precision_marks(eval_data: Optional[dict], node: dict) -> tuple:
    """从 eval_data（或节点字段兜底）推断 compile/precision 标记与 speedup。"""
    if eval_data is not None:
        compile_ok = "✓" if eval_data["compile"] else "✗"
        if eval_data["precision"] is None:
            precision_ok = "?" if not eval_data["compile"] else "✗"
        else:
            precision_ok = "✓" if eval_data["precision"] else "✗"
        speedup = eval_data["speedup"] or node.get("score") or 0.0
        return compile_ok, precision_ok, speedup
    # 回退到节点字段推断（向后兼容）
    compile_ok = "✓" if node.get("compile_success") else (
        "✗" if node.get("status") in {"failed_compile", "failed", "blocked"} else "?"
    )
    precision_ok = "✓" if node.get("precision_passed") else (
        "✗" if node.get("status") == "failed_precision" else "?"
    )
    return compile_ok, precision_ok, node.get("score") or 0.0


def _build_ledger_row(round_num: int, pidx_str: str, nid: str, node: dict,
                      results_dir: str) -> str:
    """构建 attempt-ledger.md 的单行 markdown。"""
    strategies = node.get("strategy_combination") or []
    # 反查 source_keys
    source_keys_list: list = []
    for sid in strategies:
        source_keys_list.extend(_id_to_source_keys(sid))
    source_keys_str = "; ".join(source_keys_list) if source_keys_list else "—"

    # v3.2 C8-T1: 从 evaluation_results.json 真实读 compile/precision/speedup
    eval_data = _load_eval_results_for_parallel(results_dir, pidx_str)
    compile_ok, precision_ok, speedup = _resolve_compile_precision_marks(eval_data, node)
    speedup_str = f"{speedup:.3f}" if isinstance(speedup, (int, float)) else "—"

    filtered_by = node.get("filtered_by", []) or []
    filtered_by_str = "; ".join(filtered_by) if filtered_by else "—"

    diagnosis = node.get("diagnosis") or {}
    labels = diagnosis.get("bottleneck_labels", []) or []
    labels_str = ", ".join(labels) if labels else "—"

    strategies_str = ", ".join(strategies) if strategies else "—"

    return (
        f"| {round_num} | {pidx_str} | {nid} | {strategies_str} | "
        f"`{source_keys_str}` | {compile_ok} | {precision_ok} | "
        f"{speedup_str} | {filtered_by_str} | {labels_str} |\n"
    )


@dataclass
class _LineageContext:
    """lineage 记录构建的共享上下文。"""
    nodes: dict
    round_num: int
    results_dir: str


def _build_lineage_entry(nid: str, node: dict, lc: _LineageContext, pidx_str: str) -> dict:
    """构建 lineage.jsonl 的单条记录。"""
    nodes = lc.nodes
    round_num = lc.round_num
    results_dir = lc.results_dir
    strategies = node.get("strategy_combination") or []
    source_keys_list: list = []
    for sid in strategies:
        source_keys_list.extend(_id_to_source_keys(sid))

    # mutation = 与父节点策略组合的 diff
    parent_id = node.get("parent_id") or "root"
    parent_node = nodes.get(parent_id) or {}
    parent_strategies = parent_node.get("strategy_combination") or []
    added = sorted(set(strategies) - set(parent_strategies))
    removed = sorted(set(parent_strategies) - set(strategies))

    # v3.2 C8-T1: 从 evaluation_results.json 真实读 compile/precision
    eval_data = _load_eval_results_for_parallel(results_dir, pidx_str)
    if eval_data is not None:
        compile_success = eval_data["compile"]
        precision_passed = eval_data["precision"]
        speedup = eval_data["speedup"] or node.get("score")
    else:
        compile_success = node.get("compile_success")
        precision_passed = node.get("precision_passed")
        speedup = node.get("score")

    return {
        "node_id": nid,
        "parent_id": parent_id,
        "round": round_num,
        "parallel": int(pidx_str) if pidx_str.isdigit() else pidx_str,
        "strategies": strategies,
        "source_keys": source_keys_list,
        "mutation": {"added": added, "removed": removed},
        "compile_success": compile_success,
        "precision_passed": precision_passed,
        "speedup": speedup,
        "diagnosis": node.get("diagnosis"),
        "filtered_by": node.get("filtered_by", []),
    }


def _is_lineage_passed(node: dict) -> bool:
    """v3.2 C8-T1: 用 node.status + score 判 passed，而非缺失的 compile_success 字段。"""
    return (
        node.get("status") == "passed"
        or node.get("precision_passed")
        or (node.get("score") or 0) > 0
    )


@dataclass
class _LedgerWriteContext:
    """ledger 写入的共享上下文。"""
    nodes: dict
    round_num: int
    results_dir: str
    parallel_map: dict


def _append_ledger_rows(ledger_md: str, op_name: str, wc: _LedgerWriteContext):
    """向 attempt-ledger.md 追加本轮表格行（首次创建时写表头）。"""
    ledger_is_new = not os.path.exists(ledger_md)
    with open(ledger_md, "a", encoding="utf-8") as f:
        if ledger_is_new:
            _write_ledger_header(f, op_name)
        for pidx_str, nid in wc.parallel_map.items():
            node = wc.nodes.get(nid)
            if node is None:
                continue
            f.write(_build_ledger_row(wc.round_num, pidx_str, nid, node, wc.results_dir))


def _append_lineage_entries(lineage_jsonl: str, wc: _LedgerWriteContext):
    """向 lineage.jsonl 追加每个 passed 节点一行 JSON。"""
    lineage_ctx = _LineageContext(wc.nodes, wc.round_num, wc.results_dir)
    with open(lineage_jsonl, "a", encoding="utf-8") as f:
        for pidx_str, nid in wc.parallel_map.items():
            node = wc.nodes.get(nid)
            if node is None or not _is_lineage_passed(node):
                continue
            entry = _build_lineage_entry(nid, node, lineage_ctx, pidx_str)
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _maybe_write_ledger(
    wm_path: str,
    wm: dict,
    round_num: int,
    parallel_map: dict,
    results_dir: str,
) -> None:
    """v3.2 Phase D1: 在 evo_dir/artifacts/ 追加写 attempt-ledger.md 和 lineage.jsonl。

    幂等性：使用追加模式，避免覆盖历史；第一次创建文件时写表头。

    数据来源：
    - parallel_map 映射当前轮的 parallel_idx → node_id
    - wm.decision_tree.nodes 提供节点详情
    """
    try:
        evo_dir = os.path.dirname(os.path.abspath(wm_path))
        artifacts_dir = os.path.join(evo_dir, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        ledger_md = os.path.join(artifacts_dir, "attempt-ledger.md")
        lineage_jsonl = os.path.join(artifacts_dir, "lineage.jsonl")

        nodes = wm.get("decision_tree", {}).get("nodes", {})
        op_name = wm.get("session", {}).get("op_name", "unknown")

        wc = _LedgerWriteContext(nodes, round_num, results_dir, parallel_map)
        _append_ledger_rows(ledger_md, op_name, wc)
        _append_lineage_entries(lineage_jsonl, wc)
    except OSError as e:
        # ledger 写失败不应阻塞主流程，仅 stderr 警告
        LOGGER.warning("WARN: _maybe_write_ledger failed: %s", e)


def _refresh_lineage(lineage_jsonl: str, nodes: dict) -> tuple:
    """Step 1: 用 world_model 最新 diagnosis 刷新 lineage.jsonl。

    Returns (refreshed_entries, updated_count)。
    """
    refreshed: list[dict] = []
    updated_count = 0
    with open(lineage_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            nid = entry.get("node_id")
            current_diag = (nodes.get(nid) or {}).get("diagnosis")
            if current_diag and current_diag != entry.get("diagnosis"):
                entry["diagnosis"] = current_diag
                updated_count += 1
            refreshed.append(entry)

    # Rewrite lineage.jsonl atomically
    tmp_path = lineage_jsonl + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for entry in refreshed:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    os.replace(tmp_path, lineage_jsonl)
    return refreshed, updated_count


def _entry_status_mark(value) -> str:
    """布尔/None → ✓/✗/? 状态标记（truthy→✓，False→✗，其他 falsy→?）。"""
    if value:
        return "✓"
    if value is False:
        return "✗"
    return "?"


def _build_ledger_row_from_entry(entry: dict) -> str:
    """由单条 lineage 记录构建 ledger.md 表格行。"""
    rnd = entry.get("round", "?")
    pidx = entry.get("parallel", "?")
    nid = entry.get("node_id", "?")
    strategies = entry.get("strategies") or []
    strategies_str = ", ".join(strategies) if strategies else "—"
    source_keys = entry.get("source_keys") or []
    source_keys_str = "; ".join(source_keys) if source_keys else "—"
    compile_ok = _entry_status_mark(entry.get("compile_success"))
    precision_ok = _entry_status_mark(entry.get("precision_passed"))
    sp = entry.get("speedup")
    speedup_str = f"{sp:.3f}" if isinstance(sp, (int, float)) else "—"
    filtered_by = entry.get("filtered_by") or []
    filtered_str = "; ".join(filtered_by) if filtered_by else "—"
    diag = entry.get("diagnosis") or {}
    labels = diag.get("bottleneck_labels") if isinstance(diag, dict) else None
    labels_str = ", ".join(labels) if labels else "—"
    return (
        f"| {rnd} | {pidx} | {nid} | {strategies_str} | `{source_keys_str}` | "
        f"{compile_ok} | {precision_ok} | {speedup_str} | {filtered_str} | {labels_str} |"
    )


def _regenerate_ledger_md(ledger_md: str, refreshed: list, op_name: str) -> int:
    """Step 2: 从 reconcile 后的 lineage 全量重建 attempt-ledger.md。

    Returns 写入的行数。
    """
    refreshed.sort(key=lambda e: (e.get("round", 0),
                                  e.get("parallel", 0) if isinstance(e.get("parallel"), int) else 0))
    rows = [_build_ledger_row_from_entry(entry) for entry in refreshed]

    os.makedirs(os.path.dirname(ledger_md), exist_ok=True)
    with open(ledger_md, "w", encoding="utf-8") as f:
        f.write(f"# Attempt Ledger — {op_name}\n\n")
        f.write("v3.2 ledger 自动追加产物 (finalize-ledger reconciled).\n")
        f.write("`source_keys` 列出本变体所用策略的 source_key（多个 ;` 分隔），")
        f.write("便于反向追溯到 cards/preconditions/playbooks 文件。\n\n")
        f.write("| round | parallel | node_id | strategies | source_keys | "
                "compile | precision | speedup | filtered_by | diagnosis_labels |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(r + "\n")
    return len(rows)


def _rewrite_oe_child_description(child: dict, parent: dict, parent_id: str,
                                  hint: dict) -> bool:
    """R16: rewrite stub description (from refine) with parent diagnosis info
    so it reaches ≥80 chars and references profiling facts + hint rationale."""
    diag = parent.get("diagnosis") or {}
    labels = diag.get("bottleneck_labels") or []
    prefer = hint.get("prefer") or []
    avoid = hint.get("avoid") or []
    rationale = hint.get("rationale") or ""
    diag_text = diag.get("diagnosis_text") or ""

    parts = [f"[开放探索] 基于父节点 {parent_id} 分析继续探索。"]
    if labels:
        parts.append(f"瓶颈: {', '.join(labels)}。")
    if prefer:
        parts.append(f"优先: {', '.join(prefer)}。")
    if avoid:
        parts.append(f"避免: {', '.join(avoid)}。")
    if rationale:
        parts.append(f"依据: {rationale}。")
    if diag_text:
        parts.append(f"诊断: {diag_text[:200]}。")

    new_desc = " ".join(parts)
    if len(new_desc) >= 80 and new_desc != child.get("description"):
        child["description"] = new_desc
        return True
    return False


def _apply_hint_to_oe_child(child: dict, parent: dict, parent_id: str, hint: dict) -> int:
    """对 open_exploration 子节点应用 hint（重写描述 + avoid 合并）。

    返回应用的变更数。
    """
    applied = 0
    if _rewrite_oe_child_description(child, parent, parent_id, hint):
        applied += 1
    # existing logic: avoid → ancestry_avoid
    avoid = hint.get("avoid") or []
    if avoid:
        existing = child.get("ancestry_avoid") or []
        merged = sorted(set(existing) | set(avoid))
        if merged != existing:
            child["ancestry_avoid"] = merged
            applied += 1
    return applied


def _apply_hint_to_children(parent_id: str, parent: dict, nodes: dict, hint: dict) -> int:
    """对单个 passed 父节点的 open 子节点应用 next_round_hint。

    返回应用的变更总数。
    """
    try:
        from profiling_evidence import MergeConfig, merge_strategies_with_evidence
    except ImportError:
        _pr = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if _pr not in sys.path:
            sys.path.insert(0, _pr)
        from profiling_evidence import MergeConfig, merge_strategies_with_evidence
    ev = parent.get("profiling_evidence") or {}
    ancestry_failed = _collect_ancestry_failed_strategies(parent_id, nodes)
    applied = 0
    sib_idx = 0
    for cid in parent.get("children", []):
        child = nodes.get(cid)
        if not child or child.get("status") != "open":
            continue
        if child.get("solution_ref"):
            continue  # already dispatched, don't disturb
        if child.get("mode") == "open_exploration":
            applied += _apply_hint_to_oe_child(child, parent, parent_id, hint)
            continue
        new_sc = merge_strategies_with_evidence(
            parent.get("strategy_combination", []), ev,
            ancestry_failed=ancestry_failed,
            parent_hint=hint,
            config=MergeConfig(offset=sib_idx),
        )
        if new_sc != child.get("strategy_combination"):
            child["strategy_combination"] = new_sc
            applied += 1
        sib_idx += 1
    return applied


def _apply_hints_to_open_children(nodes: dict) -> int:

    """Step 3 (v3.3): apply LLM-written next_round_hint to open children.

    Background: refine creates children BEFORE the LLM writes diagnosis, so
    at refine time `parent.diagnosis.next_round_hint = {}`. Once the LLM
    populates it and the agent calls finalize-ledger, we re-merge the
    children's strategy_combination here so the hint actually influences
    next-round derivation. Idempotent (same hint → same merge output).
    """
    try:
        from profiling_evidence import MergeConfig, merge_strategies_with_evidence
    except ImportError:
        _pr = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if _pr not in sys.path:
            sys.path.insert(0, _pr)
        from profiling_evidence import MergeConfig, merge_strategies_with_evidence

    hint_applied = 0
    for parent_id, parent in nodes.items():
        if parent.get("status") != "passed":
            continue
        hint = (parent.get("diagnosis") or {}).get("next_round_hint") or {}
        if not (hint.get("prefer") or hint.get("avoid")):
            continue
        hint_applied += _apply_hint_to_children(parent_id, parent, nodes, hint)
    return hint_applied


def _backfill_candidate_sources(nodes: dict) -> int:
    """Step 4 (v3.2 Stage 3 收口): backfill candidate_sources on nodes whose
    LLM-written diagnosis is present but Stage 3 reverse lookup wasn't run.

    candidate_sources is a per-node audit log of which strategies match the
    node's bottleneck_labels — surfaced to agent for inspection. utility
    computation does NOT depend on this field (compute_utility recomputes
    from parent.facts × parent.diagnosis × node.strategy_combination at
    call time), so this step is purely for ledger / debug visibility.
    """
    try:
        from profiling_evidence import match_strategies_by_labels
    except ImportError:
        from profiling_evidence import match_strategies_by_labels  # type: ignore[no-redef]

    candidate_sources_written = 0
    for node in nodes.values():
        diag = node.get("diagnosis")
        if not isinstance(diag, dict):
            continue
        if node.get("candidate_sources") is not None:
            continue  # idempotent
        labels = diag.get("bottleneck_labels") or []
        if not labels:
            continue
        result = match_strategies_by_labels(labels, include_unknown=True)
        node["candidate_sources"] = {
            "candidate_source_keys": result.get("candidate_source_keys", []),
            "candidate_ids": result.get("candidate_ids", []),
            "by_label": result.get("by_label", {}),
            "unknown_labels": result.get("unknown_labels", []),
        }
        candidate_sources_written += 1
    return candidate_sources_written


def cmd_finalize_ledger(args: argparse.Namespace) -> int:
    """Reconcile ledger artifacts with the latest world_model.json diagnoses.

    Background: `refine` writes ledger rows before the LLM produces
    `node.diagnosis` (LLM diagnosis is a follow-up step). Without
    reconciliation, lineage.jsonl carries `diagnosis: null` and
    attempt-ledger.md carries `diagnosis_labels: —` even after diagnoses
    are written, breaking cross-session strategy mining.

    Strategy: lineage.jsonl is the append-only truth source for which
    (round, parallel, node_id) tuples ran. For each entry, we look up
    the current `node.diagnosis` in world_model.json and refresh the
    entry's `diagnosis` field. attempt-ledger.md is then fully
    regenerated from the reconciled lineage.

    Idempotent: safe to call multiple times.
    """
    wm_path = args.wm_path
    evo_dir = args.evo_dir or os.path.dirname(os.path.abspath(wm_path))
    artifacts_dir = os.path.join(evo_dir, "artifacts")
    ledger_md = os.path.join(artifacts_dir, "attempt-ledger.md")
    lineage_jsonl = os.path.join(artifacts_dir, "lineage.jsonl")

    if not os.path.isfile(lineage_jsonl):
        LOGGER.warning("finalize-ledger: no lineage.jsonl at %s; nothing to do",
                       lineage_jsonl)
        return 0

    with open(wm_path, "r", encoding="utf-8") as f:
        wm = json.load(f)
    nodes = wm.get("decision_tree", {}).get("nodes", {})
    op_name = wm.get("session", {}).get("op_name", "unknown")

    refreshed, updated_count = _refresh_lineage(lineage_jsonl, nodes)
    rows_count = _regenerate_ledger_md(ledger_md, refreshed, op_name)

    hint_applied = _apply_hints_to_open_children(nodes)
    if hint_applied:
        # rewrite wm.json so the re-merged sc / ancestry_avoid persists
        with open(wm_path, "w", encoding="utf-8") as f:
            json.dump(wm, f, ensure_ascii=False, indent=2)

    candidate_sources_written = _backfill_candidate_sources(nodes)
    if candidate_sources_written:
        with open(wm_path, "w", encoding="utf-8") as f:
            json.dump(wm, f, ensure_ascii=False, indent=2)

    LOGGER.info("finalize-ledger: %d entries; %d diagnosis refreshed; "
                "ledger.md regenerated (%d rows); "
                "hint applied to %d open child node(s); "
                "candidate_sources backfilled for %d node(s)",
                len(refreshed), updated_count, rows_count, hint_applied,
                candidate_sources_written)
    return 0
