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
"""wm_audit.py — 世界模型审计/诊断模块（从 wm_ops.py 拆分）。

职责：
- diagnose：失败/方向诊断写回（impl_error 生成修复子节点 / strategy_infeasible 封印）
- filter-candidates：Preconditions 硬过滤候选策略
- validate-diagnosis：节点 diagnosis 字段合规校验
- verify-notes：implementation_note.txt 位置与长度事后审计
"""

import argparse
import glob
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from wm_log import DATA_LOGGER

LOGGER = logging.getLogger(__name__)

# Ensure sibling scripts are importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from wm_ledger import _all_known_strategy_ids  # noqa: E402


def _generate_fix_child(nodes: dict, node: dict, node_id: str,
                        failure_reason: str) -> str:
    """为 impl_error 节点生成修复子节点，返回 fix_id。"""
    node["difficulty"] = min(4, node.get("difficulty", 3) + 1)
    retry = node.get("retry_count", 0) + 1
    fix_id = f"{node_id}_fix{retry}"
    mode = node.get("mode", "strategy_guided")
    fix_child = {
        "id": fix_id,
        "mode": mode,
        "strategy_combination": list(node.get("strategy_combination", [])),
        "description": f"[修复实现] {failure_reason}",
        "optimization_type": node.get("optimization_type", "algorithm"),
        "difficulty": node["difficulty"],
        "depth": node.get("depth", 1) + 1,
        "parent_id": node_id,
        "status": "open",
        "score": None,
        "solution_ref": None,
        "children": [],
        "failure_type": None,
        "failure_reason": None,
        "retry_count": retry,
        "profiling_insight": None,
        "profiling_evidence": None,
    }
    nodes[fix_id] = fix_child
    node.setdefault("children", []).append(fix_id)
    return fix_id


def cmd_diagnose(args: argparse.Namespace) -> int:

    """Write failure/direction diagnosis for a node (called by agent after LLM reasoning).

    Supports two semantic modes depending on the target node's status:

      1. status=="failed" (legacy):
           - impl_error → generate fix child node, difficulty++
           - strategy_infeasible / retry>=2 → seal node (difficulty=5)

      2. status=="passed" + failure_type=="strategy_infeasible" (new: A6):
           Direction-level seal. The node itself stays passed (it did run
           successfully), but the agent has determined via semantic review
           (e.g. comparing baseline bottleneck vs evolved bottleneck, or
           observing a sibling variant along a different direction produced
           a substantially better speedup) that continuing down this
           direction is unlikely to yield further gains. We set
           direction_sealed=True and difficulty=5; soft_prune_dead_branches
           then demotes all open descendants. No fix child is generated
           because the direction has been disproven, not broken.

    Other status + failure_type combinations write failure_type/reason without
    sealing or generating children (informational).
    """
    with open(args.wm_path, "r", encoding="utf-8") as f:
        wm = json.load(f)

    nodes = wm.get("decision_tree", {}).get("nodes", {})
    node = nodes.get(args.node_id)
    if not node:
        LOGGER.error("ERROR: node '%s' not found", args.node_id)
        return 1

    node["failure_type"] = args.failure_type
    node["failure_reason"] = args.failure_reason
    status = node.get("status")

    if status == "passed" and args.failure_type == "strategy_infeasible":
        # A6: direction-level seal on a passed node.
        node["direction_sealed"] = True
        node["difficulty"] = 5
        LOGGER.info("diagnose: passed+strategy_infeasible → direction-sealed "
                    "node '%s' (difficulty=5, status stays passed)", args.node_id)
    elif args.failure_type == "impl_error" and node.get("retry_count", 0) < 2:
        # Generate fix child node
        fix_id = _generate_fix_child(nodes, node, args.node_id, args.failure_reason)
        LOGGER.info("diagnose: impl_error → generated fix child '%s'", fix_id)
    elif args.failure_type == "strategy_infeasible" or node.get("retry_count", 0) >= 2:
        node["difficulty"] = 5
        LOGGER.info("diagnose: %s → sealed node (difficulty=5)", args.failure_type)
    else:
        LOGGER.info("diagnose: wrote failure_type=%s", args.failure_type)

    # Run soft prune after diagnosis — a newly sealed node may orphan open descendants
    from wm_ops import soft_prune_dead_branches  # 延迟 import 避免循环
    pruned = soft_prune_dead_branches(nodes)
    if pruned:
        LOGGER.info("diagnose: soft-pruned %d orphaned open nodes: %s",
                    len(pruned), pruned)

    with open(args.wm_path, "w", encoding="utf-8") as f:
        json.dump(wm, f, ensure_ascii=False, indent=2)
    return 0


def _find_precond_script():
    """定位 check_preconditions.py 脚本路径。"""
    script_path = (Path(__file__).resolve().parent.parent.parent.parent
                   / "plugins-community/ops-perf-evolution/skills/evolution-strategies/scripts/check_preconditions.py")
    if script_path.exists():
        return script_path
    # 兜底：从 cwd 找
    script_path = Path("plugins-community/ops-perf-evolution/skills/"
                       "evolution-strategies/scripts/check_preconditions.py").resolve()
    return script_path if script_path.exists() else None


def _run_precond_check(script_path, candidate_ids: list, args):
    """调用 check_preconditions.py，返回 (returncode, stdout, stderr)。"""
    cmd = [
        sys.executable,
        str(script_path),
        "--strategy-ids", ",".join(candidate_ids),
        "--kernel-dir", args.kernel_dir,
    ]
    if args.baseline_eval:
        cmd.extend(["--baseline-eval", args.baseline_eval])
    if args.precond_dir:
        cmd.extend(["--precond-dir", args.precond_dir])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return -1, "", "check_preconditions timed out (60s)"
    return proc.returncode, proc.stdout, proc.stderr


def _build_filter_result(raw: dict, candidate_ids: list) -> dict:
    """整理 check_preconditions 输出为 filter 结果。"""
    passed: list[str] = []
    failed_detail: list[dict] = []
    filtered_by_keys: list[str] = []

    for sid in candidate_ids:
        info = raw.get(sid)
        if info is None:
            # 没有 Preconditions YAML 的策略 → 默认通过（fail-safe）
            passed.append(sid)
            continue
        if info.get("passed"):
            passed.append(sid)
        else:
            failed_checks = info.get("failed_checks", [])
            failed_detail.append({"id": sid, "checks": failed_checks})
            for c in failed_checks:
                check_name = c.get("id") or "unknown"
                filtered_by_keys.append(f"{sid}.precondition.{check_name}")

    return {
        "passed": passed,
        "failed": failed_detail,
        "filtered_by_keys": filtered_by_keys,
        "input_count": len(candidate_ids),
        "passed_count": len(passed),
    }


def _write_filtered_by_to_node(args, filtered_by_keys: list, result: dict):
    """可选写入：把 filtered_by 字段写入 world_model 对应节点。"""
    if not (args.wm_path and args.node_id):
        return
    with open(args.wm_path, "r", encoding="utf-8") as f:
        wm = json.load(f)
    nodes = wm.get("decision_tree", {}).get("nodes", {})
    node = nodes.get(args.node_id)
    if node is None:
        LOGGER.warning("WARN: node %s not in world_model.json", args.node_id)
        return
    # 累积式写入（多次 filter 调用结果合并）
    existing = node.get("filtered_by", []) or []
    merged = list(dict.fromkeys(existing + filtered_by_keys))
    node["filtered_by"] = merged
    with open(args.wm_path, "w", encoding="utf-8") as f:
        json.dump(wm, f, indent=2, ensure_ascii=False)
    result["wrote_to_node"] = args.node_id


def _print_filter_summary(candidate_ids: list, result: dict):
    """--summary 模式打印人类可读摘要。"""
    DATA_LOGGER.info("Input: %d candidates", len(candidate_ids))
    DATA_LOGGER.info("Passed: %s", result['passed'])
    DATA_LOGGER.info("Failed: %s", [f['id'] for f in result['failed']])
    if result['filtered_by_keys']:
        DATA_LOGGER.info("Filtered by:")
        for k in result['filtered_by_keys']:
            DATA_LOGGER.info("  - %s", k)


def cmd_filter_candidates(args: argparse.Namespace) -> int:
    """v3.2 Phase C3: 用 Preconditions 硬过滤候选策略 ID 列表。

    封装 plugins-community/ops-perf-evolution/skills/evolution-strategies/scripts/check_preconditions.py，
    在 partial-prompt 注入前剔除不适用的策略。

    可选写入：若给 --wm-path + --node-id，把 filtered_by 字段写入对应节点。
    """
    candidate_ids = [s.strip() for s in args.candidate_ids.split(",") if s.strip()]
    if not candidate_ids:
        LOGGER.error("ERROR: --candidate-ids cannot be empty")
        return 1

    script_path = _find_precond_script()
    if script_path is None:
        LOGGER.error("ERROR: check_preconditions.py not found")
        return 1

    returncode, stdout, stderr = _run_precond_check(script_path, candidate_ids, args)
    if returncode == -1:
        LOGGER.error("ERROR: %s", stderr)
        return 1
    if returncode != 0:
        LOGGER.error("check_preconditions exit %d", returncode)
        LOGGER.error("%s", stderr)
        return returncode

    # 解析 check_preconditions 的 JSON 输出
    # 输出结构: {strategy_id: {"passed": bool, "failed_checks": [...]}}
    try:
        raw = json.loads(stdout)
    except json.JSONDecodeError:
        # 不是 JSON，可能没有 Preconditions 资源 — fail-safe: 全部通过
        LOGGER.warning("WARN: check_preconditions output not JSON, "
                       "fail-safe to all-pass:\n%s", stdout[:200])
        raw = {sid: {"passed": True, "failed_checks": []} for sid in candidate_ids}

    result = _build_filter_result(raw, candidate_ids)
    _write_filtered_by_to_node(args, result["filtered_by_keys"], result)

    if args.summary:
        _print_filter_summary(candidate_ids, result)
    else:
        DATA_LOGGER.info("%s", json.dumps(result, indent=2, ensure_ascii=False))

    # 若所有候选都被过滤掉 → 警告但不 exit 失败（allow caller 决定）
    if not result["passed"]:
        LOGGER.warning("WARN: all %d candidates filtered out", len(candidate_ids))
    return 0


def _check_diag_labels(nid: str, diag: dict, issues: list, validate_labels):
    """Check 1: bottleneck_labels 非空且在词表内。"""
    labels = diag.get("bottleneck_labels", [])
    if not isinstance(labels, list) or not labels:
        issues.append({
            "node_id": nid,
            "type": "labels_missing_or_empty",
            "diagnosis": diag,
        })
        return
    v = validate_labels(labels)
    if not v["valid"]:
        issues.append({
            "node_id": nid,
            "type": "labels_unknown",
            "unknown": v["unknown"],
        })


def _check_diag_confidence(nid: str, diag: dict, issues: list):
    """Check 2: confidence ∈ [0, 1]。"""
    conf = diag.get("confidence")
    if conf is None or not isinstance(conf, (int, float)) or not (0 <= conf <= 1):
        issues.append({
            "node_id": nid,
            "type": "confidence_invalid",
            "value": conf,
        })


def _check_diag_text(nid: str, diag: dict, issues: list):
    """Check 3: diagnosis_text 非空且 ≥ 20 字符。"""
    text = diag.get("diagnosis_text", "")
    if not isinstance(text, str) or len(text.strip()) < 20:
        issues.append({
            "node_id": nid,
            "type": "diagnosis_text_too_short",
            "length": len(text.strip()) if isinstance(text, str) else 0,
        })


def _check_hint_ids(nid: str, prefer: list, avoid: list, known_ids: set, issues: list):
    """Check 4a: prefer/avoid 数量、重叠与 ID 合法性。"""
    if len(prefer) + len(avoid) > 3:
        issues.append({"node_id": nid, "type": "hint_too_many",
                       "count": len(prefer) + len(avoid)})
    if set(prefer) & set(avoid):
        issues.append({"node_id": nid, "type": "hint_prefer_avoid_overlap",
                       "overlap": sorted(set(prefer) & set(avoid))})
    if known_ids:
        bad = [s for s in (prefer + avoid) if s not in known_ids]
        if bad:
            issues.append({"node_id": nid, "type": "hint_unknown_ids",
                           "unknown": bad})


def _check_diag_hint(nid: str, diag: dict, known_ids: set, issues: list):
    """Check 4 (v3.3): passed+round_ 节点必有 next_round_hint 且内容合规。"""
    hint = diag.get("next_round_hint")
    if not isinstance(hint, dict):
        issues.append({"node_id": nid, "type": "hint_missing"})
        return
    prefer = hint.get("prefer", [])
    avoid = hint.get("avoid", [])
    rationale = hint.get("rationale", "")
    if not isinstance(prefer, list) or not isinstance(avoid, list):
        issues.append({"node_id": nid, "type": "hint_prefer_avoid_not_list"})
    else:
        _check_hint_ids(nid, prefer, avoid, known_ids, issues)
    if not isinstance(rationale, str) or not rationale.strip():
        issues.append({"node_id": nid, "type": "hint_rationale_empty"})


def _check_one_node_diagnosis(nid: str, node: dict, known_ids: set,
                              issues: list, validate_labels) -> bool:
    """校验单个节点的 diagnosis。返回是否计入 checked。"""
    if not node:
        issues.append({"node_id": nid, "type": "missing_node"})
        return False
    # v3.3: passed + round_ 节点必须有 diagnosis（含 next_round_hint）
    passed_round = (
        node.get("status") == "passed"
        and isinstance(node.get("solution_ref"), str)
        and node.get("solution_ref", "").startswith("round_")
    )
    diag = node.get("diagnosis")
    if not diag:
        if passed_round:
            issues.append({"node_id": nid, "type": "diagnosis_missing_on_passed"})
        return False  # 非 passed 节点无 diagnosis 不算问题

    _check_diag_labels(nid, diag, issues, validate_labels)
    _check_diag_confidence(nid, diag, issues)
    _check_diag_text(nid, diag, issues)
    if passed_round:
        _check_diag_hint(nid, diag, known_ids, issues)
    return True


def cmd_validate_diagnosis(args: argparse.Namespace) -> int:

    """v3.2/v3.3: 校验 world_model.json 中节点的 diagnosis 字段是否合规。

    检查项：
    1. diagnosis.bottleneck_labels ⊂ KNOWN_BOTTLENECK_LABELS（18 项词表）
    2. diagnosis.confidence ∈ [0, 1]
    3. diagnosis.diagnosis_text 非空且 ≥ 20 字符
    4. (v3.3) passed+round_ 节点必有 next_round_hint，且 prefer/avoid 合法 ID、
       rationale 非空、prefer∩avoid=∅、|prefer|+|avoid|≤3

    用途：
    - refine 阶段后调用，发现 LLM 输出不合规的诊断
    - CI 检查 ledger 中 diagnosis 字段历史合规性
    """
    try:
        from profiling_evidence import validate_labels
    except ImportError:
        _pr = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if _pr not in sys.path:
            sys.path.insert(0, _pr)
        from profiling_evidence import validate_labels

    with open(args.wm_path, "r", encoding="utf-8") as f:
        wm = json.load(f)
    nodes = wm.get("decision_tree", {}).get("nodes", {})
    known_ids = _all_known_strategy_ids()

    target_ids = [args.node_id] if args.node_id else list(nodes.keys())

    issues: list[dict] = []
    checked = 0
    for nid in target_ids:
        node = nodes.get(nid)
        if _check_one_node_diagnosis(nid, node, known_ids, issues, validate_labels):
            checked += 1

    output = {
        "checked_nodes": checked,
        "total_nodes_with_diagnosis": checked,
        "issues_count": len(issues),
        "issues": issues,
        "valid": len(issues) == 0,
    }
    DATA_LOGGER.info("%s", json.dumps(output, indent=2, ensure_ascii=False))
    if not output["valid"] and args.strict:
        return 2
    return 0


def _find_note_location(pdir: str) -> tuple:
    """查找 implementation_note.txt 的合法位置，返回 (location, size)。

    location: "top"（推荐顶层）/ "modified_files"（兼容子目录）/ None（缺失）。
    """
    top_note = os.path.join(pdir, "implementation_note.txt")
    sub_note = os.path.join(pdir, "modified_files", "implementation_note.txt")
    if os.path.isfile(top_note):
        return "top", os.path.getsize(top_note)
    if os.path.isfile(sub_note):
        return "modified_files", os.path.getsize(sub_note)
    return None, 0


def _check_one_note(evo_dir: str, pdir: str, r12_min_len: int) -> dict:
    """检查单个 partial 的 implementation_note.txt，返回审计记录。"""
    rel = os.path.relpath(pdir, evo_dir)
    location, size = _find_note_location(pdir)

    if location is None:
        status = "MISSING"
    elif size < r12_min_len:
        status = "TOO_SHORT"
    else:
        status = "PASS"

    return {
        "partial": rel,
        "status": status,
        "location": location,
        "size_bytes": size,
    }


def _print_verify_results_table(evo_dir: str, results: list,
                                pass_count: int, fail_count: int):
    """打印人类可读的审计结果表格。"""
    DATA_LOGGER.info("=== Note verification: %s ===", evo_dir)
    DATA_LOGGER.info("%-30s %-10s %-16s %8s", 'Partial', 'Status', 'Location', 'Size')
    DATA_LOGGER.info("-" * 70)
    for r in results:
        loc = r["location"] or "—"
        DATA_LOGGER.info("%-30s %-10s %-16s %8d",
                         r['partial'], r['status'], loc, r['size_bytes'])
    DATA_LOGGER.info("\nResult: %d/%d PASS, %d FAIL", pass_count, len(results), fail_count)
    if fail_count > 0:
        DATA_LOGGER.info("\nFailed partials:")
        for r in results:
            if r["status"] != "PASS":
                DATA_LOGGER.info("  - %s (%s)", r['partial'], r['status'])


def cmd_verify_notes(args: argparse.Namespace) -> int:

    """v3.2 Phase C8-FG6: 事后审计每个 partial 的 implementation_note.txt 位置和长度。

    扫描 evo_dir/round_*/parallel_*/，识别两个合法位置：
        - parallel_X/implementation_note.txt          (推荐顶层)
        - parallel_X/modified_files/implementation_note.txt (兼容子目录)
    任一存在 + ≥ r12_min_len 字符 → ✓ PASS
    否则 → ✗ MISSING / TOO_SHORT

    用于：
    - e2e 跑完后验证 R12 实际通过率（不再只看顶层路径）
    - CI 检查 ledger 引用的 partial 是否都产出元数据
    - 主 agent 在 refine 前批量检查本轮所有 partial 是否合规
    """
    r12_min_len = 100

    evo_dir = os.path.abspath(args.evo_dir)
    if not os.path.isdir(evo_dir):
        LOGGER.error("verify-notes: evo-dir not found: %s", evo_dir)
        return 2

    # 扫所有 round_*/parallel_*/ 目录
    parallel_dirs = sorted(glob.glob(os.path.join(evo_dir, "round_*", "parallel_*")))
    if not parallel_dirs:
        LOGGER.warning("verify-notes: no round_*/parallel_*/ directories under %s",
                       evo_dir)
        return 0

    results: list[dict] = []
    pass_count = 0
    fail_count = 0

    for pdir in parallel_dirs:
        entry = _check_one_note(evo_dir, pdir, r12_min_len)
        results.append(entry)
        if entry["status"] == "PASS":
            pass_count += 1
        else:
            fail_count += 1

    if args.format == "json":
        DATA_LOGGER.info("%s", json.dumps({
            "evo_dir": evo_dir,
            "total": len(results),
            "pass": pass_count,
            "fail": fail_count,
            "results": results,
        }, indent=2, ensure_ascii=False))
    else:
        _print_verify_results_table(evo_dir, results, pass_count, fail_count)

    if args.strict and fail_count > 0:
        return 2
    return 0
