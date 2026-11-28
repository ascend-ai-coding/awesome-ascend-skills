#!/usr/bin/env python3
# coding=utf-8
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Step 2 — Layer detection (sample-driven primary + explore fallback).

Architecture-neutral terminology: "block" / "layer" / "component" denote one
repeating model unit (decoder layer, encoder layer, ViT block, hybrid attention
layer, MTP head…). The script makes no commitment to what a unit represents.

Modes:

  sample (primary): triggered by --structure-spec. Takes the user's Phase 0a
    model structure (phases × scheduled layer_compositions × component types) +
    stream_sample_ack.v1 samples (Phase 0b), extracts per-stream fingerprints,
    runs stream-local sliding-window matching, and emits structure_draft.json
    (mode=stream_sample_driven, schema_version=structure_draft.stream.v1,
    components[].op_indices / op_to_component / unmatched_op_indices /
    warnings / validation). Hard warnings block by default; after user review,
    --accept-warnings explicitly releases them. The matching algorithm lives in
    sample_matching.run_stream_sample_mode.

  explore: triggered by --explore. Runs candidates + a default landmark sweep
    without producing a structure_draft. Used when the user can't yet supply
    samples — gives them a kernel-frequency reference to construct 0a/0b.

Range scope: --op-start/--op-end restrict explore to a sub-range. Sample mode
ignores op-range flags (it always scans the full sequence — samples already
encode where each component lives).

No hardcoded model-specific constants — all thresholds are derived from the
data (max period explored = max(2, count // 4)).
"""

import argparse
import json
import logging
import os
import re
import statistics
import sys
from collections import Counter

try:
    from scripts import sample_matching
except ImportError:
    import sample_matching  # 从 skill 根目录直接运行时的同目录导入


logger = logging.getLogger(__name__)


class BlockedWarningsError(RuntimeError):
    """Raised when hard/ambiguous warnings block release (already reported)."""


HARD_WARNING_CODES = {
    "primary_stream_missing",
    "sample_ack_mismatch",
    "stream_shape_mismatch",
    "composition_mismatch",
    "composition_schedule_missing",
    "auxiliary_stream_temporally_displaced",
}

AMBIGUOUS_WARNING_CODES = {
    "ambiguous_match",
    "op_membership_conflict",
    "auxiliary_stream_ambiguous",
    "stream_role_ambiguous",
}


# Auxiliary kernels: pure-format / pure-shape ops that rarely carry semantic
# block boundaries. Used to demote (not eliminate) them from anchor rankings.
# Match is exact-name after stripping a trailing version suffix (V2/V3/D),
# so fused kernels like TransposeBatchMatMul or DequantSwigluQuant — which
# start with one of these names but do real compute — are NOT misclassified.
AUX_KERNEL_NAMES = frozenset({
    "Cast", "Reshape", "Transpose", "DynamicQuant", "Dequant",
    "Squeeze", "Unsqueeze", "Slice", "Split", "View", "Contiguous",
    "Identity", "BroadcastTo", "ExpandDims", "Fill", "ZerosLike",
    "Concat", "ScatterNdUpdate", "RotaryMul", "AivKernel",
})
_VERSION_SUFFIX_RE = re.compile(r"(V\d+|D)$")


def load_ops(raw_path):
    with open(raw_path, "r") as f:
        data = json.load(f)
    return data["operators"], data.get("step_id")


def load_structure_spec(path: str) -> dict:
    with open(path) as f:
        spec = json.load(f)
    if "phases" not in spec or not isinstance(spec["phases"], list):
        raise ValueError(f"structure_spec {path} 缺 phases 列表")
    if "expected_components" not in spec:
        types = set()
        for ph in spec["phases"]:
            for comp in ph.get("layer_compositions", []):
                types.update(comp.get("components", []))
        spec["expected_components"] = sorted(types)
    return spec


def validate_inputs_json(path: str) -> dict:
    """Step 0 物料校验：prof_dir + model_script_paths 必须存在且非空。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise ValueError(
            f"[Phase 0 物料缺失] 找不到 {path}。\n"
            f"回 SKILL.md 的「前置物料 checklist」段：必须先和用户敲齐 prof 路径 + "
            f"模型脚本路径，写到 <run_dir>/inputs.json。"
        ) from e
    except json.JSONDecodeError as e:
        raise ValueError(f"[Phase 0 物料损坏] {path} 不是合法 JSON: {e}") from e
    paths = data.get("model_script_paths") or []
    if not isinstance(paths, list) or not paths:
        raise ValueError(
            f"[Phase 0 物料缺失] {path} 的 model_script_paths 为空。\n"
            f"模型脚本路径是硬约束，不允许「没有/不方便」——agent 必须先要到，"
            f"否则 Phase 0b 的 stream sample ack 无法做。"
        )
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise ValueError(
            f"[Phase 0 物料缺失] inputs.json 里的模型脚本路径不存在：{missing}"
        )
    if data.get("phase") in (None, ""):
        raise ValueError(
            f"[Phase 0 物料缺失] {path} 的 phase 字段未填。多卡/多 phase 场景下必须先和"
            f"用户敲定 (phase, rank) 组合（prefill / decode 各一），不允许 agent 自作主张默认。"
        )
    if data.get("rank") is None:
        raise ValueError(
            f"[Phase 0 物料缺失] {path} 的 rank 字段未填。必须和用户敲定具体 rank。"
        )
    return data


def _load_sample_ack(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise ValueError(
            f"[Phase 0b 未 ack] 找不到 {path}。\n"
            f"回 SKILL.md 的「Phase 0b」段：必须先和用户敲定每个 component 的 "
            f"stream sample 范围，写到 <run_dir>/sample_ack.json 后再跑 detect。"
        ) from e
    except json.JSONDecodeError as e:
        raise ValueError(f"[Phase 0b ack 文件损坏] {path}: {e}") from e


def _check_sample_ack_component(name: str, entry: dict) -> list[str]:
    """Validate one component's stream_samples entry; return problem strings."""
    problems = []
    if not entry:
        return [f"  - {name}: 完全缺失 stream sample 条目"]
    stream_samples = entry.get("stream_samples") or []
    primary = [s for s in stream_samples if s.get("role") == "primary"]
    if len(primary) != 1:
        problems.append(f"  - {name}: 必须且只能有一个 primary stream sample")
    for i, sample in enumerate(stream_samples):
        if not sample.get("stream_id"):
            problems.append(f"  - {name}[{i}]: 缺 stream_id")
        if not sample.get("op_indices"):
            problems.append(f"  - {name}[{i}]: 缺 op_indices")
    return problems


def validate_sample_ack(path: str, expected_components: list) -> dict:
    """Phase 0b ack 校验：每个 expected_component 都必须有 stream sample。"""
    ack = _load_sample_ack(path)
    comps = ack.get("components") or {}
    if not isinstance(comps, dict):
        raise ValueError(f"[Phase 0b ack 格式错] {path} 的 components 应为 dict")
    if ack.get("schema_version") != "stream_sample_ack.v1":
        raise ValueError(
            f"[Phase 0b ack 格式错] {path} 必须使用 "
            f"schema_version='stream_sample_ack.v1'"
        )
    problems = []
    for name in expected_components:
        problems.extend(_check_sample_ack_component(name, comps.get(name)))
    if problems:
        raise ValueError(
            "[Phase 0b 未 ack] sample_ack.json 不全，必须每个 expected_component 都有 "
            "用户 ack 过的 stream sample：\n" + "\n".join(problems)
        )
    return ack


def blocking_warnings(warnings: list[dict]) -> list[dict]:
    return [w for w in warnings if w.get("code") in HARD_WARNING_CODES]


def ambiguous_warnings(warnings: list[dict]) -> list[dict]:
    return [w for w in warnings if w.get("code") in AMBIGUOUS_WARNING_CODES]


def is_auxiliary(name):
    base = _VERSION_SUFFIX_RE.sub("", name)
    return base in AUX_KERNEL_NAMES


def resolve_op_range(args, ops):
    """Compute (start, end) inclusive op-index window from --op-start/--op-end.

    Returns (start, end, source_description) where description is logged.
    """
    n = len(ops)
    start = args.op_start if args.op_start is not None else 0
    end = args.op_end if args.op_end is not None else n - 1
    if start < 0 or end >= n or start > end:
        raise ValueError(f"Invalid op range [{start}, {end}] for {n} ops")
    if args.op_start is None and args.op_end is None:
        return start, end, "full range"
    return start, end, f"--op-start={start} --op-end={end}"


def block_multiset(ops, start, end):
    """Shape-aware fingerprint: (normalized_name, input_shapes) -> count."""
    cnt = Counter()
    for i in range(start, end + 1):
        name = ops[i]["normalized_name"]
        shape = ops[i].get("input_shapes", "")
        cnt[(name, shape)] += 1
    return tuple(sorted(cnt.items()))


def segment_blocks(positions, n_anchors_per_block, range_end):
    """Segment anchor positions into N-tuples; last block extends to range_end."""
    blocks = []
    for li in range(0, len(positions), n_anchors_per_block):
        anchors = positions[li: li + n_anchors_per_block]
        if len(anchors) < n_anchors_per_block:
            break
        start = anchors[0]
        next_idx = li + n_anchors_per_block
        if next_idx < len(positions):
            end = positions[next_idx] - 1
        else:
            end = range_end
        blocks.append({
            "block_idx": len(blocks),
            "anchors": anchors,
            "start": start,
            "end": end,
        })
    return blocks


def gap_cv(positions, p):
    """Mean within-phase coefficient-of-variation of inter-anchor gaps.

    For TRUE anchors appearing P times per block, partitioning gaps by i mod P
    yields uniform groups (gap_cv ≈ 0). For non-structural anchors, gap_cv
    stays high at every period. Used as a period-discovery signal.
    """
    if len(positions) < 2 * p + 1:
        return None
    gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    cvs = []
    for k in range(p):
        grp = gaps[k::p]
        if len(grp) < 2:
            continue
        mean = sum(grp) / len(grp)
        if mean <= 0:
            continue
        cvs.append(statistics.pstdev(grp) / mean)
    return sum(cvs) / len(cvs) if cvs else None


def per_period_stats(ops, positions, max_p, range_total=None, range_end=None):
    """For p in 1..max_p, run segmentation and report block/cluster stats."""
    if range_total is None:
        range_total = len(ops)
    if range_end is None:
        range_end = len(ops) - 1
    rows = []
    for p in range(1, max_p + 1):
        if len(positions) // p < 2:
            break
        blocks = segment_blocks(positions, p, range_end)
        if len(blocks) < 2:
            continue
        sigs = {block_multiset(ops, b["start"], b["end"]) for b in blocks}
        sizes = [b["end"] - b["start"] + 1 for b in blocks]
        cov = sum(sizes) / range_total
        median = statistics.median(sizes)
        cv_size = statistics.pstdev(sizes) / median if median else float("inf")
        rows.append({
            "p": p,
            "num_blocks": len(blocks),
            "clusters": len(sigs),
            "compression": len(sigs) / len(blocks),
            "coverage": cov,
            "size_cv": cv_size,
            "gap_cv": gap_cv(positions, p),
        })
    return rows


def next_kernel_distribution(ops, positions):
    """Distinct kernels immediately following each anchor position."""
    cnt = Counter()
    for p in positions:
        if p + 1 < len(ops):
            cnt[ops[p + 1]["normalized_name"]] += 1
    return cnt


def _group_positions_by_name(ops, start, end):
    """Map normalized_name -> list of op indices in [start, end] (in order)."""
    by_name = {}
    for i in range(start, end + 1):
        by_name.setdefault(ops[i]["normalized_name"], []).append(i)
    return by_name


def _top_candidates(candidates, mid_key, top_k):
    """Rank candidates in place and return the top_k slice.

    Every candidate ranking shares the same primary key (non-auxiliary first)
    and the same tie-breaker (name); callers supply only the middle
    discriminators via ``mid_key`` (negate any key that should sort desc).
    """
    candidates.sort(key=lambda c: (c["auxiliary"], *mid_key(c), c["name"]))
    return candidates[:top_k]


def build_candidates(ops, op_range, min_count=4, top_k=40):
    """Per-kernel candidacy report. No 'best' is chosen — AI picks in Step 2."""
    start, end = op_range
    by_name = _group_positions_by_name(ops, start, end)

    range_total = end - start + 1
    candidates = []
    for name, positions in by_name.items():
        if len(positions) < min_count:
            continue
        aux = is_auxiliary(name)
        # max_p is data-driven: require ≥4 samples per phase group.
        max_p = max(2, len(positions) // 4)
        per_p = per_period_stats(ops, positions, max_p, range_total, end)
        if not per_p:
            continue
        nxt = next_kernel_distribution(ops, positions)
        candidates.append({
            "name": name,
            "count": len(positions),
            "auxiliary": aux,
            "first_pos": positions[0],
            "last_pos": positions[-1],
            "next_kernels_distinct": len(nxt),
            "next_kernels": [{"name": n, "count": c}
                             for n, c in nxt.most_common(5)],
            "per_period": per_p,
        })

    # Structural-anchor quality: low next-kernel diversity first (anchors that
    # lead into a fixed template are structural), then count desc.
    return _top_candidates(
        candidates, lambda c: (c["next_kernels_distinct"], -c["count"]), top_k
    )


def _landmark_intervals(start, landmark_positions):
    """Build (exclusive_lo, exclusive_hi_landmark) intervals between landmarks."""
    intervals = []
    prev = start - 1
    for li in landmark_positions:
        intervals.append((prev, li))
        prev = li
    return intervals


def _collect_slot_offsets(positions, intervals, k):
    """Collect per-slot offsets for a kernel appearing k times per interval.

    Returns slot_offsets (list of k offset lists) or None if the kernel does
    not appear exactly k times in every interval.
    """
    pi = 0
    slot_offsets = [[] for _ in range(k)]
    for lo, hi in intervals:
        occ_positions = []
        while pi < len(positions) and positions[pi] < hi:
            if positions[pi] > lo:
                occ_positions.append(positions[pi])
            pi += 1
        if len(occ_positions) != k:
            return None
        occ_positions.sort()
        for s, pos in enumerate(occ_positions):
            slot_offsets[s].append(hi - pos)
    return slot_offsets


def _slot_stats(slot_offsets):
    """Summarize each slot's offset distribution from a landmark."""
    slot_stats = []
    for s, offs in enumerate(slot_offsets):
        mean = sum(offs) / len(offs)
        sd = statistics.pstdev(offs) if len(offs) > 1 else 0.0
        slot_stats.append({
            "slot": s,
            "mean_offset": round(mean, 2),
            "stdev_offset": round(sd, 3),
            "min_offset": min(offs),
            "max_offset": max(offs),
        })
    return slot_stats


def _landmark_candidate(name, positions, intervals, n_landmarks):
    """Build a single landmark-anchored candidate dict, or None if it fails."""
    if len(positions) % n_landmarks != 0:
        return None
    k = len(positions) // n_landmarks
    if k < 1 or k > 4:
        return None
    slot_offsets = _collect_slot_offsets(positions, intervals, k)
    if slot_offsets is None:
        return None
    slot_stats = _slot_stats(slot_offsets)
    return {
        "name": name,
        "count": len(positions),
        "auxiliary": is_auxiliary(name),
        "repeats_per_interval": k,
        "earliest_offset_from_landmark": slot_stats[0]["mean_offset"],
        "max_slot_stdev": max(s["stdev_offset"] for s in slot_stats),
        "slots": slot_stats,
    }


def build_landmark_candidates(ops, op_range, landmark_name, top_k=40):
    """Find anchor candidates by reference to a high-signal landmark kernel.

    Strategy: divide the analysis window into intervals delimited by
    consecutive landmark occurrences. A kernel X is a strong block-anchor
    candidate iff it appears EXACTLY ONCE in every interval and its offset
    to the trailing landmark is highly stable across all intervals.
    """
    start, end = op_range
    landmark_positions = [
        i for i in range(start, end + 1)
        if ops[i]["normalized_name"] == landmark_name
    ]
    if len(landmark_positions) < 2:
        return [], landmark_positions

    n_landmarks = len(landmark_positions)
    by_name = _group_positions_by_name(ops, start, end)

    intervals = _landmark_intervals(start, landmark_positions)
    candidates = []
    for name, positions in by_name.items():
        if name == landmark_name:
            continue
        cand = _landmark_candidate(name, positions, intervals, n_landmarks)
        if cand is not None:
            candidates.append(cand)

    ranked = _top_candidates(
        candidates,
        lambda c: (c["max_slot_stdev"], -c["earliest_offset_from_landmark"]),
        top_k,
    )
    return ranked, landmark_positions


def _best_period_row(per_period):
    """Pick the period with the lowest gap_cv (most stable cadence)."""
    scored = [r for r in per_period if r.get("gap_cv") is not None]
    if not scored:
        return None, None, None, None
    r = min(scored, key=lambda x: x["gap_cv"])
    return r["p"], r["gap_cv"], r["size_cv"], r["coverage"]


def _bc_cell(per_p, p):
    """Render one 'num_blocks/clusters' table cell for period p."""
    r = per_p.get(p)
    return f"{r['num_blocks']}/{r['clusters']}" if r else "-"


def print_candidates_table(candidates, total_ops):
    logger.info("Total ops: %d", total_ops)
    logger.info(
        "%-36s %3s %6s %7s %5s %7s %7s %5s %-22s %9s %9s %9s %9s",
        "name", "aux", "count", "nxt-div", "bestP", "gap_cv", "sizeCv",
        "cov", "next-top", "p=1 B/C", "p=2 B/C", "p=3 B/C", "p=4 B/C",
    )
    for c in candidates:
        per_p = {r["p"]: r for r in c["per_period"]}
        best_p, gap_cv_v, size_cv, cov = _best_period_row(c["per_period"])
        aux = "Y" if c.get("auxiliary") else "N"
        bp = str(best_p) if best_p is not None else "-"
        gcv = f"{gap_cv_v:.3f}" if gap_cv_v is not None else "-"
        scv = f"{size_cv:.3f}" if size_cv is not None else "-"
        cvs = f"{cov:.2f}" if cov is not None else "-"
        top_next = ",".join(f"{n['name']}:{n['count']}" for n in c["next_kernels"][:2])
        if len(top_next) > 22:
            top_next = top_next[:21] + "…"
        logger.info(
            "%-36s %3s %6s %7s %5s %7s %7s %5s %-22s %9s %9s %9s %9s",
            c["name"], aux, c["count"], c["next_kernels_distinct"], bp, gcv,
            scv, cvs, top_next, _bc_cell(per_p, 1), _bc_cell(per_p, 2),
            _bc_cell(per_p, 3), _bc_cell(per_p, 4),
        )


def print_landmark_table(candidates, landmark_name, n_landmarks):
    logger.info("landmark: %s (×%d)", landmark_name, n_landmarks)
    logger.info(
        "%-38s %3s %6s %3s %10s %8s %-30s",
        "name", "aux", "count", "k", "early_off", "max_sd", "slot_offsets",
    )
    for c in candidates:
        slot_str = ", ".join(f"{s['mean_offset']:.0f}" for s in c["slots"])
        aux = "Y" if c.get("auxiliary") else "N"
        logger.info(
            "%-38s %3s %6s %3s %10.2f %8.3f %-30s",
            c["name"], aux, c["count"], c["repeats_per_interval"],
            c["earliest_offset_from_landmark"], c["max_slot_stdev"], slot_str,
        )


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-r", "--raw_ops", required=True, help="raw_ops.json path")
    p.add_argument("-o", "--output", required=True,
                   help="stream sample mode → structure_draft.json; explore mode → "
                        "candidates + landmark_candidates report")
    p.add_argument("--structure-spec", default=None,
                   help="(sample mode) 0a 解析结果 JSON 路径")
    p.add_argument("--inputs", default=None,
                   help="(sample mode 必传) <run_dir>/inputs.json，Step 0 物料 "
                        "(prof_dir + model_script_paths)；缺则 exit 1")
    p.add_argument("--sample-ack", default=None,
                   help="(sample mode 必传) <run_dir>/sample_ack.json，Phase 0b "
                        "用户 ack 过的每个 component 的 stream samples；"
                        "缺则 exit 1")
    p.add_argument("--accept-warnings", action="store_true",
                   help="(sample mode) 默认 hard/ambiguous warnings 会 exit 1 "
                        "强迫复读给用户；用户明确接受后加此 flag 放行")
    p.add_argument("--explore", action="store_true",
                   help="生成 candidates + landmark_candidates；用户拿不出 sample 时探索用")
    p.add_argument("--min-count", type=int, default=4,
                   help="(explore mode) 候选最小出现次数 (default 4)")
    p.add_argument("--top-k", type=int, default=40,
                   help="(explore mode) 最多产 N 个候选 (default 40)")
    p.add_argument("--print-table", action="store_true",
                   help="(explore mode) 在 stdout 打印 summary 表")
    p.add_argument("--op-start", type=int, default=None,
                   help="(explore mode) 限定分析范围起点")
    p.add_argument("--op-end", type=int, default=None,
                   help="(explore mode) 限定分析范围终点（含）")
    return p


def _report_blocked_warnings(blocked: list[dict]) -> None:
    """Log hard/ambiguous warnings to stderr (output contract preserved)."""
    logger.warning(
        "\n[Phase 1.5 未 ack] %d 条 hard/ambiguous 警告，必须复读给用户：",
        len(blocked),
    )
    for w in blocked[:10]:
        logger.warning("  - [%s] %s", w.get("code"), w.get("message"))
    if len(blocked) > 10:
        logger.warning("  ... 还有 %d 条", len(blocked) - 10)
    logger.warning(
        "用户 ack 完每条后重跑并加 --accept-warnings；不要直接传 flag 跳过。",
    )


def run_sample_mode(args) -> None:
    """Sample-driven mode: emit structure_draft.json; raise on blocked warnings."""
    if not args.structure_spec:
        raise ValueError("sample mode 需要给 --structure-spec")
    if not args.inputs:
        raise ValueError(
            "[Phase 0 物料缺失] sample mode 必传 --inputs <run_dir>/inputs.json "
            "(prof_dir + model_script_paths)。"
        )
    if not args.sample_ack:
        raise ValueError(
            "[Phase 0b 未 ack] sample mode 必传 --sample-ack <run_dir>/sample_ack.json "
            "(用户 ack 过的每个 component stream sample)。"
        )
    ops_raw = load_ops(args.raw_ops)
    operators = ops_raw[0] if isinstance(ops_raw, tuple) else ops_raw
    spec = load_structure_spec(args.structure_spec)
    validate_inputs_json(args.inputs)
    ack = validate_sample_ack(args.sample_ack, spec["expected_components"])
    ack_components = set(ack["components"].keys())
    spec_components = set(spec["expected_components"])
    if spec_components - ack_components:
        raise ValueError(
            f"[Phase 0b 未 ack] structure_spec 的 expected_components 包含 "
            f"{spec_components - ack_components}，但 sample_ack.json 里没对应条目"
        )

    draft = sample_matching.run_stream_sample_mode(operators, spec, ack)
    out = sample_matching.stream_draft_to_dict(draft, spec)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info("stream_sample_driven structure_draft → %s", args.output)
    logger.info("  components: %d", len(out["components"]))
    logger.info("  warnings:   %d", len(out["warnings"]))

    blocked = blocking_warnings(out["warnings"]) + ambiguous_warnings(out["warnings"])
    if blocked and not args.accept_warnings:
        _report_blocked_warnings(blocked)
        raise BlockedWarningsError


def run_explore_mode(args) -> None:
    """Explore mode: emit candidates + landmark_candidates report."""
    ops, step_id = load_ops(args.raw_ops)
    range_start, range_end, range_desc = resolve_op_range(args, ops)
    op_range = (range_start, range_end)
    candidates = build_candidates(ops, op_range,
                                  min_count=args.min_count, top_k=args.top_k)
    landmarks = {}
    for lm in ("FlashAttentionScore", "FusedInferAttentionScore",
               "RmsNorm", "LayerNorm", "MoeGatingTopKHash"):
        lm_cands, _lm_positions = build_landmark_candidates(
            ops, op_range, lm, top_k=args.top_k)
        if lm_cands:
            landmarks[lm] = lm_cands
    payload = {
        "total_ops": len(ops),
        "step_id": step_id,
        "op_range": {"start": range_start, "end": range_end,
                     "ops_count": range_end - range_start + 1,
                     "source": range_desc},
        "candidates_count": len(candidates),
        "candidates": candidates,
        "landmark_candidates": landmarks,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("explore output → %s", args.output)
    logger.info("  range: %s (%d ops)", range_desc, range_end - range_start + 1)
    logger.info("  candidates: %d", len(candidates))
    if args.print_table:
        print_candidates_table(candidates, range_end - range_start + 1)
        for lm, cands in landmarks.items():
            logger.info("")
            print_landmark_table(cands, lm, sum(1 for op in ops
                                                if op.get("normalized_name") == lm))


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        stream=sys.stderr)
    args = build_arg_parser().parse_args()

    sample_mode = bool(args.structure_spec)
    if sample_mode and args.explore:
        logger.error("sample mode (--structure-spec) 不能与 --explore 同时使用")
        sys.exit(1)
    if not sample_mode and not args.explore:
        logger.error("必须指定模式：sample (--structure-spec) 或 --explore")
        sys.exit(1)

    try:
        if sample_mode:
            run_sample_mode(args)
        else:
            run_explore_mode(args)
    except BlockedWarningsError:
        sys.exit(1)
    except (ValueError, RuntimeError, OSError, KeyError) as e:
        logger.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
