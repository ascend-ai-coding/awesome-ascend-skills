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
"""Render per-component sub-item dashboard from a sample-driven draft.

Inputs:
  -d structure_draft.json   Step 2 stream_sample_driven output
                            (structure_draft.stream.v1, op_indices schema).
  -r raw_ops.json           Step 1 single-step op list (operators[]).
  -s network_spec.json      Per-network clustering rules. See schema below.
  --raw-ops-details         raw_ops_details.json from Step 1. If it has been
                            enriched by merge_theoretical_columns.py, theory
                            median columns are computed directly from it.
  --insight-annotations     Optional agent-authored high/medium annotations for
                            the final insight column. Omit on initial render.
  -o index.html             Output (single-page HTML).

Network spec schema:

    {
      "model_name": "<network>",
      "outlier": {"method": "iqr", "k": 1.5},     // or {"method": "z", "k": 2}
      "component_clusters": {
        "csa": [
          {
            "cluster": "input_norm",
            "description": "Pre-norm before Q projection",
            "rules": [{"op_name": "RmsNorm"}]
          },
          {
            "cluster": "q_compress",
            "rules": [
              {"op_name": "MatMulV3", "input_shapes_contains": "<dim>"}
            ]
          },
          {"cluster": "other", "rules": [{"catch_all": true}]}
        ],
        "moe": [...]
      }
    }

Rule fields (all listed conditions must hold; first matching cluster wins):
  op_name                exact match on normalized_name
  op_name_regex          regex on normalized_name
  input_shapes_contains  substring match on input_shapes
  input_shapes_regex     regex on input_shapes
  output_shapes_contains substring match on output_shapes
  output_shapes_regex    regex on output_shapes
  catch_all              true → matches any op (use as last fallback)

Reported sub-items per component_type:
  - one row per cluster: wall_ms = union duration of that cluster's ops in the
    instance; outliers detected on wall_ms across instances.
  - one synthetic row 'bubble': layer-level idle gap, computed as
    span_ms − union_wall_ms (gap between adjacent ops on the layer timeline).
    Outliers detected on the bubble series across instances.
  - TOTAL row: layer end-to-end span = max_op_end − min_op_start, i.e. the
    wall-clock interval a user sees when framing the layer in trace_view.
    Equals union_wall + bubble. Outliers on that span series.

Cross-cluster overlap (overlap_summary): the absolute gap is
Σ cluster_wall − union_wall (time double-counted across buckets); the
reported pct is gap / TOTAL (span) so it lines up with the layer
end-to-end time a user sees in trace_view.

Outlier method: IQR (k=1.5 default) or z-score; configured by spec.outlier.
"""

import argparse
import json
import logging
import os
import re
import html
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

# _to_float is shared with merge_theoretical_columns (same scripts dir) to keep
# a single CSV-string-to-float parser; import it rather than duplicating.
from merge_theoretical_columns import _to_float


logger = logging.getLogger(__name__)


UNMATCHED_PCT_HARD_LIMIT = 0.05    # > 5% unmatched ops 视为 Phase 1.5 漏 ack
CLUSTER_COVERAGE_MIN_PCT = 80.0    # Σ cluster_wall.median / TOTAL.median 低于此值时红字提示


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_draft_schema(draft):
    if (draft.get("mode") != "stream_sample_driven"
            or draft.get("schema_version") != "structure_draft.stream.v1"):
        raise ValueError(
            f"draft mode={draft.get('mode')!r}, "
            f"schema_version={draft.get('schema_version')!r} not supported — "
            f"render.py only accepts stream_sample_driven / "
            f"structure_draft.stream.v1 drafts."
        )
    for i, component in enumerate(draft.get("components", [])):
        if "op_indices" not in component:
            raise ValueError(f"draft component[{i}] missing op_indices")


def op_indices_from_component(component):
    return [int(i) for i in component.get("op_indices") or []]


def op_indices_from_instance(instance):
    return [int(i) for i in instance.get("op_indices") or []]


def instances_by_type_from_draft(draft):
    by_type = defaultdict(list)
    for c in draft.get("components", []):
        by_type[c["type"]].append({
            "layer_idx": c["layer_idx"],
            "phase": c["phase"],
            "op_indices": op_indices_from_component(c),
            "displaced_op_indices": [int(i) for i in c.get("displaced_op_indices") or []],
        })
    return by_type


def build_match_predicate(rule):
    """Compile a single rule dict into a fn(op) -> bool."""
    if rule.get("catch_all"):
        return lambda _op: True

    op_name = rule.get("op_name")
    op_name_re = re.compile(rule["op_name_regex"]) if rule.get("op_name_regex") else None
    in_contains = rule.get("input_shapes_contains")
    in_re = re.compile(rule["input_shapes_regex"]) if rule.get("input_shapes_regex") else None
    out_contains = rule.get("output_shapes_contains")
    out_re = re.compile(rule["output_shapes_regex"]) if rule.get("output_shapes_regex") else None

    def match(op):
        name = op.get("normalized_name", "")
        if op_name is not None and name != op_name:
            return False
        if op_name_re is not None and not op_name_re.search(name):
            return False
        ish = op.get("input_shapes", "") or ""
        if in_contains is not None and in_contains not in ish:
            return False
        if in_re is not None and not in_re.search(ish):
            return False
        osh = op.get("output_shapes", "") or ""
        if out_contains is not None and out_contains not in osh:
            return False
        if out_re is not None and not out_re.search(osh):
            return False
        return True

    return match


def compile_clusters(cluster_defs):
    """Return [(cluster_name, description, [predicate, ...])] preserving order."""
    out = []
    for entry in cluster_defs:
        name = entry["cluster"]
        desc = entry.get("description", "")
        rules = entry.get("rules", [])
        if not rules:
            raise ValueError(f"cluster {name!r} has no rules")
        preds = [build_match_predicate(r) for r in rules]
        out.append((name, desc, preds))
    return out


def classify(op, compiled):
    """Return cluster_name (first match) or None."""
    for name, _desc, preds in compiled:
        if any(p(op) for p in preds):
            return name
    return None


def _merge_intervals(iv):
    """Merge a list of (lo, hi) intervals into sorted non-overlapping spans."""
    iv = sorted(iv)
    merged = []
    for a, b in iv:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def _covered_us(merged):
    """Sum the lengths of already-merged, non-overlapping intervals (µs).

    Uses a plain running add (not the built-in ``sum``) so the result matches
    the historical accumulation bit-for-bit — CPython 3.12's ``sum`` applies
    Neumaier compensation and would perturb the last ULP of metrics output.
    """
    total = 0.0
    for lo, hi in merged:
        total += hi - lo
    return total


def wall_and_span_us(intervals):
    """Return (wall_us, span_us).

    wall = union of covered time; span = max_end − min_start. Callers that
    only need the union take ``wall_and_span_us(...)[0]``.
    """
    merged = _merge_intervals([iv for iv in intervals if iv[1] > iv[0]])
    if not merged:
        return 0.0, 0.0
    return _covered_us(merged), merged[-1][1] - merged[0][0]


def pair_overlap_us(intervals_a, intervals_b):
    """Total intersection length between two interval lists.

    Each list is already raw (not yet merged). Returns micro-seconds.
    """
    if not intervals_a or not intervals_b:
        return 0.0
    a = _merge_intervals(intervals_a)
    b = _merge_intervals(intervals_b)
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if lo < hi:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def detect_outliers(values, method="iqr", k=1.5, small_n_ratio=1.5):
    """Return set of indices in `values` that are outliers.

    n >= 4: IQR (default) or z-score, unchanged.
    2 <= n <= 3: statistical detectors are unreliable, so fall back to a
      conservative relative-spike rule — flag any value strictly above
      median * small_n_ratio. Model-agnostic (no absolute unit assumption);
      surfaces obvious single spikes (e.g. one layer 3x slower) that IQR
      cannot see at small n. Set small_n_ratio<=0 to disable the fallback.
    n < 2: never flags.
    """
    n = len(values)
    if n < 2:
        return set()
    if n < 4:
        if not small_n_ratio or small_n_ratio <= 0:
            return set()
        med = statistics.median(values)
        if med <= 0:
            return set()
        return {i for i, v in enumerate(values) if v > med * small_n_ratio}
    if method == "z":
        mean = statistics.fmean(values)
        sd = statistics.pstdev(values)
        if sd <= 0:
            return set()
        return {i for i, v in enumerate(values) if abs(v - mean) / sd > k}
    sorted_v = sorted(values)
    q1 = sorted_v[n // 4]
    q3 = sorted_v[(3 * n) // 4]
    iqr = q3 - q1
    if iqr <= 0:
        return {i for i, v in enumerate(values) if v < q1 or v > q3}
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return {i for i, v in enumerate(values) if v < lo or v > hi}


def percentile(values, p):
    if not values:
        return 0.0
    sorted_v = sorted(values)
    if len(sorted_v) == 1:
        return sorted_v[0]
    rank = (p / 100) * (len(sorted_v) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_v) - 1)
    frac = rank - lo
    return sorted_v[lo] * (1 - frac) + sorted_v[hi] * frac


def series_metrics(per_instance_us):
    """median / mean / std / p95 / min / max for a series of µs values, in ms."""
    vals_ms = [v / 1000.0 for v in per_instance_us]
    return {
        "count": len(vals_ms),
        "median_ms": statistics.median(vals_ms) if vals_ms else 0.0,
        "mean_ms": statistics.fmean(vals_ms) if vals_ms else 0.0,
        "std_ms": statistics.pstdev(vals_ms) if len(vals_ms) > 1 else 0.0,
        "p95_ms": percentile(vals_ms, 95),
        "min_ms": min(vals_ms) if vals_ms else 0.0,
        "max_ms": max(vals_ms) if vals_ms else 0.0,
    }


def outliers_for(series_us, instances, method, k):
    """Build (outlier_index_set, outlier_list) for a per-instance µs series."""
    idx = detect_outliers(series_us, method=method, k=k)
    outs = [
        {
            "phase": instances[i]["phase"],
            "layer_idx": instances[i]["layer_idx"],
            "value_ms": series_us[i] / 1000.0,
        }
        for i in sorted(idx)
    ]
    return idx, outs


@dataclass
class _ComponentAccumulator:
    """Per-component running state filled while walking instances.

    Holds the per-cluster series, slot durations, op counts/records, theory
    candidates, and the TOTAL/bubble/overlap/displaced series. All keyed by the
    fixed cluster_order so downstream sub-item building stays deterministic.
    """
    cluster_order: list
    n_inst: int
    per_inst_wall_cluster: dict = field(default_factory=dict)
    cluster_slot_durs: dict = field(default_factory=dict)
    cluster_op_counts: dict = field(default_factory=dict)
    cluster_op_records: dict = field(default_factory=dict)
    cluster_theory_candidates: dict = field(default_factory=dict)
    total_theory_candidates: list = field(default_factory=list)
    total_wall: list = field(default_factory=list)
    total_span: list = field(default_factory=list)
    total_bubble: list = field(default_factory=list)
    unmatched: defaultdict = field(default_factory=lambda: defaultdict(int))
    displaced_wall_by_stream: defaultdict = field(default_factory=lambda: defaultdict(list))
    displaced_count_by_stream: defaultdict = field(default_factory=lambda: defaultdict(int))
    per_inst_sum_minus_total: list = field(default_factory=list)
    pair_keys: list = field(default_factory=list)
    per_inst_pair_overlap: dict = field(default_factory=dict)

    def __post_init__(self):
        order = self.cluster_order
        self.per_inst_wall_cluster = {c: [] for c in order}
        self.cluster_slot_durs = {c: {} for c in order}
        self.cluster_op_counts = {c: [defaultdict(int) for _ in range(self.n_inst)]
                                  for c in order}
        self.cluster_op_records = {c: [] for c in order}
        self.cluster_theory_candidates = {c: [] for c in order}
        self.pair_keys = [(order[i], order[j])
                          for i in range(len(order))
                          for j in range(i + 1, len(order))]
        self.per_inst_pair_overlap = {p: [] for p in self.pair_keys}


@dataclass
class RenderContext:
    """Pipeline-wide inputs shared across every component section.

    Bundles the operator list, outlier config and the optional theory /
    decision inputs so the section-building call chain stays under the argument
    limit; behaviour is identical to threading the fields individually.
    """
    operators: list
    outlier_cfg: dict
    details_by_idx: dict = None
    theory_decisions: dict = None
    review_ratio: float = 0.8


@dataclass
class _TheoryContext:
    """Per-component inputs for theory-candidate aggregation."""
    component_type: str
    operators: list
    details_by_idx: dict
    decisions: dict
    review_ratio: float


@dataclass
class _TheorySlot:
    """One (sub_item, instance) slot fed into theory aggregation."""
    sub_item: str
    inst_idx: int
    inst: dict
    op_indices: list


@dataclass
class _AnalysisContext:
    """Per-component invariants threaded through instance accumulation."""
    operators: list
    compiled: list
    acc: "_ComponentAccumulator"
    theory: "_TheoryContext" = None


def _accumulate_displaced(inst, operators, acc):
    """Tally displaced aux ops into acc, returning the displaced index set."""
    displaced_idx = {int(i) for i in inst.get("displaced_op_indices") or []}
    inst_disp_ivs = defaultdict(list)
    for i in displaced_idx:
        op = operators[i]
        st, dur = op.get("start_time_us"), op.get("duration_us", 0.0)
        if st is None or dur is None:
            continue
        sid = str(op.get("stream_id"))
        inst_disp_ivs[sid].append((st, st + dur))
        acc.displaced_count_by_stream[sid] += 1
    for sid, ivs in inst_disp_ivs.items():
        wall, _ = wall_and_span_us(ivs)
        acc.displaced_wall_by_stream[sid].append(wall)
    return displaced_idx


def _classify_instance_ops(inst_idx, inst, inst_op_indices, actx):
    """Classify one instance's ops into clusters, updating actx.acc.

    Returns (by_cluster intervals, by_cluster_op_indices, all_intervals,
    all_op_indices) for this instance.
    """
    operators = actx.operators
    compiled = actx.compiled
    acc = actx.acc
    by_cluster = defaultdict(list)
    by_cluster_op_indices = defaultdict(list)
    all_intervals = []
    all_op_indices = []
    occ_counter = defaultdict(lambda: defaultdict(int))  # cluster -> op_name -> next occ
    for i in inst_op_indices:
        op = operators[i]
        st = op.get("start_time_us")
        dur = op.get("duration_us", 0.0)
        if st is None or dur is None:
            continue
        iv = (st, st + dur)
        all_intervals.append(iv)
        all_op_indices.append(i)
        cname = classify(op, compiled)
        if cname is None:
            acc.unmatched[op.get("normalized_name", "?")] += 1
            continue
        by_cluster[cname].append(iv)
        by_cluster_op_indices[cname].append(i)
        op_name = op.get("normalized_name", "?")
        acc.cluster_op_counts[cname][inst_idx][op_name] += 1
        occ = occ_counter[cname][op_name]
        occ_counter[cname][op_name] += 1
        slot = (op_name, occ)
        if slot not in acc.cluster_slot_durs[cname]:
            acc.cluster_slot_durs[cname][slot] = [None] * acc.n_inst
        acc.cluster_slot_durs[cname][slot][inst_idx] = dur
        acc.cluster_op_records[cname].append({
            "inst_idx": inst_idx,
            "op_idx": op.get("index", i),
            "phase": inst["phase"],
            "layer_idx": inst["layer_idx"],
            "op_name": op_name,
            "occurrence": occ,
        })
    return by_cluster, by_cluster_op_indices, all_intervals, all_op_indices


def _accumulate_instance(inst_idx, inst, actx):
    """Process one instance end-to-end, appending its series into actx.acc."""
    acc = actx.acc
    displaced_idx = _accumulate_displaced(inst, actx.operators, acc)
    inst_op_indices = [i for i in op_indices_from_instance(inst)
                       if i not in displaced_idx]
    by_cluster, by_cluster_op_indices, all_intervals, all_op_indices = (
        _classify_instance_ops(inst_idx, inst, inst_op_indices, actx)
    )

    sum_walls = 0.0
    for c in acc.cluster_order:
        wall_c, _ = wall_and_span_us(by_cluster.get(c, []))
        acc.per_inst_wall_cluster[c].append(wall_c)
        sum_walls += wall_c
    wall_total, span_total = wall_and_span_us(all_intervals)
    acc.total_wall.append(wall_total)
    acc.total_span.append(span_total)
    acc.total_bubble.append(max(0.0, span_total - wall_total))
    acc.per_inst_sum_minus_total.append(max(0.0, sum_walls - wall_total))
    for (a, b) in acc.pair_keys:
        acc.per_inst_pair_overlap[(a, b)].append(
            pair_overlap_us(by_cluster.get(a, []), by_cluster.get(b, []))
        )
    if actx.theory is not None:
        for c in acc.cluster_order:
            acc.cluster_theory_candidates[c].append(build_theory_candidate(
                _TheorySlot(c, inst_idx, inst, by_cluster_op_indices.get(c, [])),
                actx.theory,
            ))
        acc.total_theory_candidates.append(build_theory_candidate(
            _TheorySlot("TOTAL", inst_idx, inst, all_op_indices),
            actx.theory,
        ))


def _kernel_outliers_for_cluster(cname, instances, acc, method, k):
    """Per-slot duration outliers across instances for one cluster."""
    kernel_outliers = []
    for (op_name, occ), durs in acc.cluster_slot_durs[cname].items():
        valid_idx = [i for i, d in enumerate(durs) if d is not None]
        if len(valid_idx) < 4:
            continue
        valid_vals = [durs[i] for i in valid_idx]
        slot_out_pos = detect_outliers(valid_vals, method=method, k=k)
        if not slot_out_pos:
            continue
        baseline = statistics.median(valid_vals)
        for pos in sorted(slot_out_pos):
            ii = valid_idx[pos]
            kernel_outliers.append({
                "op_name": op_name,
                "occurrence": occ,
                "instance_idx": ii,
                "phase": instances[ii]["phase"],
                "layer_idx": instances[ii]["layer_idx"],
                "duration_us": durs[ii],
                "baseline_median_us": baseline,
            })
    kernel_outliers.sort(key=lambda x: -abs(x["duration_us"] - x["baseline_median_us"]))
    return kernel_outliers


def _kernel_count_anomalies_for_cluster(cname, instances, acc):
    """Per-op-name count anomalies (extras / missing) for one cluster."""
    kernel_count_anomalies = []
    counts_per_inst = acc.cluster_op_counts[cname]
    all_names = set()
    for d in counts_per_inst:
        all_names.update(d.keys())
    for op_name in sorted(all_names):
        vec = [d.get(op_name, 0) for d in counts_per_inst]
        ctr = Counter(vec)
        modal_count, modal_freq = ctr.most_common(1)[0]
        if modal_freq == acc.n_inst:
            continue
        for ii, cnt in enumerate(vec):
            if cnt == modal_count:
                continue
            kernel_count_anomalies.append({
                "op_name": op_name,
                "instance_idx": ii,
                "phase": instances[ii]["phase"],
                "layer_idx": instances[ii]["layer_idx"],
                "count": cnt,
                "modal_count": modal_count,
            })
    kernel_count_anomalies.sort(
        key=lambda x: (-abs(x["count"] - x["modal_count"]),
                       x["op_name"], x["instance_idx"])
    )
    return kernel_count_anomalies


def _build_cluster_sub_items(compiled, instances, acc, method, k):
    """Build the per-cluster sub-item dicts in spec order."""
    sub_items = []
    for cname, desc, _preds in compiled:
        series = acc.per_inst_wall_cluster[cname]
        idx, outs = outliers_for(series, instances, method, k)
        sub_items.append({
            "name": cname,
            "kind": "cluster",
            "description": desc,
            "metric": series_metrics(series),
            "outliers": outs,
            "outlier_idx": sorted(idx),
            "per_instance_ms": [v / 1000.0 for v in series],
            "kernel_outliers": _kernel_outliers_for_cluster(cname, instances, acc, method, k),
            "kernel_count_anomalies": _kernel_count_anomalies_for_cluster(cname, instances, acc),
            "op_records": acc.cluster_op_records[cname],
            "theoretical": summarize_theory(acc.cluster_theory_candidates[cname]),
        })
    return sub_items


def _build_overlap_summary(acc):
    """Cross-cluster overlap summary: gap vs union + top pairwise overlaps."""
    per_inst_sum_minus_total = acc.per_inst_sum_minus_total
    total_span = acc.total_span
    med_gap = statistics.median(per_inst_sum_minus_total) if per_inst_sum_minus_total else 0.0
    max_gap = max(per_inst_sum_minus_total) if per_inst_sum_minus_total else 0.0
    med_total = statistics.median(total_span) if total_span else 0.0
    pair_summaries = []
    for (a, b), vals in acc.per_inst_pair_overlap.items():
        if not vals:
            continue
        med = statistics.median(vals)
        if med <= 0:
            continue
        pair_summaries.append({
            "cluster_a": a,
            "cluster_b": b,
            "median_overlap_ms": med / 1000.0,
            "median_overlap_pct": (med / med_total * 100) if med_total else 0.0,
            "max_overlap_ms": max(vals) / 1000.0,
        })
    pair_summaries.sort(key=lambda p: -p["median_overlap_ms"])
    return {
        "median_gap_ms": med_gap / 1000.0,
        "median_gap_pct": (med_gap / med_total * 100) if med_total else 0.0,
        "max_gap_ms": max_gap / 1000.0,
        "top_pairs": pair_summaries[:8],
    }


def _build_displaced_summary(acc):
    """Per-stream summary of displaced aux ops, as % of median TOTAL span."""
    med_total_span = statistics.median(acc.total_span) if acc.total_span else 0.0
    summary = []
    for sid, walls in sorted(acc.displaced_wall_by_stream.items()):
        med_wall = statistics.median(walls) if walls else 0.0
        summary.append({
            "stream_id": sid,
            "op_count": acc.displaced_count_by_stream[sid],
            "median_wall_ms": med_wall / 1000.0,
            "pct_of_total": ((med_wall / med_total_span * 100)
                             if walls and med_total_span else 0.0),
        })
    return summary


def analyze_component_type(component_type, instances, compiled, ctx):
    """For each instance: classify ops, compute per-cluster wall + layer bubble.

    Returns list of sub-items (clusters in spec order, then synthetic 'bubble')
    plus a 'total' dict (layer wall) and an 'unmatched' counter.
    Each sub-item: {name, kind, description, metric, outliers, per_instance_ms,
                    outlier_idx, kernel_outliers}
      outlier_idx flags instance indices whose cluster-wall is an outlier.
      kernel_outliers flags individual kernels: each cluster collects every op,
      slotted by (op_name, occurrence_within_instance). Each slot's durations
      across instances run through detect_outliers; flagged kernels list their
      phase/layer + duration vs slot median.
      kernel_count_anomalies flags instances whose op-name count deviates from
      the modal count across the cluster (extras / missing ops per layer).
    """
    method = ctx.outlier_cfg.get("method", "iqr")
    k = ctx.outlier_cfg.get("k", 1.5)
    cluster_order = [name for name, _d, _p in compiled]
    acc = _ComponentAccumulator(cluster_order=cluster_order, n_inst=len(instances))
    theory_present = _has_theory_columns(ctx.details_by_idx)   # 理论性能可选；缺则列留空
    theory = (
        _TheoryContext(component_type, ctx.operators, ctx.details_by_idx,
                       ctx.theory_decisions or {}, ctx.review_ratio)
        if theory_present else None
    )
    actx = _AnalysisContext(operators=ctx.operators, compiled=compiled,
                            acc=acc, theory=theory)

    for inst_idx, inst in enumerate(instances):
        _accumulate_instance(inst_idx, inst, actx)

    sub_items = _build_cluster_sub_items(compiled, instances, acc, method, k)

    idx, outs = outliers_for(acc.total_bubble, instances, method, k)
    sub_items.append({
        "name": "bubble",
        "kind": "bubble",
        "description": "layer idle gap (span − wall)",
        "metric": series_metrics(acc.total_bubble),
        "outliers": outs,
        "outlier_idx": sorted(idx),
        "per_instance_ms": [v / 1000.0 for v in acc.total_bubble],
    })

    idx, outs = outliers_for(acc.total_span, instances, method, k)
    total = {
        "description": "layer end-to-end span (max_end − min_start, 含 bubble)",
        "metric": series_metrics(acc.total_span),
        "outliers": outs,
        "outlier_idx": sorted(idx),
        "per_instance_ms": [v / 1000.0 for v in acc.total_span],
        "theoretical": summarize_theory(acc.total_theory_candidates),
    }

    overlap_summary = _build_overlap_summary(acc)
    displaced_summary = _build_displaced_summary(acc)
    return sub_items, total, dict(acc.unmatched), overlap_summary, displaced_summary


def fmt_duration_us(value_us):
    if value_us is None:
        return "—"
    v = float(value_us)
    return f"{v:.1f} µs" if abs(v) < 1000.0 else f"{v / 1000.0:.3f} ms"


def fmt_duration_ms(value_ms):
    if value_ms is None:
        return "—"
    return fmt_duration_us(float(value_ms) * 1000.0)


def fmt_outliers(outs):
    return ", ".join(
        f"{o['phase']}#{o['layer_idx']} ({fmt_duration_ms(o['value_ms'])})"
        for o in outs
    ) or "—"


def _has_theory_columns(details_by_idx):
    """理论列是否存在（Step 1.5 是否跑过）。理论性能是可选的——外部
    operator-theoretical-perf skill 可能不存在；没跑则 render 把理论列留空，不阻塞。"""
    if not details_by_idx:
        return False
    return any("theoretical_operator_time_us" in d for d in details_by_idx.values())


def _detail_theory_us(detail):
    if not detail:
        return None
    supported = detail.get("theory_supported")
    if isinstance(supported, str) and supported.strip().lower() in {
        "false", "0", "no", "n/a", "unsupported",
    }:
        return None
    if supported is False:
        return None
    return _to_float(detail.get("theoretical_operator_time_us"))


def load_theory_decisions(path):
    if not path:
        return {}
    data = load_json(path)
    decisions = data.get("decisions", data if isinstance(data, list) else [])
    out = {}
    for item in decisions:
        # Omit BOTH phase and layer_idx (→ None) for a slot-wide wildcard that
        # covers every instance of (component_type, sub_item); lookup tries the
        # exact (phase,layer) key first, then the (None,None) wildcard. Only
        # component_type + sub_item are mandatory.
        key = (
            item.get("component_type"),
            item.get("sub_item"),
            item.get("phase"),
            item.get("layer_idx"),
        )
        if key[0] is None or key[1] is None:
            continue
        decision = {
            "reason": item.get("reason", "agent annotation"),
            "semantic_note": item.get("semantic_note"),
            "stream_semantics": item.get("stream_semantics") or {},
        }
        if item.get("selected_stream") is not None:
            decision["selected_stream"] = str(item.get("selected_stream"))
        out[key] = decision
    return out


def _aggregate_theory_streams(op_indices, operators, details_by_idx):
    """Bucket ops by stream_id and tally theory/observed-union → (rows, missing)."""
    streams = defaultdict(lambda: {
        "op_count": 0,
        "supported_count": 0,
        "unsupported_count": 0,
        "theoretical_sum_us": 0.0,
        "intervals": [],
    })
    missing_theory = 0
    for i in op_indices:
        op = operators[i]
        stream = str(op.get("stream_id", "unknown"))
        bucket = streams[stream]
        bucket["op_count"] += 1
        st = op.get("start_time_us")
        dur = op.get("duration_us")
        if st is not None and dur is not None:
            bucket["intervals"].append((float(st), float(st) + float(dur)))
        detail = details_by_idx.get(op.get("index", i))
        theory_us = _detail_theory_us(detail)
        if theory_us is not None:
            bucket["supported_count"] += 1
            bucket["theoretical_sum_us"] += theory_us
        else:
            bucket["unsupported_count"] += 1
            missing_theory += 1

    stream_rows = []
    for stream, row in streams.items():
        op_count = row["op_count"]
        supported = row["supported_count"]
        stream_rows.append({
            "stream_id": stream,
            "op_count": op_count,
            "supported_count": supported,
            "unsupported_count": row["unsupported_count"],
            "supported_pct": supported / op_count * 100.0 if op_count else 0.0,
            "observed_union_us": wall_and_span_us(row["intervals"])[0],
            "theoretical_sum_us": row["theoretical_sum_us"] if supported else None,
        })
    stream_rows.sort(key=lambda x: x["observed_union_us"], reverse=True)
    return stream_rows, missing_theory


def _select_theory_stream(stream_rows, decision, review_ratio):
    """Pick the stream row → (selected, selection_source, warning, needs_review)."""
    selected = stream_rows[0] if stream_rows else None
    selection_source = "script_max_observed_union_us"
    selection_warning = None
    if decision and decision.get("selected_stream") is not None:
        requested = decision["selected_stream"]
        match = next((r for r in stream_rows if r["stream_id"] == requested), None)
        if match:
            selected = match
            selection_source = "agent_decision"
        else:
            selection_warning = f"agent selected missing stream {requested!r}"

    needs_agent_review = False
    if len(stream_rows) >= 2 and stream_rows[0]["observed_union_us"] > 0:
        ratio = stream_rows[1]["observed_union_us"] / stream_rows[0]["observed_union_us"]
        needs_agent_review = ratio >= review_ratio and selection_source != "agent_decision"
    return selected, selection_source, selection_warning, needs_agent_review


def build_theory_candidate(slot, ctx):
    """Aggregate per-stream theory for one (component, sub_item, instance) slot.

    slot is a _TheorySlot (sub_item + instance); ctx is a _TheoryContext holding
    the component type and the shared operators/details/decisions inputs.
    """
    if not ctx.details_by_idx:
        return None
    inst = slot.inst
    stream_rows, missing_theory = _aggregate_theory_streams(
        slot.op_indices, ctx.operators, ctx.details_by_idx
    )

    # exact (ct, sub, phase, layer) → full wildcard (ct, sub, None, None).
    # The wildcard lets one decision cover all instances of a slot.
    decision = (
        ctx.decisions.get((ctx.component_type, slot.sub_item,
                           inst.get("phase"), inst.get("layer_idx")))
        or ctx.decisions.get((ctx.component_type, slot.sub_item, None, None))
    )
    selected, selection_source, selection_warning, needs_agent_review = (
        _select_theory_stream(stream_rows, decision, ctx.review_ratio)
    )

    theory_us = selected.get("theoretical_sum_us") if selected else None
    return {
        "instance_idx": slot.inst_idx,
        "phase": inst.get("phase"),
        "layer_idx": inst.get("layer_idx"),
        "selected_stream": selected.get("stream_id") if selected else None,
        "selection_source": selection_source,
        "selection_reason": (
            decision.get("reason") if decision else "largest observed timeline union"
        ),
        "selection_warning": selection_warning,
        "needs_agent_review": needs_agent_review,
        "multi_stream": len(stream_rows) > 1,
        "semantic_note": decision.get("semantic_note") if decision else None,
        "stream_semantics": decision.get("stream_semantics") if decision else {},
        "effective_theoretical_us": theory_us,
        "missing_theory_count": missing_theory,
        "stream_candidates": stream_rows,
    }


def summarize_theory(candidates):
    candidates = [c for c in candidates if c is not None]
    if not candidates:
        return None
    theory_values = [
        c["effective_theoretical_us"]
        for c in candidates
        if c.get("effective_theoretical_us") is not None
    ]
    return {
        "median_theoretical_ms": (
            statistics.median(theory_values) / 1000.0 if theory_values else None
        ),
        "theoretical_metric": series_metrics(theory_values) if theory_values else None,
        "supported_instance_count": len(theory_values),
        "instance_count": len(candidates),
        "needs_agent_review_count": sum(
            1 for c in candidates if c.get("needs_agent_review")
        ),
        "multi_stream_count": sum(1 for c in candidates if c.get("multi_stream")),
        "missing_semantic_note_count": sum(
            1 for c in candidates
            if c.get("multi_stream") and not c.get("semantic_note")
        ),
        "semantic_notes": sorted({
            c.get("semantic_note")
            for c in candidates
            if c.get("multi_stream") and c.get("semantic_note")
        }),
        "missing_theory_count": sum(c.get("missing_theory_count", 0) for c in candidates),
        "per_instance": candidates,
    }


def render_theory_cell(summary, actual_median_ms):
    if not summary:
        return "—"
    med = summary.get("median_theoretical_ms")
    if med is None:
        return "<span class='muted'>N/A</span>"
    ratio = None
    if actual_median_ms and med > 0:
        ratio = actual_median_ms / med
    elif summary.get("actual_over_theoretical_median") is not None:
        ratio = summary["actual_over_theoretical_median"]

    notes = []
    inst_n = summary.get("instance_count")
    supported_n = summary.get("supported_instance_count")
    if inst_n is not None and supported_n is not None and supported_n < inst_n:
        notes.append(f"{supported_n}/{inst_n} inst")
    review_n = summary.get("needs_agent_review_count") or 0
    if review_n:
        notes.append(f"{review_n} review")
    multi_n = summary.get("multi_stream_count") or 0
    if multi_n:
        notes.append(f"multi-stream {multi_n} inst")
        semantic_notes = summary.get("semantic_notes") or []
        if semantic_notes:
            notes.extend(semantic_notes[:2])
            if len(semantic_notes) > 2:
                notes.append(f"+{len(semantic_notes) - 2} notes")
        missing_sem_n = summary.get("missing_semantic_note_count") or 0
        if missing_sem_n:
            notes.append(f"{missing_sem_n} semantic note required")
    miss_n = summary.get("missing_theory_count") or 0
    if miss_n:
        notes.append(f"{miss_n} unsupported")

    ratio_html = f"<span class='theory-ratio'>wall/theory {ratio:.2f}x</span>" if ratio else ""
    note_html = f"<span class='theory-note'>{html.escape('; '.join(notes))}</span>" if notes else ""
    return (
        f"<span class='theory-main'>{fmt_duration_ms(med)}</span>"
        f"{ratio_html}{note_html}"
    )


def _insight_confidence(review_or_record):
    if not review_or_record:
        return None
    conf = review_or_record.get("confidence")
    if conf is None and isinstance(review_or_record.get("agent_review"), dict):
        conf = review_or_record["agent_review"].get("confidence")
    if conf is None:
        return None
    text = str(conf).strip().lower()
    return text if text in {"high", "medium"} else None


def _insight_summary(record):
    for key in ("summary", "semantic_summary", "reason"):
        val = record.get(key)
        if val:
            return str(val)
    return ""


def _row_insight_targets(record):
    raw_targets = []
    if isinstance(record.get("target"), dict):
        raw_targets.append(record["target"])
    elif isinstance(record.get("targets"), list):
        raw_targets.extend(t for t in record["targets"] if isinstance(t, dict))
    else:
        raw_targets.append(record)

    targets = []
    for target in raw_targets:
        component_type = (
            target.get("component_type")
            or target.get("primary_component_type")
            or record.get("component_type")
            or record.get("primary_component_type")
        )
        sub_item = (
            target.get("sub_item")
            or target.get("cluster")
            or target.get("primary_sub_item")
            or record.get("sub_item")
            or record.get("cluster")
            or record.get("primary_sub_item")
        )
        if not component_type or not sub_item:
            continue
        targets.append({
            "component_type": component_type,
            "sub_item": sub_item,
            "mapping_type": (
                target.get("mapping_type")
                or record.get("mapping_type")
                or "direct"
            ),
            "mapping_note": target.get("mapping_note") or record.get("mapping_note") or "",
            "related_targets": (
                target.get("related_targets")
                or record.get("related_targets")
                or []
            ),
        })
    return targets


def _format_related_targets(targets):
    if not isinstance(targets, list):
        return ""
    labels = []
    for target in targets:
        if not isinstance(target, dict):
            continue
        component_type = target.get("component_type")
        sub_item = target.get("sub_item") or target.get("cluster")
        if component_type and sub_item:
            labels.append(f"{component_type}/{sub_item}")
    return ", ".join(labels)


def _add_row_insight(lookup, record):
    conf = _insight_confidence(record)
    if conf not in {"high", "medium"}:
        return
    targets = _row_insight_targets(record)
    if not targets:
        return
    summary = _insight_summary(record)
    if not summary:
        return
    evidence = record.get("evidence")
    if isinstance(evidence, (dict, list)):
        evidence = json.dumps(evidence, ensure_ascii=False)
    source = record.get("source", "agent")
    category = (
        record.get("category")
        or record.get("insight_category")
        or (os.path.splitext(os.path.basename(str(source)))[0] if source else "")
    )
    for target in targets:
        lookup[(target["component_type"], target["sub_item"])].append({
            "confidence": conf,
            "summary": summary,
            "category": str(category) if category else "",
            "source": source,
            "evidence": str(evidence) if evidence else "",
            "mapping_type": target.get("mapping_type") or "direct",
            "mapping_note": target.get("mapping_note") or "",
            "related_targets": _format_related_targets(target.get("related_targets")),
        })


def load_agent_insight_annotations(path):
    """Load agent-authored main report insight annotations.

    The renderer does not infer insight semantics from Step 5 JSONs. The agent
    explicitly decides which rows deserve a high/medium note and writes:

      {"items": [{"target": {"component_type": "moe", "sub_item": "expert",
                             "mapping_type": "direct"},
                  "category": "operator_jitter", "confidence": "high",
                  "summary": "...", "evidence": "..."}]}

    Initial Step 3 render passes no path, so the final column remains blank.
    """
    lookup = defaultdict(list)
    if not path:
        return lookup
    data = load_json(path)
    records = data.get("items", data if isinstance(data, list) else [])
    for rec in records:
        _add_row_insight(lookup, rec)

    order = {"high": 0, "medium": 1}
    for key, vals in lookup.items():
        vals.sort(key=lambda x: (
            order.get(x["confidence"], 9),
            x.get("category") or "",
            x["summary"],
        ))
        lookup[key] = vals
    return lookup


def render_insight_cell(items):
    if not items:
        return ""
    parts = []
    for item in items:
        conf = html.escape(item["confidence"])
        summary = html.escape(item["summary"])
        category = html.escape(item.get("category") or "")
        source = html.escape(item.get("source") or "")
        evidence = html.escape(item.get("evidence") or "")
        mapping_type = item.get("mapping_type") or "direct"
        mapping_note = item.get("mapping_note") or ""
        related_targets = item.get("related_targets") or ""
        category_html = (
            f"<span class='insight-category'>{category}</span> " if category else ""
        )
        evidence_html = f"<span class='insight-evidence'>{evidence}</span>" if evidence else ""
        mapping_bits = []
        if mapping_type != "direct":
            mapping_bits.append(f"mapping: {mapping_type}")
        if mapping_note:
            mapping_bits.append(mapping_note)
        if related_targets:
            mapping_bits.append(f"related: {related_targets}")
        mapping_html = (
            f"<span class='insight-mapping'>{html.escape('; '.join(mapping_bits))}</span>"
            if mapping_bits else ""
        )
        parts.append(
            f"<div class='insight-note {conf}'>"
            f"<span class='insight-conf'>{conf}</span> {category_html}{summary}"
            f"<span class='insight-source'>{source}</span>{mapping_html}{evidence_html}"
            f"</div>"
        )
    return "".join(parts)


def _op_status_tags(rec, dur_anom_map, count_anom_map):
    """Status tags + row class for one op record. Returns (tags, row_cls)."""
    tags = []
    row_cls = ""
    key = (rec["op_name"], rec["occurrence"])
    if key in dur_anom_map:
        da = dur_anom_map[key]
        base = da["baseline_us"]
        dur = da["duration_us"]
        is_slow = dur > base
        verb = "偏慢" if is_slow else "偏快"
        pct = ((dur - base) / base * 100.0) if base > 0 else 0.0
        tag_cls = "status slow" if is_slow else "status fast"
        tags.append(
            f"<span class='{tag_cls}'>{verb} ({pct:+.0f}% vs 中位 "
            f"{fmt_duration_us(base)})</span>"
        )
        row_cls = "row-slow" if is_slow else "row-fast"
    ca = count_anom_map.get(rec["op_name"])
    if ca and ca["count"] > ca["modal_count"] and \
            rec["occurrence"] >= ca["modal_count"]:
        tags.append(
            f"<span class='status extra'>多出 "
            f"({ca['count']} vs 多数层 {ca['modal_count']})</span>"
        )
        if not row_cls:
            row_cls = "row-extra"
    return tags, row_cls


def _op_data_row(rec, op, columns, dur_anom_map, count_anom_map):
    """Render one op record row (status column + CSV columns)."""
    tags, row_cls = _op_status_tags(rec, dur_anom_map, count_anom_map)
    status_cell = " ".join(tags) if tags else "—"
    cells = [f"<td class='status-cell'>{status_cell}</td>"]
    for c in columns:
        v = op.get(c, "")
        if v is None:
            v = ""
        elif isinstance(v, float):
            v = f"{v:.6g}"
        else:
            v = str(v)
        cells.append(f"<td>{html.escape(v)}</td>")
    row_attr = f" class='{row_cls}'" if row_cls else ""
    return f"<tr{row_attr}>" + "".join(cells) + "</tr>"


def _missing_op_rows(count_anom_map, columns):
    """Synthetic rows for ops missing/under-counted vs the modal layer."""
    body_rows = []
    for op_name, ca in count_anom_map.items():
        if ca["count"] >= ca["modal_count"]:
            continue
        missing_n = ca["modal_count"] - ca["count"]
        verb = "缺失" if ca["count"] == 0 else "少出"
        tag = (
            f"<span class='status missing'>{verb} {missing_n} 个 "
            f"(本层 {ca['count']} / 多数层 {ca['modal_count']})</span>"
        )
        cells = [f"<td class='status-cell'>{tag}</td>"]
        for c in columns:
            if c == "name":
                cells.append(f"<td>{html.escape(op_name)}</td>")
            else:
                cells.append("<td class='missing-cell'>—</td>")
        body_rows.append("<tr class='row-missing'>" + "".join(cells) + "</tr>")
    return body_rows


def render_inst_op_table(op_records, details_by_idx, dur_anom_map, count_anom_map):
    """Per-instance op table with a leading status column.

    op_records: list of {op_idx, op_name, occurrence, ...} for one instance.
    dur_anom_map: {(op_name, occurrence): {duration_us, baseline_us}} flagged
        as duration outliers within this cluster, scoped to this instance.
    count_anom_map: {op_name: {count, modal_count}} for this instance — covers
        both 多出 (count > modal) and 缺失/少出 (count < modal). Missing ops
        are not in op_records, so we emit synthetic rows at the end.
    """
    if not op_records or not details_by_idx:
        return ""
    rows = []
    columns = None
    for rec in op_records:
        op = details_by_idx.get(rec["op_idx"])
        if op is None:
            continue
        if columns is None:
            columns = list(op.keys())
        rows.append((rec, op))
    if not rows or not columns:
        return ""

    header_cells = "<th class='status-col'>状态</th>" + "".join(
        f"<th>{html.escape(c)}</th>" for c in columns
    )

    body_rows = [
        _op_data_row(rec, op, columns, dur_anom_map, count_anom_map)
        for rec, op in rows
    ]
    body_rows.extend(_missing_op_rows(count_anom_map, columns))

    return (
        "<div class='subwrap'><table class='subtbl'>"
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )


@dataclass
class _ChipSpec:
    """Inputs for rendering one sub-item's per-instance chip row."""
    per_inst: list
    outlier_idx: list = field(default_factory=list)
    kernel_outliers: list = None
    kernel_count_anomalies: list = None
    op_records: list = None
    median_ms: float = None


def _chip_ref_median(per_inst, median_ms):
    """Reference median for chip tiering: explicit median or non-zero median."""
    ref_median = median_ms
    if ref_median is None or ref_median <= 0:
        nz = sorted(v for v in per_inst if v > 0)
        if nz:
            mid = len(nz) // 2
            ref_median = nz[mid] if len(nz) % 2 else (nz[mid - 1] + nz[mid]) / 2
    return ref_median


def _chip_outlier_class(v, ref_median):
    """(css class, tag html) for an outlier chip given its value vs median."""
    if ref_median is not None and ref_median > 0:
        dev = abs(v - ref_median) / ref_median
    else:
        dev = 0.0
    if dev >= 0.5:
        tier = "tier-3"
    elif dev >= 0.2:
        tier = "tier-2"
    else:
        tier = "tier-1"
    if ref_median is not None and ref_median > 0 and v < ref_median:
        return f"chip outlier-fast {tier}", "<span class='tag'>异常·快</span>"
    return f"chip outlier-slow {tier}", "<span class='tag'>异常·慢</span>"


def _chip_anom_note(dur_anoms, count_anoms):
    """Compact slow/fast/extra/missing count note for one instance's chip."""
    slow_n = sum(1 for da in dur_anoms.values() if da["duration_us"] > da["baseline_us"])
    fast_n = sum(1 for da in dur_anoms.values() if da["duration_us"] < da["baseline_us"])
    extra_n = sum(1 for ca in count_anoms.values() if ca["count"] > ca["modal_count"])
    missing_n = sum(1 for ca in count_anoms.values() if ca["count"] < ca["modal_count"])
    anom_parts = []
    if slow_n:
        anom_parts.append(f"<span class='hot'>{slow_n}慢</span>")
    if fast_n:
        anom_parts.append(f"<span class='cool'>{fast_n}快</span>")
    if extra_n:
        anom_parts.append(f"<span class='hot'>{extra_n}多</span>")
    if missing_n:
        anom_parts.append(f"<span class='cool'>{missing_n}缺</span>")
    return (
        f"<span class='anom-note'>{' '.join(anom_parts)}</span>"
        if anom_parts else ""
    )


def _index_chip_anomalies(spec, details_by_idx):
    """Build per-instance lookup maps for chip records and anomalies."""
    expandable = bool(spec.op_records) and bool(details_by_idx)
    records_by_inst = defaultdict(list)
    dur_anom_by_inst = defaultdict(dict)
    count_anom_by_inst = defaultdict(dict)
    if expandable:
        for rec in spec.op_records:
            records_by_inst[rec["inst_idx"]].append(rec)
    for ko in spec.kernel_outliers or []:
        dur_anom_by_inst[ko["instance_idx"]][(ko["op_name"], ko["occurrence"])] = {
            "duration_us": ko["duration_us"],
            "baseline_us": ko["baseline_median_us"],
        }
    for ka in spec.kernel_count_anomalies or []:
        count_anom_by_inst[ka["instance_idx"]][ka["op_name"]] = {
            "count": ka["count"],
            "modal_count": ka["modal_count"],
        }
    return expandable, records_by_inst, dur_anom_by_inst, count_anom_by_inst


def _render_chip(i, v, meta_i, ctx):
    """Render a single instance chip. ctx carries shared chip-row state."""
    is_out = i in ctx["outlier_set"]
    ref_median = ctx["ref_median"]
    if is_out:
        cls, tag = _chip_outlier_class(v, ref_median)
    else:
        cls, tag = "chip", ""
    max_v = ctx["max_v"]
    bar_pct = (v / max_v * 100.0) if max_v > 0 else 0.0
    bar = f"<span class='bar' style='width:{bar_pct:.1f}%'></span>"
    med_note = ""
    if is_out and ref_median is not None and ref_median > 0:
        pct = (v - ref_median) / ref_median * 100.0
        med_note = (
            f"<span class='med-note'>vs 中位 {fmt_duration_ms(ref_median)}"
            f" ({pct:+.0f}%)</span>"
        )
    dur_anoms = ctx["dur_anom_by_inst"].get(i, {})
    count_anoms = ctx["count_anom_by_inst"].get(i, {})
    anom_note = _chip_anom_note(dur_anoms, count_anoms)
    summary_inner = (
        f"{bar}{tag}"
        f"<span class='lbl'>{html.escape(meta_i['phase'])}#{meta_i['layer_idx']}</span>"
        f"<span class='val'>{fmt_duration_ms(v)}</span>{med_note}{anom_note}"
    )
    expandable = ctx["expandable"]
    inst_records = ctx["records_by_inst"].get(i) if expandable else None
    inst_has_anom = expandable and (
        i in ctx["dur_anom_by_inst"] or i in ctx["count_anom_by_inst"]
    )
    if inst_records or inst_has_anom:
        panel = render_inst_op_table(
            inst_records or [], ctx["details_by_idx"], dur_anoms, count_anoms,
        )
        return (
            f"<details class='chip-d'>"
            f"<summary class='{cls}'>{summary_inner}</summary>"
            f"<div class='panel'>{panel}</div>"
            f"</details>"
        )
    return f"<span class='{cls}'>{summary_inner}</span>"


def chip_cell(spec, meta, details_by_idx):
    """Render a sub-item's full per-instance chip list as an HTML fragment."""
    expandable, records_by_inst, dur_anom_by_inst, count_anom_by_inst = (
        _index_chip_anomalies(spec, details_by_idx)
    )
    ctx = {
        "outlier_set": set(spec.outlier_idx or []),
        "max_v": max(spec.per_inst) if spec.per_inst else 0.0,
        "ref_median": _chip_ref_median(spec.per_inst, spec.median_ms),
        "expandable": expandable,
        "records_by_inst": records_by_inst,
        "dur_anom_by_inst": dur_anom_by_inst,
        "count_anom_by_inst": count_anom_by_inst,
        "details_by_idx": details_by_idx,
    }
    chips = []
    prev_phase = None
    for i, v in enumerate(spec.per_inst):
        meta_i = meta[i]
        if prev_phase is not None and meta_i["phase"] != prev_phase:
            chips.append("<span class='phasebrk'></span>")
        prev_phase = meta_i["phase"]
        chips.append(_render_chip(i, v, meta_i, ctx))
    return "<div class='chips'>" + "".join(chips) + "</div>"


_CSS = """
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           color: #222; max-width: 1400px; margin: 24px auto; padding: 0 16px; }
    h1 { margin-bottom: 4px; }
    .meta { color: #666; font-size: 13px; margin-bottom: 24px; }
    .muted { color: #888; }
    h2 { margin-top: 32px; border-bottom: 2px solid #ddd; padding-bottom: 4px; }
    h2 small { color: #888; font-weight: normal; font-size: 14px; }
    table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13px;
            table-layout: fixed; }
    col.sub { width: 130px; }
    col.desc { width: 230px; }
    col.med { width: 80px; }
    col.theory { width: 110px; }
    col.insight { width: 190px; }
    th, td { border: 1px solid #ddd; padding: 6px 10px; vertical-align: top; }
    th { background: #f5f5f5; text-align: left; }
    td.med { text-align: right; }
    td.theory { text-align: right; font-variant-numeric: tabular-nums; }
    .theory-main { display: block; font-weight: 600; }
    .theory-ratio, .theory-note { display: block; color: #666; font-size: 11px; }
    .theory-note { color: #8a5810; }
    td.insight { font-size: 12px; }
    .insight-note { margin: 0 0 6px; line-height: 1.35; }
    .insight-conf { display: inline-block; min-width: 48px; margin-right: 4px;
                    border-radius: 999px; padding: 1px 6px; font-size: 10px;
                    font-weight: 700; text-transform: uppercase; }
    .insight-note.high .insight-conf { background: #fee2e2; color: #991b1b; }
    .insight-note.medium .insight-conf { background: #fef3c7; color: #92400e; }
    .insight-category { display: inline-block; margin-right: 4px;
                        border-radius: 4px; padding: 1px 5px;
                        background: #e8eef8; color: #23436d;
                        font-size: 10px; font-weight: 700; }
    .insight-source, .insight-evidence, .insight-mapping {
        display: block; color: #777; font-size: 10.5px; margin-top: 2px; }
    .insight-mapping { color: #8a5810; }
    td.sub { font-weight: 600; }
    td.desc { color: #666; font-size: 12px; }
    tr.bubble td.sub { font-style: italic; }
    tr.total { background: #f9f9f9; font-weight: bold; }
    .chips { display: flex; flex-wrap: wrap; gap: 4px; align-items: stretch; }
    .chip { position: relative; display: inline-block; padding: 2px 6px;
            border-radius: 3px; background: #eef2f7; font-size: 11px;
            font-family: ui-monospace, "SF Mono", Consolas, monospace;
            white-space: nowrap; overflow: hidden; min-width: 70px; }
    .chip > .bar { position: absolute; left: 0; top: 0; bottom: 0;
                   background: #5a8fb8; opacity: 0.22; z-index: 0; }
    .chip > .tag, .chip > .lbl, .chip > .val { position: relative; z-index: 1; }
    .chip .lbl { color: #667; margin-right: 4px; }
    .chip.outlier-slow { font-weight: 600; }
    .chip.outlier-slow > .bar { background: #a32020; opacity: 0.28; }
    .chip.outlier-slow.tier-1 { background: #fdecec; border: 1px solid #f5b8b8;
                                color: #a32020; }
    .chip.outlier-slow.tier-1 .lbl { color: #a32020; }
    .chip.outlier-slow.tier-2 { background: #fbcaca; border: 1px solid #e07070;
                                color: #8a1818; }
    .chip.outlier-slow.tier-2 .lbl { color: #8a1818; }
    .chip.outlier-slow.tier-3 { background: #f59090; border: 1px solid #c84040;
                                color: #6f0e0e; }
    .chip.outlier-slow.tier-3 .lbl { color: #6f0e0e; }
    .chip.outlier-fast { font-weight: 600; }
    .chip.outlier-fast > .bar { background: #2d6a30; opacity: 0.28; }
    .chip.outlier-fast.tier-1 { background: #eaf3ea; border: 1px solid #b5d7b5;
                                color: #2d6a30; }
    .chip.outlier-fast.tier-1 .lbl { color: #2d6a30; }
    .chip.outlier-fast.tier-2 { background: #c8e3c8; border: 1px solid #88c08a;
                                color: #1f5022; }
    .chip.outlier-fast.tier-2 .lbl { color: #1f5022; }
    .chip.outlier-fast.tier-3 { background: #95cf99; border: 1px solid #5fa666;
                                color: #0f3a12; }
    .chip.outlier-fast.tier-3 .lbl { color: #0f3a12; }
    .chip .tag { font-weight: 600; margin-right: 4px;
                 font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    .phasebrk { border-left: 1px dashed #bbb; margin: 0 2px; align-self: stretch; }
    .legend { background: #fafafa; border: 1px solid #e0e0e0; border-radius: 4px;
              padding: 8px 12px; margin: 8px 0 18px; font-size: 12px;
              line-height: 1.7; }
    .legend .key { display: inline-block; padding: 1px 6px; border-radius: 3px;
                   border: 1px solid #ccc; margin: 0 4px; font-family: ui-monospace,
                   "SF Mono", Consolas, monospace; font-size: 11px; }
    .legend .key.slow { background: #fde2e2; border-color: #f0a0a0; color: #a32020; }
    .legend .key.fast { background: #e2f0e2; border-color: #a0d0a0; color: #2d6a30; }
    .unmatched { background: #fff8e7; padding: 8px 12px; border-left: 3px solid #e0a020;
                 margin: 8px 0; font-size: 13px; }
    .unmatched code { background: #f0e8d8; padding: 1px 4px; border-radius: 3px; }
    details.chip-d { display: block; padding: 0; margin: 0; min-width: 0; }
    details.chip-d[open] { flex: 1 1 100%; }
    details.chip-d > summary { list-style: none; cursor: pointer; outline: none;
                               user-select: none; }
    details.chip-d > summary::-webkit-details-marker { display: none; }
    details.chip-d > summary::marker { content: ""; }
    details.chip-d > .panel { display: none; }
    details.chip-d[open] > .panel { display: block; margin: 6px 0 4px; }
    details.chip-d[open] > summary.chip { box-shadow: 0 0 0 2px #5a8fb8; }
    details.chip-d[open] > summary.chip.outlier-slow { box-shadow: 0 0 0 2px #a32020; }
    details.chip-d[open] > summary.chip.outlier-fast { box-shadow: 0 0 0 2px #2d6a30; }
    .chip .med-note { color: #555; font-size: 10.5px; margin-left: 6px;
                      font-family: ui-monospace, "SF Mono", Consolas, monospace; }
    .chip.outlier-slow .med-note { color: #7a3030; }
    .chip.outlier-fast .med-note { color: #2d6a30; }
    .chip .anom-note { position: relative; z-index: 1; font-size: 10.5px;
                       margin-left: 6px; font-family: ui-monospace, "SF Mono",
                       Consolas, monospace; }
    .chip .anom-note .hot { color: #a32020; font-weight: 600; margin-right: 3px; }
    .chip .anom-note .cool { color: #2d6a30; font-weight: 600; margin-right: 3px; }
    .subwrap { overflow-x: auto; max-height: 320px; overflow-y: auto;
               border: 1px solid #e0e0e0; border-radius: 3px; margin-top: 4px; }
    table.subtbl { width: max-content; min-width: 100%; border-collapse: collapse;
                   font-size: 10.5px; font-family: ui-monospace, "SF Mono", Consolas, monospace;
                   table-layout: auto; margin: 0; }
    table.subtbl th, table.subtbl td { padding: 2px 6px; border: 1px solid #e8e8e8;
                                       white-space: nowrap; vertical-align: top; }
    table.subtbl th { background: #f0f4f8; position: sticky; top: 0;
                      font-weight: 600; font-family: -apple-system, BlinkMacSystemFont,
                      "Segoe UI", sans-serif; font-size: 10.5px; }
    table.subtbl td.where { font-weight: 600; background: #fff; }
    table.subtbl tr:nth-child(even) td { background: #f8f8f8; }
    table.subtbl tr:nth-child(even) td.where { background: #f4f4f4; }
    table.subtbl th.status-col, table.subtbl td.status-cell {
        min-width: 140px; max-width: 240px; white-space: normal;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    table.subtbl .status { display: inline-block; padding: 1px 5px; margin: 0 3px 2px 0;
                           border-radius: 3px; border: 1px solid #ccc; font-weight: 600;
                           font-size: 10px; white-space: nowrap; }
    table.subtbl .status.slow { background: #fde2e2; border-color: #f0a0a0; color: #a32020; }
    table.subtbl .status.fast { background: #e2f0e2; border-color: #a0d0a0; color: #2d6a30; }
    table.subtbl .status.extra { background: #fde2e2; border-color: #f0a0a0; color: #a32020; }
    table.subtbl .status.missing { background: #e2f0e2; border-color: #a0d0a0; color: #2d6a30; }
    table.subtbl tr.row-slow td, table.subtbl tr.row-slow:nth-child(even) td {
        background: #fceaea; }
    table.subtbl tr.row-fast td, table.subtbl tr.row-fast:nth-child(even) td {
        background: #e8f3e8; }
    table.subtbl tr.row-extra td, table.subtbl tr.row-extra:nth-child(even) td {
        background: #fdeede; }
    table.subtbl tr.row-missing td, table.subtbl tr.row-missing:nth-child(even) td {
        background: #eef5ee; font-style: italic; }
    table.subtbl td.missing-cell { color: #aaa; }
    .coverage-warn { background: #fde8e8; border-left: 3px solid #c83030;
                     padding: 8px 12px; margin: 8px 0; font-size: 13px;
                     color: #6e0e0e; }
    .coverage-warn .head { font-weight: 600; margin-bottom: 4px; }
    .overlap-warn { background: #fff4e0; border-left: 3px solid #d48820;
                    padding: 8px 12px; margin: 8px 0; font-size: 13px; }
    .overlap-warn .head { font-weight: 600; color: #8a5810; margin-bottom: 4px; }
    .overlap-warn .pairs { margin-top: 4px; font-family: ui-monospace,
                           "SF Mono", Consolas, monospace; font-size: 12px;
                           color: #6a4408; }
    .overlap-warn .pairs .pair { display: inline-block; margin: 2px 6px 2px 0;
                                 background: #f8e8c8; padding: 1px 6px;
                                 border-radius: 3px; }
    .overlap-info { background: #eef4fb; border-left: 3px solid #3b73a8;
                    padding: 8px 12px; margin: 8px 0; font-size: 13px;
                    color: #2a4a66; }
    .overlap-info .head { font-weight: 600; color: #2a4a66; margin-bottom: 4px; }
    .overlap-info .pairs { margin-top: 4px; font-family: ui-monospace,
                           "SF Mono", Consolas, monospace; font-size: 12px;
                           color: #2a4a66; }
    .overlap-info .pairs .pair { display: inline-block; margin: 2px 6px 2px 0;
                                 background: #dde8f4; padding: 1px 6px;
                                 border-radius: 3px; }
    """


def _render_section_warnings(sec, parts):
    """Append coverage / displaced / overlap warning panels for a section."""
    csp = sec.get("cluster_sum_pct_of_total")
    if csp is not None and csp < CLUSTER_COVERAGE_MIN_PCT:
        parts.append(
            f"<div class='coverage-warn'>"
            f"<div class='head'>⚠ cluster wall 中位之和 = {csp:.1f}% TOTAL "
            f"(&lt; {CLUSTER_COVERAGE_MIN_PCT:.0f}%)</div>"
            f"<div>该 component 大量算子落进 bubble / 未匹配 / 漏算子。"
            f"查下方 Unmatched 面板与 cluster 规则；可能是 catch_all 把真正语义"
            f"算子吞了，或某段 op 完全没被规则覆盖。</div>"
            f"</div>"
        )
    for d in sec.get("displaced") or []:
        parts.append(
            f"<div class='overlap-info'>"
            f"<div class='head'>⟂ 辅流 stream {html.escape(str(d['stream_id']))} "
            f"时间脱节，已从 TOTAL/bubble 排除</div>"
            f"<div>{d['op_count']} op；median wall "
            f"{fmt_duration_ms(d['median_wall_ms'])}/实例 "
            f"({d['pct_of_total']:.1f}% of TOTAL)。op 仍 matched 在该 component，"
            f"逐层明细见 splits/ 对应文件夹。</div>"
            f"</div>"
        )
    _render_overlap_warning(sec.get("overlap") or {}, parts)


def _overlap_pair_chips(pairs):
    """Render the pair chips fragment for an overlap panel."""
    return "".join(
        f"<span class='pair'>{html.escape(p['cluster_a'])} ↔ "
        f"{html.escape(p['cluster_b'])}: "
        f"{fmt_duration_ms(p['median_overlap_ms'])} "
        f"({p['median_overlap_pct']:.1f}% of TOTAL)</span>"
        for p in pairs
    )


def _render_overlap_warning(ov, parts):
    """Append the cross-cluster overlap warning / info panel for a section."""
    gap_pct = ov.get("median_gap_pct", 0.0)
    if gap_pct >= 5.0:
        pairs = ov.get("top_pairs", []) or []
        pair_chips = _overlap_pair_chips(pairs)
        parts.append(
            f"<div class='overlap-warn'>"
            f"<div class='head'>⚠ cluster 桶之间存在显著时间轴 overlap，"
            f"cluster wall 之和重复统计 ≈ {fmt_duration_ms(ov.get('median_gap_ms', 0))} "
            f"({gap_pct:.1f}% of TOTAL, 中位)；这种分桶不能直接反映真实"
            f"性能分布。</div>"
            f"<div>主要重叠对（按中位 overlap 降序，可能来自跨 stream 并发）"
            f"</div>"
            f"<div class='pairs'>{pair_chips or '—'}</div>"
            f"</div>"
        )
    elif gap_pct > 0.0:
        # 少量 overlap：过滤掉 < 1µs 的 floating-point 噪声项后，
        # 若仍有 pair 则插一个浅蓝 info 面板提示是哪些桶在跨 stream 并发。
        pairs = [p for p in (ov.get("top_pairs", []) or [])
                 if p.get("median_overlap_ms", 0.0) >= 0.001][:8]
        if pairs:
            pair_chips = _overlap_pair_chips(pairs)
            parts.append(
                f"<div class='overlap-info'>"
                f"<div class='head'>ℹ cluster 桶之间存在少量 timeline overlap"
                f"（中位 {fmt_duration_ms(ov.get('median_gap_ms', 0))}, "
                f"{gap_pct:.1f}% of TOTAL）；占比低于警告阈值，但下列桶对仍"
                f"有跨 stream 并发，cluster wall 之和会略微重复统计这部分。</div>"
                f"<div class='pairs'>{pair_chips}</div>"
                f"</div>"
            )


def _render_sub_item_row(item, ct, meta, details_by_idx, row_insights):
    """Render one sub-item table row."""
    m = item["metric"]
    row_cls = " class='bubble'" if item["kind"] == "bubble" else ""
    is_cluster = item["kind"] == "cluster"
    chip_html = chip_cell(
        _ChipSpec(
            per_inst=item["per_instance_ms"],
            outlier_idx=item.get("outlier_idx", []),
            kernel_outliers=item.get("kernel_outliers"),
            kernel_count_anomalies=item.get("kernel_count_anomalies"),
            op_records=item.get("op_records") if is_cluster else None,
            median_ms=m["median_ms"],
        ),
        meta, details_by_idx,
    )
    return (
        f"<tr{row_cls}>"
        f"<td class='sub'>{html.escape(item['name'])}</td>"
        f"<td class='desc'>{html.escape(item['description'])}</td>"
        f"<td class='med'>{fmt_duration_ms(m['median_ms'])}</td>"
        f"<td class='theory'>{render_theory_cell(item.get('theoretical'), m['median_ms'])}</td>"
        f"<td>{chip_html}</td>"
        f"<td class='insight'>{render_insight_cell(row_insights.get((ct, item['name'])))}</td>"
        f"</tr>"
    )


def _render_total_row(tot, ct, meta, details_by_idx, row_insights):
    """Render the TOTAL table row for a section."""
    m = tot["metric"]
    chip_html = chip_cell(
        _ChipSpec(
            per_inst=tot["per_instance_ms"],
            outlier_idx=tot.get("outlier_idx", []),
            median_ms=m["median_ms"],
        ),
        meta, details_by_idx,
    )
    return (
        f"<tr class='total'>"
        f"<td class='sub'>TOTAL</td>"
        f"<td class='desc'>{html.escape(tot['description'])}</td>"
        f"<td class='med'>{fmt_duration_ms(m['median_ms'])}</td>"
        f"<td class='theory'>{render_theory_cell(tot.get('theoretical'), m['median_ms'])}</td>"
        f"<td>{chip_html}</td>"
        f"<td class='insight'>{render_insight_cell(row_insights.get((ct, 'TOTAL')))}</td>"
        f"</tr>"
    )


def _render_section(sec, details_by_idx, row_insights, parts):
    """Render a full component section (heading, warnings, table, unmatched)."""
    ct = sec["component_type"]
    ic = sec["instance_count"]
    meta = sec.get("instances_meta", [])
    parts.append(f"<h2>{html.escape(ct)} <small>({ic} instances)</small></h2>")
    _render_section_warnings(sec, parts)
    parts.append(
        "<table>"
        "<colgroup>"
        "<col class='sub'><col class='desc'><col class='med'><col class='theory'><col><col class='insight'>"
        "</colgroup>"
        "<thead><tr>"
        "<th>sub-item</th><th>description</th>"
        "<th style='text-align:right'>median</th>"
        "<th style='text-align:right'>theory median</th>"
        "<th>per-instance &mdash; outliers highlighted</th>"
        "<th>insight</th>"
        "</tr></thead><tbody>"
    )
    for item in sec["sub_items"]:
        parts.append(_render_sub_item_row(item, ct, meta, details_by_idx, row_insights))
    parts.append(_render_total_row(sec["total"], ct, meta, details_by_idx, row_insights))
    parts.append("</tbody></table>")
    if sec["unmatched"]:
        top = sorted(sec["unmatched"].items(), key=lambda kv: -kv[1])[:20]
        items = " ".join(f"<code>{html.escape(n)}</code>:{c}" for n, c in top)
        parts.append(
            f"<div class='unmatched'><b>Unmatched ops</b> "
            f"({sum(sec['unmatched'].values())} total): {items}</div>"
        )


def render_html(model_name, sections, outlier_cfg, details_by_idx=None, row_insights=None):
    """Assemble the full single-page HTML dashboard string."""
    row_insights = row_insights or {}
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(model_name)} cluster breakdown</title>",
        f"<style>{_CSS}</style></head><body>",
        f"<h1>{html.escape(model_name)}</h1>",
        f"<div class='meta'>outlier rule: <code>{html.escape(outlier_cfg.get('method', 'iqr'))}</code> "
        f"k={outlier_cfg.get('k', 1.5)} &middot; "
        f"cluster wall = 该 cluster 算子的时间并集 &middot; "
        f"bubble = 层内算子间空隙（span − 总 wall 并集） &middot; "
        f"TOTAL = 层 end-to-end span（max_end − min_start, 含 bubble）"
        f"；HTML 自动按量级显示 µs / ms</div>",
        "<div class='legend'>"
        "<b>读图说明</b><br>"
        "每行 cluster 是一个 chip 列表，每个 chip 表示一个实例的 wall time。"
        "<span class='key slow'>红 chip = 比中位慢</span>、"
        "<span class='key fast'>绿 chip = 比中位快</span>（均为 IQR 异常），"
        "其旁直接标 <code>vs 中位 X µs/ms (±N%)</code>。<br>"
        "chip 标签后用紧凑标注汇总该实例算子级异常计数："
        "<code><span style='color:#a32020;font-weight:600'>N慢</span></code>/"
        "<code><span style='color:#2d6a30;font-weight:600'>N快</span></code>"
        "（同名同序号算子耗时偏离多数层）、"
        "<code><span style='color:#a32020;font-weight:600'>N多</span></code>/"
        "<code><span style='color:#2d6a30;font-weight:600'>N缺</span></code>"
        "（算子出现次数偏离多数层）。<br>"
        "若 <code>raw_ops_details.json</code> 已由 <code>operator_analysis.csv</code> 合并理论列，"
        "<code>theory median</code> 表示该 sub-item 在 critical stream 上的理论耗时中位数；"
        "多流时默认取 observed union 最大流，标 <code>review</code> 的项需 agent 判定流选择。<br>"
        "<b>点击 chip</b> 展开该实例的算子明细子表（bubble/TOTAL 不可展）。"
        "子表第一列「状态」与行底色给出每条异常的具体算子与数值。"
        "</div>",
    ]

    for sec in sections:
        _render_section(sec, details_by_idx, row_insights, parts)
    parts.append("</body></html>")
    return "\n".join(parts)


def print_history_reminder(run_dir):
    """<run_dir>/HISTORY.md 缺失时打软提示，返回是否提示过。

    Hook #6：不 exit 1，HISTORY 是收尾物，硬卡反而打断 agent 的写日志节奏。
    """
    history_path = os.path.join(run_dir, "HISTORY.md")
    if os.path.exists(history_path):
        return False
    logger.info(
        "ℹ HISTORY.md 缺失：%s\n"
        "  本次 run 完成后请 append 一条条目（label / prof / 复用-新建 / "
        "模型结构 / sample 锁定 / cluster 规则要点 / 渲染产物 / 关键观察）。\n"
        "  模板见 references/history_template.md。",
        history_path,
    )
    return True


# Communication cores never live inside an attn/ffn component, so they are
# excluded from the gate denominator on every architecture. Ops without a core
# field degrade naturally to "UNKNOWN" → counted → raw %, so no separate
# fallback branch is needed. (AI_CPU is already dropped upstream in sample mode.)
NON_LAYER_CORES = {"COMMUNICATION", "HCCL"}


def enforce_unmatched_gates(draft, operators, *, accept=False,
                            limit=UNMATCHED_PCT_HARD_LIMIT):
    """Gate when too many *compute* ops fell outside every component — the
    signal that a component/sample was mis-declared.

    The gate metric excludes communication because those are expected to be
    unmatched everywhere; gating on raw % mis-fires on comm/sampling-heavy
    workloads (e.g. decode + MoE all-to-all). The message prints the per-core
    breakdown so the agent can tell "expected IO/sampling" from "missed a real
    compute component". --accept-unmatched releases the gate.
    """
    unmatched_set = {int(i) for i in (draft.get("unmatched_op_indices") or [])}
    total_ops = len(operators)
    if total_ops <= 0:
        return

    by_idx = {int(o.get("index", n)): o for n, o in enumerate(operators)}
    core_counts = {}
    for i in unmatched_set:
        core = (by_idx.get(i, {}) or {}).get("accelerator_core") or "UNKNOWN"
        core_counts[core] = core_counts.get(core, 0) + 1

    def is_gated(op):  # comm excluded; missing core → "UNKNOWN" → gated (= raw %)
        return (op.get("accelerator_core") or "UNKNOWN") not in NON_LAYER_CORES
    gated_total = sum(1 for o in operators if is_gated(o)) or total_ops
    gated_unmatched = sum(1 for i in unmatched_set if is_gated(by_idx.get(i, {}) or {}))
    pct = gated_unmatched / gated_total

    if pct <= limit:
        return

    logger.warning(
        "[Step 3 unmatched gate] %d compute-core unmatched (通信类已排除) = %.1f%% "
        "(阈值 %.0f%%; 总 unmatched %d/%d)。",
        gated_unmatched, pct * 100, limit * 100, len(unmatched_set), total_ops,
    )
    if core_counts:
        brk_parts = []
        for core, n in sorted(core_counts.items(), key=lambda kv: -kv[1]):
            brk_parts.append(f"{core}×{n}")
        logger.warning("  - unmatched 按 accelerator_core: %s", ", ".join(brk_parts))
    sample = sorted(unmatched_set)[:20]
    logger.warning("  - unmatched_op_indices sample: %s", sample)
    logger.warning(
        "  - 判断：若 unmatched 主要是通信/采样/embedding/IO 等非 layer flow，"
        "属正常，加 --accept-unmatched 放行；若有大段 compute 算子漏匹配，"
        "回 Phase 0a/0b 补 sample 或声明 component。可用 --unmatched-limit 调阈值。"
    )

    if accept:
        logger.warning("[--accept-unmatched] 用户已知情放行，继续渲染。")
        return
    raise RuntimeError("unmatched compute-core ratio exceeds limit")


def build_argparser():
    """Build the render.py CLI argument parser."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-d", "--draft", required=True,
                   help="structure_draft.json from Step 2 stream_sample_driven mode")
    p.add_argument("-r", "--raw_ops", required=True,
                   help="raw_ops.json from Step 1")
    p.add_argument("--raw-ops-details", dest="raw_ops_details",
                   help="raw_ops_details.json from Step 1 (full CSV rows). "
                        "When provided, each cluster row gets a collapsible "
                        "sub-table with every clustered op's full CSV row. "
                        "If enriched by merge_theoretical_columns.py, theory "
                        "median columns are computed from these per-kernel fields.")
    p.add_argument("--theory-decisions", dest="theory_decisions",
                   help="optional JSON with agent stream-selection overrides "
                        "for ambiguous multi-stream theory aggregation.")
    p.add_argument("--theory-review-ratio", type=float, default=0.80,
                   help="mark theory stream choice for review when second "
                        "largest observed stream union >= this ratio of the "
                        "largest (default: 0.80).")
    p.add_argument("--insight-annotations", dest="insight_annotations",
                   help="optional agent-authored JSON for the final insight "
                        "column. Only high/medium items are rendered. Omit on "
                        "the first Step 3 render to leave the column blank.")
    p.add_argument("-s", "--spec", required=True,
                   help="network_spec.json (per-network clustering rules)")
    p.add_argument("-o", "--output", required=True,
                   help="output HTML path")
    p.add_argument("--label",
                   help="run label embedded in metrics.json (default: parent "
                        "dir name of -o). compare_runs.py keys runs by label.")
    p.add_argument("--accept-unmatched", action="store_true",
                   help="跳过 compute-core unmatched%% > 阈值 的硬阻断。"
                        "默认 exit 1 强迫 agent 看 per-core 分解后再决定是否放行"
                        "（通信类已自动排除出 gate 指标）。")
    p.add_argument("--unmatched-limit", type=float,
                   default=UNMATCHED_PCT_HARD_LIMIT,
                   help=f"compute-core unmatched 比例硬阈值 (默认 "
                        f"{UNMATCHED_PCT_HARD_LIMIT}); 通信/AICPU 等非 layer "
                        f"core 不计入。重 IO/采样负载可调高。")
    return p


def _instances_meta(instances):
    """Compact [{phase, layer_idx}] meta list for a component's instances."""
    return [
        {"phase": inst["phase"], "layer_idx": inst["layer_idx"]}
        for inst in instances
    ]


def _no_rules_instance_series(comp_type, instances, ctx):
    """Per-instance (span, bubble, theory-candidate) series for a rule-less type."""
    theory = (
        _TheoryContext(comp_type, ctx.operators, ctx.details_by_idx,
                       ctx.theory_decisions or {}, ctx.review_ratio)
        if ctx.details_by_idx else None
    )
    total_span = []
    total_bubble = []
    total_theory_candidates = []
    for inst_idx, inst in enumerate(instances):
        ivs = []
        op_indices = []
        for i in op_indices_from_instance(inst):
            st = ctx.operators[i].get("start_time_us")
            dur = ctx.operators[i].get("duration_us")
            if st is None or dur is None:
                continue
            ivs.append((st, st + dur))
            op_indices.append(i)
        w, span = wall_and_span_us(ivs)
        total_span.append(span)
        total_bubble.append(max(0.0, span - w))
        if theory is not None:
            total_theory_candidates.append(build_theory_candidate(
                _TheorySlot("TOTAL", inst_idx, inst, op_indices), theory,
            ))
    return total_span, total_bubble, total_theory_candidates


def build_section_no_rules(comp_type, instances, ctx):
    """Section for a component type with no cluster rules: bubble + TOTAL only."""
    total_span, total_bubble, total_theory_candidates = _no_rules_instance_series(
        comp_type, instances, ctx
    )
    method = ctx.outlier_cfg.get("method", "iqr")
    k = ctx.outlier_cfg.get("k", 1.5)
    b_idx, b_outs = outliers_for(total_bubble, instances, method, k)
    t_idx, t_outs = outliers_for(total_span, instances, method, k)
    unmatched_total = sum(len(op_indices_from_instance(inst)) for inst in instances)
    return {
        "component_type": comp_type,
        "instance_count": len(instances),
        "instances_meta": _instances_meta(instances),
        "sub_items": [{
            "name": "bubble",
            "kind": "bubble",
            "description": "layer idle gap (span − wall)",
            "metric": series_metrics(total_bubble),
            "outliers": b_outs,
            "outlier_idx": sorted(b_idx),
            "per_instance_ms": [v / 1000.0 for v in total_bubble],
        }],
        "total": {
            "description": "layer end-to-end span (max_end − min_start, 含 bubble)",
            "metric": series_metrics(total_span),
            "outliers": t_outs,
            "outlier_idx": sorted(t_idx),
            "per_instance_ms": [v / 1000.0 for v in total_span],
            "theoretical": summarize_theory(total_theory_candidates),
        },
        "unmatched": {"<no rules defined for this component type>": unmatched_total},
        "overlap": {"median_gap_ms": 0.0, "median_gap_pct": 0.0,
                    "max_gap_ms": 0.0, "top_pairs": []},
    }


def build_section_with_rules(comp_type, instances, compiled, ctx):
    """Section for a component type with cluster rules: full breakdown."""
    sub_items, total, unmatched, overlap_summary, displaced_summary = analyze_component_type(
        comp_type, instances, compiled, ctx,
    )
    cluster_medians = [s["metric"]["median_ms"] for s in sub_items
                       if s.get("kind") == "cluster"]
    total_median = total["metric"]["median_ms"]
    cluster_sum_pct = (sum(cluster_medians) / total_median * 100
                       if cluster_medians and total_median > 0 else None)
    return {
        "component_type": comp_type,
        "instance_count": len(instances),
        "instances_meta": _instances_meta(instances),
        "sub_items": sub_items,
        "total": total,
        "unmatched": unmatched,
        "overlap": overlap_summary,
        "cluster_sum_pct_of_total": cluster_sum_pct,
        "displaced": displaced_summary,
    }


def build_sections(by_type, cluster_defs, ctx):
    """Build and order one section per component type from the draft."""
    sections = []
    for comp_type, instances in by_type.items():
        if comp_type not in cluster_defs:
            sections.append(build_section_no_rules(comp_type, instances, ctx))
            continue
        compiled = compile_clusters(cluster_defs[comp_type])
        sections.append(build_section_with_rules(comp_type, instances, compiled, ctx))

    spec_order = list(cluster_defs.keys())
    sections.sort(key=lambda s: (
        spec_order.index(s["component_type"]) if s["component_type"] in spec_order
        else len(spec_order),
        s["component_type"],
    ))
    return sections


@dataclass
class _MetricsMeta:
    """Report header/provenance fields for the metrics.json payload."""
    label: str
    model_name: str
    outlier_cfg: dict
    details_payload: dict
    theory_decisions_path: str
    insight_annotations_path: str


def build_metrics_payload(meta, sections):
    """Assemble the metrics.json payload mirroring the rendered sections."""
    return {
        "label": meta.label,
        "model_name": meta.model_name,
        "outlier": meta.outlier_cfg,
        "generated_at": int(time.time()),
        "theoretical_perf_source": (
            meta.details_payload.get("theoretical_perf_source")
            if meta.details_payload else None
        ),
        "theory_decisions": meta.theory_decisions_path,
        "insight_annotations": meta.insight_annotations_path,
        "sections": [
            {
                "component_type": sec["component_type"],
                "instance_count": sec["instance_count"],
                "instances_meta": sec.get("instances_meta", []),
                "sub_items": sec["sub_items"],
                "total": sec["total"],
                "unmatched": sec["unmatched"],
                "overlap": sec.get("overlap", {}),
                "cluster_sum_pct_of_total": sec.get("cluster_sum_pct_of_total"),
                "displaced": sec.get("displaced", []),
            }
            for sec in sections
        ],
    }


def log_section_summary(sections):
    """Emit a per-section one-line summary to the run log."""
    for sec in sections:
        ct = sec["component_type"]
        ic = sec["instance_count"]
        unmatched_n = sum(sec["unmatched"].values())
        clusters_n = sum(1 for s in sec["sub_items"] if s["kind"] == "cluster")
        logger.info("  %s: %d instances, %d clusters + bubble, unmatched ops: %d",
                    ct, ic, clusters_n, unmatched_n)


def _load_details(raw_ops_details):
    """Load optional raw_ops_details.json → (payload, by_index_map)."""
    if not raw_ops_details:
        return None, None
    payload = load_json(raw_ops_details)
    details_ops = payload.get("operators", [])
    by_idx = {o.get("index"): o for o in details_ops if o.get("index") is not None}
    return payload, by_idx


def _write_metrics(metrics_meta, sections, out_dir):
    """Build the metrics payload and write metrics.json into out_dir."""
    metrics_path = os.path.join(out_dir, "metrics.json")
    metrics_payload = build_metrics_payload(metrics_meta, sections)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)
    logger.info("metrics  → %s (label=%s)", metrics_path, metrics_meta.label)


def run(args):
    """Execute the render pipeline for parsed CLI args."""
    draft = load_json(args.draft)
    validate_draft_schema(draft)
    ops_data = load_json(args.raw_ops)
    operators = ops_data["operators"]
    enforce_unmatched_gates(draft, operators, accept=args.accept_unmatched,
                            limit=args.unmatched_limit)
    spec = load_json(args.spec)
    details_payload, details_by_idx = _load_details(args.raw_ops_details)
    theory_decisions = load_theory_decisions(args.theory_decisions)

    model_name = spec.get("model_name", "model")
    outlier_cfg = spec.get("outlier", {"method": "iqr", "k": 1.5})
    cluster_defs = spec.get("component_clusters") or {}
    if not cluster_defs:
        raise ValueError("network_spec.json 缺 component_clusters")

    render_ctx = RenderContext(
        operators=operators,
        outlier_cfg=outlier_cfg,
        details_by_idx=details_by_idx,
        theory_decisions=theory_decisions,
        review_ratio=args.theory_review_ratio,
    )
    by_type = instances_by_type_from_draft(draft)
    sections = build_sections(by_type, cluster_defs, render_ctx)

    row_insights = load_agent_insight_annotations(args.insight_annotations)
    html_out = render_html(
        model_name, sections, outlier_cfg, details_by_idx, row_insights
    )
    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html_out)
    logger.info("dashboard → %s", args.output)

    label = args.label or os.path.basename(out_dir) or "run"
    metrics_meta = _MetricsMeta(
        label=label,
        model_name=model_name,
        outlier_cfg=outlier_cfg,
        details_payload=details_payload,
        theory_decisions_path=(os.path.abspath(args.theory_decisions)
                               if args.theory_decisions else None),
        insight_annotations_path=(os.path.abspath(args.insight_annotations)
                                  if args.insight_annotations else None),
    )
    _write_metrics(metrics_meta, sections, out_dir)

    log_section_summary(sections)

    run_dir = os.path.dirname(os.path.abspath(args.draft)) or "."
    print_history_reminder(run_dir)


def main():
    """CLI entry point. Returns a process exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )
    args = build_argparser().parse_args()
    try:
        run(args)
    except Exception as exc:  # 顶层 CLI 入口兜底，转退出码
        logger.error("render failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
