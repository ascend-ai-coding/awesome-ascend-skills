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
"""Step 5 — 洞察事实提取（脚本角色，纯机械，不做语义判断）。

执行者边界（见 references/insight_workflow.md §1）：
  脚本（本文件）：join / group-by / 排序 / 区间扫描 / 统计 / 选 Top-K 代表 / 落 JSON。
  agent：定义 movement taxonomy、审阅可信度、写 agent_review（summary/confidence/
         reason/fusion_candidate/elimination_direction）、写 final_conclusions。

因此本脚本**绝不**内置：模型/component/cluster/op/stream 名、搬运或融合 op 规则、
“理论不可靠”算子名单、性能原因、confidence。这些一律来自 agent：
  --movement-taxonomy <json>   agent 定义的 movement family（Insight 5 必需，缺省则只
                               产 op-frequency 候选清单 _taxonomy_candidates.json）。
  --annotations <json>         agent 审阅后填的 agent_review，按 stable key 合并。
缺省时 agent_review 字段留空，并产 _review_stub.json 列出所有待审 key。

排序口径：偏离/抖动一律按**绝对增量**（gap / delta）为主键——绝对量对不同模型/shape
更稳健，且天然压低“理论值极小→ratio 虚高”的小算子噪声（是否可信仍由 agent 用
confidence 标注）。展示层（render_insights.py）再按 confidence 重排。

用法:
  python gen_insights.py \
    --metrics <run>/runs/<label>/metrics.json \
    --raw-ops <run>/raw_ops.json \
    --structure-draft <run>/structure_draft.json \
    --raw-ops-details <run>/raw_ops_details.json \
    --out-dir <run>/runs/<label>/insights \
    [--movement-taxonomy <agent.json>] [--annotations <agent.json>] \
    [--top-k 5] [--small-n-ratio 1.5]

输入事实来源：metrics.json（cluster/bubble/TOTAL/outlier/theoretical 摘要、op_records
的 op→cluster 归属）、raw_ops.json（核类型/时间/stream/连续 vector 扫描/搬运邻接）、
raw_ops_details.json（per-kernel duration_over_theoretical 等理论列）、structure_draft.json
（op_to_component / component 边界）。
"""
import argparse
import json
import logging
import os
import statistics
import sys
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


def load(p):
    """读取 JSON 文件并返回解析后的对象。"""
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path):
    """把对象写成 UTF-8 JSON 文件（缩进 2，保留非 ASCII），覆盖正常/异常路径。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def median(xs):
    """空序列返回 0.0，否则返回中位数。"""
    return statistics.median(xs) if xs else 0.0


def us(ms):
    """毫秒转微秒（保留 1 位），非数值返回 None。"""
    return round(ms * 1000, 1) if isinstance(ms, (int, float)) else None


def small_n_outlier_idx(values, ratio):
    """与 render.detect_outliers 同口径：n>=4 用 IQR，2<=n<=3 用相对尖峰兜底。"""
    n = len(values)
    if n < 2:
        return set()
    if n < 4:
        med = median(values)
        if med <= 0 or not ratio or ratio <= 0:
            return set()
        return {i for i, v in enumerate(values) if v > med * ratio}
    sv = sorted(values)
    q1, q3 = sv[n // 4], sv[(3 * n) // 4]
    iqr = q3 - q1
    if iqr <= 0:
        return {i for i, v in enumerate(values) if v < q1 or v > q3}
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return {i for i, v in enumerate(values) if v < lo or v > hi}


EMPTY_REVIEW = {"summary": "", "confidence": "", "reason": "", "needs_followup": None}


def review_for(stub_registry, key, annotations, extra_fields=None):
    """登记一个待审 key，返回合并后的 agent_review（缺省全空，等 agent 填）。"""
    base = dict(EMPTY_REVIEW)
    if extra_fields:
        base.update(extra_fields)
    stub_registry[key] = dict(base)
    if annotations and key in annotations:
        merged = dict(base)
        merged.update({k: v for k, v in annotations[key].items() if v is not None})
        return merged
    return base


def topk_with_annotated(cands, k, key_fn, annotations):
    """取前 k（已按事实排序），但任何被 agent 注解过的候选强制保留。

    agent 可越过脚本排序选要展示的项。
    """
    if not annotations:
        return cands[:k]
    annotated = [c for c in cands if key_fn(c) in annotations]
    head = cands[:k]
    seen = {id(c) for c in head}
    for c in annotated:
        if id(c) not in seen:
            head.append(c)
            seen.add(id(c))
    return head


def parse_args(argv=None):
    """解析 CLI 参数（接口与原脚本完全一致）。"""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--raw-ops", required=True)
    ap.add_argument("--structure-draft", required=True)
    ap.add_argument("--raw-ops-details", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--movement-taxonomy", default=None,
                    help="agent 定义的 movement family JSON（Insight 5）。缺省只产候选清单。")
    ap.add_argument("--annotations", default=None,
                    help="agent agent_review 注解 JSON（按 stable key 合并）。")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--small-n-ratio", type=float, default=1.5)
    return ap.parse_args(argv)


class InsightContext:
    """汇总各 Insight 共享的派生上下文（避免大量函数参数）。"""

    def __init__(self, args):
        self.args = args
        self.top_k = args.top_k
        self.small_n_ratio = args.small_n_ratio
        self.out_dir = args.out_dir
        self.metrics = load(args.metrics)
        raw = load(args.raw_ops)
        self.draft = load(args.structure_draft)
        details = load(args.raw_ops_details)
        annotations = load(args.annotations) if args.annotations else None
        if annotations and "annotations" in annotations:
            annotations = annotations["annotations"]
        self.annotations = annotations
        self.taxonomy = load(args.movement_taxonomy) if args.movement_taxonomy else None

        self.ops = sorted(raw["operators"], key=lambda o: o["index"])
        self.byidx = {o["index"]: o for o in self.ops}
        self.o2c = self.draft.get("op_to_component", {})
        self.comp_by_id = {c["component_id"]: c for c in self.draft.get("components", [])}
        self.det_by_idx = {d["index"]: d for d in details.get("operators", [])
                           if d.get("index") is not None}
        self.stub = {}  # stable key -> empty review (for _review_stub.json)

        # op_idx -> (component_type, cluster) 来自 metrics op_records（不重评 network_spec 规则）
        self.op_cluster = {}
        for s in self.metrics["sections"]:
            ct = s["component_type"]
            for si in s["sub_items"]:
                if si.get("kind") != "cluster":
                    continue
                for rec in si.get("op_records", []):
                    self.op_cluster[rec["op_idx"]] = (ct, si["name"])

    @staticmethod
    def env_of(cobj):
        """component 的 op_indices 包络 [min, max]，cobj 为空返回 None。"""
        if not cobj:
            return None
        return [min(cobj["op_indices"]), max(cobj["op_indices"])]

    def find_comp(self, phase, layer, ctype):
        """按 (phase, layer_idx, type) 找 component，找不到返回 None。"""
        for c in self.draft.get("components", []):
            if c["phase"] == phase and c["layer_idx"] == layer and c["type"] == ctype:
                return c
        return None

    def cl_of(self, idx):
        """op 索引对应的 cluster 名，缺失返回 'unknown'。"""
        if idx in self.op_cluster:
            return self.op_cluster.get(idx, (None, "unknown"))[1]
        return "unknown"


def _persistent_module_entry(ctx, section, bub):
    """构造单个 component_type 的 persistent module bubble 条目。"""
    ct = section["component_type"]
    bpi = bub.get("per_instance_ms", [])
    tpi = section["total"].get("per_instance_ms", [])
    med = median(bpi)
    pct = [(b / t * 100) for b, t in zip(bpi, tpi) if t]
    meta = section.get("instances_meta", [])
    key = f"persistent:{ct}"
    rep = []
    for i in sorted(range(len(bpi)), key=lambda i: -bpi[i])[:ctx.top_k]:
        mi = meta[i] if i < len(meta) else {}
        env = ctx.env_of(ctx.find_comp(mi.get("phase"), mi.get("layer_idx"), ct))
        rep.append({"phase": mi.get("phase"), "layer_idx": mi.get("layer_idx"),
                    "bubble_us": us(bpi[i]), "op_range_envelope": env})
    high_ratio = round(sum(1 for p in pct if p > 5) / len(pct), 2) if pct else 0
    return {
        "component_type": ct, "instance_count": len(bpi),
        "median_bubble_us": us(med),
        "p90_bubble_us": us(sorted(bpi)[int(0.9 * (len(bpi) - 1))]),
        "max_bubble_us": us(max(bpi)),
        "median_bubble_pct_of_total": round(median(pct), 2) if pct else None,
        "high_bubble_instance_ratio": high_ratio,
        "representative_instances": rep,
        "agent_review": review_for(ctx.stub, key, ctx.annotations),
        "_sort": med,
    }


def _instance_bubble_outliers(ctx, section, bub, med):
    """同类型基线下的实例级 bubble outlier 候选。"""
    ct = section["component_type"]
    bpi = bub.get("per_instance_ms", [])
    tpi = section["total"].get("per_instance_ms", [])
    pct = [(b / t * 100) for b, t in zip(bpi, tpi) if t]
    meta = section.get("instances_meta", [])
    out = []
    for i in small_n_outlier_idx(bpi, ctx.small_n_ratio):
        mi = meta[i] if i < len(meta) else {}
        env = ctx.env_of(ctx.find_comp(mi.get("phase"), mi.get("layer_idx"), ct))
        ikey = f"instance_bubble:{mi.get('phase')}:{mi.get('layer_idx')}:{ct}"
        out.append({
            "phase": mi.get("phase"), "layer_idx": mi.get("layer_idx"),
            "component_type": ct, "bubble_us": us(bpi[i]),
            "bubble_pct_of_total": round(pct[i], 2) if i < len(pct) else None,
            "delta_vs_type_median_us": us(bpi[i] - med),
            "ratio_vs_type_median": round(bpi[i] / med, 2) if med else None,
            "evidence": {"op_range_envelope": env},
            "agent_review": review_for(ctx.stub, ikey, ctx.annotations),
            "_sort": bpi[i] - med,
        })
    return out


def insight_module_bubble(ctx):
    """Insight 1: persistent module bubbles + 实例级 bubble outliers。"""
    persistent = []
    inst_out = []
    for s in ctx.metrics["sections"]:
        bub = next((si for si in s["sub_items"] if si.get("name") == "bubble"), None)
        if not bub:
            continue
        bpi = bub.get("per_instance_ms", [])
        if not bpi:
            continue
        med = median(bpi)
        persistent.append(_persistent_module_entry(ctx, s, bub))
        inst_out.extend(_instance_bubble_outliers(ctx, s, bub, med))
    persistent.sort(key=lambda x: -x["_sort"])
    inst_out.sort(key=lambda x: -x["_sort"])
    for x in persistent + inst_out:
        x.pop("_sort", None)
    k = ctx.top_k
    dump_json({"schema_version": "insight.module_bubble.v1",
               "selection_policy": {"top_k": k, "baseline": "same_component_type",
                                    "sort": "median_bubble_us / delta_vs_type_median_us"},
               "persistent_module_bubbles": persistent[:k],
               "instance_bubble_outliers": inst_out[:k]},
              f"{ctx.out_dir}/module_bubble.json")
    return persistent, inst_out


def _cluster_jitter_candidates(ctx):
    """cluster 级抖动候选（cluster wall = interval union）。"""
    cluster_cand = []
    for s in ctx.metrics["sections"]:
        ct = s["component_type"]
        meta = s.get("instances_meta", [])
        for si in s["sub_items"]:
            if si.get("kind") != "cluster":
                continue
            pi = si.get("per_instance_ms", [])  # cluster wall = interval union (per doc)
            if len(pi) < 2:
                continue
            med = median(pi)
            for i in small_n_outlier_idx(pi, ctx.small_n_ratio):
                mi = meta[i] if i < len(meta) else {}
                env = ctx.env_of(ctx.find_comp(mi.get("phase"), mi.get("layer_idx"), ct))
                key = (f"cluster_jitter:{ct}:{si['name']}:"
                       f"{mi.get('phase')}:{mi.get('layer_idx')}")
                cluster_cand.append({
                    "component_type": ct, "cluster": si["name"],
                    "phase": mi.get("phase"), "layer_idx": mi.get("layer_idx"),
                    "duration_us": us(pi[i]), "baseline_median_us": us(med),
                    "delta_us": us(pi[i] - med),
                    "ratio_vs_baseline": round(pi[i] / med, 2) if med else None,
                    "evidence": {"op_range_envelope": env},
                    "agent_review": review_for(ctx.stub, key, ctx.annotations),
                    "_sort": pi[i] - med,
                })
    cluster_cand.sort(key=lambda x: -(x["_sort"] or 0))
    return cluster_cand


def _op_jitter_candidates(ctx):
    """operator 级抖动候选，消费每个 cluster sub_item 的 kernel_outliers。

    render.py 已按 (op_name, occurrence) 在 cluster 实例内分槽，故同名 op 在不同
    语义位置仍分开（occurrence）。无 kernel_outliers 的 cluster 不产 operator 候选。
    """
    op_slot = defaultdict(list)
    for s in ctx.metrics["sections"]:
        ct = s["component_type"]
        for si in s["sub_items"]:
            if si.get("kind") != "cluster":
                continue
            for ko in si.get("kernel_outliers", []):
                slot = (ct, si["name"], ko.get("op_name"), ko.get("occurrence"))
                op_slot[slot].append(ko)
    op_cand = []
    for (ct, cl, nm, occ), kos in op_slot.items():
        worst = max(kos, key=lambda k: abs((k.get("duration_us") or 0)
                                           - (k.get("baseline_median_us") or 0)))
        dur = worst.get("duration_us") or 0
        base = worst.get("baseline_median_us") or 0
        key = f"op_jitter:{ct}:{cl}:{nm}:{occ}"
        op_cand.append({
            "component_type": ct, "cluster": cl, "op_name": nm, "occurrence": occ,
            "phase": worst.get("phase"), "layer_idx": worst.get("layer_idx"),
            "duration_us": round(dur, 1), "baseline_median_us": round(base, 1),
            "delta_us": round(dur - base, 1),
            "ratio_vs_baseline": round(dur / base, 2) if base else None,
            "agent_review": review_for(ctx.stub, key, ctx.annotations),
            "_sort": abs(dur - base),
        })
    op_cand.sort(key=lambda x: -x["_sort"])
    return op_cand


def insight_operator_jitter(ctx):
    """Insight 2: cluster 级 + operator 级抖动候选。"""
    cluster_cand = _cluster_jitter_candidates(ctx)
    op_cand = _op_jitter_candidates(ctx)
    k = ctx.top_k
    cc = topk_with_annotated(
        cluster_cand, k,
        lambda c: (f"cluster_jitter:{c['component_type']}:{c['cluster']}:"
                   f"{c['phase']}:{c['layer_idx']}"),
        ctx.annotations)
    oc = topk_with_annotated(
        op_cand, k,
        lambda c: (f"op_jitter:{c['component_type']}:{c['cluster']}:"
                   f"{c['op_name']}:{c['occurrence']}"),
        ctx.annotations)
    for x in cluster_cand + op_cand:
        x.pop("_sort", None)
    cluster_src = "metrics.sections[].sub_items[kind=cluster].per_instance_ms"
    cluster_sem = "cluster wall = interval union, not sum(duration_us)"
    op_src = "metrics.sections[].sub_items[kind=cluster].kernel_outliers"
    dump_json({"schema_version": "insight.operator_jitter.v1",
               "selection_policy": {"baseline": "same_component_type_and_cluster",
                                    "top_k": k,
                                    "cluster_metric_source": cluster_src,
                                    "cluster_metric_semantics": cluster_sem,
                                    "operator_metric_source": op_src,
                                    "operator_slot": "(component_type, cluster, op_name, occurrence)",
                                    "sort": "delta_us (absolute)"},
               "cluster_jitter_candidates": cc,
               "operator_jitter_candidates": oc},
              f"{ctx.out_dir}/operator_jitter.json")
    return cluster_cand, op_cand


def _sub_item_rep_instances(ctx, section, name, si, th):
    """sub_item 偏离的代表实例（按实际耗时降序取 Top-K）。"""
    ct = section["component_type"]
    meta = section.get("instances_meta", [])
    pi = si.get("per_instance_ms", [])
    tpi = th.get("per_instance", [])
    rep = []
    for i in sorted(range(len(pi)), key=lambda i: -pi[i])[:ctx.top_k]:
        mi = meta[i] if i < len(meta) else {}
        cobj_r = ctx.find_comp(mi.get("phase"), mi.get("layer_idx"), ct)
        rep_ops = [oi for oi in (cobj_r.get("op_indices", []) if cobj_r else [])
                   if name == "TOTAL" or ctx.op_cluster.get(oi, (None, None))[1] == name]
        has_theo = (i < len(tpi)
                    and tpi[i].get("effective_theoretical_us") is not None)
        theo_ms = round(tpi[i]["effective_theoretical_us"] / 1000, 4) if has_theo else None
        rep.append({"phase": mi.get("phase"), "layer_idx": mi.get("layer_idx"),
                    "actual_ms": round(pi[i], 4),
                    "theoretical_ms": theo_ms,
                    "op_indices": rep_ops,
                    "op_range_envelope": [min(rep_ops), max(rep_ops)] if rep_ops else None})
    return rep


def _sub_item_deviation(ctx):
    """sub_item / TOTAL 级理论偏离候选。"""
    sub_dev = []
    for s in ctx.metrics["sections"]:
        ct = s["component_type"]
        items = [(si["name"], si) for si in s["sub_items"] if si["name"] != "other"]
        items.append(("TOTAL", s["total"]))
        for name, si in items:
            th = si.get("theoretical") or {}
            amed = si["metric"].get("median_ms")
            tmed = th.get("median_theoretical_ms")
            if not (amed and tmed):
                continue
            gap = (amed - tmed) * 1000
            if abs(gap) < 1:
                continue
            rep = _sub_item_rep_instances(ctx, s, name, si, th)
            key = f"sub_dev:{ct}:{name}"
            support_ratio = round(th.get("supported_instance_count", 0)
                                  / max(th.get("instance_count", 1), 1), 2)
            sub_dev.append({
                "component_type": ct, "sub_item": name,
                "actual_median_ms": round(amed, 4),
                "theoretical_median_ms": round(tmed, 4),
                "wall_over_theoretical_median": round(amed / tmed, 2) if tmed else None,
                "absolute_gap_us": round(gap, 1),
                # 理论量级——agent 据此判可信度（极小理论值→ratio 不可信），脚本不替它判
                "theoretical_magnitude_us": round(tmed * 1000, 1),
                "support_ratio": support_ratio,
                "representative_instances": rep,
                "agent_review": review_for(ctx.stub, key, ctx.annotations),
                "_sort": gap,
            })
    sub_dev.sort(key=lambda x: -x["_sort"])
    return sub_dev


def _occurrence_of_ops(ctx):
    """每个 op 在 (component, cluster) 内的 occurrence 序号。

    occurrence = op_name 在 (component instance, cluster) 内第 N 次出现，镜像
    render.py 的 per-instance occ_counter，使同一语义位置可跨实例比较。
    """
    occ_of = {}
    occ_counter = defaultdict(lambda: defaultdict(int))
    for opidx in sorted(ctx.op_cluster):
        o = ctx.byidx.get(opidx)
        if not o:
            continue
        _ct, _cl = ctx.op_cluster[opidx]
        scope = (ctx.o2c.get(str(opidx)), _cl)
        nm = o["normalized_name"]
        occ_of[opidx] = occ_counter[scope][nm]
        occ_counter[scope][nm] += 1
    return occ_of


def _op_deviation_locations(ctx, top):
    """单个 op slot 的 Top-K 偏离位置。"""
    locs = []
    for _dot, opidx, oi, dur, th_us, bt in top:
        cobj = ctx.comp_by_id.get(ctx.o2c.get(str(opidx)), {})
        locs.append({"phase": cobj.get("phase"), "layer_idx": cobj.get("layer_idx"),
                     "org_index": oi, "duration_us": round(dur, 1) if dur else None,
                     "theoretical_us": round(th_us, 1) if th_us else None,
                     "bound_type": bt})  # 可能为 None（理论 skill 未填）
    return locs


def _op_slot_deviation(ctx):
    """operator slot 级理论偏离候选，来自 raw_ops_details duration_over_theoretical。"""
    occ_of = _occurrence_of_ops(ctx)
    op_slot = defaultdict(list)
    for opidx, (ct, cl) in ctx.op_cluster.items():
        o = ctx.byidx.get(opidx)
        d = ctx.det_by_idx.get(opidx)
        if not o or not d or not d.get("theory_supported"):
            continue
        dot = d.get("duration_over_theoretical")
        if dot is None:
            continue
        slot = (ct, cl, o["normalized_name"], occ_of.get(opidx))
        op_slot[slot].append(
            (dot, opidx, o.get("org_index"), o.get("duration_us"),
             d.get("theoretical_operator_time_us"), d.get("bound_type")))
    op_dev = []
    for (ct, cl, nm, occ), lst in op_slot.items():
        if len(lst) < 3:
            continue
        dots = [x[0] for x in lst]
        gaps = [(x[3] - x[4]) for x in lst if x[3] is not None and x[4] is not None]
        if not gaps:
            continue
        mgap = median(gaps)
        if mgap <= 1:
            continue
        top = sorted(lst, key=lambda x: -x[0])[:ctx.top_k]
        locs = _op_deviation_locations(ctx, top)
        key = f"op_dev:{ct}:{cl}:{nm}:{occ}"
        op_dev.append({
            "component_type": ct, "cluster": cl, "op_name": nm, "occurrence": occ,
            "duration_over_theoretical_median": round(median(dots), 2),
            "absolute_gap_us_median": round(mgap, 1),
            "max_duration_over_theoretical": round(max(dots), 2),
            "supported_count": len(lst),
            "top_locations": locs,
            "agent_review": review_for(ctx.stub, key, ctx.annotations),
            "_sort": mgap,
        })
    op_dev.sort(key=lambda x: -x["_sort"])
    return op_dev


def insight_theoretical_deviation(ctx):
    """Insight 3: sub_item 级 + operator slot 级理论偏离候选。"""
    data_limits = []
    if not ctx.det_by_idx or not any(d.get("theory_supported")
                                     for d in ctx.det_by_idx.values()):
        data_limits.append(
            "raw_ops_details 缺理论列或 theory_supported 全空——回 Step 1.5 重跑理论注入。")
    sub_dev = _sub_item_deviation(ctx)
    op_dev = _op_slot_deviation(ctx)
    k = ctx.top_k
    sd = topk_with_annotated(
        sub_dev, k,
        lambda c: f"sub_dev:{c['component_type']}:{c['sub_item']}",
        ctx.annotations)
    od = topk_with_annotated(
        op_dev, k,
        lambda c: (f"op_dev:{c['component_type']}:{c['cluster']}:"
                   f"{c['op_name']}:{c['occurrence']}"),
        ctx.annotations)
    for x in sub_dev + op_dev:
        x.pop("_sort", None)
    note = ("按绝对 gap 排序（稳健，压低理论值极小的虚高 ratio）；"
            "可信度由 agent 用 confidence 标")
    dump_json({"schema_version": "insight.theoretical_deviation.v1",
               "selection_policy": {"top_k": k, "ranking_object": "logical_slot",
                                    "singleton_instances": "evidence_only",
                                    "sub_item_sort": ["absolute_gap_us"],
                                    "operator_slot_sort": ["absolute_gap_us_median"],
                                    "note": note},
               "data_limits": data_limits,
               "sub_item_deviation_candidates": sd,
               "operator_slot_deviation_candidates": od,
               "unsupported_summary": []},
              f"{ctx.out_dir}/theoretical_deviation.json")
    return sub_dev, op_dev


def _vector_runs(ctx):
    """连续 AI_VECTOR_CORE 段（长度 >= 5）。"""
    runs = []
    cur = []
    for o in ctx.ops:
        if o.get("accelerator_core") == "AI_VECTOR_CORE":
            cur.append(o)
        else:
            if len(cur) >= 5:
                runs.append(cur)
            cur = []
    if len(cur) >= 5:
        runs.append(cur)
    return runs


def _run_signature(vector_run):
    """把一段连续 vector run 折叠成 run-length 签名字符串。"""
    names = [o["normalized_name"] for o in vector_run]
    out, i = [], 0
    while i < len(names):
        j = i
        while j < len(names) and names[j] == names[i]:
            j += 1
        out.append(f"{names[i]} x{j - i}" if j - i > 1 else names[i])
        i = j
    return " -> ".join(out)


def _vector_pattern_samples(ctx, runs):
    """单个 pattern 的代表样本（按耗时降序取前 3）。"""
    samples = []
    for r in sorted(runs, key=lambda r: -sum(o["duration_us"] for o in r))[:3]:
        i0 = r[0]["index"]
        cobj = ctx.comp_by_id.get(ctx.o2c.get(str(i0), ""), {}) or {}
        samples.append({"phase": cobj.get("phase"), "layer_idx": cobj.get("layer_idx"),
                        "component_type": cobj.get("type", "unmatched"),
                        "cluster": ctx.cl_of(i0),
                        "op_indices": [o["index"] for o in r],
                        "op_range_envelope": [i0, r[-1]["index"]],
                        "duration_us": round(sum(o["duration_us"] for o in r), 1)})
    return samples


def insight_vector_sequences(ctx):
    """Insight 4: 连续 vector 序列模式候选。"""
    runs = _vector_runs(ctx)
    patt = defaultdict(list)
    for r in runs:
        patt[_run_signature(r)].append(r)
    pats = []
    for s_sig, rs in patt.items():
        tot = sum(o["duration_us"] for r in rs for o in r)
        comps = sorted({(ctx.comp_by_id.get(ctx.o2c.get(str(o["index"]), ""), {})
                         or {}).get("type", "unmatched")
                        for o in rs[0]})
        clusters = sorted({ctx.cl_of(o["index"]) for o in rs[0]})
        samples = _vector_pattern_samples(ctx, rs)
        key = f"vec:{s_sig}"
        typical = round(median([sum(o["duration_us"] for o in r) for r in rs]), 1)
        review = review_for(ctx.stub, key, ctx.annotations,
                            {"semantic_summary": "", "fusion_candidate": "", "confidence": ""})
        pats.append({"pattern_signature": s_sig, "occurrences": len(rs),
                     "total_duration_us": round(tot, 1),
                     "typical_duration_us": typical,
                     "components": comps,
                     "clusters": clusters,
                     "representative_samples": samples,
                     "agent_review": review,
                     "_sort": tot})
    pats.sort(key=lambda x: -x["_sort"])
    pk = topk_with_annotated(pats, ctx.top_k,
                             lambda c: f"vec:{c['pattern_signature']}", ctx.annotations)
    for x in pats:
        x.pop("_sort", None)
    dump_json({"schema_version": "insight.vector_sequence_candidates.v1",
               "selection_policy": {"min_ops": 5, "min_total_us": 20.0, "top_k": ctx.top_k,
                                    "group_by": "pattern_signature", "sort": "total_duration_us"},
               "patterns": pk},
              f"{ctx.out_dir}/vector_sequence_candidates.json")
    return pats


def write_taxonomy_candidates(ctx):
    """taxonomy 候选清单（op-name × core 频次）——给 agent 定 movement family 用。"""
    op_freq = defaultdict(lambda: {"count": 0, "total_us": 0.0, "cores": Counter()})
    for o in ctx.ops:
        e = op_freq[o["normalized_name"]]
        e["count"] += 1
        e["total_us"] += o.get("duration_us", 0)
        e["cores"][o.get("accelerator_core")] += 1
    tax_cand = []
    for k, v in sorted(op_freq.items(), key=lambda kv: -kv[1]["total_us"]):
        tax_cand.append({"op_name": k, "count": v["count"],
                         "total_duration_us": round(v["total_us"], 1),
                         "cores": dict(v["cores"])})
    note = ("agent 据此定义 movement taxonomy（哪些是 layout/shape/indexing/"
            "cache_update/communication_prep 搬运），写成 --movement-taxonomy JSON。"
            "脚本不替你判哪些是搬运。")
    dump_json({"note": note, "op_frequency": tax_cand},
              f"{ctx.out_dir}/_taxonomy_candidates.json")


def _movement_neighbors(ctx, i):
    """op i 在 stream 内与全局的前后邻居 normalized_name。"""
    o = ctx.byidx[i]
    sid = str(o.get("stream_id"))
    same = sorted([x for x in ctx.ops if str(x.get("stream_id")) == sid],
                  key=lambda x: x.get("start_time_us", 0))
    pos = next((k for k, x in enumerate(same) if x["index"] == i), None)
    has_next = pos is not None and pos + 1 < len(same)
    return {
        "prev_same_stream": same[pos - 1]["normalized_name"] if pos and pos > 0 else None,
        "next_same_stream": same[pos + 1]["normalized_name"] if has_next else None,
        "prev_global": ctx.byidx.get(i - 1, {}).get("normalized_name"),
        "next_global": ctx.byidx.get(i + 1, {}).get("normalized_name"),
    }


def _movement_aggregate(ctx, op2fam):
    """按 family × (component_type, cluster) 聚合搬运 op。"""
    fam_agg = defaultdict(lambda: defaultdict(lambda: {"n": 0, "dur": 0.0, "locs": []}))
    fam_tot = defaultdict(lambda: {"n": 0, "dur": 0.0})
    for o in ctx.ops:
        fam = op2fam.get(o["normalized_name"])
        if not fam:
            continue
        i = o["index"]
        cid = ctx.o2c.get(str(i), "unmatched")
        cobj = ctx.comp_by_id.get(cid, {})
        ct = cobj.get("type", "unmatched")
        cl = ctx.op_cluster.get(i, (ct, "unmatched"))[1] if i in ctx.op_cluster else "unmatched"
        fam_agg[fam][(ct, cl)]["n"] += 1
        fam_agg[fam][(ct, cl)]["dur"] += o.get("duration_us", 0)
        fam_agg[fam][(ct, cl)]["locs"].append((o.get("duration_us", 0), i, o.get("org_index"), cobj))
        fam_tot[fam]["n"] += 1
        fam_tot[fam]["dur"] += o.get("duration_us", 0)
    return fam_agg, fam_tot


def _write_movement_empty(ctx):
    """没有 agent taxonomy 时只产候选清单，families 留空。"""
    note = ("未提供 --movement-taxonomy：请 agent 先用 _taxonomy_candidates.json "
            "定义 movement family，再带 --movement-taxonomy 重跑。")
    dump_json({"schema_version": "insight.data_movement_ops.v1",
               "agent_taxonomy": {"families": [], "needs_user_review": []},
               "families": [],
               "_note": note},
              f"{ctx.out_dir}/data_movement_ops.json")


def insight_data_movement(ctx):
    """Insight 5: data movement ops（依赖 agent 的 movement taxonomy）。"""
    if not ctx.taxonomy:
        _write_movement_empty(ctx)
        return []
    taxonomy = ctx.taxonomy
    tax_families = taxonomy.get("families",
                                taxonomy if isinstance(taxonomy, list) else [])
    op2fam = {op: f["family"] for f in tax_families for op in f.get("ops", [])}
    fam_agg, fam_tot = _movement_aggregate(ctx, op2fam)
    families_out = []
    for fam in sorted(fam_agg, key=lambda f: -fam_tot[f]["dur"]):
        mods = []
        for (ct, cl), v in sorted(fam_agg[fam].items(),
                                  key=lambda kv: -kv[1]["dur"])[:ctx.top_k]:
            top = sorted(v["locs"], key=lambda x: -x[0])[:ctx.top_k]
            locs = [{"phase": cobj.get("phase"), "layer_idx": cobj.get("layer_idx"),
                     "duration_us": round(dur, 1), "op_indices": [i],
                     "op_range_envelope": [i, i], "op_idx": i, "org_index": oi,
                     "neighbor_context": _movement_neighbors(ctx, i)}
                    for dur, i, oi, cobj in top]
            key = f"move:{fam}:{ct}:{cl}"
            rev = review_for(ctx.stub, key, ctx.annotations,
                             {"semantic_summary": "", "elimination_direction": "",
                              "confidence": ""})
            mods.append({"component_type": ct, "cluster": cl,
                         "total_duration_us": round(v["dur"], 1), "occurrences": v["n"],
                         "top_locations": locs, "agent_review": rev})
        fam_ops = next((f.get("ops", []) for f in tax_families if f["family"] == fam), [])
        families_out.append({"family": fam, "ops": fam_ops,
                             "total_duration_us": round(fam_tot[fam]["dur"], 1),
                             "occurrences": fam_tot[fam]["n"], "by_module": mods})
    dump_json({"schema_version": "insight.data_movement_ops.v1",
               "agent_taxonomy": {"families": tax_families,
                                  "needs_user_review": taxonomy.get("needs_user_review", [])},
               "families": families_out[:ctx.top_k]},
              f"{ctx.out_dir}/data_movement_ops.json")
    return families_out


def write_review_stub(ctx):
    """落 _review_stub.json：所有待审 stable key 的空 agent_review。"""
    note = ("agent 填 agent_review 后改名/复制为 annotations.json，"
            "带 --annotations 重跑 gen。key 稳定，"
            "summary/confidence(high|medium|low)/reason 等留给 agent。")
    dump_json({"note": note,
               "annotations": {k: v for k, v in sorted(ctx.stub.items())}},
              f"{ctx.out_dir}/_review_stub.json")


def log_summary(ctx, counts):
    """打印进度/统计摘要到 stderr（人读，非下游消费契约）。"""
    annotations = ctx.annotations
    annotated_n = (len([k for k in ctx.stub if annotations and k in annotations])
                   if annotations else 0)
    movement = "(no taxonomy)" if not ctx.taxonomy else counts["movement"]
    logger.info("gen_insights → %s", ctx.out_dir)
    logger.info("  candidates: bubble %d+%d, jitter %d+%d, theory %d+%d, "
                "vector %d, movement %s",
                counts["persistent"], counts["inst_out"],
                counts["cluster_cand"], counts["op_cand"],
                counts["sub_dev"], counts["op_dev"],
                counts["pats"], movement)
    stub_tail = "" if annotations else "  (空 agent_review，待 agent 填 _review_stub.json)"
    logger.info("  review keys: %d  ·  annotated: %d%s", len(ctx.stub), annotated_n, stub_tail)
    if not ctx.taxonomy:
        logger.warning("  ⚠ Insight 5 待 movement taxonomy：见 _taxonomy_candidates.json，"
                       "定义后带 --movement-taxonomy 重跑")


def run(args):
    """执行五类 Insight 提取并落盘（业务主流程，异常向上抛出）。"""
    ctx = InsightContext(args)
    os.makedirs(ctx.out_dir, exist_ok=True)
    persistent, inst_out = insight_module_bubble(ctx)
    cluster_cand, op_cand = insight_operator_jitter(ctx)
    sub_dev, op_dev = insight_theoretical_deviation(ctx)
    pats = insight_vector_sequences(ctx)
    write_taxonomy_candidates(ctx)
    families_out = insight_data_movement(ctx)
    write_review_stub(ctx)
    log_summary(ctx, {
        "persistent": len(persistent), "inst_out": len(inst_out),
        "cluster_cand": len(cluster_cand), "op_cand": len(op_cand),
        "sub_dev": len(sub_dev), "op_dev": len(op_dev),
        "pats": len(pats), "movement": len(families_out),
    })


def main(argv=None):
    """CLI 主入口：解析参数、配置日志、捕获异常并以非 0 退出。"""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    try:
        run(args)
    except Exception as e:
        logger.error("gen_insights failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
