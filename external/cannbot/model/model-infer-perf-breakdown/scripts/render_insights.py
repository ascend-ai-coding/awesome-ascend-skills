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
"""Step 5 — 洞察 HTML 渲染（脚本角色）。

以 references/insight_html_template.html 为骨架，服务端预填五类 insight 表 + 常驻
结论区。**只渲染已落盘的 JSON 和 final_conclusions.md，不重算、不推断语义**。

- 表格按 agent_review.confidence (high>medium>low) 重排，同档按事实数值降序，取 Top5，
  表头标 "Top5"。某数组空时保留 tab，显示“本次未发现候选”。
- 结论卡 / 数据限制：从 final_conclusions.md 的 "### High|Medium|Low" 和 "## 数据限制"
  小节解析（agent 写什么就显示什么，脚本不造）。
- 列布局按**固定 schema 字段**派生，不枚举模型/component/cluster 名（这些来自数据）。

用法:
  python render_insights.py --insights-dir <run>/runs/<label>/insights \
    --model-name "<name>" --label <label> [--template <path>] -o <.../insights/index.html>
"""
import argparse
import html
import json
import logging
import os
import re
import sys

logger = logging.getLogger(__name__)

CONF = {"high": 0, "medium": 1, "low": 2, "": 3, None: 3}
DEF_TEMPLATE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "references", "insight_html_template.html")


def load(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def esc(x):
    return html.escape(str(x)) if x is not None else "—"


def conf_of(row):
    return (row.get("agent_review") or {}).get("confidence") or ""


def badge(c):
    c = c or "—"
    cls = c if c in ("high", "medium", "low") else "low"
    return f'<span class="confidence {cls}">{esc(c)}</span>'


def metric(x, unit="µs"):
    return f'<span class="metric">{esc(x)}{unit if x is not None else ""}</span>'


def order(rows, value_key):
    """confidence 升序(high 在前) → 事实数值降序。"""
    def vk(r):
        v = r.get(value_key)
        return v if isinstance(v, (int, float)) else 0
    return sorted(rows, key=lambda r: (CONF.get(conf_of(r), 3), -vk(r)))


def table(headers, rows, top=5):
    h = "".join(f"<th>{esc(x)}</th>" for x in headers)
    if not rows:
        return (f'<thead><tr>{h}</tr></thead><tbody><tr>'
                f'<td colspan="{len(headers)}" class="empty">本次未发现候选</td></tr></tbody>')
    body = ""
    for r in rows[:top]:
        body += "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
    return f"<thead><tr>{h}</tr></thead><tbody>{body}</tbody>"


def summ(row):
    return esc((row.get("agent_review") or {}).get("summary") or "")


def _module_bubble_tables(mb):
    """module_bubble persistent (sort by median bubble) + instance outliers。"""
    out = {}
    rows = []
    for d in order(mb.get("persistent_module_bubbles", []), "median_bubble_us"):
        rep = (d.get("representative_instances") or [{}])[0]
        rows.append([esc(d.get("component_type")),
                     metric(f"{d.get('median_bubble_us')} / {d.get('median_bubble_pct_of_total')}%", ""),
                     metric(f"{d.get('p90_bubble_us')}/{d.get('max_bubble_us')}"),
                     esc(f"{rep.get('phase','')} L{rep.get('layer_idx','')}"),
                     badge(conf_of(d)), summ(d)])
    out["module_bubble.json:persistent_module_bubbles"] = table(
        ["component_type", "median bubble / %", "p90/max", "rep loc", "conf", "agent summary"], rows)

    rows = []
    for d in order(mb.get("instance_bubble_outliers", []), "bubble_us"):
        rows.append([esc(f"{d.get('phase')} L{d.get('layer_idx')} {d.get('component_type')}"),
                     metric(d.get("bubble_us")), metric(d.get("delta_vs_type_median_us")),
                     badge(conf_of(d)), summ(d)])
    out["module_bubble.json:instance_bubble_outliers"] = table(
        ["phase/layer/comp", "bubble", "delta vs median", "conf", "agent summary"], rows)
    return out


def _operator_jitter_tables(oj):
    """cluster / operator jitter candidates。"""
    out = {}
    rows = []
    for d in order(oj.get("cluster_jitter_candidates", []), "delta_us"):
        rows.append([esc(f"{d.get('component_type')}/{d.get('cluster')}"),
                     metric(d.get("duration_us")),
                     metric(f"base {d.get('baseline_median_us')} · x{d.get('ratio_vs_baseline')}", ""),
                     esc(f"{d.get('phase')} L{d.get('layer_idx')}"), badge(conf_of(d)), summ(d)])
    out["operator_jitter.json:cluster_jitter_candidates"] = table(
        ["comp/cluster", "wall (union)", "baseline/ratio", "loc", "conf", "agent summary"], rows)

    rows = []
    for d in order(oj.get("operator_jitter_candidates", []), "delta_us"):
        rows.append([esc(f"{d.get('component_type')}/{d.get('cluster')}/{d.get('op_name')}#{d.get('occurrence')}"),
                     metric(d.get("duration_us")),
                     metric(f"base {d.get('baseline_median_us')} · x{d.get('ratio_vs_baseline')}", ""),
                     esc(f"{d.get('phase')} L{d.get('layer_idx')}"),
                     badge(conf_of(d)), summ(d)])
    out["operator_jitter.json:operator_jitter_candidates"] = table(
        ["comp/cluster/op#occ", "max dur", "baseline/ratio", "loc", "conf", "agent summary"], rows)
    return out


def _theoretical_deviation_tables(td):
    """sub_item / operator_slot 理论偏差候选。"""
    out = {}
    rows = []
    for d in order(td.get("sub_item_deviation_candidates", []), "absolute_gap_us"):
        am = d.get("actual_median_ms")
        tm = d.get("theoretical_median_ms")
        rows.append([esc(f"{d.get('component_type')}/{d.get('sub_item')}"),
                     metric(f"{am*1000:.1f} / {tm*1000:.1f}" if am and tm else "—"),
                     metric(f"x{d.get('wall_over_theoretical_median')} · {d.get('absolute_gap_us')}", ""),
                     badge(conf_of(d)), summ(d)])
    out["theoretical_deviation.json:sub_item_deviation_candidates"] = table(
        ["comp/sub_item", "actual / theory", "ratio / gap", "conf", "agent summary"], rows)

    rows = []
    for d in order(td.get("operator_slot_deviation_candidates", []), "absolute_gap_us_median"):
        t0 = (d.get("top_locations") or [{}])[0]
        rows.append([esc(f"{d.get('component_type')}/{d.get('cluster')}/{d.get('op_name')}#{d.get('occurrence')}"),
                     metric(f"x{d.get('duration_over_theoretical_median')}", ""),
                     metric(d.get("absolute_gap_us_median")),
                     esc(f"org{t0.get('org_index')} {t0.get('duration_us')}/{t0.get('theoretical_us')}µs"),
                     badge(conf_of(d)), summ(d)])
    out["theoretical_deviation.json:operator_slot_deviation_candidates"] = table(
        ["comp/cluster/op#occ", "dur/theory", "gap median", "rep loc", "conf", "agent summary"], rows)
    return out


def _vector_sequence_table(vs):
    """vector_sequence_candidates patterns。"""
    rows = []
    for d in order(vs.get("patterns", []), "total_duration_us"):
        s0 = (d.get("representative_samples") or [{}])[0]
        ar = d.get("agent_review") or {}
        loc_parts = []
        if s0.get("phase") is not None or s0.get("layer_idx") is not None:
            loc_parts.append(f"{s0.get('phase')} L{s0.get('layer_idx')}")
        cc = "/".join(y for y in [s0.get("component_type"), s0.get("cluster")] if y)
        if cc:
            loc_parts.append(cc)
        loc = " ".join(loc_parts) or "—"
        rows.append([f"<code>{esc(d.get('pattern_signature'))}</code>",
                     metric(f"{d.get('occurrences')}x / {d.get('total_duration_us')}", ""),
                     esc(",".join((d.get("components") or [])[:2])),
                     esc(ar.get("fusion_candidate") or "—"), badge(conf_of(d)),
                     esc(ar.get("semantic_summary") or "")
                     + f" <span class='muted'>[{esc(loc)}]</span>"])
    return {"vector_sequence_candidates.json:patterns": table(
        ["pattern_signature", "occ / total", "modules", "fusion?", "conf", "semantic summary"], rows)}


def _data_movement_table(dm):
    """data_movement: one row per family, represented by its HIGHEST-confidence
    by_module (not just the largest-duration one) so an agent's medium/high
    judgment on a smaller-but-interesting module isn't hidden behind a large
    un-annotated module. Tie-break by module duration.
    """
    fam_rows = []
    for f in dm.get("families", []):
        mods = f.get("by_module") or [{}]
        b0 = min(mods, key=lambda b: (CONF.get(conf_of(b), 3),
                                      -(b.get("total_duration_us") or 0)))
        fam_rows.append({"family": f.get("family"), "f": f, "b0": b0,
                         "total_duration_us": f.get("total_duration_us"),
                         "agent_review": b0.get("agent_review", {})})
    kindmap = {fx.get("family"): fx.get("movement_kind")
               for fx in (dm.get("agent_taxonomy", {}) or {}).get("families", [])}
    rows = []
    for fr in order(fam_rows, "total_duration_us"):
        f, b0 = fr["f"], fr["b0"]
        ar = b0.get("agent_review", {})
        rows.append([esc(f"{f.get('family')} → {b0.get('component_type','-')}/{b0.get('cluster','-')}"),
                     metric(f"{f.get('occurrences')}x / {f.get('total_duration_us')}", ""),
                     esc(kindmap.get(f.get("family")) or "—"),
                     esc(ar.get("elimination_direction") or "—"),
                     badge(conf_of(b0)), esc(ar.get("reason") or "")])
    return {"data_movement_ops.json:families": table(
        ["family → module", "count / total", "kind", "direction", "conf", "redundancy reason"], rows)}


def build_tables(ins):
    tables = {}
    tables.update(_module_bubble_tables(ins.get("module_bubble.json", {})))
    tables.update(_operator_jitter_tables(ins.get("operator_jitter.json", {})))
    tables.update(_theoretical_deviation_tables(ins.get("theoretical_deviation.json", {})))
    tables.update(_vector_sequence_table(ins.get("vector_sequence_candidates.json", {})))
    tables.update(_data_movement_table(ins.get("data_movement_ops.json", {})))
    return tables


def parse_conclusions(md_text):
    """从 final_conclusions.md 抽 High/Medium/Low 核心结论 + 数据限制（纯解析，不造）。"""
    findings = {"high": [], "medium": [], "low": []}
    limits = []
    section = None
    grp = None
    for line in (md_text or "").splitlines():
        s = line.strip()
        mh = re.match(r"^#{2,3}\s*(High|Medium|Low)\b", s, re.I)
        if mh:
            grp = mh.group(1).lower()
            section = "findings"
            continue
        if re.match(r"^##\s*数据限制", s):
            section = "limits"
            grp = None
            continue
        if re.match(r"^##\s", s):  # any other h2 ends the current section
            section = None         # (### sub-headers don't match ^##\s)
            grp = None
        if s.startswith("- ") or s.startswith("* "):
            txt = s[2:].strip()
            if section == "findings" and grp in findings:
                findings[grp].append(txt)
            elif section == "limits":
                limits.append(txt)
    return findings, limits


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--insights-dir", required=True)
    ap.add_argument("--model-name", default="model")
    ap.add_argument("--label", default="")
    ap.add_argument("--generated-at", default="")
    ap.add_argument("--template", default=DEF_TEMPLATE)
    ap.add_argument("-o", "--output", required=True)
    return ap.parse_args()


def read_insights(insights_dir):
    """读取五类 insight JSON（缺失则空 dict）。"""
    ins = {}
    for fn in ["module_bubble.json", "operator_jitter.json", "theoretical_deviation.json",
               "vector_sequence_candidates.json", "data_movement_ops.json"]:
        p = os.path.join(insights_dir, fn)
        ins[fn] = load(p) if os.path.exists(p) else {}
    return ins


def read_conclusions_md(insights_dir):
    """读取 final_conclusions.md（缺失则空字符串）。"""
    mdp = os.path.join(insights_dir, "final_conclusions.md")
    if not os.path.exists(mdp):
        return ""
    with open(mdp, encoding="utf-8") as f:
        return f.read()


def fill_placeholders(tpl, args):
    """替换模板里的标量占位符。"""
    replacements = {"{{model_name}}": args.model_name, "{{label}}": args.label,
                    "{{generated_at}}": args.generated_at,
                    "{{metrics_path}}": "../metrics.json", "{{raw_ops_path}}": "raw_ops.json",
                    "{{structure_draft_path}}": "structure_draft.json"}
    for k, v in replacements.items():
        tpl = tpl.replace(k, v)
    return tpl


def fill_tables(tpl, tables):
    """把 <table data-source="..."></table> 占位替换为渲染好的表体。"""
    def fill(mo):
        src = re.search(r'data-source="([^"]+)"', mo.group(0)).group(1)
        inner = tables.get(src) or table([src], [])
        return f'<table data-source="{src}">{inner}</table>'
    return re.sub(r'<table data-source="[^"]+"[^>]*></table>', fill, tpl)


def fill_finding_cards(tpl, findings):
    """渲染 high/medium/low 结论卡。"""
    def card(grp):
        items = findings.get(grp) or []
        lis = "".join(f"<li>{esc(x)}</li>" for x in items) or "<li class='muted'>—</li>"
        return f'<span class="confidence {grp}">{grp}</span><ul>{lis}</ul>'
    for grp in ("high", "medium", "low"):
        placeholder = (f'<div class="finding-card" data-confidence-group="{grp}">'
                       f'<span class="confidence {grp}">{grp}</span>'
                       f'<!-- {grp} findings --></div>')
        rendered = f'<div class="finding-card" data-confidence-group="{grp}">{card(grp)}</div>'
        tpl = tpl.replace(placeholder, rendered)
    return tpl


def fill_limits(tpl, limits):
    """渲染数据限制小节。"""
    if limits:
        lim_html = "<ul>" + "".join(f"<li>{esc(x)}</li>" for x in limits) + "</ul>"
    else:
        lim_html = "<p class='muted'>—</p>"
    return tpl.replace('<div class="empty"><!-- render final_conclusions.md 数据限制 --></div>',
                       f"<div>{lim_html}</div>")


def fill_source_list(tpl):
    """渲染证据来源列表。"""
    ev = ("<ul><li>主报告: <code>../index.html</code></li>"
          "<li>切片证据: <code>../../../splits/</code>"
          "（manifest.json 索引每个 component / unmatched 段）</li>"
          "<li>op 定位: org_index(Excel 行=org_index+2) / op_idx(致密 index) / op_range_envelope</li>"
          "<li>结论与 Top5 证据表: <code>final_conclusions.md</code></li></ul>")
    placeholder = ('<div class="source-list"><!-- Step 3 HTML, split dirs, '
                   'op_indices / op_range_envelope / org_index links --></div>')
    return tpl.replace(placeholder, f'<div class="source-list">{ev}</div>')


def render(args):
    """构建 insight HTML 并写入输出路径，返回渲染统计。"""
    ins = read_insights(args.insights_dir)
    md = read_conclusions_md(args.insights_dir)

    tables = build_tables(ins)
    findings, limits = parse_conclusions(md)

    with open(args.template, encoding="utf-8") as f:
        tpl = f.read()
    tpl = fill_placeholders(tpl, args)
    tpl = fill_tables(tpl, tables)
    tpl = fill_finding_cards(tpl, findings)
    tpl = fill_limits(tpl, limits)
    tpl = fill_source_list(tpl)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(tpl)
    return tpl, tables, findings


def main():
    args = parse_args()
    tpl, tables, findings = render(args)
    leftover = tpl.count("{{")
    logger.info("render_insights → %s (%d bytes, tables=%d, "
                "findings h/m/l=%d/%d/%d, placeholders_left=%d)",
                args.output, len(tpl), len(tables),
                len(findings["high"]), len(findings["medium"]), len(findings["low"]),
                leftover)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        main()
    except Exception as exc:  # 顶层 CLI 入口兜底，转非零退出码
        logger.error("render_insights failed: %s", exc)
        sys.exit(1)
