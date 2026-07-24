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
"""Aggregate multiple render.py metrics.json files into a comparison HTML.

Each row is one sub-item under a component_type (one cluster, the synthetic
`bubble`, or the TOTAL layer-wall). Each column is one run. Cells show
`median_ms (Δ%)` where Δ is relative to the baseline (first column unless
overridden). Outliers per (sub-item, run) are listed inline so the reader
can see how the outlier set shifted between versions.

Inputs:
  -r runs_dir                 directory containing N */metrics.json files;
                              auto-discovered, sorted by mtime ascending.
  --runs path1 path2 …        explicit list of metrics.json paths (overrides
                              -r ordering).
  --baseline LABEL            label of the run to use as Δ baseline. Default:
                              first run (by order).
  --metric NAME               metric to compare. One of:
                                  median_ms (default) / mean_ms / p95_ms /
                                  max_ms / std_ms
  -o output                   output HTML path.
"""

import argparse
import glob
import html
import json
import logging
import os
import sys

logger = logging.getLogger(__name__)


def load_metrics(path):
    with open(path) as f:
        data = json.load(f)
    data["__path"] = path
    return data


def pick_metric(metric_dict, name):
    return float(metric_dict.get(name, 0.0) or 0.0)


def delta_pct(new, base):
    """Return (Δ% as float, formatted string). None when baseline is 0."""
    if base == 0:
        if new == 0:
            return 0.0, "0.0%"
        return None, "new"
    pct = (new - base) / base * 100.0
    sign = "+" if pct > 0 else ""
    return pct, f"{sign}{pct:.1f}%"


def collect_rows(runs, metric):
    """Build flat row table.

    Returns list[dict], each:
      { component_type, sub_item, kind, description,
        per_run: {label -> {value, outliers}} }
    Rows are in first-seen order: spec/cluster order within each section,
    then 'bubble', then 'TOTAL'.
    """
    rows = {}
    order = []

    def ensure(key, **fields):
        if key not in rows:
            rows[key] = {**fields, "per_run": {}}
            order.append(key)
        return rows[key]

    for run in runs:
        label = run["label"]
        for sec in run.get("sections", []):
            ct = sec["component_type"]
            for item in sec.get("sub_items", []):
                key = (ct, item["name"])
                row = ensure(key,
                             component_type=ct,
                             sub_item=item["name"],
                             kind=item["kind"],
                             description=item.get("description", ""))
                row["per_run"][label] = {
                    "value": pick_metric(item["metric"], metric),
                    "outliers": item.get("outliers", []),
                }
            tot = sec.get("total") or {}
            key = (ct, "TOTAL")
            row = ensure(key,
                         component_type=ct,
                         sub_item="TOTAL",
                         kind="total",
                         description=tot.get("description",
                                             "layer wall (union of every op)"))
            row["per_run"][label] = {
                "value": pick_metric(tot.get("metric", {}), metric),
                "outliers": tot.get("outliers", []),
            }
    return [rows[k] for k in order]


def fmt_outlier_set(outs):
    """Compact rendering of an outlier list."""
    if not outs:
        return "—"
    return ", ".join(f"{o['phase']}#{o['layer_idx']}" for o in outs)


def _format_outlier_keys(items):
    """Render a sorted (phase, layer_idx) list as 'phase#layer' strings."""
    return [f"{p}#{i}" for p, i in items]


def classify_outlier_changes(base_outs, new_outs):
    """Return (added, removed, kept) lists of 'phase#layer' strings."""
    base_set = {(o["phase"], o["layer_idx"]) for o in base_outs}
    new_set = {(o["phase"], o["layer_idx"]) for o in new_outs}
    added = sorted(new_set - base_set)
    removed = sorted(base_set - new_set)
    kept = sorted(new_set & base_set)
    return (
        _format_outlier_keys(added),
        _format_outlier_keys(removed),
        _format_outlier_keys(kept),
    )


_CSS = """
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           color: #222; max-width: 1500px; margin: 24px auto; padding: 0 16px; }
    h1 { margin-bottom: 4px; }
    .meta { color: #666; font-size: 13px; margin-bottom: 18px; }
    table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 12px; }
    th, td { border: 1px solid #ddd; padding: 5px 8px; vertical-align: top; }
    th { background: #f5f5f5; }
    th.num, td.num { text-align: right; }
    th.label, td.label, td.desc { text-align: left; }
    td.desc { color: #888; font-size: 11px; }
    tr.section-head td { background: #eef3f8; font-weight: bold; text-align: left;
                         font-size: 12px; padding: 5px 8px; }
    tr.bubble td.label { font-style: italic; }
    tr.total { background: #f9f9f9; font-weight: bold; }
    .pos { color: #b04040; }
    .neg { color: #2d8a4f; }
    .zero { color: #777; }
    .cell { display: block; min-width: 100px; }
    .cell .ms { font-weight: 500; }
    .cell .delta { font-size: 11px; margin-left: 4px; }
    .cell .outliers { display: block; color: #888; font-size: 10.5px; margin-top: 2px;
                       max-width: 220px; word-break: break-word; line-height: 1.35; }
    .cell .outliers .tag { display: inline-block; margin-right: 6px; white-space: nowrap; }
    .cell .outliers .added { color: #b04040; }
    .cell .outliers .removed { color: #2d8a4f; }
    .cell .outliers .kept { color: #999; }
    .cell .outliers .label { color: #aaa; margin-right: 4px; font-style: italic; }
    .footnote { color: #777; font-size: 11px; margin-top: 16px; }
    """


def _render_head(model_name, baseline_label, metric, num_runs):
    """Build the document head, title, and meta line as a list of HTML parts."""
    metric_label = metric.replace("_ms", "") + " (ms)"
    return [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(model_name)} run history</title>",
        f"<style>{_CSS}</style></head><body>",
        f"<h1>{html.escape(model_name)} — run history</h1>",
        f"<div class='meta'>metric: <b>{html.escape(metric_label)}</b> &middot; "
        f"baseline: <b>{html.escape(baseline_label)}</b> &middot; "
        f"runs: {num_runs} &middot; "
        f"cells: value · Δ% vs baseline · outlier change tags "
        f"(<span class='pos'>+new</span> / <span class='neg'>−gone</span> / "
        f"<span style='color:#999'>=still</span>)</div>",
    ]


def _render_table_header(labels, baseline_label):
    """Build the <thead> row, one column per run label."""
    parts = [
        "<table><thead><tr>",
        "<th class='label'>component / sub-item</th>",
        "<th class='label'>description</th>",
    ]
    for lab in labels:
        marker = " (baseline)" if lab == baseline_label else ""
        parts.append(f"<th class='num'>{html.escape(lab + marker)}</th>")
    parts.append("</tr></thead><tbody>")
    return parts


def _row_class(kind):
    """Map a row kind to its CSS class name ('' when none)."""
    if kind == "total":
        return "total"
    if kind == "bubble":
        return "bubble"
    return ""


def _render_baseline_cell(value, outs):
    """Render a baseline (or base-missing) cell listing the full outlier set."""
    if outs:
        body = (f"<span class='label'>outliers:</span>"
                f"<span class='kept'>{html.escape(fmt_outlier_set(outs))}</span>")
    else:
        body = "<span class='label'>outliers:</span>—"
    return (
        f"<td class='num'><span class='cell'>"
        f"<span class='ms'>{value:.3f}</span>"
        f"<span class='outliers'>{body}</span></span></td>"
    )


def _render_outlier_tags(base_outs, outs):
    """Build the outlier-change tag span comparing base vs this cell."""
    added, removed, kept = classify_outlier_changes(base_outs, outs)
    tags = []
    for s in added:
        tags.append(f"<span class='tag added'>+{html.escape(s)}</span>")
    for s in removed:
        tags.append(f"<span class='tag removed'>−{html.escape(s)}</span>")
    for s in kept:
        tags.append(f"<span class='tag kept'>={html.escape(s)}</span>")
    if not tags:
        return ""
    return "<span class='outliers'>" + "".join(tags) + "</span>"


def _render_compare_cell(value, outs, base):
    """Render a non-baseline cell with Δ% and outlier-change tags vs base."""
    _, d_str = delta_pct(value, base["value"])
    dcls = "pos" if d_str.startswith("+") else "neg" if d_str.startswith("-") else "zero"
    outs_html = _render_outlier_tags(base.get("outliers", []), outs)
    return (
        f"<td class='num'><span class='cell'>"
        f"<span class='ms'>{value:.3f}</span>"
        f"<span class='delta {dcls}'>{html.escape(d_str)}</span>"
        f"{outs_html}</span></td>"
    )


def _render_cell(cell, lab, baseline_label, base):
    """Render one table cell for a given run label."""
    if cell is None:
        return "<td class='num'>—</td>"
    value = cell["value"]
    outs = cell.get("outliers", [])
    if lab == baseline_label or base is None:
        return _render_baseline_cell(value, outs)
    return _render_compare_cell(value, outs, base)


def _render_row(row, labels, baseline_label):
    """Render one data <tr> (cells across all run labels) as HTML parts."""
    cls_attr = ""
    row_cls = _row_class(row["kind"])
    if row_cls:
        cls_attr = f" class='{row_cls}'"
    parts = [
        f"<tr{cls_attr}>",
        f"<td class='label'><b>{html.escape(row['sub_item'])}</b></td>",
        f"<td class='desc'>{html.escape(row['description'])}</td>",
    ]
    base = row["per_run"].get(baseline_label)
    for lab in labels:
        cell = row["per_run"].get(lab)
        parts.append(_render_cell(cell, lab, baseline_label, base))
    parts.append("</tr>")
    return parts


def _render_body_rows(rows, labels, baseline_label):
    """Render all data rows, inserting a section head when component_type changes."""
    parts = []
    current_ct = None
    for row in rows:
        if row["component_type"] != current_ct:
            current_ct = row["component_type"]
            parts.append(
                f"<tr class='section-head'><td colspan='{2 + len(labels)}'>"
                f"{html.escape(current_ct)}</td></tr>"
            )
        parts.extend(_render_row(row, labels, baseline_label))
    return parts


def render_html(runs, rows, baseline_label, metric):
    """Aggregate runs/rows into the comparison HTML document string."""
    model_name = runs[0].get("model_name", "model")
    labels = [r["label"] for r in runs]

    parts = _render_head(model_name, baseline_label, metric, len(runs))
    parts.extend(_render_table_header(labels, baseline_label))
    parts.extend(_render_body_rows(rows, labels, baseline_label))
    parts.append("</tbody></table>")
    parts.append(
        "<div class='footnote'>Outlier method: IQR (k=1.5) by default. "
        "Baseline cell lists this run's full outlier set; comparison cells "
        "list only the changes vs baseline.</div>"
    )
    parts.append("</body></html>")
    return "\n".join(parts)


def build_arg_parser_args():
    """Build the argument parser and return the parsed CLI arguments."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("-r", "--runs-dir",
                     help="directory holding N */metrics.json files")
    src.add_argument("--runs", nargs="+",
                     help="explicit list of metrics.json paths in order")
    p.add_argument("--baseline",
                   help="label of baseline run (default: first run by order)")
    p.add_argument("--metric", default="median_ms",
                   choices=["median_ms", "mean_ms", "p95_ms", "max_ms", "std_ms"])
    p.add_argument("-o", "--output", required=True, help="output HTML path")
    return p.parse_args()


def _resolve_paths(args):
    """Resolve the ordered list of metrics.json paths from CLI args."""
    if args.runs_dir:
        paths = sorted(
            glob.glob(os.path.join(args.runs_dir, "*", "metrics.json")),
            key=os.path.getmtime,
        )
        if not paths:
            raise ValueError(f"no */metrics.json found under {args.runs_dir!r}")
        return paths
    return args.runs


def _resolve_baseline(labels, requested):
    """Validate labels are unique and pick/validate the baseline label."""
    if len(set(labels)) != len(labels):
        dupes = sorted({lab for lab in labels if labels.count(lab) > 1})
        raise ValueError(f"duplicate labels in runs: {dupes}")
    baseline = requested or labels[0]
    if baseline not in labels:
        raise ValueError(f"baseline {baseline!r} not in run labels {labels}")
    return baseline


def main():
    args = build_arg_parser_args()

    paths = _resolve_paths(args)
    runs = [load_metrics(p_) for p_ in paths]
    labels = [r["label"] for r in runs]
    baseline = _resolve_baseline(labels, args.baseline)

    rows = collect_rows(runs, args.metric)
    out_html = render_html(runs, rows, baseline, args.metric)
    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(out_html)
    logger.info(
        "history → %s (%d runs, baseline=%s, metric=%s)",
        args.output, len(runs), baseline, args.metric,
    )
    for r in runs:
        logger.info("  %-20s ← %s", r["label"], r["__path"])


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    try:
        main()
    except Exception as exc:  # 顶层 CLI 入口兜底，转非零退出码
        logger.error("%s", exc)
        sys.exit(1)
