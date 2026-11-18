#!/usr/bin/env python3
"""Render normalized operator test records as offline HTML and portable CSV."""

import argparse
import csv
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote

SCHEMA_VERSION = "1.0"
SUITES = ("function", "accuracy", "memory", "performance", "determinism")
STATUSES = (
    "PASS", "FAIL", "ERROR", "TIMEOUT", "UNSUPPORTED", "NOT_RUN",
    "NOT_APPLICABLE", "BLOCKED", "MEASURED",
)
EVIDENCE_CLASSES = {
    "synthetic": "合成示例",
    "hardware": "设备测试记录",
    "unverified": "未经验证的记录",
}
KEY_FIELDS = ("case_id", "phase", "suite", "backend", "reference")
RESULT_FIELDS = (
    "schema_version", "run_id", "standard_id", "evidence_class", "case_id",
    "operator", "phase", "suite", "backend", "reference", "status", "detail",
    "evidence_json", "reproduction", "metadata_json",
)
METRIC_FIELDS = (
    "schema_version", "run_id", *KEY_FIELDS, "operator", "name", "value", "unit", "limit",
)


def _object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _array(value, label):
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _string(value, label, allow_empty=False):
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{label} must be a {'possibly empty' if allow_empty else 'non-empty'} string")
    if "\0" in value:
        raise ValueError(f"{label} contains a null character")
    return value


def _number(value, label):
    is_integer = type(value) is int
    is_finite_float = type(value) is float and math.isfinite(value)
    if not (is_integer or is_finite_float):
        raise ValueError(f"{label} must be a finite number")


def _unique_json_keys(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON key: {name}")
        result[name] = value
    return result


def _json(value, indent=None):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=indent)


def _artifact_path(value, root, label):
    path = _string(value, label)
    parts = path.split("/")
    if (
        any(part in {"", ".", ".."} for part in parts)
        or any(char in path for char in "\\:")
        or any(ord(char) < 32 or ord(char) == 127 for char in path)
    ):
        raise ValueError(f"{label} must be a safe POSIX relative path")
    resolved = (root / path).resolve()
    if root not in resolved.parents:
        raise ValueError(f"{label} resolves outside the bundle directory")
    return resolved


def validate(bundle, root):
    """Check the reporting schema without evaluating any acceptance standard."""
    _object(bundle, "bundle")
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    run = _object(bundle.get("run"), "run")
    for field in ("id", "standard_id", "evidence_class"):
        _string(run.get(field), f"run.{field}")
    if run["evidence_class"] not in EVIDENCE_CLASSES:
        raise ValueError("run.evidence_class must be synthetic, hardware, or unverified")
    _object(run.get("metadata", {}), "run.metadata")
    _json(run)

    rows = _array(bundle.get("results"), "results")
    seen = set()
    for index, row in enumerate(rows):
        label = f"results[{index}]"
        _object(row, label)
        for field in (*KEY_FIELDS, "operator", "status"):
            _string(row.get(field), f"{label}.{field}", allow_empty=field == "reference")
        _string(row.get("detail", ""), f"{label}.detail", allow_empty=True)
        if row["suite"] not in SUITES or row["status"] not in STATUSES:
            raise ValueError(f"{label} has an unknown suite or status")
        key = tuple(row[field] for field in KEY_FIELDS)
        if key in seen:
            raise ValueError(f"{label} duplicates result key {key!r}")
        seen.add(key)
        _json(_object(row.get("metadata", {}), f"{label}.metadata"))

        names = set()
        metrics = _array(row.get("metrics", []), f"{label}.metrics")
        for metric_index, metric in enumerate(metrics):
            metric_label = f"{label}.metrics[{metric_index}]"
            _object(metric, metric_label)
            name = _string(metric.get("name"), f"{metric_label}.name")
            if name in names:
                raise ValueError(f"{label} has duplicate metric name {name!r}")
            names.add(name)
            _string(metric.get("unit"), f"{metric_label}.unit", allow_empty=True)
            _number(metric.get("value"), f"{metric_label}.value")
            if metric.get("limit") is not None:
                _number(metric["limit"], f"{metric_label}.limit")

        for path in _array(row.get("evidence", []), f"{label}.evidence"):
            _artifact_path(path, root, f"{label}.evidence")
        if row.get("reproduction") is not None:
            _artifact_path(row["reproduction"], root, f"{label}.reproduction")

    return run, rows


def _csv_cell(value):
    if isinstance(value, str):
        leading = value.lstrip()
        if value.startswith(("\t", "\r", "\n", "'")) or leading.startswith(("=", "+", "-", "@")):
            return "'" + value
    return value


def _write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        for row in rows:
            writer.writerow([_csv_cell(row.get(field, "")) for field in fields])


def _result_rows(run, rows):
    for row in rows:
        yield {
            **row,
            "schema_version": SCHEMA_VERSION,
            "run_id": run["id"],
            "standard_id": run["standard_id"],
            "evidence_class": run["evidence_class"],
            "evidence_json": _json(row.get("evidence", [])),
            "metadata_json": _json(row.get("metadata", {})),
        }


def _metric_rows(run, rows):
    for row in rows:
        for metric in row.get("metrics", []):
            yield {
                **{field: row[field] for field in (*KEY_FIELDS, "operator")},
                **{field: metric.get(field) for field in ("name", "value", "unit", "limit")},
                "schema_version": SCHEMA_VERSION,
                "run_id": run["id"],
            }


def summarize(run, rows, root):
    counts = {name: dict.fromkeys(STATUSES, 0) for name in SUITES}
    missing = set()
    for row in rows:
        counts[row["suite"]][row["status"]] += 1
        paths = list(row.get("evidence", []))
        if row.get("reproduction"):
            paths.append(row["reproduction"])
        for path in paths:
            if not _artifact_path(path, root, "artifact").is_file():
                missing.add(path)
    return {
        "schema_version": SCHEMA_VERSION,
        "run": run,
        "case_count": len({row["case_id"] for row in rows}),
        "result_count": len(rows),
        "metric_count": sum(len(row.get("metrics", [])) for row in rows),
        "status_counts": {status: sum(counts[suite][status] for suite in SUITES) for status in STATUSES},
        "suites": counts,
        "missing_artifacts": sorted(missing),
    }


def _text(value):
    return html.escape(str(value), quote=True)


def _artifact_link(path, root, output):
    target = _artifact_path(path, root, "artifact")
    if not target.is_file():
        return f'<span class="missing">{_text(path)}（缺失）</span>'
    relative = Path(os.path.relpath(target, output)).as_posix()
    href = _text(quote(relative, safe="/"))
    return f'<a href="{href}">{_text(path)}</a>'


def _summary_table(summary):
    headings = "".join(f"<th scope=\"col\">{name}</th>" for name in STATUSES)
    rows = []
    for suite, counts in summary["suites"].items():
        cells = "".join(f"<td>{counts[status]}</td>" for status in STATUSES)
        rows.append(f"<tr><th scope=\"row\">{suite}</th>{cells}</tr>")
    return (
        '<div class="scroll"><table class="summary"><thead><tr><th scope="col">专项</th>'
        f"{headings}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _select(label, name, options):
    values = ['<option value="">全部</option>']
    for option in options:
        escaped = _text(option)
        values.append(f'<option value="{escaped}">{escaped}</option>')
    return f'<label>{label}<select id="{name}">{"".join(values)}</select></label>'


def _details(row, root, output):
    sections = []
    if row.get("detail"):
        sections.append(f'<p class="note">{_text(row["detail"])}</p>')
    if row.get("metrics"):
        items = []
        for metric in row["metrics"]:
            value = _text(metric["value"])
            limit = metric.get("limit")
            suffix = "" if limit is None else f"（阈值 {_text(limit)}）"
            items.append(f'<dt>{_text(metric["name"])}</dt><dd>{value} {_text(metric["unit"])}{suffix}</dd>')
        sections.append(f'<dl class="metrics">{"".join(items)}</dl>')
    if row.get("metadata"):
        sections.append(f'<pre>{_text(_json(row["metadata"], indent=2))}</pre>')
    if row.get("evidence"):
        items = "".join(f"<li>{_artifact_link(path, root, output)}</li>" for path in row["evidence"])
        sections.append(f"<ul>{items}</ul>")
    if not sections:
        return '<span class="muted">无</span>'
    return f'<details><summary>查看详情</summary>{"".join(sections)}</details>'


def _html_row(row, root, output):
    cells = [f"<td>{_text(row[field])}</td>" for field in ("case_id", "operator", "phase", "suite", "backend", "reference")]
    status = _text(row["status"])
    cells.append(f'<td><span class="status {status}">{status}</span></td>')
    replay = row.get("reproduction")
    cells.append(f'<td>{_artifact_link(replay, root, output) if replay else "—"}</td>')
    cells.append(f"<td>{_details(row, root, output)}</td>")
    return f"<tr>{''.join(cells)}</tr>\n"


STYLE = """\
:root { color-scheme: light; --ink:#142536; --muted:#526677; --line:#dce4ec; --panel:#fff; --accent:#125d8e; }
* { box-sizing:border-box; }
body { margin:0; color:var(--ink); background:#f3f6fa; font:16px/1.55 Inter, system-ui, sans-serif; }
main { max-width:1480px; margin:auto; padding:clamp(16px,4vw,48px); }
header { border-bottom:2px solid var(--accent); padding-bottom:18px; }
h1 { margin:0 0 10px; font-size:clamp(24px,4vw,40px); letter-spacing:-.03em; }
h2 { margin:28px 0 10px; font-size:20px; }
p { margin:6px 0; }
nav { display:flex; gap:16px; flex-wrap:wrap; margin-top:12px; }
a { color:var(--accent); text-underline-offset:.18em; }
.run-info, .totals, .filters { display:flex; flex-wrap:wrap; gap:12px 24px; }
.run-info { color:var(--muted); }
.totals { margin-top:20px; }
.totals > span { padding:12px 16px; background:var(--panel); border:1px solid var(--line); border-radius:8px; }
.totals strong { margin-inline-end:8px; font-size:22px; font-variant-numeric:tabular-nums; }
.scroll { overflow:auto; border:1px solid var(--line); border-radius:8px; background:var(--panel); }
table { border-collapse:collapse; width:100%; text-align:left; font-size:14px; }
th, td { padding:9px 10px; border-bottom:1px solid var(--line); vertical-align:top; overflow-wrap:anywhere; }
thead th { background:#eaf1f7; white-space:nowrap; }
tbody tr:last-child > * { border-bottom:0; }
tbody tr:nth-child(even) { background:#f8fafd; }
.summary td { font-variant-numeric:tabular-nums; }
.filters { align-items:end; margin-bottom:10px; }
label { display:grid; gap:4px; font-size:14px; }
input, select, button { border:1px solid #9eafbe; border-radius:6px; padding:6px 8px; color:var(--ink); background:#fff; font:inherit; }
:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.status { display:inline-block; border-radius:5px; padding:1px 6px; font-weight:600; background:#e9eef3; }
.status.PASS { background:#d8f4e5; color:#0a5535; }
.status.FAIL,.status.ERROR,.status.TIMEOUT { background:#fce3e3; color:#8b1d1d; }
.status.BLOCKED,.status.UNSUPPORTED { background:#fff0c7; color:#744702; }
.status.MEASURED { background:#dcecf9; color:#154369; }
.muted, .count { color:var(--muted); }
.missing { color:#8b1d1d; }
summary { cursor:pointer; color:var(--accent); }
pre, .note { white-space:pre-wrap; overflow-wrap:anywhere; }
.metrics { display:grid; grid-template-columns:auto 1fr; gap:3px 12px; }
.metrics dd { margin:0; }
@media print { .filters { display:none; } .scroll { overflow:visible; } main { padding:0; } }
"""

FILTER = """\
const rows = Array.from(document.querySelectorAll("#records tbody tr"));
const query = document.getElementById("query");
const suite = document.getElementById("suite");
const status = document.getElementById("status");
const backend = document.getElementById("backend");
const count = document.getElementById("count");
const issueStates = new Set(["FAIL", "ERROR", "TIMEOUT", "BLOCKED"]);
const issues = document.getElementById("issues");
function update() {
  const needle = query.value.trim().toLowerCase();
  let matches = 0;
  for (const row of rows) {
    const cells = row.cells;
    const state = cells[6].textContent;
    const show = (!needle || row.textContent.toLowerCase().includes(needle)) &&
      (!suite.value || cells[3].textContent === suite.value) &&
      (!backend.value || cells[4].textContent === backend.value) &&
      (!status.value || state === status.value) &&
      (!issues.checked || issueStates.has(state));
    row.hidden = !show;
    if (show) matches++;
  }
  count.textContent = '显示 ' + matches + ' / ' + rows.length + ' 条';
}
for (const control of [query, suite, backend, status, issues]) {
  control.addEventListener("input", update);
}
"""


def _write_html(path, run, rows, summary, root):
    run_id = _text(run["id"])
    standard = _text(run["standard_id"])
    evidence = EVIDENCE_CLASSES[run["evidence_class"]]
    missing_count = len(summary["missing_artifacts"])
    labels = ("用例", "算子", "阶段", "专项", "被测后端", "参考后端", "状态", "复现", "详情")
    columns = "".join(f'<th scope="col">{label}</th>' for label in labels)
    filters = "".join((
        _select("专项", "suite", SUITES),
        _select("状态", "status", STATUSES),
        _select("被测后端", "backend", sorted({row["backend"] for row in rows})),
    ))
    metadata = ""
    if run.get("metadata"):
        metadata = f'<details><summary>运行信息</summary><pre>{_text(_json(run["metadata"], indent=2))}</pre></details>'
    with path.open("w", encoding="utf-8") as file:
        file.write(f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>算子测试报告 · {run_id}</title><style>{STYLE}</style></head>
<body><main><header><h1>算子测试报告</h1>
<div class="run-info"><span>运行 {run_id}</span><span>标准 {standard}</span><span>{evidence}</span></div>
<nav aria-label="下载"><a href="results.csv">结果 CSV</a><a href="metrics.csv">指标 CSV</a>
<a href="summary.json">汇总 JSON</a></nav>{metadata}</header>
<div class="totals"><span><strong>{summary["case_count"]}</strong>用例</span>
<span><strong>{summary["result_count"]}</strong>结果</span>
<span><strong>{summary["metric_count"]}</strong>指标</span>
<span><strong>{missing_count}</strong>缺失附件</span></div>
<h2>专项统计</h2>{_summary_table(summary)}
<p class="muted">统计来自输入记录。报告生成器只汇总各条记录的状态。</p>
<h2>用例明细</h2><div class="filters">
<label>搜索用例<input id="query" type="search"></label>{filters}
<label><span><input id="issues" type="checkbox"> 仅失败、错误、超时及阻塞</span></label></div>
<p class="count" id="count" role="status">显示 {len(rows)} / {len(rows)} 条</p>
<div class="scroll"><table id="records"><thead><tr>{columns}</tr></thead><tbody>
""")
        for row in rows:
            file.write(_html_row(row, root, path.parent))
        file.write(f"</tbody></table></div></main><script>{FILTER}</script></body></html>\n")


def render(bundle_path, output):
    bundle_path = bundle_path.resolve(strict=True)
    root = bundle_path.parent
    with bundle_path.open(encoding="utf-8-sig") as source:
        bundle = json.load(source, object_pairs_hook=_unique_json_keys)
    run, rows = validate(bundle, root)
    summary = summarize(run, rows, root)
    # A fresh output directory keeps an earlier report intact if another run fails.
    output.mkdir(parents=True, exist_ok=False)
    output = output.resolve()
    _write_csv(output / "results.csv", RESULT_FIELDS, _result_rows(run, rows))
    _write_csv(output / "metrics.csv", METRIC_FIELDS, _metric_rows(run, rows))
    (output / "summary.json").write_text(_json(summary, indent=2) + "\n", encoding="utf-8")
    _write_html(output / "report.html", run, rows, summary, root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="normalized results JSON")
    parser.add_argument("--out", required=True, type=Path, help="new output directory")
    args = parser.parse_args()
    try:
        render(args.bundle, args.out)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"report: {error}\n")


if __name__ == "__main__":
    main()
