#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Step 3 Task D Contract Gate.

This verifier checks one final contract: S2P2_cases.json must fully and
legally cover the path / tilingkey / param_def entry declarations produced by
Task D. It does not inspect S2P2_gen_cases.py implementation details and does
not review test design quality.

Usage:
  python3 s3_task_d_gate.py \
    --output-dir /path/to/tests/whitebox \
    [--report-json S3_verification_report.json] \
    [--report-md S3_verification_report.md]
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

_logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    status: str
    issues: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def pct(covered: int, total: int) -> str:
    if total == 0:
        return "100%"
    return f"{covered * 100 / total:.2f}%"


def dtype_param(param_def: Dict[str, Any]) -> str:
    dtype_tensors = param_def.get("dtype_tensors", [])
    if not dtype_tensors or not isinstance(dtype_tensors[0], dict):
        return ""
    param = dtype_tensors[0].get("param")
    return param if isinstance(param, str) else ""


def _process_dtype_entries(
    group_id: str,
    dtype_name: str,
    dtype_entries: Any,
    entries: Set[Tuple[str, str, str, int]],
    paths: Set[str],
) -> None:
    if not isinstance(dtype_name, str) or not isinstance(dtype_entries, list):
        return
    for entry in dtype_entries:
        if not isinstance(entry, dict):
            continue
        path = entry.get("path")
        key = entry.get("key")
        if isinstance(path, str) and isinstance(key, int):
            entries.add((group_id, dtype_name, path, key))
            paths.add(path)


def collect_param_def_entries(param_def: Dict[str, Any]) -> Tuple[Set[Tuple[str, str, str, int]], Set[str]]:
    entries: Set[Tuple[str, str, str, int]] = set()
    paths: Set[str] = set()
    for group in param_def.get("groups", []):
        if not isinstance(group, dict):
            continue
        group_id = group.get("id")
        per_dtype = group.get("per_dtype", {})
        if not isinstance(group_id, str) or not isinstance(per_dtype, dict):
            continue
        for dtype_name, dtype_entries in per_dtype.items():
            _process_dtype_entries(group_id, dtype_name, dtype_entries, entries, paths)
    return entries, paths


def _normalize_expected_entries(
    entries: Set[Tuple[str, str, str, int]],
    dtype_field: str,
) -> Set[Tuple[str, str, str, int]]:
    if dtype_field:
        return entries
    return {(group, "", path, key) for group, _dtype, path, key in entries}


@dataclass
class _ExpectedSets:
    path_ids: Set[str] = field(default_factory=set)
    reachable_paths: Set[str] = field(default_factory=set)
    expected_keys: Set[int] = field(default_factory=set)
    expected_entries: Set[Tuple[str, str, str, int]] = field(default_factory=set)
    param_def_paths: Set[str] = field(default_factory=set)
    dtype_field: str = ""


def _extract_expected_sets(path_list: Dict[str, Any], param_def: Dict[str, Any]) -> _ExpectedSets:
    paths = path_list.get("paths", []) if isinstance(path_list, dict) else []
    path_ids = {p.get("id") for p in paths if isinstance(p, dict) and isinstance(p.get("id"), str)}
    reachable_paths = {
        p.get("id")
        for p in paths
        if isinstance(p, dict) and isinstance(p.get("id"), str) and p.get("reachability") == "reachable"
    }
    raw_keys = param_def.get("tiling_keys", [])
    expected_keys = {k for k in (raw_keys if isinstance(raw_keys, list) else []) if isinstance(k, int)}
    expected_entries, param_def_paths = collect_param_def_entries(param_def)
    dtype_field = dtype_param(param_def)
    expected_entries = _normalize_expected_entries(expected_entries, dtype_field)
    return _ExpectedSets(
        path_ids=path_ids,
        reachable_paths=reachable_paths,
        expected_keys=expected_keys,
        expected_entries=expected_entries,
        param_def_paths=param_def_paths,
        dtype_field=dtype_field,
    )


@dataclass
class _CaseData:
    case_paths: Set[str] = field(default_factory=set)
    case_keys: Set[int] = field(default_factory=set)
    case_entries: Set[Tuple[str, str, str, int]] = field(default_factory=set)
    unknown_paths: Set[str] = field(default_factory=set)
    unknown_keys: Set[int] = field(default_factory=set)
    unknown_entries: Set[Tuple[str, str, str, int]] = field(default_factory=set)
    malformed: List[Dict[str, Any]] = field(default_factory=list)


def _validate_single_case(idx: int, case: Any, exp: _ExpectedSets, cd: _CaseData) -> None:
    if not isinstance(case, dict):
        cd.malformed.append({"index": idx, "problem": "case must be object", "found": type(case).__name__})
        return
    path = case.get("path")
    key = case.get("key")
    group = case.get("_group")
    dtype_value = case.get(exp.dtype_field) if exp.dtype_field else ""
    missing = []
    if not isinstance(path, str):
        missing.append("path")
    if not isinstance(key, int):
        missing.append("key")
    if not isinstance(group, str):
        missing.append("_group")
    if exp.dtype_field and not isinstance(dtype_value, str):
        missing.append(exp.dtype_field)
    if missing:
        cd.malformed.append({
            "index": idx,
            "problem": "missing or invalid required case fields",
            "fields": missing,
        })
        return
    cd.case_paths.add(path)
    cd.case_keys.add(key)
    entry = (group, dtype_value, path, key)
    cd.case_entries.add(entry)
    if path not in exp.path_ids:
        cd.unknown_paths.add(path)
    if key not in exp.expected_keys:
        cd.unknown_keys.add(key)
    if entry not in exp.expected_entries:
        cd.unknown_entries.add(entry)


def _build_coverage_issues(cd: _CaseData, exp: _ExpectedSets) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    missing_reachable = sorted(exp.reachable_paths - cd.case_paths)
    missing_pd_paths = sorted(exp.param_def_paths - cd.case_paths)
    missing_keys = sorted(exp.expected_keys - cd.case_keys)
    missing_entries = sorted(exp.expected_entries - cd.case_entries)
    if cd.malformed:
        issues.append({"problem": "malformed cases found", "detail": cd.malformed[:20], "count": len(cd.malformed)})
    if cd.unknown_paths:
        issues.append({"problem": "cases reference unknown paths", "detail": sorted(cd.unknown_paths)})
    if cd.unknown_keys:
        issues.append({"problem": "cases reference unknown tilingkeys", "detail": sorted(cd.unknown_keys)})
    if cd.unknown_entries:
        issues.append({
            "problem": "cases do not map to param_def entries",
            "detail": [list(v) for v in sorted(cd.unknown_entries)[:20]],
            "count": len(cd.unknown_entries),
        })
    if missing_reachable:
        issues.append({"problem": "reachable paths missing from cases", "detail": missing_reachable})
    if missing_pd_paths:
        issues.append({"problem": "param_def paths missing from cases", "detail": missing_pd_paths})
    if missing_keys:
        issues.append({"problem": "tilingkeys missing from cases", "detail": missing_keys})
    if missing_entries:
        issues.append({
            "problem": "param_def entries missing from cases",
            "detail": [list(v) for v in missing_entries[:20]],
            "count": len(missing_entries),
        })
    return issues


def _build_coverage_metrics(cd: _CaseData, cases_len: int, exp: _ExpectedSets) -> Dict[str, Any]:
    covered_reachable = len(exp.reachable_paths & cd.case_paths)
    covered_pd_paths = len(exp.param_def_paths & cd.case_paths)
    covered_keys = len(exp.expected_keys & cd.case_keys)
    covered_entries = len(exp.expected_entries & cd.case_entries)
    return {
        "case_count": cases_len,
        "dtype_param": exp.dtype_field,
        "reachable_path_count": len(exp.reachable_paths),
        "covered_reachable_path_count": covered_reachable,
        "reachable_path_coverage": pct(covered_reachable, len(exp.reachable_paths)),
        "param_def_path_count": len(exp.param_def_paths),
        "covered_param_def_path_count": covered_pd_paths,
        "param_def_path_coverage": pct(covered_pd_paths, len(exp.param_def_paths)),
        "expected_tilingkey_count": len(exp.expected_keys),
        "covered_tilingkey_count": covered_keys,
        "tilingkey_coverage": pct(covered_keys, len(exp.expected_keys)),
        "param_def_entry_count": len(exp.expected_entries),
        "covered_param_def_entry_count": covered_entries,
        "param_def_entry_coverage": pct(covered_entries, len(exp.expected_entries)),
        "missing_reachable_paths": sorted(exp.reachable_paths - cd.case_paths),
        "missing_param_def_paths": sorted(exp.param_def_paths - cd.case_paths),
        "unknown_paths": sorted(cd.unknown_paths),
        "missing_tilingkeys": sorted(exp.expected_keys - cd.case_keys),
        "unknown_tilingkeys": sorted(cd.unknown_keys),
        "missing_param_def_entries": [list(v) for v in sorted(exp.expected_entries - cd.case_entries)],
    }


def validate_cases_path_key_coverage(path_list: Dict[str, Any], param_def: Dict[str, Any], cases: Any) -> GateResult:
    exp = _extract_expected_sets(path_list, param_def)
    if not isinstance(cases, list):
        issues = [{"problem": "S2P2_cases.json must be a list", "detail": type(cases).__name__}]
        metrics = _build_coverage_metrics(_CaseData(), 0, exp)
        return GateResult(status="fail", issues=issues, metrics=metrics)
    cd = _CaseData()
    for idx, case in enumerate(cases):
        _validate_single_case(idx, case, exp, cd)
    issues = _build_coverage_issues(cd, exp)
    metrics = _build_coverage_metrics(cd, len(cases), exp)
    return GateResult(status="pass" if not issues else "fail", issues=issues, metrics=metrics)


def render_markdown(report: Dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "---",
        f"op_name: {report['op_name']}",
        f"platform: {report['platform']}",
        f"status: {report['status']}",
        "checks_total: 1",
        f"checks_pass: {1 if report['status'] == 'pass' else 0}",
        f"checks_fail: {1 if report['status'] == 'fail' else 0}",
        "checks_warn: 0",
        "---",
        "",
        f"# Task D 契约门禁报告：{report['op_name']}",
        "",
        "## 总览",
        "",
        "| 项目 | 值 |",
        "|------|---|",
        f"| 算子 | {report['op_name']} |",
        f"| 平台 | {report['platform']} |",
        f"| 全局状态 | **{report['status']}** |",
        f"| case 数量 | {metrics['case_count']} |",
        "",
        "## 覆盖率",
        "",
        "| 覆盖项 | 覆盖数 / 总数 | 覆盖率 |",
        "|--------|-------------|--------|",
        f"| reachable path | {metrics['covered_reachable_path_count']} / "
        f"{metrics['reachable_path_count']} | {metrics['reachable_path_coverage']} |",
        f"| param_def path | {metrics['covered_param_def_path_count']} / "
        f"{metrics['param_def_path_count']} | {metrics['param_def_path_coverage']} |",
        f"| tilingkey | {metrics['covered_tilingkey_count']} / "
        f"{metrics['expected_tilingkey_count']} | {metrics['tilingkey_coverage']} |",
        f"| param_def entry | {metrics['covered_param_def_entry_count']} / "
        f"{metrics['param_def_entry_count']} | {metrics['param_def_entry_coverage']} |",
        "",
    ]
    if report.get("issues"):
        lines.extend([f"## Issues（共 {len(report['issues'])} 项）", ""])
        for idx, issue in enumerate(report["issues"], start=1):
            lines.append(f"### F{idx}: {issue.get('problem', '')}")
            lines.append("")
            lines.append(f"- detail: {issue.get('detail', '')}")
            if "count" in issue:
                lines.append(f"- count: {issue['count']}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(description="Step 3 Task D cases.json coverage gate")
    parser.add_argument("--output-dir", required=True, help="tests/whitebox output dir")
    parser.add_argument("--report-json", default="S3_verification_report.json", help="Output JSON report filename")
    parser.add_argument("--report-md", default="S3_verification_report.md", help="Output Markdown report filename")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    path_list = load_json(output_dir / "S2P1_path_list.json")
    param_def = load_json(output_dir / "S2P2_param_def.json")
    cases = load_json(output_dir / "S2P2_cases.json")

    result = validate_cases_path_key_coverage(path_list, param_def, cases)
    report = {
        "op_name": path_list.get("op_name", path_list.get("operator", "unknown")),
        "platform": param_def.get("platform", "unknown"),
        "status": result.status,
        "scope": "cases_path_key_coverage",
        "check": "cases_path_key_coverage",
        "checks_total": 1,
        "checks_pass": 1 if result.status == "pass" else 0,
        "checks_fail": 1 if result.status == "fail" else 0,
        "checks_warn": 0,
        "metrics": result.metrics,
        "issues": result.issues,
        "inputs": {
            "path_list": str(output_dir / "S2P1_path_list.json"),
            "param_def": str(output_dir / "S2P2_param_def.json"),
            "cases": str(output_dir / "S2P2_cases.json"),
        },
        "limitations": [
            "Does not inspect S2P2_gen_cases.py implementation details.",
            "Does not judge sampling quality or case distribution beyond coverage.",
            "Does not re-review source code semantics.",
        ],
    }

    json_path = output_dir / args.report_json
    md_path = output_dir / args.report_md
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    _logger.info("Wrote: %s", json_path)
    _logger.info("Wrote: %s", md_path)
    _logger.info("Status: %s", result.status)
    _logger.info("reachable path coverage: %s", result.metrics["reachable_path_coverage"])
    _logger.info("tilingkey coverage: %s", result.metrics["tilingkey_coverage"])
    _logger.info("param_def entry coverage: %s", result.metrics["param_def_entry_coverage"])
    return 0 if result.status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
