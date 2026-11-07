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
"""Assemble format-check and clause-review YAML into one Markdown report.

Usage:
  python3 workflow.assemble_report.py --dir <yaml_dir> --output <report.md>

The input directory may contain one aggregate ``type: format`` result and any
number of ordinary ``type: clause`` results. Matching the AscendC behavior,
parseable ``_dupN`` files all participate in the report, malformed YAML is
warned about and skipped, and an incorrect ``confidence_value`` is recalculated
and written back.
"""

import argparse
from collections import Counter, defaultdict
import logging
import os
import sys
from typing import Any

import yaml


logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

CLAUSE_STATUSES = {"PASS", "FAIL", "SUSPICIOUS"}
FORMAT_STATUSES = {"PASS", "FAIL", "ERROR", "PARTIAL"}
MAX_LINT_EXAMPLES_PER_FILE_CODE = 3
MAX_FORMAT_DIFF_LINES_PER_FILE = 50


def _md_cell(value: Any) -> str:
    """Escape text for one Markdown table cell."""
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _parse_score(value: Any) -> int:
    """Parse +40%/-15%; invalid values become zero as in AscendC."""
    text = str(value or "").strip().replace("%", "").replace("+", "")
    try:
        return int(text)
    except ValueError:
        try:
            return int(float(text))
        except ValueError:
            return 0


def _confidence_total(data: dict, fname: str) -> int | None:
    """Return the clamped evidence total after validating containers."""
    evidence = data.get("evidence")
    if not evidence:
        return None
    if not isinstance(evidence, dict):
        raise ValueError(
            f"{fname}: evidence must be a mapping, got {type(evidence).__name__}"
        )

    total = 0
    for key in ("positive", "negative"):
        items = evidence.get(key, []) or []
        if not isinstance(items, list):
            raise ValueError(
                f"{fname}: evidence.{key} must be a list, got {type(items).__name__}"
            )
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(
                    f"{fname}: evidence.{key} items must be mappings, got {type(item).__name__}"
                )
            total += _parse_score(item.get("score"))
    return max(0, min(100, total))


def _fix_confidence(data: dict, fname: str) -> bool:
    """Correct confidence_value in memory; return whether it changed."""
    total = _confidence_total(data, fname)
    if total is None:
        return False
    evidence = data["evidence"]
    correct_value = f"{total}%"
    if str(evidence.get("confidence_value", "")).strip() == correct_value:
        return False
    evidence["confidence_value"] = correct_value
    return True


def _writeback_yaml(data: dict, path: str, fname: str) -> None:
    """Write corrected confidence back, matching AscendC behavior."""
    try:
        with open(path, "w", encoding="utf-8") as output_file:
            yaml.safe_dump(
                data,
                output_file,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            )
    except OSError as error:
        logger.warning("置信度修正写回失败: %s (%s)", fname, error)


def _positive_int(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return int(value) >= 1
    except (TypeError, ValueError):
        return False


def _validate_clause(data: dict, fname: str) -> list[str]:
    """Validate a clause result after confidence correction."""
    errors = []
    required = (
        "clause_id",
        "clause_title",
        "canonical_id",
        "submission_key",
        "status",
    )
    for field in required:
        if not data.get(field):
            errors.append(f"{fname}: 缺少必填字段 {field}")

    status = data.get("status")
    if status not in CLAUSE_STATUSES:
        errors.append(f"{fname}: status 非法: {status!r}")
        return errors
    if status == "PASS":
        if "evidence" in data or "confidence" in data:
            errors.append(f"{fname}: PASS 禁止携带 evidence/confidence")
        return errors

    for field in ("confidence", "problem_desc", "fix_suggestion"):
        if not data.get(field):
            errors.append(f"{fname}: {status} 缺少必填字段 {field}")

    snippet = data.get("code_snippet")
    if not isinstance(snippet, dict):
        errors.append(f"{fname}: code_snippet 必须是 mapping")
    else:
        if not snippet.get("file_path"):
            errors.append(f"{fname}: code_snippet.file_path 缺失")
        if not snippet.get("code"):
            errors.append(f"{fname}: code_snippet.code 缺失")
        start_valid = _positive_int(snippet.get("start_line"))
        end_valid = _positive_int(snippet.get("end_line"))
        if not start_valid:
            errors.append(f"{fname}: code_snippet.start_line 必须是一基正整数")
        if not end_valid:
            errors.append(f"{fname}: code_snippet.end_line 必须是一基正整数")
        if start_valid and end_valid:
            if int(snippet["end_line"]) < int(snippet["start_line"]):
                errors.append(f"{fname}: code_snippet.end_line 不能小于 start_line")

    try:
        total = _confidence_total(data, fname)
    except ValueError as error:
        errors.append(str(error))
        return errors
    if total is None:
        errors.append(f"{fname}: {status} 缺少 evidence")
        return errors

    if total >= 80:
        expected_status, expected_confidence = "FAIL", "HIGH"
    elif total >= 70:
        expected_status, expected_confidence = "SUSPICIOUS", "MED"
    else:
        expected_status, expected_confidence = "SUSPICIOUS", "LOW"
    if status != expected_status or data.get("confidence") != expected_confidence:
        errors.append(
            f"{fname}: {total}% 应映射为 {expected_status}/{expected_confidence}，实际 {status}/{data.get('confidence')!s}"
        )
    return errors


def _validate_format_section(section: Any, section_name: str, fname: str) -> list[str]:
    """Validate one optional language section in aggregate format YAML."""
    if section is None:
        return []
    if not isinstance(section, dict):
        return [f"{fname}: {section_name} 必须是 mapping"]
    errors = []
    for key in ("files_checked", "lint_issues", "format_issues"):
        if key not in section:
            continue
        value = section[key]
        if key == "files_checked" and isinstance(value, int):
            continue
        if not isinstance(value, list):
            errors.append(f"{fname}: {section_name}.{key} 必须是 list")
    return errors


def _validate_format(data: dict, fname: str) -> list[str]:
    """Validate an aggregate deterministic format-check result."""
    errors = []
    if not data.get("check_id"):
        errors.append(f"{fname}: format YAML 缺少 check_id")
    if data.get("status") not in FORMAT_STATUSES:
        errors.append(f"{fname}: format status 非法: {data.get('status')!r}")
    if "tools" in data and not isinstance(data["tools"], dict):
        errors.append(f"{fname}: tools 必须是 mapping")
    for section_name in ("python", "cpp", "markdown"):
        errors.extend(
            _validate_format_section(data.get(section_name), section_name, fname)
        )
    for key in ("skipped_files", "execution_errors"):
        if key in data and not isinstance(data[key], list):
            errors.append(f"{fname}: {key} 必须是 list")
    return errors


def load_yaml_files(yaml_dir: str) -> tuple[list[dict], list[dict], list[str]]:
    """Load parseable YAML; duplicate files deliberately stay independent."""
    clause_results = []
    format_results = []
    skipped_files = []
    schema_errors = []
    corrected_count = 0

    for fname in sorted(os.listdir(yaml_dir)):
        if not fname.endswith((".yaml", ".yml")):
            continue
        path = os.path.join(yaml_dir, fname)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as input_file:
                data = yaml.safe_load(input_file)
        except (OSError, yaml.YAMLError) as error:
            logger.warning("跳过无法解析的 yaml: %s (%s)", fname, error)
            skipped_files.append(fname)
            continue
        if not isinstance(data, dict):
            skipped_files.append(fname)
            continue

        result_type = data.get("type", "clause")
        if result_type == "format":
            schema_errors.extend(_validate_format(data, fname))
            format_results.append(data)
            continue
        if result_type != "clause":
            schema_errors.append(f"{fname}: 不支持的 YAML type: {result_type!r}")
            continue

        try:
            corrected = _fix_confidence(data, fname)
        except ValueError as error:
            schema_errors.append(str(error))
            corrected = False
        if corrected:
            _writeback_yaml(data, path, fname)
            corrected_count += 1
        schema_errors.extend(_validate_clause(data, fname))
        clause_results.append(data)

    if schema_errors:
        logger.error("%d 个 yaml schema 校验失败:", len(schema_errors))
        for error in schema_errors:
            logger.error("  %s", error)
        raise ValueError(f"{len(schema_errors)} 个 yaml schema 校验失败")
    if corrected_count:
        logger.info("置信度校验修正 %d 个 yaml 的 confidence_value", corrected_count)
    return clause_results, format_results, skipped_files


def _pct(number: int, total: int) -> str:
    return "0%" if total == 0 else f"{number * 100 / total:.1f}%"


def _clause_stats(results: list[dict]) -> dict[str, int]:
    stats = {"total": len(results), "PASS": 0, "FAIL": 0, "SUSPICIOUS": 0}
    for result in results:
        status = result.get("status")
        if status in CLAUSE_STATUSES:
            stats[status] += 1
    return stats


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _files_checked_count(section: dict) -> int:
    files = section.get("files_checked", [])
    if isinstance(files, int):
        return max(0, files)
    return len(files) if isinstance(files, list) else 0


def _format_stats(format_results: list[dict]) -> dict[str, int]:
    stats = {
        "python_files": 0,
        "cpp_files": 0,
        "markdown_files": 0,
        "lint_issues": 0,
        "python_format_issues": 0,
        "cpp_format_issues": 0,
    }
    for result in format_results:
        python = result.get("python") if isinstance(result.get("python"), dict) else {}
        cpp = result.get("cpp") if isinstance(result.get("cpp"), dict) else {}
        markdown = (
            result.get("markdown") if isinstance(result.get("markdown"), dict) else {}
        )
        stats["python_files"] += _files_checked_count(python)
        stats["cpp_files"] += _files_checked_count(cpp)
        stats["markdown_files"] += _files_checked_count(markdown)
        stats["lint_issues"] += len(_as_list(python.get("lint_issues")))
        stats["python_format_issues"] += len(_as_list(python.get("format_issues")))
        stats["cpp_format_issues"] += len(_as_list(cpp.get("format_issues")))
    return stats


def _review_issue_stats(clause_results: list[dict]) -> dict[str, int]:
    stats = {"python": 0, "cpp": 0, "markdown": 0, "other": 0}
    for result in clause_results:
        if result.get("status") == "PASS":
            continue
        snippet = result.get("code_snippet")
        path = str(snippet.get("file_path", "")) if isinstance(snippet, dict) else ""
        extension = os.path.splitext(path)[1].lower()
        if extension in {".py", ".pyi"}:
            stats["python"] += 1
        elif extension in {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".icc"}:
            stats["cpp"] += 1
        elif extension == ".md":
            stats["markdown"] += 1
        else:
            stats["other"] += 1
    return stats


def _issue_path(issue: Any) -> str:
    if isinstance(issue, str):
        return issue
    if not isinstance(issue, dict):
        return ""
    return str(
        issue.get("file_path") or issue.get("filename") or issue.get("file") or ""
    )


def _issue_line(issue: dict) -> Any:
    if issue.get("line") is not None:
        return issue["line"]
    location = issue.get("location")
    return location.get("row", "") if isinstance(location, dict) else ""


def _issue_fix(issue: dict) -> str:
    value = issue.get("fix_suggestion") or issue.get("fix") or "—"
    if isinstance(value, dict):
        return str(value.get("message") or value.get("applicability") or value)
    return str(value)


def _collect_format_items(
    format_results: list[dict], section_name: str, field: str
) -> list[Any]:
    items = []
    for result in format_results:
        section = result.get(section_name)
        if isinstance(section, dict):
            items.extend(_as_list(section.get(field)))
    return items


def _format_item_text(item: Any) -> str:
    """Render one skipped-file or execution-error item as readable text."""
    if not isinstance(item, dict):
        return str(item)
    path = (
        item.get("file_path")
        or item.get("file")
        or item.get("path")
        or item.get("tool")
    )
    reason = item.get("reason") or item.get("message") or item.get("error")
    if path and reason:
        return f"{path}：{reason}"
    return str(path or reason or item)


def _render_format_metadata(lines: list[str], format_results: list[dict]) -> None:
    """Render tool versions plus skipped files and execution failures."""
    tools = []
    skipped = []
    errors = []
    for result in format_results:
        tool_data = result.get("tools")
        if isinstance(tool_data, dict):
            tools.extend(tool_data.items())
        skipped.extend(_as_list(result.get("skipped_files")))
        errors.extend(_as_list(result.get("execution_errors")))

    if tools:
        lines.extend(
            [
                "### 格式检查工具",
                "",
                "| 工具 | 版本/状态 |",
                "|---|---|",
            ]
        )
        for name, detail in tools:
            if isinstance(detail, dict):
                version = detail.get("version")
                status = detail.get("status")
                if version and status:
                    version = f"{version}（{status}）"
                else:
                    version = version or status or detail
            else:
                version = detail
            lines.append(f"| {_md_cell(name)} | {_md_cell(version)} |")
        lines.append("")

    if skipped:
        lines.extend(["### 未检查文件", ""])
        lines.extend(f"- {_format_item_text(item)}" for item in skipped)
        lines.append("")

    if errors:
        lines.extend(["### 工具安装或执行失败", ""])
        lines.extend(f"- {_format_item_text(item)}" for item in errors)
        lines.append("")


def _render_format_summary(
    lines: list[str], format_results: list[dict], clause_results: list[dict]
) -> None:
    lines.extend(["## 2. 检查摘要", ""])
    review = _review_issue_stats(clause_results)
    review_total = sum(review.values())
    if not format_results:
        lines.extend(
            [
                "> 本次 workflow 未收到格式检查 YAML；格式状态记为“未执行”，不能视为通过。",
                "",
                "| 语言 | 文件数 | Lint问题数 | 格式问题数 | 必要Review问题数 |",
                "|---|---:|---:|---:|---:|",
                f"| Python / TileLang | 未执行 | 未执行 | 未执行 | {review['python']} |",
                f"| C/C++ | 未执行 | — | 未执行 | {review['cpp']} |",
                f"| Markdown | 未执行 | — | — | {review['markdown']} |",
                f"| 其他/无法分类 | — | — | — | {review['other']} |",
                f"| **合计** | 未执行 | 未执行 | 未执行 | {review_total} |",
                "",
            ]
        )
        return

    stats = _format_stats(format_results)
    statuses = Counter(str(result.get("status")) for result in format_results)
    status_text = "、".join(f"{key} {value}" for key, value in sorted(statuses.items()))
    file_total = stats["python_files"] + stats["cpp_files"] + stats["markdown_files"]
    format_total = stats["python_format_issues"] + stats["cpp_format_issues"]
    lines.extend(
        [
            f"- 格式检查结果：{status_text}",
            "",
            "| 语言 | 文件数 | Lint问题数 | 格式问题数 | 必要Review问题数 |",
            "|---|---:|---:|---:|---:|",
            f"| Python / TileLang | {stats['python_files']} | {stats['lint_issues']} "
            f"| {stats['python_format_issues']} | {review['python']} |",
            f"| C/C++ | {stats['cpp_files']} | — | {stats['cpp_format_issues']} | {review['cpp']} |",
            f"| Markdown | {stats['markdown_files']} | — | — | {review['markdown']} |",
            f"| 其他/无法分类 | — | — | — | {review['other']} |",
            f"| **合计** | {file_total} | {stats['lint_issues']} | {format_total} | {review_total} |",
            "",
        ]
    )
    _render_format_metadata(lines, format_results)


def _format_description(issue: Any) -> str:
    if not isinstance(issue, dict):
        return "需要格式化"
    return str(issue.get("message") or issue.get("description") or "需要格式化")


def _issue_code(issue: Any) -> str:
    if not isinstance(issue, dict):
        return "未分类"
    return str(issue.get("code") or "未分类")


def _unique_text(values: list[str]) -> str:
    return "；".join(dict.fromkeys(value for value in values if value)) or "—"


def _lint_example(issue: Any) -> str:
    if not isinstance(issue, dict):
        return str(issue)
    line = _issue_line(issue)
    source = issue.get("source") or issue.get("line_text") or ""
    if line and source:
        return f"L{line}: {source}"
    if line:
        return f"L{line}"
    return str(source or "位置未提供")


def _render_lint_groups(lines: list[str], issues: list[Any]) -> None:
    """Render bounded Ruff examples while retaining totals from the full YAML."""
    groups: dict[str, list[Any]] = defaultdict(list)
    for issue in issues:
        groups[_issue_code(issue)].append(issue)

    lines.extend(
        [
            "#### Lint问题（按Ruff代码分组）",
            "",
            "| Ruff代码 | 总数 | 示例（最多3处） | 描述 | 修复建议 |",
            "|---|---:|---|---|---|",
        ]
    )
    omitted = 0
    for code, grouped in sorted(groups.items()):
        examples = grouped[:MAX_LINT_EXAMPLES_PER_FILE_CODE]
        omitted += max(0, len(grouped) - len(examples))
        descriptions = [
            str(item.get("message", "")) for item in examples if isinstance(item, dict)
        ]
        fixes = [_issue_fix(item) for item in examples if isinstance(item, dict)]
        lines.append(
            f"| {_md_cell(code)} | {len(grouped)} "
            f"| {_md_cell('；'.join(_lint_example(item) for item in examples))} "
            f"| {_md_cell(_unique_text(descriptions))} "
            f"| {_md_cell(_unique_text(fixes))} |"
        )
    lines.append("")
    if omitted:
        lines.extend(
            [
                f"> 本文件另有 {omitted} 个 Ruff 问题未展开；完整结果保留在汇总 format YAML。",
                "",
            ]
        )


def _render_diff_preview(lines: list[str], diffs: list[str]) -> None:
    """Render at most 50 diff lines for one file."""
    diff_lines = []
    for diff in diffs:
        if diff:
            diff_lines.extend(str(diff).rstrip().splitlines())
    if not diff_lines:
        return

    preview = diff_lines[:MAX_FORMAT_DIFF_LINES_PER_FILE]
    lines.extend(["", "```diff", *preview, "```"])
    omitted = len(diff_lines) - len(preview)
    if omitted:
        lines.extend(
            [
                f"> diff 共 {len(diff_lines)} 行，仅展示前 {len(preview)} 行；其余 {omitted} 行保留在汇总 format YAML。",
                "",
            ]
        )


def _render_python_format(lines: list[str], format_results: list[dict]) -> None:
    lines.extend(["## 3. Python Lint与格式问题", ""])
    if not format_results:
        lines.extend(["本次未执行格式检查。", ""])
        return
    lint_issues = _collect_format_items(format_results, "python", "lint_issues")
    format_issues = _collect_format_items(format_results, "python", "format_issues")
    if not lint_issues and not format_issues:
        lines.extend(["未发现 Python Lint 或格式问题。", ""])
        return

    by_file: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: {"lint": [], "format": []}
    )
    for issue in lint_issues:
        by_file[_issue_path(issue) or "未知文件"]["lint"].append(issue)
    for issue in format_issues:
        by_file[_issue_path(issue) or "未知文件"]["format"].append(issue)

    for file_path in sorted(by_file):
        lines.extend([f"### {file_path}", ""])
        file_lint = by_file[file_path]["lint"]
        if file_lint:
            _render_lint_groups(lines, file_lint)

        file_format = by_file[file_path]["format"]
        if file_format:
            lines.extend(["#### 格式问题", ""])
            for issue in file_format:
                lines.append(f"- 具体问题：{_format_description(issue)}")
            diffs = [
                str(issue.get("diff", ""))
                for issue in file_format
                if isinstance(issue, dict)
            ]
            _render_diff_preview(lines, diffs)
            lines.append("")


def _render_cpp_format(lines: list[str], format_results: list[dict]) -> None:
    lines.extend(["## 4. C/C++格式问题", ""])
    if not format_results:
        lines.extend(["本次未执行格式检查。", ""])
        return
    issues = _collect_format_items(format_results, "cpp", "format_issues")
    if not issues:
        lines.extend(["未发现 C/C++ 格式问题。", ""])
        return
    by_file: dict[str, list[Any]] = defaultdict(list)
    for issue in issues:
        by_file[_issue_path(issue) or "未知文件"].append(issue)
    for file_path in sorted(by_file):
        file_issues = by_file[file_path]
        lines.extend([f"### {file_path}", ""])
        for issue in file_issues:
            lines.append(f"- 具体问题：{_format_description(issue)}")
        diffs = [
            str(issue.get("diff", ""))
            for issue in file_issues
            if isinstance(issue, dict)
        ]
        _render_diff_preview(lines, diffs)
        lines.append("")


def _infer_fence_language(file_path: str) -> str:
    extension = os.path.splitext(file_path)[1].lower()
    return {
        ".py": "python",
        ".pyi": "python",
        ".md": "markdown",
        ".sh": "bash",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".toml": "toml",
        ".c": "c",
        ".cc": "cpp",
        ".cpp": "cpp",
        ".cxx": "cpp",
        ".h": "cpp",
        ".hpp": "cpp",
    }.get(extension, "")


def _finding_sort_key(result: dict) -> tuple[str, int, str]:
    snippet = result.get("code_snippet")
    if not isinstance(snippet, dict):
        return "", 0, str(result.get("canonical_id", ""))
    try:
        start_line = int(snippet.get("start_line", 0))
    except (TypeError, ValueError):
        start_line = 0
    return (
        str(snippet.get("file_path", "")),
        start_line,
        str(result.get("canonical_id", "")),
    )


def _render_evidence(lines: list[str], evidence: dict) -> None:
    for label, key in (("正向证据", "positive"), ("负向证据", "negative")):
        items = evidence.get(key, []) or []
        if not items:
            continue
        lines.extend(
            [
                f"**{label}**",
                "",
                "| 证据类型 | 分值 | 证据描述 |",
                "|---|---:|---|",
            ]
        )
        for item in items:
            lines.append(
                f"| {_md_cell(item.get('type', ''))} | {_md_cell(item.get('score', ''))} "
                f"| {_md_cell(item.get('desc', ''))} |"
            )
        lines.append("")
    confidence_value = evidence.get("confidence_value")
    if confidence_value:
        lines.append(
            f"自信值 = clamp(Σ正向 + Σ负向, 0, 100) = {confidence_value}；80% 为 FAIL 阈值。"
        )


def _render_finding(result: dict) -> str:
    canonical = result.get("canonical_id") or result.get("clause_id", "")
    lines = [f"### [{canonical}] {result.get('clause_title', '')}", ""]
    lines.append(
        f"- **状态**：{result.get('status', '')} | **置信度**：{result.get('confidence', '')}"
    )
    lines.append(f"- **问题描述**：{result.get('problem_desc', '')}")

    snippet = result.get("code_snippet")
    if isinstance(snippet, dict):
        file_path = str(snippet.get("file_path", ""))
        lines.extend(
            [
                f"- **文件**：{file_path}",
                f"- **行号**：{snippet.get('start_line', '')}-{snippet.get('end_line', '')}",
                "- **问题片段**：",
                "",
                f"```{_infer_fence_language(file_path)}",
                str(snippet.get("code", "")).rstrip(),
                "```",
            ]
        )

    evidence = result.get("evidence")
    if isinstance(evidence, dict):
        lines.extend(["", "- **假设检验证据**：", ""])
        _render_evidence(lines, evidence)
    lines.extend(["", f"- **修复建议**：{result.get('fix_suggestion', '')}"])
    return "\n".join(lines)


def _classify_findings(
    results: list[dict],
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    high, medium, low, out_of_range = [], [], [], []
    for result in results:
        if result.get("status") == "PASS":
            continue
        if result.get("out_of_range"):
            out_of_range.append(result)
        elif result.get("confidence") == "HIGH":
            high.append(result)
        elif result.get("confidence") == "MED":
            medium.append(result)
        else:
            low.append(result)
    for group in (high, medium, low, out_of_range):
        group.sort(key=_finding_sort_key)
    return high, medium, low, out_of_range


def _render_clause_sections(lines: list[str], clause_results: list[dict]) -> None:
    stats = _clause_stats(clause_results)
    lines.extend(
        [
            "## 5. 必要代码检视统计",
            "",
            "| 状态 | 条例结果数 | 占比 |",
            "|---|---:|---:|",
            f"| PASS | {stats['PASS']} | {_pct(stats['PASS'], stats['total'])} |",
            f"| FAIL | {stats['FAIL']} | {_pct(stats['FAIL'], stats['total'])} |",
            f"| SUSPICIOUS | {stats['SUSPICIOUS']} | {_pct(stats['SUSPICIOUS'], stats['total'])} |",
            "",
        ]
    )
    passed = sorted(
        str(result.get("canonical_id") or result.get("clause_id"))
        for result in clause_results
        if result.get("status") == "PASS"
    )
    if passed:
        lines.extend(
            ["### 已通过条例", "", ", ".join(f"`{item}`" for item in passed), ""]
        )

    high, medium, low, out_of_range = _classify_findings(clause_results)
    for title, findings in (
        ("## 6. 发现问题（HIGH置信度）", high),
        ("## 7. 需关注（MED置信度）", medium),
        ("## 8. 疑似（LOW置信度）", low),
    ):
        if not findings:
            continue
        lines.extend([title, ""])
        for finding in findings:
            lines.extend([_render_finding(finding), ""])
    if out_of_range:
        lines.extend(
            [
                "## 9. PR范围外备注",
                "",
                "> 以下发现不属于本次PR diff的正式范围，但按兼容字段保留展示。",
                "",
            ]
        )
        for finding in out_of_range:
            lines.extend([_render_finding(finding), ""])


def _render_problem_stats(
    lines: list[str], format_results: list[dict], clause_results: list[dict]
) -> None:
    lines.extend(["## 10. 问题统计摘要", ""])
    if not format_results:
        lines.extend(
            [
                "### Lint与格式问题",
                "",
                "本次未执行格式检查，不能生成Lint或格式问题统计。",
            ]
        )
    else:
        lint_issues = _collect_format_items(format_results, "python", "lint_issues")
        lint_codes = Counter(
            str(issue.get("code") or "未分类")
            for issue in lint_issues
            if isinstance(issue, dict)
        )
        lines.extend(
            [
                "### Lint问题（按错误代码）",
                "",
                "| 代码 | 出现次数 |",
                "|---|---:|",
            ]
        )
        if lint_codes:
            for code, count in sorted(lint_codes.items()):
                lines.append(f"| {_md_cell(code)} | {count} |")
        else:
            lines.append("| — | 0 |")

        format_stats = _format_stats(format_results)
        lines.extend(
            [
                "",
                "### 格式问题",
                "",
                "| 语言 | 问题数 |",
                "|---|---:|",
                f"| Python / TileLang | {format_stats['python_format_issues']} |",
                f"| C/C++ | {format_stats['cpp_format_issues']} |",
            ]
        )

    lines.extend(
        [
            "",
            "### 必要Review问题（按条例身份）",
            "",
            "| Canonical条例身份 | FAIL | SUSPICIOUS |",
            "|---|---:|---:|",
        ]
    )
    review_counts: dict[str, Counter] = defaultdict(Counter)
    for result in clause_results:
        if result.get("status") == "PASS":
            continue
        canonical = str(result.get("canonical_id") or result.get("clause_id") or "")
        review_counts[canonical][str(result.get("status"))] += 1
    if review_counts:
        for canonical, counts in sorted(review_counts.items()):
            lines.append(
                f"| {_md_cell(canonical)} | {counts['FAIL']} | {counts['SUSPICIOUS']} |"
            )
    else:
        lines.append("| — | 0 | 0 |")
    lines.append("")


def assemble_report(
    clause_results: list[dict], format_results: list[dict], skipped_files: list[str]
) -> str:
    """Render the complete fused report."""
    stats = _clause_stats(clause_results)
    lines = [
        "# 代码格式与必要检视报告",
        "",
        "## 1. 检视概览",
        "",
        "- **生成时间**：{{TIMESTAMP}}",
        "- **检查对象**：{{CODE_FILE}}",
        "- **文件类型**：{{FILE_TYPES}}",
        "- **代码侧别**：{{SIDE}}",
        "- **检视规则**：{{DOC_LIST}}",
        f"- **条例结果数**：{stats['total']}",
        f"- **格式检查YAML数**：{len(format_results)}",
        "",
    ]
    if skipped_files:
        lines.extend(
            [
                f"- **跳过的无效YAML**：{', '.join(skipped_files)}",
                "",
            ]
        )
    _render_format_summary(lines, format_results, clause_results)
    _render_python_format(lines, format_results)
    _render_cpp_format(lines, format_results)
    _render_clause_sections(lines, clause_results)
    _render_problem_stats(lines, format_results, clause_results)
    lines.extend(
        [
            "## 11. 下一步操作",
            "",
            "```bash",
            "# 重新查看格式问题",
            "ruff check <file>",
            "ruff format --diff <file>",
            "clang-format --style=file <file> | diff -u <file> -",
            "```",
            "",
            "- 格式问题只能在全部检查完成并获得用户明确同意后自动修复。",
            "- references检视发现属于语义问题，只提供证据和建议，不自动修改。",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YAML目录 → 融合Markdown检视报告")
    parser.add_argument("--dir", required=True, help="collector YAML输出目录")
    parser.add_argument("--output", required=True, help="报告Markdown输出路径")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    yaml_dir = os.path.abspath(args.dir)
    output_path = os.path.abspath(args.output)
    if not os.path.isdir(yaml_dir):
        logger.error("目录不存在: %s", yaml_dir)
        return 2

    try:
        clause_results, format_results, skipped_files = load_yaml_files(yaml_dir)
    except ValueError:
        return 1

    report = assemble_report(clause_results, format_results, skipped_files)
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as error:
            logger.error("创建输出目录失败: %s", error)
            return 2
    try:
        with open(output_path, "w", encoding="utf-8") as output_file:
            output_file.write(report)
    except OSError as error:
        logger.error("写报告失败: %s", error)
        return 2

    stats = _clause_stats(clause_results)
    logger.info("报告拼接完成")
    logger.info("  yaml目录: %s", yaml_dir)
    logger.info("  报告路径: %s", output_path)
    logger.info("  format yaml: %d 个", len(format_results))
    logger.info(
        "  条例统计: 总%d / PASS %d / FAIL %d / SUSPICIOUS %d",
        stats["total"],
        stats["PASS"],
        stats["FAIL"],
        stats["SUSPICIOUS"],
    )
    if skipped_files:
        logger.info("  跳过无法解析或非mapping的yaml: %d 个", len(skipped_files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
