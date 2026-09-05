#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Generate the mandatory per-run Issue handling summary report."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from runtime_paths import LATEST_REPORT, REPORTS_DIR, path_text  # noqa: E402
from cli_output import write_stdout  # noqa: E402

SENSITIVE_KEYS = {
    "access_token",
    "authorization",
    "gitcode_token",
    "password",
    "private_token",
    "secret",
    "token",
}
SECRET_PATTERNS = (
    re.compile(r"(?i)(access_token\s*[=:]\s*)[^&\s]+"),
    re.compile(r"(?i)(private-token\s*[=:]\s*)[^\s]+"),
    re.compile(r"(?i)(authorization\s*[=:]\s*(?:bearer\s+)?)[^\s]+"),
    re.compile(r"(?i)gitcode_pat_[A-Za-z0-9_-]+"),
)
LOGGER = logging.getLogger(__name__)


def _write_stdout(text: str) -> None:
    """Write the JSON result protocol to stdout."""
    write_stdout(text)


def redact_text(value: str) -> str:
    redacted = value
    for pattern in SECRET_PATTERNS:
        if pattern.pattern.lower().startswith("(?i)gitcode_pat"):
            redacted = pattern.sub("[REDACTED]", redacted)
        else:
            redacted = pattern.sub(r"\1[REDACTED]", redacted)
    return redacted


def sanitize(value: Any, key: str = "") -> Any:
    normalized = key.casefold().replace("-", "_")
    if normalized in SENSITIVE_KEYS or normalized.endswith("_access_token"):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): sanitize(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def present(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def scalar(value: Any, default: str = "unknown") -> str:
    if not present(value):
        return default
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def inline(value: Any, default: str = "unknown") -> str:
    return scalar(value, default).replace("|", "\\|").replace("\n", "<br>")


def items(value: Any) -> list[Any]:
    if not present(value):
        return []
    return value if isinstance(value, list) else [value]


def safe_run_id(value: Any) -> str:
    run_id = re.sub(r"[^A-Za-z0-9._-]+", "-", scalar(value, "unknown-run"))
    return run_id.strip(".-") or "unknown-run"


ROUTING_ONLY_CATEGORIES = {"self_assigned", "auto_assign_via_pr"}
SUBSTANTIVE_STAGES = {
    "diagnose",
    "reproduce",
    "implement",
    "validate",
    "deliver",
    "comment",
}


def process_stages(issue: dict[str, Any]) -> set[str]:
    return {
        str(entry.get("stage", ""))
        for entry in items(issue.get("process_log"))
        if isinstance(entry, dict)
    }


def should_report_issue(issue: Any) -> bool:
    """Return whether this run substantively handled an Issue.

    Self-assignment and PR-author assignment are routing observations, not
    handling results.  New states should set handled_in_run explicitly; the
    stage fallback keeps older run states readable without reintroducing
    classification-only noise.
    """
    if not isinstance(issue, dict):
        return False
    if str(issue.get("category", "")) in ROUTING_ONLY_CATEGORIES:
        return False
    explicit = issue.get("handled_in_run")
    if isinstance(explicit, bool):
        return explicit
    if issue.get("bucket") == "need_attention":
        return True
    return bool(process_stages(issue) & SUBSTANTIVE_STAGES)


def normalize_report_scope(state: dict[str, Any]) -> int:
    """Keep only handled Issues/groups and return the excluded item count."""
    run = state.setdefault("run", {})
    raw_issues = state.get("issues") if isinstance(state.get("issues"), list) else []
    report_issues = [issue for issue in raw_issues if should_report_issue(issue)]
    excluded = len(raw_issues) - len(report_issues)

    prior_total = run.get("issues_scanned_total", run.get("issues_total"))
    if not isinstance(prior_total, int) or prior_total < len(raw_issues):
        prior_total = len(raw_issues)
    run["issues_scanned_total"] = prior_total
    run["issues_total"] = len(report_issues)
    state["issues"] = report_issues

    handled_ids = {str(issue.get("iid")) for issue in report_issues}
    groups = state.get("groups") if isinstance(state.get("groups"), list) else []
    report_groups = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        contains_handled_issue = any(
            str(member) in handled_ids for member in items(group.get("members"))
        )
        if contains_handled_issue:
            report_groups.append(group)
    state["groups"] = report_groups
    return excluded


REQUIRED_ISSUE_FIELDS = (
    "iid",
    "url",
    "title",
    "author",
    "bucket",
    "category",
    "handling_status",
    "resolution_status",
    "result_summary",
    "next_action",
)
REQUIRED_LOG_FIELDS = ("time", "stage", "action", "result", "evidence")


def _validate_process_log(issue: dict[str, Any], label: str):
    process_log = issue.get("process_log")
    if not isinstance(process_log, list) or not process_log:
        return [
            f"{label}.process_log must contain at least the classification action"
        ], set()
    errors = []
    stages = set()
    for index, entry in enumerate(process_log):
        if not isinstance(entry, dict):
            errors.append(f"{label}.process_log[{index}] must be an object")
            continue
        for field in REQUIRED_LOG_FIELDS:
            missing_value = field != "evidence" and not present(entry.get(field))
            if field not in entry or missing_value:
                errors.append(
                    f"{label}.process_log[{index}].{field} is required; "
                    "use 'unknown' when unavailable"
                )
        stages.add(str(entry.get("stage", "")))
    return errors, stages


def _expected_stages(issue: dict[str, Any], group: dict[str, Any]) -> set[str]:
    expected = {"triage"}
    if issue.get("bucket") == "need_attention":
        expected.add("diagnose")
    if present(issue.get("reproduction_status")):
        expected.add("reproduce")
    if present(issue.get("changed_files")) or present(group.get("changed_files")):
        expected.add("implement")
    if present(issue.get("tests")) or present(group.get("tests")):
        expected.add("validate")
    delivery_fields = ("commit_sha", "pr_url", "published_branch")
    if any(
        present(issue.get(field)) or present(group.get(field))
        for field in delivery_fields
    ):
        expected.add("deliver")
    return expected


def _validate_issue(issue: Any, index: int, group_by_id: dict[str, Any]):
    if not isinstance(issue, dict):
        return [f"issues[{index}] must be an object"]
    label = f"Issue {issue.get('iid', index)}"
    errors = [
        f"{label}.{field} is required; use 'unknown' when unavailable"
        for field in REQUIRED_ISSUE_FIELDS
        if not present(issue.get(field))
    ]
    log_errors, stages = _validate_process_log(issue, label)
    errors.extend(log_errors)
    if not stages:
        return errors
    group = group_by_id.get(str(issue.get("group_id")), {})
    missing_stages = sorted(_expected_stages(issue, group) - stages)
    if missing_stages:
        errors.append(
            f"{label}.process_log missing stages: {', '.join(missing_stages)}"
        )
    return errors


def validate_state(state: dict[str, Any]) -> list[str]:
    run = state.get("run")
    if not isinstance(run, dict):
        return ["run must be an object"]
    errors = [
        f"run.{field} is required"
        for field in ("run_id", "started_at", "completed_at", "overall_status")
        if not present(run.get(field))
    ]
    issues = state.get("issues")
    if not isinstance(issues, list):
        return errors + ["issues must be a list"]
    if run.get("issues_total") != len(issues):
        errors.append(
            f"run.issues_total ({run.get('issues_total')!r}) must equal "
            f"issues length ({len(issues)})"
        )
    groups = state.get("groups") if isinstance(state.get("groups"), list) else []
    group_by_id = {
        str(group.get("group_id")): group
        for group in groups
        if isinstance(group, dict) and present(group.get("group_id"))
    }
    for index, issue in enumerate(issues):
        errors.extend(_validate_issue(issue, index, group_by_id))
    return errors


def render_process_log(log: Any) -> list[str]:
    entries = items(log)
    if not entries:
        return []
    meaningful = [
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("stage") != "triage"
    ]
    if not meaningful:
        meaningful = [entry for entry in entries if isinstance(entry, dict)]
    lines = [
        "| 动作 | 结果 |",
        "| --- | --- |",
    ]
    for entry in meaningful:
        lines.append(
            "| {} | {} |".format(
                inline(entry.get("action")),
                inline(entry.get("result")),
            )
        )
    return lines


def render_issue(issue: dict[str, Any], group: dict[str, Any] | None) -> list[str]:
    iid = scalar(issue.get("iid"))
    title = scalar(issue.get("title"))
    lines = [f"### Issue #{iid}：{title}", ""]
    group = group or {}
    lines.extend(
        [
            f"- 结果：{scalar(issue.get('result_summary'))}",
            f"- 状态：{scalar(issue.get('handling_status'))} / {scalar(issue.get('resolution_status'))}",
            f"- 下一步：{scalar(issue.get('next_action'))}",
            f"- 链接：{scalar(issue.get('url'))}",
        ]
    )

    root_cause = scalar(issue.get("final_root_cause"), "")
    if root_cause and not root_cause.startswith(("不适用", "unknown")):
        lines.append(f"- 根因：{root_cause}")

    process_lines = render_process_log(issue.get("process_log"))
    if process_lines:
        lines.extend(["", "#### 核心动作", "", *process_lines])

    changed_files = issue.get("changed_files") or group.get("changed_files")
    tests = issue.get("tests") or group.get("tests")
    pr = issue.get("pr_url") or group.get("pr_url") or group.get("published_branch")
    commit = issue.get("commit_sha") or group.get("commit_sha")
    if any(present(value) for value in (changed_files, tests, pr, commit)):
        lines.extend(["", "#### 变更与交付", ""])
        if present(changed_files):
            lines.append(f"- 文件：{scalar(changed_files)}")
        if present(tests):
            lines.append(f"- 验证：{scalar(tests)}")
        if present(commit):
            lines.append(f"- Commit：{scalar(commit)}")
        if present(pr):
            lines.append(f"- PR/推送：{scalar(pr)}")

    blockers = items(issue.get("blockers"))
    risks = items(issue.get("remaining_risks") or issue.get("validation_boundary"))
    if blockers or risks:
        lines.extend(["", "#### 卡点与风险", ""])
        lines.extend(f"- {scalar(value)}" for value in blockers + risks)
    lines.append("")
    return lines


def render_report(state: dict[str, Any], state_path: Path, output_path: Path) -> str:
    run = state["run"]
    issues = state.get("issues") if isinstance(state.get("issues"), list) else []
    groups = state.get("groups") if isinstance(state.get("groups"), list) else []
    group_by_id = {
        str(group.get("group_id")): group
        for group in groups
        if isinstance(group, dict) and present(group.get("group_id"))
    }
    generated_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    lines = [
        "# Issue 处理结果",
        "",
        f"> Run `{inline(run.get('run_id'))}`，生成时间 `{generated_at}`。",
        "",
        "## 概览",
        "",
        f"- 仓库：{scalar(run.get('repository') or run.get('repo'))}",
        f"- 范围：{scalar(run.get('time_scope'))}",
        f"- 状态：{scalar(run.get('overall_status'))}",
        f"- 扫描：{scalar(run.get('issues_scanned_total'), str(len(issues)))} 个 Issue",
        f"- 实际处理：{len(issues)} 个 Issue",
    ]

    lines.extend(["", "## 本次实际处理的 Issue", ""])
    if not issues:
        lines.append("本次未实际处理任何 Issue。")
    for issue in issues:
        group = group_by_id.get(str(issue.get("group_id")))
        lines.extend(render_issue(issue, group))

    blockers = items(state.get("internal_blockers"))
    boundaries = items(state.get("validation_boundaries"))
    show_run_limits = bool(issues) or run.get("overall_status") in {
        "partial",
        "blocked",
    }
    if show_run_limits and (blockers or boundaries):
        lines.extend(["## 卡点与验证边界", ""])
        lines.extend(f"- {scalar(value)}" for value in blockers + boundaries)
        lines.append("")
    lines.append("")
    return "\n".join(lines)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, help="Complete run-state JSON file")
    parser.add_argument("--output", help="Historical Markdown output path")
    parser.add_argument(
        "--latest",
        default=path_text(LATEST_REPORT),
        help=f"Latest-report path (default: {path_text(LATEST_REPORT)})",
    )
    parser.add_argument(
        "--no-latest", action="store_true", help="Do not update latest.md"
    )
    parser.add_argument(
        "--strict", action="store_true", help="Reject incomplete per-Issue state"
    )
    return parser.parse_args(argv)


def _load_state(args):
    state_path = Path(args.state)
    try:
        raw_state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Error: cannot read run state: {exc}") from exc
    if not isinstance(raw_state, dict):
        raise ValueError("Error: run state root must be an object")

    state = sanitize(raw_state)
    excluded_observations = normalize_report_scope(state)
    errors = validate_state(state) if args.strict else []
    if errors:
        raise ValueError(
            "Error: summary report state is incomplete:\n"
            + "\n".join(f"- {error}" for error in errors)
        )
    return state, excluded_observations


def _write_report_artifacts(state, args, excluded_observations):
    run = state.setdefault("run", {})
    run_id = safe_run_id(run.get("run_id"))
    output_path = (
        Path(args.output)
        if args.output
        else REPORTS_DIR / run_id / "summary.md"
    )
    canonical_state_path = output_path.parent / "run_state.json"
    run["report_generated"] = True
    run["report_path"] = output_path.as_posix()
    report = render_report(state, canonical_state_path, output_path)

    write_text(
        canonical_state_path, json.dumps(state, ensure_ascii=False, indent=2) + "\n"
    )
    write_text(output_path, report)
    latest_path = None
    if not args.no_latest:
        latest_path = Path(args.latest)
        write_text(latest_path, report)

    result = {
        "report_path": output_path.as_posix(),
        "run_state_path": canonical_state_path.as_posix(),
        "latest_path": latest_path.as_posix() if latest_path else None,
        "issues": len(state.get("issues", [])),
        "excluded_observations": excluded_observations,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        state, excluded_observations = _load_state(args)
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 2
    result = _write_report_artifacts(state, args, excluded_observations)
    _write_stdout(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
