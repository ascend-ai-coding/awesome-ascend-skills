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
from report_runs import create_run, save_run, RUN_NAME
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
RESPONSIBILITIES = {"handle", "list-only", "ignore"}
SUBSTANTIVE_STAGES = {
    "diagnose",
    "reproduce",
    "implement",
    "validate",
    "deliver",
    "comment",
}


def process_stages(issue: dict[str, Any]) -> set[str]:
    return {str(entry.get("stage", "")) for entry in items(issue.get("process_log")) if isinstance(entry, dict)}


def should_report_issue(issue: Any) -> bool:
    """Return whether this run substantively handled an Issue.

    Historical self-assignment is a routing observation. A response-stage
    PR-owner handoff keeps its actionable category and handled_in_run flag.
    New states should set handled_in_run explicitly; the stage fallback keeps
    older run states readable without reintroducing classification-only noise.
    """
    if not isinstance(issue, dict):
        return False
    responsibility = issue.get("responsibility")
    if responsibility == "ignore" or responsibility == "list-only":
        return False
    if str(issue.get("category", "")) in ROUTING_ONLY_CATEGORIES:
        return False
    explicit = issue.get("handled_in_run")
    if isinstance(explicit, bool):
        return explicit
    if issue.get("bucket") == "need_attention":
        return True
    return bool(process_stages(issue) & SUBSTANTIVE_STAGES)


def _issue_key(issue: Any) -> str:
    if not isinstance(issue, dict):
        return ""
    return str(issue.get("iid") or issue.get("url") or "")


def _route_report_issues(candidates):
    report_issues = []
    listed_issues = []
    seen: set[str] = set()
    for issue in candidates:
        if not isinstance(issue, dict):
            continue
        key = _issue_key(issue)
        responsibility = issue.get("responsibility")
        if responsibility == "list-only":
            if key and key in seen:
                continue
            listed_issues.append(issue)
            if key:
                seen.add(key)
        elif responsibility != "ignore" and should_report_issue(issue):
            if key and key in seen:
                continue
            report_issues.append(issue)
            if key:
                seen.add(key)
    return report_issues, listed_issues


def normalize_report_scope(state: dict[str, Any]) -> int:
    """Normalize responsibility routing while retaining enough state to regenerate."""
    run = state.setdefault("run", {})
    raw_issues = state.get("issues") if isinstance(state.get("issues"), list) else []
    prior_listed = state.get("listed_issues") if isinstance(state.get("listed_issues"), list) else []
    # A prior canonical listed entry wins over a duplicate in issues.
    report_issues, listed_issues = _route_report_issues(list(prior_listed) + raw_issues)

    prior_total = run.get("issues_scanned_total", run.get("issues_total"))
    if not isinstance(prior_total, int) or prior_total < len(raw_issues):
        prior_total = len(raw_issues)
    run["issues_scanned_total"] = prior_total
    run["issues_total"] = len(report_issues)
    run["issues_listed_total"] = len(listed_issues)
    state["issues"] = report_issues
    state["listed_issues"] = listed_issues

    handled_ids = {str(issue.get("iid")) for issue in report_issues}
    groups = state.get("groups") if isinstance(state.get("groups"), list) else []
    report_groups = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        members = items(group.get("members"))
        if any(str(member) in handled_ids for member in members):
            report_groups.append(group)
    state["groups"] = report_groups
    return len(raw_issues) - len(report_issues)


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
        return [f"{label}.process_log must contain at least the classification action"], set()
    errors = []
    stages = set()
    for index, entry in enumerate(process_log):
        if not isinstance(entry, dict):
            errors.append(f"{label}.process_log[{index}] must be an object")
            continue
        for field in REQUIRED_LOG_FIELDS:
            missing_value = field != "evidence" and not present(entry.get(field))
            if field not in entry or missing_value:
                errors.append(f"{label}.process_log[{index}].{field} is required; use 'unknown' when unavailable")
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
    if any(present(issue.get(field)) or present(group.get(field)) for field in delivery_fields):
        expected.add("deliver")
    return expected


def _candidate_map(candidates, label, errors):
    seen = set()
    per_operator: dict[str, set[str]] = {}
    required = ("login", "operators", "contribution", "evidence", "identity_evidence", "limitation")
    for candidate in candidates:
        if not isinstance(candidate, dict):
            errors.append(f"{label}.owner candidate must be an object")
            continue
        for key in required:
            if not present(candidate.get(key)):
                errors.append(f"{label}.owner candidate requires {key}")
        login = str(candidate.get("login", "")).casefold()
        if login in seen:
            errors.append(f"{label}.duplicate owner candidate login")
        seen.add(login)
        for operator in items(candidate.get("operators")):
            per_operator.setdefault(str(operator), set()).add(login)
    for operator, logins in per_operator.items():
        if len(logins) > 5:
            errors.append(f"{label}.{operator} owner candidates must be at most 5")
    return per_operator


def _validate_analysis_operators(analysis, operators, per_operator, label, errors):
    if not isinstance(operators, list):
        errors.append(f"{label}.owner analysis operators must be a list")
        return
    target_operators = [str(operator) for operator in operators]
    if len(set(target_operators)) != len(target_operators):
        errors.append(f"{label}.duplicate target operator")
    uncovered = {str(operator) for operator in items(analysis.get("uncovered_operators"))}
    target_set = set(target_operators)
    for operator in per_operator:
        if operator not in target_set:
            errors.append(f"{label}.{operator} candidate operator is not in operators")
    for operator in target_operators:
        if not per_operator.get(operator) and operator not in uncovered:
            errors.append(f"{label}.{operator} must have candidates or be listed in uncovered_operators")


def _validate_candidate_analysis(analysis, label, errors):
    if analysis is None:
        return
    if not isinstance(analysis, dict):
        errors.append(f"{label}.owner_candidate_analysis must be an object")
        return
    candidates = analysis.get("candidates", [])
    if not isinstance(candidates, list):
        errors.append(f"{label}.owner candidates must be a list")
        return
    per_operator = _candidate_map(candidates, label, errors)
    if analysis.get("operators") is not None:
        _validate_analysis_operators(
            analysis,
            analysis["operators"],
            per_operator,
            label,
            errors,
        )


def _validate_issue(issue: Any, index: int, group_by_id: dict[str, Any]):
    if not isinstance(issue, dict):
        return [f"issues[{index}] must be an object"]
    label = f"Issue {issue.get('iid', index)}"
    errors = [
        f"{label}.{field} is required; use 'unknown' when unavailable"
        for field in REQUIRED_ISSUE_FIELDS
        if not present(issue.get(field))
    ]
    responsibility = issue.get("responsibility")
    if present(responsibility) and responsibility not in RESPONSIBILITIES:
        errors.append(f"{label}.responsibility must be one of: {', '.join(sorted(RESPONSIBILITIES))}")
    _validate_candidate_analysis(issue.get("owner_candidate_analysis"), label, errors)
    log_errors, stages = _validate_process_log(issue, label)
    errors.extend(log_errors)
    if not stages:
        return errors
    group = group_by_id.get(str(issue.get("group_id")), {})
    missing_stages = sorted(_expected_stages(issue, group) - stages)
    if missing_stages:
        errors.append(f"{label}.process_log missing stages: {', '.join(missing_stages)}")
    return errors


def _validate_listed_issue(issue: Any, index: int):
    if not isinstance(issue, dict):
        return [f"listed_issues[{index}] must be an object"]
    label = f"listed Issue {issue.get('iid', index)}"
    errors = []
    for field in ("iid", "url", "responsibility_summary"):
        if not present(issue.get(field)):
            errors.append(f"{label}.{field} is required")
    if issue.get("responsibility") != "list-only":
        errors.append(f"{label}.responsibility must be 'list-only'")
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
        errors.append(f"run.issues_total ({run.get('issues_total')!r}) must equal issues length ({len(issues)})")
    groups = state.get("groups") if isinstance(state.get("groups"), list) else []
    group_by_id = {
        str(group.get("group_id")): group
        for group in groups
        if isinstance(group, dict) and present(group.get("group_id"))
    }
    for index, issue in enumerate(issues):
        errors.extend(_validate_issue(issue, index, group_by_id))
    listed = state.get("listed_issues", [])
    if not isinstance(listed, list):
        errors.append("listed_issues must be a list")
    else:
        if run.get("issues_listed_total") != len(listed):
            errors.append(
                f"run.issues_listed_total ({run.get('issues_listed_total')!r}) must equal "
                f"listed_issues length ({len(listed)})"
            )
        for index, issue in enumerate(listed):
            errors.extend(_validate_listed_issue(issue, index))
    return errors


def render_process_log(log: Any) -> list[str]:
    entries = items(log)
    if not entries:
        return []
    meaningful = [entry for entry in entries if isinstance(entry, dict) and entry.get("stage") != "triage"]
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


def render_owner_candidates(analysis: Any) -> list[str]:
    if not isinstance(analysis, dict):
        return []
    lines = ["", "#### 候选责任人（供线下确认）", ""]
    candidates = items(analysis.get("candidates"))
    target_operators = [str(operator) for operator in items(analysis.get("operators"))]
    uncovered = {str(operator) for operator in items(analysis.get("uncovered_operators"))}
    rows: list[tuple[str, str, str]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        candidate_operators = [str(operator) for operator in items(candidate.get("operators"))]
        if target_operators:
            ordered = [operator for operator in target_operators if operator in candidate_operators]
            ordered.extend(operator for operator in candidate_operators if operator not in target_operators)
            candidate_operators = ordered
        if not candidate_operators:
            candidate_operators = ["待确认"]
        login = scalar(candidate.get("login"))
        description = candidate.get("summary_description") or candidate.get("contribution")
        for operator in candidate_operators:
            rows.append((operator, login, scalar(description)))
    covered = {operator for operator, _, _ in rows}
    for operator in target_operators or sorted(uncovered):
        if operator in uncovered and operator not in covered:
            rows.append((operator, "待确认", "未找到可靠候选"))
    if not rows:
        rows.append(("待确认", "待确认", "未找到可靠候选"))
    lines.extend(["| 算子 | GitCode账号 | 简述 |", "| --- | --- | --- |"])
    for operator, login, description in rows:
        account = "待确认" if login == "待确认" else f"[@{inline(login)}](https://gitcode.com/{inline(login)})"
        lines.append(f"| {inline(operator)} | {account} | {inline(description)} |")
    lines.append("")
    return lines


def _render_delivery(issue: dict[str, Any], group: dict[str, Any]) -> list[str]:
    changed_files = issue.get("changed_files") or group.get("changed_files")
    tests = issue.get("tests") or group.get("tests")
    pr = issue.get("pr_url") or group.get("pr_url") or group.get("published_branch")
    commit = issue.get("commit_sha") or group.get("commit_sha")
    lines = []
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
    return lines


def _render_response_artifacts(issue):
    lines = []
    if issue.get("response_status") == "exempt_self_authored_pr":
        lines.append("- 首响：自提免首响，本轮仅补齐负责人。")
    for name, artifact in (issue.get("response_artifacts") or {}).items():
        if isinstance(artifact, dict) and artifact.get("path"):
            path = str(artifact["path"])
            marker = "/issues/issue-"
            if marker in path:
                path = "issues/issue-" + path.rsplit(marker, 1)[1]
            lines.append(f"- 响应材料：[{inline(name)}](<{path}>)")

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

    for code in items(issue.get("related_code")):
        if isinstance(code, dict) and code.get("path") and code.get("url"):
            kind = "目录" if code.get("kind") == "directory" else "文件"
            revision = f"（版本：{inline(code['revision'])}）" if code.get("revision") else ""
            lines.append(f"- 关联代码（{kind}）：[{inline(code['path'])}](<{code['url']}>){revision}")
    if present(issue.get("related_code_note")):
        lines.append(f"- 关联代码说明：{scalar(issue['related_code_note'])}")

    root_cause = scalar(issue.get("final_root_cause"), "")
    if root_cause and not root_cause.startswith(("不适用", "unknown")):
        lines.append(f"- 根因：{root_cause}")

    if issue.get("assignment_provisional") is True:
        candidate = inline(issue.get("assigned_candidate") or "待确认")
        if issue.get("assignment_status") == "verified":
            if issue.get("assignment_source") == "fallback_user":
                lines.append(f"- 临时指派：无候选，已临时指派兜底接收人 @{candidate}，请确认真正负责人。")
            else:
                lines.append(f"- 临时指派：已临时指派 @{candidate}，请确认真正负责人。")
        else:
            lines.append(f"- 临时指派：@{candidate} 尚未回查成功，不能视为已指派。")

    lines.extend(_render_response_artifacts(issue))

    lines.extend(render_owner_candidates(issue.get("owner_candidate_analysis")))

    process_lines = render_process_log(issue.get("process_log"))
    if process_lines:
        lines.extend(["", "#### 核心动作", "", *process_lines])

    lines.extend(_render_delivery(issue, group))

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
    listed_issues = state.get("listed_issues") if isinstance(state.get("listed_issues"), list) else []
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
        f"- 仅列举：{len(listed_issues)} 个 Issue",
    ]

    lines.extend(["", "## 本次实际处理的 Issue", ""])
    if not issues:
        lines.append("本次未实际处理任何 Issue。")
    for issue in issues:
        group = group_by_id.get(str(issue.get("group_id")))
        lines.extend(render_issue(issue, group))

    if listed_issues:
        lines.extend(["## 仅列举", ""])
        for issue in listed_issues:
            lines.append(
                f"- [#{inline(issue.get('iid'))}]({inline(issue.get('url'))})："
                f"{inline(issue.get('responsibility_summary'))}"
            )
        lines.append("")

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
    parser.add_argument("--no-latest", action="store_true", help="Do not update latest.md")
    parser.add_argument("--strict", action="store_true", help="Reject incomplete per-Issue state")
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
        raise ValueError("Error: summary report state is incomplete:\n" + "\n".join(f"- {error}" for error in errors))
    return state, excluded_observations


def _write_report_artifacts(state, args, excluded_observations):
    run = state.setdefault("run", {})
    if args.output:
        output_path = Path(args.output)
    elif run.get("report_directory"):
        output_path = Path(run["report_directory"]) / "summary.md"
    elif RUN_NAME.fullmatch(str(run.get("run_id", ""))) and (REPORTS_DIR / run["run_id"] / "run.json").is_file():
        output_path = REPORTS_DIR / run["run_id"] / "summary.md"
    else:
        # Legacy state stays readable; new default output always uses a dated round.
        started = run.get("started_at")
        try:
            parsed = datetime.fromisoformat(str(started).replace("Z", "+00:00"))
            timestamp = parsed.replace(tzinfo=timezone.utc).timestamp() if parsed.tzinfo is None else parsed.timestamp()
        except ValueError:
            timestamp = None
        metadata = create_run(REPORTS_DIR, run.get("repository", ""), run.get("mode", "single"), timestamp)
        run["legacy_run_id"] = run.get("run_id")
        metadata["legacy_run_id"] = run.get("run_id")
        save_run(REPORTS_DIR, metadata)
        run["run_id"] = metadata["run_id"]
        output_path = REPORTS_DIR / metadata["run_id"] / "summary.md"
    managed = (output_path.parent / "run.json").is_file()
    canonical_state_path = output_path.parent / ("_internal/run_state.json" if managed else "run_state.json")
    if managed:
        run["report_directory"] = output_path.parent.as_posix()
    run["report_generated"] = True
    run["report_path"] = output_path.as_posix()
    report = render_report(state, canonical_state_path, output_path)

    write_text(canonical_state_path, json.dumps(state, ensure_ascii=False, indent=2) + "\n")
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
