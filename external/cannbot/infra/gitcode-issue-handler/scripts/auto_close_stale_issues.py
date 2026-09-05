#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Close answered question Issues after a configurable quiet period.

The command is conservative by design. It only selects open question-type
Issues with a substantive reply from a known participant other than the Issue
reporter, no newer reporter reply, and no linked PR. Dry-run is the default;
pass --apply to post the closure notice and close the Issue.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import NamedTuple

import requests

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import classify_issues as classifier  # noqa: E402
import fetch_issues as issue_api  # noqa: E402
import followup_state  # noqa: E402
from cli_output import write_stdout  # noqa: E402

from gitcode_client import (  # noqa: E402
    api_get,
    api_patch,
    api_post,
    session_rate_limiter,
)
from runtime_paths import rate_limit_path  # noqa: E402

DEFAULT_INACTIVE_HOURS = 48.0
DEFAULT_COMMENT = (
    "您好，当前问题已经解决，我们计划关闭此issue，后续您如果还有疑问，"
    "欢迎重新给我们提issue，我们会继续提供问题支撑"
)
DEFAULT_QUESTION_LABELS = ("question", "问题咨询", "咨询")
DEFAULT_QUESTION_TITLE_MARKERS = (
    "[question|",
    "[question]",
    "问题咨询",
    "咨询",
    "请问",
)
PR_URL_RE = re.compile(
    r"https?://gitcode\.com/[^\s)]+/(?:pull|pulls|merge_requests)/\d+",
    re.IGNORECASE,
)
ASSIGN_RE = re.compile(r"^\s*/assign(?:\s|$)", re.IGNORECASE)
MORE_INFO_MARKERS = (
    "请提供",
    "请补充",
    "麻烦提供",
    "能否提供",
    "复现步骤",
    "完整日志",
    "报错日志",
    "版本信息",
    "环境信息",
)
LOGGER = logging.getLogger(__name__)


class ClosePolicy(NamedTuple):
    """Eligibility settings shared by evaluation and close-time rechecks."""

    now: datetime
    inactive_hours: float
    closure_comment: str = DEFAULT_COMMENT
    question_labels: tuple[str, ...] = DEFAULT_QUESTION_LABELS
    title_markers: tuple[str, ...] = DEFAULT_QUESTION_TITLE_MARKERS


class IssueApiContext(NamedTuple):
    """Connection values needed to update one repository's Issues."""

    session: object
    api_base: str
    owner: str
    repo: str
    token: str


class RuntimeConfig(NamedTuple):
    """Validated command configuration."""

    api: IssueApiContext
    repo_path: str
    policy: ClosePolicy
    pr_fetch_pages: int
    linkage_budget: int
    apply_changes: bool
    followup_state_file: str
    followup_watches: dict


def _write_stdout(text: str) -> None:
    """Write the JSON result protocol to stdout."""
    write_stdout(text)


def load_settings(path: str | None, *, allow_legacy: bool = True) -> dict:
    """Load shared repository settings and optional auto_close overrides."""
    cfg = classifier.load_config(path, allow_legacy=allow_legacy)
    config_path = classifier.resolve_config_path(path, allow_legacy=allow_legacy)
    raw = {}
    try:
        import yaml

        with config_path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream) or {}
    except (OSError, ValueError) as exc:
        raise ValueError(f"Error: cannot read auto_close config — {exc}") from exc
    auto_close = raw.get("auto_close") or {}
    if not isinstance(auto_close, dict):
        raise ValueError("Error: 'auto_close' config must be an object")
    cfg["auto_close"] = auto_close
    return cfg


def _labels(issue: dict) -> set[str]:
    result = set()
    for label in issue.get("labels") or []:
        if isinstance(label, dict):
            value = label.get("name") or label.get("label_name") or ""
        else:
            value = str(label)
        if value.strip():
            result.add(value.strip().casefold())
    return result


def is_question_issue(
    issue: dict,
    question_labels: tuple[str, ...] = DEFAULT_QUESTION_LABELS,
    title_markers: tuple[str, ...] = DEFAULT_QUESTION_TITLE_MARKERS,
) -> bool:
    configured_labels = {
        item.strip().casefold() for item in question_labels if item.strip()
    }
    if _labels(issue) & configured_labels:
        return True
    title = str(issue.get("title") or "").casefold()
    return any(
        marker.strip().casefold() in title for marker in title_markers if marker.strip()
    )


def _comment_author(comment: dict) -> str | None:
    author = classifier.get_comment_author(comment)
    if not author or author.strip().casefold() == "unknown":
        return None
    return author.strip()


def _comment_time(comment: dict) -> datetime | None:
    parsed = issue_api.parse_iso(comment.get("created_at", ""))
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _is_substantive_handler_reply(body: str, closure_comment: str) -> bool:
    stripped = (body or "").strip()
    if not stripped or stripped == closure_comment or ASSIGN_RE.match(stripped):
        return False
    return not any(marker in stripped for marker in MORE_INFO_MARKERS)


def _parse_comments(issue: dict) -> list[tuple[datetime, str, str]] | None:
    parsed_comments = []
    for comment in issue.get("comments") or []:
        author = _comment_author(comment)
        created_at = _comment_time(comment)
        if author is None or created_at is None:
            return None
        parsed_comments.append((created_at, author, str(comment.get("body") or "")))
    return sorted(parsed_comments, key=lambda item: item[0])


def _latest_handler_comment(
    comments: list[tuple[datetime, str, str]],
    reporter: str,
    closure_comment: str,
) -> tuple[datetime, str, str] | None:
    handler_comments = []
    for item in comments:
        from_reporter = item[1].casefold() == reporter.casefold()
        is_closure_comment = item[2].strip() == closure_comment
        if not from_reporter and not is_closure_comment:
            handler_comments.append(item)
    return handler_comments[-1] if handler_comments else None


def _has_later_reporter_reply(
    comments: list[tuple[datetime, str, str]],
    reporter: str,
    after: datetime,
    closure_comment: str,
) -> bool:
    for item in comments:
        is_later = item[0] > after
        from_reporter = item[1].casefold() == reporter.casefold()
        is_closure_comment = item[2].strip() == closure_comment
        if is_later and from_reporter and not is_closure_comment:
            return True
    return False


def _inactive_decision(
    handler_comment: tuple[datetime, str, str], policy: ClosePolicy
) -> dict:
    last_reply_at, handler, _ = handler_comment
    now = policy.now
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    inactive_for = now - last_reply_at
    inactive_hours = round(inactive_for.total_seconds() / 3600, 3)
    if inactive_for < timedelta(hours=policy.inactive_hours):
        return {
            "eligible": False,
            "reason": "quiet_period_not_reached",
            "last_handler_reply_at": last_reply_at.isoformat(),
            "inactive_hours": inactive_hours,
        }
    return {
        "eligible": True,
        "reason": "answered_and_inactive",
        "handler": handler,
        "last_handler_reply_at": last_reply_at.isoformat(),
        "inactive_hours": inactive_hours,
    }


def _watched_wait_decision(
    issue: dict,
    watch: dict,
    comments: list[tuple[datetime, str, str]],
    reporter: str,
    policy: ClosePolicy,
) -> dict:
    """Evaluate an explicit awaiting-reporter watch, including info requests."""
    if watch.get("conversation_state") != "awaiting_reporter":
        return {"eligible": False, "reason": "followup_watch_not_waiting"}
    baseline = issue_api.parse_iso(watch.get("last_maintainer_comment_at", ""))
    waiting_since = issue_api.parse_iso(watch.get("waiting_since", ""))
    if baseline is None or waiting_since is None:
        return {"eligible": False, "reason": "followup_watch_incomplete"}
    if baseline.tzinfo is None:
        baseline = baseline.replace(tzinfo=timezone.utc)
    if waiting_since.tzinfo is None:
        waiting_since = waiting_since.replace(tzinfo=timezone.utc)
    if _has_later_reporter_reply(comments, reporter, baseline, policy.closure_comment):
        return {
            "eligible": False,
            "reason": "reporter_replied_after_handler",
            "last_handler_reply_at": baseline.isoformat(),
        }
    synthetic = (waiting_since, "maintainer", "follow-up watch")
    decision = _inactive_decision(synthetic, policy)
    if decision["eligible"]:
        decision["reason"] = "watched_awaiting_reporter_inactive"
    return decision


def evaluate_issue(
    issue: dict, policy: ClosePolicy, followup_watch: dict | None = None
) -> dict:
    """Return a deterministic eligibility decision without external writes."""
    state = str(issue.get("state") or "").casefold()
    if state not in ("open", "opened"):
        return {"eligible": False, "reason": "not_open"}
    if not is_question_issue(issue, policy.question_labels, policy.title_markers):
        return {"eligible": False, "reason": "not_question"}

    reporter = str(issue.get("author") or "").strip()
    if not reporter:
        return {"eligible": False, "reason": "reporter_unknown"}

    parsed_comments = _parse_comments(issue)
    if parsed_comments is None:
        return {"eligible": False, "reason": "comment_evidence_incomplete"}
    if followup_watch:
        return _watched_wait_decision(
            issue, followup_watch, parsed_comments, reporter, policy
        )
    handler_comment = _latest_handler_comment(
        parsed_comments, reporter, policy.closure_comment
    )
    if handler_comment is None:
        return {"eligible": False, "reason": "no_handler_reply"}

    if not _is_substantive_handler_reply(handler_comment[2], policy.closure_comment):
        return {"eligible": False, "reason": "latest_handler_comment_not_answer"}

    last_reply_at = handler_comment[0]
    if _has_later_reporter_reply(
        parsed_comments, reporter, last_reply_at, policy.closure_comment
    ):
        return {
            "eligible": False,
            "reason": "reporter_replied_after_handler",
            "last_handler_reply_at": last_reply_at.isoformat(),
        }

    return _inactive_decision(handler_comment, policy)


def issue_mentions_pr(issue: dict) -> bool:
    parts = [str(issue.get("description") or issue.get("body") or "")]
    parts.extend(
        str(comment.get("body") or "") for comment in issue.get("comments") or []
    )
    return bool(PR_URL_RE.search("\n".join(parts)))


def pr_exclusion_reason(
    issue: dict,
    issue_pr_map: dict[str, list[dict]],
    association_complete: bool,
) -> str | None:
    """Return the conservative PR-related reason that blocks auto-close."""
    number = issue.get("iid") or issue.get("number")
    if issue_mentions_pr(issue) or issue_pr_map.get(str(number)):
        return "linked_pr"
    if not association_complete:
        return "association_scan_incomplete"
    return None


def _comment_exists(comments: list[dict], body: str) -> bool:
    return any(str(comment.get("body") or "").strip() == body for comment in comments)


def close_issue(
    api: IssueApiContext,
    issue: dict,
    policy: ClosePolicy,
    followup_watch: dict | None = None,
) -> dict:
    """Refresh, comment, verify, close, and verify one previously selected Issue."""
    number = issue.get("iid") or issue.get("number")
    issue_url = f"{api.api_base}/repos/{api.owner}/{api.repo}/issues/{number}"
    comments_url = f"{issue_url}/comments"

    fresh_raw = issue_api.get_single_issue(api, number)
    fresh = issue_api.normalize_issue(fresh_raw)
    fresh["comments"] = issue_api.get_issue_comments(api, number)
    decision = evaluate_issue(fresh, policy, followup_watch)
    if not decision["eligible"]:
        return {"status": "skipped_after_refresh", "reason": decision["reason"]}

    if not _comment_exists(fresh["comments"], policy.closure_comment):
        response = api_post(
            api.session,
            comments_url,
            api.token,
            data={"body": policy.closure_comment},
        )
        if response.status_code >= 400:
            return {"status": "comment_failed", "http_status": response.status_code}

    verified_comments = issue_api.get_issue_comments(api, number)
    if not _comment_exists(verified_comments, policy.closure_comment):
        return {"status": "comment_failed", "reason": "comment_not_found_after_post"}

    fresh["comments"] = verified_comments
    race_check = evaluate_issue(fresh, policy, followup_watch)
    if not race_check["eligible"]:
        return {"status": "skipped_before_close", "reason": race_check["reason"]}

    response = api_patch(
        api.session, issue_url, api.token, json_data={"state": "closed"}
    )
    if response.status_code >= 400:
        return {"status": "close_failed", "http_status": response.status_code}
    verified = api_get(api.session, issue_url, api.token)
    if str(verified.get("state") or "").casefold() != "closed":
        return {"status": "close_failed", "reason": "state_not_closed_after_patch"}
    return {"status": "closed", "comment": policy.closure_comment}


def _as_tuple(value, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)
    raise ValueError(
        "Error: auto_close labels and title markers must be strings or arrays of strings"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help=f"Config path (default: {classifier.DEFAULT_CONFIG_FILE})",
    )
    parser.add_argument("--token", default=None, help="Defaults to GITCODE_TOKEN")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Post the comment and close eligible Issues",
    )
    parser.add_argument(
        "--hours", type=float, default=None, help="Override the quiet period"
    )
    parser.add_argument("--comment", default=None, help="Override the closure comment")
    parser.add_argument("--pr-fetch-pages", type=int, default=None)
    parser.add_argument("--pr-linkage-api-budget", type=int, default=None)
    return parser.parse_args(argv)


def _close_policy(args, auto_cfg) -> ClosePolicy:
    inactive_hours = float(
        args.hours
        if args.hours is not None
        else auto_cfg.get("inactive_hours", DEFAULT_INACTIVE_HOURS)
    )
    if inactive_hours <= 0:
        raise ValueError("Error: quiet period must be greater than zero")
    closure_comment = str(
        args.comment or auto_cfg.get("comment") or DEFAULT_COMMENT
    ).strip()
    if not closure_comment:
        raise ValueError("Error: closure comment must not be empty")
    return ClosePolicy(
        now=datetime.now(timezone.utc),
        inactive_hours=inactive_hours,
        closure_comment=closure_comment,
        question_labels=_as_tuple(
            auto_cfg.get("question_labels"), DEFAULT_QUESTION_LABELS
        ),
        title_markers=_as_tuple(
            auto_cfg.get("question_title_markers"), DEFAULT_QUESTION_TITLE_MARKERS
        ),
    )


def _association_limits(args, cfg, auto_cfg) -> tuple[int, int]:
    pr_fetch_pages = int(
        args.pr_fetch_pages or auto_cfg.get("pr_fetch_pages", cfg["pr_fetch_pages"])
    )
    linkage_budget = int(
        args.pr_linkage_api_budget
        or auto_cfg.get("pr_linkage_api_budget", cfg["pr_linkage_api_budget"])
    )
    return pr_fetch_pages, linkage_budget


def _build_runtime(args: argparse.Namespace) -> RuntimeConfig:
    """Validate CLI/config values and build a compact runtime context."""
    cfg = load_settings(args.config, allow_legacy=args.config is None)
    auto_cfg = cfg["auto_close"]
    token = issue_api.resolve_token(args.token)
    repo_path = cfg["repo"]
    if "/" not in repo_path:
        raise ValueError("Error: config 'repo' must use owner/repo format")
    owner, repo = repo_path.split("/", 1)
    api_base = cfg["gitcode_api"].rstrip("/")
    policy = _close_policy(args, auto_cfg)
    pr_fetch_pages, linkage_budget = _association_limits(args, cfg, auto_cfg)
    api = IssueApiContext(
        issue_api.make_session(rate_limit_dir=rate_limit_path(cfg["cache_dir"])),
        api_base,
        owner,
        repo,
        token,
    )
    followup_cfg = cfg["follow_up"]
    followup_state_file = str(followup_cfg["state_file"])
    state = followup_state.load_followup_state(followup_state_file, repo_path)
    return RuntimeConfig(
        api,
        repo_path,
        policy,
        pr_fetch_pages,
        linkage_budget,
        args.apply,
        followup_state_file,
        state["issues"],
    )


def _evaluate_issues(
    issues: list[dict], policy: ClosePolicy, watches: dict
) -> tuple[dict[str, dict], list[dict]]:
    decisions = {}
    candidates = []
    for issue in issues:
        decision = evaluate_issue(issue, policy, watches.get(str(issue.get("iid"))))
        decisions[str(issue.get("iid"))] = decision
        if decision["eligible"]:
            candidates.append(issue)
    return decisions, candidates


def _scan_associations(
    candidates: list[dict], runtime: RuntimeConfig
) -> tuple[dict[str, list[dict]], bool, dict]:
    if not candidates:
        return {}, True, {"pr_fetch": None, "linkage_fallback": None}

    created_values = [
        issue_api.parse_iso(issue.get("created_at", "")) for issue in candidates
    ]
    known_created_values = [value for value in created_values if value is not None]
    since_iso = (
        min(known_created_values).isoformat()
        if len(known_created_values) == len(candidates)
        else None
    )
    rate_limiter = session_rate_limiter(runtime.api.session)
    pr_fetch_options = classifier.PRFetchOptions(
        api_base=runtime.api.api_base,
        repo=runtime.repo_path,
        token=runtime.api.token,
        since_iso=since_iso,
        max_pages=runtime.pr_fetch_pages,
        rate_limiter=rate_limiter,
    )
    prs, pr_diagnostics = classifier.fetch_recent_prs(pr_fetch_options)
    linkage_options = classifier.LinkageOptions(
        runtime.api.api_base,
        runtime.repo_path,
        runtime.api.token,
        target_issue_numbers=[issue.get("iid") for issue in candidates],
        api_budget=runtime.linkage_budget,
        rate_limiter=rate_limiter,
    )
    issue_pr_map, linkage_diagnostics = classifier.build_issue_pr_map(
        prs, linkage_options
    )
    diagnostics = {
        "pr_fetch": pr_diagnostics,
        "linkage_fallback": linkage_diagnostics,
    }
    complete = pr_diagnostics["complete"] and linkage_diagnostics["complete"]
    return issue_pr_map, complete, diagnostics


def _handle_issue(
    issue: dict,
    decision: dict,
    issue_pr_map: dict[str, list[dict]],
    association_complete: bool,
    runtime: RuntimeConfig,
) -> dict:
    number = issue.get("iid")
    result = {
        "number": number,
        "title": issue.get("title", ""),
        "eligible": decision["eligible"],
        "reason": decision["reason"],
        "action": "skipped",
    }
    if not decision["eligible"]:
        return result
    pr_reason = pr_exclusion_reason(issue, issue_pr_map, association_complete)
    if pr_reason:
        result.update({"eligible": False, "reason": pr_reason})
        return result
    if not runtime.apply_changes:
        result.update({"action": "would_close", **decision})
        return result
    try:
        watch = runtime.followup_watches.get(str(number))
        action = close_issue(runtime.api, issue, runtime.policy, watch)
    except requests.RequestException as exc:
        action = {
            "status": "action_failed",
            "reason": issue_api.redact_token(str(exc)),
        }
    action_status = action.get("status", "action_failed")
    result.update({"action": action_status, "details": action})
    if action_status == "closed":
        try:
            removed = followup_state.resolve_issue(
                runtime.followup_state_file, runtime.repo_path, str(number)
            )
            result.get("details", {})["followup_watch_removed"] = removed
        except (OSError, ValueError) as exc:
            result.get("details", {})["followup_watch_cleanup"] = {
                "status": "failed",
                "reason": str(exc),
            }
    return result


def _build_summary(
    issues: list[dict],
    results: list[dict],
    runtime: RuntimeConfig,
    association_scan: dict,
) -> dict:
    failed_actions = {"comment_failed", "close_failed", "action_failed"}
    return {
        "mode": "apply" if runtime.apply_changes else "dry_run",
        "inactive_hours": runtime.policy.inactive_hours,
        "total_open": len(issues),
        "would_close": sum(item["action"] == "would_close" for item in results),
        "closed": sum(item["action"] == "closed" for item in results),
        "skipped": sum(
            item["action"] == "skipped" or item["action"].startswith("skipped_")
            for item in results
        ),
        "failed": sum(item["action"] in failed_actions for item in results),
        "association_scan": association_scan,
        "issues": results,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        runtime = _build_runtime(args)
        raw_issues = issue_api.get_issues(runtime.api, state="opened")
        issues = [issue_api.normalize_issue(item) for item in raw_issues]
        issue_api.enrich_issues_with_comments(
            runtime.api,
            issues,
        )
        decisions, candidates = _evaluate_issues(
            issues, runtime.policy, runtime.followup_watches
        )
        issue_pr_map, complete, association_scan = _scan_associations(
            candidates, runtime
        )
        results = [
            _handle_issue(
                issue,
                decisions[str(issue.get("iid"))],
                issue_pr_map,
                complete,
                runtime,
            )
            for issue in issues
        ]
        summary = _build_summary(issues, results, runtime, association_scan)
        _write_stdout(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1 if summary["failed"] else 0
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 1
    except requests.RequestException as exc:
        LOGGER.error(
            "Error: GitCode request failed — %s",
            issue_api.redact_token(str(exc)),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
