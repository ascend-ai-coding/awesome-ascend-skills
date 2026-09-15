#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""
Classify open GitCode issues into action buckets using a deterministic
decision tree.

Shared HTTP / token / URL utilities are imported from gitcode-toolkit's
gitcode_client module; this script only contains classification-specific
business logic (PR fetching, decision tree, report formatting).

Pipeline:
    # Standard usage: comments are fetched on demand after PR association
    python scripts/fetch_issues.py --since 2026-08-04 \\
      > .cannbot/gitcode-issue-handler/data/issues.json
    python scripts/classify_issues.py \\
      --input .cannbot/gitcode-issue-handler/data/issues.json

    # Default is interactive/dry-run: do not POST /assign comments
    python scripts/classify_issues.py \\
      --config .cannbot/gitcode-issue-handler/config/classify_config.yaml

    # Write mode is batch-only and requires explicit upstream authorization
    python scripts/classify_issues.py \\
      --config .cannbot/gitcode-issue-handler/config/classify_config.yaml \\
      --authorization-mode approved_batch

Config (YAML, see classify_config.yaml.example):
    repo: cann/ops-math
    gitcode_api: https://api.gitcode.com/api/v5
    last_check_file: .cannbot/gitcode-issue-handler/data/last_check.json
    report_file: .cannbot/gitcode-issue-handler/reports/classification.txt

Token (only needed for auto-assign side effect):
    GITCODE_TOKEN env var. A token alone never enables writes. Auto-assignment
    additionally requires --authorization-mode approved_batch. The legacy
    --no-auto-assign option remains a force-dry-run override.

Decision tree and reason strings mirror the reference plugin script.
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple

import requests

_HERE = Path(__file__).resolve().parent
_TOOLKIT_SCRIPTS = _HERE.parent.parent / "gitcode-toolkit" / "scripts"
sys.path.insert(0, str(_TOOLKIT_SCRIPTS))
from gitcode_client import (  # noqa: E402
    SharedRateLimiter,
    make_session,
    api_get,
    api_post,
    parse_iso,
    TZ_CHINA,
    DEFAULT_GITCODE_API_BASE,
)

sys.path.insert(0, str(_HERE))
from fetch_cache import load_pr_links, save_pr_links  # noqa: E402
from fetch_issues import RepoApiContext, enrich_issues_with_comments  # noqa: E402
from cli_output import write_stdout  # noqa: E402
from runtime_paths import (  # noqa: E402
    CLASSIFICATION_REPORT,
    CLASSIFY_CONFIG,
    FETCH_CACHE,
    FOLLOWUP_WATCH_STATE,
    LAST_CHECK_STATE,
    LEGACY_CLASSIFY_CONFIG,
    compatible_read_path,
    migrate_legacy_runtime_defaults,
    path_text,
    rate_limit_path,
)

DEFAULT_CONFIG_FILE = path_text(CLASSIFY_CONFIG)
DEFAULT_LAST_CHECK_FILE = path_text(LAST_CHECK_STATE)
DEFAULT_REPORT_FILE = path_text(CLASSIFICATION_REPORT)
DEFAULT_CACHE_DIR = path_text(FETCH_CACHE)
DEFAULT_LOOKBACK_DAYS = 7
PR_FETCH_PAGES = 5
PR_LINKAGE_API_BUDGET = 3
_ASSIGN_PATTERN = re.compile(
    r"\A\s*/assign\s+(?:@\S+|\[@[^\]\s]+\]\([^)]+\))\s*\Z",
    re.IGNORECASE,
)
_MENTION_ONLY_PATTERN = re.compile(
    r"\A\s*(?:\[\s*)?@[^\s\],，。.!！?？)]+"
    r"(?:\s*\]\([^)]+\))?\s*[,，。.!！?？]?\s*\Z"
)
_SYSTEM_BOT_LOGINS = frozenset({"cann-robot"})
_SYSTEM_COMMENT_PATTERNS = (
    re.compile(
        r"\A\s*(?:#{1,6}\s*notice\s*)?"
        r"this\s+issue\s+can\s+not\s+be\s+assigned\s+to\s+\**yourself\**\.\s*"
        r"please\s+try\s+to\s+assign\s+to\s+the\s+repository\s+members\.\s*\Z",
        re.IGNORECASE,
    ),
)
_ASSIGNEE_WAIT_PATTERNS = (
    re.compile(
        r"(?:已|已经|正在)?(?:联系|转交|转给|反馈给|同步给).{0,24}"
        r"(?:责任人|负责人|owner|maintainer|assignee)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:等待|请).{0,24}(?:责任人|负责人|owner|maintainer|assignee)"
        r".{0,24}(?:处理|回复|确认|反馈|分析|跟进)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:已|已经)?(?:转交|指派|分配)(?:给|至)?.{0,32}(?:处理|跟进)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:assigned|forwarded|escalated).{0,24}" r"(?:owner|maintainer|assignee)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:waiting|pending).{0,24}(?:owner|maintainer|assignee)",
        re.IGNORECASE,
    ),
)
LOGGER = logging.getLogger(__name__)


class LinkageOptions(NamedTuple):
    """Repository and scan settings for native PR-to-Issue linkage."""

    api_base: str
    repo: str
    token: str
    target_issue_numbers: object = None
    api_budget: int = PR_LINKAGE_API_BUDGET
    scan_mode: str = "ambiguous"
    cache_dir: object = DEFAULT_CACHE_DIR
    rate_limiter: object = None


class PRFetchOptions(NamedTuple):
    """Repository and paging settings for one recent-PR scan."""

    api_base: str
    repo: str
    token: str
    since_iso: str | None = None
    max_pages: int = PR_FETCH_PAGES
    rate_limiter: object = None


class ClassificationOptions(NamedTuple):
    """External evidence and side-effect policy for one classification."""

    issue_pr_map: dict
    post_fn: object
    dry_run: bool
    association_scan_complete: bool = True
    comment_scan_complete: bool = True


class CommentTimeline(NamedTuple):
    """Normalized participant comments in chronological order."""

    reporter: str
    assignee: str
    reporter_comments: list[dict]
    maintainer_comments: list[dict]
    latest_reporter: dict | None
    latest_maintainer: dict | None


class WatchSignals(NamedTuple):
    """Signals derived from an explicit persisted follow-up watch."""

    watch: dict
    state: str
    latest_assignee: dict | None
    reporter_followup: bool
    assignee_followup: bool


class InferredSignals(NamedTuple):
    """Signals derived from explicit hand-off wording in public comments."""

    latest_wait: dict | None
    waiting: bool
    assignee_followup: bool
    reporter_followup: bool


class ConversationAnalysis(NamedTuple):
    """Inputs needed to render the final conversation decision."""

    timeline: CommentTimeline
    watch: WatchSignals
    inferred: InferredSignals
    state: str
    pending_reporter: bool
    core_closed: bool
    custom_state: str


class IssueEvidence(NamedTuple):
    """Normalized evidence consumed by the classification decision tree."""

    number: object
    author: str | None
    assignee: str | None
    effective_comments: list[dict]
    active_prs: list[dict]
    followup_selected: bool


def _write_stdout(text):
    """Write the classifier JSON output protocol to stdout."""
    write_stdout(text)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def resolve_config_path(path=None, *, allow_legacy=True):
    """Prefer canonical config and read the former root file only as fallback."""
    requested = CLASSIFY_CONFIG if path is None else Path(path)
    if not allow_legacy:
        return requested
    return compatible_read_path(
        requested,
        canonical=CLASSIFY_CONFIG,
        legacy=LEGACY_CLASSIFY_CONFIG,
    )


def load_config(path=None, repo_override=None, *, allow_legacy=True):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "Error: PyYAML required. Install with: pip install pyyaml"
        ) from exc

    cfg = {}
    config_path = resolve_config_path(path, allow_legacy=allow_legacy)
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    if allow_legacy:
        cfg = migrate_legacy_runtime_defaults(cfg, config_path)
    if repo_override:
        cfg["repo"] = repo_override

    if "repo" not in cfg or not cfg["repo"]:
        raise ValueError(
            "Error: repository is unknown. Pass --repo owner/repo, use input "
            "from fetch_issues.py, or set repo in the config file."
        )

    cfg.setdefault("gitcode_api", DEFAULT_GITCODE_API_BASE)
    cfg.setdefault("last_check_file", DEFAULT_LAST_CHECK_FILE)
    cfg.setdefault("report_file", DEFAULT_REPORT_FILE)
    cfg.setdefault("cache_dir", DEFAULT_CACHE_DIR)
    cfg.setdefault("pr_fetch_pages", PR_FETCH_PAGES)
    cfg.setdefault("pr_linkage_api_budget", PR_LINKAGE_API_BUDGET)
    cfg.setdefault("pr_linkage_scan_mode", "ambiguous")
    if cfg.get("pr_linkage_scan_mode") not in {"ambiguous", "all"}:
        raise ValueError("Error: pr_linkage_scan_mode must be 'ambiguous' or 'all'")
    follow_up = cfg.setdefault("follow_up", {})
    if not isinstance(follow_up, dict):
        raise ValueError("Error: follow_up config must be an object")
    follow_up.setdefault("state_file", path_text(FOLLOWUP_WATCH_STATE))
    follow_up.setdefault("waiting_status", "挂起")
    follow_up.setdefault("active_status", "进行中")
    follow_up.setdefault("lookback_days", 30)
    follow_up.setdefault("fetch_pages", 10)
    follow_up.setdefault("poll_hours", 24)
    follow_up.setdefault("stale_hours", 48)
    return cfg


# --------------------------------------------------------------------------- #
# last_check.json state
# --------------------------------------------------------------------------- #
def load_last_check(path, repo):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("repo") == repo:
            return data.get("last_all_clear_time")
    except (json.JSONDecodeError, OSError):
        pass
    return None


def save_last_check(path, repo):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    now_iso = datetime.now(TZ_CHINA).strftime("%Y-%m-%dT%H:%M:%S+08:00")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"last_all_clear_time": now_iso, "repo": repo},
            f,
            ensure_ascii=False,
            indent=2,
        )


def get_since(path, repo):
    last = load_last_check(path, repo)
    if last:
        return last
    return (datetime.now(TZ_CHINA) - timedelta(days=DEFAULT_LOOKBACK_DAYS)).strftime(
        "%Y-%m-%dT00:00:00+08:00"
    )


def _normalize_since(value):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=TZ_CHINA).isoformat()
    except ValueError:
        parsed = parse_iso(value)
        return parsed.isoformat() if parsed is not None else value


def resolve_time_scope(raw, ignore_last_check, last_check_file, repo):
    """Resolve one authoritative input scope and whether to filter by update."""
    filters = raw.get("filters", {}) if isinstance(raw, dict) else {}
    if ignore_last_check:
        return None, False, "full_input"
    if filters.get("mode") == "single":
        return None, False, "single_issue"
    if filters.get("since"):
        return _normalize_since(filters["since"]), False, "fetch_since"
    return get_since(last_check_file, repo), True, "last_check"


# --------------------------------------------------------------------------- #
# PR fetching + issue→PR mapping
# --------------------------------------------------------------------------- #
def fetch_pr_linked_issues(api_base, repo, pr_number, token, rate_limiter=None):
    url = f"{api_base}/repos/{repo}/pulls/{pr_number}/issues"
    session = make_session(rate_limiter=rate_limiter)
    return api_get(session, url, token)


def fetch_recent_prs(options: PRFetchOptions):
    """Fetch recent PRs and stop once the updated-time window is exhausted."""
    url = f"{options.api_base}/repos/{options.repo}/pulls"
    since_dt = parse_iso(options.since_iso) if options.since_iso else None
    session = make_session(rate_limiter=options.rate_limiter)
    diagnostics = {
        "complete": True,
        "pages_requested": 0,
        "warnings": [],
    }
    all_prs = []
    for page in range(1, options.max_pages + 1):
        params = {
            "state": "all",
            "page": page,
            "per_page": 100,
            "sort": "updated",
            "direction": "desc",
        }
        try:
            batch = api_get(session, url, token, params=params)
        except Exception as exc:
            diagnostics["complete"] = False
            diagnostics["warnings"].append(
                f"PR list page {page} failed: {type(exc).__name__}"
            )
            break
        diagnostics["pages_requested"] += 1
        batch = batch if isinstance(batch, list) else []
        if not batch:
            break
        all_prs.extend(batch)
        if len(batch) < 100:
            break
        if since_dt:
            updated_values = [parse_iso(pr.get("updated_at", "")) for pr in batch]
            if any(
                updated is not None and updated < since_dt for updated in updated_values
            ):
                break
    else:
        diagnostics["complete"] = False
        diagnostics["warnings"].append(
            f"PR scan reached configured page limit ({options.max_pages})"
        )
    return all_prs, diagnostics


class LinkageScanState:
    """Mutable state for one bounded native-linkage scan."""

    def __init__(self, options):
        self.options = options
        self.targets = (
            {
                str(number)
                for number in options.target_issue_numbers
                if number is not None
            }
            if options.target_issue_numbers is not None
            else None
        )
        self.parsed_refs = {}
        self.api_budget_used = 0
        self.incomplete_numbers = set()
        self.diagnostics = {
            "scan_mode": options.scan_mode,
            "candidates": 0,
            "api_calls": 0,
            "cache_hits": 0,
            "api_errors": 0,
            "budget_exhausted": False,
            "incomplete_issue_numbers": [],
        }


def _text_refs(pr, targets):
    text = f"{pr.get('body') or ''} {pr.get('title') or ''}"
    refs = set(re.findall(r"#(\d+)", text) + re.findall(r"/issues/(\d+)", text))
    return text, refs & targets if targets is not None else refs


def _linkage_candidates(prs, state):
    candidates = []
    for pr in prs:
        pr_num = pr.get("number")
        if pr_num is None:
            continue
        text, refs = _text_refs(pr, state.targets)
        state.parsed_refs[pr_num] = refs
        if state.options.scan_mode == "all":
            if not refs:
                candidates.append((pr, set(state.targets or [])))
            continue
        if not state.targets:
            continue
        head_ref = (pr.get("head") or {}).get("ref") or ""
        ambiguous_refs = set()
        for ref in state.targets - refs:
            pattern = rf"(?<!\d){re.escape(ref)}(?!\d)"
            if re.search(pattern, f"{text} {head_ref}"):
                ambiguous_refs.add(ref)
        if ambiguous_refs:
            candidates.append((pr, ambiguous_refs))
    state.diagnostics["candidates"] = len(candidates)
    return candidates


def _linked_numbers(pr, ambiguous_refs, state):
    options = state.options
    cached = load_pr_links(options.cache_dir, options.repo, pr)
    if cached is not None:
        state.diagnostics["cache_hits"] += 1
        return cached
    if state.api_budget_used >= max(0, options.api_budget):
        state.diagnostics["budget_exhausted"] = True
        state.incomplete_numbers.update(ambiguous_refs or (state.targets or []))
        return None

    state.api_budget_used += 1
    state.diagnostics["api_calls"] += 1
    try:
        linked = fetch_pr_linked_issues(
            options.api_base,
            options.repo,
            pr["number"],
            options.token,
            options.rate_limiter,
        )
    except requests.RequestException:
        state.diagnostics["api_errors"] += 1
        state.incomplete_numbers.update(ambiguous_refs or (state.targets or []))
        return None
    linked_items = linked if isinstance(linked, list) else []
    linked_numbers = []
    for item in linked_items:
        if isinstance(item, dict) and item.get("number") is not None:
            linked_numbers.append(item.get("number"))
    save_pr_links(options.cache_dir, options.repo, pr, linked_numbers)
    return linked_numbers


def _resolve_linkage_candidates(candidates, state):
    for pr, ambiguous_refs in candidates:
        linked_numbers = _linked_numbers(pr, ambiguous_refs, state)
        if linked_numbers is None:
            continue
        for number in linked_numbers:
            ref = str(number)
            if state.targets is None or ref in state.targets:
                state.parsed_refs[pr["number"]].add(ref)


def _invert_pr_refs(prs, parsed_refs):
    issue_pr_map = {}
    for pr in prs:
        pr_num = pr.get("number")
        if pr_num is None:
            continue
        pr_author = (pr.get("user") or {}).get("login", "")
        pr_info = {
            "pr_number": pr_num,
            "pr_state": pr.get("state"),
            "pr_merged": bool(pr.get("merged")),
            "pr_title": (pr.get("title") or "")[:60],
            "pr_author": pr_author,
        }
        for ref in parsed_refs.get(pr_num, set()):
            issue_pr_map.setdefault(ref, []).append(pr_info)
    return issue_pr_map


def build_issue_pr_map(prs, options: LinkageOptions):
    """Build Issue-to-PR evidence using text refs and bounded native linkage."""
    if options.scan_mode not in {"ambiguous", "all"}:
        raise ValueError("scan_mode must be 'ambiguous' or 'all'")
    state = LinkageScanState(options)
    candidates = _linkage_candidates(prs, state)
    _resolve_linkage_candidates(candidates, state)
    state.diagnostics["incomplete_issue_numbers"] = sorted(
        state.incomplete_numbers, key=lambda value: int(value)
    )
    state.diagnostics["complete"] = (
        state.diagnostics["api_errors"] == 0
        and not state.diagnostics["budget_exhausted"]
    )
    return _invert_pr_refs(prs, state.parsed_refs), state.diagnostics


# --------------------------------------------------------------------------- #
# Comment shape helpers
# --------------------------------------------------------------------------- #
def is_only_assign_comments(comments):
    """Return whether every effective comment is an assignment command."""
    if not comments:
        return False
    for c in comments:
        body = (c.get("body") or "").strip()
        if not _ASSIGN_PATTERN.fullmatch(body):
            return False
    return True


def get_comment_author(comment):
    """Return a comment author's login for normalized or raw API shapes."""
    author = comment.get("author")
    if isinstance(author, str):
        return author
    if isinstance(author, dict):
        return author.get("login")
    user = comment.get("user")
    if isinstance(user, dict):
        return user.get("login")
    return None


def _is_system_comment(comment) -> bool:
    """Recognize explicit system events or known bot-generated templates."""
    if comment.get("system") is True:
        return True
    author = get_comment_author(comment)
    if not author or author.strip().casefold() not in _SYSTEM_BOT_LOGINS:
        return False
    body = str(comment.get("body") or "")
    return any(pattern.fullmatch(body) for pattern in _SYSTEM_COMMENT_PATTERNS)


def get_effective_comments(comments, issue_author):
    """Keep comments with a known author other than the issue reporter."""
    normalized_issue_author = issue_author.strip().casefold() if issue_author else None
    effective_comments = []
    for comment in comments:
        if _is_system_comment(comment):
            continue
        body = str(comment.get("body") or "")
        if _MENTION_ONLY_PATTERN.fullmatch(body):
            continue
        comment_author = get_comment_author(comment)
        if not comment_author:
            continue
        normalized_comment_author = comment_author.strip().casefold()
        if normalized_comment_author == "unknown":
            continue
        if (
            normalized_issue_author
            and normalized_comment_author == normalized_issue_author
        ):
            continue
        effective_comments.append(comment)
    return effective_comments


def _comment_time(comment):
    """Return a normalized comment time without inventing missing evidence."""
    return parse_iso(comment.get("created_at", ""))


def _comment_id(comment):
    value = comment.get("id")
    return str(value) if value is not None else None


def _indicates_assignee_wait(body: str) -> bool:
    """Recognize explicit hand-off text without treating every reply as pending."""
    return any(pattern.search(body) for pattern in _ASSIGNEE_WAIT_PATTERNS)


def _comment_after_watch(entry, watch, watch_time) -> bool:
    """Return whether a comment is newer than the watch baseline."""
    if not entry or not watch:
        return False
    baseline_id = watch.get("last_maintainer_comment_id")
    if baseline_id is not None and entry["id"] == str(baseline_id):
        return False
    if watch_time is not None and entry["parsed_at"] is not None:
        return entry["parsed_at"] > watch_time
    return True


def _comment_timeline(issue) -> CommentTimeline:
    """Normalize public comments into reporter and maintainer timelines."""
    reporter = str(issue.get("author") or "").strip()
    normalized_reporter = reporter.casefold()
    assignee = str(issue.get("assignee") or "").strip()
    comments = list(issue.get("comments") or [])
    if comments and all(_comment_time(comment) is not None for comment in comments):
        comments.sort(key=_comment_time)
    reporter_comments = []
    maintainer_comments = []
    for index, comment in enumerate(comments):
        if _is_system_comment(comment):
            continue
        author = get_comment_author(comment)
        body = str(comment.get("body") or "").strip()
        if not author or author.casefold() == "unknown" or not body:
            continue
        if _ASSIGN_PATTERN.fullmatch(body) or _MENTION_ONLY_PATTERN.fullmatch(body):
            continue
        entry = {
            "index": index,
            "id": _comment_id(comment),
            "created_at": comment.get("created_at", ""),
            "parsed_at": _comment_time(comment),
            "author": author,
            "body": body,
        }
        if normalized_reporter and author.casefold() == normalized_reporter:
            reporter_comments.append(entry)
        else:
            maintainer_comments.append(entry)
    return CommentTimeline(
        reporter,
        assignee,
        reporter_comments,
        maintainer_comments,
        reporter_comments[-1] if reporter_comments else None,
        maintainer_comments[-1] if maintainer_comments else None,
    )


def _latest_author_comment(comments, author):
    if not author:
        return None
    normalized = author.casefold()
    for entry in reversed(comments):
        if entry.get("author", "").casefold() == normalized:
            return entry
    return None


def _latest_wait_comment(comments):
    for entry in reversed(comments):
        if _indicates_assignee_wait(entry.get("body", "")):
            return entry
    return None


def _watch_signals(issue, timeline: CommentTimeline) -> WatchSignals:
    watch = issue.get("followup_watch") or {}
    watch_state = str(watch.get("conversation_state") or "").strip().casefold()
    watch_time = parse_iso(watch.get("last_maintainer_comment_at", ""))
    reporter_after_watch = bool(
        timeline.latest_reporter
        and watch
        and _comment_after_watch(timeline.latest_reporter, watch, watch_time)
    )
    awaited_assignee = str(watch.get("assignee") or timeline.assignee).strip()
    latest_assignee = _latest_author_comment(
        timeline.maintainer_comments, awaited_assignee
    )
    assignee_after_watch = bool(
        watch_state == "awaiting_assignee"
        and _comment_after_watch(latest_assignee, watch, watch_time)
    )
    return WatchSignals(
        watch,
        watch_state,
        latest_assignee,
        reporter_after_watch,
        assignee_after_watch,
    )


def _inferred_signals(timeline: CommentTimeline, watch: WatchSignals):
    latest_wait = _latest_wait_comment(timeline.maintainer_comments)
    normalized_reporter = timeline.reporter.casefold()
    inferred_assignee_wait = bool(
        not watch.watch
        and timeline.assignee
        and timeline.assignee.casefold() != normalized_reporter
        and latest_wait
    )
    inferred_assignee_followup = bool(
        inferred_assignee_wait
        and watch.latest_assignee
        and watch.latest_assignee["index"] > latest_wait["index"]
    )
    reporter_after_inferred_wait = bool(
        inferred_assignee_wait
        and timeline.latest_reporter
        and timeline.latest_reporter["index"] > latest_wait["index"]
        and (
            not inferred_assignee_followup
            or timeline.latest_reporter["index"] > watch.latest_assignee["index"]
        )
    )
    return InferredSignals(
        latest_wait,
        inferred_assignee_wait,
        inferred_assignee_followup,
        reporter_after_inferred_wait,
    )


def _conversation_state(issue, timeline, watch, inferred, pending_reporter):
    core_closed = str(issue.get("state") or "").casefold() == "closed"
    custom_state = str(issue.get("issue_state") or "").strip()
    terminal_custom = custom_state.casefold() in {
        "已完成",
        "已解决",
        "已拒绝",
        "已取消",
        "已验收",
    }
    if pending_reporter:
        return (
            "reopened_followup"
            if core_closed or terminal_custom
            else "reporter_followup"
        )
    if watch.assignee_followup or inferred.assignee_followup:
        return "assignee_followup"
    if watch.state == "awaiting_assignee" or inferred.waiting:
        return "awaiting_assignee"
    if watch.watch:
        return "awaiting_reporter"
    if timeline.latest_maintainer:
        return "maintainer_replied"
    return "awaiting_maintainer"


def _conversation_output(analysis: ConversationAnalysis) -> dict:
    timeline, watch, inferred = analysis.timeline, analysis.watch, analysis.inferred
    if analysis.pending_reporter:
        pending_since = timeline.latest_reporter["created_at"]
    elif watch.assignee_followup or inferred.assignee_followup:
        pending_since = watch.latest_assignee["created_at"]
    elif inferred.waiting:
        pending_since = inferred.latest_wait["created_at"]
    else:
        pending_since = None
    waiting_on = {
        "awaiting_reporter": "reporter",
        "awaiting_assignee": "assignee",
    }.get(analysis.state, "maintainer")
    waiting_since = watch.watch.get("waiting_since") if watch.watch else None
    if not watch.watch and inferred.waiting:
        waiting_since = inferred.latest_wait["created_at"]
    return {
        "state": analysis.state,
        "waiting_on": waiting_on,
        "latest_reporter_comment_id": (
            timeline.latest_reporter["id"] if timeline.latest_reporter else None
        ),
        "latest_reporter_comment_at": (
            timeline.latest_reporter["created_at"] if timeline.latest_reporter else None
        ),
        "latest_maintainer_comment_id": (
            timeline.latest_maintainer["id"] if timeline.latest_maintainer else None
        ),
        "latest_maintainer_comment_at": (
            timeline.latest_maintainer["created_at"]
            if timeline.latest_maintainer
            else None
        ),
        "pending_since": pending_since,
        "waiting_since": waiting_since,
        "reopen_required": bool(analysis.pending_reporter and analysis.core_closed),
        "activate_required": bool(
            analysis.state
            in {"reporter_followup", "reopened_followup", "assignee_followup"}
            and analysis.custom_state.casefold() != "进行中"
        ),
        "waiting_status_reconcile_required": bool(
            analysis.state in {"awaiting_reporter", "awaiting_assignee"}
            and analysis.custom_state.casefold() != "挂起"
        ),
        "waiting_watch_required": bool(
            analysis.state == "awaiting_assignee" and not watch.watch
        ),
    }


def analyze_conversation(issue):
    """Determine whose turn is actionable from ordered Issue comments."""
    timeline = _comment_timeline(issue)
    watch = _watch_signals(issue, timeline)
    inferred = _inferred_signals(timeline, watch)
    reporter_after_maintainer = bool(
        timeline.latest_reporter
        and timeline.latest_maintainer
        and timeline.latest_reporter["index"] > timeline.latest_maintainer["index"]
    )
    pending_reporter = bool(
        watch.reporter_followup
        or reporter_after_maintainer
        or inferred.reporter_followup
    )
    state = _conversation_state(issue, timeline, watch, inferred, pending_reporter)

    core_closed = str(issue.get("state") or "").casefold() == "closed"
    custom_state = str(issue.get("issue_state") or "").strip()
    analysis = ConversationAnalysis(
        timeline, watch, inferred, state, pending_reporter, core_closed, custom_state
    )
    return _conversation_output(analysis)


def followup_sla(pending_since, now=None):
    """Return a one-business-day follow-up response clock."""
    started = parse_iso(pending_since)
    if started is None:
        return "unknown"
    due = started + timedelta(days=1)
    while due.weekday() >= 5:
        due += timedelta(days=1)
    current = now or datetime.now(TZ_CHINA)
    if current.tzinfo is None:
        current = current.replace(tzinfo=TZ_CHINA)
    if current > due:
        return "breached"
    if current >= due - timedelta(hours=4):
        return "at_risk"
    return "pending"


def should_fetch_comments(issue, issue_pr_map):
    """Fetch only when comments can still change the classification result."""
    sources = set(issue.get("fetch_sources") or [])
    if issue.get("followup_watch") or sources & {"updated", "watchlist"}:
        return True, "followup_detection_required"
    author = issue.get("author")
    assignee = issue.get("assignee")
    if author and assignee and author.casefold() == assignee.casefold():
        return False, "self_assigned"
    number = issue.get("number") or issue.get("iid")
    active_prs = [
        pr
        for pr in issue_pr_map.get(str(number), [])
        if pr.get("pr_state") == "open" or pr.get("pr_merged")
    ]
    if active_prs:
        return False, "active_linked_pr"
    return True, "classification_required"


def filter_by_updated_since(issues, since_iso):
    """Keep Issues updated since the boundary, including unparseable dates."""
    if not since_iso:
        return issues
    since_dt = parse_iso(since_iso)
    if since_dt is None:
        return issues
    filtered = []
    for issue in issues:
        sources = set(issue.get("fetch_sources") or [])
        if sources & {"updated", "watchlist"}:
            filtered.append(issue)
            continue
        updated = parse_iso(issue.get("updated_at", ""))
        if updated is None or updated >= since_dt:
            filtered.append(issue)
    return filtered


# --------------------------------------------------------------------------- #
# Decision tree
# --------------------------------------------------------------------------- #
def _classification_result(bucket, category, reason, auto_action=None):
    return {
        "bucket": bucket,
        "category": category,
        "reason": reason,
        "auto_action": auto_action,
    }


def _classify_unassigned_pr(number, author_login, active_pr_refs, options):
    pr_authors = list({pr["pr_author"] for pr in active_pr_refs if pr.get("pr_author")})
    if not pr_authors:
        return _classification_result(
            "need_attention",
            "needs_manual_no_pr_author",
            "无负责人，已有关联PR，但无法获取PR作者，需手动处理",
        )
    if author_login and any(
        author_login.casefold() == pr_author.casefold() for pr_author in pr_authors
    ):
        return _classification_result(
            "no_attention",
            "self_assigned",
            f"自提 issue（提出者 {author_login} 已提交关联PR），无需自动指派",
        )

    author = pr_authors[0]
    assign_comment = f"/assign @{author}"
    if options.dry_run:
        action = {
            "type": "comment",
            "body": assign_comment,
            "success": None,
            "dry_run": True,
        }
        reason = f"无负责人，已有关联PR(作者: {', '.join(pr_authors)})，[预览] 将自动指派 @{author}"
        return _classification_result(
            "no_attention", "auto_assign_via_pr", reason, action
        )

    success = options.post_fn(number, assign_comment, author)
    action = {
        "type": "comment",
        "body": assign_comment,
        "success": success,
        "dry_run": False,
    }
    if success:
        reason = (
            f"无负责人，已有关联PR(作者: {', '.join(pr_authors)})，已自动指派 @{author}"
        )
        return _classification_result(
            "no_attention", "auto_assign_via_pr", reason, action
        )
    reason = (
        f"无负责人，已有关联PR(作者: {', '.join(pr_authors)})，自动指派失败，需手动处理"
    )
    return _classification_result(
        "need_attention", "auto_assign_failed", reason, action
    )


def _classify_unassigned(effective_comments):
    if not effective_comments:
        return _classification_result(
            "need_attention",
            "needs_first_look",
            "无负责人、无关联PR、无非提出者评论回复，需要关注并分配",
        )
    if is_only_assign_comments(effective_comments):
        return _classification_result(
            "need_attention",
            "needs_only_assign_cmd",
            "无负责人，评论仅为指派命令，需要关注并分配",
        )
    return _classification_result(
        "no_attention",
        "replied_no_owner",
        "无负责人，已有非提出者评论回复，不需要关注",
    )


def _classify_assigned(assignee_login, active_pr_refs, effective_comments):
    if active_pr_refs:
        return _classification_result(
            "no_attention",
            "our_team_done_with_pr",
            f"负责人 {assignee_login}，已有关联PR，已处理",
        )
    if not effective_comments:
        return _classification_result(
            "need_attention",
            "our_team_needs_work",
            f"负责人 {assignee_login}，无关联PR且无非提出者评论回复，需要处理",
        )
    if is_only_assign_comments(effective_comments):
        return _classification_result(
            "need_attention",
            "our_team_only_assign_cmd",
            f"负责人 {assignee_login}，评论仅为指派命令，需要处理",
        )
    return _classification_result(
        "no_attention",
        "our_team_replied",
        f"负责人 {assignee_login}，已有非提出者评论回复，可能已处理",
    )


def _issue_evidence(issue, options: ClassificationOptions) -> IssueEvidence:
    number = issue.get("number") or issue.get("iid")
    author_login = issue.get("author")
    assignee_login = issue.get("assignee")
    effective_comments = get_effective_comments(
        issue.get("comments", []) or [], author_login
    )
    pr_refs = options.issue_pr_map.get(str(number), [])
    active_pr_refs = [
        pr for pr in pr_refs if pr.get("pr_state") == "open" or pr.get("pr_merged")
    ]
    followup_selected = bool(
        issue.get("followup_watch")
        or set(issue.get("fetch_sources") or []) & {"updated", "watchlist"}
    )
    return IssueEvidence(
        number,
        author_login,
        assignee_login,
        effective_comments,
        active_pr_refs,
        followup_selected,
    )


def _classify_conversation(conversation):
    state = conversation.get("state")
    if state in {"reporter_followup", "reopened_followup"}:
        reason = "提出者在维护者回复后新增评论，需要优先跟进"
        if state == "reopened_followup":
            reason += "；Issue 当前为关闭或终态，回复前应恢复为进行中"
        return _classification_result("need_attention", state, reason)
    if state == "assignee_followup":
        return _classification_result(
            "need_attention",
            "assignee_followup",
            "责任人在挂起后新增回复，需要恢复进行中并判断是否已经解决",
        )
    if state == "awaiting_assignee":
        if conversation.get("waiting_watch_required") or conversation.get(
            "waiting_status_reconcile_required"
        ):
            return _classification_result(
                "need_attention",
                "awaiting_assignee_setup",
                "已有首响且明确等待责任人处理，需切为挂起并建立 watchlist 跟踪",
            )
        return _classification_result(
            "no_attention",
            "awaiting_assignee",
            "Issue 已挂起并等待责任人处理，由 watchlist 持续跟踪",
        )
    if state == "awaiting_reporter":
        if conversation.get("waiting_status_reconcile_required"):
            return _classification_result(
                "need_attention",
                "awaiting_reporter_setup",
                "Issue 正在等待提出者补充，需将自定义状态恢复为挂起",
            )
        return _classification_result(
            "no_attention",
            "awaiting_reporter",
            "维护者已请求提出者补充，Issue 保持挂起并由 watchlist 持续跟踪",
        )
    return None


def _classify_remaining(evidence: IssueEvidence, options: ClassificationOptions):
    if (
        evidence.author
        and evidence.assignee
        and evidence.author.casefold() == evidence.assignee.casefold()
    ):
        reason = f"自提 issue（提出者 {evidence.author} 即负责人），已在自行处理"
        return _classification_result("no_attention", "self_assigned", reason)
    comments_unhelpful = not evidence.effective_comments or is_only_assign_comments(
        evidence.effective_comments
    )
    if (
        not options.association_scan_complete
        and not evidence.active_prs
        and comments_unhelpful
    ):
        return _classification_result(
            "need_attention",
            "association_scan_incomplete",
            "PR 列表扫描不完整，保留待自动重试，不执行外部动作",
        )
    if evidence.assignee is not None:
        return _classify_assigned(
            evidence.assignee, evidence.active_prs, evidence.effective_comments
        )
    if evidence.active_prs:
        return _classify_unassigned_pr(
            evidence.number, evidence.author, evidence.active_prs, options
        )
    return _classify_unassigned(evidence.effective_comments)


def classify_one(issue, options: ClassificationOptions):
    """Apply the deterministic decision tree to one Issue."""
    evidence = _issue_evidence(issue, options)
    if not options.comment_scan_complete and (
        evidence.followup_selected or not evidence.active_prs
    ):
        return _classification_result(
            "need_attention",
            "comment_scan_incomplete",
            "评论获取未完成，保留待自动续跑，不执行外部动作",
        )
    conversation_result = _classify_conversation(analyze_conversation(issue))
    if conversation_result:
        return conversation_result
    return _classify_remaining(evidence, options)


# --------------------------------------------------------------------------- #
# Report (Chinese, matches reference script format)
# --------------------------------------------------------------------------- #
def format_report(need_attention, no_attention, since_iso, all_clear):
    now_str = datetime.now(TZ_CHINA).strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"===== 待处理 Issue ({now_str}) =====", ""]

    lines.append(f"【进入处理流程 ({len(need_attention)} 个）】")
    if need_attention:
        for item in need_attention:
            lines.append(f'  #{item["number"]} {item["title"]}')
            lines.append(f'    负责人: {item["assignee"] or "无"}')
            lines.append(f'    评论数: {item["comments_count"]}')
            lines.append(f'    原因: {item["reason"]}')
            lines.append(f'    链接: {item["url"]}')
            lines.append("")
    else:
        lines.append("  无")
        lines.append("")

    return "\n".join(lines)


def apply_processing_mode(result, single_mode):
    """Make an explicit single Issue actionable without losing triage evidence."""
    routed = dict(result)
    routed["classification_bucket"] = result["bucket"]
    routed["must_handle"] = bool(single_mode)
    routed["single_issue_override"] = bool(
        single_mode and result["bucket"] != "need_attention"
    )
    if single_mode:
        routed["bucket"] = "need_attention"
    return routed


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def read_input(args):
    try:
        if args.input:
            with open(args.input, "r", encoding="utf-8") as f:
                return json.load(f)
        text = sys.stdin.read()
        if not text.strip():
            raise ValueError(
                "no JSON input received; the upstream fetch command may " "have failed"
            )
        return json.loads(text)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Error: cannot read classifier input — {exc}") from exc


def extract_issues(raw):
    if isinstance(raw, dict):
        return raw.get("issues", []) or []
    if isinstance(raw, list):
        return raw
    raise ValueError("Error: input JSON must be an object with 'issues' or an array")


def write_report(path, content):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


class ClassifierRuntime(NamedTuple):
    """Validated values shared across one classifier run."""

    args: object
    cfg: dict
    issues: list
    repo: str
    api_base: str
    single_mode: bool
    since_iso: object
    time_scope_source: str
    token: str
    dry_run: bool
    post_fn: object
    rate_limiter: object


def _add_input_args(parser):
    parser.add_argument(
        "--config",
        default=None,
        help=f"Optional YAML config (default: {DEFAULT_CONFIG_FILE})",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="Repository as owner/repo; overrides input metadata and config",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Read issues JSON from file instead of stdin",
    )


def _add_run_policy_args(parser):
    parser.add_argument(
        "--authorization-mode",
        choices=("interactive", "approved_batch"),
        default="interactive",
        help=(
            "External-write authorization. Default 'interactive' is dry-run. "
            "Use 'approved_batch' only after the user explicitly approves the "
            "displayed batch scope."
        ),
    )
    parser.add_argument(
        "--no-auto-assign",
        action="store_true",
        help=(
            "Force dry-run and do not POST /assign comments, even when "
            "--authorization-mode approved_batch is present."
        ),
    )
    parser.add_argument(
        "--ignore-last-check",
        action="store_true",
        help="Classify the complete supplied input. An input filters.since value "
        "already takes precedence over last_check automatically.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable durable comment and PR-link cache reads and writes",
    )
    parser.add_argument(
        "--refresh-comments",
        action="store_true",
        help="Ignore valid cached comments and fetch required comments again",
    )
    parser.add_argument(
        "--full-pr-linkage-scan",
        action="store_true",
        help="Use the native linkage API for every PR without a text link. "
        "Default: only verify ambiguous target-number matches.",
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Classify open GitCode issues using a deterministic decision tree. "
            "Reads issues JSON from stdin (or --input), fetches related PRs, "
            "applies the tree, optionally posts /assign comments, and writes "
            "JSON to stdout plus a Chinese report to report_file."
        )
    )
    _add_input_args(parser)
    _add_run_policy_args(parser)
    return parser.parse_args(argv)


def load_runtime(args):
    raw = read_input(args)
    filters = raw.get("filters", {}) if isinstance(raw, dict) else {}
    cfg = load_config(
        args.config,
        repo_override=args.repo or filters.get("repository"),
        allow_legacy=args.config is None,
    )
    issues = extract_issues(raw)
    repo = cfg["repo"]
    api_base = cfg["gitcode_api"]
    single_mode = filters.get("mode") == "single"
    if single_mode and args.authorization_mode == "approved_batch":
        raise ValueError(
            "Error: approved_batch authorization is valid only for batch input"
        )
    since_iso, apply_updated_filter, time_scope_source = resolve_time_scope(
        raw,
        args.ignore_last_check,
        cfg["last_check_file"],
        repo,
    )
    if apply_updated_filter:
        issues = filter_by_updated_since(issues, since_iso)
    token = os.environ.get("GITCODE_TOKEN", "")
    rate_limiter = SharedRateLimiter(rate_limit_path(cfg["cache_dir"]))
    dry_run = (
        args.authorization_mode != "approved_batch"
        or args.no_auto_assign
        or not token
        or single_mode
    )
    post_session = make_session(rate_limiter=rate_limiter) if not dry_run else None

    def post_fn(number, body, expected_assignee):
        if not token:
            return False
        url = f"{api_base}/repos/{repo}/issues/{number}/comments"
        try:
            response = api_post(post_session, url, token, data={"body": body})
            if response.status_code >= 400:
                return False
            issue_url = f"{api_base}/repos/{repo}/issues/{number}"
            issue = api_get(post_session, issue_url, token)
            assignee = issue.get("assignee") if isinstance(issue, dict) else None
            assignee_login = (
                assignee.get("login") if isinstance(assignee, dict) else None
            )
            if not assignee_login and isinstance(issue, dict):
                assignees = issue.get("assignees") or []
                if assignees and isinstance(assignees[0], dict):
                    assignee_login = assignees[0].get("login")
            return bool(
                assignee_login
                and assignee_login.casefold() == expected_assignee.casefold()
            )
        except requests.RequestException:
            return False

    return ClassifierRuntime(
        args,
        cfg,
        issues,
        repo,
        api_base,
        single_mode,
        since_iso,
        time_scope_source,
        token,
        dry_run,
        post_fn,
        rate_limiter,
    )


def _finish_empty_run(runtime):
    if not runtime.single_mode:
        save_last_check(runtime.cfg["last_check_file"], runtime.repo)
    output = {
        "total": 0,
        "mode": "single" if runtime.single_mode else "batch",
        "authorization_mode": runtime.args.authorization_mode,
        "dry_run": True,
        "transport": runtime.rate_limiter.snapshot(),
        "by_bucket": {"need_attention": 0, "no_attention": 0},
        "since": runtime.since_iso,
        "time_scope": {
            "source": runtime.time_scope_source,
            "since": runtime.since_iso,
        },
        "all_clear": True,
        "issues": [],
    }
    _write_stdout(json.dumps(output, indent=2, ensure_ascii=False))
    write_report(
        runtime.cfg["report_file"],
        format_report([], [], runtime.since_iso, all_clear=True),
    )


def _collect_evidence(runtime):
    pr_fetch_options = PRFetchOptions(
        api_base=runtime.api_base,
        repo=runtime.repo,
        token=runtime.token,
        since_iso=runtime.since_iso,
        max_pages=int(runtime.cfg["pr_fetch_pages"]),
        rate_limiter=runtime.rate_limiter,
    )
    prs, pr_fetch_diagnostics = fetch_recent_prs(pr_fetch_options)
    issue_numbers = [
        issue.get("number") or issue.get("iid") for issue in runtime.issues
    ]
    linkage_options = LinkageOptions(
        runtime.api_base,
        runtime.repo,
        runtime.token,
        target_issue_numbers=issue_numbers,
        api_budget=int(runtime.cfg["pr_linkage_api_budget"]),
        scan_mode=(
            "all"
            if runtime.args.full_pr_linkage_scan
            else runtime.cfg["pr_linkage_scan_mode"]
        ),
        cache_dir=None if runtime.args.no_cache else runtime.cfg["cache_dir"],
        rate_limiter=runtime.rate_limiter,
    )
    issue_pr_map, linkage_diagnostics = build_issue_pr_map(prs, linkage_options)
    owner, repo_name = runtime.repo.split("/", 1)
    comment_api = RepoApiContext(
        make_session(rate_limiter=runtime.rate_limiter),
        runtime.api_base,
        owner,
        repo_name,
        runtime.token,
    )
    comment_diagnostics = enrich_issues_with_comments(
        comment_api,
        runtime.issues,
        cache_dir=None if runtime.args.no_cache else runtime.cfg["cache_dir"],
        refresh=runtime.args.refresh_comments,
        should_fetch=lambda issue: should_fetch_comments(issue, issue_pr_map),
    )
    return issue_pr_map, pr_fetch_diagnostics, linkage_diagnostics, comment_diagnostics


def _classified_item(issue, routed, issue_pr_map):
    number = issue.get("number") or issue.get("iid")
    conversation = analyze_conversation(issue)
    return {
        "number": number,
        "title": issue.get("title", ""),
        "url": issue.get("url", ""),
        "assignee": issue.get("assignee"),
        "comments_count": issue.get("comments_count", 0) or 0,
        "created_at": issue.get("created_at", ""),
        "updated_at": issue.get("updated_at", ""),
        "issue_age_days": issue.get("issue_age_days"),
        "first_response_sla": issue.get("first_response_sla", "unknown"),
        "state": issue.get("state", ""),
        "issue_state": issue.get("issue_state", ""),
        "fetch_sources": issue.get("fetch_sources", []),
        "conversation_state": conversation["state"],
        "waiting_on": conversation["waiting_on"],
        "latest_reporter_comment_id": conversation["latest_reporter_comment_id"],
        "latest_reporter_comment_at": conversation["latest_reporter_comment_at"],
        "latest_maintainer_comment_id": conversation["latest_maintainer_comment_id"],
        "latest_maintainer_comment_at": conversation["latest_maintainer_comment_at"],
        "followup_pending_since": conversation["pending_since"],
        "waiting_since": conversation["waiting_since"],
        "followup_sla": followup_sla(conversation["pending_since"]),
        "reopen_required": conversation["reopen_required"],
        "activate_required": conversation["activate_required"],
        "waiting_status_reconcile_required": conversation[
            "waiting_status_reconcile_required"
        ],
        "waiting_watch_required": conversation["waiting_watch_required"],
        "linked_prs": issue_pr_map.get(str(number), []),
        "bucket": routed["bucket"],
        "classification_bucket": routed["classification_bucket"],
        "category": routed["category"],
        "reason": routed["reason"],
        "auto_action": routed["auto_action"],
        "must_handle": routed["must_handle"],
        "single_issue_override": routed["single_issue_override"],
    }


def attention_sort_key(item):
    """Prioritize new reporter turns before ordinary first-look work."""
    category = item.get("category")
    followup = category in {
        "reporter_followup",
        "reopened_followup",
        "assignee_followup",
        "awaiting_assignee_setup",
        "awaiting_reporter_setup",
    }
    followup_urgent = followup and item.get("followup_sla") in {
        "breached",
        "at_risk",
    }
    first_response_urgent = item.get("first_response_sla") in {
        "breached",
        "at_risk",
    }
    age = item.get("issue_age_days")
    age = float(age) if isinstance(age, (int, float)) else -1
    if followup_urgent:
        rank = 0
    elif followup:
        rank = 1
    elif first_response_urgent:
        rank = 2
    elif age >= 7:
        rank = 3
    elif age >= 5:
        rank = 4
    else:
        rank = 5
    created = parse_iso(item.get("created_at", ""))
    return rank, created or datetime.max.replace(tzinfo=TZ_CHINA)


def _classify_all(runtime, issue_pr_map, pr_diagnostics, linkage_diagnostics):
    need_attention = []
    no_attention = []
    incomplete_linkage = set(linkage_diagnostics["incomplete_issue_numbers"])
    for issue in runtime.issues:
        number = issue.get("number") or issue.get("iid")
        comments_status = (issue.get("comments_fetch") or {}).get("status")
        options = ClassificationOptions(
            issue_pr_map,
            runtime.post_fn,
            runtime.dry_run,
            association_scan_complete=(
                pr_diagnostics["complete"] and str(number) not in incomplete_linkage
            ),
            comment_scan_complete=comments_status != "error",
        )
        routed = apply_processing_mode(
            classify_one(issue, options), runtime.single_mode
        )
        target = (
            need_attention if routed["bucket"] == "need_attention" else no_attention
        )
        target.append(_classified_item(issue, routed, issue_pr_map))
    need_attention.sort(key=attention_sort_key)
    return need_attention, no_attention


def _finish_run(runtime, classified, diagnostics):
    need_attention, no_attention = classified
    pr_diagnostics, linkage_diagnostics, comment_diagnostics = diagnostics
    all_clear = not need_attention
    if all_clear and not runtime.single_mode:
        save_last_check(runtime.cfg["last_check_file"], runtime.repo)
    output = {
        "total": len(runtime.issues),
        "mode": "single" if runtime.single_mode else "batch",
        "authorization_mode": runtime.args.authorization_mode,
        "by_bucket": {
            "need_attention": len(need_attention),
            "no_attention": len(no_attention),
        },
        "since": runtime.since_iso,
        "time_scope": {
            "source": runtime.time_scope_source,
            "since": runtime.since_iso,
        },
        "all_clear": all_clear,
        "dry_run": runtime.dry_run,
        "transport": runtime.rate_limiter.snapshot(),
        "comment_fetch": comment_diagnostics,
        "association_scan": {
            "pr_fetch": pr_diagnostics,
            "linkage_fallback": linkage_diagnostics,
        },
        "issues": need_attention + no_attention,
    }
    _write_stdout(json.dumps(output, indent=2, ensure_ascii=False))
    write_report(
        runtime.cfg["report_file"],
        format_report(need_attention, no_attention, runtime.since_iso, all_clear),
    )


def main(argv=None):
    args = parse_args(argv)

    try:
        runtime = load_runtime(args)
    except (OSError, ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 2
    if not runtime.issues:
        _finish_empty_run(runtime)
        return 0
    evidence = _collect_evidence(runtime)
    issue_pr_map, pr_diagnostics, linkage_diagnostics, comment_diagnostics = evidence
    classified = _classify_all(
        runtime, issue_pr_map, pr_diagnostics, linkage_diagnostics
    )
    _finish_run(
        runtime,
        classified,
        (pr_diagnostics, linkage_diagnostics, comment_diagnostics),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
