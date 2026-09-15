#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""
Fetch issues from GitCode.

This is the GitCode adaptation of the original CodeHub fetch_issues.py.
The CLI interface and JSON output contract are kept identical so the
surrounding skill (issue resolver) works unchanged.

Shared HTTP / token / URL utilities are imported from gitcode-toolkit's
gitcode_client module; this script only contains issue-specific business
logic (pagination, normalisation, time filtering).

Usage:
    # Fetch open issues plus all-state/watchlist follow-up candidates
    python fetch_issues.py --url https://gitcode.com/cann/ops-math --token <token>

    # Use environment variables instead of CLI args
    export GITCODE_URL=https://gitcode.com/cann/ops-math
    export GITCODE_TOKEN=<token>
    python fetch_issues.py

    # Fetch today's open issues plus follow-up candidates
    python fetch_issues.py --today

    # Fetch issues created since a specific date
    python fetch_issues.py --since 2026-04-10

    # Fetch issues created within a date range
    python fetch_issues.py --since 2026-04-10 --until 2026-04-13

    # Fetch a full comment snapshot (classification normally fetches on demand)
    python fetch_issues.py --with-comments

    # Fetch a single issue by URL (comments included by default)
    python fetch_issues.py --issue https://gitcode.com/cann/ops-math/issues/2170

    # Single issue without comments
    python fetch_issues.py --issue <url> --no-comments

Resolution order for each parameter:
    --url   → GITCODE_URL   env var  (no built-in default; one must be provided)
    --token → GITCODE_TOKEN env var  (no built-in default; one must be provided)
    --issue → takes precedence over --url; the two are mutually exclusive.

The API base URL defaults to https://api.gitcode.com/api/v5 when the repo URL
is on gitcode.com; override with --api-base or GITCODE_API_BASE for self-hosted.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import NamedTuple

import requests

# --------------------------------------------------------------------------- #
# Import shared GitCode API client from gitcode-toolkit
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.realpath(__file__))
_TOOLKIT_SCRIPTS = os.path.normpath(
    os.path.join(_HERE, "..", "..", "gitcode-toolkit", "scripts")
)
sys.path.insert(0, _TOOLKIT_SCRIPTS)
from gitcode_client import (  # noqa: E402
    GitCodeClientError,
    resolve_token,
    resolve_api_base,
    parse_repo_path,
    parse_issue_url,
    make_session,
    api_get,
    rate_limit_metrics,
    redact_token,
    parse_iso,
    TZ_CHINA,
    STATE_MAP,
)

sys.path.insert(0, _HERE)
from fetch_cache import load_comments, save_comments  # noqa: E402
from runtime_paths import FETCH_CACHE, path_text, rate_limit_path  # noqa: E402
from followup_state import (  # noqa: E402
    DEFAULT_STATE_FILE,
    advance_updated_cursor,
    load_followup_config,
    load_followup_state,
)
from cli_output import write_stdout  # noqa: E402

DEFAULT_CACHE_DIR = path_text(FETCH_CACHE)
DEFAULT_FOLLOWUP_LOOKBACK_DAYS = 30
DEFAULT_FOLLOWUP_FETCH_PAGES = 10
LOGGER = logging.getLogger(__name__)


class RepoApiContext(NamedTuple):
    """Repository-scoped values shared by Issue API operations."""

    session: object
    api_base: str
    owner: str
    repo: str
    token: str


def _write_stdout(text):
    """Write the JSON output protocol to stdout."""
    write_stdout(text)


# --------------------------------------------------------------------------- #
# CLI URL resolution (fetch_issues-specific, not a shared utility)
# --------------------------------------------------------------------------- #
def resolve_url(cli_url):
    url = cli_url or os.environ.get("GITCODE_URL")
    if not url:
        raise ValueError(
            "Error: Repository URL not provided.\n"
            "Pass --url <url> or set the GITCODE_URL environment variable."
        )
    return url


# --------------------------------------------------------------------------- #
# Domain operations
# --------------------------------------------------------------------------- #
def get_issues(
    api: RepoApiContext,
    state="opened",
    created_since=None,
    created_until_exclusive=None,
):
    """Fetch all issues with pagination.

    CLI state values ('opened'/'closed'/'all') are mapped to GitCode's
    'open'/'closed'/'all' internally.
    """
    gc_state = STATE_MAP.get(state, state)
    url = f"{api.api_base}/repos/{api.owner}/{api.repo}/issues"
    all_issues = []
    page = 1
    while True:
        params = {
            "state": gc_state,
            "page": page,
            "per_page": 100,
            "sort": "created",
            "direction": "desc",
        }
        data = api_get(api.session, url, api.token, params=params)
        batch = data if isinstance(data, list) else []
        if not batch:
            break
        reached_since_boundary = False
        for issue in batch:
            created_at = parse_iso(issue.get("created_at", ""))
            if created_at is None:
                all_issues.append(issue)
                continue
            if created_since and created_at < created_since:
                reached_since_boundary = True
                continue
            if created_until_exclusive and created_at >= created_until_exclusive:
                continue
            all_issues.append(issue)
        if reached_since_boundary:
            break
        if len(batch) < 100:
            break
        page += 1
    return all_issues


def get_updated_issues(
    api: RepoApiContext,
    updated_since: datetime,
    *,
    max_pages: int = DEFAULT_FOLLOWUP_FETCH_PAGES,
):
    """Fetch all-state Issues updated since a cursor.

    The cursor is advanced only when this scan is complete. This catches old
    or closed Issues whose reporter or awaited assignee added a new comment.
    """
    url = f"{api.api_base}/repos/{api.owner}/{api.repo}/issues"
    issues = []
    diagnostics = {
        "complete": True,
        "pages_requested": 0,
        "boundary": updated_since.isoformat(),
        "warnings": [],
    }
    for page in range(1, max_pages + 1):
        params = {
            "state": "all",
            "page": page,
            "per_page": 100,
            "sort": "updated",
            "direction": "desc",
        }
        try:
            data = api_get(api.session, url, api.token, params=params)
        except requests.RequestException as exc:
            diagnostics["complete"] = False
            diagnostics["warnings"].append(
                f"updated Issue page {page} failed: {type(exc).__name__}"
            )
            break
        diagnostics["pages_requested"] += 1
        if not isinstance(data, list):
            diagnostics["complete"] = False
            diagnostics["warnings"].append(
                f"updated Issue page {page} returned an invalid response shape"
            )
            break
        batch = data
        if not batch:
            break
        reached_boundary = False
        for issue in batch:
            updated_at = parse_iso(issue.get("updated_at", ""))
            if updated_at is not None and updated_at < updated_since:
                reached_boundary = True
                continue
            issues.append(issue)
        if reached_boundary or len(batch) < 100:
            break
    else:
        diagnostics["complete"] = False
        diagnostics["warnings"].append(
            f"updated Issue scan reached page limit ({max_pages})"
        )
    return issues, diagnostics


def get_watched_issues(api: RepoApiContext, watched: dict):
    """Refresh every watched Issue directly, regardless of core state."""
    issues = []
    diagnostics = {"complete": True, "requested": 0, "errors": []}
    for number in sorted(watched, key=str):
        diagnostics["requested"] += 1
        try:
            issue = get_single_issue(api, number)
        except requests.RequestException as exc:
            diagnostics["complete"] = False
            diagnostics["errors"].append(
                {"issue_number": number, "error": type(exc).__name__}
            )
            continue
        if not isinstance(issue, dict):
            diagnostics["complete"] = False
            diagnostics["errors"].append(
                {"issue_number": number, "error": "invalid_response_shape"}
            )
            continue
        issue["followup_watch"] = watched[number]
        issues.append(issue)
    return issues, diagnostics


def merge_issue_sources(*sources):
    """Merge Issue snapshots by IID while preserving how each was selected."""
    merged = {}
    for source_name, issues in sources:
        for raw in issues:
            number = raw.get("number") or raw.get("iid")
            if number is None:
                continue
            key = str(number)
            existing = merged.get(key, {})
            source_names = set(existing.get("fetch_sources") or [])
            source_names.add(source_name)
            watch = raw.get("followup_watch") or existing.get("followup_watch")
            existing.update(raw)
            existing["fetch_sources"] = sorted(source_names)
            if watch:
                existing["followup_watch"] = watch
            merged[key] = existing
    return list(merged.values())


def get_single_issue(api: RepoApiContext, issue_number):
    """Fetch one issue by number. Returns the raw issue dict."""
    url = f"{api.api_base}/repos/{api.owner}/{api.repo}/issues/{issue_number}"
    return api_get(api.session, url, api.token)


def get_issue_comments(api: RepoApiContext, issue_number, *, page_diagnostics=None):
    """Fetch all comments for an issue.

    GitCode's comments API (GitHub/Gitea-style) does not surface a `system`
    flag, so all comments are returned. System-generated activity events
    typically do not appear in this endpoint to begin with.
    """
    url = f"{api.api_base}/repos/{api.owner}/{api.repo}/issues/{issue_number}/comments"
    all_comments = []
    page = 1
    while True:
        params = {"page": page, "per_page": 100, "sort": "asc"}
        data = api_get(api.session, url, api.token, params=params)
        if page_diagnostics is not None:
            page_diagnostics["comment_pages"] += 1
        batch = data if isinstance(data, list) else []
        if not batch:
            break
        all_comments.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return [
        {
            "id": c.get("id"),
            "author": (c.get("user") or {}).get("login")
            or (c.get("author") or {}).get("login")
            or "unknown",
            "body": c.get("body", ""),
            "created_at": c.get("created_at", ""),
            "updated_at": c.get("updated_at", ""),
        }
        for c in all_comments
    ]


def normalize_issue(raw):
    """Map a raw GitCode issue to the output schema the skill expects.

    Output keys: iid, title, description, labels, state, created_at
    (plus a few useful GitCode-native extras).
    """
    labels_raw = raw.get("labels") or []
    labels = []
    for lb in labels_raw:
        if isinstance(lb, str):
            labels.append(lb)
        elif isinstance(lb, dict):
            name = lb.get("name") or lb.get("label_name")
            if name:
                labels.append(name)

    assignee = raw.get("assignee")
    assignee_login = assignee.get("login") if isinstance(assignee, dict) else None
    if not assignee_login:
        assignees_raw = raw.get("assignees") or []
        if (
            isinstance(assignees_raw, list)
            and assignees_raw
            and isinstance(assignees_raw[0], dict)
        ):
            assignee_login = assignees_raw[0].get("login")

    user = raw.get("user") or {}
    author_login = user.get("login") if isinstance(user, dict) else None
    if not author_login:
        author_obj = raw.get("author") or {}
        author_login = author_obj.get("login") if isinstance(author_obj, dict) else None

    return {
        "iid": raw.get("number"),
        "number": raw.get("number"),
        "title": raw.get("title", ""),
        "description": raw.get("body", ""),
        "labels": labels,
        "state": raw.get("state", ""),
        "issue_state": raw.get("issue_state", ""),
        "issue_state_id": (raw.get("issue_state_detail") or {}).get("id"),
        "finished_at": raw.get("finished_at", ""),
        "created_at": raw.get("created_at", ""),
        "updated_at": raw.get("updated_at", ""),
        "url": raw.get("html_url", ""),
        "author": author_login,
        "assignee": assignee_login,
        "comments_count": raw.get("comments", 0),
        "fetch_sources": raw.get("fetch_sources", []),
        "followup_watch": raw.get("followup_watch"),
    }


def _mark_skipped(issue, diagnostics, reason):
    issue["comments"] = []
    issue["comments_fetch"] = {"status": "skipped", "reason": reason}
    diagnostics["skipped"] += 1


def _reuse_or_skip_comments(issue, diagnostics, refresh, should_fetch):
    """Return true after satisfying an Issue without cache/API access."""
    if issue.get("iid") is None and issue.get("number") is None:
        _mark_skipped(issue, diagnostics, "missing_issue_number")
        return True

    existing_status = (issue.get("comments_fetch") or {}).get("status")
    if "comments" in issue and not refresh and existing_status != "error":
        issue["comments_fetch"] = {"status": "input"}
        diagnostics["input_hits"] += 1
        return True

    if "comments_count" in issue and int(issue.get("comments_count", 0) or 0) == 0:
        _mark_skipped(issue, diagnostics, "no_comments")
        return True

    if should_fetch is None:
        return False
    decision = should_fetch(issue)
    fetch_required, reason = (
        decision if isinstance(decision, tuple) else (bool(decision), "not_required")
    )
    if fetch_required:
        return False
    _mark_skipped(issue, diagnostics, reason)
    return True


def enrich_issues_with_comments(
    api: RepoApiContext,
    issues,
    *,
    cache_dir=DEFAULT_CACHE_DIR,
    refresh=False,
    should_fetch=None,
):
    """Attach comments in-place, persisting each successful Issue fetch.

    ``should_fetch`` may return a bool or ``(bool, reason)``. A failed Issue
    request is recorded and does not discard comments already cached for other
    Issues, so the next run resumes from the unfinished item.
    """
    repo_key = f"{api.owner}/{api.repo}"
    diagnostics = {
        "complete": True,
        "api_requests": 0,
        "comment_pages": 0,
        "cache_hits": 0,
        "input_hits": 0,
        "skipped": 0,
        "errors": [],
    }
    for issue in issues:
        num = issue.get("iid") or issue.get("number")
        if _reuse_or_skip_comments(issue, diagnostics, refresh, should_fetch):
            continue

        cached = None if refresh else load_comments(cache_dir, repo_key, issue)
        if cached is not None:
            issue["comments"] = cached
            issue["comments_fetch"] = {"status": "cache"}
            diagnostics["cache_hits"] += 1
            continue

        diagnostics["api_requests"] += 1
        try:
            comments = get_issue_comments(api, num, page_diagnostics=diagnostics)
        except requests.RequestException as exc:
            issue["comments"] = []
            issue["comments_fetch"] = {
                "status": "error",
                "error": type(exc).__name__,
            }
            diagnostics["complete"] = False
            diagnostics["errors"].append(
                {"issue_number": num, "error": type(exc).__name__}
            )
            continue
        issue["comments"] = comments
        issue["comments_fetch"] = {"status": "api"}
        save_comments(cache_dir, repo_key, issue, comments)
    return diagnostics


# --------------------------------------------------------------------------- #
# Time filtering (client-side, TZ +08:00)
# --------------------------------------------------------------------------- #
def parse_date(date_str):
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(
            f"Error: invalid date format '{date_str}', expected YYYY-MM-DD"
        ) from exc
    return dt.replace(tzinfo=TZ_CHINA)


def filter_issues_by_time(issues, since=None, until=None):
    if since is None and until is None:
        return issues
    filtered = []
    for issue in issues:
        sources = set(issue.get("fetch_sources") or [])
        if sources & {"updated", "watchlist"}:
            filtered.append(issue)
            continue
        created_at = parse_iso(issue.get("created_at", ""))
        if created_at is None:
            continue
        if since and created_at < since:
            continue
        if until and created_at >= until + timedelta(days=1):
            continue
        filtered.append(issue)
    return filtered


def filter_issues_by_self_assigned(issues):
    """Exclude issues where author == assignee (self-submitted issues).

    These are issues the reporter assigned to themselves — they're already
    being handled by the reporter and don't need external intervention.
    """
    filtered = []
    for issue in issues:
        author = issue.get("author")
        assignee = issue.get("assignee")
        if author and assignee and author == assignee:
            continue
        filtered.append(issue)
    return filtered


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _add_connection_args(parser):
    conn = parser.add_argument_group("connection")
    conn.add_argument(
        "--url",
        default=None,
        help="GitCode repository URL (or set GITCODE_URL env var). "
        "Use --issue for single-issue mode.",
    )
    conn.add_argument(
        "--issue",
        default=None,
        metavar="URL",
        help="Single-issue URL (e.g. https://gitcode.com/owner/repo/issues/123). "
        "Mutually exclusive with --url; bypasses list/state/time filtering.",
    )
    conn.add_argument(
        "--token",
        default=None,
        help="GitCode access token (or set GITCODE_TOKEN env var)",
    )
    conn.add_argument(
        "--api-base",
        default=None,
        help="Override API base URL, e.g. https://api.gitcode.com/api/v5 "
        "(or set GITCODE_API_BASE env var)",
    )


def _add_filter_args(parser):
    parser.add_argument(
        "--state",
        default="opened",
        choices=["opened", "closed", "all"],
        help="Issue state filter. 'opened' is mapped to GitCode 'open' internally.",
    )
    time_group = parser.add_argument_group("time filters")
    time_group.add_argument(
        "--today",
        action="store_true",
        help="Only show issues created today (local timezone +08:00). "
        "Overrides --since/--until.",
    )
    time_group.add_argument(
        "--since",
        metavar="YYYY-MM-DD",
        help="Only show issues created on or after this date",
    )
    time_group.add_argument(
        "--until",
        metavar="YYYY-MM-DD",
        help="Only show issues created on or before this date",
    )
    parser.add_argument(
        "--exclude-self-assigned",
        action="store_true",
        help="Exclude issues where author == assignee in list mode.",
    )


def _add_comment_args(parser):
    parser.add_argument(
        "--with-comments",
        action="store_true",
        help="Include all comments in list mode.",
    )
    parser.add_argument(
        "--no-comments",
        action="store_true",
        help="Skip comments in single-issue mode.",
    )
    cache_group = parser.add_argument_group("cache")
    cache_group.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help=f"Durable fetch cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    cache_group.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable durable comment cache reads and writes",
    )
    cache_group.add_argument(
        "--refresh-comments",
        action="store_true",
        help="Ignore valid cached comments and fetch them again",
    )


def _add_followup_args(parser):
    group = parser.add_argument_group("follow-up tracking")
    group.add_argument(
        "--config",
        default=None,
        help=(
            "Optional classify config; defaults to the canonical config when "
            "that file exists"
        ),
    )
    group.add_argument(
        "--no-follow-up",
        action="store_true",
        help="Disable all-state updated scanning and watchlist refresh",
    )
    group.add_argument(
        "--follow-up-state-file",
        default=None,
        help=f"Follow-up watch/cursor state (default: {DEFAULT_STATE_FILE})",
    )
    group.add_argument(
        "--follow-up-lookback-days",
        type=int,
        default=None,
        help="Initial all-state updated-at lookback when no cursor exists",
    )
    group.add_argument(
        "--follow-up-fetch-pages",
        type=int,
        default=None,
        help="Safety limit for all-state updated-at scanning",
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="GitCode issue fetcher (adapted from the CodeHub version)"
    )
    _add_connection_args(parser)
    _add_filter_args(parser)
    _add_comment_args(parser)
    _add_followup_args(parser)
    return parser.parse_args(argv)


def _single_issue_output(args, token):
    owner, repo, issue_number = parse_issue_url(args.issue)
    api_base = resolve_api_base(args.api_base, args.issue)
    session = make_session(rate_limit_dir=rate_limit_path(args.cache_dir))
    api = RepoApiContext(session, api_base, owner, repo, token)
    issues = [normalize_issue(get_single_issue(api, issue_number))]
    comment_fetch = None
    if not args.no_comments:
        comment_fetch = enrich_issues_with_comments(
            api,
            issues,
            cache_dir=None if args.no_cache else args.cache_dir,
            refresh=args.refresh_comments,
        )
    output = {
        "total": 1,
        "filters": {
            "mode": "single",
            "repository": f"{owner}/{repo}",
            "issue": issue_number,
            "state": None,
            "since": None,
            "until": None,
        },
        "issues": issues,
    }
    if comment_fetch is not None:
        output["comment_fetch"] = comment_fetch
    output["transport"] = rate_limit_metrics(session)
    return output


def _date_range(args):
    if args.today:
        today = datetime.now(TZ_CHINA).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return today, today
    since = parse_date(args.since) if args.since else None
    until = parse_date(args.until) if args.until else None
    return since, until


def _followup_options(args):
    """Resolve follow-up fetch settings from CLI, config, then defaults."""
    followup_cfg = load_followup_config(args.config)

    state_file = (
        args.follow_up_state_file
        or followup_cfg.get("state_file")
        or DEFAULT_STATE_FILE
    )
    lookback_days = int(
        args.follow_up_lookback_days
        if args.follow_up_lookback_days is not None
        else followup_cfg.get("lookback_days", DEFAULT_FOLLOWUP_LOOKBACK_DAYS)
    )
    fetch_pages = int(
        args.follow_up_fetch_pages
        if args.follow_up_fetch_pages is not None
        else followup_cfg.get("fetch_pages", DEFAULT_FOLLOWUP_FETCH_PAGES)
    )
    if lookback_days <= 0 or fetch_pages <= 0:
        raise ValueError(
            "Error: follow-up lookback days and fetch pages must be greater than zero"
        )
    return str(state_file), lookback_days, fetch_pages


def _collect_followup_sources(args, api, repository, primary_issues):
    """Merge updated/watchlist sources and persist a complete scan cursor."""
    diagnostics = {
        "enabled": not args.no_follow_up,
        "updated_scan": None,
        "watchlist_refresh": None,
        "cursor_advanced": False,
    }
    if args.no_follow_up:
        return primary_issues, diagnostics

    state_file, lookback_days, fetch_pages = _followup_options(args)
    state = load_followup_state(state_file, repository)
    scan_started = datetime.now(TZ_CHINA)
    cursor = parse_iso(state.get("updated_cursor"))
    if cursor is None:
        cursor = scan_started - timedelta(days=lookback_days)
    updated_issues, updated_diagnostics = get_updated_issues(
        api, cursor, max_pages=fetch_pages
    )
    watched_issues, watch_diagnostics = get_watched_issues(
        api, state.get("issues") or {}
    )
    raw_issues = merge_issue_sources(
        ("primary", primary_issues),
        ("updated", updated_issues),
        ("watchlist", watched_issues),
    )
    diagnostics.update(
        {
            "updated_scan": updated_diagnostics,
            "watchlist_refresh": watch_diagnostics,
            "cursor_before": cursor.isoformat(),
            "state_file": state_file,
            "lookback_days": lookback_days,
            "fetch_pages": fetch_pages,
        }
    )
    if updated_diagnostics.get("complete"):
        advance_updated_cursor(state_file, repository, scan_started.isoformat())
        diagnostics["cursor_advanced"] = True
        diagnostics["cursor_after"] = scan_started.isoformat()
    return raw_issues, diagnostics


def _batch_output(args, token):
    repo_url = resolve_url(args.url)
    api_base = resolve_api_base(args.api_base, repo_url)
    owner, repo = parse_repo_path(repo_url)
    session = make_session(rate_limit_dir=rate_limit_path(args.cache_dir))
    api = RepoApiContext(session, api_base, owner, repo, token)
    repository = f"{owner}/{repo}"
    since_dt, until_dt = _date_range(args)
    primary_issues = get_issues(
        api,
        state=args.state,
        created_since=since_dt,
        created_until_exclusive=(until_dt + timedelta(days=1) if until_dt else None),
    )
    raw_issues, followup_diagnostics = _collect_followup_sources(
        args, api, repository, primary_issues
    )
    issues = [normalize_issue(item) for item in raw_issues]
    issues = filter_issues_by_time(issues, since=since_dt, until=until_dt)
    if args.exclude_self_assigned:
        issues = filter_issues_by_self_assigned(issues)
    comment_fetch = None
    if args.with_comments:
        comment_fetch = enrich_issues_with_comments(
            api,
            issues,
            cache_dir=None if args.no_cache else args.cache_dir,
            refresh=args.refresh_comments,
        )
    output = {
        "total": len(issues),
        "filters": {
            "mode": "batch",
            "repository": repository,
            "state": args.state,
            "since": since_dt.strftime("%Y-%m-%d") if since_dt else None,
            "until": until_dt.strftime("%Y-%m-%d") if until_dt else None,
            "exclude_self_assigned": args.exclude_self_assigned,
            "follow_up": followup_diagnostics,
        },
        "issues": issues,
    }
    if comment_fetch is not None:
        output["comment_fetch"] = comment_fetch
    output["transport"] = rate_limit_metrics(session)
    return output


def main(argv=None):
    args = parse_args(argv)
    if args.issue and args.url:
        LOGGER.error("Error: --issue and --url are mutually exclusive.")
        return 1
    if not args.issue and not args.url:
        LOGGER.error(
            "Error: provide --url <repo-url> or --issue <issue-url> "
            "(or set GITCODE_URL)."
        )
        return 1
    try:
        token = resolve_token(args.token)
        output = (
            _single_issue_output(args, token)
            if args.issue
            else _batch_output(args, token)
        )
        _write_stdout(json.dumps(output, indent=2, ensure_ascii=False))
        return 0
    except GitCodeClientError as exc:
        LOGGER.error("%s", exc)
        return 1
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 1
    except requests.HTTPError as e:
        # Errors from api_get already carry a token-redacted message.
        LOGGER.error("Error: API request failed — %s", e)
        return 1
    except requests.RequestException as e:
        msg = redact_token(str(e))
        LOGGER.error("Error: network request failed — %s", msg)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
